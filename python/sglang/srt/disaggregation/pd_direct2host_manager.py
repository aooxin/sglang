"""PDDirect2HostManager: decode-side pinned host buffer + async H2D pipeline.

Owns a "transit" host KV pool that prefill RDMA-writes into directly via
mooncake, then asynchronously copies pages onto the device on a dedicated
CUDA stream. The device allocator is only touched right before the H2D
copy starts, so the device KV-cache is freed up for larger decode batches.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.nvtx_utils import nvtx_annotated_method

try:
    from transfer_kernel.kvcacheio import transfer_kv_multi_pool_fuse
except ImportError:  # pragma: no cover
    transfer_kv_multi_pool_fuse = None

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)

# TODO: support other layout.
_TRANSIT_LAYOUT = "layer_first"
_MULTI_POOL_FUSE_NUM_WARPS_PER_BLOCK = 16


class PDD2HAbortDisposition(Enum):
    REMOVE_NOW = auto()
    DEFER_UNTIL_H2D_DONE = auto()


class PDDirect2HostManager:
    """Decode-side pinned host pool + async H2D pipeline."""

    def __init__(
        self,
        req_to_token_pool: "ReqToTokenPool",
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
        server_args: "ServerArgs",
        sliding_window_size: Optional[int] = None,
        draft_token_to_kv_pool: Optional[KVCache] = None,
        draft_token_to_kv_pool_list: Optional[List[KVCache]] = None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.server_args = server_args
        self.sliding_window_size = sliding_window_size
        self.page_size = token_to_kv_pool_allocator.page_size

        device_pool = token_to_kv_pool_allocator.get_kvcache()

        draft_pools = list(draft_token_to_kv_pool_list or [])
        if not draft_pools and draft_token_to_kv_pool is not None:
            draft_pools.append(draft_token_to_kv_pool)

        self.transit_pools = [self._make_host_kv_pool(device_pool)]
        self.transit_state_pools: List[MHATokenToKVPoolHost] = []
        target_state_pool = self._make_host_state_pool(device_pool)
        if target_state_pool is not None:
            self.transit_state_pools.append(target_state_pool)

        for draft_pool in draft_pools:
            self.transit_pools.append(self._make_host_kv_pool(draft_pool))
            draft_state_pool = self._make_host_state_pool(draft_pool)
            if draft_state_pool is not None:
                self.transit_state_pools.append(draft_state_pool)

        if target_state_pool is None and self.transit_state_pools:
            raise NotImplementedError(
                "PD direct-to-host only supports draft state pools when the "
                "target pool also has SWA state."
            )

        self.transit_pool = self.transit_pools[0]
        self.transit_state_pool = (
            self.transit_state_pools[0] if self.transit_state_pools else None
        )

        for pool in self.transit_pools[1:]:
            if pool.size < self.transit_pool.size:
                raise RuntimeError(
                    "PD direct-to-host draft host pool is smaller than target "
                    f"host pool: draft={pool.size}, target={self.transit_pool.size}"
                )
        if self.transit_state_pool is not None:
            for pool in self.transit_state_pools[1:]:
                if pool.size < self.transit_state_pool.size:
                    raise RuntimeError(
                        "PD direct-to-host draft state host pool is smaller than "
                        "target state host pool: "
                        f"draft={pool.size}, target={self.transit_state_pool.size}"
                    )

        h2d_stream_priority, _ = torch.cuda.Stream.priority_range()
        self.h2d_stream = torch.cuda.Stream(priority=h2d_stream_priority)
        self.h2d_gate_event: Optional[torch.cuda.Event] = None
        # ack_id -> (req, host_indices, device_indices, event, state_indices, keepalive)
        self.ongoing_h2d: Dict[
            int,
            Tuple[
                "Req",
                torch.Tensor,
                torch.Tensor,
                torch.cuda.Event,
                Optional[torch.Tensor],
                List[Any],
            ],
        ] = {}
        # Requests that the scheduler has aborted between start_h2d and the
        # event firing. Keyed by id(req) and handled when the H2D is completed.
        self.aborted_req_ids: Set[int] = set()
        self._next_ack_id = 0
        self.retract_in_progress = False

        logger.info(
            "PDDirect2HostManager: transit pool size=%d num_pools=%d "
            "(state=%s num_state_pools=%d)",
            self.transit_pool.size,
            len(self.transit_pools),
            (
                self.transit_state_pool.size
                if self.transit_state_pool is not None
                else None
            ),
            len(self.transit_state_pools),
        )

    def _allocatable_tokens(self, req: "Req") -> Tuple[int, int, int, int]:
        host_indices = getattr(req, "host_kv_indices", None)
        if host_indices is None:
            return 0, 0, 0, 0

        need_full = len(host_indices)
        host_state_indices = getattr(req, "host_state_indices", None)
        need_swa = len(host_state_indices) if host_state_indices is not None else 0

        if hasattr(self.token_to_kv_pool_allocator, "full_available_size"):
            full_available = self.token_to_kv_pool_allocator.full_available_size()
            swa_available = self.token_to_kv_pool_allocator.swa_available_size()
        else:
            full_available = self.token_to_kv_pool_allocator.available_size()
            swa_available = 0

        return need_full, full_available, need_swa, swa_available

    def _has_device_room_for_h2d(
        self, req: "Req", reserved_decode_tokens: int
    ) -> bool:
        need_full, full_available, need_swa, swa_available = self._allocatable_tokens(
            req
        )
        if need_full + reserved_decode_tokens > full_available:
            return False
        if need_swa > swa_available:
            return False
        return True

    def can_admit_h2d(self, req: "Req", reserved_decode_tokens: int) -> bool:
        if self.retract_in_progress:
            return False
        if self._h2d_gate_busy():
            return False
        return self._has_device_room_for_h2d(req, reserved_decode_tokens)

    def begin_retract(self) -> None:
        self.retract_in_progress = True
        if self.ongoing_h2d:
            self.h2d_stream.synchronize()
            self.h2d_gate_event = None

    def end_retract(self) -> None:
        self.retract_in_progress = False

    # Pool-level handshake info

    def _make_host_kv_pool(self, device_pool: KVCache):
        host_size = self.server_args.pd_direct2host_host_size
        host_ratio = self.server_args.pd_direct2host_host_ratio

        if isinstance(device_pool, SWAKVPool):
            device_pool = device_pool.full_kv_pool
        if isinstance(device_pool, MLATokenToKVPool):
            return MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=host_ratio,
                host_size=host_size if host_size is not None else 0,
                page_size=self.page_size,
                layout=_TRANSIT_LAYOUT,
            )
        if isinstance(device_pool, MHATokenToKVPool):
            return MHATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=host_ratio,
                host_size=host_size if host_size is not None else 0,
                page_size=self.page_size,
                layout=_TRANSIT_LAYOUT,
            )
        raise NotImplementedError(
            f"PDDirect2HostManager does not support device pool "
            f"{type(device_pool).__name__}"
        )

    def _make_host_state_pool(self, device_pool: KVCache):
        if not isinstance(device_pool, SWAKVPool):
            return None
        host_size = self.server_args.pd_direct2host_host_size
        host_ratio = self.server_args.pd_direct2host_host_ratio
        return MHATokenToKVPoolHost(
            device_pool=device_pool.swa_kv_pool,
            host_to_device_ratio=host_ratio,
            host_size=host_size if host_size is not None else 0,
            page_size=self.page_size,
            layout=_TRANSIT_LAYOUT,
        )

    def get_host_kv_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs, data_lens, item_lens = [], [], []
        for pool in self.transit_pools:
            ptrs, lens, items = self._buf_infos_for(pool)
            data_ptrs += ptrs
            data_lens += lens
            item_lens += items
        return data_ptrs, data_lens, item_lens

    def get_host_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs, data_lens, item_lens = [], [], []
        for pool in self.transit_state_pools:
            ptrs, lens, items = self._buf_infos_for(pool)
            data_ptrs += ptrs
            data_lens += lens
            item_lens += items
        return data_ptrs, data_lens, item_lens

    def _buf_infos_for(self, host_pool) -> Tuple[List[int], List[int], List[int]]:
        """Mirror device-side get_contiguous_buf_infos for a host pool.

        Returns (data_ptrs, data_lens, item_lens). For MHA it's
        [k_layer_0..k_layer_{L-1}, v_layer_0..v_layer_{L-1}]; for MLA it's
        [layer_0..layer_{L-1}]. layer_first layout guarantees each per-layer
        slice is a contiguous tensor view.
        """
        if isinstance(host_pool, MHATokenToKVPoolHost):
            k_buffer = host_pool.k_buffer
            v_buffer = host_pool.v_buffer
            layer_num = host_pool.layer_num
            data_ptrs = [k_buffer[i].data_ptr() for i in range(layer_num)] + [
                v_buffer[i].data_ptr() for i in range(layer_num)
            ]
            data_lens = [k_buffer[i].nbytes for i in range(layer_num)] + [
                v_buffer[i].nbytes for i in range(layer_num)
            ]
            item_lens = [
                k_buffer[i][0].nbytes * self.page_size for i in range(layer_num)
            ] + [v_buffer[i][0].nbytes * self.page_size for i in range(layer_num)]
            return data_ptrs, data_lens, item_lens
        if isinstance(host_pool, MLATokenToKVPoolHost):
            kv_buffer = host_pool.kv_buffer
            layer_num = host_pool.layer_num
            data_ptrs = [kv_buffer[i].data_ptr() for i in range(layer_num)]
            data_lens = [kv_buffer[i].nbytes for i in range(layer_num)]
            item_lens = [
                kv_buffer[i][0].nbytes * self.page_size for i in range(layer_num)
            ]
            return data_ptrs, data_lens, item_lens
        raise TypeError(f"Unsupported host pool {type(host_pool).__name__}")

    # Per-request host-side preallocation

    def pre_alloc(self, req: "Req") -> bool:
        """Allocate host transit slots and request bookkeeping for a decode req.

        Device KV slots are allocated later in ``start_h2d`` after prefill has
        finished writing into the host transit pool.
        """
        seq_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        need_pages = (seq_len + self.page_size - 1) // self.page_size
        need_size = need_pages * self.page_size

        host_indices = self.transit_pool.alloc(need_size)
        if host_indices is None:
            return False

        host_state_indices: Optional[torch.Tensor] = None
        if self.transit_state_pool is not None:
            window = (
                self.sliding_window_size
                if self.sliding_window_size is not None
                else seq_len
            )
            window_start = max(0, seq_len - window)
            window_start = (window_start // self.page_size) * self.page_size
            swa_buffer_len = need_size - window_start
            host_state_indices = self.transit_state_pool.alloc(swa_buffer_len)
            if host_state_indices is None:
                self.transit_pool.free(host_indices)
                return False
            logger.debug(
                "[pd_d2h pre_alloc] rid=%s seq_len=%d need_size=%d "
                "window_start=%d swa_buffer_len=%d",
                getattr(req, "rid", "?"),
                seq_len,
                need_size,
                window_start,
                swa_buffer_len,
            )

        req_pool_indices = self.req_to_token_pool.alloc([req])
        if req_pool_indices is None:
            self.transit_pool.free(host_indices)
            if host_state_indices is not None and self.transit_state_pool is not None:
                self.transit_state_pool.free(host_state_indices)
            return False
        req.req_pool_idx = req_pool_indices[0]
        req.host_kv_indices = host_indices
        req.host_state_indices = host_state_indices
        req.host_state_seq_len = seq_len

        req.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_input_len(len(req.fill_ids))

        if self.sliding_window_size is not None:
            swa_evicted_seqlen = max(seq_len - self.sliding_window_size, 0)
        else:
            swa_evicted_seqlen = 0
        if self.page_size > 1:
            swa_evicted_seqlen = (
                swa_evicted_seqlen // self.page_size
            ) * self.page_size
        req.swa_evicted_seqlen = swa_evicted_seqlen
        return True

    # Start async H2D after prefill transfer succeeds

    @nvtx_annotated_method("pd_d2h.start_h2d")
    def start_h2d(self, req: "Req") -> bool:
        """Allocate device slots and enqueue the host->device copy.

        Returns False if device allocation unexpectedly fails after admission.

        For SWA models the device allocator's pre_alloc returns full-pool
        indices and stashes the matching SWA-pool indices in its internal
        full->swa mapping; we translate to the SWA-pool indices here when
        we need to schedule the per-layer SWA H2D.
        """
        host_indices = req.host_kv_indices
        need_size = host_indices.shape[0]
        seq_len_orig = getattr(req, "host_state_seq_len", need_size)

        io_backend = self.server_args.hicache_io_backend
        keepalive: List[Any] = []

        alloc_stream = torch.cuda.current_stream()
        device_indices = self._device_alloc(need_size, seq_len=seq_len_orig)
        if device_indices is None:
            return False

        device_state_indices: Optional[torch.Tensor] = None
        if self.transit_state_pool is not None:
            translator = getattr(
                self.token_to_kv_pool_allocator,
                "translate_loc_from_full_to_swa",
                None,
            )
            if translator is None:
                # SWA transit pool was constructed but the allocator does not
                # have an SWA mapping — bug in either dispatch or allocator.
                self.token_to_kv_pool_allocator.free(device_indices)
                raise RuntimeError(
                    "SWA transit pool present but allocator has no "
                    "translate_loc_from_full_to_swa()."
                )

            swa_buffer_len = req.host_state_indices.shape[0]
            device_state_indices = translator(device_indices[-swa_buffer_len:]).to(
                torch.int64
            )
            logger.debug(
                "[pd_d2h start_h2d] rid=%s device_indices_len=%d "
                "swa_buffer_len=%d device_state_indices_len=%d",
                getattr(req, "rid", "?"),
                device_indices.shape[0],
                swa_buffer_len,
                device_state_indices.shape[0],
            )

        kv_host_indices_for_io = host_indices
        kv_device_indices_for_io = device_indices
        state_host_indices_for_io = getattr(req, "host_state_indices", None)
        state_device_indices_for_io = device_state_indices

        if io_backend == "kernel":
            kv_host_indices_for_io = host_indices.to(
                device_indices.device, non_blocking=True
            )
            keepalive.append(kv_host_indices_for_io)
            if state_host_indices_for_io is not None:
                state_host_indices_for_io = state_host_indices_for_io.to(
                    device_indices.device, non_blocking=True
                )
                keepalive.append(state_host_indices_for_io)

        self._write_req_to_token(req, device_indices)
        alloc_event = torch.cuda.Event()
        alloc_event.record(alloc_stream)
        keepalive.append(alloc_event)

        with torch.cuda.stream(self.h2d_stream):
            self.h2d_stream.wait_event(alloc_event)
            if not self._load_to_device_multi_pool_groups(
                groups=[
                    (
                        self.transit_pools,
                        kv_host_indices_for_io,
                        kv_device_indices_for_io,
                    ),
                    (
                        self.transit_state_pools,
                        state_host_indices_for_io,
                        state_device_indices_for_io,
                    ),
                ],
                io_backend=io_backend,
                keepalive=keepalive,
            ):
                for transit_pool in self.transit_pools:
                    self._load_to_device_pool(
                        transit_pool=transit_pool,
                        host_indices=kv_host_indices_for_io,
                        device_indices=kv_device_indices_for_io,
                        io_backend=io_backend,
                    )
                if self.transit_state_pools:
                    for transit_state_pool in self.transit_state_pools:
                        self._load_to_device_pool(
                            transit_pool=transit_state_pool,
                            host_indices=state_host_indices_for_io,
                            device_indices=state_device_indices_for_io,
                            io_backend=io_backend,
                        )
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
            self.h2d_gate_event = event

        ack_id = self._next_ack_id
        self._next_ack_id += 1
        req.pd_d2h_ack_id = ack_id
        self.ongoing_h2d[ack_id] = (
            req,
            host_indices,
            device_indices,
            event,
            device_state_indices,
            keepalive,
        )
        return True

    # Query and complete finished H2D copies

    def is_h2d_ready(self, req: "Req") -> bool:
        ack_id = getattr(req, "pd_d2h_ack_id", None)
        if ack_id is None:
            return False
        entry = self.ongoing_h2d.get(ack_id)
        if entry is None:
            return False
        return entry[3].query()

    def _h2d_gate_busy(self) -> bool:
        event = self.h2d_gate_event
        if event is None:
            return False
        if event.query():
            self.h2d_gate_event = None
            return False
        return True

    def complete_h2d(self, req: "Req") -> Optional["Req"]:
        ack_id = getattr(req, "pd_d2h_ack_id", None)
        if ack_id is None:
            return None
        entry = self.ongoing_h2d.get(ack_id)
        if entry is None:
            return None
        (
            req,
            host_indices,
            device_indices,
            event,
            _device_state_indices,
            _io_keepalive,
        ) = entry
        if not event.query():
            return None
        torch.cuda.current_stream().wait_event(event)
        del self.ongoing_h2d[ack_id]
        req.pd_d2h_ack_id = None
        if id(req) in self.aborted_req_ids:
            self.aborted_req_ids.discard(id(req))
            self._free_host_slots(req, host_indices)
            self.token_to_kv_pool_allocator.free(device_indices)
            self._free_req_slot_if_allocated(req)
            return None
        self._free_host_slots(req, host_indices)
        return req

    # ---- failure / cancel paths ----

    def abort(self, req: "Req") -> PDD2HAbortDisposition:
        """Abort host/device resources and report when queue removal is safe."""
        if any(id(req) == id(other) for (other, *_rest) in self.ongoing_h2d.values()):
            self.aborted_req_ids.add(id(req))
            return PDD2HAbortDisposition.DEFER_UNTIL_H2D_DONE
        host_indices = getattr(req, "host_kv_indices", None)
        if host_indices is not None:
            self._free_host_slots(req, host_indices)
        self._free_req_slot_if_allocated(req)
        return PDD2HAbortDisposition.REMOVE_NOW

    def release_failed(self, req: "Req") -> None:
        """Release host/device state for a transfer that prefill reported
        as Failed before any H2D was started."""
        host_indices = getattr(req, "host_kv_indices", None)
        if host_indices is not None:
            self._free_host_slots(req, host_indices)
        self._free_req_slot_if_allocated(req)

    # ---- helpers ----

    def _device_alloc(
        self,
        num_tokens: int,
        seq_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if hasattr(self.token_to_kv_pool_allocator, "pre_alloc"):
            chunked_size = num_tokens
            if self.sliding_window_size is not None:
                # Keep this formula aligned with the host-state swa_buffer_len
                # in pre_alloc(); otherwise full->SWA translation can hit slot 0.
                effective_seq_len = seq_len if seq_len is not None else num_tokens
                align_start = (
                    max(effective_seq_len - self.sliding_window_size, 0)
                    // self.page_size
                    * self.page_size
                )
                chunked_size = num_tokens - align_start
            result = self.token_to_kv_pool_allocator.pre_alloc(
                num_tokens,
                chunked_size,
            )
            if result is not None and self.sliding_window_size is not None:
                W = self.sliding_window_size
                ps = self.page_size
                effective_seq_len = seq_len if seq_len is not None else num_tokens
                align_start_alloc = max(effective_seq_len - W, 0) // ps * ps
                swa_need_size_alloc = num_tokens - align_start_alloc
                logger.debug(
                    "[pd_d2h _device_alloc] num_tokens=%d seq_len=%s "
                    "align_start_alloc=%d swa_need_size_alloc=%d "
                    "device_indices_len=%d",
                    num_tokens,
                    seq_len,
                    align_start_alloc,
                    swa_need_size_alloc,
                    result.shape[0],
                )
            return result
        if self.page_size == 1:
            return self.token_to_kv_pool_allocator.alloc(num_tokens)
        device = self.token_to_kv_pool_allocator.device
        prefix_lens = torch.empty((1,), dtype=torch.int64, device=device)
        prefix_lens.fill_(0)
        seq_lens = torch.empty((1,), dtype=torch.int64, device=device)
        seq_lens.fill_(num_tokens)
        last_loc = torch.empty((1,), dtype=torch.int64, device=device)
        last_loc.fill_(-1)
        return self.token_to_kv_pool_allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=seq_lens,
            seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int64),
            last_loc=last_loc,
            extend_num_tokens=num_tokens,
        )

    def _load_to_device_pool(
        self,
        transit_pool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if transit_pool.layer_num <= 0:
            return

        load_all_layers = getattr(transit_pool, "load_to_device_all_layer", None)
        if io_backend == "direct" and load_all_layers is not None:
            load_all_layers(
                device_pool=transit_pool.device_pool,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend=io_backend,
            )
            return

        # The kernel backend has fused all-layer helpers too. Keep PD D2H on
        # per-layer H2D for now; this is the future switch point after fused
        # all-layer H2D is validated for the direct-to-host pipeline.
        for layer_id in range(
            transit_pool.start_layer,
            transit_pool.start_layer + transit_pool.layer_num,
        ):
            transit_pool.load_to_device_per_layer(
                device_pool=transit_pool.device_pool,
                host_indices=host_indices,
                device_indices=device_indices,
                layer_id=layer_id,
                io_backend=io_backend,
            )

    def _load_to_device_multi_pool_groups(
        self,
        groups,
        io_backend: str,
        keepalive: List[Any],
    ) -> bool:
        if io_backend != "kernel" or transfer_kv_multi_pool_fuse is None:
            return False

        descriptors = []
        for transit_pools, host_indices, device_indices in groups:
            if not transit_pools:
                continue
            if host_indices is None or device_indices is None:
                return False
            for transit_pool in transit_pools:
                get_desc = getattr(transit_pool, "get_multi_pool_h2d_descriptors", None)
                if get_desc is None:
                    return False
                desc = get_desc(
                    device_pool=transit_pool.device_pool,
                    host_indices=host_indices,
                    device_indices=device_indices,
                    io_backend=io_backend,
                )
                if desc is None:
                    return False
                descriptors.append(desc)
        if not descriptors:
            return False

        src_k_layers = []
        dst_k_layers = []
        src_v_layers = []
        dst_v_layers = []
        src_indices = []
        dst_indices = []
        pool_num_items = []
        pool_k_item_sizes = []
        pool_v_item_sizes = []
        layer_to_pool = []
        layer_to_pool_layer = []
        max_num_items = 0
        block_quota = None

        for desc in descriptors:
            pool_base = len(pool_num_items)
            src_k_layers.append(desc["src_k_layer_ptrs"])
            dst_k_layers.append(desc["dst_k_layer_ptrs"])
            src_v_layers.append(desc["src_v_layer_ptrs"])
            dst_v_layers.append(desc["dst_v_layer_ptrs"])
            src_indices.extend(desc["src_indices"])
            dst_indices.extend(desc["dst_indices"])
            pool_num_items.extend(desc["pool_num_items"])
            pool_k_item_sizes.extend(desc["pool_k_item_sizes"])
            pool_v_item_sizes.extend(desc["pool_v_item_sizes"])
            desc_layer_to_pool = desc.get("layer_to_pool")
            desc_layer_to_pool_layer = desc.get("layer_to_pool_layer")
            if desc_layer_to_pool is None:
                desc_layer_to_pool = [0] * desc["src_k_layer_ptrs"].numel()
            if desc_layer_to_pool_layer is None:
                desc_layer_to_pool_layer = list(range(desc["src_k_layer_ptrs"].numel()))
            layer_to_pool.extend([pool_base + x for x in desc_layer_to_pool])
            layer_to_pool_layer.extend(desc_layer_to_pool_layer)
            max_num_items = max(max_num_items, int(desc["max_num_items"]))
            block_quota = desc["block_quota"] if block_quota is None else block_quota

        device = device_indices.device
        src_indices_ptrs = torch.tensor(
            [x.data_ptr() for x in src_indices], dtype=torch.uint64, device=device
        )
        dst_indices_ptrs = torch.tensor(
            [x.data_ptr() for x in dst_indices], dtype=torch.uint64, device=device
        )
        layer_to_pool_tensor = torch.tensor(
            layer_to_pool, dtype=torch.int64, device=device
        )
        layer_to_pool_layer_tensor = torch.tensor(
            layer_to_pool_layer, dtype=torch.int64, device=device
        )
        pool_num_items_tensor = torch.tensor(
            pool_num_items, dtype=torch.int64, device=device
        )
        pool_k_item_sizes_tensor = torch.tensor(
            pool_k_item_sizes, dtype=torch.int64, device=device
        )
        pool_v_item_sizes_tensor = torch.tensor(
            pool_v_item_sizes, dtype=torch.int64, device=device
        )
        launch_tensors = [
            torch.cat(src_k_layers),
            torch.cat(dst_k_layers),
            torch.cat(src_v_layers),
            torch.cat(dst_v_layers),
            src_indices_ptrs,
            dst_indices_ptrs,
            layer_to_pool_tensor,
            layer_to_pool_layer_tensor,
            pool_num_items_tensor,
            pool_k_item_sizes_tensor,
            pool_v_item_sizes_tensor,
        ]
        keepalive.extend(launch_tensors)

        transfer_kv_multi_pool_fuse(
            *launch_tensors,
            max_num_items,
            block_quota,
            num_warps_per_block=_MULTI_POOL_FUSE_NUM_WARPS_PER_BLOCK,
        )
        return True

    def _write_req_to_token(self, req: "Req", device_indices: torch.Tensor) -> None:
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, len(device_indices))), device_indices
        )

    def _free_host_slots(self, req: "Req", host_indices: torch.Tensor) -> None:
        self.transit_pool.free(host_indices)
        host_state_indices = getattr(req, "host_state_indices", None)
        if host_state_indices is not None and self.transit_state_pool is not None:
            self.transit_state_pool.free(host_state_indices)
        req.host_kv_indices = None
        req.host_state_indices = None

    def _free_req_slot_if_allocated(self, req: "Req") -> None:
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None or req_pool_idx == -1:
            return
        self.req_to_token_pool.free(req)
