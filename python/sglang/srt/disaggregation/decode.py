"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.pd_direct2host_manager import (
    PDD2HAbortDisposition,
    PDDirect2HostManager,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    poll_and_all_reduce,
    poll_and_all_reduce_with_staging,
    prepare_abort,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.nvtx_utils import nvtx_annotated_method
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils.common import ceil_align, floor_align
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()

REQUEST_ABORT_ERR_TYPE = "request_abort"


class PDD2HState(Enum):
    # Waiting for prefill to RDMA-write KV into decode host slots and send metadata.
    WAIT_PREFILL = auto()
    # Prefill transfer is committed; host KV is ready, but H2D has not started.
    HOST_READY = auto()
    # Device KV is allocated and host-to-device copy has been enqueued.
    H2D_INFLIGHT = auto()
    # The request is aborted, but device slots remain owned until H2D completes.
    ABORT_PENDING_H2D = auto()


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        if mamba_size is not None:
            effective_mamba_size = min(mamba_size, size + pre_alloc_size)
            if mamba_size > size + pre_alloc_size:
                logger.warning(
                    "mamba_size (%d) exceeds size + pre_alloc_size (%d), "
                    "capping effective_mamba_size to %d",
                    mamba_size,
                    size + pre_alloc_size,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = size + pre_alloc_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    pd_d2h_state: PDD2HState = PDD2HState.WAIT_PREFILL

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    def abort_receiver_if_active(self) -> None:
        if self.kv_receiver is not None:
            self.kv_receiver.abort()


def _is_request_abort(req: Req) -> bool:
    return isinstance(req.finished_reason, FINISH_ABORT) and (
        req.finished_reason.err_type == REQUEST_ABORT_ERR_TYPE
    )


def _notify_request_abort(scheduler: Scheduler, decode_req: DecodeRequest) -> None:
    if _is_request_abort(decode_req.req):
        return
    decode_req.req.finished_reason = FINISH_ABORT(err_type=REQUEST_ABORT_ERR_TYPE)
    scheduler.send_to_tokenizer.send_output(
        AbortReq(rid=decode_req.req.rid), decode_req.req
    )


class PDD2HFusedPoller:
    """Fixed gloo collective sequence per scheduler iteration for pd-d2h.

    Replaces the two independent — and conditionally executed — all_gather
    pairs in DecodePreallocQueue and DecodeTransferQueue with a fixed header,
    and queue snapshot sequence. This removes queue-state-dependent collective
    branching across ranks.

    The fixed-size header carries a monotonically increasing sequence number.
    After the header gather, all ranks verify that the sequence matches before
    issuing the optional queue snapshot gather.
    """

    def __init__(self, gloo_group: ProcessGroup, tp_rank: int):
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.seq = 0

    def poll(
        self,
        prealloc_entries: List[Tuple[int, int, int, int, int, int]],
        transfer_entries: List[Tuple[int, int, int, int, int, int]],
        admission_snapshot: Tuple[int, int, int, int, int],
    ) -> Tuple[Dict[int, int], Dict[int, int], Optional[int], bool]:
        """Gather queue snapshots and select a TP-wide H2D candidate room.

        Returns (prealloc_polls_by_room, transfer_polls_by_room,
        h2d_candidate_room, h2d_admit).
        """
        group = self.gloo_group
        world_size = torch.distributed.get_world_size(group=group)
        self.seq += 1

        header = torch.tensor(
            [
                self.seq,
                len(prealloc_entries),
                len(transfer_entries),
                *admission_snapshot,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        gathered_headers = torch.empty(
            (world_size, 8), dtype=torch.int64, device="cpu"
        )
        torch.distributed.all_gather_into_tensor(
            gathered_headers.flatten(), header, group=group
        )

        seqs = gathered_headers[:, 0].tolist()
        if len(set(seqs)) != 1:
            raise RuntimeError(
                "pd-d2h collective schedule desync detected: "
                f"tp_rank={self.tp_rank} local(seq={self.seq}) "
                f"gathered seqs={seqs}. "
                "A scheduler rank diverged from the collective lockstep; "
                "raising to trigger a clean restart instead of deadlocking."
            )

        counts_p = gathered_headers[:, 1].tolist()
        counts_t = gathered_headers[:, 2].tolist()
        max_p = max(counts_p)
        max_t = max(counts_t)
        if max_p + max_t == 0:
            return {}, {}, None, False

        stride = max_p + max_t
        local_entries = torch.full(
            (stride, 6), -1, dtype=torch.int64, device="cpu"
        )
        for i, entry in enumerate(prealloc_entries):
            local_entries[i] = torch.tensor(entry, dtype=torch.int64)
        for i, entry in enumerate(transfer_entries):
            local_entries[max_p + i] = torch.tensor(entry, dtype=torch.int64)

        gathered_entries = torch.empty(
            (world_size, stride, 6), dtype=torch.int64, device="cpu"
        )
        torch.distributed.all_gather_into_tensor(
            gathered_entries.flatten(), local_entries.flatten(), group=group
        )

        prealloc_polls = self._consensus_polls_by_room(
            gathered_entries[:, :max_p],
            counts_p,
            world_size,
            int(KVPoll.Bootstrapping),
        )
        transfer_polls = self._consensus_transfer_polls_by_room(
            gathered_entries[:, max_p:],
            counts_t,
            world_size,
        )
        h2d_candidate_room = self._common_room_with_state(
            gathered_entries[:, max_p:],
            counts_t,
            world_size,
            PDD2HState.HOST_READY.value,
        )
        h2d_admit = self._h2d_admit_from_snapshot(
            h2d_candidate_room,
            gathered_headers,
            gathered_entries[:, max_p:],
            counts_t,
        )
        return prealloc_polls, transfer_polls, h2d_candidate_room, h2d_admit

    @staticmethod
    def _consensus_polls_by_room(
        gathered_entries: torch.Tensor,
        counts: List[int],
        world_size: int,
        fallback_status: int,
    ) -> Dict[int, int]:
        """Compute per-room consensus poll status from gathered queue entries.

        A room reaches a non-fallback status only when every rank reported it;
        Failed wins immediately; otherwise the min status across ranks is used.
        """
        present_counts: Dict[int, int] = {}
        min_status_by_room: Dict[int, int] = {}
        for entries, count in zip(gathered_entries, counts):
            seen_rooms = set()
            for i in range(count):
                room = int(entries[i, 0].item())
                poll = int(entries[i, 1].item())
                if room in seen_rooms:
                    min_status_by_room[room] = min(
                        min_status_by_room.get(room, poll), poll
                    )
                    continue
                seen_rooms.add(room)
                present_counts[room] = present_counts.get(room, 0) + 1
                min_status_by_room[room] = min(
                    min_status_by_room.get(room, poll), poll
                )

        polls_by_room: Dict[int, int] = {}
        for room, min_status in min_status_by_room.items():
            if min_status == KVPoll.Failed:
                polls_by_room[room] = int(KVPoll.Failed)
            elif present_counts.get(room, 0) == world_size:
                polls_by_room[room] = min_status
            else:
                polls_by_room[room] = int(fallback_status)
        return polls_by_room

    @staticmethod
    def _common_room_with_state(
        gathered_entries: torch.Tensor,
        counts: List[int],
        world_size: int,
        required_state: int,
    ) -> Optional[int]:
        """Return the smallest room reported in the required state by every rank."""
        states_by_room: Dict[int, Dict[int, int]] = {}
        for rank, (entries, count) in enumerate(zip(gathered_entries, counts)):
            for i in range(count):
                room = int(entries[i, 0].item())
                state = int(entries[i, 2].item())
                states_by_room.setdefault(room, {})[rank] = state

        common_rooms = [
            room
            for room, rank_states in states_by_room.items()
            if len(rank_states) == world_size
            and all(state == required_state for state in rank_states.values())
        ]
        return min(common_rooms) if common_rooms else None

    @classmethod
    def _consensus_transfer_polls_by_room(
        cls,
        gathered_entries: torch.Tensor,
        counts: List[int],
        world_size: int,
    ) -> Dict[int, int]:
        polls_by_room = cls._consensus_polls_by_room(
            gathered_entries,
            counts,
            world_size,
            int(KVPoll.Transferring),
        )
        entries_by_room: Dict[int, Dict[int, Tuple[int, int, int]]] = {}
        for rank, (entries, count) in enumerate(zip(gathered_entries, counts)):
            for i in range(count):
                room = int(entries[i, 0].item())
                entries_by_room.setdefault(room, {})[rank] = (
                    int(entries[i, 1].item()),
                    int(entries[i, 2].item()),
                    int(entries[i, 3].item()),
                )

        for room, poll in list(polls_by_room.items()):
            if poll != KVPoll.Success:
                continue
            rank_entries = entries_by_room.get(room, {})
            if len(rank_entries) != world_size:
                polls_by_room[room] = int(KVPoll.Transferring)
                continue
            if all(
                state == PDD2HState.WAIT_PREFILL.value
                for _, state, _ in rank_entries.values()
            ):
                metadata_rooms = [
                    metadata_room for _, _, metadata_room in rank_entries.values()
                ]
                if any(
                    metadata_room not in (0, room)
                    for metadata_room in metadata_rooms
                ):
                    polls_by_room[room] = int(KVPoll.Failed)
                elif not all(
                    metadata_room == room for metadata_room in metadata_rooms
                ):
                    polls_by_room[room] = int(KVPoll.Transferring)
        return polls_by_room

    @staticmethod
    def _h2d_admit_from_snapshot(
        candidate_room: Optional[int],
        gathered_headers: torch.Tensor,
        gathered_transfer_entries: torch.Tensor,
        transfer_counts: List[int],
    ) -> bool:
        if candidate_room is None:
            return False

        for rank, (entries, count) in enumerate(
            zip(gathered_transfer_entries, transfer_counts)
        ):
            full_available = int(gathered_headers[rank, 3].item())
            swa_available = int(gathered_headers[rank, 4].item())
            reserved_decode_tokens = int(gathered_headers[rank, 5].item())
            retract_in_progress = int(gathered_headers[rank, 6].item())
            h2d_gate_busy = int(gathered_headers[rank, 7].item())

            candidate_entry = next(
                (
                    entries[i]
                    for i in range(count)
                    if int(entries[i, 0].item()) == candidate_room
                ),
                None,
            )
            if candidate_entry is None:
                return False
            need_full = int(candidate_entry[4].item())
            need_swa = int(candidate_entry[5].item())
            if not PDDirect2HostManager.can_admit_h2d_from_snapshot(
                need_full,
                need_swa,
                (
                    full_available,
                    swa_available,
                    reserved_decode_tokens,
                    retract_in_progress,
                    h2d_gate_busy,
                ),
            ):
                return False
        return True

class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        draft_token_to_kv_pool_list: Optional[List[KVCache]],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self._has_pre_alloc = hasattr(token_to_kv_pool_allocator, "pre_alloc")
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.draft_token_to_kv_pool_list = draft_token_to_kv_pool_list or []
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        assert num_reserved_decode_tokens % self.token_to_kv_pool_allocator.page_size == 0
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)
        self._init_hybrid_extra_tokens()

    def _init_hybrid_extra_tokens(self):
        self.evict_len_per_step = None
        self.swa_extra_need_tokens = 0
        if self.scheduler.model_config.is_hybrid_swa:
            self.evict_len_per_step = envs.SGLANG_SWA_EVICT_LENGTH_PER_STEP.get()
            if not self.scheduler.spec_algorithm.is_none():
                self.swa_extra_need_tokens = max(
                    self.scheduler.server_args.speculative_num_steps
                    * self.scheduler.server_args.speculative_eagle_topk,
                    self.scheduler.server_args.speculative_num_draft_tokens,
                )
                self.swa_extra_need_tokens = ceil_align(
                    self.swa_extra_need_tokens, self.scheduler.page_size
                )

    @staticmethod
    def _to_metadata_indices(indices) -> np.ndarray:
        arr = (
            indices.detach().cpu().numpy()
            if isinstance(indices, torch.Tensor)
            else np.asarray(indices)
        )
        if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(np.int32).max):
            raise ValueError(
                f"KV metadata index out of int32 range: min={arr.min()} max={arr.max()}"
            )
        return arr.astype(np.int32, copy=False)

    def _pre_alloc_and_get_kv_indices(
        self, decode_req: DecodeRequest, pd_d2h_mgr, origin_input_len: int
    ) -> Optional[np.ndarray]:
        req = decode_req.req

        if pd_d2h_mgr is not None:
            if not pd_d2h_mgr.pre_alloc(req):
                return None
            return self._to_metadata_indices(req.host_kv_indices[:origin_input_len])

        dst_kv_indices = self._pre_alloc(req)
        if self.scheduler.enable_hisparse:
            return self._to_metadata_indices(dst_kv_indices[:origin_input_len])

        kv_indices_full = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :origin_input_len
        ]
        return self._to_metadata_indices(kv_indices_full)

    def _get_state_indices(
        self, decode_req: DecodeRequest, pd_d2h_mgr, page_size: int
    ):
        req = decode_req.req

        if pd_d2h_mgr is not None:
            if req.host_state_indices is None:
                return None
            state_indices = self._to_metadata_indices(req.host_state_indices)
            return kv_to_page_indices(state_indices, page_size)

        if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
            return [
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx
                ]
                .cpu()
                .numpy()
            ]

        if isinstance(self.token_to_kv_pool, SWAKVPool):
            seq_len = len(req.origin_input_ids)
            window_size = self.scheduler.swa_retention_window
            window_start = max(0, seq_len - window_size)
            window_start = (window_start // page_size) * page_size
            window_kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, window_start:seq_len
            ]
            window_kv_indices_swa = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    window_kv_indices_full
                )
            )
            state_indices = self._to_metadata_indices(window_kv_indices_swa)
            return kv_to_page_indices(state_indices, page_size)

        if isinstance(self.token_to_kv_pool, NSATokenToKVPool):
            seq_len = len(req.origin_input_ids)
            kv_indices_full = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ]
            state_indices = self._to_metadata_indices(kv_indices_full)
            return kv_to_page_indices(state_indices, self.token_to_kv_pool.page_size)

        return None

    def _bootstrap_room(self, decode_req: DecodeRequest) -> int:
        bootstrap_room = decode_req.req.bootstrap_room
        if bootstrap_room is None:
            raise RuntimeError(
                "Disagg decode prealloc request has no bootstrap_room: "
                f"rid={decode_req.req.rid}"
            )
        return int(bootstrap_room)

    def local_poll_entries(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Snapshot prealloc room, poll, local state, and metadata placeholder.

        Pure local computation — the collective exchange is done by
        PDD2HFusedPoller so that all ranks issue exactly one collective per
        scheduler iteration regardless of local queue state.
        """
        entries: List[Tuple[int, int, int, int, int, int]] = []
        for decode_req in self.queue:
            room = self._bootstrap_room(decode_req)
            if decode_req.waiting_for_input:
                entries.append((room, int(KVPoll.WaitingForInput), 1, -1, -1, -1))
            else:
                entries.append(
                    (room, int(decode_req.kv_receiver.poll()), 0, -1, -1, -1)
                )
        return entries

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        if self.scheduler.enable_hisparse:
            # Direct-to-host: register host pool pointers so P writes to D's host memory
            host_pool = self.scheduler.hisparse_coordinator.mem_pool_host
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                host_pool.get_contiguous_buf_infos()
            )
        else:
            kv_data_ptrs, kv_data_lens, kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
        if len(self.draft_token_to_kv_pool_list):
            for draft_kv_pool in self.draft_token_to_kv_pool_list:
                draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                    draft_kv_pool.get_contiguous_buf_infos()
                )
                kv_data_ptrs += draft_kv_data_ptrs
                kv_data_lens += draft_kv_data_lens
                kv_item_lens += draft_kv_item_lens
        elif self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        # HiSparse Host pool has page_size=1; use it when hisparse is enabled
        kv_args.page_size = (
            1 if self.scheduler.enable_hisparse else self.token_to_kv_pool.page_size
        )

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        if len(self.draft_token_to_kv_pool_list):
            for draft_kv_pool in self.draft_token_to_kv_pool_list:
                if hasattr(draft_kv_pool, "get_state_buf_infos"):
                    state_data_ptrs, state_data_lens, state_item_lens = (
                        draft_kv_pool.get_state_buf_infos()
                    )
                    kv_args.state_data_ptrs += state_data_ptrs
                    kv_args.state_data_lens += state_data_lens
                    kv_args.state_item_lens += state_item_lens
        elif hasattr(self.draft_token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.draft_token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs += state_data_ptrs
            kv_args.state_data_lens += state_data_lens
            kv_args.state_item_lens += state_item_lens

        # PD direct-to-host: when enabled, the decode side advertises
        # pinned host buffers as the RDMA destinations instead of device
        # KV pools. The manager owns the lifetime of those buffers; we
        # only forward the pool-level (ptr, len, item_len) tuples here.
        pd_d2h_mgr = getattr(self.scheduler, "pd_direct2host_manager", None)
        if pd_d2h_mgr is not None:
            (
                kv_args.host_kv_data_ptrs,
                kv_args.host_kv_data_lens,
                kv_args.host_kv_item_lens,
            ) = pd_d2h_mgr.get_host_kv_buf_infos()
            (
                kv_args.host_state_data_ptrs,
                kv_args.host_state_data_lens,
                kv_args.host_state_item_lens,
            ) = pd_d2h_mgr.get_host_state_buf_infos()
        else:
            kv_args.host_kv_data_ptrs = []
            kv_args.host_kv_data_lens = []
            kv_args.host_kv_item_lens = []
            kv_args.host_state_data_ptrs = []
            kv_args.host_state_data_lens = []
            kv_args.host_state_item_lens = []

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(req)

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def abort_matching(self, recv_req: AbortReq) -> None:
        matched_ids = set()
        for decode_req in self.queue:
            if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                matched_ids.add(id(decode_req))
                decode_req.abort_receiver_if_active()
                _notify_request_abort(self.scheduler, decode_req)

        if matched_ids:
            self.pending_reqs = [
                decode_req
                for decode_req in self.pending_reqs
                if id(decode_req) not in matched_ids
            ]

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if prefill_info.follow_bootstrap_room:
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def _full_required_tokens(
        self, req: Req, page_size: int, include_output_ids: bool = True
    ) -> int:
        req_len = req.seqlen if include_output_ids else len(req.origin_input_ids)
        return ceil_align(
            req_len + self.num_reserved_decode_tokens,
            page_size,
        )

    def _swa_required_tokens(
        self, req: Req, page_size: int, include_output_ids: bool = True
    ) -> int:
        req_len = req.seqlen if include_output_ids else len(req.origin_input_ids)
        return ceil_align(
            min(
                req_len,
                self.scheduler.swa_retention_window,
            )
            + min(self.num_reserved_decode_tokens, self.evict_len_per_step)
            + self.swa_extra_need_tokens,
            page_size,
        )

    def _swa_tokens_for_retracted_req(self, req: Req, page_size: int) -> int:
        return ceil_align(
            min(
                req.seqlen,
                self.scheduler.swa_retention_window,
            )
            + self.evict_len_per_step,
            page_size,
        )

    def _retractable_tokens(self) -> Tuple[int, Optional[int]]:
        retractable_tokens = 0
        swa_retractable_tokens = 0
        is_hybrid_swa = self.scheduler.model_config.is_hybrid_swa
        sliding_window_size = self.scheduler.swa_retention_window
        for req in self.scheduler.running_batch.reqs:
            req_len = req.seqlen
            retractable_tokens += req_len
            if is_hybrid_swa:
                swa_retractable_tokens += min(req_len, sliding_window_size)
        return retractable_tokens, (swa_retractable_tokens if is_hybrid_swa else None)

    def _max_full_tokens_for_req(
        self, req: Req, retractable_tokens: int, page_size: int
    ) -> int:
        return ceil_align(
            max(
                len(req.origin_input_ids)
                + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_tokens,
                0,
            ),
            page_size,
        )

    def _max_swa_tokens_for_req(
        self, req: Req, swa_retractable_tokens: int, page_size: int
    ) -> int:
        return ceil_align(
            max(
                min(
                    len(req.origin_input_ids),
                    self.scheduler.swa_retention_window,
                )
                + self.swa_extra_need_tokens
                + min(
                    req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                    self.evict_len_per_step,
                )
                - swa_retractable_tokens,
                0,
            ),
            page_size,
        )

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)
        page_size = self.token_to_kv_pool_allocator.page_size

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break
            if not self.scheduler.model_config.is_hybrid_swa:
                required_tokens_for_request = self._full_required_tokens(req, page_size)
                if required_tokens_for_request > allocatable_tokens:
                    break
                allocatable_tokens -= required_tokens_for_request
            else:
                full_allocatable_tokens, swa_allocatable_tokens = allocatable_tokens
                full_required_tokens_for_request = self._full_required_tokens(
                    req, page_size
                )
                swa_required_tokens_for_request = self._swa_required_tokens(
                    req, page_size
                )
                if (
                    full_required_tokens_for_request > full_allocatable_tokens
                    or swa_required_tokens_for_request > swa_allocatable_tokens
                ):
                    break
                full_allocatable_tokens -= full_required_tokens_for_request
                swa_allocatable_tokens -= swa_required_tokens_for_request
                allocatable_tokens = (full_allocatable_tokens, swa_allocatable_tokens)

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self,
        rids_to_check: Optional[List[str]] = None,
        polls_by_room: Optional[Dict[int, int]] = None,
    ) -> None:
        pd_d2h_mgr = getattr(self.scheduler, "pd_direct2host_manager", None)
        if not self.queue and pd_d2h_mgr is None:
            return

        if (
            pd_d2h_mgr is None
            and self.queue
            and all(decode_req.waiting_for_input for decode_req in self.queue)
        ):
            return

        if pd_d2h_mgr is not None:
            if polls_by_room is None:
                raise RuntimeError(
                    "PD D2H handshake polling requires fused poll consensus; "
                    "pipeline parallelism is not supported with PD D2H."
                )
            # Consensus was exchanged by PDD2HFusedPoller; local lookup only.
            polls = [
                polls_by_room.get(
                    self._bootstrap_room(decode_req), int(KVPoll.Bootstrapping)
                )
                for decode_req in self.queue
            ]
        else:
            polls = poll_and_all_reduce(
                [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
            )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                if _is_request_abort(decode_req.req):
                    continue
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    decode_req.abort_receiver_if_active()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self,
        rids_to_check: Optional[List[str]] = None,
        polls_by_room: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check, polls_by_room=polls_by_room)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens, swa_retractable_tokens = self._retractable_tokens()
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens,
            swa_retractable_tokens=swa_retractable_tokens,
            count_retracted=True,
        )
        if self.scheduler.enable_hisparse:
            page_size = 1  # host pool page_size
        else:
            page_size = self.token_to_kv_pool_allocator.page_size
        if self.scheduler.model_config.is_hybrid_swa:
            assert swa_retractable_tokens is not None

        # First, remove all failed requests from the queue
        pd_d2h_mgr = getattr(self.scheduler, "pd_direct2host_manager", None)
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                if not _is_request_abort(decode_req.req):
                    self.scheduler.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                if pd_d2h_mgr is not None:
                    pd_d2h_mgr.abort(decode_req.req)
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            req = decode_req.req
            required_tokens_for_request = self._full_required_tokens(
                req, page_size, include_output_ids=False
            )
            max_tokens_for_request = self._max_full_tokens_for_req(
                req, retractable_tokens, page_size
            )
            if not self.scheduler.model_config.is_hybrid_swa:
                if (
                    max(
                        required_tokens_for_request,
                        max_tokens_for_request,
                    )
                    > allocatable_tokens
                ):
                    break
                if required_tokens_for_request > allocatable_tokens:
                    break

                allocatable_tokens -= required_tokens_for_request
            else:
                full_allocatable_tokens, swa_allocatable_tokens = allocatable_tokens
                full_required_tokens_for_request = required_tokens_for_request
                swa_required_tokens_for_request = self._swa_required_tokens(
                    req, page_size, include_output_ids=False
                )
                full_max_tokens_for_request = max_tokens_for_request
                swa_max_tokens_for_request = self._max_swa_tokens_for_req(
                    req, swa_retractable_tokens, page_size
                )
                # Estimate whether this request would exceed the token limit in the most extreme case,
                # assuming all currently running requests are retracted.
                if (
                    max(
                        required_tokens_for_request,
                        full_max_tokens_for_request,
                    )
                    > full_allocatable_tokens
                ) or (
                    max(
                        swa_required_tokens_for_request,
                        swa_max_tokens_for_request,
                    )
                    > swa_allocatable_tokens
                ):
                    break
                if (
                    full_required_tokens_for_request > full_allocatable_tokens
                    or swa_required_tokens_for_request > swa_allocatable_tokens
                ):
                    logger.warning(
                        f"Not enough space for req: {decode_req.req.rid} {decode_req.req.bootstrap_room=}, "
                        f"full_required_tokens_for_request: {full_required_tokens_for_request} > full_allocatable_tokens: {full_allocatable_tokens}, "
                        f"swa_required_tokens_for_request: {swa_required_tokens_for_request} > swa_allocatable_tokens: {swa_allocatable_tokens}"
                    )
                    break
                full_allocatable_tokens -= full_required_tokens_for_request
                swa_allocatable_tokens -= swa_required_tokens_for_request
                allocatable_tokens = (full_allocatable_tokens, swa_allocatable_tokens)

            origin_input_len = len(decode_req.req.origin_input_ids)
            kv_indices = self._pre_alloc_and_get_kv_indices(
                decode_req, pd_d2h_mgr, origin_input_len
            )
            if kv_indices is None:
                # Host pool full — keep req in queue, retry next tick.
                break

            state_indices = self._get_state_indices(decode_req, pd_d2h_mgr, page_size)

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, page_size)
            decode_req.kv_receiver.send_metadata(
                page_indices, decode_req.metadata_buffer_index, state_indices
            )
            if (
                self.transfer_queue.enable_staging
                and decode_req.kv_receiver.require_staging
            ):
                self.transfer_queue.staging_handler.register_decode_req(
                    decode_req.req.bootstrap_room, decode_req
                )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()
            if self.scheduler.enable_metrics:
                ts = decode_req.req.time_stats
                self.scheduler.metrics_collector.observe_decode_bootstrap_time(
                    ts.bootstrap_done_time - ts.decode_prealloc_queue_entry_time
                )
                self.scheduler.metrics_collector.observe_decode_alloc_wait_time(
                    ts.decode_transfer_queue_entry_time - ts.bootstrap_done_time
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(
        self,
        retractable_tokens: Optional[int] = None,
        swa_retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Union[int, Tuple[int, int]]:
        page_size = self.token_to_kv_pool_allocator.page_size
        running_reqs = self.scheduler.running_batch.reqs
        if (
            retractable_tokens is not None
            and len(running_reqs) > 0
        ):
            need_space_for_single_req = 0
            need_swa_space_for_single_req = 0
            is_hybrid_swa = self.scheduler.model_config.is_hybrid_swa
            if is_hybrid_swa:
                assert swa_retractable_tokens is not None
            for req in running_reqs:
                need_space_for_single_req = max(
                    need_space_for_single_req,
                    self._max_full_tokens_for_req(
                        req, retractable_tokens, page_size
                    ),
                )
                if is_hybrid_swa:
                    need_swa_space_for_single_req = max(
                        need_swa_space_for_single_req,
                        self._max_swa_tokens_for_req(
                            req, swa_retractable_tokens, page_size
                        ),
                    )
        else:
            need_space_for_single_req = 0
            need_swa_space_for_single_req = 0
        total_reqs = (
            len(running_reqs)
            + len(self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
        )
        if not self.scheduler.model_config.is_hybrid_swa:
            available_size = self.token_to_kv_pool_allocator.available_size()
            reserved_tokens = self.num_reserved_decode_tokens * total_reqs

            allocatable_tokens = available_size - max(
                # preserve some space for future decode
                reserved_tokens,
                # make sure each request can finish if reach max_tokens with all other requests retracted
                need_space_for_single_req,
            )

            # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
            #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
            if (
                self.scheduler.last_batch
                and self.scheduler.last_batch.forward_mode.is_prebuilt()
            ):
                allocatable_tokens -= self.num_reserved_decode_tokens * len(
                    self.scheduler.last_batch.reqs
                )

            if count_retracted:
                allocatable_tokens -= sum(
                    self._full_required_tokens(req, page_size)
                    for req in self.retracted_queue
                )
            return allocatable_tokens
        else:
            need_full_space_for_single_req = need_space_for_single_req

            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
            swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()

            reserved_full = self.num_reserved_decode_tokens * total_reqs
            alloc_full = full_available_size - max(
                reserved_full,
                need_full_space_for_single_req,
            )

            # SWA side: no reserved decode tokens, only window budget + optional spec extra

            alloc_swa = swa_available_size - need_swa_space_for_single_req

            if (
                self.scheduler.last_batch
                and self.scheduler.last_batch.forward_mode.is_prebuilt()
            ):
                alloc_full -= self.num_reserved_decode_tokens * len(
                    self.scheduler.last_batch.reqs
                )

            if count_retracted:
                alloc_full -= sum(
                    self._full_required_tokens(req, page_size)
                    for req in self.retracted_queue
                )

                alloc_swa -= sum(
                    # pre-alloc for prefill + alloc for decode
                    self._swa_tokens_for_retracted_req(req, page_size)
                    for req in self.retracted_queue
                )

            return (alloc_full, alloc_swa)

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        # Alloc all tokens for the prebuilt req (except for the reserved input token for decoding)
        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len

        if self.scheduler.enable_hisparse:
            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_logical_only(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )
            # Allocate host indices for the RDMA transfer target
            host_indices = coordinator.mem_pool_host.alloc(fill_len)
            if host_indices is None:
                raise RuntimeError(
                    f"HiSparse host mem pool alloc failed for {fill_len} tokens "
                    f"in _pre_alloc (req {req.rid})"
                )
            host_indices = host_indices.to(device=coordinator.device)
            coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        elif self._has_pre_alloc:
            kv_loc = self.token_to_kv_pool_allocator.pre_alloc(
                fill_len,
                self.scheduler.swa_retention_window,
            )
        elif self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(fill_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

        assert (
            kv_loc is not None
        ), "KV cache is full! There is a bug in memory estimation."

        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_input_len(len(req.fill_ids))
        # 会在 maybe_evict_swa 中使用
        req.swa_evicted_seqlen = floor_align(
            max(
                len(req.origin_input_ids)
                + max(len(req.output_ids) - 1, 0)
                - self.scheduler.swa_retention_window,
                0,
            ),
            self.token_to_kv_pool_allocator.page_size,
        )

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None

    @property
    def num_h2d_queue_reqs(self):
        return sum(
            1
            for decode_req in self.queue
            if decode_req.pd_d2h_state is PDD2HState.HOST_READY
        )

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)
        if self.enable_staging:
            for dr in decode_reqs:
                if dr.kv_receiver.require_staging:
                    self.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)

    def abort_matching(self, recv_req: AbortReq) -> None:
        pd_d2h_mgr = getattr(self.scheduler, "pd_direct2host_manager", None)
        remaining_queue = []
        for decode_req in self.queue:
            if not (recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid)):
                remaining_queue.append(decode_req)
                continue

            logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
            if _is_request_abort(decode_req.req):
                remaining_queue.append(decode_req)
                continue

            decode_req.abort_receiver_if_active()
            _notify_request_abort(self.scheduler, decode_req)

            if pd_d2h_mgr is None:
                remaining_queue.append(decode_req)
                continue

            if decode_req.pd_d2h_state is PDD2HState.WAIT_PREFILL:
                remaining_queue.append(decode_req)
                continue

            disposition = pd_d2h_mgr.abort(decode_req.req)
            if disposition is PDD2HAbortDisposition.DEFER_UNTIL_H2D_DONE:
                decode_req.pd_d2h_state = PDD2HState.ABORT_PENDING_H2D
                remaining_queue.append(decode_req)
                continue

            self._release_queue_entry(decode_req)

        self.queue = remaining_queue

    def _release_queue_entry(self, decode_req: DecodeRequest) -> None:
        if self.enable_staging and self.staging_handler.is_staging_room(
            decode_req.req.bootstrap_room
        ):
            self.staging_handler.unregister_decode_req(decode_req.req.bootstrap_room)

        idx = decode_req.metadata_buffer_index
        if idx != -1:
            self.req_to_metadata_buffer_idx_allocator.free(idx)
            decode_req.metadata_buffer_index = -1

    def _bootstrap_room(self, decode_req: DecodeRequest) -> int:
        bootstrap_room = decode_req.req.bootstrap_room
        if bootstrap_room is None:
            raise RuntimeError(
                "Disagg decode transfer request has no bootstrap_room: "
                f"rid={decode_req.req.rid}"
            )
        return int(bootstrap_room)

    def _queue_indices_by_bootstrap_room(self) -> List[int]:
        return sorted(
            range(len(self.queue)),
            key=lambda i: self._bootstrap_room(self.queue[i]),
        )

    def _h2d_reserved_decode_tokens(self) -> int:
        total_reqs = len(self.scheduler.running_batch.reqs)
        last_batch = self.scheduler.last_batch
        if last_batch is not None and last_batch.forward_mode.is_prebuilt():
            total_reqs += len(last_batch.reqs)
        return self.scheduler.server_args.num_reserved_decode_tokens * total_reqs

    def find_by_bootstrap_room(self, bootstrap_room: int) -> Optional[DecodeRequest]:
        matches = [
            decode_req
            for decode_req in self.queue
            if self._bootstrap_room(decode_req) == bootstrap_room
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "Duplicate bootstrap_room in disagg decode transfer queue: "
                f"tp_rank={self.tp_rank} bootstrap_room={bootstrap_room}"
            )
        return matches[0] if matches else None

    def start_h2d(self, pd_d2h_mgr, decode_req: DecodeRequest) -> None:
        if not pd_d2h_mgr.start_h2d(decode_req.req):
            raise RuntimeError(
                f"H2D allocation failed after TP admission: rid={decode_req.req.rid}"
            )
        decode_req.pd_d2h_state = PDD2HState.H2D_INFLIGHT
        decode_req.req.time_stats.set_decode_h2d_wait_start_time()

    def _local_metadata_room(self, decode_req: DecodeRequest) -> int:
        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            return self._bootstrap_room(decode_req)
        idx = decode_req.metadata_buffer_index
        *_, output_bootstrap_room = self.metadata_buffers.get_buf(idx)
        return int(output_bootstrap_room[0].item())

    def local_poll_entries(
        self, pd_d2h_mgr
    ) -> List[Tuple[int, int, int, int, int, int]]:
        """Snapshot transfer room, poll, state, and visible metadata room.

        Pure local computation — the collective exchange is done by
        PDD2HFusedPoller (see DecodePreallocQueue.local_poll_entries).
        """
        entries: List[Tuple[int, int, int, int, int, int]] = []
        for decode_req in self.queue:
            room = self._bootstrap_room(decode_req)
            need_full, need_swa = pd_d2h_mgr.h2d_required_tokens(decode_req.req)
            metadata_room = -1
            if decode_req.pd_d2h_state is PDD2HState.WAIT_PREFILL:
                poll = int(decode_req.kv_receiver.poll())
                if poll == KVPoll.Success:
                    metadata_room = self._local_metadata_room(decode_req)
            elif decode_req.pd_d2h_state in (
                PDD2HState.H2D_INFLIGHT,
                PDD2HState.ABORT_PENDING_H2D,
            ):
                poll = (
                    int(KVPoll.Success)
                    if pd_d2h_mgr.is_h2d_ready(decode_req.req)
                    else int(KVPoll.Transferring)
                )
            else:
                poll = int(KVPoll.Transferring)
            entries.append(
                (
                    room,
                    poll,
                    decode_req.pd_d2h_state.value,
                    metadata_room,
                    need_full,
                    need_swa,
                )
            )
        return entries

    def _commit_transfer_to_req(
        self,
        decode_req: DecodeRequest,
        set_wait_queue_time: bool = True,
    ) -> bool:
        """
        Returns:
            True if the request should be removed from the queue (success or corruption)
            False if metadata not ready yet (keep in queue for next poll)
        """
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            mm_image_tokens,
            mm_audio_tokens,
            mm_video_tokens,
            dflash_draft_seq_len,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        # Case 3: Success - commit the transfer
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        decode_req.req.mm_image_tokens = mm_image_tokens[0].item()
        decode_req.req.mm_audio_tokens = mm_audio_tokens[0].item()
        decode_req.req.mm_video_tokens = mm_video_tokens[0].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states
        if self.spec_algorithm.is_dflash():
            # PD + DFlash: prefill side already populated draft KV pool; decode
            # only needs the cursor to rebuild DFlashDraftInput in process_prebuilt.
            decode_req.req.dflash_draft_seq_len = int(
                dflash_draft_seq_len[0].item()
            )

        if decode_req.req.return_logprob:
            decode_req.req.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.output_top_logprobs_val.append(
                output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist()
            )
            decode_req.req.output_top_logprobs_idx.append(
                output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist()
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        if set_wait_queue_time:
            decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue, self.staging_handler, self.gloo_group
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def _pop_transferred_pd_d2h(
        self,
        pd_d2h_mgr,
        rids_to_check: Optional[List[str]] = None,
        polls_by_room: Optional[Dict[int, int]] = None,
    ) -> List[Req]:
        if polls_by_room is None:
            raise RuntimeError(
                "PD D2H transfer polling requires fused poll consensus; "
                "pipeline parallelism is not supported with PD D2H."
            )
        polls = [
            polls_by_room.get(
                self._bootstrap_room(decode_req), int(KVPoll.Transferring)
            )
            for decode_req in self.queue
        ]

        transferred_reqs = []
        indices_to_remove = set()
        for i in self._queue_indices_by_bootstrap_room():
            decode_req = self.queue[i]
            poll = polls[i]
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                if decode_req.kv_receiver is not None:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                is_request_abort = _is_request_abort(decode_req.req)
                if not is_request_abort:
                    logger.error(error_message)
                    prepare_abort(
                        decode_req.req,
                        error_message,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    self.scheduler.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                if is_request_abort or decode_req.pd_d2h_state in (
                    PDD2HState.H2D_INFLIGHT,
                    PDD2HState.ABORT_PENDING_H2D,
                ):
                    disposition = pd_d2h_mgr.abort(decode_req.req)
                    if disposition is PDD2HAbortDisposition.DEFER_UNTIL_H2D_DONE:
                        decode_req.pd_d2h_state = PDD2HState.ABORT_PENDING_H2D
                        continue
                else:
                    pd_d2h_mgr.release_failed(decode_req.req)
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics and not is_request_abort:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue

            if poll == KVPoll.Success:
                if decode_req.pd_d2h_state in (
                    PDD2HState.H2D_INFLIGHT,
                    PDD2HState.ABORT_PENDING_H2D,
                ):
                    state_before_complete = decode_req.pd_d2h_state
                    ready_req = pd_d2h_mgr.complete_h2d(decode_req.req)
                    if ready_req is None:
                        log_fn = (
                            logger.warning
                            if state_before_complete is PDD2HState.ABORT_PENDING_H2D
                            else logger.error
                        )
                        log_fn(
                            "pd-d2h complete_h2d returned None: "
                            "tp_rank=%s rid=%s bootstrap_room=%s state=%s poll=%s",
                            self.tp_rank,
                            decode_req.req.rid,
                            decode_req.req.bootstrap_room,
                            state_before_complete.name,
                            poll,
                        )
                    if (
                        ready_req is not None
                        and decode_req.pd_d2h_state is PDD2HState.H2D_INFLIGHT
                    ):
                        ready_req.time_stats.set_decode_h2d_wait_finish_time()
                        ready_req.time_stats.set_wait_queue_entry_time()
                        transferred_reqs.append(ready_req)
                    indices_to_remove.add(i)
                    continue

                if decode_req.pd_d2h_state is not PDD2HState.WAIT_PREFILL:
                    logger.error(
                        "pd-d2h invalid state for transfer commit: "
                        "tp_rank=%s rid=%s bootstrap_room=%s state=%s poll=%s",
                        self.tp_rank,
                        decode_req.req.rid,
                        decode_req.req.bootstrap_room,
                        decode_req.pd_d2h_state.name,
                        poll,
                    )
                should_remove = self._commit_transfer_to_req(
                    decode_req, set_wait_queue_time=False
                )
                if should_remove:
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        indices_to_remove.add(i)
                        is_request_abort = _is_request_abort(decode_req.req)
                        if not is_request_abort:
                            self.scheduler.stream_output(
                                [decode_req.req], decode_req.req.return_logprob
                            )
                        if self.scheduler.enable_hisparse:
                            self.scheduler.hisparse_coordinator.request_finished(
                                decode_req.req
                            )
                        pd_d2h_mgr.abort(decode_req.req)
                        if self.scheduler.enable_metrics and not is_request_abort:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        decode_req.req.time_stats.set_decode_transferred_time()
                        self.req_to_metadata_buffer_idx_allocator.free(
                            decode_req.metadata_buffer_index
                        )
                        decode_req.metadata_buffer_index = -1
                        decode_req.pd_d2h_state = PDD2HState.HOST_READY
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            self._release_queue_entry(self.queue[i])

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def pop_transferred(
        self,
        rids_to_check: Optional[List[str]] = None,
        polls_by_room: Optional[Dict[int, int]] = None,
    ) -> List[Req]:
        pd_d2h_mgr = getattr(self.scheduler, "pd_direct2host_manager", None)
        if pd_d2h_mgr is not None:
            return self._pop_transferred_pd_d2h(
                pd_d2h_mgr,
                rids_to_check,
                polls_by_room=polls_by_room,
            )

        if not self.queue:
            return []

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = poll_and_all_reduce(
                [dr.kv_receiver for dr in self.queue], self.gloo_group
            )

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            decode_req = self.queue[i]
            poll = polls[i]
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                if decode_req.kv_receiver is not None:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                is_request_abort = _is_request_abort(decode_req.req)
                if not is_request_abort:
                    logger.error(error_message)
                    prepare_abort(
                        decode_req.req,
                        error_message,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    self.scheduler.stream_output(
                        [decode_req.req], decode_req.req.return_logprob
                    )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics and not is_request_abort:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                should_remove = self._commit_transfer_to_req(decode_req)
                if should_remove:
                    # Check if request was aborted due to corruption
                    indices_to_remove.add(i)
                    if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                        is_request_abort = _is_request_abort(decode_req.req)
                        if not is_request_abort:
                            self.scheduler.stream_output(
                                [decode_req.req], decode_req.req.return_logprob
                            )
                        if self.scheduler.enable_hisparse:
                            self.scheduler.hisparse_coordinator.request_finished(
                                decode_req.req
                            )
                        release_kv_cache(
                            decode_req.req, self.tree_cache, is_insert=False
                        )
                        if self.scheduler.enable_metrics and not is_request_abort:
                            self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                    else:
                        transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            self._release_queue_entry(self.queue[i])

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:

    @torch.no_grad()
    @nvtx_annotated_method("scheduler.disagg_decode.event_loop_normal")
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    @nvtx_annotated_method("scheduler.disagg_decode.event_loop_overlap")
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            self._apply_war_barrier()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @nvtx_annotated_method("scheduler.disagg_decode.get_next_batch")
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch()
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.process_batch_result_prebuilt(new_prebuilt_batch)
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        self.running_batch.hisparse_coordinator = (
                            self.hisparse_coordinator
                        )
                else:
                    self.running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if self.running_batch.is_empty():
            ret = None
        else:
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None

        ret = self.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return ret

    @nvtx_annotated_method("scheduler.disagg_decode.get_new_prebuilt_batch")
    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                req.init_next_round_input(self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    @nvtx_annotated_method("scheduler.disagg_decode.process_decode_queue")
    def process_decode_queue(self: Scheduler):
        pd_d2h_mgr = getattr(self, "pd_direct2host_manager", None)

        prealloc_polls_by_room: Optional[Dict[int, int]] = None
        transfer_polls_by_room: Optional[Dict[int, int]] = None
        if pd_d2h_mgr is not None:
            if not hasattr(self, "pd_d2h_fused_poller"):
                self.pd_d2h_fused_poller = PDD2HFusedPoller(
                    gloo_group=self.attn_tp_cpu_group,
                    tp_rank=self.tp_rank,
                )
            reserved_decode_tokens = (
                self.disagg_decode_transfer_queue._h2d_reserved_decode_tokens()
            )
            prealloc_entries = self.disagg_decode_prealloc_queue.local_poll_entries()
            transfer_entries = self.disagg_decode_transfer_queue.local_poll_entries(
                pd_d2h_mgr
            )
            admission_snapshot = pd_d2h_mgr.h2d_admission_snapshot(
                reserved_decode_tokens
            )
            (
                prealloc_polls_by_room,
                transfer_polls_by_room,
                candidate_room,
                h2d_admit,
            ) = self.pd_d2h_fused_poller.poll(
                prealloc_entries,
                transfer_entries,
                admission_snapshot,
            )
            h2d_candidate = (
                self.disagg_decode_transfer_queue.find_by_bootstrap_room(
                    candidate_room
                )
                if candidate_room is not None
                else None
            )
            if candidate_room is not None and h2d_candidate is None:
                raise RuntimeError(
                    "TP-wide H2D candidate is missing from the local queue: "
                    f"tp_rank={self.tp_rank} bootstrap_room={candidate_room}"
                )
            if h2d_admit:
                if h2d_candidate is None:
                    raise RuntimeError("TP admitted H2D without a shared candidate")
                self.disagg_decode_transfer_queue.start_h2d(
                    pd_d2h_mgr, h2d_candidate
                )

        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated(
                polls_by_room=prealloc_polls_by_room
            )
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = self.disagg_decode_transfer_queue.pop_transferred(
                polls_by_room=transfer_polls_by_room
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
                self.waiting_queue.extend(transferred_reqs)
            else:
                self.waiting_queue.extend(transferred_reqs)
