from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import random
import threading
import time
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1024
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        swa_host_indices: Optional[torch.Tensor] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        self.swa_host_indices = swa_host_indices

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: "StorageOperation"):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        swa_host_indices: Optional[torch.Tensor] = None,
    ):
        self.request_id = request_id
        self.swa_valid_mask: Optional[torch.Tensor] = None
        self.full_hit_count: int = 0
        self.swa_hit_count: int = 0

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        self.io_start_time = 0.0
        self.io_finish_time = 0.0

        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            swa_host_indices=swa_host_indices,
        )

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag

    def mark_io_start(self):
        with self._lock:
            if self.io_start_time == 0.0:
                self.io_start_time = time.monotonic()

    def mark_io_finish(self):
        with self._lock:
            if self.io_start_time > 0.0 and self.io_finish_time == 0.0:
                self.io_finish_time = time.monotonic()

    def get_io_duration(self) -> Optional[float]:
        with self._lock:
            if self.io_start_time == 0.0 or self.io_finish_time == 0.0:
                return None
            return self.io_finish_time - self.io_start_time


class FaultInjector:
    """L3 TP mismatch fault injection for testing.

    Provides three independent fault injection points controlled by environment
    variables (all default to 0.0 = disabled):

    - SGLANG_TEST_GCACHE_FI_EXISTS_PROB:
        Injection site: _storage_hit_query(), before results are returned.
        Effect: current rank reports a smaller storage_hit_count; for hybrid
        models, randomly flips some swa_valid_per_page entries to False.
        Test target: all_reduce(MIN) alignment in prefetch_thread_func.

    - SGLANG_TEST_GCACHE_FI_TRANSFER_PROB:
        Injection site: _page_get_zero_copy(), before operation.increment(inc).
        Effect: current rank under-reports completed tokens by dropping some
        pages from the reported increment.
        Test target: all_reduce(MIN) alignment in check_prefetch_progress.

    - SGLANG_TEST_GCACHE_FI_TIMEOUT_PROB:
        Injection site: _page_transfer() batch loop, before page_get_func.
        Effect: current rank sleeps SGLANG_TEST_GCACHE_FI_TIMEOUT_SLEEP_S
        seconds (default 5.0s), simulating GCache IO timeout.
        Test target: is_prefetch_timeout detection and all_reduce(MAX, states)
        broadcast in can_terminate_prefetch.

    Each TP rank rolls dice independently so different ranks may trigger
    differently, accurately simulating real-world TP inconsistency.
    """

    def __init__(
        self,
        tp_rank: int,
        tp_world_size: int,
        page_size: int,
        is_hybrid: bool,
    ):
        from sglang.srt.environ import envs

        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.page_size = page_size
        self.is_hybrid = is_hybrid

        self._fi_exists_prob = envs.SGLANG_TEST_GCACHE_FI_EXISTS_PROB.get()
        self._fi_transfer_prob = envs.SGLANG_TEST_GCACHE_FI_TRANSFER_PROB.get()
        self._fi_timeout_prob = envs.SGLANG_TEST_GCACHE_FI_TIMEOUT_PROB.get()
        self._fi_timeout_sleep_s = (
            envs.SGLANG_TEST_GCACHE_FI_TIMEOUT_SLEEP_S.get()
            if self._fi_timeout_prob > 0.0
            else 0.0
        )

        # Each injection point uses a dedicated RNG pair to avoid cross-thread
        # contention. The prefetch thread calls maybe_inject_exists_fault while
        # the IO aux thread calls maybe_inject_transfer_fault/maybe_inject_timeout_fault.
        _seed_base = self.tp_rank * 10007 + os.getpid() + int(time.monotonic() * 1e6)
        self._exists_trigger_rng = random.Random(_seed_base)
        self._exists_amount_rng = random.Random(_seed_base + 1)
        self._transfer_trigger_rng = random.Random(_seed_base + 2)
        self._transfer_amount_rng = random.Random(_seed_base + 3)
        self._timeout_trigger_rng = random.Random(_seed_base + 4)

        # Track operations that have already triggered timeout injection (at most once per operation)
        self._fi_timeout_triggered: set = set()

        self.enabled = (
            self._fi_exists_prob > 0.0
            or self._fi_transfer_prob > 0.0
            or self._fi_timeout_prob > 0.0
        )

        if self.enabled:
            logger.warning(
                "[FaultInjection] L3 TP mismatch simulation ENABLED — "
                "exists_prob=%s, transfer_prob=%s, timeout_prob=%s, "
                "timeout_sleep_s=%s, tp_rank=%d, tp_world_size=%d",
                self._fi_exists_prob,
                self._fi_transfer_prob,
                self._fi_timeout_prob,
                self._fi_timeout_sleep_s,
                self.tp_rank,
                self.tp_world_size,
            )

    def maybe_inject_transfer_fault(self, inc: int) -> int:
        """Fault injection for the IO-transfer stage.

        Each TP rank rolls dice independently. When triggered, the current rank
        under-reports the completed token count for this batch, simulating an
        RDMA transfer interruption.

        Prerequisites:
        - _fi_transfer_prob > 0
        - TP world size > 1 (single-GPU has no all_reduce to test)
        - At least 2 pages completed in this batch (trimming < 2 pages is meaningless)

        Returns:
            The (possibly reduced) ``inc`` value.
        """
        if (
            self._fi_transfer_prob > 0.0
            and self.tp_world_size > 1
            and inc >= 2 * self.page_size
            and self._transfer_trigger_rng.random() < self._fi_transfer_prob
        ):
            orig_inc = inc
            drop_pages = self._transfer_amount_rng.randint(
                1, max(1, (inc // self.page_size) // 2)
            )
            inc -= drop_pages * self.page_size

            logger.warning(
                "[FaultInjection][transfer] tp_rank=%d: "
                "completed_tokens inc %d -> %d (%d pages dropped)",
                self.tp_rank,
                orig_inc,
                inc,
                drop_pages,
            )
        return inc

    def maybe_inject_timeout_fault(self, operation_id: int):
        """Fault injection for the transfer thread: simulate IO timeout.

        Triggers at most once per operation. When triggered, the current rank
        sleeps for ``_fi_timeout_sleep_s`` seconds, simulating a GCache IO hang.

        Prerequisites:
        - _fi_timeout_prob > 0
        - TP world size > 1 (single-GPU has no cross-rank timeout propagation to test)
        - This operation has not been injected before
        """
        if (
            self._fi_timeout_prob > 0.0
            and self.tp_world_size > 1
            and operation_id not in self._fi_timeout_triggered
            and self._timeout_trigger_rng.random() < self._fi_timeout_prob
        ):
            self._fi_timeout_triggered.add(operation_id)
            logger.warning(
                "[FaultInjection][timeout] tp_rank=%d: operation_id=%d, "
                "sleeping %.1fs to simulate IO timeout",
                self.tp_rank,
                operation_id,
                self._fi_timeout_sleep_s,
            )
            time.sleep(self._fi_timeout_sleep_s)
            logger.warning(
                "[FaultInjection][timeout] tp_rank=%d: operation_id=%d, "
                "sleep finished, resuming remaining batches",
                self.tp_rank,
                operation_id,
            )

    def maybe_inject_exists_fault(
        self,
        hash_values: list,
        storage_query_count: int,
        swa_valid_per_page: list,
    ) -> tuple:
        """Fault injection for the exists (metadata-query) stage.

        Each TP rank rolls dice independently. When triggered, the current rank
        reports a smaller storage_hit_count and optionally degrades some
        swa_valid_per_page entries to False.

        Prerequisites:
        - _fi_exists_prob > 0
        - TP world size > 1
        - At least 2 pages hit (trimming below 2 may trigger threshold revocation
          and mask bugs in the all_reduce logic itself)

        Returns:
            Tuple of (hash_values, storage_query_count, swa_valid_per_page).
        """
        if (
            self._fi_exists_prob > 0.0
            and self.tp_world_size > 1
            and len(hash_values) >= 2
            and self._fi_trigger_rng.random() < self._fi_exists_prob
        ):
            drop_pages = self._exists_amount_rng.randint(
                1, max(1, len(hash_values) // 2)
            )
            orig_count = storage_query_count
            orig_pages = len(hash_values)

            hash_values = hash_values[: orig_pages - drop_pages]
            storage_query_count = len(hash_values) * self.page_size

            # Hybrid model: sync-trim swa_valid_per_page and randomly flip some
            # entries to False, simulating partial SWA data missing in GCache on
            # this rank while other ranks still report valid.
            if self.is_hybrid and swa_valid_per_page:
                swa_valid_per_page = swa_valid_per_page[: len(hash_values)]
                flip_count = self._exists_amount_rng.randint(
                    0, max(0, len(swa_valid_per_page) // 2)
                )
                if flip_count > 0:
                    flip_indices = self._exists_amount_rng.sample(
                        range(len(swa_valid_per_page)), flip_count
                    )
                    for idx in flip_indices:
                        swa_valid_per_page[idx] = False

            logger.warning(
                "[FaultInjection][exists] tp_rank=%d: "
                "storage_hit_count %d -> %d (%d pages dropped)%s",
                self.tp_rank,
                orig_count,
                storage_query_count,
                drop_pages,
                (
                    f", swa_valid_per_page={swa_valid_per_page}"
                    if self.is_hybrid
                    else ""
                ),
            )
        return hash_values, storage_query_count, swa_valid_per_page

    def cleanup_timeout_tracking(self, operation_id: int):
        """Remove a completed operation from the timeout-injection tracking set."""
        self._fi_timeout_triggered.discard(operation_id)


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        enable_storage_metrics: bool = False,
        sliding_window_size: Optional[int] = None,
    ):
        self.tp_group = tp_group
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False

        self.has_draft_kv_pool = False
        self.has_draft_kv_pool_list = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.mem_pool_device_draft_list = []
        self.mem_pool_host_draft_list = []
        self.storage_backend = None
        self.storage_backend_type = None
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enable_storage_metrics = enable_storage_metrics
        self.metrics_collector = None
        self.sliding_window_size = sliding_window_size
        self.sliding_window_pages = (
            sliding_window_size // self.page_size
            if sliding_window_size is not None
            else None
        )
        self.is_hybrid_swa = sliding_window_size is not None

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        # Host full + optional swa
        self.host_mem_release_queue = Queue()
        # Host swa only
        self.swa_only_host_mem_release_queue = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.hicache_storage import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            self.prefetch_capacity_limit = max(
                0, int(0.8 * (self.mem_pool_host.size - self.mem_pool_device.size))
            )
            # granularity of batch storage IO operations, in number of pages
            self.storage_batch_size = 128
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # create a new communication group for synchronizing storage operations across TP workers
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            if self.tp_world_size > 1:
                from sglang.srt.distributed.parallel_state import (
                    create_custom_parallel_group,
                )

                group_ranks = torch.distributed.get_process_group_ranks(self.tp_group)
                self.prefetch_tp_group = create_custom_parallel_group(
                    group_ranks=group_ranks, backend="gloo"
                )

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl", "gcache"]) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()

            # Fault injection for L3 TP mismatch simulation (test only)
            self._fault_injector = FaultInjector(
                tp_rank=self.tp_rank,
                tp_world_size=self.tp_world_size,
                page_size=self.page_size,
                is_hybrid=self.is_hybrid_swa,
            )
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            try:
                if hasattr(self, "prefetch_tp_group"):
                    try:
                        torch.distributed.destroy_process_group(self.prefetch_tp_group)
                    except Exception:
                        pass
                    self.prefetch_tp_group = None
            except Exception:
                pass
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process group created for storage ops.
        try:
            if (
                hasattr(self, "prefetch_tp_group")
                and self.prefetch_tp_group is not None
            ):
                try:
                    torch.distributed.destroy_process_group(self.prefetch_tp_group)
                except Exception:
                    pass
                self.prefetch_tp_group = None
        except Exception:
            pass

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        is_mla_backend = isinstance(self.mem_pool_device, MLATokenToKVPool)
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_mla_backend
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            is_mla_model=is_mla_backend,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.stop_event.set()
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()

        self.stop_event.clear()
        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def alloc_host(
        self,
        device_indices: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], int, int]:
        if self.is_hybrid_swa:
            swa_device_indices = (
                self.mem_pool_device_allocator.translate_loc_from_full_to_swa(
                    device_indices
                )
            )
            return self.mem_pool_host.alloc_host(device_indices, swa_device_indices)

        need_size = len(device_indices)
        host_indices = self.mem_pool_host.alloc(need_size)
        if host_indices is None:
            lack_full_size = need_size - self.mem_pool_host.available_size()
            return None, max(0, lack_full_size), 0
        return host_indices, 0, 0

    def get_swa_host_indices(
        self,
        full_host_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.is_hybrid_swa:
            return None
        return self.mem_pool_host.full_to_swa_index_mapping[full_host_indices]

    def get_valid_swa_host_len(
        self,
        full_host_indices: torch.Tensor,
    ) -> int:
        if not self.is_hybrid_swa:
            return 0
        swa_host_indices = self.mem_pool_host.full_to_swa_index_mapping[
            full_host_indices
        ]
        return torch.count_nonzero(swa_host_indices).item()

    def write(
        self,
        device_indices: torch.Tensor,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ):
        """
        Back up KV caches from device memory to host memory.
        """
        assert host_indices is not None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft_kv_pool_list:
                for i, draft_device_pool in enumerate(self.mem_pool_device_draft_list):
                    self.mem_pool_host_draft_list[i].backup_from_device_all_layer(
                        draft_device_pool,
                        host_indices,
                        device_indices,
                        self.io_backend,
                    )
            elif self.has_draft_kv_pool:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def assert_host_indices(
        self,
        host_indices: torch.Tensor,
        host_hit_length: int,
    ):
        if not self.is_hybrid_swa:
            return
        assert self.sliding_window_size is not None and self.sliding_window_size > 0
        # TODO(yingchun): swa_host_indices 与调用者重复计算，可传入参数直接用
        swa_host_indices = self.mem_pool_host.full_to_swa_index_mapping[host_indices]
        ensure_swa_length = min(host_hit_length, self.sliding_window_size)
        # TODO(yingchun): this assert has D2H op
        assert swa_host_indices[-ensure_swa_length:].min().item() > 0, (
            f"Last {ensure_swa_length} SWA slots must be valid, "
            f"got {swa_host_indices[-ensure_swa_length:].tolist()}"
        )

    def alloc_device(
        self,
        host_indices: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        if not self.is_hybrid_swa:
            need_size = len(host_indices)
            device_indices = self.mem_pool_device_allocator.alloc(need_size)
            if device_indices is None:
                lack_full_size = (
                    need_size - self.mem_pool_device_allocator.available_size()
                )
                return None, max(0, lack_full_size), 0
            return device_indices, 0, 0

        swa_host_indices = self.mem_pool_host.full_to_swa_index_mapping[host_indices]
        swa_valid_mask = swa_host_indices != 0
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            self.assert_host_indices(host_indices, self.sliding_window_size)
            swa_valid_mask[: -self.sliding_window_size] = False

        return self.mem_pool_device_allocator.alloc_device(host_indices, swa_valid_mask)

    def load(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        assert device_indices is not None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )

    def move_indices(self, op: CacheOperation):
        host_indices, device_indices = op.host_indices, op.device_indices
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(op)
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft_kv_pool_list:
                    for j, draft_host_pool in enumerate(self.mem_pool_host_draft_list):
                        if i < self.mem_pool_host_draft_list[j].layer_num:
                            draft_host_pool.load_to_device_per_layer(
                                self.mem_pool_device_draft_list[j],
                                host_indices,
                                device_indices,
                                i,
                                self.io_backend,
                            )
                elif self.has_draft_kv_pool:
                    if i < self.mem_pool_host_draft.layer_num:
                        self.mem_pool_host_draft.load_to_device_per_layer(
                            self.mem_pool_device_draft,
                            host_indices,
                            device_indices,
                            i,
                            self.io_backend,
                        )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> Tuple[int, int]:
        assert device_indices is not None
        if not self.is_hybrid_swa:
            self.mem_pool_device_allocator.free(device_indices)
            return len(device_indices), 0

        swa_device_indices = (
            self.mem_pool_device_allocator.translate_loc_from_full_to_swa(
                device_indices
            )
        )
        swa_freed_size = torch.count_nonzero(swa_device_indices).item()
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices), swa_freed_size

    def set_draft_kv_pool(
        self,
        draft_kv_pool,
        draft_kv_pool_list,
        draft_host_kv_pool,
        draft_host_kv_pool_list,
    ):
        if draft_kv_pool_list:
            self.has_draft_kv_pool_list = True
            self.mem_pool_device_draft_list = draft_kv_pool_list
            self.mem_pool_host_draft_list = draft_host_kv_pool_list
        elif draft_kv_pool:
            self.has_draft_kv_pool = True
            self.mem_pool_device_draft = draft_kv_pool
            self.mem_pool_host_draft = draft_host_kv_pool

    def evict_host(
            self, host_indices: torch.Tensor, backup_only: bool = True
    ) -> Tuple[int, int]:
        assert host_indices is not None
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        freed = self.mem_pool_host.free(host_indices)
        if isinstance(freed, tuple):
            return int(freed[0]), int(freed[1])
        return int(freed), 0

    def prefetch(
        self,
        request_id: str,
        full_host_indices: torch.Tensor,
        swa_host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id,
            full_host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            swa_host_indices=swa_host_indices,
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def append_swa_only_host_mem_release(
        self, swa_only_to_release: Optional[torch.Tensor]
    ):
        if swa_only_to_release is None:
            assert (
                not self.is_hybrid_swa
            ), "swa_only_to_release must not be None in SWA mode"
            return
        if swa_only_to_release.numel() == 0:
            return

        page_size = self.mem_pool_host.page_size
        paged = swa_only_to_release.view(-1, page_size)
        valid_count_per_page = paged.count_nonzero(dim=1)
        assert (
            (valid_count_per_page == 0) | (valid_count_per_page == page_size)
        ).all(), f"Each page must be all-zero or all-valid, got {valid_count_per_page.tolist()}"

        valid_page_mask = valid_count_per_page > 0
        if valid_page_mask.any():
            # TODO(yingchun): 单独 put 入队，lock 获取频繁。可考虑批量入队。
            for page in paged[valid_page_mask]:
                self.swa_only_host_mem_release_queue.put(page)

    def append_full_and_swa_mem_release_separately(
        self, full_host_indices_to_release: torch.Tensor
    ):
        if full_host_indices_to_release.numel() == 0:
            return

        swa_host_indices = self.mem_pool_host.unmapping_full_and_swa_indices(
            full_host_indices_to_release
        )
        self.append_host_mem_release(full_host_indices_to_release)
        self.append_swa_only_host_mem_release(swa_host_indices)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        if self._fault_injector.enabled:
            inc = self._fault_injector.maybe_inject_transfer_fault(inc)
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        operation.mark_io_start()
        try:
            # Transfer batch by batch
            prefix_keys = operation.prefix_keys
            for i in range(0, len(operation.hash_value), self.storage_batch_size):
                batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
                batch_host_indices = operation.host_indices[
                    i * self.page_size : (i + len(batch_hashes)) * self.page_size
                ]
                prev_completed_tokens = operation.completed_tokens
                extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
                # 使用 swa_valid_mask 过滤需要 batch get 的 SWA KV cache
                if self.is_hybrid_swa and operation.swa_valid_mask is not None:
                    extra_info.swa_needed_per_page = operation.swa_valid_mask[
                        i : i + len(batch_hashes)
                    ]

                if self._fault_injector.enabled:
                    self._fault_injector.maybe_inject_timeout_fault(operation.id)

                # Get one batch token, and update the completed_tokens if succeed
                t0_io = time.perf_counter()
                self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
                if self.metrics_collector is not None:
                    self.metrics_collector.log_prefetch_io_latency(
                        time.perf_counter() - t0_io
                    )
                # Check termination
                if (
                    operation.completed_tokens
                    != prev_completed_tokens + len(batch_hashes) * self.page_size
                ):
                    operation.mark_terminate()
                    break  # Some operations fail or operation terminated by controller

                if prefix_keys and len(prefix_keys) > 0:
                    prefix_keys += batch_hashes
        finally:
            operation.mark_io_finish()

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_full_and_swa_mem_release_separately(
                    operation.host_indices[operation.completed_tokens :]
                )
                if self._fault_injector.enabled:
                    self._fault_injector.cleanup_timeout_tracking(operation.id)
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    # TODO(yingchun): 在 _compute_safe_storage_prefix 中只考虑了 storage 中的 SWA 安全，未考虑 device 和 host 上的 SWA，
    #  更好的做法是同时考虑 device 和 host 的拼接
    # TODO(yingchun): 与 simple_swa_radix_cache.compute_swa_safe_len() 相似，考虑重构复用
    @staticmethod
    def _compute_safe_storage_prefix(
        swa_valid_per_page: torch.Tensor,
        sliding_window_pages: int,
    ) -> int:
        """
        Find the longest prefix of pages L such that the trailing SWA window
        ``[max(0, L - sliding_window_pages), L)`` all have valid SWA.

        Same algorithm as SimpleSWARadixCache._compute_safe_prefix_len but at
        page granularity, using tensor operations.
        """
        full_prefix_pages = len(swa_valid_per_page)
        if full_prefix_pages == 0:
            return 0

        # Fast path: tail window all valid → full prefix is usable.
        if swa_valid_per_page[
            max(0, full_prefix_pages - sliding_window_pages) : full_prefix_pages
        ].all():
            return full_prefix_pages

        # Slow path: prefix-sum scan to find the longest safe L.
        # invalid_cumsum[i] = number of invalid pages in [0, i)
        invalid_cumsum = (~swa_valid_per_page).int().cumsum(0)
        # prepend a 0 so invalid_cumsum_ext[i] = count of invalid in [0, i)
        invalid_cumsum_ext = torch.cat(
            [torch.zeros(1, dtype=torch.int), invalid_cumsum]
        )

        best = 0
        # TODO(yingchun): 可向量化：构造 L = arange(1, N+1), win_starts = (L - window).clamp(0)，一次比较
        #  invalid_cumsum_ext[L] - invalid_cumsum_ext[win_starts] == 0，取最后一个 True 位置。
        for l in range(1, full_prefix_pages + 1):
            win_start = max(0, l - sliding_window_pages)
            if invalid_cumsum_ext[l] - invalid_cumsum_ext[win_start] == 0:
                best = l
        return best

    def _storage_hit_query(
        self, operation
    ) -> Tuple[list[str], int, int, Optional[torch.Tensor]]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_values = []
        swa_valid_per_page = []

        for start in range(
            0, len(tokens_to_fetch), self.page_size * self.storage_batch_size
        ):
            end = min(
                start + self.page_size * self.storage_batch_size, len(tokens_to_fetch)
            )
            batch_tokens = tokens_to_fetch[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), self.page_size):
                last_hash = self.get_hash_str(
                    batch_tokens[i : i + self.page_size], last_hash
                )
                batch_hashes.append(last_hash)
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            t0_query = time.perf_counter()
            if self.is_hybrid_swa:
                hit_page_num, part_swa_mask = self.storage_backend.batch_exists_hybrid(
                    batch_hashes, extra_info
                )
                assert hit_page_num <= len(batch_hashes)
                assert hit_page_num == len(part_swa_mask)
                # 确定哪些 SWA 是有效的
                swa_valid_per_page.extend(part_swa_mask)
            else:
                hit_page_num = self.storage_backend.batch_exists(
                    batch_hashes, extra_info
                )
            if self.metrics_collector is not None:
                self.metrics_collector.log_storage_query_latency(
                    time.perf_counter() - t0_query
                )
            hash_values.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        if self._fault_injector.enabled:
            hash_values, storage_query_count, swa_valid_per_page = (
                self._fault_injector.maybe_inject_exists_fault(
                    hash_values,
                    storage_query_count,
                    swa_valid_per_page,
                )
            )

        if self.is_hybrid_swa:
            # 对于 hybrid 模型，需要裁剪到 SWA 安全的部分
            swa_valid_mask = torch.tensor(swa_valid_per_page, dtype=torch.bool)
            swa_valid_len = self._compute_safe_storage_prefix(
                swa_valid_mask,
                self.sliding_window_pages,
            )
            assert swa_valid_len <= len(tokens_to_fetch) // self.page_size
            return (
                hash_values[:swa_valid_len],
                swa_valid_len * self.page_size,
                storage_query_count,
                swa_valid_mask[:swa_valid_len],
            )
        else:
            return hash_values, storage_query_count, storage_query_count, None

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                # 通过 batch_exists 查看 storage 上存在的 KV cache（含 SWA 的有效区间 mask）
                (
                    hash_value,
                    storage_hit_count,
                    full_query_count,
                    valid_swa_mask,
                ) = self._storage_hit_query(operation)
                hit_pages = storage_hit_count // self.page_size
                operation.full_hit_count = full_query_count
                operation.swa_hit_count = storage_hit_count if self.is_hybrid_swa else 0
                assert (
                    len(hash_value) * self.page_size == storage_hit_count
                ), f"{len(hash_value)=}, {storage_hit_count=}"
                assert valid_swa_mask is None or len(valid_swa_mask) == len(
                    hash_value
                ), f"{len(valid_swa_mask)=}, {len(hash_value)=}"

                # TODO(yingchun): 通信的数据更多了，可以优化到新的队列/线程中
                #  各 TP 同步，对齐 storage_hit_count 和 valid_swa_mask
                if self.tp_world_size > 1:
                    new_input_pages = len(operation.token_ids) // self.page_size
                    storage_hit_count_tensor = torch.empty(
                        1 + new_input_pages, dtype=torch.int
                    )
                    storage_hit_count_tensor[0] = storage_hit_count
                    swa_mask_len = (
                        len(valid_swa_mask) if valid_swa_mask is not None else 0
                    )
                    # FIXME(yingchun): storage_hit_count_tensor[swa_mask_len + 1:] 元素未初始化
                    if swa_mask_len > 0:
                        storage_hit_count_tensor[1 : swa_mask_len + 1] = (
                            valid_swa_mask.int()
                        )
                    torch.distributed.all_reduce(
                        storage_hit_count_tensor,
                        op=torch.distributed.ReduceOp.MIN,
                        group=self.prefetch_tp_group,
                    )
                    storage_hit_count = storage_hit_count_tensor[0].item()
                    hit_pages = storage_hit_count // self.page_size
                    if swa_mask_len > 0:
                        valid_swa_mask = storage_hit_count_tensor[
                            1 : hit_pages + 1
                        ].bool()

                # 没有达到 prefetch 阈值，则放弃从 storage 预取
                # TODO(yingchun): 跳过预取可能造成在树上回退很长，硬算过长。
                #  可以把阈值设置成 0
                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    revoke_tokens = len(operation.host_indices)
                    self.append_host_mem_release(operation.host_indices)
                    self.append_swa_only_host_mem_release(operation.swa_host_indices)
                    if self.metrics_collector is not None:
                        self.metrics_collector.log_revoke_tokens(revoke_tokens)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                    continue

                # 在 hybrid 模型下，若 SWA 不足 sliding window 大小，则放弃预取
                if self.is_hybrid_swa:
                    # 各 TP 同步后，valid_swa_mask 会收缩，使用收缩后的再算一次 SWA 安全长度
                    # TODO(yingchun): 可以比较下与同步前的 valid_swa_mask 是否变化，未变化则可以跳过
                    swa_valid_len = self._compute_safe_storage_prefix(
                        valid_swa_mask,
                        self.sliding_window_pages,
                    )
                    if swa_valid_len < self.sliding_window_pages:
                        self.prefetch_revoke_queue.put(operation.request_id)
                        revoke_tokens = len(operation.host_indices)
                        self.append_host_mem_release(operation.host_indices)
                        self.append_swa_only_host_mem_release(
                            operation.swa_host_indices
                        )
                        if self.metrics_collector is not None:
                            self.metrics_collector.log_revoke_tokens(revoke_tokens)
                        logger.debug(
                            f"Revoking prefetch for request {operation.request_id} due to insufficient SWA hits ({swa_valid_len})."
                        )
                        continue

                # 截取需要从 storage prefetch 的部分
                operation.hash_value = hash_value[:hit_pages]
                # FIXME(yingchun): 需要先 free，再截断
                operation.host_indices = operation.host_indices[:storage_hit_count]
                # free the pre-allocated memory for pages that are not hit
                self.append_host_mem_release(operation.host_indices[storage_hit_count:])
                # 把一开始未做 mapping 的 full 和 swa 的 host memory 做 mapping
                if self.is_hybrid_swa:
                    assert valid_swa_mask is not None
                    trimmed_swa_mask, surplus_swa = (
                        self.mem_pool_host.mapping_full_and_swa_indices(
                            operation.host_indices,
                            operation.swa_host_indices,
                            valid_swa_mask[:hit_pages],
                        )
                    )
                    # free the SWA ONLY pre-allocated memory for pages that are not hit
                    if surplus_swa is not None:
                        self.append_swa_only_host_mem_release(surplus_swa)
                    # Reset swa_host_indices to None after being mapped
                    operation.swa_host_indices = None
                    operation.swa_valid_mask = trimmed_swa_mask

                logger.debug(
                    f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                )
                self.prefetch_buffer.put(operation)

            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), self.storage_batch_size):
            batch_hashes = operation.hash_value[i : i + self.storage_batch_size]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            assert len(batch_hashes) > 0, f"{len(batch_hashes)=}"
            assert (
                len(batch_hashes)
                == len(batch_host_indices) // self.mem_pool_host.page_size
            ), f"{len(batch_hashes)=}, {len(batch_host_indices)=}, {self.mem_pool_host.page_size=}"
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break
            logger.debug(f"Write page to storage: {len(batch_hashes)} pages succeeded.")

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
