from __future__ import annotations

import abc
import atexit
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
    SWATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.simple_swa_radix_cache import (
    SimpleSWARadixCache,
    compute_swa_safe_len,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.utils import convert_to_steps_key
from sglang.srt.observability.metrics_collector import StorageMetricsCollector

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

"""
从根到叶，状态严格按照 backuped → device_only → evicted 的顺序排列，每段都可以为空，但不会交替出现。
具体约束：
1. evicted 只出现在连续的尾部段（eviction 从叶向根传播）
2. evicted 节点必有 host backup（无 backup 的节点被 `_evict_regular` 直接删除）
3. backuped(device+host) 从根部开始连续（祖先 hit_count >= 子孙 hit_count）
4. device_only 在 backuped 和 evicted 之间（已达到 device 但还没达到 backup 阈值的节点）

"""


class BaseHiRadixCache(abc.ABC):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, SWAKVPool):
            self.token_to_kv_pool_host = SWATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                device=params.token_to_kv_pool_allocator.device,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"BaseHiRadixCache only supports MHA, MLA and SWA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            enable_storage_metrics=self.enable_storage_metrics,
            sliding_window_size=params.sliding_window_size,
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        self.enable_hicache_metrics = self._enable_metrics_flag
        if self.enable_hicache_metrics:
            from prometheus_client import Counter

            hicache_labels = {
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
            }
            self._hicache_labels = hicache_labels

            self._write_backup_host_full_alloc_retry_total = Counter(
                name="sglang:write_backup_host_full_alloc_retry_total",
                documentation="Number of write_backup retries triggered by insufficient full-attention host memory.",
                labelnames=hicache_labels.keys(),
            )
            self._write_backup_host_swa_alloc_retry_total = Counter(
                name="sglang:write_backup_host_swa_alloc_retry_total",
                documentation="Number of write_backup retries triggered by insufficient SWA host memory.",
                labelnames=hicache_labels.keys(),
            )
            self._load_back_device_full_alloc_retry_total = Counter(
                name="sglang:load_back_device_full_alloc_retry_total",
                documentation="Number of load_back retries triggered by insufficient full-attention device memory.",
                labelnames=hicache_labels.keys(),
            )
            self._load_back_device_swa_alloc_retry_total = Counter(
                name="sglang:load_back_device_swa_alloc_retry_total",
                documentation="Number of load_back retries triggered by insufficient SWA device memory.",
                labelnames=hicache_labels.keys(),
            )
            self._write_backup_host_full_alloc_failure_total = Counter(
                name="sglang:write_backup_host_full_alloc_failure_total",
                documentation="Number of write_backup failures due to insufficient full-attention host memory.",
                labelnames=hicache_labels.keys(),
            )
            self._write_backup_host_swa_alloc_failure_total = Counter(
                name="sglang:write_backup_host_swa_alloc_failure_total",
                documentation="Number of write_backup failures due to insufficient SWA host memory.",
                labelnames=hicache_labels.keys(),
            )
            self._load_back_device_full_alloc_failure_total = Counter(
                name="sglang:load_back_device_full_alloc_failure_total",
                documentation="Number of load_back failures due to insufficient full-attention device memory.",
                labelnames=hicache_labels.keys(),
            )
            self._load_back_device_swa_alloc_failure_total = Counter(
                name="sglang:load_back_device_swa_alloc_failure_total",
                documentation="Number of load_back failures due to insufficient SWA device memory.",
                labelnames=hicache_labels.keys(),
            )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.prefetch_io_duration_by_reqid = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.sliding_window_size = params.sliding_window_size

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
                self.cache_controller.metrics_collector = self.storage_metrics_collector
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
        n_swa_only_release: Optional[int] = None,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                self.prefetch_io_duration_by_reqid.pop(req_id, None)
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        def _drain_swa_only_release():
            if n_swa_only_release is None:
                return
            swa_indices_list = []
            for swa_indices in _drain_queue(
                cc.swa_only_host_mem_release_queue, n_swa_only_release
            ):
                swa_indices_list.append(swa_indices)
            if swa_indices_list:
                swa_host_indices = torch.cat(swa_indices_list, dim=0)
                cc.mem_pool_host.free_swa_only(swa_host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()
        _drain_swa_only_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.prefetch_io_duration_by_reqid.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def _log_write_backup_host_alloc_retry(
        self, lack_full: bool, lack_swa: bool
    ) -> None:
        if lack_full:
            self._write_backup_host_full_alloc_retry_total.labels(
                **self._hicache_labels
            ).inc(1)
        if lack_swa:
            self._write_backup_host_swa_alloc_retry_total.labels(
                **self._hicache_labels
            ).inc(1)

    def _log_write_backup_host_alloc_failure(
        self, lack_full: bool, lack_swa: bool
    ) -> None:
        if lack_full:
            self._write_backup_host_full_alloc_failure_total.labels(
                **self._hicache_labels
            ).inc(1)
        if lack_swa:
            self._write_backup_host_swa_alloc_failure_total.labels(
                **self._hicache_labels
            ).inc(1)

    def _log_load_back_device_alloc_retry(
        self, lack_full: bool, lack_swa: bool
    ) -> None:
        if lack_full:
            self._load_back_device_full_alloc_retry_total.labels(
                **self._hicache_labels
            ).inc(1)
        if lack_swa:
            self._load_back_device_swa_alloc_retry_total.labels(
                **self._hicache_labels
            ).inc(1)

    def _log_load_back_device_alloc_failure(
        self, lack_full: bool, lack_swa: bool
    ) -> None:
        if lack_full:
            self._load_back_device_full_alloc_failure_total.labels(
                **self._hicache_labels
            ).inc(1)
        if lack_swa:
            self._load_back_device_swa_alloc_failure_total.labels(
                **self._hicache_labels
            ).inc(1)

    def write_backup(self, node: TreeNode, write_back=False):
        full_host_indices, lack_full_size, lack_swa_size = (
            self.cache_controller.alloc_host(
                device_indices=node.value,
            )
        )
        if full_host_indices is None:
            assert lack_full_size > 0 or lack_swa_size > 0
            if self.enable_hicache_metrics:
                self._log_write_backup_host_alloc_retry(
                    lack_full=lack_full_size > 0,
                    lack_swa=lack_swa_size > 0,
                )
            self.evict_host(lack_full_size, lack_swa_size)
            full_host_indices, lack_full_size, lack_swa_size = (
                self.cache_controller.alloc_host(
                    device_indices=node.value,
                )
            )
        if full_host_indices is None:
            if self.enable_hicache_metrics:
                self._log_write_backup_host_alloc_failure(
                    lack_full=lack_full_size > 0,
                    lack_swa=lack_swa_size > 0,
                )
            return 0, 0

        self.cache_controller.write(
            device_indices=node.value,
            host_indices=full_host_indices,
            node_id=node.id,
        )
        node.host_value = full_host_indices
        assert len(node.host_value) > 0
        self.ongoing_write_through[node.id] = {
            "node": node,
            "key": RadixKey(token_ids=node.key.token_ids),
            "hash_value": list(node.hash_value) if node.hash_value else None,
            "host_value": full_host_indices,
        }
        if not write_back:
            # no need to lock nodes if write back
            self.inc_lock_ref(node)

        return len(full_host_indices), self.cache_controller.get_valid_swa_host_len(
            full_host_indices
        )

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        snapshot = self.ongoing_write_through.pop(ack_id)
                        backuped_node = snapshot["node"]
                        # TODO(yingchun): 注意 write_back 时，存储的不是之前的快照值。
                        #  split node 之后的新节点会延迟到它自己触发 write_back 时再存储。
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            cc = self.cache_controller
            states = torch.tensor(
                [
                    finish_count,
                    cc.mem_pool_host.full_pool_host.available_size(),
                    cc.mem_pool_host.swa_pool_host.available_size(),
                ],
                dtype=torch.int,
                device="cpu",
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            (
                finish_count,
                n_host_full_available_size,
                n_host_swa_available_size,
            ) = states.tolist()
            # TODO(yingchun): 重复代码，可封装为函数
            assert (
                n_host_full_available_size
                == cc.mem_pool_host.full_pool_host.available_size()
            )
            assert (
                n_host_swa_available_size
                == cc.mem_pool_host.swa_pool_host.available_size()
            )

        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                snapshot = self.ongoing_write_through.pop(ack_id)
                backuped_node = snapshot["node"]
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    current_key = backuped_node.key
                    current_hash_value = backuped_node.hash_value
                    current_host_value = backuped_node.host_value
                    try:
                        # 当 node 发生 split 时，此时的 hash_value, host_value 已经是之前把 node 放入 ongoing_write_through
                        # 时的子集，缺失了部分数据，这里是保证写到 storage 的数据与入队时的一致。
                        backuped_node.key = snapshot["key"]
                        backuped_node.hash_value = snapshot["hash_value"]
                        backuped_node.host_value = snapshot["host_value"]
                        self.write_backup_storage(backuped_node)
                    finally:
                        # Restore current values to keep tree structure intact
                        backuped_node.key = current_key
                        backuped_node.hash_value = current_hash_value
                        backuped_node.host_value = current_host_value
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def _to_radix_key(self, token_ids: List[int]) -> RadixKey:
        """Convert raw token_ids to a RadixKey for tree walking.

        Must use list (not tuple) to match scheduler's RadixKey format,
        since _key_match_paged compares slices directly and list != tuple.
        """
        return RadixKey(token_ids=list(token_ids))

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        full_evicted = 0
        swa_evicted = 0
        write_back_nodes = []
        while (full_evicted < full_num_tokens or swa_evicted < swa_num_tokens) and len(
            eviction_heap
        ):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    freed_full, freed_swa = self.write_backup(x, write_back=True)
                    full_evicted += freed_full
                    swa_evicted += freed_swa
                    write_back_nodes.append(x)
                else:
                    freed_full, freed_swa = self._evict_regular(x)
                    full_evicted += freed_full
                    swa_evicted += freed_swa
            else:
                freed_full, freed_swa = self._evict_backuped(x)
                full_evicted += freed_full
                swa_evicted += freed_swa

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        if full_evicted < full_num_tokens or swa_evicted < swa_num_tokens:
            logger.info(
                f"Evicted {full_evicted}/{full_num_tokens} full tokens and "
                f"{swa_evicted}/{swa_num_tokens} swa tokens from device cache"
            )

        self.update_eviction_metrics(full_evicted + swa_evicted, start_time)
        return EvictResult(
            num_tokens_evicted=full_evicted, swa_num_tokens_evicted=swa_evicted
        )

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU demotion: no BlockRemoved since block is still reachable via load_back
        full_evicted, swa_evicted = self.cache_controller.evict_device(node.value)
        assert full_evicted > 0
        self.evictable_size_ -= full_evicted
        # TODO(yingchun): swa_evictable_size 是在 SimpleSWARadixCache 的 _collect_swa_sizes() 中动态计算的，那里的实时计算会更费一些。这里可以优化？
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return full_evicted, swa_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        self._record_remove_event(node)
        # TODO(yingchun): 使用 cache_controller.evict_device() 重构
        swa_device_indices = self.cache_controller.mem_pool_device_allocator.translate_loc_from_full_to_swa(
            node.value
        )
        full_evicted = len(node.value)
        swa_evicted = torch.count_nonzero(swa_device_indices).item()
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        self._delete_leaf(node)
        return full_evicted, swa_evicted

    def evict_host(self, full_num_tokens: int, swa_num_tokens: int = 0):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        full_evicted = 0
        swa_evicted = 0
        while (full_evicted < full_num_tokens or swa_evicted < swa_num_tokens) and len(
            eviction_heap
        ):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict host value of already-evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit BlockRemoved so the router removes this block from its index.
            self._record_remove_event(x)
            freed_full, freed_swa = self.cache_controller.evict_host(x.host_value)
            full_evicted += freed_full
            swa_evicted += freed_swa

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancestor_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancestor_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancestor_node)
            return None

        full_device_indices, lack_full_size, lack_swa_size = (
            self.cache_controller.alloc_device(
                host_indices=host_indices,
            )
        )
        if full_device_indices is None:
            assert lack_full_size > 0 or lack_swa_size > 0
            if self.enable_hicache_metrics:
                self._log_load_back_device_alloc_retry(
                    lack_full=lack_full_size > 0,
                    lack_swa=lack_swa_size > 0,
                )
            if self.debug_mode:
                self.cache_controller.assert_host_indices(
                    host_indices=host_indices,
                    host_hit_length=self.sliding_window_size,
                )
            self.evict(EvictParams(num_tokens=lack_full_size, swa_num_tokens=lack_swa_size))
            if self.debug_mode:
                self.cache_controller.assert_host_indices(
                    host_indices=host_indices,
                    host_hit_length=self.sliding_window_size,
                )
            full_device_indices, lack_full_size, lack_swa_size = (
                self.cache_controller.alloc_device(
                    host_indices=host_indices,
                )
            )

        # no sufficient GPU memory to load back KV caches
        if full_device_indices is None:
            if self.enable_hicache_metrics:
                self._log_load_back_device_alloc_failure(
                    lack_full=lack_full_size > 0,
                    lack_swa=lack_swa_size > 0,
                )
            self.dec_lock_ref(ancestor_node)
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.cache_controller.load(
            host_indices=host_indices,
            device_indices=full_device_indices,
            node_id=last_hit_node.id,
        )
        self.dec_lock_ref(ancestor_node)

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = full_device_indices[
                offset : offset + len(node.host_value)
            ].clone()
            offset += len(node.host_value)
            assert len(node.value) == len(node.host_value)
        assert offset == len(full_device_indices)
        self.evictable_size_ += len(full_device_indices)
        # TODO(yingchun): swa_evictable_size 同上
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(
                len(full_device_indices)
            )

        return full_device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            storage_metrics = self.cache_controller.storage_backend.get_stats()
            if storage_metrics is not None:
                cc = self.cache_controller
                storage_metrics.prefetch_queue_length = cc.prefetch_queue.qsize()
                storage_metrics.prefetch_buffer_length = (
                    cc.prefetch_buffer.qsize() if hasattr(cc, "prefetch_buffer") else 0
                )
                storage_metrics.ongoing_prefetch_queue_length = len(
                    self.ongoing_prefetch
                )
            self.storage_metrics_collector.log_storage_metrics(storage_metrics)

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        if cc.is_hybrid_swa:
            qsizes = torch.tensor(
                [
                    cc.prefetch_revoke_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                    cc.swa_only_host_mem_release_queue.qsize(),
                    cc.mem_pool_host.full_pool_host.available_size(),
                    cc.mem_pool_host.swa_pool_host.available_size(),
                ],
                dtype=torch.int,
            )
        else:
            qsizes = torch.tensor(
                [
                    cc.prefetch_revoke_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                ],
                dtype=torch.int,
            )

        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_swa_only_release = None
        if cc.is_hybrid_swa:
            n_revoke, n_backup, n_release, n_swa_only_release, n_host_full_available_size, n_host_swa_available_size = map(
                int, qsizes.tolist()
            )
            assert (
                n_host_full_available_size
                == cc.mem_pool_host.full_pool_host.available_size()
            )
            assert (
                n_host_swa_available_size == cc.mem_pool_host.swa_pool_host.available_size()
            )
        else:
            n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
            n_swa_only_release=n_swa_only_release,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            cc = self.cache_controller
            states = torch.tensor(
                [
                    1 - int(can_terminate),
                    int(operation_terminated),
                    cc.mem_pool_host.full_pool_host.available_size(),
                    cc.mem_pool_host.swa_pool_host.available_size(),
                ],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            (
                can_terminate,
                operation_terminated,
                n_host_full_available_size,
                n_host_swa_available_size,
            ) = states.tolist()
            # all TP can_terminate are True
            can_terminate = can_terminate == 0
            # any TP operation_terminated is True
            operation_terminated = operation_terminated == 1
            # TODO(yingchun): Defensive programming only, maybe can be removed
            assert (
                n_host_full_available_size
                == cc.mem_pool_host.full_pool_host.available_size()
            )
            assert (
                n_host_swa_available_size
                == cc.mem_pool_host.swa_pool_host.available_size()
            )
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            assert operation.swa_host_indices is None
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to hiradix cache
            cc = self.cache_controller
            states = torch.tensor(
                [
                    min_completed_tokens,
                    # TODO(yingchun): Defensive programming only, maybe can be removed
                    cc.mem_pool_host.full_pool_host.available_size(),
                    cc.mem_pool_host.swa_pool_host.available_size(),
                ],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            (
                min_completed_tokens,
                n_host_full_available_size,
                n_host_swa_available_size,
            ) = states.tolist()
            assert (
                n_host_full_available_size
                == cc.mem_pool_host.full_pool_host.available_size()
            )
            assert (
                n_host_swa_available_size
                == cc.mem_pool_host.swa_pool_host.available_size()
            )

        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_full_and_swa_mem_release_separately(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        self.prefetch_io_duration_by_reqid[req_id] = operation.get_io_duration()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetch_dropped_tokens(matched_length)
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            assert operation.swa_host_indices is None
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def pop_prefetch_io_duration(self, req_id: str) -> Optional[float]:
        return self.prefetch_io_duration_by_reqid.pop(req_id, None)

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        page_aligned_len = len(key)
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        # 在 tree 上 (device and/or host) 匹配前缀，得到 device 上的命中结果``value``和最后一个匹配的 node ``last_node``（可能是 evicted 状态，即仅 host 上有）
        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        # (无 device + 有 host) 的情况，需要从 host 上数有多少 token 是命中的
        host_hit_length = 0
        last_host_node = last_node
        # 从 tree 上匹配到的节点向 root 回溯，找出需要从 host 上 load back 的长度（host_hit_length），更新在 device 上的最后一个节点（last_node）
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        # 从 tree 上匹配到的节点向 root 回溯，找出已在 host 上 backuped 的最后一个节点（last_host_node）
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> bool:
        new_input_tokens = (
            convert_to_steps_key(new_input_tokens, self.speculative_num_steps)
            if self.is_eagle
            else new_input_tokens
        )
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return False

        last_host_node.protect_host()
        full_host_indices, swa_host_indices, lack_full_size, lack_swa_size = (
            self.cache_controller.mem_pool_host.alloc_unmapped_host_by_len(
                prefetch_length
            )
        )
        if full_host_indices is None:
            assert lack_full_size > 0 or lack_swa_size > 0
            self.evict_host(lack_full_size, lack_swa_size)
            full_host_indices, swa_host_indices, _, _ = (
                self.cache_controller.mem_pool_host.alloc_unmapped_host_by_len(
                    prefetch_length
                )
            )
        if full_host_indices is None:
            assert swa_host_indices is None
            last_host_node.release_host()
            # no sufficient host memory for prefetch
            return False
        operation = self.cache_controller.prefetch(
            req_id,
            full_host_indices,
            swa_host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            full_host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)
        return True

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        # 收集的是 device 上的数据
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        # FIXME(yingchun): 不能赋值！
        new_node.host_ref_counter = child.host_ref_counter
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_io_duration_by_reqid.pop(rid, None)
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            assert operation.swa_host_indices is None
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_full_and_swa_mem_release_separately(
            host_indices[:completed_tokens]
        )
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)


def _collect_evicted_host_indices(
    last_host_node: TreeNode, last_device_node: TreeNode
) -> torch.Tensor:
    """
    Collect host_value indices from evicted nodes on the matched path.

    When ``host_hit_length > 0``, ``last_host_node`` in the MatchResult is
    the actual match endpoint (deepest matched node, which is evicted and
    therefore always backuped). ``last_device_node`` is the first non-evicted
    ancestor found by walking up.

    We walk UP from ``last_host_node`` to ``last_device_node``, collecting
    each evicted node's ``host_value`` in reverse, then reverse the list to
    get root-to-leaf order. This avoids the ambiguity of walking DOWN through
    potentially branching children.
    """

    host_values = []
    node = last_host_node
    while node is not last_device_node:
        assert node.evicted, "Only evicted nodes should be traversed for host indices"
        assert node.backuped, "Evicted node must have host backup"
        host_values.append(node.host_value)
        node = node.parent

    if not host_values:
        raise RuntimeError(
            "_collect_evicted_host_indices called with last_host_node == "
            "last_device_node; expected at least one evicted node in between."
        )

    # Reverse to get root-to-leaf order
    host_values.reverse()
    return torch.cat(host_values)


class HiRadixCache(BaseHiRadixCache, RadixCache):
    pass


class HiSimpleSWARadixCache(BaseHiRadixCache, SimpleSWARadixCache):
    """
    BaseHiRadixCache with SWA-aware prefix matching.

    Combines BaseHiRadixCache's hierarchical (device → host → storage) KV cache
    management with SimpleSWARadixCache's SWA safety checking.

    For a matched prefix to be reusable, the SWA (Sliding Window Attention)
    mappings in the tail window must all be valid (non-zero). This applies to
    both:
    - **Device-resident** portion: checked via the device allocator's
      ``full_to_swa_index_mapping``.
    - **Host-resident** portion (evicted nodes to be loaded back): checked via
      the host pool's ``full_to_swa_index_mapping``, since ``alloc_device()``
      will restore SWA mappings from the host mapping during load-back.
    """

    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
    ):
        # Validate SWA-specific preconditions before BaseHiRadixCache initialization
        # (BaseHiRadixCache will allocate host pool / cache controller; we want to
        # fail fast if the allocator type is wrong).
        assert isinstance(
            params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        ), (
            f"HiSimpleSWARadixCache requires SWATokenToKVPoolAllocator, "
            f"got {type(params.token_to_kv_pool_allocator)}"
        )
        assert (
            params.sliding_window_size is not None and params.sliding_window_size > 0
        ), f"sliding_window_size must be positive, got {params.sliding_window_size}"

        # Initialize BaseHiRadixCache (which initializes RadixCache via super()).
        # We don't chain through SimpleSWARadixCache.__init__ because it would
        # re-invoke RadixCache.__init__ via its own super(); instead we set the
        # SimpleSWA-specific fields manually below to avoid double init.
        BaseHiRadixCache.__init__(self, params, server_args)

        # SimpleSWA-specific state mirrored from SimpleSWARadixCache.__init__.
        self.sliding_window_size = params.sliding_window_size
        self.swa_active_freed_tokens: int = 0
        self.swa_evicted_from_tree: int = 0
        self.full_evicted_from_tree: int = 0
        self._last_raw_match_len: int = 0

        # Force load_back_threshold to 0 for SWA mode: any host-resident hit
        # must be loaded back to device before it can be safely reused, since
        # SWA validity is checked across the combined device+host prefix.
        self.load_back_threshold = 0
        logger.warning(
            "Initializing HiSimpleSWARadixCache and force set load_back_threshold = 0"
        )

    def _compute_safe_prefix_len(
        self, full_matched_result: MatchResult
    ) -> tuple[int, int]:
        """Compute the SWA-safe prefix length for both device and host portions.

        The full matched prefix consists of two contiguous parts:
          [device_indices (on GPU)] [host-only evicted nodes (to be loaded back)]

        For the combined prefix of length ``total_len = device_len + host_len``,
        the tail window ``[max(0, total_len - sliding_window_size), total_len)``
        must have valid SWA mappings. Delegates to ``compute_swa_safe_len()``
        for the core sliding-window validity check.

        Returns:
            A tuple ``(swa_safe_device_len, swa_safe_host_len)`` where:

            - **swa_safe_device_len** (int): Number of leading device_indices
              that are SWA-safe.  Always ``<= device_len``.
            - **swa_safe_host_len** (int):
              - ``int >= 0`` — the number of host tokens (counted from the
                device/host boundary toward the leaf) that are SWA-safe.
                ``0`` means no host portion is usable.  Note that when
                ``swa_safe_device_len < device_len``, ``swa_safe_host_len``
                will always be ``0`` because trimming the device portion
                invalidates the host portion as well.
        """
        device_indices = full_matched_result.device_indices
        device_len = len(device_indices)
        host_hit_length = full_matched_result.host_hit_length
        total_len = device_len + host_hit_length

        # 1. Fast path: nothing matched, return immediately without any SWA checks
        if total_len == 0:
            assert (
                device_len == 0 and host_hit_length == 0
            ), f"{device_len=}, {host_hit_length=}"
            return 0, 0

        # 2. Build the combined SWA index vector for the full matched prefix.
        swa_parts = []

        # 2.1 Device portion: lookup from device allocator's mapping.
        if device_len > 0:
            device_swa = self.token_to_kv_pool_allocator.full_to_swa_index_mapping[
                device_indices
            ]
            swa_parts.append(device_swa)

        # 2.2 Host portion: lookup from host pool's mapping (keyed by host_value indices).
        if host_hit_length > 0:
            # Collect host_value indices from the evicted tail nodes.
            # When host_hit_length > 0, last_host_node is the actual match
            # endpoint (deepest evicted node), and last_device_node is the
            # first non-evicted ancestor. Walk UP from last_host_node to
            # last_device_node to collect host_indices.
            host_indices = _collect_evicted_host_indices(
                full_matched_result.last_host_node,
                full_matched_result.last_device_node,
            )
            assert (
                host_hit_length == host_indices.numel()
            ), f"{host_hit_length=}, {host_indices.numel()=}"
            host_swa = self.cache_controller.get_swa_host_indices(host_indices)
            swa_parts.append(host_swa)

        swa_indices = torch.cat(swa_parts) if len(swa_parts) > 1 else swa_parts[0]
        assert len(swa_indices) == total_len, f"{len(swa_indices)=}, {total_len=}"

        # 3. Delegate to the shared SWA safety check algorithm.
        safe_total_len = compute_swa_safe_len(swa_indices, self.sliding_window_size)
        assert safe_total_len <= total_len, f"{safe_total_len=}, {total_len=}"

        # 4. Split the safe length back into device and host portions.
        safe_device_len = min(safe_total_len, device_len)
        safe_host_len = max(0, safe_total_len - device_len)
        return safe_device_len, safe_host_len

    def match_prefix(self, params):
        """SWA-aware match: trim either device or host portion based on SWA validity.

        ``params`` is a ``MatchPrefixParams`` (current API). We delegate to
        ``BaseHiRadixCache.match_prefix`` for the raw tree match (which returns a
        ``MatchResult`` with possibly-non-empty ``host_hit_length``), then apply
        SWA safety checks to trim either the device portion or the host portion.
        """
        assert self.load_back_threshold == 0

        full_matched_result = BaseHiRadixCache.match_prefix(self, params)
        full_matched_device_len = len(full_matched_result.device_indices)
        full_matched_host_len = full_matched_result.host_hit_length
        self._last_raw_match_len = full_matched_device_len + full_matched_host_len

        if full_matched_device_len + full_matched_host_len == 0:
            assert (
                full_matched_device_len == 0 and full_matched_host_len == 0
            ), f"{full_matched_device_len=}, {full_matched_host_len=}"
            return full_matched_result

        swa_matched_device_len, swa_matched_host_len = self._compute_safe_prefix_len(
            full_matched_result
        )
        logger.debug(
            f"{full_matched_device_len=}, {full_matched_host_len=}, {swa_matched_device_len=}, {swa_matched_host_len=}"
        )

        # Trim device_indices if needed.
        if swa_matched_device_len < full_matched_device_len:
            assert (
                swa_matched_host_len == 0
            ), f"When device needs trimming, host must be 0: {swa_matched_host_len=}"
            if swa_matched_device_len == 0:
                # Nothing is SWA-safe — return empty match at root.
                return MatchResult(
                    device_indices=torch.empty(
                        (0,), dtype=torch.int64, device=self.device
                    ),
                    last_device_node=self.root_node,
                    last_host_node=self.root_node,
                    host_hit_length=0,
                )
        else:
            assert (
                swa_matched_device_len == full_matched_device_len
            ), f"{swa_matched_device_len=}, {full_matched_device_len=}"
            assert (
                0 <= swa_matched_host_len <= full_matched_host_len
            ), f"{swa_matched_host_len=} must be in [0, {full_matched_host_len=}]"

        swa_matched_total_len = swa_matched_device_len + swa_matched_host_len
        if self.is_eagle:
            swa_matched_total_len = (
                (swa_matched_total_len + self.speculative_num_steps)
                if swa_matched_total_len > 0
                else 0
            )
        key = params.key
        swa_matched_params = MatchPrefixParams(
            key=RadixKey(key.token_ids[:swa_matched_total_len], key.extra_key),
            req=params.req,
        )
        # TODO(yingchun): 需要做优化避免此处的二次 match_prefix
        swa_rematched_result = BaseHiRadixCache.match_prefix(self, swa_matched_params)
        logger.debug(
            f"{len(swa_rematched_result.device_indices)=}, {swa_rematched_result.host_hit_length=}"
        )
        assert (
            swa_rematched_result.host_hit_length
            + len(swa_rematched_result.device_indices)
            == swa_matched_device_len + swa_matched_host_len
        ), f"{swa_rematched_result.host_hit_length=} + {len(swa_rematched_result.device_indices)=} != {swa_matched_device_len=} + {swa_matched_host_len=}"
        if self.debug_mode:
            if len(swa_rematched_result.last_host_node.host_value) == 0:
                assert (
                    swa_rematched_result.host_hit_length == 0
                    and swa_rematched_result.last_host_node
                    is swa_rematched_result.last_device_node
                ), f"{swa_rematched_result.host_hit_length=}, {swa_rematched_result.last_host_node=}, {swa_rematched_result.last_device_node=}"
            elif swa_rematched_result.host_hit_length > 0:
                self.cache_controller.assert_host_indices(
                    swa_rematched_result.last_host_node.host_value,
                    swa_rematched_result.host_hit_length,
                )

        return swa_rematched_result
