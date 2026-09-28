"""GCache MHA Full/SWA/draft storage for UnifiedRadixCache / HiCache."""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_dp_rank
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.gcache.client import get_client_runtime
from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig
from sglang.srt.mem_cache.storage.gcache.key_codec import GCacheKeyCodec
from sglang.srt.mem_cache.storage.gcache.metrics import GCacheMetrics
from sglang.srt.observability.metrics_collector import StorageMetrics

logger = logging.getLogger(__name__)


class GcacheStorage(HiCacheStorage):
    # HybridCacheController may intersect the complete endpoint sets across TP
    # ranks instead of reducing maxima that can fall in another rank's SWA gap.
    supports_restorable_prefixes = True

    def __init__(
        self, storage_config: HiCacheStorageConfig, mem_pool_host: HostKVCache
    ):
        self._validate_topology(storage_config)
        self._validate_pool(mem_pool_host)
        self.key_codec = GCacheKeyCodec(
            storage_config.model_name,
            storage_config.tp_rank,
            envs.SGLANG_HICACHE_DRAFT_MODEL_NAME.get(),
        )
        self.tp_rank = storage_config.tp_rank
        # The controller already supplied attention TP coordinates. DP belongs
        # in the process config, never in the shared cache object identity.
        self.dp_rank = get_attention_dp_rank()
        self.client_config = GCacheClientConfig.from_env(storage_config.extra_config)
        self._runtime = get_client_runtime(
            self.client_config, self.dp_rank, self.tp_rank
        )
        self.config_path = self._runtime.config_path
        self.gcache_client = self._runtime.client
        self.mem_pool_host = mem_pool_host
        self.registered_pools: dict[PoolName, HostKVCache] = {}
        # Keep the actual allocations alive, including when enumerated through
        # offset views of a shared K/V buffer.
        self._registered_storages: dict[tuple[int, int], torch.UntypedStorage] = {}
        self._pool_ranges: dict[PoolName, tuple[tuple[int, int], ...]] = {}
        self._component_sizes: dict[PoolName, list[int]] = {}
        self._registration_lock = threading.Lock()
        self._condition = threading.Condition()
        self._active_operations = 0
        self._closed = False
        self.enable_storage_metrics = storage_config.enable_storage_metrics
        self._metrics = StorageMetrics()
        self._metrics_lock = threading.Lock()
        self._gcache_metrics = (
            GCacheMetrics(self.tp_rank, self.dp_rank)
            if self.enable_storage_metrics
            else None
        )

    @staticmethod
    def _validate_topology(config: HiCacheStorageConfig) -> None:
        if config.is_mla_model:
            raise NotImplementedError("GCache supports MHA KV only")
        if config.pp_size != 1 or config.attn_cp_size != 1:
            raise NotImplementedError("GCache requires pp_size=1 and attn_cp_size=1")
        if config.pp_rank != 0 or config.attn_cp_rank != 0:
            raise ValueError("GCache requires zero PP and attention CP ranks")
        if config.tp_size < 1 or not 0 <= config.tp_rank < config.tp_size:
            raise ValueError("Invalid GCache attention TP rank/size")
        if config.should_split_heads or config.tp_lcm_size not in (
            None,
            config.tp_size,
        ):
            raise NotImplementedError(
                "GCache does not support heterogeneous TP or split heads"
            )

    @staticmethod
    def _validate_pool(pool: HostKVCache) -> None:
        # AsymmetricMHATokenToKVPoolHost inherits this class and exposes two
        # independent allocations with different per-page byte lengths.
        if not isinstance(pool, MHATokenToKVPoolHost):
            raise TypeError("GCache requires an MHA host pool")
        if pool.layout not in ("page_first", "page_first_direct"):
            raise ValueError(f"Unsupported GCache host layout: {pool.layout}")
        if getattr(pool, "dcp_size", 1) != 1:
            raise NotImplementedError("GCache does not support DCP host pools")
        if getattr(pool, "mtp_draft_device_pools", ()):
            raise NotImplementedError("GCache does not support packed MTP draft layers")
        if pool.page_size <= 0 or pool.size < pool.page_size or pool.layer_num <= 0:
            raise ValueError("GCache requires a non-empty MHA host pool")

    @contextmanager
    def _operation(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("GCache backend is closed")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_operations -= 1
                self._condition.notify_all()

    def close(self) -> None:
        """Drain this backend without calling the SDK's global fini()."""
        with self._condition:
            self._closed = True
            self._condition.wait_for(lambda: self._active_operations == 0)
            self.registered_pools.clear()
            self._pool_ranges.clear()
            self._component_sizes.clear()
            self._registered_storages.clear()
            self.mem_pool_host = None
        # The process runtime retains the actual allocations: the pinned SDK
        # cannot unregister them, and reattach must reuse their registration.

    @staticmethod
    def _iter_tensors(buffers):
        if buffers is None:
            return
        if isinstance(buffers, torch.Tensor):
            if buffers.numel() > 0:
                yield buffers
        elif isinstance(buffers, (list, tuple)):
            for buffer in buffers:
                yield from GcacheStorage._iter_tensors(buffer)
        else:
            raise TypeError(f"Unsupported GCache host buffer type: {type(buffers)}")

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        if mem_pool_host is not self.mem_pool_host:
            raise ValueError("Cannot replace the GCache primary host pool")
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name) -> None:
        if host_pool_name not in (
            PoolName.KV,
            PoolName.SWA,
            PoolName.DRAFT,
            PoolName.DRAFT_SWA,
        ):
            raise NotImplementedError(f"Unsupported GCache pool: {host_pool_name}")
        name = PoolName(host_pool_name)
        with self._operation(), self._registration_lock:
            if name == PoolName.KV and host_pool is not self.mem_pool_host:
                raise ValueError("Cannot replace the GCache primary host pool")
            existing = self.registered_pools.get(name)
            if existing is not None:
                if existing is not host_pool:
                    raise ValueError(f"Cannot replace registered GCache pool {name}")
                return
            self._validate_pool(host_pool)
            accessor = getattr(host_pool, "get_hybrid_pool_buffer", None)
            buffers = accessor() if callable(accessor) else host_pool.kv_buffer
            allocations = {}
            for tensor in self._iter_tensors(buffers):
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError("GCache requires contiguous CPU host buffers")
                storage = tensor.untyped_storage()
                if storage.nbytes():
                    allocations[storage.data_ptr(), storage.nbytes()] = storage

            # Validate against this pool's allocations before native registration.
            # A pointer into a different registered pool is never a valid range.
            ptrs, sizes = host_pool.get_page_buffer_meta(
                torch.arange(host_pool.page_size, dtype=torch.int64)
            )
            self._validate_metadata(ptrs, sizes, 1, allocations)
            self._runtime.register(allocations)
            self._registered_storages.update(allocations)
            self._pool_ranges[name] = tuple(allocations)
            self._component_sizes[name] = list(sizes)
            self.registered_pools[name] = host_pool

    @staticmethod
    def _validate_metadata(ptrs, sizes, pages: int, ranges) -> None:
        if len(ptrs) != pages * 2 or len(sizes) != pages * 2:
            raise ValueError("GCache requires exactly one K and one V range per page")
        for ptr, size in zip(ptrs, sizes):
            if type(ptr) is not int or type(size) is not int or size <= 0:
                raise ValueError("Invalid GCache page pointer or byte length")
            if not any(
                base <= ptr and ptr + size <= base + length for base, length in ranges
            ):
                raise ValueError("GCache page range is outside registered host memory")

    def _page_metadata(self, name: PoolName, keys: list[str], indices: torch.Tensor):
        pool = self.registered_pools[name]
        page_size = pool.page_size
        if (
            not isinstance(indices, torch.Tensor)
            or indices.device.type != "cpu"
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.ndim != 1
            or indices.numel() != len(keys) * page_size
        ):
            raise ValueError(
                "GCache host_indices must contain one complete CPU integer page per key"
            )
        rows = indices.reshape(-1, page_size)
        starts = rows[:, 0]
        if (
            torch.any(starts < 0)
            or torch.any(starts + page_size > pool.size)
            or torch.any(starts % page_size != 0)
            or len(set(starts.tolist())) != len(keys)
            or not torch.equal(rows, starts[:, None] + torch.arange(page_size))
        ):
            raise ValueError("GCache requires distinct, aligned, contiguous host pages")
        ptrs, sizes = pool.get_page_buffer_meta(indices)
        self._validate_metadata(ptrs, sizes, len(keys), self._pool_ranges[name])
        if list(sizes) != self._component_sizes[name] * len(keys):
            raise ValueError("GCache host page byte lengths changed after registration")
        return ptrs, sizes

    def _sdk_call(self, operation: str, keys: list[str], *args, **kwargs):
        try:
            result = getattr(self.gcache_client, f"batch_{operation}")(
                keys, *args, **kwargs
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) != len(keys)
                or any(type(value) is not bool for value in result)
            ):
                raise ValueError(
                    f"GCache {operation} returned an invalid boolean result list"
                )
            return list(result)
        except Exception:
            # Controller workers must receive an explicit failure, not die or
            # publish pages whose bytes were only partially initialized.
            logger.exception("GCache %s failed for %d objects", operation, len(keys))
            return None

    @staticmethod
    def _page_results(components: list[bool]) -> list[bool]:
        return [
            components[i] and components[i + 1] for i in range(0, len(components), 2)
        ]

    @staticmethod
    def _prefix_length(pages: list[bool]) -> int:
        return pages.index(False) if False in pages else len(pages)

    def _pool_exists(self, name: PoolName, keys: list[str]) -> list[bool]:
        mask = self._sdk_call(
            "exists",
            self.key_codec.component_keys(name, keys),
            for_read=True,
            size_list=self._component_sizes[name] * len(keys),
        )
        return self._page_results(mask) if mask is not None else [False] * len(keys)

    def batch_exists(
        self, keys: list[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if not keys:
            return 0
        try:
            with self._operation():
                return self._prefix_length(self._pool_exists(PoolName.KV, keys))
        except Exception:
            logger.exception("GCache primary KV existence query failed")
            return 0

    def _batch_io(
        self,
        name: PoolName,
        keys: list[str],
        host_indices: torch.Tensor,
        *,
        is_set: bool,
    ) -> list[bool]:
        if not keys:
            return []
        started = time.perf_counter()
        prepared = transferred = None
        pages = [False] * len(keys)
        transferred_bytes = 0
        try:
            with self._operation():
                ptrs, sizes = self._page_metadata(name, keys, host_indices)
                object_keys = self.key_codec.component_keys(name, keys)
                prepared = time.perf_counter()
                if is_set:
                    components = self._sdk_call(
                        "exists", object_keys, for_read=False, size_list=sizes
                    )
                    missing, written = [], None
                    if components is not None:
                        missing = [
                            i for i, exists in enumerate(components) if not exists
                        ]
                        if missing:
                            written = self._sdk_call(
                                "set",
                                [object_keys[i] for i in missing],
                                [ptrs[i] for i in missing],
                                [sizes[i] for i in missing],
                            )
                    transferred = time.perf_counter()
                    if written is not None:
                        for i, ok in zip(missing, written):
                            components[i] = ok
                            if ok:
                                transferred_bytes += sizes[i]
                else:
                    components = self._sdk_call("get", object_keys, ptrs, sizes)
                    transferred = time.perf_counter()
                    if components is not None:
                        transferred_bytes = sum(
                            size for size, ok in zip(sizes, components) if ok
                        )
                if components is not None:
                    pages = self._page_results(components)
        except Exception:
            logger.exception("GCache %s %s failed", name, "write" if is_set else "read")

        if self.enable_storage_metrics:
            finished = time.perf_counter()
            prepared = prepared if prepared is not None else finished
            transferred = transferred if transferred is not None else finished
            elapsed = max(finished - started, 1e-9)
            bandwidth = transferred_bytes / (1 << 30) / elapsed
            with self._metrics_lock:
                if is_set:
                    self._metrics.backup_pgs.append(sum(pages))
                    self._metrics.backup_bandwidth.append(bandwidth)
                else:
                    self._metrics.prefetch_pgs.append(sum(pages))
                    self._metrics.prefetch_bandwidth.append(bandwidth)
            self._gcache_metrics.observe(
                is_set=is_set,
                pages=len(keys),
                transferred_bytes=transferred_bytes,
                times=(started, prepared, transferred, finished),
            )
        return pages

    def batch_get_v1(self, keys, host_indices, extra_info=None) -> list[bool]:
        return self._batch_io(PoolName.KV, keys, host_indices, is_set=False)

    def batch_set_v1(self, keys, host_indices, extra_info=None) -> list[bool]:
        return self._batch_io(PoolName.KV, keys, host_indices, is_set=True)

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    def batch_exists_v2(
        self, keys, pool_transfers=None, extra_info=None
    ) -> PoolTransferResult:
        if not keys:
            return PoolTransferResult(0, {}, [])
        try:
            with self._operation():
                kv_pages = self._prefix_length(self._pool_exists(PoolName.KV, keys))
                hit_count = {PoolName.KV: kv_pages} if kv_pages else {}
                restorable = set(range(1, kv_pages + 1))
                for transfer in pool_transfers or []:
                    if not restorable:
                        break
                    # Prefetch transfers may contain window-length placeholders.
                    # The primary hash list always defines the candidate pages.
                    page_exists = self._pool_exists(transfer.name, keys[:kv_pages])
                    if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                        boundary = self._prefix_length(page_exists)
                        pool_restorable = set(range(1, boundary + 1))
                    elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                        trailing = max(1, len(transfer.keys or []))
                        consecutive = 0
                        pool_restorable = set()
                        for end, exists in enumerate(page_exists, start=1):
                            consecutive = consecutive + 1 if exists else 0
                            if consecutive >= min(end, trailing):
                                pool_restorable.add(end)
                    else:
                        raise ValueError(
                            f"Unsupported GCache hit policy: {transfer.hit_policy}"
                        )
                    if pool_restorable:
                        hit_count[transfer.name] = max(pool_restorable)
                    restorable.intersection_update(pool_restorable)
                endpoints = sorted(restorable)
                return PoolTransferResult(
                    endpoints[-1] if endpoints else 0, hit_count, endpoints
                )
        except Exception:
            logger.exception("GCache multi-pool existence query failed")
            return PoolTransferResult(0, {}, [])

    def _batch_io_v2(self, transfers: list[PoolTransfer], *, is_set: bool):
        return {
            transfer.name: self._batch_io(
                transfer.name, transfer.keys or [], transfer.host_indices, is_set=is_set
            )
            for transfer in transfers
        }

    def batch_get_v2(
        self, transfers: list[PoolTransfer], extra_info=None
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self, transfers: list[PoolTransfer], extra_info=None
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, is_set=True)

    def get(self, key: str, target_location: Optional[Any] = None, target_sizes=None):
        raise NotImplementedError("GCache requires the zero-copy batch_get_v1 API")

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError("GCache requires the zero-copy batch_get_v1 API")

    def set(self, key: str, value=None, target_location=None, target_sizes=None):
        raise NotImplementedError("GCache requires the zero-copy batch_set_v1 API")

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise NotImplementedError("GCache requires the zero-copy batch_set_v1 API")

    def get_stats(self) -> StorageMetrics:
        with self._metrics_lock:
            result = self._metrics
            self._metrics = StorageMetrics()
        return result
