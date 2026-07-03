import logging
import os
import time
from typing import Any, List, Optional

import torch
from gcache_sdk import GCacheClient

from sglang.srt.layers.dp_attention import get_attention_dp_rank
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, SWATokenToKVPoolHost
from sglang.srt.observability.metrics_collector import StorageMetrics

logger = logging.getLogger(__name__)

GCACHE_CONFIG_FILE = "/sgl-workspace/gcache.flags"


class GcacheStorage(HiCacheStorage):
    def __init__(
        self, hicache_config: HiCacheStorageConfig, mem_pool_host: HostKVCache
    ):
        self.dp_rank = get_attention_dp_rank()
        self.tp_rank = hicache_config.tp_rank
        # self.tp_size = hicache_config.tp_size
        self.model_name: str = hicache_config.model_name
        assert self.model_name, "Empty model name"

        # Cache the constant key prefix/suffixes so the per-page hot loop only
        # concatenates the variable middle part instead of formatting a full
        # f-string for every key.
        self._key_prefix = f"{self.model_name}_"
        self._key_suffix_full_k = f"_{self.tp_rank}_full_k"
        self._key_suffix_full_v = f"_{self.tp_rank}_full_v"
        self._key_suffix_swa_k = f"_{self.tp_rank}_swa_k"
        self._key_suffix_swa_v = f"_{self.tp_rank}_swa_v"

        # TODO(yingchun): move to python/sglang/srt/environ.py
        master_addr = os.environ.get("GCACHE_MASTER_ADDR")
        cache_group = os.environ.get("GCACHE_CACHE_GROUP")
        if not master_addr or not cache_group:
            raise RuntimeError(
                "Gcache need env GCACHE_MASTER_ADDR and GCACHE_CACHE_GROUP"
            )

        client_threads = os.environ.get("GCACHE_CLIENT_THREADS")
        if not client_threads:
            client_threads = "8"

        traffic_class = os.environ.get("GCACHE_TRAFFIC_CLASS")
        if not traffic_class:
            traffic_class = "-1"

        hpn_ifname = os.environ.get("HPN_IB_IFNAME")
        obsolete_ifname = os.environ.get("GCACHE_CLIENT_IFNAME")
        if hpn_ifname:
            client_ifname = hpn_ifname
        elif obsolete_ifname:
            client_ifname = obsolete_ifname
        else:
            raise RuntimeError(
                "Gcache need env HPN_IB_IFNAME or GCACHE_CLIENT_IFNAME"
            )

        # In case there is only one NIC
        if obsolete_ifname and len(obsolete_ifname.split(",")) == 1:
            client_ifname = obsolete_ifname

        # # Each client uses one NIC
        # nic_names = client_ifname.split(",")
        # nic_index = (self.dp_rank * self.tp_size + self.tp_rank) % len(nic_names)
        # selected_ifname = nic_names[nic_index]

        subprocess_config = GCACHE_CONFIG_FILE + f".dp{self.dp_rank}tp{self.tp_rank}"
        self.generate_config_file(
            GCACHE_CONFIG_FILE,
            subprocess_config,
            master_addr,
            cache_group,
            client_ifname,
            client_threads,
            traffic_class,
        )

        self.gcache_client = GCacheClient(conf_file=subprocess_config)

        self.enable_storage_metrics = hicache_config.enable_storage_metrics
        self.gb_per_page = None
        self.prefetch_pgs: list = []
        self.backup_pgs: list = []
        self.prefetch_bandwidth: list = []
        self.backup_bandwidth: list = []
        self.mem_pool_host = mem_pool_host
        self._last_batch_get_end_ts: float | None = None

        if self.enable_storage_metrics:
            from prometheus_client import Counter, Histogram

            self._metric_labels = {
                "storage_backend": "gcache",
                "tp_rank": self.tp_rank,
                "dp_rank": self.dp_rank,
            }
            self._prefetch_bytes_total = Counter(
                name="sglang:gcache_prefetch_bytes_total",
                documentation="Cumulative bytes fetched from gcache (per rank).",
                labelnames=self._metric_labels.keys(),
            )
            self._backup_bytes_total = Counter(
                name="sglang:gcache_backup_bytes_total",
                documentation="Cumulative bytes written to gcache (per rank).",
                labelnames=self._metric_labels.keys(),
            )
            # Fine-grained timing of batch_get_v1 to locate bandwidth bottleneck.
            _seg_buckets = [
                0.00005,
                0.0001,
                0.0002,
                0.0005,
                0.001,
                0.002,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
            self._prefetch_preprocess_seconds = Histogram(
                name="sglang:gcache_prefetch_preprocess_seconds",
                documentation="Time spent in _batch_preprocess (tolist, ptr/key build).",
                labelnames=self._metric_labels.keys(),
                buckets=_seg_buckets,
            )
            self._prefetch_rdma_seconds = Histogram(
                name="sglang:gcache_prefetch_rdma_seconds",
                documentation="Time spent inside gcache_client.batch_get (pure RDMA + server).",
                labelnames=self._metric_labels.keys(),
                buckets=_seg_buckets,
            )
            self._prefetch_postprocess_seconds = Histogram(
                name="sglang:gcache_prefetch_postprocess_seconds",
                documentation="Time spent in _batch_postprocess.",
                labelnames=self._metric_labels.keys(),
                buckets=_seg_buckets,
            )
            _bw_buckets = [
                0.5,
                1,
                2,
                5,
                10,
                15,
                20,
                25,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
            ]
            self._prefetch_rdma_bandwidth = Histogram(
                name="sglang:gcache_prefetch_rdma_bandwidth",
                documentation="RDMA-only bandwidth in GiB/s (bytes / batch_get duration).",
                labelnames=self._metric_labels.keys(),
                buckets=_bw_buckets,
            )
            _keys_buckets = [1, 4, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1024]
            self._prefetch_batch_keys = Histogram(
                name="sglang:gcache_prefetch_batch_keys",
                documentation="Number of pages (keys) per batch_get_v1 call.",
                labelnames=self._metric_labels.keys(),
                buckets=_keys_buckets,
            )
            _idle_buckets = [
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
            ]
            self._prefetch_idle_seconds = Histogram(
                name="sglang:gcache_prefetch_idle_seconds",
                documentation="Wall-clock gap between consecutive batch_get_v1 calls (NIC idle time).",
                labelnames=self._metric_labels.keys(),
                buckets=_idle_buckets,
            )

        self.warmup()

    def generate_config_file(
        self,
        config_file: str,
        subprocess_config_file: str,
        master_addr: str,
        cache_group: str,
        client_ifname: str,
        client_threads: str,
        traffic_class: str,
    ):
        config_map = {
            "master_addr": master_addr,
            "cache_group": cache_group,
            "client_ifname": client_ifname,
            "client_threads": client_threads,
            "traffic_class": traffic_class,
        }
        new_lines = []
        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                for key, new_val in config_map.items():
                    if line_strip.startswith(f"--{key}="):
                        new_line = f"--{key}={new_val}\n"
                        new_lines.append(new_line)
                        break
                else:
                    new_lines.append(line)
        with open(subprocess_config_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    def register_buffer(self, tensor: torch.Tensor, hint_buffer: str):
        if self.gcache_client is None:
            raise RuntimeError("Gcache store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret_code = self.gcache_client.register_memory(ptr, size)
        if ret_code:
            logger.error(f"Failed to register {hint_buffer}, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register {hint_buffer} to Gcache Store, error code: {ret_code}"
            )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        assert self.mem_pool_host == mem_pool_host
        super().register_mem_pool_host(mem_pool_host)
        assert isinstance(self.mem_pool_host, SWATokenToKVPoolHost)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ], "gcache store storage backend only support page first or page first direct layout"
        try:
            # If k and v head dim are inconsistent, host_k_buffer and host_v_buffer are two separate memory addresses when allocating memory, so they need to be registered separately
            self.register_buffer(self.mem_pool_host.full_pool_host.k_buffer, "full K buffer")
            self.register_buffer(self.mem_pool_host.full_pool_host.v_buffer, "full V buffer")
            self.register_buffer(self.mem_pool_host.swa_pool_host.k_buffer, "SWA K buffer")
            self.register_buffer(self.mem_pool_host.swa_pool_host.v_buffer, "SWA V buffer")

        except TypeError as err:
            logger.error("Failed to register buffer to Gcache Store: %s", err)
            raise TypeError("Gcache Store Register Buffer Error.") from err

        size_per_token = mem_pool_host.get_size_per_token()
        self.gb_per_page = size_per_token * mem_pool_host.page_size / (1 << 30)

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        query_keys, size_list = self._convert_hybrid_keys(keys, with_size=True)
        logger.debug(f"gcache: begin check {len(query_keys)} keys")
        key_multiplier = 4
        assert len(query_keys) == len(keys) * key_multiplier
        exist_mask = self.gcache_client.batch_exists(
            query_keys, for_read=True, size_list=size_list
        )
        assert len(exist_mask) == len(query_keys)

        prefix_success = 0
        for exist in exist_mask:
            if exist:
                prefix_success += 1
            else:
                break

        logger.debug(
            f"gcache: end check {len(query_keys)} keys, {prefix_success} exist"
        )
        prefix_success = prefix_success // key_multiplier
        return prefix_success

    def batch_exists_hybrid(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> tuple[int, list[bool]]:
        query_keys, size_list = self._convert_hybrid_keys(keys, with_size=True)
        logger.debug(f"gcache: begin hybrid check {len(query_keys)} keys")
        key_multiplier = 4
        assert len(query_keys) == len(keys) * key_multiplier
        exist_mask: List[bool] = self.gcache_client.batch_exists(
            query_keys, for_read=True, size_list=size_list
        )
        assert len(exist_mask) == len(query_keys)

        full_prefix_success = len(keys)
        for i in range(full_prefix_success):
            ii = i * key_multiplier
            if not (exist_mask[ii] and exist_mask[ii + 1]):
                full_prefix_success = i
                break

        valid_swa_mask = [
            exist_mask[ii + 2] and exist_mask[ii + 3]
            for ii in range(0, full_prefix_success * key_multiplier, key_multiplier)
        ]

        logger.debug(
            f"gcache: end hybrid check {len(query_keys)} keys, "
            f"{full_prefix_success=}, swa_valid={sum(valid_swa_mask)}/{len(valid_swa_mask)}"
        )
        return full_prefix_success, valid_swa_mask

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if len(keys) == 0:
            return [True] * len(keys)

        start_time = time.perf_counter()
        gcache_keys, value_ptr_list, value_size_list, swa_valid_list = (
            self._batch_preprocess(keys, host_indices)
        )
        logger.debug(f"gcache: begin set {len(gcache_keys)} keys")

        exist_result = self.gcache_client.batch_exists(
            gcache_keys, for_read=False, size_list=value_size_list
        )
        assert len(exist_result) == len(gcache_keys)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [False] * len(gcache_keys)

        for i in range(len(gcache_keys)):
            if not exist_result[i]:
                set_keys.append(gcache_keys[i])
                set_buffer_ptrs.append(value_ptr_list[i])
                set_buffer_sizes.append(value_size_list[i])
                set_indices.append(i)
            else:
                set_results[i] = True

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            t0_set = time.perf_counter()
            partial_set_results = self.gcache_client.batch_set(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            t1_set = time.perf_counter()
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = partial_set_results[i]

            if self.enable_storage_metrics:
                self.backup_pgs.append(len(set_keys))
                total_bytes = sum(set_buffer_sizes)
                self.backup_bandwidth.append(
                    total_bytes / (t1_set - t0_set) / (1 << 30)
                )
                self._backup_bytes_total.labels(**self._metric_labels).inc(total_bytes)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1e6
        success_count = sum(1 for each in set_results if each)

        ret = self._batch_postprocess(set_results, swa_valid_list)
        if len(gcache_keys) == success_count:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"gcache: end set {len(gcache_keys)} keys, success count: {success_count}, API results: {set_results}, "
                    f"ret: {ret}, cost time: {execution_time:.2f} us"
                )
        else:
            logger.error(
                f"gcache: batch set failed: {len(gcache_keys)} keys, success count: {success_count}"
            )

        return ret

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if len(keys) == 0:
            return [True] * len(keys)

        swa_needed_per_page = (
            extra_info.swa_needed_per_page if extra_info is not None else None
        )

        t0 = time.perf_counter()
        gcache_keys, value_ptr_list, value_size_list, swa_valid_list = (
            self._batch_preprocess(keys, host_indices, swa_needed_per_page)
        )
        assert len(gcache_keys) == len(value_ptr_list) == len(value_size_list)
        logger.debug(f"gcache: begin get {len(gcache_keys)} keys")

        t1 = time.perf_counter()
        results = self.gcache_client.batch_get(
            gcache_keys, value_ptr_list, value_size_list
        )
        t2 = time.perf_counter()
        assert len(results) == len(gcache_keys)
        success_count = sum(1 for each in results if each)

        ret = self._batch_postprocess(results, swa_valid_list)
        t3 = time.perf_counter()
        execution_time = (t3 - t0) * 1e6

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            total_bytes = sum(value_size_list)
            self.prefetch_bandwidth.append(total_bytes / (t3 - t0) / (1 << 30))
            labels = self._metric_labels
            self._prefetch_bytes_total.labels(**labels).inc(total_bytes)
            self._prefetch_preprocess_seconds.labels(**labels).observe(t1 - t0)
            self._prefetch_rdma_seconds.labels(**labels).observe(t2 - t1)
            self._prefetch_postprocess_seconds.labels(**labels).observe(t3 - t2)
            rdma_dt = t2 - t1
            if rdma_dt > 0:
                self._prefetch_rdma_bandwidth.labels(**labels).observe(
                    total_bytes / rdma_dt / (1 << 30)
                )
            self._prefetch_batch_keys.labels(**labels).observe(len(keys))
            if self._last_batch_get_end_ts is not None:
                idle = t0 - self._last_batch_get_end_ts
                if idle >= 0:
                    self._prefetch_idle_seconds.labels(**labels).observe(idle)
            self._last_batch_get_end_ts = t3

        logger.debug(
            f"gcache: end get {len(gcache_keys)} keys, success count: {success_count}, "
            f"cost time: {execution_time:.2f} us"
        )

        return ret

    ####################################################################################################################

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("gcache: deprecated single set. SHOULD NOT HAPPEN")

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("gcache: deprecated batch set. SHOULD NOT HAPPEN")

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError("gcache: deprecated single get. SHOULD NOT HAPPEN")

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError("gcache: deprecated batch get. SHOULD NOT HAPPEN")

    def exists(self, key) -> bool:
        raise NotImplementedError("gcache: deprecated single exists. SHOULD NOT HAPPEN")

    ####################################################################################################################

    def warmup(self):
        num_warmup = 2048
        keys = ["gcache_warmup_key" + str(i) for i in range(num_warmup)]
        logger.info("begin warming up gcache client")
        start_time = time.perf_counter()
        # Just check exists, in order to create RDMA connections in advance
        self.batch_exists(keys)
        logger.info(
            f"finish warming up gcache client, cost {time.perf_counter() - start_time:.2f} seconds"
        )

    def _convert_hybrid_keys(
        self,
        keys: List[str],
        swa_valid_list: Optional[List[bool]] = None,
        with_size: bool = False,
    ):
        if swa_valid_list is not None:
            assert len(keys) == len(swa_valid_list)
        prefix = self._key_prefix
        suffix_full_k = self._key_suffix_full_k
        suffix_full_v = self._key_suffix_full_v
        suffix_swa_k = self._key_suffix_swa_k
        suffix_swa_v = self._key_suffix_swa_v
        ret = []
        for i in range(len(keys)):
            mid = prefix + keys[i]
            ret.append(mid + suffix_full_k)
            ret.append(mid + suffix_full_v)
            if swa_valid_list is None or swa_valid_list[i]:
                ret.append(mid + suffix_swa_k)
                ret.append(mid + suffix_swa_v)

        size_list = []
        if with_size:
            full_and_swa = self.mem_pool_host.get_page_element_sizes()
            full_only = full_and_swa[:2]
            if swa_valid_list is None:
                size_list = list(full_and_swa) * len(keys)
            else:
                for valid in swa_valid_list:
                    size_list.extend(full_and_swa if valid else full_only)
        return ret, size_list

    def _batch_preprocess(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        swa_needed_per_page: Optional[list[bool]] = None,
    ):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        assert isinstance(self.mem_pool_host, SWATokenToKVPoolHost)

        ptr_list, element_size_list, swa_valid_list = (
            self.mem_pool_host.get_page_buffer_meta(host_indices, swa_needed_per_page)
        )
        assert len(swa_valid_list) == len(keys)

        key_list, _ = self._convert_hybrid_keys(keys, swa_valid_list)

        assert len(key_list) == len(ptr_list) == len(element_size_list)
        return key_list, ptr_list, element_size_list, swa_valid_list

    def _batch_postprocess(
        self,
        results: List[bool],
        swa_valid_list: List[bool],
    ) -> List[bool]:
        ret = []
        idx = 0
        for swa_valid in swa_valid_list:
            step = 4 if swa_valid else 2
            ret.append(all(results[idx : idx + step]))
            idx += step
        assert idx == len(results)
        return ret

    def get_stats(self):
        # Atomic swap under GIL to avoid race with IO threads appending
        prefetch_pgs, self.prefetch_pgs = self.prefetch_pgs, []
        backup_pgs, self.backup_pgs = self.backup_pgs, []
        prefetch_bandwidth, self.prefetch_bandwidth = self.prefetch_bandwidth, []
        backup_bandwidth, self.backup_bandwidth = self.backup_bandwidth, []

        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs = prefetch_pgs
        storage_metrics.backup_pgs = backup_pgs
        storage_metrics.prefetch_bandwidth = prefetch_bandwidth
        storage_metrics.backup_bandwidth = backup_bandwidth
        return storage_metrics
