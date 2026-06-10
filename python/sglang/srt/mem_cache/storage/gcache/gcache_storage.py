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

        # TODO(yingchun): move to python/sglang/srt/environ.py
        master_addr = os.environ.get("GCACHE_MASTER_ADDR")
        cache_group = os.environ.get("GCACHE_CACHE_GROUP")
        if not master_addr or not cache_group:
            raise RuntimeError(
                "Gcache need env GCACHE_MASTER_ADDR and GCACHE_CACHE_GROUP"
            )

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
        )

        self.gcache_client = GCacheClient(conf_file=subprocess_config)

        self.enable_storage_metrics = hicache_config.enable_storage_metrics
        self.gb_per_page = None
        self.prefetch_pgs: list = []
        self.backup_pgs: list = []
        self.prefetch_bandwidth: list = []
        self.backup_bandwidth: list = []
        self.mem_pool_host = mem_pool_host

        self.warmup()

    def generate_config_file(
        self,
        config_file,
        subprocess_config_file,
        master_addr,
        cache_group,
        client_ifname,
    ):
        config_map = {
            "master_addr": master_addr,
            "cache_group": cache_group,
            "client_ifname": client_ifname,
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

            if self.enable_storage_metrics and self.gb_per_page is not None:
                self.backup_pgs.append(len(set_keys))
                self.backup_bandwidth.append(
                    len(set_keys) / (t1_set - t0_set) * self.gb_per_page
                )

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

        start_time = time.perf_counter()
        gcache_keys, value_ptr_list, value_size_list, swa_valid_list = (
            self._batch_preprocess(keys, host_indices, swa_needed_per_page)
        )
        assert len(gcache_keys) == len(value_ptr_list) == len(value_size_list)
        logger.debug(f"gcache: begin get {len(gcache_keys)} keys")

        results = self.gcache_client.batch_get(
            gcache_keys, value_ptr_list, value_size_list
        )
        assert len(results) == len(gcache_keys)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1e6
        success_count = sum(1 for each in results if each)

        if self.enable_storage_metrics and self.gb_per_page is not None:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        logger.debug(
            f"gcache: end get {len(gcache_keys)} keys, success count: {success_count}, "
            f"cost time: {execution_time:.2f} us"
        )

        return self._batch_postprocess(results, swa_valid_list)

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
        ret = []
        for i in range(len(keys)):
            key_ = keys[i]
            ret.append(f"{self.model_name}_{key_}_{self.tp_rank}_full_k")
            ret.append(f"{self.model_name}_{key_}_{self.tp_rank}_full_v")
            if swa_valid_list is None or swa_valid_list[i]:
                ret.append(f"{self.model_name}_{key_}_{self.tp_rank}_swa_k")
                ret.append(f"{self.model_name}_{key_}_{self.tp_rank}_swa_v")

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
