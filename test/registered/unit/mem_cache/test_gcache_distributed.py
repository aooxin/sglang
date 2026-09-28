"""Two real Gloo ranks querying one network KV server, without GPU kernels."""

import os
import sys
import threading
from array import array
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_mock_server import GCacheMockServer, NetworkGCacheClient
from sglang.test.gcache_test_utils import (
    fill_pool,
    make_mha_pool,
    make_storage_config,
    page_bytes,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _run_rank(rank, rendezvous, address, config_dir, scenario):
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
        StorageExistenceCache,
    )
    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
        PrefetchOperation,
    )
    from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.storage.gcache import gcache_storage
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    backend = None
    try:
        # Only the external SDK transport and process configuration are supplied
        # by the harness. Storage, collectives, ACK reduction and frees are real.
        sys.modules["gcache_sdk"] = SimpleNamespace(GCacheClient=NetworkGCacheClient)
        gcache_storage.get_attention_dp_rank = lambda: 0
        os.environ["GCACHE_MASTER_ADDR"] = address
        os.environ["GCACHE_CACHE_GROUP"] = "test"
        os.environ["HPN_IB_IFNAME"] = "lo"
        os.environ.pop("GCACHE_CONFIG_FILE", None)
        full = make_mha_pool(size=16)
        swa = make_mha_pool(size=16, asymmetric=True, layers=1)
        config = make_storage_config(
            full,
            Path(config_dir) / str(rank),
            tp_rank=rank,
            enable_storage_metrics=False,
        )
        backend = gcache_storage.GcacheStorage(config, full)
        backend.register_mem_pool_host(full)
        backend.register_mem_host_pool_v2(swa, PoolName.SWA)
        fill_pool(full, seed=rank * 31)
        fill_pool(swa, seed=91 + rank * 17)
        tokens = list(range(12))
        hashes = get_storage_hash_str(tokens, None, page_size=2)
        assert backend.batch_set_v1(hashes, torch.arange(12)) == [True] * 6
        mask = [1, 1, 0, 1, 1, 1] if rank == 0 else [1, 1, 1, 1, 0, 0]
        selected = [i for i, exists in enumerate(mask) if exists]
        assert backend.batch_set_v2(
            [
                PoolTransfer(
                    PoolName.SWA,
                    torch.tensor([j for i in selected for j in (2 * i, 2 * i + 1)]),
                    keys=[hashes[i] for i in selected],
                )
            ]
        )[PoolName.SWA] == [True] * len(selected)
        expected_full = [page_bytes(full, i) for i in (0, 2)]
        expected_swa = [page_bytes(swa, i) for i in (0, 2)]
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = 2
        cc.storage_backend = backend
        cc.storage_backend_type = "gcache"
        cc.prefetch_hits_sync_groups = [torch.distributed.group.WORLD]
        cc.prefetch_completion_sync_groups = [torch.distributed.group.WORLD]
        cc.storage_stop_event = threading.Event()
        cc.prefetch_queue = Queue()
        cc.prefetch_hit_queue = Queue()
        cc.prefetch_sync_queue = Queue()
        cc.host_mem_release_queue = Queue()
        cc.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(PoolName.KV, full, None, lambda layer: layer, True),
                PoolEntry(PoolName.SWA, swa, None, lambda layer: layer),
            ]
        )
        cc._init_extra_host_mem_release_queues()
        cc.page_get_func = cc._page_get_zero_copy
        transfer = PoolTransfer(
            PoolName.SWA, keys=["window"] * 2, hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        operation = PrefetchOperation(
            CacheRequestHandle("distributed", 0), tokens, pool_transfers=[transfer]
        )
        if scenario == "cancel" and rank == 1:
            operation.mark_terminate()
        cc.prefetch_queue.put(operation)
        worker = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
        worker.start()
        try:
            received = cc.prefetch_hit_queue.get(timeout=15)
            assert received is operation
        finally:
            cc.storage_stop_event.set()
            cc.prefetch_queue.put(None)
            worker.join(timeout=5)
        assert not worker.is_alive()
        if scenario == "cancel":
            assert operation.storage_hit_count == 0
            return
        # Rank maxima are 6 and 4, but rank 0 cannot restore endpoint 4.
        assert operation.storage_hit_count == 4
        assert operation.hash_value == hashes[:2]

        operation.host_indices = cc.mem_pool_host.alloc(4)
        transfer.host_indices = cc.mem_pool_host.alloc(4, pool=PoolName.SWA)
        cc.prefetch_tokens_occupied = 4
        for pool in (full, swa):
            pool.k_buffer.zero_()
            pool.v_buffer.zero_()
        if rank == 1 and scenario in ("swa_failure", "full_failure"):
            kind = "swa" if scenario == "swa_failure" else "full"
            backend.gcache_client._call(
                "control",
                action="configure",
                operation="get",
                fault={"fail_keys": [f"model_{hashes[0]}_1_{kind}_v"]},
            )
        torch.distributed.barrier()
        cc._page_transfer(operation)
        # One primary batch ACK and one sidecar ACK on every rank, even when
        # rank 1 could not read the primary pool and skipped sidecar I/O.
        assert cc.prefetch_sync_queue.qsize() == 2
        while not cc.prefetch_sync_queue.empty():
            ack = cc.prefetch_sync_queue.get_nowait()
            cc._reduce_prefetch_ack(ack)
            if ack.completed_tokens is not None:
                operation.completed_tokens = ack.completed_tokens
            if ack.pool_hits is not None:
                operation.pool_storage_result.update_extra_pool_hit_pages(ack.pool_hits)
                operation.pool_transfers_done = True

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=2)
        cache.host_memory_mode = "cache"
        cache.cache_controller = cc
        cache.storage_existence_cache = StorageExistenceCache()
        cache.storage_existence_cache.add(PoolName.KV, operation.hash_value)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.buffer_pipeline = None
        cache.ongoing_prefetch = {operation.handle: operation}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        accepted = cache._check_hybrid_prefetch_result(
            operation.handle,
            operation,
            operation.completed_tokens,
            operation.hash_value,
            operation.host_indices,
            0,
            None,
            RadixKey(array("q", tokens[:4])),
        )
        assert accepted == (scenario == "restore")
        if accepted:
            assert [page_bytes(full, i) for i in (0, 2)] == expected_full
            assert [page_bytes(swa, i) for i in (0, 2)] == expected_swa
            cc.append_host_mem_release(operation.host_indices, [transfer])
        else:
            assert cache.prefetch_loaded_tokens_by_reqid[operation.handle] == 0
            assert operation.handle not in cache.ongoing_prefetch
            assert cc.prefetch_tokens_occupied == 0
            assert not cache.storage_existence_cache.contains_all(
                PoolName.KV, hashes[:2]
            )
            # The completed-request ACK owns the unfinished primary tail.
            cc.append_host_mem_release(
                operation.host_indices[operation.completed_tokens :]
            )
        while not cc.host_mem_release_queue.empty():
            cc.mem_pool_host.free(cc.host_mem_release_queue.get_nowait())
        for name, queue in cc.extra_host_mem_release_queues.items():
            while not queue.empty():
                cc.mem_pool_host.free(queue.get_nowait(), pool=name)
        assert full.available_size() == full.size
        assert swa.available_size() == swa.size
        assert not full.slot_used.any() and not swa.slot_used.any()
    finally:
        if backend is not None:
            backend.close()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["restore", "swa_failure", "full_failure", "cancel"]
)
def test_two_rank_prefetch_and_failure_reclamation(tmp_path, scenario):
    with GCacheMockServer() as server:
        torch.multiprocessing.spawn(
            _run_rank,
            args=(
                str(tmp_path / "rendezvous"),
                server.address,
                str(tmp_path),
                scenario,
            ),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
