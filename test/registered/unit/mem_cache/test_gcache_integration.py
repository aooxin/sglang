"""CPU integration of the GCache backend with the 0520 HiCache entry points."""

import argparse
import builtins
import importlib
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory
from sglang.test.gcache_test_utils import (
    fill_pool,
    make_mha_pool,
    make_storage_config,
    page_bytes,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def test_cli_accepts_gcache():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model-path", "dummy", "--hicache-storage-backend", "gcache"]
    )
    assert args.hicache_storage_backend == "gcache"


@pytest.mark.parametrize(
    "io_backend,want", [("kernel", "page_first"), ("direct", "page_first_direct")]
)
def test_gcache_layout_resolution_uses_declarations(io_backend, want):
    from sglang.srt.arg_groups.hicache_hook import resolve_storage_layout_compatibility
    from sglang.srt.arg_groups.overrides import resolving_view

    args = SimpleNamespace(
        hicache_storage_backend="gcache",
        hicache_mem_layout="layer_first",
        hicache_io_backend=io_backend,
    )
    resolve_storage_layout_compatibility(args)
    assert resolving_view(args).hicache_mem_layout == want
    assert args.hicache_mem_layout == "layer_first"
    assert args.hicache_io_backend == io_backend


@pytest.mark.parametrize(
    "overrides",
    [
        {"pp_size": 2},
        {"attn_cp_size": 2},
        {"hicache_mem_layout": "page_head"},
        {"hicache_io_backend": "kernel_ascend"},
    ],
)
def test_gcache_unsupported_launch_combinations_fail_early(overrides):
    from sglang.srt.arg_groups.hicache_hook import handle_hicache

    fields = dict(
        enable_unified_cache_external_linker=False,
        enable_hierarchical_cache=True,
        disaggregation_mode="null",
        hicache_host_memory_mode="cache",
        hicache_size=0,
        hicache_ratio=2,
        hicache_storage_backend="gcache",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
    )
    fields.update(overrides)
    with pytest.raises((ValueError, NotImplementedError), match="GCache|gcache"):
        handle_hicache(SimpleNamespace(**fields))


def test_factory_creates_gcache_backend(backend_factory):
    factory = backend_factory
    backend = factory(via_factory=True)
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    assert backend.batch_exists(["h0"]) == 1


def test_other_backend_does_not_import_gcache_sdk(monkeypatch, tmp_path):
    from sglang.srt.mem_cache.storage import backend_factory

    attempted = []
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "gcache_sdk" or name.startswith("gcache_sdk."):
            attempted.append(name)
            raise ModuleNotFoundError("SDK intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setitem(sys.modules, "gcache_sdk", None)
    # Re-run the registration module, then perform real file-backend I/O.
    importlib.reload(backend_factory)
    (tmp_path / "file").mkdir()
    monkeypatch.setenv(
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", str(tmp_path / "file")
    )
    config = make_storage_config(SimpleNamespace(layout="page_first"), tmp_path)
    backend = backend_factory.StorageBackendFactory.create_backend("file", config, None)
    value = torch.arange(8, dtype=torch.uint8)
    assert backend.set("h0", value)
    assert torch.equal(
        backend.get("h0", target_location=torch.empty_like(value)), value
    )
    assert attempted == []


@pytest.mark.parametrize("startup", [True, False])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_unified_plain_kv_strategy_and_controller_v1_dispatch(
    monkeypatch, tmp_path, backend_factory, startup, asymmetric
):
    from sglang.srt.managers import cache_controller
    from sglang.srt.mem_cache import l2_transfer
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    factory = backend_factory
    pool = make_mha_pool(asymmetric=asymmetric)
    fill_pool(pool)
    device_pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    device_pool.layer_num = pool.layer_num
    device_pool.device = "cpu"
    allocator = SimpleNamespace(get_kvcache=lambda: device_pool)
    parallel = SimpleNamespace(
        tp_rank=5, tp_size=8, attn_tp_rank=1, attn_tp_size=2, pp_rank=0, pp_size=1
    )
    memory = SimpleNamespace(
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_host_memory_mode="cache",
    )
    monkeypatch.setattr(cache_controller, "get_parallel", lambda: parallel)
    monkeypatch.setattr(cache_controller, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(cache_controller, "get_attention_dp_rank", lambda: 2)
    monkeypatch.setattr(assembler, "get_memory", lambda: memory)
    monkeypatch.setattr(assembler, "build_kv_host_pool", lambda **kwargs: pool)
    # Only hardware execution and background scheduling are substituted. The
    # strategy, group, controller init/attach, factory, and page I/O are real.
    device_events = SimpleNamespace(Event=object, Stream=object)
    monkeypatch.setattr(cache_controller, "device_module", device_events)
    monkeypatch.setattr(l2_transfer, "device_module", device_events)
    monkeypatch.setattr(HybridCacheController, "_create_sync_groups", lambda self: [])
    monkeypatch.setattr(
        HybridCacheController, "_start_storage_threads", lambda self: None
    )
    params = SimpleNamespace(
        page_size=pool.page_size,
        token_to_kv_pool_allocator=allocator,
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
        mtp_draft_device_pools=(),
    )
    strategy = assembler._select_strategy(device_pool, {ComponentType.FULL})
    config = make_storage_config(pool, tmp_path)
    stack = strategy.build(
        cache=SimpleNamespace(),
        kvcache=device_pool,
        params=params,
        server_args=SimpleNamespace(),
        load_cache_event=threading.Event(),
        storage_backend="gcache" if startup else None,
        storage_backend_extra_config=config.extra_config,
        model_name="model",
    )
    controller = stack.cache_controller
    if not startup:
        controller.attach_storage_backend(
            "gcache",
            model_name="model",
            storage_backend_extra_config=config.extra_config,
            host_pools=stack.host_pool_group.entries,
        )
    backend = controller.storage_backend
    try:
        assert controller.storage_config.tp_rank == 1
        assert controller.storage_config.tp_size == 2
        assert backend.registered_pools[PoolName.KV] is pool
        assert len(factory.clients) == 1
        assert len(backend.gcache_client.registrations) == (2 if asymmetric else 1)
        assert (
            controller.page_get_func.__func__
            is cache_controller.HiCacheController._page_get_zero_copy
        )
        assert (
            controller.page_set_func.__func__
            is cache_controller.HiCacheController._page_set_zero_copy
        )
        expected = page_bytes(pool, 4)
        assert controller.page_set_func(["h0"], torch.tensor([4, 5])) is True
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        assert (
            controller.page_get_func(
                SimpleNamespace(request_id="test"), ["h0"], torch.tensor([2, 3])
            )
            == 1
        )
        assert page_bytes(pool, 2) == expected
        assert set(factory.server.objects) == {"model_h0_1_full_k", "model_h0_1_full_v"}
    finally:
        controller.detach_storage_backend()
    # The pinned SDK close() is global fini(), so detach must not call it.
    assert backend.gcache_client.closed is False


def _cpu_device_pool(*, layers=2, asymmetric=False):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.size = 8
    pool.layer_num = layers
    pool.start_layer, pool.end_layer = 0, layers
    pool.head_num, pool.head_dim = 2, 4
    pool.v_head_dim = 6 if asymmetric else 4
    pool.dtype = pool.store_dtype = torch.float16
    pool.device = "cpu"
    return pool


def _cpu_swa_stack(monkeypatch, tmp_path, *, startup):
    from sglang.srt.managers import cache_controller
    from sglang.srt.mem_cache import l2_transfer
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host import common
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    parallel = SimpleNamespace(
        tp_rank=1,
        tp_size=2,
        attn_tp_rank=1,
        attn_tp_size=2,
        pp_rank=0,
        pp_size=1,
        dcp_enabled=False,
    )
    memory = SimpleNamespace(
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_host_memory_mode="cache",
        hicache_mem_layout="page_first",
        hicache_ratio=2,
        hicache_size=0,
        hicache_storage_backend="gcache",
    )
    monkeypatch.setattr(cache_controller, "get_parallel", lambda: parallel)
    monkeypatch.setattr(assembler, "get_parallel", lambda: parallel)
    monkeypatch.setattr(cache_controller, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(assembler, "get_memory", lambda: memory)
    monkeypatch.setattr(common, "get_memory", lambda: memory)
    events = SimpleNamespace(Event=object, Stream=object)
    monkeypatch.setattr(cache_controller, "device_module", events)
    monkeypatch.setattr(l2_transfer, "device_module", events)
    monkeypatch.setattr(HybridCacheController, "_create_sync_groups", lambda self: [])
    monkeypatch.setitem(
        common.ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, *, dtype, **kwargs: torch.zeros(dims, dtype=dtype),
    )
    target = SWAKVPool.__new__(SWAKVPool)
    target.full_kv_pool = _cpu_device_pool()
    target.swa_kv_pool = _cpu_device_pool(asymmetric=True)
    target.layers_mapping = {0: (0, False), 1: (0, True), 2: (1, False), 3: (1, True)}
    target.layer_num, target.device = 4, "cpu"
    allocator = SimpleNamespace(get_kvcache=lambda: target, swa_attn_allocator=None)
    params = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=allocator,
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
        mtp_draft_device_pools=(),
    )
    config = make_storage_config(SimpleNamespace(layout="page_first"), tmp_path)
    strategy = assembler._select_strategy(
        target, {ComponentType.FULL, ComponentType.SWA}
    )
    stack = strategy.build(
        cache=SimpleNamespace(),
        kvcache=target,
        params=params,
        server_args=SimpleNamespace(),
        load_cache_event=threading.Event(),
        storage_backend="gcache" if startup else None,
        storage_backend_extra_config=config.extra_config,
        model_name="model",
    )
    return stack, config.extra_config


@pytest.mark.parametrize("startup", [True, False])
@pytest.mark.parametrize("swa_draft", [False, True])
def test_swa_and_draft_assembler_with_storage_worker_io(
    backend_factory, monkeypatch, tmp_path, startup, swa_draft
):
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    stack, extra = _cpu_swa_stack(monkeypatch, tmp_path, startup=startup)
    cc = stack.cache_controller
    try:
        if not startup:
            cc.attach_storage_backend(
                "gcache",
                model_name="model",
                storage_backend_extra_config=extra,
                host_pools=stack.host_pool_group.entries,
            )
        draft_device = _cpu_device_pool(layers=1, asymmetric=True)
        if swa_draft:
            draft = SWAKVPool.__new__(SWAKVPool)
            draft.swa_kv_pool, draft.full_kv_pool = draft_device, None
        else:
            draft = draft_device
        specs, entries = assembler.build_hicache_draft_sidecars(
            draft_device_pools=(draft,), tree_cache=SimpleNamespace(cache_controller=cc)
        )
        name = PoolName.DRAFT_SWA if swa_draft else PoolName.DRAFT
        source = PoolName.SWA if swa_draft else PoolName.KV
        assert len(entries) == len(specs) == 1
        assert specs[0].pool_name == name and specs[0].indices_from_pool == source
        cc.register_host_pool_entry(entries[0])
        assert set(cc.storage_backend.registered_pools) == {
            PoolName.KV,
            PoolName.SWA,
            name,
        }
        pools = cc.storage_backend.registered_pools
        for seed, pool in enumerate(pools.values()):
            fill_pool(pool, seed=seed * 83)
        expected = {
            PoolName.KV: [page_bytes(pools[PoolName.KV], i) for i in (2, 6)],
            PoolName.SWA: [page_bytes(pools[PoolName.SWA], i) for i in (8, 0)],
            name: [
                page_bytes(pools[name], i) for i in ((8, 0) if swa_draft else (2, 6))
            ],
        }
        keys = ["h0", "h1"]

        def transfers(swa_indices, *, writing):
            return [
                PoolTransfer(
                    PoolName.SWA,
                    swa_indices,
                    keys=keys if writing else ["window"] * 2,
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
                PoolTransfer(
                    name,
                    keys=None if writing else ["window"] * 2,
                    indices_from_pool=source,
                    hit_policy=specs[0].hit_policy,
                ),
            ]

        operation_id = cc.write_storage(
            torch.tensor([2, 3, 6, 7]),
            [],
            hash_value=keys,
            extra_pools=transfers(torch.tensor([8, 9, 0, 1]), writing=True),
        )
        written = cc.ack_backup_queue.get(timeout=5)
        assert written.id == operation_id and written.completed_tokens == 4
        for pool in pools.values():
            pool.k_buffer.zero_()
            pool.v_buffer.zero_()
        read = PrefetchOperation(
            CacheRequestHandle("test", 0),
            [],
            pool_transfers=transfers(torch.tensor([2, 3, 6, 7]), writing=False),
        )
        read.hash_value, read.host_indices = keys, torch.tensor([4, 5, 0, 1])
        cc.prefetch_buffer.put(read)
        acks = [cc.ack_prefetch_queue.get(timeout=5) for _ in range(3)]
        assert acks[-1].completed_req
        assert acks[0].completed_tokens == 4
        assert acks[1].pool_hits[PoolName.SWA] == 2
        assert [page_bytes(pools[PoolName.KV], i) for i in (4, 0)] == expected[
            PoolName.KV
        ]
        assert [page_bytes(pools[PoolName.SWA], i) for i in (2, 6)] == expected[
            PoolName.SWA
        ]
        assert [
            page_bytes(pools[name], i) for i in ((2, 6) if swa_draft else (4, 0))
        ] == expected[name]
        if swa_draft:
            assert not any(
                key.startswith("d:") and "_full_" in key
                for key in backend_factory.server.objects
            )
    finally:
        cc.detach_storage_backend()


def test_attach_rolls_back_if_side_pool_registration_fails(
    backend_factory, monkeypatch, tmp_path
):
    stack, extra = _cpu_swa_stack(monkeypatch, tmp_path, startup=False)
    cc = stack.cache_controller
    backend_factory.server.configure("register", fail_after=1)
    try:
        with pytest.raises(RuntimeError, match="registration failure"):
            cc.attach_storage_backend(
                "gcache",
                model_name="model",
                storage_backend_extra_config=extra,
                host_pools=stack.host_pool_group.entries,
            )
        assert not cc.enable_storage and cc.storage_backend is None
        assert all(
            not thread.is_alive()
            for thread in (
                cc.prefetch_thread,
                cc.prefetch_io_aux_thread,
                cc.prefetch_sync_thread,
                cc.backup_thread,
            )
        )
        backend_factory.server.configure("register")
        cc.attach_storage_backend(
            "gcache",
            model_name="model",
            storage_backend_extra_config=extra,
            host_pools=stack.host_pool_group.entries,
        )
        assert cc.enable_storage
        assert len(backend_factory.clients[0].registrations) == 3
    finally:
        cc.detach_storage_backend()


def test_dynamic_registration_failure_does_not_publish_pool(
    backend_factory, monkeypatch, tmp_path
):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.pool_host import PoolEntry

    stack, _ = _cpu_swa_stack(monkeypatch, tmp_path, startup=True)
    cc = stack.cache_controller
    entry = PoolEntry(
        PoolName.DRAFT, make_mha_pool(layers=1), None, lambda layer: layer
    )
    backend_factory.server.configure("register", fail_after=3)
    try:
        with pytest.raises(RuntimeError, match="registration failure"):
            cc.register_host_pool_entry(entry)
        assert PoolName.DRAFT not in cc.mem_pool_host.entry_map
        assert PoolName.DRAFT not in cc.extra_host_mem_release_queues
        backend_factory.server.configure("register")
        cc.register_host_pool_entry(entry)
        assert cc.mem_pool_host.entry_map[PoolName.DRAFT] is entry
    finally:
        cc.detach_storage_backend()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
