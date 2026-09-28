"""Opt-in byte round trips through the real, pinned GCache SDK and service.

Requires GCACHE_MASTER_ADDR, GCACHE_CACHE_GROUP, a client NIC and an installed
SDK. Set SGLANG_TEST_GCACHE=1 explicitly, using a dedicated test cache group.
This checks native host-memory I/O; it does not launch a model or test GPU kernels.
"""

import os
import uuid

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=30,
    suite="base-a-test-cpu",
    disabled="Requires an internal GCache test service and SGLANG_TEST_GCACHE=1.",
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_GCACHE") != "1",
    reason="Native GCache integration is opt-in; CPU unit tests use the local mockserver.",
)


def _host_pool(layout, asymmetric):
    from sglang.srt.mem_cache.pool_host.mha import (
        AsymmetricMHATokenToKVPoolHost,
        MHATokenToKVPoolHost,
    )

    cls = AsymmetricMHATokenToKVPoolHost if asymmetric else MHATokenToKVPoolHost
    pool = cls.__new__(cls)
    pool.layout, pool.dtype, pool.device = layout, torch.float16, "cpu"
    pool.size, pool.page_size, pool.page_num = 128, 16, 8
    pool.layer_num, pool.head_num, pool.head_dim = 2, 2, 8
    pool.v_head_dim = 12 if asymmetric else 8
    prefix = (
        (pool.size, pool.layer_num)
        if layout == "page_first"
        else (pool.page_num, pool.layer_num, pool.page_size)
    )
    k_shape = (*prefix, pool.head_num, pool.head_dim)
    pool.kv_buffer = (
        (
            torch.zeros(k_shape, dtype=pool.dtype),
            torch.zeros((*prefix, pool.head_num, pool.v_head_dim), dtype=pool.dtype),
        )
        if asymmetric
        else torch.zeros((2, *k_shape), dtype=pool.dtype)
    )
    return pool


def _page_bytes(pool, start):
    return [
        (
            buf[start : start + pool.page_size]
            if pool.layout == "page_first"
            else buf[start // pool.page_size]
        )
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
        for buf in (pool.k_buffer, pool.v_buffer)
    ]


@pytest.fixture(scope="module")
def connection_config(tmp_path_factory):
    return {
        "config_dir": str(tmp_path_factory.mktemp("gcache_sdk")),
        "rpc_socket_type": os.environ.get("GCACHE_TEST_RPC_SOCKET_TYPE", "rdma"),
        "timeout_us": 5_000_000,
    }


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_native_full_swa_and_draft_roundtrip(
    monkeypatch, connection_config, layout, asymmetric
):
    from sglang.srt.mem_cache.hicache_storage import (
        HiCacheStorageConfig,
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.storage.gcache import gcache_storage

    # The standalone host test has no SGLang scheduler / attention group.
    # Only that process coordinate is supplied; SDK I/O is entirely native.
    monkeypatch.setattr(gcache_storage, "get_attention_dp_rank", lambda: 0)
    namespace = f"sglang-gcache-smoke-{uuid.uuid4().hex}"
    monkeypatch.setenv("SGLANG_HICACHE_DRAFT_MODEL_NAME", f"d:{namespace}")
    names = [PoolName.KV, PoolName.SWA, PoolName.DRAFT, PoolName.DRAFT_SWA]
    sources = {name: _host_pool(layout, asymmetric) for name in names}
    targets = {name: _host_pool(layout, asymmetric) for name in names}
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=layout == "page_first",
        model_name=namespace,
        extra_config=connection_config,
    )
    writer = gcache_storage.GcacheStorage(config, sources[PoolName.KV])
    reader = gcache_storage.GcacheStorage(config, targets[PoolName.KV])
    keys = ["page0", "page1"]
    written_keys = []
    source_indices = torch.tensor(list(range(64, 80)) + list(range(16)))
    target_indices = torch.tensor(list(range(32, 48)) + list(range(96, 112)))
    try:
        for seed, name in enumerate(names):
            writer.register_mem_host_pool_v2(sources[name], name)
            reader.register_mem_host_pool_v2(targets[name], name)
            for component, buf in enumerate(
                (sources[name].k_buffer, sources[name].v_buffer)
            ):
                raw = buf.view(torch.uint8).flatten()
                raw.copy_(
                    (torch.arange(raw.numel()) + seed * 53 + component * 17).remainder(
                        251
                    )
                )
            written_keys.extend(writer.key_codec.component_keys(name, keys))
        assert writer.batch_set_v1(keys, source_indices) == [True, True]
        transfers = [
            PoolTransfer(
                name,
                source_indices,
                keys=keys,
                hit_policy=PoolHitPolicy.TRAILING_PAGES
                if name in (PoolName.SWA, PoolName.DRAFT_SWA)
                else PoolHitPolicy.ALL_PAGES,
            )
            for name in names[1:]
        ]
        expected_results = {name: [True, True] for name in names[1:]}
        assert writer.batch_set_v2(transfers) == expected_results
        assert reader.batch_exists_v2(keys, transfers).restorable_prefix_pages == [1, 2]
        assert reader.batch_get_v1(keys, target_indices) == [True, True]
        for transfer in transfers:
            transfer.host_indices = target_indices
        assert reader.batch_get_v2(transfers) == expected_results
        for name in names:
            assert [_page_bytes(targets[name], i) for i in (32, 96)] == [
                _page_bytes(sources[name], i) for i in (64, 0)
            ]
    finally:
        try:
            if written_keys:
                writer.gcache_client.batch_delete(written_keys)
        finally:
            reader.close()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
