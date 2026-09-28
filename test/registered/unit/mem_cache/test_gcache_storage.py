"""GCache primary KV contract using real host metadata and actual CPU bytes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory
from sglang.test.gcache_test_utils import fill_pool, make_mha_pool, page_bytes

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.uint8])
def test_noncontiguous_page_roundtrip(backend_factory, layout, asymmetric, dtype):
    source = make_mha_pool(layout, asymmetric, dtype)
    dest = make_mha_pool(layout, asymmetric, dtype)
    fill_pool(source)
    writer = backend_factory(source)
    reader = backend_factory(dest)
    expected = [page_bytes(source, 4), page_bytes(source, 0)]

    assert writer.batch_set_v1(["h0", "h1"], torch.tensor([4, 5, 0, 1])) == [True, True]
    assert reader.batch_exists(["h0", "h1"]) == 2
    assert reader.batch_get_v1(["h0", "h1"], torch.tensor([2, 3, 6, 7])) == [True, True]
    assert page_bytes(dest, 2) == expected[0]
    assert page_bytes(dest, 6) == expected[1]
    assert all(not any(data) for start in (0, 4) for data in page_bytes(dest, start))
    assert backend_factory.server.objects["model_h0_1_full_k"] == expected[0][0]
    assert backend_factory.server.objects["model_h0_1_full_v"] == expected[0][1]
    assert len(writer.gcache_client.registrations) == (4 if asymmetric else 2)


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parent / "fixtures/gcache_legacy_contract.json").read_text()
    )["page_bytes"],
    ids=lambda case: f"{case['layout']}-asymmetric={case['asymmetric']}",
)
def test_primary_page_bytes_match_legacy_vectors(backend_factory, monkeypatch, case):
    from sglang.srt.mem_cache.pool_host import mha

    pool = make_mha_pool(case["layout"], case["asymmetric"])
    # Exercise target init_kv_buffer as well as target address metadata, with
    # only the host allocation replaced by ordinary CPU memory.
    pool.device_pool = SimpleNamespace(device="cpu")
    pool.pin_memory = False
    pool.allocator = None
    monkeypatch.setitem(
        mha.ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, *, dtype, **kwargs: torch.zeros(dims, dtype=dtype),
    )
    pool.kv_buffer = pool.init_kv_buffer()
    for component, buffer in enumerate((pool.k_buffer, pool.v_buffer)):
        dim = pool.head_dim if component == 0 else pool.v_head_dim
        logical = (
            torch.arange(
                pool.size * pool.layer_num * pool.head_num * dim, dtype=pool.dtype
            ).reshape(pool.size, pool.layer_num, pool.head_num, dim)
            + component * 1000
        )
        physical = (
            logical
            if pool.layout == "page_first"
            else logical.reshape(
                pool.page_num, pool.page_size, pool.layer_num, pool.head_num, dim
            ).transpose(1, 2)
        )
        buffer.copy_(physical)
    backend = backend_factory(pool)
    assert backend.batch_set_v1(["h0", "h1"], torch.tensor([4, 5, 0, 1])) == [
        True,
        True,
    ]
    values = [
        backend_factory.server.objects[f"model_{key}_1_full_{component}"]
        for key in ("h0", "h1")
        for component in ("k", "v")
    ]
    assert [len(value) for value in values] == case["component_sizes"]
    assert [hashlib.sha256(value).hexdigest() for value in values] == case[
        "component_sha256"
    ]


def test_kv_alias_and_storage_views_register_once(backend_factory):
    pool = make_mha_pool()
    pool.get_hybrid_pool_buffer = lambda: [
        pool.k_buffer.flatten()[1:],
        [None, pool.v_buffer],
        torch.empty(0),
    ]
    backend = backend_factory(pool)
    backend.register_mem_pool_host(pool)
    backend.register_mem_host_pool_v2(pool, PoolName.KV)
    backend.register_mem_host_pool_v2(pool, PoolName.KV)
    storage = pool.kv_buffer.untyped_storage()
    assert backend.gcache_client.registrations == [
        (storage.data_ptr(), storage.nbytes())
    ]
    assert backend.registered_pools[PoolName.KV] is pool
    fill_pool(pool)
    assert backend.batch_set_v1(["h0"], torch.tensor([0, 1])) == [True]


def test_attention_tp_keys_and_dp_config_filename(backend_factory):
    backend = backend_factory(tp_rank=3, tp_size=4)
    assert backend.batch_set_v1(["h0"], torch.tensor([0, 1])) == [True]
    assert set(backend_factory.server.objects) == {
        "model_h0_3_full_k",
        "model_h0_3_full_v",
    }
    assert "dp2tp3" in Path(backend.gcache_client.options["conf_file"]).name


def test_exists_counts_only_complete_leading_pages(backend_factory):
    backend = backend_factory()
    keys = ["h0", "h1", "h2"]
    assert backend.batch_set_v1(keys, torch.arange(6)) == [True] * 3
    backend_factory.server.delete("model_h1_1_full_v")
    assert backend.batch_exists(keys) == 1
    assert backend.exists("h0") is True
    assert backend.exists("h1") is False
    query = backend_factory.server.calls[-1]
    assert query["for_read"] is True
    assert query["sizes"] == [
        len(value) for value in page_bytes(backend.mem_pool_host, 0)
    ]


def test_set_only_missing_components_and_requires_both(backend_factory):
    backend = backend_factory()
    pool = backend.mem_pool_host
    fill_pool(pool)
    k, v = page_bytes(pool, 0)
    backend_factory.server.put(
        {
            "model_h0_1_full_k": b"x" * len(k),
            "model_h1_1_full_k": k,
            "model_h1_1_full_v": v,
        }
    )
    backend_factory.server.configure("set", fail_keys=["model_h2_1_full_v"])
    assert backend.batch_set_v1(["h0", "h1", "h2"], torch.arange(6)) == [
        True,
        True,
        False,
    ]
    query, write = backend_factory.server.calls[-2:]
    assert query["for_read"] is False
    assert write["keys"] == [
        "model_h0_1_full_v",
        "model_h2_1_full_k",
        "model_h2_1_full_v",
    ]
    assert backend_factory.server.objects["model_h0_1_full_k"] == b"x" * len(k)
    backend_factory.server.clear_calls()
    assert backend.batch_set_v1(["h0", "h1"], torch.arange(4)) == [True, True]
    assert [call["operation"] for call in backend_factory.server.calls] == ["exists"]


def test_get_reports_partial_pages_without_promoting_a_later_hit(backend_factory):
    backend = backend_factory()
    assert backend.batch_set_v1(["h0", "h1", "h2"], torch.arange(6)) == [True] * 3
    backend_factory.server.configure("get", fail_keys=["model_h1_1_full_k"])
    assert backend.batch_get_v1(["h0", "h1", "h2"], torch.arange(6)) == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "bad_result", [None, [], [True], [True] * 3, [1, 1], [True, "true"]]
)
@pytest.mark.parametrize("operation", ["exists", "get", "set"])
def test_malformed_sdk_results_fail_closed(backend_factory, operation, bad_result):
    backend = backend_factory()
    backend_factory.server.configure(operation, result=bad_result)
    if operation == "exists":
        assert backend.batch_exists(["h0"]) == 0
    else:
        assert getattr(backend, f"batch_{operation}_v1")(
            ["h0"], torch.tensor([0, 1])
        ) == [False]


@pytest.mark.parametrize("operation", ["exists", "get", "set"])
def test_sdk_exception_is_a_cache_failure(backend_factory, operation):
    backend = backend_factory()
    backend_factory.server.configure(operation, error="injected server failure")
    if operation == "exists":
        assert backend.batch_exists(["h0"]) == 0
        assert backend.batch_set_v1(["h0"], torch.tensor([0, 1])) == [False]
        assert not any(
            call["operation"] == "set" for call in backend_factory.server.calls
        )
    else:
        assert getattr(backend, f"batch_{operation}_v1")(
            ["h0"], torch.tensor([0, 1])
        ) == [False]


def test_failed_write_preserves_existing_page_success(backend_factory):
    backend = backend_factory()
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    backend_factory.server.configure("set", error="unavailable")
    assert backend.batch_set_v1(["h0", "h1"], torch.arange(4)) == [True, False]


def test_empty_batches_do_not_call_sdk(backend_factory):
    backend = backend_factory()
    backend_factory.server.clear_calls()
    indices = torch.empty(0, dtype=torch.int64)
    assert backend.batch_exists([]) == 0
    assert backend.batch_get_v1([], indices) == []
    assert backend.batch_set_v1([], indices) == []
    assert backend_factory.server.calls == []


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([0]),
        torch.tensor([1, 2]),
        torch.tensor([0, 2]),
        torch.tensor([-2, -1]),
        torch.tensor([8, 9]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0, 1]]),
        torch.tensor([0, 1, 0, 1]),
    ],
)
def test_invalid_host_indices_fail_before_sdk_io(backend_factory, indices):
    backend = backend_factory()
    keys = ["h0", "h1"] if indices.numel() == 4 else ["h0"]
    backend_factory.server.clear_calls()
    assert backend.batch_get_v1(keys, indices) == [False] * len(keys)
    assert backend.batch_set_v1(keys, indices) == [False] * len(keys)
    assert backend_factory.server.calls == []


def test_invalid_metadata_cannot_read_unregistered_memory(backend_factory):
    backend = backend_factory()
    backend.mem_pool_host.get_page_buffer_meta = lambda indices: ([1, 2], [4, 4])
    backend_factory.server.clear_calls()
    assert backend.batch_get_v1(["h0"], torch.arange(2)) == [False]
    assert backend_factory.server.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_mla_model": True},
        {"pp_size": 2},
        {"attn_cp_size": 2},
        {"should_split_heads": True},
        {"tp_lcm_size": 4},
        {"tp_rank": 2},
        {"tp_size": 0},
        {"model_name": ""},
    ],
)
def test_unsupported_topology_rejected_before_client_init(backend_factory, overrides):
    with pytest.raises((ValueError, NotImplementedError)):
        backend_factory(**overrides)
    assert backend_factory.clients == []


@pytest.mark.parametrize("layout", ["layer_first", "page_head"])
def test_unsupported_layout_rejected_before_client_init(backend_factory, layout):
    pool = make_mha_pool()
    pool.layout = layout
    with pytest.raises((ValueError, NotImplementedError), match="layout"):
        backend_factory(pool)
    assert backend_factory.clients == []


def test_unsupported_pool_rejected(backend_factory):
    with pytest.raises((TypeError, NotImplementedError), match="MHA"):
        backend_factory(SimpleNamespace(layout="page_first"))
    assert backend_factory.clients == []


@pytest.mark.parametrize(
    "field,value", [("dcp_size", 2), ("mtp_draft_device_pools", (object(),))]
)
def test_unscoped_pool_geometry_is_rejected(backend_factory, field, value):
    pool = make_mha_pool()
    setattr(pool, field, value)
    with pytest.raises(NotImplementedError, match="DCP|draft"):
        backend_factory(pool)
    assert backend_factory.clients == []


@pytest.mark.parametrize(
    "pool_name", [PoolName.MAMBA, PoolName.INDEXER, PoolName.DRAFT_INDEXER]
)
def test_unsupported_pool_names_rejected(backend_factory, pool_name):
    backend = backend_factory()
    with pytest.raises(NotImplementedError, match="pool"):
        backend.register_mem_host_pool_v2(make_mha_pool(), pool_name)
    transfer = PoolTransfer(name=pool_name, keys=["h0"], host_indices=torch.arange(2))
    assert backend.batch_exists_v2(["h0"], [transfer]).kv_hit_pages == 0
    assert backend.batch_get_v2([transfer]) == {pool_name: [False]}
    assert backend.batch_set_v2([transfer]) == {pool_name: [False]}


def test_registration_errors_propagate(backend_factory):
    backend = backend_factory(register=False)
    backend_factory.server.configure("register", error="cannot register")
    with pytest.raises(RuntimeError, match="register"):
        backend.register_mem_pool_host(backend.mem_pool_host)


def test_rebinding_primary_pool_is_rejected(backend_factory):
    backend = backend_factory()
    with pytest.raises(ValueError, match="pool"):
        backend.register_mem_host_pool_v2(make_mha_pool(), PoolName.KV)


def test_tensor_copy_api_is_explicitly_unsupported(backend_factory):
    backend = backend_factory()
    for method, argument in (
        ("get", "h0"),
        ("set", "h0"),
        ("batch_get", ["h0"]),
        ("batch_set", ["h0"]),
    ):
        with pytest.raises(NotImplementedError, match="v1|zero.copy"):
            getattr(backend, method)(argument)


def test_storage_metrics_count_successful_pages_and_drain(backend_factory):
    backend = backend_factory()
    backend_factory.server.configure("set", fail_keys=["model_h1_1_full_v"])
    assert backend.batch_set_v1(["h0", "h1"], torch.arange(4)) == [True, False]
    assert backend.batch_get_v1(["h0", "h1"], torch.arange(4)) == [True, False]
    stats = backend.get_stats()
    assert stats.backup_pgs == [1]
    assert stats.prefetch_pgs == [1]
    assert len(stats.backup_bandwidth) == len(stats.prefetch_bandwidth) == 1
    assert stats.backup_bandwidth[0] > 0 and stats.prefetch_bandwidth[0] > 0
    assert backend.get_stats().backup_pgs == []
    assert stats.backup_pgs == [1]


def test_metrics_can_be_disabled(backend_factory):
    backend = backend_factory(enable_storage_metrics=False)
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    assert backend.get_stats().backup_pgs == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
