"""Named-pool transfers and recoverable SWA endpoints through the KV server."""

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory
from sglang.test.gcache_test_utils import fill_pool, make_mha_pool, page_bytes

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def write_mask(backend, name, pool, mask, *, seed=0):
    fill_pool(pool, seed)
    selected = [i for i, present in enumerate(mask) if present]
    transfer = PoolTransfer(
        name=name,
        keys=[f"h{i}" for i in selected],
        host_indices=torch.tensor(
            [
                j
                for i in selected
                for j in range(i * pool.page_size, (i + 1) * pool.page_size)
            ],
            dtype=torch.int64,
        ),
    )
    assert backend.batch_set_v2([transfer]) == {name: [True] * len(selected)}


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
def test_full_and_swa_use_independent_page_geometry(backend_factory, layout):
    full_source = make_mha_pool(layout, size=16)
    full_dest = make_mha_pool(layout, size=16)
    swa_source = make_mha_pool(
        layout, asymmetric=True, size=24, page_size=4, layers=2, v_head_dim=7
    )
    swa_dest = make_mha_pool(
        layout, asymmetric=True, size=24, page_size=4, layers=2, v_head_dim=7
    )
    writer, reader = backend_factory(full_source), backend_factory(full_dest)
    writer.register_mem_host_pool_v2(swa_source, PoolName.SWA)
    reader.register_mem_host_pool_v2(swa_dest, PoolName.SWA)
    fill_pool(full_source, seed=17)
    fill_pool(swa_source, seed=103)
    keys = ["h0", "h1"]
    full_bytes = [page_bytes(full_source, i) for i in (8, 2)]
    swa_bytes = [page_bytes(swa_source, i) for i in (16, 4)]
    assert writer.batch_set_v1(keys, torch.tensor([8, 9, 2, 3])) == [True, True]
    assert writer.batch_set_v2(
        [
            PoolTransfer(
                PoolName.SWA, torch.tensor([16, 17, 18, 19, 4, 5, 6, 7]), keys=keys
            )
        ]
    ) == {PoolName.SWA: [True, True]}
    assert reader.batch_get_v1(keys, torch.tensor([4, 5, 0, 1])) == [True, True]
    assert reader.batch_get_v2(
        [
            PoolTransfer(
                PoolName.SWA, torch.tensor([8, 9, 10, 11, 20, 21, 22, 23]), keys=keys
            )
        ]
    ) == {PoolName.SWA: [True, True]}
    assert [page_bytes(full_dest, i) for i in (4, 0)] == full_bytes
    assert [page_bytes(swa_dest, i) for i in (8, 20)] == swa_bytes
    assert all(not any(data) for data in page_bytes(swa_dest, 0))
    query = reader.batch_exists_v2(
        keys, [PoolTransfer(PoolName.SWA, keys=["placeholder"])]
    )
    assert query.kv_hit_pages == 2
    assert query.restorable_prefix_pages == [1, 2]
    swa_query = backend_factory.server.calls[-1]
    assert swa_query["sizes"] == [len(value) for value in swa_bytes[0]] * 2
    assert all("placeholder" not in key for key in swa_query["keys"])


@pytest.mark.parametrize(
    "mask,window,policy,expected",
    [
        ([1, 1, 0, 1, 1, 1], 2, PoolHitPolicy.TRAILING_PAGES, [1, 2, 5, 6]),
        ([1, 1, 0, 1, 1, 0], 2, PoolHitPolicy.TRAILING_PAGES, [1, 2, 5]),
        ([0, 1, 1, 1, 1, 1], 2, PoolHitPolicy.TRAILING_PAGES, [3, 4, 5, 6]),
        ([1, 1, 0, 1, 1, 1], 2, PoolHitPolicy.ALL_PAGES, [1, 2]),
        ([1, 1, 1, 1, 1, 1], 9, PoolHitPolicy.TRAILING_PAGES, [1, 2, 3, 4, 5, 6]),
        ([0, 0, 0, 0, 0, 0], 2, PoolHitPolicy.TRAILING_PAGES, []),
    ],
)
def test_swa_reports_all_valid_endpoints(
    backend_factory, mask, window, policy, expected
):
    full = make_mha_pool(size=16)
    swa = make_mha_pool(size=24, page_size=4, layers=1)
    backend = backend_factory(full)
    backend.register_mem_host_pool_v2(swa, PoolName.SWA)
    write_mask(backend, PoolName.KV, full, [1] * 6)
    write_mask(backend, PoolName.SWA, swa, mask, seed=33)
    backend_factory.server.clear_calls()
    result = backend.batch_exists_v2(
        [f"h{i}" for i in range(6)],
        [PoolTransfer(PoolName.SWA, keys=["unused"] * window, hit_policy=policy)],
    )
    assert result.restorable_prefix_pages == expected
    assert result.kv_hit_pages == (expected[-1] if expected else 0)
    assert result.extra_pool_hit_pages[PoolName.KV] == 6
    if expected:
        assert result.extra_pool_hit_pages[PoolName.SWA] == expected[-1]
    assert all(
        "unused" not in key
        for call in backend_factory.server.calls
        for key in call["keys"]
    )


def test_two_trailing_pools_intersect_endpoints_not_maxima(backend_factory):
    full = make_mha_pool(size=16)
    backend = backend_factory(full)
    write_mask(backend, PoolName.KV, full, [1] * 6)
    for name, mask in (
        (PoolName.SWA, [1, 1, 0, 1, 1, 1]),
        (PoolName.DRAFT_SWA, [1, 1, 1, 1, 0, 0]),
    ):
        pool = make_mha_pool(size=16, layers=1)
        backend.register_mem_host_pool_v2(pool, name)
        write_mask(backend, name, pool, mask)
    result = backend.batch_exists_v2(
        [f"h{i}" for i in range(6)],
        [
            PoolTransfer(
                name, keys=["window"] * 2, hit_policy=PoolHitPolicy.TRAILING_PAGES
            )
            for name in (PoolName.SWA, PoolName.DRAFT_SWA)
        ],
    )
    # min(6, 4) would select an endpoint inside the SWA hole.
    assert result.extra_pool_hit_pages[PoolName.SWA] == 6
    assert result.extra_pool_hit_pages[PoolName.DRAFT_SWA] == 4
    assert result.restorable_prefix_pages == [1, 2]
    assert result.kv_hit_pages == 2


def test_main_prefix_stops_on_a_missing_component(backend_factory):
    full = make_mha_pool(size=16)
    swa = make_mha_pool(size=16)
    backend = backend_factory(full)
    backend.register_mem_host_pool_v2(swa, PoolName.SWA)
    write_mask(backend, PoolName.KV, full, [1] * 6)
    write_mask(backend, PoolName.SWA, swa, [1] * 6)
    backend_factory.server.delete("model_h2_1_full_v")
    result = backend.batch_exists_v2(
        [f"h{i}" for i in range(6)],
        [
            PoolTransfer(
                PoolName.SWA,
                keys=["window"] * 2,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    assert result.kv_hit_pages == 2
    assert result.restorable_prefix_pages == [1, 2]


def test_v2_partial_read_and_repair_missing_v(backend_factory):
    backend = backend_factory()
    swa = make_mha_pool(asymmetric=True)
    backend.register_mem_host_pool_v2(swa, PoolName.SWA)
    fill_pool(swa, seed=63)
    transfer = PoolTransfer(PoolName.SWA, torch.arange(6), keys=["h0", "h1", "h2"])
    assert backend.batch_set_v2([transfer])[PoolName.SWA] == [True] * 3
    backend_factory.server.delete("model_h1_1_swa_v")
    backend_factory.server.clear_calls()
    assert backend.batch_set_v2([transfer])[PoolName.SWA] == [True] * 3
    assert backend_factory.server.calls[-1]["keys"] == ["model_h1_1_swa_v"]
    backend_factory.server.configure("get", fail_keys=["model_h1_1_swa_k"])
    assert backend.batch_get_v2([transfer])[PoolName.SWA] == [True, False, True]


def test_side_pool_indices_validated_against_its_own_pool(backend_factory):
    backend = backend_factory(make_mha_pool(size=32))
    swa = make_mha_pool(size=8, page_size=4)
    backend.register_mem_host_pool_v2(swa, PoolName.SWA)
    backend_factory.server.clear_calls()
    for indices in (torch.arange(2), torch.arange(8, 12), torch.tensor([0, 1, 3, 2])):
        transfer = PoolTransfer(PoolName.SWA, indices, keys=["h0"])
        assert backend.batch_get_v2([transfer]) == {PoolName.SWA: [False]}
        assert backend.batch_set_v2([transfer]) == {PoolName.SWA: [False]}
    assert backend_factory.server.calls == []


def test_side_pool_metadata_must_not_point_into_primary(backend_factory):
    backend = backend_factory()
    swa = make_mha_pool()
    backend.register_mem_host_pool_v2(swa, PoolName.SWA)
    swa.get_page_buffer_meta = backend.mem_pool_host.get_page_buffer_meta
    backend_factory.server.clear_calls()
    assert backend.batch_get_v2(
        [PoolTransfer(PoolName.SWA, torch.arange(2), keys=["h0"])]
    ) == {PoolName.SWA: [False]}
    assert backend_factory.server.calls == []


def test_empty_v2_batches_do_not_contact_server(backend_factory):
    backend = backend_factory()
    backend_factory.server.clear_calls()
    assert backend.batch_exists_v2([]).restorable_prefix_pages == []
    assert backend.batch_get_v2([]) == {}
    assert backend.batch_set_v2([]) == {}
    empty = PoolTransfer(PoolName.SWA, torch.empty(0, dtype=torch.int64), keys=[])
    assert backend.batch_get_v2([empty]) == {PoolName.SWA: []}
    assert backend.batch_set_v2([empty]) == {PoolName.SWA: []}
    assert backend_factory.server.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
