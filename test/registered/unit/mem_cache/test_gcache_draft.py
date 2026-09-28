"""Draft identities, bytes and missing sidecars through the network mockserver."""

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory
from sglang.test.gcache_test_utils import fill_pool, make_mha_pool, page_bytes

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("name", [PoolName.DRAFT, PoolName.DRAFT_SWA])
@pytest.mark.parametrize("draft_model", [None, "shared-draft"])
def test_draft_and_target_bytes_are_isolated(
    backend_factory, monkeypatch, name, draft_model
):
    if draft_model is not None:
        monkeypatch.setenv("SGLANG_HICACHE_DRAFT_MODEL_NAME", draft_model)
    backend = backend_factory()
    draft = make_mha_pool(asymmetric=True, layers=1)
    backend.register_mem_host_pool_v2(draft, name)
    fill_pool(backend.mem_pool_host, seed=19)
    fill_pool(draft, seed=114)
    expected_target = page_bytes(backend.mem_pool_host, 0)
    expected_draft = page_bytes(draft, 4)
    transfer = PoolTransfer(name, torch.tensor([4, 5]), keys=["h0"])
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    assert backend.batch_set_v2([transfer]) == {name: [True]}
    prefix = draft_model or "d:model"
    kind = "swa" if name == PoolName.DRAFT_SWA else "full"
    stored = backend_factory.server.objects
    assert stored["model_h0_1_full_k"] == expected_target[0]
    assert stored[f"{prefix}_h0_1_{kind}_k"] == expected_draft[0]
    assert stored[f"{prefix}_h0_1_{kind}_v"] == expected_draft[1]
    draft.k_buffer.zero_()
    draft.v_buffer.zero_()
    transfer.host_indices = torch.tensor([2, 3])
    assert backend.batch_get_v2([transfer]) == {name: [True]}
    assert page_bytes(draft, 2) == expected_draft
    assert page_bytes(backend.mem_pool_host, 0) == expected_target


def test_named_draft_can_be_reused_by_another_target(backend_factory, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_DRAFT_MODEL_NAME", "shared")
    writer = backend_factory(model_name="target-a")
    reader = backend_factory(model_name="target-b")
    source, dest = make_mha_pool(layers=1), make_mha_pool(layers=1)
    writer.register_mem_host_pool_v2(source, PoolName.DRAFT)
    reader.register_mem_host_pool_v2(dest, PoolName.DRAFT)
    fill_pool(source, seed=73)
    expected = page_bytes(source, 0)
    assert writer.batch_set_v2(
        [PoolTransfer(PoolName.DRAFT, torch.arange(2), keys=["h0"])]
    ) == {PoolName.DRAFT: [True]}
    assert reader.batch_get_v2(
        [PoolTransfer(PoolName.DRAFT, torch.arange(2), keys=["h0"])]
    ) == {PoolName.DRAFT: [True]}
    assert page_bytes(dest, 0) == expected
    assert set(backend_factory.server.objects) == {
        "shared_h0_1_full_k",
        "shared_h0_1_full_v",
    }


def test_all_swa_draft_never_requires_a_full_draft_object(backend_factory):
    backend = backend_factory()
    draft_swa = make_mha_pool(layers=1)
    backend.register_mem_host_pool_v2(draft_swa, PoolName.DRAFT_SWA)
    assert PoolName.DRAFT not in backend.registered_pools
    assert backend.batch_set_v1(["h0", "h1"], torch.arange(4)) == [True, True]
    transfer = PoolTransfer(
        PoolName.DRAFT_SWA,
        torch.arange(4),
        keys=["h0", "h1"],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    assert backend.batch_set_v2([transfer]) == {PoolName.DRAFT_SWA: [True, True]}
    result = backend.batch_exists_v2(["h0", "h1"], [transfer])
    assert result.kv_hit_pages == 2
    assert result.restorable_prefix_pages == [1, 2]
    assert not any(
        key.startswith("d:") and "_full_" in key
        for call in backend_factory.server.calls
        for key in call["keys"]
    )
    assert len(backend.gcache_client.registrations) == 2


@pytest.mark.parametrize("component", ["k", "v"])
@pytest.mark.parametrize("page_size,draft_layers", [(2, 1), (64, 5)])
def test_missing_draft_component_limits_prefix_and_reports_read_failure(
    backend_factory, component, page_size, draft_layers
):
    backend = backend_factory(make_mha_pool(size=4 * page_size, page_size=page_size))
    draft = make_mha_pool(size=4 * page_size, page_size=page_size, layers=draft_layers)
    backend.register_mem_host_pool_v2(draft, PoolName.DRAFT)
    fill_pool(draft, seed=41)
    expected = [page_bytes(draft, i * page_size) for i in range(3)]
    keys = ["h0", "h1", "h2"]
    indices = torch.arange(3 * page_size)
    transfer = PoolTransfer(
        PoolName.DRAFT, indices, keys=keys, indices_from_pool=PoolName.KV
    )
    assert backend.batch_set_v1(keys, indices) == [True] * 3
    assert backend.batch_set_v2([transfer])[PoolName.DRAFT] == [True] * 3
    backend_factory.server.delete(f"d:model_h1_1_full_{component}")
    assert backend.batch_exists(keys) == 3
    assert backend.batch_exists_v2(keys, [transfer]).restorable_prefix_pages == [1]
    draft.k_buffer.zero_()
    draft.v_buffer.zero_()
    assert backend.batch_get_v2([transfer])[PoolName.DRAFT] == [True, False, True]
    assert page_bytes(draft, 0) == expected[0]
    assert page_bytes(draft, 2 * page_size) == expected[2]


@pytest.mark.parametrize("component", ["k", "v"])
def test_five_layer_draft_swa_requires_complete_1024_token_tail(
    backend_factory, component
):
    page_size, pages = 64, 20
    backend = backend_factory(
        make_mha_pool(size=pages * page_size, page_size=page_size)
    )
    draft = make_mha_pool(size=pages * page_size, page_size=page_size, layers=5)
    backend.register_mem_host_pool_v2(draft, PoolName.DRAFT_SWA)
    keys = [f"h{i}" for i in range(pages)]
    indices = torch.arange(pages * page_size)
    assert backend.batch_set_v1(keys, indices) == [True] * pages
    assert (
        backend.batch_set_v2([PoolTransfer(PoolName.DRAFT_SWA, indices, keys=keys)])[
            PoolName.DRAFT_SWA
        ]
        == [True] * pages
    )
    query = PoolTransfer(
        PoolName.DRAFT_SWA,
        keys=["window"] * 16,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    # An early missing page only disqualifies prefixes whose 1024-token
    # draft tail still covers it. Longer prefixes can be restored again.
    backend_factory.server.delete(f"d:model_h1_1_swa_{component}")
    result = backend.batch_exists_v2(keys, [query])
    assert result.kv_hit_pages == pages
    assert result.restorable_prefix_pages == [1, 18, 19, 20]
    # A missing page in the last 16 pages must not count as a complete hit.
    backend_factory.server.delete(f"d:model_h18_1_swa_{component}")
    assert backend.batch_exists_v2(keys, [query]).restorable_prefix_pages == [1, 18]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
