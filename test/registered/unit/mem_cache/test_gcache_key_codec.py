"""Legacy GCache object naming and storage-hash compatibility."""

import json
from pathlib import Path

import pytest

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_legacy_component_keys():
    from sglang.srt.mem_cache.storage.gcache.key_codec import GCacheKeyCodec

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/gcache_legacy_contract.json").read_text()
    )
    for case in fixture["keys"]:
        codec = GCacheKeyCodec(
            case["model_name"], case["tp_rank"], case.get("draft_model_name")
        )
        assert (
            codec.component_keys(PoolName(case["pool"]), case["hashes"])
            == case["objects"]
        )


def test_namespaces_and_shards_do_not_collide():
    from sglang.srt.mem_cache.storage.gcache.key_codec import GCacheKeyCodec

    a = GCacheKeyCodec("model", 0)
    b = GCacheKeyCodec("model", 1)
    assert a.component_keys(PoolName.KV, ["h0"]) != b.component_keys(
        PoolName.KV, ["h0"]
    )
    assert a.component_keys(PoolName.KV, ["h0"]) != a.component_keys(
        PoolName.DRAFT, ["h0"]
    )
    draft_a = GCacheKeyCodec("target-a", 0, "shared-draft")
    draft_b = GCacheKeyCodec("target-b", 0, "shared-draft")
    assert draft_a.component_keys(PoolName.DRAFT, ["h0"]) == [
        "shared-draft_h0_0_full_k",
        "shared-draft_h0_0_full_v",
    ]
    assert draft_a.component_keys(PoolName.DRAFT, ["h0"]) == draft_b.component_keys(
        PoolName.DRAFT, ["h0"]
    )


def test_empty_batch_and_unknown_pool():
    from sglang.srt.mem_cache.storage.gcache.key_codec import GCacheKeyCodec

    codec = GCacheKeyCodec("model", 0)
    assert codec.component_keys(PoolName.KV, []) == []
    with pytest.raises(ValueError, match="pool"):
        codec.component_keys(PoolName.MAMBA, ["h0"])


@pytest.mark.parametrize("model,rank", [("", 0), ("   ", 0), ("model", -1)])
def test_rejects_invalid_identity(model, rank):
    from sglang.srt.mem_cache.storage.gcache.key_codec import GCacheKeyCodec

    with pytest.raises(ValueError):
        GCacheKeyCodec(model, rank)


def test_native_hash_matches_legacy_vectors():
    from sglang.srt.mem_cache.utils import get_hash_str

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/gcache_legacy_contract.json").read_text()
    )
    for case in fixture["hashes"]:
        assert get_hash_str(case["tokens"], case["prior_hash"]) == case["hash"]


def test_request_namespace_preserves_isolation():
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    class Tokens(list):
        extra_key = None
        cache_salt = None

    plain = Tokens([1, 2, 3, 4])
    adapter = Tokens(plain)
    adapter.extra_key = "adapter-a"
    tenant = Tokens(plain)
    tenant.cache_salt = "tenant-a"
    assert (
        len(
            {
                get_storage_hash_str(plain),
                get_storage_hash_str(adapter),
                get_storage_hash_str(tenant),
            }
        )
        == 3
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
