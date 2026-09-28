"""Registration ownership and detach while real network requests are in flight."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory
from sglang.test.gcache_test_utils import fill_pool, make_mha_pool, page_bytes

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def test_reattach_reuses_client_registration_and_config(backend_factory):
    pool = make_mha_pool(asymmetric=True)
    fill_pool(pool, seed=51)
    first = backend_factory(pool)
    expected = page_bytes(pool, 0)
    assert first.batch_set_v1(["h0"], torch.arange(2)) == [True]
    config_path = first.config_path
    client = first.gcache_client
    for _ in range(3):
        first.close()
        first.close()
        assert not client.closed
        assert first.batch_get_v1(["h0"], torch.arange(2)) == [False]
        first = backend_factory(pool)
        assert first.gcache_client is client
        assert first.config_path == config_path
        assert len(client.registrations) == 2
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        assert first.batch_get_v1(["h0"], torch.arange(2)) == [True]
        assert page_bytes(pool, 0) == expected
    assert len(list(config_path.parent.glob("*.flags"))) == 1
    assert len(backend_factory.clients) == 1


def test_detach_waits_for_inflight_network_read(backend_factory):
    pool = make_mha_pool()
    fill_pool(pool)
    backend = backend_factory(pool)
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    backend_factory.server.clear_calls()
    backend_factory.server.configure("get", hold=True)
    closed = threading.Event()

    def close():
        backend.close()
        closed.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        read = workers.submit(backend.batch_get_v1, ["h0"], torch.arange(2))
        backend_factory.server.wait_for_call("get")
        detach = workers.submit(close)
        try:
            assert not closed.wait(timeout=0.05)
            assert backend.registered_pools[PoolName.KV] is pool
        finally:
            backend_factory.server.release("get")
        assert read.result(timeout=5) == [True]
        detach.result(timeout=5)
    assert closed.is_set()
    assert backend.registered_pools == {}
    assert not backend.gcache_client.closed


def test_registered_allocation_outlives_detach_without_sdk_unregister(backend_factory):
    backend = backend_factory()
    allocation = backend.mem_pool_host.kv_buffer.untyped_storage()
    reference = weakref.ref(allocation)
    backend.close()
    del allocation
    gc.collect()
    # The pinned SDK has no unregister API. Releasing this allocation while
    # its native registration survives would allow stale RDMA registrations.
    assert reference() is not None


def test_partial_registration_failure_can_retry_without_double_registration(
    backend_factory,
):
    pool = make_mha_pool(asymmetric=True)
    backend = backend_factory(pool, register=False)
    backend_factory.server.configure("register", fail_after=1)
    with pytest.raises(RuntimeError, match="registration failure"):
        backend.register_mem_pool_host(pool)
    assert backend.registered_pools == {}
    assert len(backend.gcache_client.registrations) == 1
    backend_factory.server.configure("register")
    backend.register_mem_pool_host(pool)
    assert len(backend.gcache_client.registrations) == 2
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]


def test_failed_client_init_does_not_leave_flags(backend_factory, tmp_path):
    backend_factory.server.configure("init", error="initialization failed")
    with pytest.raises(RuntimeError, match="initialization failed"):
        backend_factory()
    assert list((tmp_path / "flags").glob("*.flags")) == []
    backend_factory.server.configure("init")
    backend = backend_factory()
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]


def test_singleton_cannot_silently_ignore_changed_connection_config(
    backend_factory, monkeypatch
):
    backend = backend_factory()
    backend.close()
    monkeypatch.setenv("GCACHE_CACHE_GROUP", "another-group")
    with pytest.raises(ValueError, match="restart|process"):
        backend_factory()
    assert len(backend_factory.clients) == 1


def test_network_timeout_is_a_read_failure(backend_factory):
    backend = backend_factory(
        extra_config={
            "config_dir": str(backend_factory.config_dir),
            "log_dir": str(backend_factory.log_dir),
            "timeout_us": 100_000,
        }
    )
    backend_factory.server.configure("get", delay=0.25)
    assert backend.batch_get_v1(["h0"], torch.arange(2)) == [False]


def test_repeated_metrics_registration_and_exact_transfer_bytes(backend_factory):
    from prometheus_client import REGISTRY

    pool = make_mha_pool(asymmetric=True)
    backend = backend_factory(pool)
    labels = {"storage_backend": "gcache", "tp_rank": "1", "dp_rank": "2"}
    name = "sglang:gcache_backup_bytes_total"
    before = REGISTRY.get_sample_value(name, labels) or 0
    bytes_per_page = sum(len(value) for value in page_bytes(pool, 0))
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    backend.close()
    backend = backend_factory(pool)
    # An already-stored page succeeds without counting another byte transfer.
    assert backend.batch_set_v1(["h0"], torch.arange(2)) == [True]
    assert REGISTRY.get_sample_value(name, labels) - before == bytes_per_page
    for metric in (
        "preprocess_seconds",
        "rdma_seconds",
        "postprocess_seconds",
        "batch_keys",
    ):
        assert REGISTRY.get_sample_value(f"sglang:gcache_backup_{metric}_count", labels)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
