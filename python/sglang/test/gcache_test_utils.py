"""Real CPU host pools and the network GCache mockserver test fixture."""

import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.test.gcache_mock_server import GCacheMockServer, NetworkGCacheClient


def make_mha_pool(
    layout="page_first",
    asymmetric=False,
    dtype=torch.float16,
    *,
    size=8,
    page_size=2,
    layers=3,
    heads=2,
    head_dim=4,
    v_head_dim=None,
):
    from sglang.srt.mem_cache.pool_host.mha import (
        AsymmetricMHATokenToKVPoolHost,
        MHATokenToKVPoolHost,
    )

    cls = AsymmetricMHATokenToKVPoolHost if asymmetric else MHATokenToKVPoolHost
    pool = cls.__new__(cls)
    pool.layout = layout
    pool.size = size
    pool.page_size = page_size
    pool.page_num = pool.size // pool.page_size
    pool.layer_num = layers
    pool.head_num = heads
    pool.head_dim = head_dim
    pool.v_head_dim = v_head_dim or (6 if asymmetric else head_dim)
    pool.dtype = dtype
    pool.device = "cpu"
    pool.dcp_size = 1
    pool.can_use_write_back_jit = False
    if layout == "page_first":
        prefix = (pool.size, pool.layer_num)
    else:
        prefix = (pool.page_num, pool.layer_num, pool.page_size)
    k_shape = (*prefix, pool.head_num, pool.head_dim)
    v_shape = (*prefix, pool.head_num, pool.v_head_dim)
    pool.kv_buffer = (
        (torch.zeros(k_shape, dtype=dtype), torch.zeros(v_shape, dtype=dtype))
        if asymmetric
        else torch.zeros((2, *k_shape), dtype=dtype)
    )
    pool.lock = threading.RLock()
    pool.free_slots = torch.arange(pool.size, dtype=torch.int64)
    pool.release_slots = []
    pool.num_release_slots = 0
    pool.slot_used = torch.zeros(pool.size, dtype=torch.bool)
    return pool


def make_storage_config(pool, tmp_path, **overrides):
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

    config = dict(
        tp_rank=1,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=True,
        is_page_first_layout=pool.layout == "page_first",
        model_name="model",
        extra_config={
            "config_dir": str(tmp_path / "flags"),
            "log_dir": str(tmp_path / "logs"),
        },
    )
    config.update(overrides)
    return HiCacheStorageConfig(**config)


def make_backend_factory(monkeypatch, tmp_path):
    from sglang.srt.mem_cache.storage.gcache import gcache_storage

    server = GCacheMockServer()
    clients = []
    backends = []

    def client_factory(**kwargs):
        # Match the pinned SDK's process singleton, including across detach.
        if not clients:
            clients.append(NetworkGCacheClient(**kwargs))
        return clients[0]

    monkeypatch.setitem(
        sys.modules, "gcache_sdk", SimpleNamespace(GCacheClient=client_factory)
    )
    monkeypatch.setattr(gcache_storage, "get_attention_dp_rank", lambda: 2)
    monkeypatch.setenv("GCACHE_MASTER_ADDR", server.address)
    monkeypatch.setenv("GCACHE_CACHE_GROUP", "test")
    monkeypatch.setenv("HPN_IB_IFNAME", "eth1")
    monkeypatch.delenv("GCACHE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("SGLANG_HICACHE_DRAFT_MODEL_NAME", raising=False)

    def create(pool=None, *, register=True, via_factory=False, **overrides):
        pool = pool if pool is not None else make_mha_pool()
        config = make_storage_config(pool, tmp_path, **overrides)
        if via_factory:
            from sglang.srt.mem_cache.storage import StorageBackendFactory

            backend = StorageBackendFactory.create_backend("gcache", config, pool)
        else:
            backend = gcache_storage.GcacheStorage(config, pool)
        backends.append(backend)
        if register:
            backend.register_mem_pool_host(pool)
        return backend

    def close():
        for backend in backends:
            if hasattr(backend, "close"):
                backend.close()
        server.close()

    create.clients = clients
    create.server = server
    create.config_dir = tmp_path / "flags"
    create.log_dir = tmp_path / "logs"
    create.close = close
    return create


@pytest.fixture
def backend_factory(monkeypatch, tmp_path):
    factory = make_backend_factory(monkeypatch, tmp_path)
    try:
        yield factory
    finally:
        factory.close()


def fill_pool(pool, seed=0):
    for component, buffer in enumerate((pool.k_buffer, pool.v_buffer)):
        raw = buffer.view(torch.uint8).flatten()
        raw.copy_((torch.arange(raw.numel()) + component * 37 + seed).remainder(251))


def page_bytes(pool, start):
    """Independent oracle: slice tensors, without using address metadata."""
    result = []
    for buffer in (pool.k_buffer, pool.v_buffer):
        page = (
            buffer[start : start + pool.page_size]
            if pool.layout == "page_first"
            else buffer[start // pool.page_size]
        )
        result.append(page.contiguous().view(torch.uint8).numpy().tobytes())
    return result
