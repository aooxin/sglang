import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.pool_host.mha import AsymmetricMHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

# These tests use AsymmetricMHATokenToKVPoolHost methods and let that class call
# the real sgl-kernel transfer ops.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="asymmetric host-pool tests require CUDA."
)

DEVICE = "cuda"
PAGE_SIZE = 16
NUM_LAYERS = 3
TOTAL_ITEMS = PAGE_SIZE * 8
HEAD_NUM = 4
K_HEAD_DIM = 192
V_HEAD_DIM = 128
DTYPES = [torch.float16, torch.bfloat16]


def token_indices_for_pages(pages, page_size=PAGE_SIZE, device=None):
    indices = torch.cat(
        [
            torch.arange(
                int(page) * page_size,
                (int(page) + 1) * page_size,
                dtype=torch.int64,
            )
            for page in pages.tolist()
        ]
    )
    return indices if device is None else indices.to(device)


def fill_with_offset(tensor, offset):
    data = torch.arange(tensor.numel(), device=tensor.device, dtype=torch.float32)
    tensor.copy_(((data + offset) % 251).view_as(tensor))


def make_host_pool(dtype, layout="page_first"):
    host = AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost)
    host.layout = layout
    host.page_size = PAGE_SIZE
    host.page_num = TOTAL_ITEMS // PAGE_SIZE
    host.layer_num = NUM_LAYERS
    host.head_num = HEAD_NUM
    host.head_dim = K_HEAD_DIM
    host.v_head_dim = V_HEAD_DIM
    host.dtype = dtype
    host.can_use_write_back_jit = False
    host.can_use_jit = False
    if layout == "layer_first":
        k_dims = (NUM_LAYERS, TOTAL_ITEMS, HEAD_NUM, K_HEAD_DIM)
        v_dims = (NUM_LAYERS, TOTAL_ITEMS, HEAD_NUM, V_HEAD_DIM)
    elif layout == "page_first":
        k_dims = (TOTAL_ITEMS, NUM_LAYERS, HEAD_NUM, K_HEAD_DIM)
        v_dims = (TOTAL_ITEMS, NUM_LAYERS, HEAD_NUM, V_HEAD_DIM)
    elif layout == "page_first_direct":
        k_dims = (host.page_num, NUM_LAYERS, PAGE_SIZE, HEAD_NUM, K_HEAD_DIM)
        v_dims = (host.page_num, NUM_LAYERS, PAGE_SIZE, HEAD_NUM, V_HEAD_DIM)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    host.kv_buffer = (
        torch.zeros(k_dims, dtype=dtype).pin_memory(),
        torch.zeros(v_dims, dtype=dtype).pin_memory(),
    )
    if layout == "layer_first":
        host.k_data_refs = list(host.k_buffer.unbind())
        host.v_data_refs = list(host.v_buffer.unbind())
        host.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in host.k_data_refs], dtype=torch.uint64, device=DEVICE
        )
        host.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in host.v_data_refs], dtype=torch.uint64, device=DEVICE
        )
    return host


def make_device_pool(dtype):
    k_buffer = [
        torch.empty(TOTAL_ITEMS, HEAD_NUM, K_HEAD_DIM, dtype=dtype, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    v_buffer = [
        torch.empty(TOTAL_ITEMS, HEAD_NUM, V_HEAD_DIM, dtype=dtype, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    for layer_id in range(NUM_LAYERS):
        fill_with_offset(k_buffer[layer_id], layer_id * 1000)
        fill_with_offset(v_buffer[layer_id], layer_id * 1000 + 100)

    return SimpleNamespace(
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [x.data_ptr() for x in k_buffer], dtype=torch.uint64, device=DEVICE
        ),
        v_data_ptrs=torch.tensor(
            [x.data_ptr() for x in v_buffer], dtype=torch.uint64, device=DEVICE
        ),
    )


def _host_k_tokens(host, indices, layer_id):
    if host.layout == "layer_first":
        return host.k_buffer[layer_id, indices]
    if host.layout == "page_first":
        return host.k_buffer[indices, layer_id]
    pages = indices[::PAGE_SIZE] // PAGE_SIZE
    return host.k_buffer[pages, layer_id].reshape(-1, HEAD_NUM, K_HEAD_DIM)


def _host_v_tokens(host, indices, layer_id):
    if host.layout == "layer_first":
        return host.v_buffer[layer_id, indices]
    if host.layout == "page_first":
        return host.v_buffer[indices, layer_id]
    pages = indices[::PAGE_SIZE] // PAGE_SIZE
    return host.v_buffer[pages, layer_id].reshape(-1, HEAD_NUM, V_HEAD_DIM)


def assert_backup_matches_device(host, device_pool, host_indices_host, device_indices):
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            _host_k_tokens(host, host_indices_host, layer_id),
            device_pool.k_buffer[layer_id][device_indices].cpu(),
        )
        torch.testing.assert_close(
            _host_v_tokens(host, host_indices_host, layer_id),
            device_pool.v_buffer[layer_id][device_indices].cpu(),
        )


def assert_load_matches_host(host, device_pool, host_indices_host, load_indices):
    for layer_id in range(NUM_LAYERS):
        torch.testing.assert_close(
            device_pool.k_buffer[layer_id][load_indices],
            _host_k_tokens(host, host_indices_host, layer_id).to(DEVICE),
        )
        torch.testing.assert_close(
            device_pool.v_buffer[layer_id][load_indices],
            _host_v_tokens(host, host_indices_host, layer_id).to(DEVICE),
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_asymmetric_mha_kernel_page_first_roundtrip(dtype):
    # Covers D2H backup + H2D load through AsymmetricMHATokenToKVPoolHost using
    # MiMoV2's real K/V head dims and the real MLA single-buffer kernels.
    host = make_host_pool(dtype)
    device_pool = make_device_pool(dtype)

    device_pages = torch.tensor([1, 2, 3], dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], dtype=torch.int64)
    load_pages = torch.tensor([4, 5, 6], dtype=torch.int64)
    device_indices_host = token_indices_for_pages(device_pages)
    host_indices_host = token_indices_for_pages(host_pages)
    load_indices_host = token_indices_for_pages(load_pages)
    device_indices = device_indices_host.to(DEVICE)
    host_indices = host_indices_host.to(DEVICE)
    load_indices = load_indices_host.to(DEVICE)

    host.backup_from_device_all_layer(
        device_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    assert_backup_matches_device(
        host, device_pool, host_indices_host, device_indices_host
    )

    for layer_id in range(NUM_LAYERS):
        device_pool.k_buffer[layer_id].zero_()
        device_pool.v_buffer[layer_id].zero_()
        host.load_to_device_per_layer(
            device_pool, host_indices, load_indices, layer_id, io_backend="kernel"
        )
    torch.cuda.synchronize()
    assert_load_matches_host(host, device_pool, host_indices_host, load_indices_host)


@pytest.mark.parametrize("dtype", DTYPES)
def test_asymmetric_mha_direct_page_first_direct_roundtrip(dtype):
    # Covers D2H backup + H2D load through AsymmetricMHATokenToKVPoolHost using
    # page_first_direct/direct. K and V are copied through separate direct calls
    # because their per-token strides differ.
    host = make_host_pool(dtype, layout="page_first_direct")
    device_pool = make_device_pool(dtype)
    direct_stream = torch.cuda.Stream()

    device_pages = torch.tensor([1, 2, 3], dtype=torch.int64)
    host_pages = torch.tensor([0, 1, 2], dtype=torch.int64)
    load_pages = torch.tensor([4, 5, 6], dtype=torch.int64)
    device_indices_host = token_indices_for_pages(device_pages)
    host_indices_host = token_indices_for_pages(host_pages)
    load_indices_host = token_indices_for_pages(load_pages)

    with torch.cuda.stream(direct_stream):
        host.backup_from_device_all_layer(
            device_pool, host_indices_host, device_indices_host, io_backend="direct"
        )
    direct_stream.synchronize()
    assert_backup_matches_device(
        host, device_pool, host_indices_host, device_indices_host
    )

    with torch.cuda.stream(direct_stream):
        for layer_id in range(NUM_LAYERS):
            device_pool.k_buffer[layer_id].zero_()
            device_pool.v_buffer[layer_id].zero_()
            host.load_to_device_per_layer(
                device_pool,
                host_indices_host,
                load_indices_host,
                layer_id,
                io_backend="direct",
            )
    direct_stream.synchronize()
    assert_load_matches_host(host, device_pool, host_indices_host, load_indices_host)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", ["kernel", "jit", "direct"])
def test_asymmetric_mha_layer_first_roundtrip(dtype, backend):
    host = make_host_pool(dtype, layout="layer_first")
    host.can_use_jit = backend == "jit"
    io_backend = "kernel" if backend == "jit" else backend
    pool = make_device_pool(dtype)
    ptrs, sizes, items = host.get_contiguous_buf_infos()
    assert ptrs == [x.data_ptr() for x in host.k_data_refs + host.v_data_refs]
    assert (
        sizes
        == [TOTAL_ITEMS * HEAD_NUM * K_HEAD_DIM * dtype.itemsize] * NUM_LAYERS
        + [TOTAL_ITEMS * HEAD_NUM * V_HEAD_DIM * dtype.itemsize] * NUM_LAYERS
    )
    assert (
        items
        == [PAGE_SIZE * HEAD_NUM * K_HEAD_DIM * dtype.itemsize] * NUM_LAYERS
        + [PAGE_SIZE * HEAD_NUM * V_HEAD_DIM * dtype.itemsize] * NUM_LAYERS
    )
    # Noncontiguous, permuted pages catch wrong offsets and request mixing.
    src = token_indices_for_pages(torch.tensor([3, 1, 6]))
    dst = token_indices_for_pages(torch.tensor([5, 0, 2]))
    load = token_indices_for_pages(torch.tensor([2, 7, 4]))
    indices_device = "cpu" if io_backend == "direct" else DEVICE
    src_i, dst_i, load_i = [x.to(indices_device) for x in (src, dst, load)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        host.backup_from_device_all_layer(pool, dst_i, src_i, io_backend)
    stream.synchronize()
    assert_backup_matches_device(host, pool, dst, src)
    for buf in pool.k_buffer + pool.v_buffer:
        buf.fill_(-1)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in range(NUM_LAYERS):
            host.load_to_device_per_layer(pool, dst_i, load_i, layer, io_backend)
    stream.synchronize()
    assert_load_matches_host(host, pool, dst, load)
    untouched = torch.ones(TOTAL_ITEMS, dtype=torch.bool)
    untouched[load] = False
    for buf in pool.k_buffer + pool.v_buffer:
        assert torch.all(buf[untouched] == -1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
