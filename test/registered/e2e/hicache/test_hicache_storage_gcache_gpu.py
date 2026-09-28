"""Real CUDA -> host -> network KV -> host -> CUDA byte round trips.

Guards the boundary between GCache's page metadata and HiCache's physical
layouts: a wrong K/V length, layer stride, pool, or page address must change
the restored device bytes. CUDA allocation, host registration, L2 kernels,
streams and events are real. The fixture supplies standalone rank coordinates
and replaces the external SDK transport with the network client used by the
independent in-memory GCache mockserver.
"""

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gcache_test_utils import backend_factory as backend_factory

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GCache device round trips require CUDA."
)


def _indices(pages, page_size):
    return torch.tensor(
        [
            token
            for page in pages
            for token in range(page * page_size, (page + 1) * page_size)
        ],
        dtype=torch.int64,
    )


def _make_pools(layout, dtype, name, asymmetric, host_pools, page_size, draft_layers):
    layers, heads, k_dim, v_dim = {
        PoolName.KV: (3, 2, 192 if asymmetric else 128, 128),
        PoolName.SWA: (2, 1, 192, 128),
        PoolName.DRAFT: (draft_layers, 2, 64, 64),
        PoolName.DRAFT_SWA: (draft_layers, 1, 64, 32),
    }[name]
    device = MHATokenToKVPool(
        size=page_size * 8,
        page_size=page_size,
        dtype=dtype,
        head_num=heads,
        head_dim=k_dim,
        v_head_dim=v_dim,
        layer_num=layers,
        device="cuda",
        enable_memory_saver=False,
        kv_cache_layout="nhd",
    )
    cls = AsymmetricMHATokenToKVPoolHost if k_dim != v_dim else MHATokenToKVPoolHost
    host = cls(
        device,
        host_to_device_ratio=2,
        host_size=0,
        page_size=page_size,
        layout=layout,
        pin_memory=True,
        device="cpu",
        pool_label=name,
    )
    host_pools.append(host)
    assert host.k_buffer.is_pinned() and host.v_buffer.is_pinned()
    if layout == "page_first":
        assert host.can_use_write_back_jit, "The supported staged CUDA kernel must load"
    for buffer in (*device.k_buffer, *device.v_buffer, host.k_buffer, host.v_buffer):
        buffer.zero_()
    return device, host


def _transfer(pair, host_indices, device_indices, *, io_backend, backup):
    device, host = pair
    # Staged D2H takes CPU destination indices for cudaMemcpyBatchAsync.
    # Kernel H2D takes CUDA indices; direct I/O takes CPU indices both ways.
    cpu_host_indices = io_backend == "direct" or (
        backup and host.can_use_write_back_jit
    )
    return L2Transfer(
        host,
        device,
        host_indices if cpu_host_indices else host_indices.cuda(),
        device_indices if io_backend == "direct" else device_indices.cuda(),
    )


def _fill_device(pool, seed):
    for component, buffers in enumerate((pool.k_buffer, pool.v_buffer)):
        for layer, buffer in enumerate(buffers):
            data = torch.arange(buffer.numel(), device="cuda", dtype=torch.int64)
            data = (data + seed + component * 71 + layer * 19).remainder(251)
            buffer.copy_((data.float() / 16).reshape_as(buffer))


def _expected_objects(expected, layout, page_size):
    """Pack the original GPU values by the documented layout, without host metadata."""
    objects = {}
    for name, components in expected.items():
        model, kind = {
            PoolName.KV: ("model", "full"),
            PoolName.SWA: ("model", "swa"),
            PoolName.DRAFT: ("d:model", "full"),
            PoolName.DRAFT_SWA: ("d:model", "swa"),
        }[name]
        for page in range(3):
            for component, data in zip(("k", "v"), components):
                value = data[:, page * page_size : (page + 1) * page_size]
                if layout == "page_first":
                    value = value.transpose(0, 1)
                objects[f"{model}_h{page}_1_{kind}_{component}"] = (
                    value.contiguous().view(torch.uint8).numpy().tobytes()
                )
    return objects


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("asymmetric", [False, True], ids=["mha", "asymmetric-mha"])
@pytest.mark.parametrize("draft_name", [PoolName.DRAFT, PoolName.DRAFT_SWA])
@pytest.mark.parametrize(
    "page_size,draft_layers", [(16, 1), (64, 5)], ids=["small", "dflash-five-layer"]
)
def test_device_bytes_survive_gcache_restore(
    backend_factory, layout, dtype, asymmetric, draft_name, page_size, draft_layers
):
    io_backend = "kernel" if layout == "page_first" else "direct"
    names = (PoolName.KV, PoolName.SWA, draft_name)
    host_pools = []
    writer = reader = None
    with get_context().override_server_args(
        enable_hierarchical_cache=True,
        hicache_storage_backend="gcache",
        hicache_mem_layout=layout,
        hicache_io_backend=io_backend,
        page_size=page_size,
    ):
        try:
            sources = {
                name: _make_pools(
                    layout, dtype, name, asymmetric, host_pools, page_size, draft_layers
                )
                for name in names
            }
            targets = {
                name: _make_pools(
                    layout, dtype, name, asymmetric, host_pools, page_size, draft_layers
                )
                for name in names
            }
            writer = backend_factory(sources[PoolName.KV][1])
            reader = backend_factory(targets[PoolName.KV][1])
            for name in names[1:]:
                writer.register_mem_host_pool_v2(sources[name][1], name)
                reader.register_mem_host_pool_v2(targets[name][1], name)

            device_source = _indices([5, 1, 3], page_size)
            host_source = _indices([0, 2, 4], page_size)
            host_target = _indices([1, 3, 5], page_size)
            device_target = _indices([4, 0, 6], page_size)
            engine = L2TransferEngine(io_backend)
            producer = torch.cuda.Stream()
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                for seed, name in enumerate(names):
                    _fill_device(sources[name][0], 37 * seed + 11)
                completion = engine.submit_device_to_host(
                    [
                        _transfer(
                            sources[name],
                            host_source,
                            device_source,
                            io_backend=io_backend,
                            backup=True,
                        )
                        for name in names
                    ]
                )
            completion.finish_event.synchronize()
            expected = {
                name: [
                    torch.stack([buffer[device_source].cpu() for buffer in buffers])
                    for buffers in (
                        sources[name][0].k_buffer,
                        sources[name][0].v_buffer,
                    )
                ]
                for name in names
            }

            keys = ["h0", "h1", "h2"]
            assert writer.batch_set_v1(keys, host_source) == [True] * 3
            transfers = [
                PoolTransfer(name, host_source, keys=keys) for name in names[1:]
            ]
            assert writer.batch_set_v2(transfers) == {
                name: [True] * 3 for name in names[1:]
            }
            assert backend_factory.server.objects == _expected_objects(
                expected, layout, page_size
            )

            # The reader must recover from the server after all writer buffers
            # are erased and the writing backend has detached.
            for device, host in sources.values():
                for buffer in (
                    *device.k_buffer,
                    *device.v_buffer,
                    host.k_buffer,
                    host.v_buffer,
                ):
                    buffer.zero_()
            writer.close()
            query_transfers = [
                PoolTransfer(
                    name,
                    keys=["window"] * 2,
                    hit_policy=PoolHitPolicy.ALL_PAGES
                    if name == PoolName.DRAFT
                    else PoolHitPolicy.TRAILING_PAGES,
                )
                for name in names[1:]
            ]
            assert reader.batch_exists_v2(
                keys, query_transfers
            ).restorable_prefix_pages == [1, 2, 3]
            assert reader.batch_get_v1(keys, host_target) == [True] * 3
            assert reader.batch_get_v2(
                [PoolTransfer(name, host_target, keys=keys) for name in names[1:]]
            ) == {name: [True] * 3 for name in names[1:]}

            completion = engine.submit_host_to_device(
                [
                    _transfer(
                        targets[name],
                        host_target,
                        device_target,
                        io_backend=io_backend,
                        backup=False,
                    )
                    for name in names
                ],
                layer_num=max(targets[name][1].layer_num for name in names),
            )
            completion.finish_event.synchronize()
            for name in names:
                device = targets[name][0]
                for component, buffers in enumerate((device.k_buffer, device.v_buffer)):
                    for layer, buffer in enumerate(buffers):
                        restored = buffer[device_target].cpu()
                        assert torch.equal(
                            restored.view(torch.uint8),
                            expected[name][component][layer].view(torch.uint8),
                        )
                        untouched = torch.ones(buffer.shape[0], dtype=torch.bool)
                        untouched[device_target] = False
                        assert torch.count_nonzero(buffer[untouched]).item() == 0
        finally:
            torch.cuda.synchronize()
            for backend in (reader, writer):
                if backend is not None:
                    backend.close()
            for host in host_pools:
                host.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
