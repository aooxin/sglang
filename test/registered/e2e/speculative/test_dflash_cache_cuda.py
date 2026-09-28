"""DFlash cache lifetimes and PD handoff primitives on real CUDA buffers.

These tests do not load production weights or replace a full two-role PD run.
GCache's actual socket round trip is covered by the HiCache GPU test suite.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from sglang.srt.configs.model_config import is_hybrid_swa_model
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
    setup_state_kv_args,
)
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_hicache_draft_sidecars,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode
from sglang.srt.speculative.dflash_disaggregation import build_dflash_disagg_draft_input
from sglang.srt.speculative.dflash_utils import get_dflash_layer_types
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlash cache handoff tests require CUDA"
)


def _mha_pool(size):
    return MHATokenToKVPool(
        size=size,
        page_size=64,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=5,
        device="cuda",
        enable_memory_saver=False,
        kv_cache_layout="nhd",
    )


def _allocate(allocator, tokens):
    prefix_cpu = torch.tensor([0], dtype=torch.int64)
    length_cpu = torch.tensor([tokens], dtype=torch.int64)
    slots = allocator.alloc_extend(
        prefix_cpu.cuda(),
        prefix_cpu,
        length_cpu.cuda(),
        length_cpu,
        torch.tensor([-1], dtype=torch.int64, device="cuda"),
        tokens,
    )
    assert slots is not None
    return slots


def test_full_draft_sidecar_survives_target_swa_eviction_and_reuse():
    cfg = Qwen3Config(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=5,
        dflash_config={"use_swa": True, "swa_window_size": 1024},
    )
    assert get_dflash_layer_types(cfg) == ["sliding_attention"] * 5
    # The attention mask does not opt this checkpoint into shared SWA storage.
    assert not is_hybrid_swa_model(cfg.architectures, cfg)
    with get_context().override_server_args(
        enable_hierarchical_cache=True,
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        page_size=64,
        speculative_algorithm="DFLASH",
    ):
        target = SWAKVPool(
            size=40 * 64,
            size_swa=24 * 64,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            device="cuda",
            kv_cache_layout="nhd",
        )
        draft = _mha_pool(target.size)
        allocator = SWATokenToKVPoolAllocator(
            target.size,
            target.size_swa,
            64,
            torch.bfloat16,
            "cuda",
            target,
            True,
        )
        slots = _allocate(allocator, 18 * 64)
        expected = []
        for layer, (k, v) in enumerate(zip(draft.k_buffer, draft.v_buffer)):
            data = torch.arange(slots.numel(), device="cuda").remainder(31).to(k.dtype)
            k[slots] = (data + layer * 32)[:, None, None]
            v[slots] = (data + layer * 32 + 7)[:, None, None]
            expected.append((k[slots[-1024:]].clone(), v[slots[-1024:]].clone()))
        # Target keeps only the last 384 tokens here, shorter than the draft's
        # 1024-token mask. Its freed SWA pages really are reused by another req.
        evicted = slots[:-384]
        freed_swa = allocator.translate_loc_from_full_to_swa(evicted).clone()
        full_available = allocator.full_available_size()
        allocator.free_swa(evicted)
        assert allocator.full_available_size() == full_available
        assert (
            torch.count_nonzero(allocator.translate_loc_from_full_to_swa(evicted)) == 0
        )
        other = _allocate(allocator, 12 * 64)
        other_swa = allocator.translate_loc_from_full_to_swa(other)
        assert torch.isin(other_swa, freed_swa).any()
        for buffer in (*target.swa_kv_pool.k_buffer, *target.swa_kv_pool.v_buffer):
            buffer[other_swa] = -19
        for layer, (k, v) in enumerate(zip(draft.k_buffer, draft.v_buffer)):
            torch.testing.assert_close(
                k[slots[-1024:]], expected[layer][0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                v[slots[-1024:]], expected[layer][1], rtol=0, atol=0
            )

        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
            )
        )
        worker._draft_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=draft,
                model_config=SimpleNamespace(hf_config=cfg),
            )
        )
        plan = worker._build_hicache_draft_plan()
        assert plan.mode == HiCacheDraftMode.SIDECAR
        assert plan.device_pools == (draft,)
        assert worker.primary_draft_kv_pool is draft
        tree = SimpleNamespace(
            cache_controller=SimpleNamespace(
                page_size=64,
                mem_pool_host=SimpleNamespace(logical_size=2 * target.size),
            )
        )
        specs, entries = build_hicache_draft_sidecars(
            draft_device_pools=plan.device_pools,
            tree_cache=tree,
        )
        try:
            assert len(specs) == len(entries) == 1
            assert specs[0].pool_name == PoolName.DRAFT
            assert specs[0].indices_from_pool == PoolName.KV
            assert specs[0].hit_policy == PoolHitPolicy.ALL_PAGES
            assert entries[0].host_pool.layer_num == 5
            assert entries[0].device_pool is draft
            assert entries[0].host_pool.size >= target.size

            # The PD path uses all ten draft K/V entries in FULL index space;
            # target SWA remains a distinct state component with its own ids.
            ptrs, lens, item_lens = draft.get_contiguous_buf_infos()
            buffers = draft.k_buffer + draft.v_buffer
            assert len(ptrs) == len(lens) == len(item_lens) == 10
            assert ptrs == [buffer.data_ptr() for buffer in buffers]
            assert lens == [buffer.nbytes for buffer in buffers]
            assert item_lens == [64 * 2 * 64 * 2] * 10
            decoder_draft = _mha_pool(8 * 64)
            dst_ptrs, dst_lens, dst_items = decoder_draft.get_contiguous_buf_infos()
            assert dst_items == item_lens and dst_lens != lens
            assert set(ptrs).isdisjoint(dst_ptrs)
            assert (
                build_kv_layer_ids(
                    token_to_kv_pool=target,
                    draft_token_to_kv_pool=draft,
                    num_draft_entries=10,
                    num_hidden_layers=4,
                )
                == []
            )
            assert build_transfer_entry_pairs([], [], 14, 14) == [
                (i, i) for i in range(14)
            ]
            kv_args = KVArgs()
            setup_state_kv_args(kv_args, target, draft, total_kv_layers=4)
            assert kv_args.state_types == [StateType.SWA]
            assert kv_args.state_data_ptrs == [target.get_state_buf_infos()[0]]
            assert set(ptrs).isdisjoint(kv_args.state_data_ptrs[0])
        finally:
            torch.cuda.synchronize()
            for entry in entries:
                entry.host_pool.destroy()


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_pd_draft_input_preserves_lengths_and_bonus_tokens(overlap, count):
    with get_context().override_server_args(speculative_algorithm="DFLASH"):
        req_pool = ReqToTokenPool(8, 2048, "cuda", enable_memory_saver=False)
        future = FutureMap(
            device=torch.device("cuda"),
            spec_algo=SpeculativeAlgorithm.DFLASH,
            req_to_token_pool=req_pool,
        )
        indices = torch.tensor([5, 2, 7][:count], dtype=torch.int64, device="cuda")
        lengths = torch.tensor(
            [1023, 1024, 1088][:count], device="cuda", dtype=torch.int64
        )
        bonus = torch.tensor([31, 97, 103][:count], device="cuda", dtype=torch.int64)
        batch = SimpleNamespace(
            seq_lens=lengths,
            enable_overlap=overlap,
            req_pool_indices=indices,
            req_pool_indices_cpu=indices.cpu(),
        )
        spec = build_dflash_disagg_draft_input(batch, bonus, future)
        torch.testing.assert_close(spec.new_seq_lens, lengths)
        torch.testing.assert_close(spec.bonus_tokens, bonus)
        assert spec.hidden_states.shape == (count, 0)
        if not overlap:
            assert spec.future_indices is None
            assert future.publish_ready is None
        else:
            assert torch.equal(spec.future_indices, indices)
            if count:
                # Resolve through the actual GPU FutureMap path in a different
                # order, as filtering/rebatching on decode can reorder requests.
                order = torch.arange(count - 1, -1, -1, device="cuda")
                spec.future_indices = indices[order]
                batch.req_pool_indices = spec.future_indices
                batch.req_pool_indices_cpu = batch.req_pool_indices.cpu()
                batch.spec_info = spec
                future.resolve_seq_lens_cpu(batch)
                future._resolve_spec_extras(batch)
                torch.testing.assert_close(batch.seq_lens, lengths[order])
                torch.testing.assert_close(batch.seq_lens_cpu, lengths[order].cpu())
                torch.testing.assert_close(spec.bonus_tokens, bonus[order])
                assert batch.seq_lens_sum == int(lengths.sum())
                assert not future.need_topk and not future.need_hidden_states
            else:
                assert future.publish_ready is None
        torch.cuda.synchronize()
