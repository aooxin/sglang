"""Real TP4/DP2 collectives for DFlash; no model checkpoint is required.

Exercises the production TP context, embedding cache, vocabulary sampler and
spec broadcasts with one DP idle and with unequal active batch sizes. This is
a distributed component test, not a full MiMo/DeepEP/P-to-D serving test.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-h100")

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="DFlash TP4/DP2 requires four CUDA devices"
)


def _run_dp_rank(rank, rendezvous):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_attn_tp_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.srt.speculative.dflash_worker_v2 import (
        DFlashWorkerV2,
        _DflashDraftSampler,
    )
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
    from sglang.srt.speculative.spec_utils import draft_tp_context

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    with get_context().override_server_args(
        device="cuda",
        tp_size=4,
        dp_size=2,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        disable_custom_all_reduce=True,
    ):
        set_custom_all_reduce(False)
        init_distributed_environment(
            world_size=4,
            rank=rank,
            local_rank=rank,
            distributed_init_method=rendezvous,
            backend="nccl",
            timeout=60,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=4,
            attention_data_parallel_size=2,
            backend="nccl",
        )
        completed = False
        try:
            attn_group = get_attn_tp_group()
            dp_rank, local_rank = divmod(rank, 2)
            assert attn_group.ranks == [dp_rank * 2, dp_rank * 2 + 1]
            sync = SpecTpSync(attn_group)

            # Distinct per-DP sentinel weights catch accidental gathering from
            # the peer DP group even when real replicas normally share weights.
            for vocab in (128, 133):
                embedding = VocabParallelEmbedding(
                    vocab,
                    16,
                    use_attn_tp_group=True,
                    params_dtype=torch.float32,
                ).to(device)
                full = torch.arange(vocab * 16, device=device, dtype=torch.float32)
                full = full.reshape(vocab, 16) + dp_rank * 10000
                shard = embedding.shard_indices
                with torch.no_grad():
                    embedding.weight.fill_(-999)
                    embedding.weight[: shard.num_org_elements].copy_(
                        full[shard.org_vocab_start_index : shard.org_vocab_end_index]
                    )
                worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
                worker.ps = SimpleNamespace(tp_rank=rank)
                worker._full_embed_gpu = None
                worker._target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                        model_config=SimpleNamespace(vocab_size=vocab),
                    )
                )
                worker._cache_full_embed_weight()
                torch.testing.assert_close(worker._full_embed_gpu, full, rtol=0, atol=0)

            generator = torch.Generator(device=device).manual_seed(13)
            weight = torch.randn(128, 16, device=device, generator=generator)
            sampler = _DflashDraftSampler(
                weight=weight[local_rank * 64 : (local_rank + 1) * 64].contiguous(),
                block_size=8,
                num_org=64,
                org_vocab_start=local_rank * 64,
                max_bs=3,
                tp_group=attn_group,
            )
            for step, batches in enumerate(((1, 0), (1, 3), (0, 2), (2, 1))):
                bs = batches[dp_rank]
                if bs:
                    with draft_tp_context(attn_group):
                        assert get_parallel().tp_size == 2
                        assert get_parallel().tp_rank == local_rank
                        assert get_tp_group() is attn_group
                        generator.manual_seed(100 + step * 7 + dp_rank)
                        hidden = torch.randn(
                            bs * 8, 16, device=device, generator=generator
                        )
                        sampler(hidden)
                        expected = (
                            hidden.reshape(bs, 8, 16)[:, 1:].reshape(-1, 16) @ weight.T
                        ).argmax(-1)
                        torch.testing.assert_close(sampler.out[: bs * 7], expected)
                        value = torch.tensor([rank], device=device)
                        sync.sync(SpecTpSyncSite.DFLASH_TARGET, value)
                        assert value.item() == dp_rank * 2
                # The target still requires every rank, including idle DP.
                assert get_parallel().tp_size == 4
                value = torch.tensor([rank + 1.0], device=device)
                dist.all_reduce(value, group=get_tp_group().device_group)
                assert value.item() == 10
            torch.cuda.synchronize()
            completed = True
        finally:
            # On failure let mp.spawn terminate peers immediately; collective
            # teardown could otherwise hide the assertion behind a timeout.
            if completed:
                destroy_model_parallel()
                destroy_distributed_environment()


def test_dflash_dp_subgroups_with_idle_and_unequal_batches(tmp_path):
    mp.spawn(_run_dp_rank, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4)
