"""Real H20/FA3 window boundaries and five-layer DFlash KV materialization."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="DFlash legacy FA3 tests require a Hopper CUDA device",
)


def _config():
    return Qwen3Config(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=5,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
        dflash_config={
            "target_layer_ids": [0, 11, 23, 35, 47],
            "block_size": 8,
            "use_swa": True,
            "swa_window_size": 1024,
            "attention_value_scale": 0.612,
            "attention_sink_bias": True,
        },
    )


def _pool(dtype, size=1280):
    return MHATokenToKVPool(
        size=size,
        page_size=64,
        dtype=dtype,
        head_num=2,
        head_dim=64,
        layer_num=5,
        device="cuda",
        enable_memory_saver=False,
    )


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def draft(request):
    with (
        get_context().override_server_args(
            tp_size=1,
            dp_size=1,
            page_size=64,
            attention_backend="fa3",
            speculative_algorithm="DFLASH",
            speculative_num_draft_tokens=8,
            speculative_eagle_topk=1,
            speculative_draft_attention_backend="fa3",
        ),
        get_parallel().override(tp_rank=0, attn_tp_rank=0),
    ):
        torch.manual_seed(53)
        # Follow the loader's dtype scope; casting the complete module would
        # incorrectly downcast its FP32 RoPE cache as well.
        with set_default_torch_dtype(request.param):
            model = DFlashDraftModel(_config()).to(device="cuda")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.fill_(1)
                else:
                    param.normal_(std=0.03)
        yield model


@pytest.mark.parametrize("prefix", [1023, 1024, 1088])
@pytest.mark.parametrize("implementation", ["kernel", "backend"])
def test_fa3_legacy_window_matches_explicit_bidirectional_mask(
    draft, prefix, implementation
):
    from sgl_kernel import flash_attn as fa3

    if implementation == "backend" and not hasattr(fa3, "get_scheduler_metadata"):
        pytest.skip("The installed sgl-kernel predates the FA3 backend scheduler API")
    length, width = prefix + 8, 8
    dtype = draft.layers[0].self_attn.qkv_proj.weight.dtype
    pool = _pool(dtype)
    for buffer in (*pool.k_buffer, *pool.v_buffer):
        buffer.zero_()
    # The left-edge sentinel distinguishes 1024 from 1023. The last draft
    # token distinguishes bidirectional attention from a causal draft mask.
    edge = max(prefix - 1024, 0)
    pool.v_buffer[0][edge, :, 0] = 1024
    pool.v_buffer[0][length - 1, :, 1] = 128
    req_pool = SimpleNamespace(
        size=1,
        req_to_token=torch.arange(1280, device="cuda")[None],
    )
    mc = SimpleNamespace(
        is_encoder_decoder=False,
        context_len=1280,
        attention_arch=AttentionArch.MHA,
        is_local_attention_model=False,
        head_dim=64,
        hf_text_config=SimpleNamespace(num_attention_heads=2),
        get_num_kv_heads=lambda tp: 2,
    )
    runner = SimpleNamespace(
        sliding_window_size=draft.get_attention_sliding_window_size(),
        model_config=mc,
        device="cuda",
        req_to_token_pool=req_pool,
        token_to_kv_pool=pool,
        kv_cache_dtype=dtype,
        kv_cache_dtype_str="auto",
        page_size=64,
        ps=SimpleNamespace(attn_cp_size=1, tp_size=1),
        is_draft_worker=True,
        kv_index_translator=KVIndexTranslator(
            req_to_token=req_pool.req_to_token,
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=object(),
            page_size=64,
            device="cuda",
        ),
    )
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    metadata = FlashAttentionMetadata(
        cache_seqlens_int32=lengths,
        max_seq_len_q=width,
        max_seq_len_k=length,
        cu_seqlens_q=torch.tensor([0, width], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        page_table=torch.arange(20, device="cuda", dtype=torch.int32)[None],
    )
    positions = torch.arange(prefix, length, device="cuda")
    batch = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=1,
        input_ids=torch.zeros(width, device="cuda", dtype=torch.long),
        req_pool_indices=torch.tensor([0], device="cuda"),
        seq_lens=lengths,
        out_cache_loc=positions,
        seq_lens_sum=length,
        positions=positions,
    )
    q = torch.zeros(width, 128, device="cuda", dtype=dtype)
    sinks = draft.layers[0].self_attn.attention_sink_bias
    with torch.no_grad():
        sinks.copy_(torch.tensor([0.5, -0.5], device="cuda"))
    layer = draft.layers[0].self_attn.attn
    if implementation == "backend":
        backend = FlashAttentionBackend(runner)
        backend.forward_metadata = metadata
        output = backend.forward_extend(
            q,
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
            sinks=sinks,
        ).reshape(width, 2, 64)
    else:
        # Also exercise the native FA3 operation independently of the backend
        # constructor, so older CUDA test environments can check the model's
        # window units and causality against the same explicit mask oracle.
        causal = layer.attn_type == AttentionType.DECODER
        output = fa3.flash_attn_with_kvcache(
            q=q.reshape(width, 2, 64),
            k_cache=pool.k_buffer[0].reshape(-1, 64, 2, 64),
            v_cache=pool.v_buffer[0].reshape(-1, 64, 2, 64),
            cache_seqlens=lengths,
            page_table=metadata.page_table,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=width,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=(
                layer.sliding_window_size,
                0 if causal else layer.sliding_window_size,
            ),
            sinks=sinks,
        ).reshape(width, 2, 64)
    keys = torch.arange(length, device="cuda")
    mask = (keys[None] >= positions[:, None] - 1024) & (
        keys[None] <= positions[:, None] + 1024
    )
    values = pool.v_buffer[0][:length].float()
    reference = torch.einsum("qk,khd->qhd", mask.float(), values)
    reference /= mask.sum(-1)[:, None, None] + sinks.float().exp()[None, :, None]
    torch.testing.assert_close(
        output.float(),
        reference,
        atol=1e-3,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
    )


def test_five_layer_kv_projection_matches_full_qkv_and_drops_padding(draft):
    dtype = draft.layers[0].self_attn.qkv_proj.weight.dtype
    pool = _pool(dtype)
    for buffer in (*pool.k_buffer, *pool.v_buffer):
        buffer.fill_(17)
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker._target_worker = SimpleNamespace()
    worker.model_runner = SimpleNamespace(device=torch.device("cuda"))
    worker.draft_model_runner = SimpleNamespace(tp_group=None, token_to_kv_pool=pool)
    worker.draft_model = draft
    worker.draft_tp_context = nullcontext
    worker._use_fused_kv_materialize = False
    worker._fused_kv_helper = None
    hidden = torch.randn(196, 5 * 128, device="cuda", dtype=dtype)
    positions = torch.arange(193, device="cuda", dtype=torch.int64)
    slots = torch.cat(
        [torch.arange(p * 64, (p + 1) * 64, device="cuda") for p in (5, 1, 3, 7)]
    )[:193]
    worker._append_target_hidden_to_draft_kv_by_loc(
        target_hidden=hidden,
        cache_loc=slots,
        positions=positions,
    )
    with torch.inference_mode():
        context = draft.project_target_hidden(hidden[:193])
        for index, layer in enumerate(draft.layers):
            attn = layer.self_attn
            packed = F.linear(context, attn.qkv_proj.weight, attn.qkv_proj.bias)
            _, key, value = packed.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            key = (
                F.rms_norm(
                    key.reshape(-1, 64).float(),
                    (64,),
                    attn.k_norm.weight.float(),
                    attn.k_norm.variance_epsilon,
                )
                .to(key.dtype)
                .reshape_as(key)
            )
            # The CUDA kernel rotates in FP32 before casting back. The native
            # helper follows its input dtype, so pass FP32 to avoid introducing
            # extra BF16 rounding of both trigonometric values and products.
            key_fp32 = key.float()
            _, key = attn.rotary_emb.forward_native(
                positions, torch.zeros_like(key_fp32), key_fp32
            )
            key = key.to(dtype)
            tolerance = 1e-2 if dtype == torch.bfloat16 else 3e-3
            torch.testing.assert_close(
                pool.k_buffer[index][slots],
                key.reshape(-1, 2, 64),
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                pool.v_buffer[index][slots],
                (value * 0.612).reshape(-1, 2, 64),
                atol=tolerance,
                rtol=tolerance,
            )
            assert torch.all(pool.k_buffer[index][0] == 17)
    torch.cuda.synchronize()
