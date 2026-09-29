"""MiMo checkpoint bytes and SM90 DeepGEMM scale layouts must agree."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import fp8
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.srt.runtime_context import get_context, get_flags, get_parallel


@pytest.fixture(autouse=True)
def single_rank_context():
    with (
        get_context().override_server_args(
            tp_size=1, moe_runner_backend="triton", moe_a2a_backend="none"
        ),
        get_parallel().override(tp_size=1),
    ):
        yield


def _load_experts(*, serialized_e8m0=True, scale_dtype=None):
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    if serialized_e8m0:
        config["store_dtype"] = "mxfp4"
    quant = Fp8Config.from_config(config)
    quant.is_fp4_experts = True

    experts = object.__new__(FusedMoE)
    nn.Module.__init__(experts)
    experts.moe_runner_config = SimpleNamespace(is_gated=True)
    experts.quant_config = quant
    experts.quant_method = Fp8MoEMethod(quant)
    experts.scheme = None
    experts.moe_tp_size = 1
    experts.moe_tp_rank = 0
    experts.use_flashinfer_trtllm_moe = False
    experts.use_triton_kernels = False
    experts.use_padded_loading = False
    experts.use_presharded_weights = False
    experts._has_fused_shared = False
    experts.quant_method.create_weights(
        experts,
        num_experts=1,
        hidden_size=128,
        intermediate_size_per_partition=128,
        params_dtype=torch.bfloat16,
        fp4_scale_dtype=scale_dtype,
        weight_loader=experts._weight_loader_impl,
    )

    model = object.__new__(MiMoV2ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=1, encoder_only=False, tie_word_embeddings=False
    )
    model._is_multimodal = False
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = experts
    model.model.layers = nn.ModuleList([layer])

    scales = (
        torch.tensor([126, 127, 128, 129], dtype=torch.uint8)
        if serialized_e8m0
        else torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    ).repeat(128, 1)
    packed = torch.full((128, 64), 170, dtype=torch.uint8)
    weights = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"model.layers.0.mlp.experts.0.{projection}"
        weights.extend([(f"{prefix}.weight", packed), (f"{prefix}.weight_scale", scales)])
    model.load_weights(weights)
    return experts


def test_mimo_loader_preserves_e8m0_exponents_and_packed_weights():
    experts = _load_experts()
    expected = torch.tensor([0.5, 1.0, 2.0, 4.0]).repeat(256, 1)
    assert experts.w13_weight_scale_inv.dtype == torch.uint8
    torch.testing.assert_close(
        experts.w13_weight_scale_inv[0].view(torch.float8_e8m0fnu).float(), expected
    )
    assert torch.all(experts.w13_weight.view(torch.uint8) == 170)
    assert torch.all(experts.w2_weight.view(torch.uint8) == 170)


def test_other_fp4_checkpoints_keep_numeric_fp32_scales():
    experts = _load_experts(serialized_e8m0=False)
    assert experts.w13_weight_scale_inv.dtype == torch.float32
    torch.testing.assert_close(
        experts.w13_weight_scale_inv[0, 0], torch.tensor([0.5, 1.0, 2.0, 4.0])
    )


@pytest.mark.parametrize(
    "a2a,mode,needs_companion",
    [("deepep", "normal", False), ("deepep", "low_latency", True),
     ("deepep", "auto", True), ("none", "auto", True)],
)
def test_sm90_scale_layouts_follow_reachable_dispatch_modes(a2a, mode, needs_companion):
    # Bypass only the loading bug above so this test independently exercises
    # real post-load processing, including the normal-only memory saving.
    experts = _load_experts(scale_dtype=torch.uint8)
    original = experts.w13_weight_scale_inv.detach().clone()
    with (
        get_context().override_server_args(
            moe_runner_backend="deep_gemm", moe_a2a_backend=a2a, deepep_mode=mode
        ),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.DEEP_GEMM,
            a2a_backend=MoeA2ABackend(a2a),
            deepep_mode=DeepEPMode(mode),
        ),
        patch.object(fp8, "_is_cpu", False),
        patch.object(fp8, "get_platform", return_value=SimpleNamespace(is_sm90=True)),
    ):
        experts.quant_method.process_weights_after_loading(experts)

    torch.testing.assert_close(experts.w13_weight_scale_inv, original)
    assert experts.w13_weight_scale_inv.stride(-2) == 1
    assert experts.w2_weight_scale_inv.stride(-2) == 1
    for name in ("w13_weight_scale_inv_fp32", "w2_weight_scale_inv_fp32"):
        if needs_companion:
            scale = getattr(experts, name)
            assert scale.dtype == torch.float32
            assert scale.is_contiguous()
            torch.testing.assert_close(scale[0, 0], torch.tensor([0.5, 1.0, 2.0, 4.0]))
        else:
            assert not hasattr(experts, name)
