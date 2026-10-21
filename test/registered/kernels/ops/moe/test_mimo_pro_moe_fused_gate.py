"""Correctness tests for the MiMo-V2 Pro fused gate JIT kernel.

The kernel replaces the Triton ``moe_fused_gate`` router for MiMo-V2 Pro routing
(384 experts, one group, sigmoid + correction bias), so it is checked against a
definition-based torch reference and against that Triton router. Weight
comparisons scatter back to a dense ``[M, 384]`` layout, so per-row slot order
does not matter.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.moe.mimo_pro_moe_fused_gate import (
    MIMO_PRO_NUM_EXPERTS,
    mimo_pro_moe_fused_gate,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
NUM_EXPERTS = MIMO_PRO_NUM_EXPERTS
TOPK = 8
SCALE = 2.5


def _scatter_by_expert(weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], NUM_EXPERTS), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, ids.long(), weights.float())
    return dense


def _reference_gate(logits, bias, topk, renormalize, scale, apply_scale):
    """Sigmoid + bias selection (lowest id wins ties), unbiased weights."""
    scores = logits.float().sigmoid()
    work = scores + bias.unsqueeze(0)
    arange = torch.arange(NUM_EXPERTS, device=logits.device).unsqueeze(0)
    weights = torch.empty(logits.shape[0], topk, dtype=torch.float32, device=DEVICE)
    ids = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=DEVICE)
    for k in range(topk):
        best, _ = work.max(dim=1, keepdim=True)
        winner = torch.where(work == best, arange, NUM_EXPERTS).min(dim=1).values
        ids[:, k] = winner.to(torch.int32)
        weights[:, k] = scores.gather(1, winner.unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.unsqueeze(1), float("-inf"))
    if renormalize:
        weights = weights / weights.sum(dim=1, keepdim=True)
    if apply_scale:
        weights = weights * scale
    return weights, ids


def _make_inputs(num_tokens: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device=DEVICE) * 2.0
    bias = torch.randn(NUM_EXPERTS, device=DEVICE) * 0.5
    return logits.to(dtype), bias


# 512 is the last token count of the small-batch kernel; 4099 leaves a partial
# last CTA in the warp-per-token kernel (6 tokens per CTA).
@pytest.mark.parametrize("num_tokens", [1, 37, 512, 513, 4099])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "renormalize,apply_scale", [(True, False), (True, True), (False, True)]
)
def test_matches_reference(num_tokens, dtype, renormalize, apply_scale):
    logits, bias = _make_inputs(num_tokens, dtype, seed=num_tokens)

    weights, ids = mimo_pro_moe_fused_gate(
        logits, bias, TOPK, renormalize, SCALE, apply_scale
    )
    ref_weights, ref_ids = _reference_gate(
        logits, bias, TOPK, renormalize, SCALE, apply_scale
    )

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    torch.testing.assert_close(
        _scatter_by_expert(weights, ids),
        _scatter_by_expert(ref_weights, ref_ids),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("num_tokens", [64, 2048])
def test_matches_triton_router(num_tokens):
    """Must agree with the Triton router it replaces for MiMo-V2 Pro."""
    logits, bias = _make_inputs(num_tokens, torch.float32, seed=7)

    weights, ids = mimo_pro_moe_fused_gate(logits, bias, TOPK, True, 1.0, False)
    triton_weights, triton_ids = moe_fused_gate(
        logits,
        bias,
        TOPK,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )

    torch.testing.assert_close(
        _scatter_by_expert(weights, ids),
        _scatter_by_expert(triton_weights, triton_ids),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("num_tokens", [3, 600])
def test_ties_pick_lowest_expert_ids(num_tokens):
    """Equal biased scores must resolve to the lowest ids, in both kernels."""
    logits = torch.zeros(num_tokens, NUM_EXPERTS, device=DEVICE)
    bias = torch.zeros(NUM_EXPERTS, device=DEVICE)

    _, ids = mimo_pro_moe_fused_gate(logits, bias, TOPK, True, 1.0, False)

    expected = torch.arange(TOPK, dtype=torch.int32, device=DEVICE)
    assert torch.equal(ids, expected.expand(num_tokens, TOPK))


def test_empty_batch():
    logits = torch.empty(0, NUM_EXPERTS, device=DEVICE)
    bias = torch.zeros(NUM_EXPERTS, device=DEVICE)

    weights, ids = mimo_pro_moe_fused_gate(logits, bias, TOPK, True, 1.0, False)

    assert weights.shape == (0, TOPK) and ids.shape == (0, TOPK)


@pytest.mark.parametrize(
    "num_experts,num_fused_shared_experts,routed",
    [(NUM_EXPERTS, 0, True), (256, 0, False), (NUM_EXPERTS, 1, False)],
)
def test_biased_grouped_topk_routes_only_mimo_pro_shape(
    num_experts, num_fused_shared_experts, routed
):
    """With SGLANG_USE_FUSED_MOE_GATE on, only the 384-expert, one-group,
    no-shared-expert shape may take this kernel; others keep their routers."""
    from sglang.srt.layers.moe import topk as topk_mod

    logits = torch.randn(16, num_experts, device=DEVICE)
    bias = torch.zeros(num_experts, device=DEVICE)
    spy = mock.Mock(wraps=mimo_pro_moe_fused_gate)
    with (
        mock.patch.object(topk_mod, "_use_mimo_pro_fused_gate", True),
        mock.patch.object(topk_mod, "MIMO_PRO_NUM_EXPERTS", NUM_EXPERTS, create=True),
        mock.patch.object(topk_mod, "mimo_pro_moe_fused_gate", spy, create=True),
    ):
        topk_mod.biased_grouped_topk_gpu(
            logits,
            logits,
            bias,
            TOPK + num_fused_shared_experts,
            True,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=num_fused_shared_experts,
        )

    assert spy.called == routed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
