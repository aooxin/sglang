from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MIMO_PRO_NUM_EXPERTS = 384


@cache_once
def _jit_mimo_pro_moe_fused_gate_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"mimo_pro_moe_fused_gate: unsupported logit dtype {dtype}; "
            "expected float32, float16 or bfloat16"
        )
    args = make_cpp_args(dtype)
    return load_jit(
        "mimo_pro_moe_fused_gate",
        *args,
        cuda_files=["moe/mimo_pro_moe_fused_gate.cuh"],
        cuda_wrappers=[("mimo_pro_moe_fused_gate", f"mimo_pro_moe_fused_gate<{args}>")],
        # Matches the -use_fast_math the original sgl-kernel build used.
        extra_cuda_cflags=["--use_fast_math"],
    )


def mimo_pro_moe_fused_gate(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid + bias + top-k router for MiMo-V2 Pro (384 experts, one group).

    Experts are picked by ``sigmoid(logit) + correction_bias`` (lowest id wins
    ties); the returned weights are the unbiased sigmoid scores, optionally
    renormalized to sum to 1 and then multiplied by ``routed_scaling_factor``.

    Args:
        gating_output: [num_tokens, 384] router logits, fp32 / fp16 / bf16, contiguous.
        correction_bias: [384] fp32.
        topk: experts selected per token.

    Returns:
        (topk_weights [num_tokens, topk] fp32, topk_ids [num_tokens, topk] int32)
    """
    # Shape, dtype and device are checked by the C++ launcher; keep this path thin.
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    module = _jit_mimo_pro_moe_fused_gate_module(gating_output.dtype)
    module.mimo_pro_moe_fused_gate(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids
