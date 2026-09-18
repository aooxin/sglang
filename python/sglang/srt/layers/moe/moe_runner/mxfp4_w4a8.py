from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)


@dataclass
class Mxfp4W4A8QuantInfo(MoeQuantInfo):
    w13_weight: Any
    w2_weight: Any


def _validate_runner_config(runner_config: MoeRunnerConfig) -> None:
    if runner_config.activation != "silu":
        raise NotImplementedError("mxfp4_w4a8 currently supports silu MoE only.")
    if not runner_config.is_gated:
        raise NotImplementedError("mxfp4_w4a8 requires a gated MoE layer.")
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "mxfp4_w4a8 does not support apply_router_weight_on_input."
        )
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
    ):
        raise NotImplementedError(
            "mxfp4_w4a8 does not support non-standard SwiGLU alpha/clamp."
        )


def _deepep_normal_num_tokens_to_gpu(
    num_recv_tokens_per_expert: list[int],
    device: torch.device,
) -> torch.Tensor:
    from sglang.srt.layers.moe.moe_runner.deep_gemm import copy_list_to_gpu_no_ce
    from sglang.srt.utils.offloader import get_offloader

    if get_offloader().forbid_copy_engine_usage:
        return copy_list_to_gpu_no_ce(num_recv_tokens_per_expert).to(device)

    return torch.tensor(
        num_recv_tokens_per_expert,
        device="cpu",
        dtype=torch.int32,
        pin_memory=True,
    ).to(device, non_blocking=True)


def _mxfp4_w4a8_deepep_normal(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: Mxfp4W4A8QuantInfo,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather, ep_scatter
    from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_humming import (
        mxfp4_w4a8_deepep_normal_humming,
    )

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output

    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens == 0:
        return DeepEPNormalCombineInput(
            hidden_states=torch.empty_like(hidden_states, dtype=torch.bfloat16),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    device = hidden_states.device
    hidden_size = hidden_states.shape[1]
    num_tokens_per_expert_gpu = _deepep_normal_num_tokens_to_gpu(
        num_recv_tokens_per_expert,
        device=device,
    )
    expert_start_for_scatter = torch.empty_like(num_tokens_per_expert_gpu)
    expert_start = torch.empty_like(num_tokens_per_expert_gpu)
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    output_index = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    if hidden_states.dtype == torch.float8_e4m3fn:
        if hidden_states_scale is None or hidden_states_scale.dtype != torch.float32:
            raise NotImplementedError(
                "mxfp4_w4a8 DeepEP normal FP8 path requires float32 scales."
            )
        input_tensor = torch.empty(
            (all_tokens, hidden_size),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        input_tensor_scale = torch.empty(
            (all_tokens, hidden_size // 128),
            device=device,
            dtype=torch.float32,
        )
        ep_scatter(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_tokens_per_expert_gpu,
            expert_start_for_scatter,
            input_tensor,
            input_tensor_scale,
            m_indices,
            output_index,
            expert_start_out=expert_start,
        )
    else:
        if hidden_states.dtype != torch.bfloat16:
            raise NotImplementedError(
                "mxfp4_w4a8 DeepEP normal path supports only FP8 or BF16 input."
            )

        # DeepEP BF16 dispatch does not provide activation scales. Quantize each
        # received token before the expert-major reorder; quantization is
        # per-token, so this is numerically equivalent to quantizing after the
        # reorder and lets the existing FP8 ep_scatter kernel remain the single
        # reorder implementation.
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        quantized_hidden_states, quantized_hidden_states_scale = (
            sglang_per_token_group_quant_fp8(hidden_states, 128)
        )
        input_tensor = torch.empty(
            (all_tokens, hidden_size),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        input_tensor_scale = torch.empty(
            (all_tokens, hidden_size // 128),
            device=device,
            dtype=torch.float32,
        )
        ep_scatter(
            quantized_hidden_states,
            quantized_hidden_states_scale,
            topk_ids,
            num_tokens_per_expert_gpu,
            expert_start_for_scatter,
            input_tensor,
            input_tensor_scale,
            m_indices,
            output_index,
            expert_start_out=expert_start,
        )
        del quantized_hidden_states, quantized_hidden_states_scale

    down_output = mxfp4_w4a8_deepep_normal_humming(
        input_tensor,
        input_tensor_scale,
        expert_start,
        num_tokens_per_expert_gpu,
        quant_info.w13_weight,
        quant_info.w2_weight,
    )

    output = torch.empty(
        hidden_states.shape,
        device=device,
        dtype=torch.bfloat16,
    )
    ep_gather(down_output, topk_ids, topk_weights, output_index, output)
    return DeepEPNormalCombineInput(
        hidden_states=output,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


@register_fused_func("deepep", "mxfp4_w4a8")
def fused_experts_deepep_to_mxfp4_w4a8(
    dispatch_output: DeepEPLLDispatchOutput | DeepEPNormalDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput | DeepEPNormalCombineInput:
    if not DispatchOutputChecker.format_is_deepep(dispatch_output):
        raise NotImplementedError("mxfp4_w4a8 currently supports DeepEP dispatch only.")
    _validate_runner_config(runner_config)
    if not isinstance(quant_info, Mxfp4W4A8QuantInfo):
        raise TypeError(f"Unexpected quant_info type: {type(quant_info)}")
    if quant_info.w13_weight is None or quant_info.w2_weight is None:
        raise RuntimeError("mxfp4_w4a8 requires both Humming expert weights.")

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        return _mxfp4_w4a8_deepep_normal(dispatch_output, quant_info)

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
    ) = dispatch_output
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "mxfp4_w4a8 requires DeepEP FP8 dispatch output. "
            "Start the server with --deepep-dispatcher-output-dtype fp8."
        )
    if hidden_states_scale is None:
        raise NotImplementedError(
            "mxfp4_w4a8 Humming low-latency path requires FP8 scales."
        )

    from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_humming import (
        mxfp4_w4a8_deepep_ll_humming,
    )

    output = mxfp4_w4a8_deepep_ll_humming(
        hidden_states,
        hidden_states_scale,
        masked_m,
        expected_m,
        quant_info.w13_weight,
        quant_info.w2_weight,
    )
    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
