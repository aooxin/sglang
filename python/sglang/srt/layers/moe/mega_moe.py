# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SM90 (Hopper) Mega-MoE: DeepGEMM's fused dispatch->GEMM->SwiGLU->GEMM->combine kernel."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    get_dp_global_num_tokens,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.utils import ceil_div

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


_MEGA_MOE_SYMM_BUFFER: dict = {}


def validate_mega_moe_token_cap(server_args: ServerArgs) -> None:
    """Reject at startup any config whose batch bounds can exceed the symm
    buffer cap. The runtime checks below fire mid-serving (or not at all
    during cuda graph capture, where should_use_mega_moe skips them), so
    misconfigurations must be caught before the server starts taking traffic.
    """
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()

    # The MoE input is the SCATTERED layout: per-rank share = tokens / attn_tp
    # (= tp/dp with DP attention, tp without; see LayerCommunicator). Computed
    # here because initialize_dp_attention runs later, in TpModelWorker init.
    from sglang.srt.layers.dp_attention import compute_dp_attention_world_info

    _, attn_tp_size, _ = compute_dp_attention_world_info(
        server_args.enable_dp_attention,
        0,
        server_args.tp_size,
        server_args.dp_size,
    )
    draft_mult = server_args.speculative_num_draft_tokens or 1

    bounds = {}
    # -1 disables chunking: a single prompt may then reach --context-length
    # tokens per rank, which no symm-buffer cap can bound.
    if not (server_args.chunked_prefill_size and server_args.chunked_prefill_size > 0):
        raise ValueError(
            "mega MoE requires chunked prefill: --chunked-prefill-size -1 is "
            "not supported, because an un-chunked prompt can carry up to "
            "--context-length tokens per rank and cannot be bounded by "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK."
        )

    if server_args.disaggregation_mode != "decode":
        # NOTE: must run after ServerArgs.__post_init__: with DP attention,
        # chunked_prefill_size there has already been divided by dp_size
        # (i.e. it is the per-DP-group budget, not the CLI value).
        bounds["--chunked-prefill-size"] = ceil_div(
            server_args.chunked_prefill_size, attn_tp_size
        )

    if server_args.disaggregation_mode != "prefill":
        if not server_args.disable_cuda_graph and server_args.cuda_graph_max_bs:
            bounds["--cuda-graph-max-bs (x draft tokens)"] = ceil_div(
                server_args.cuda_graph_max_bs * draft_mult, attn_tp_size
            )
        if server_args.max_running_requests is not None:
            bounds["--max-running-requests (x draft tokens)"] = ceil_div(
                server_args.max_running_requests * draft_mult, attn_tp_size
            )

    violations = {name: tokens for name, tokens in bounds.items() if tokens > cap}
    if violations:
        details = ", ".join(
            f"{name} allows up to {tokens} tokens/rank"
            for name, tokens in violations.items()
        )
        raise ValueError(
            f"mega MoE: SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK={cap} "
            f"is too small for the configured batch bounds: {details}. "
            f"Raise the env var or lower the offending option."
        )


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        # Entries are per-DP-group totals, but the MoE input is the SCATTERED
        # layout: each EP rank carries only 1/attn_tp of its group's tokens
        # (prepare_mlp_sync_batch pads each entry to a multiple of attn_tp for
        # exactly this split). The symm buffer is sized per rank, so compare
        # against the per-rank share; with dp == tp the divisor is 1, which is
        # why the group/rank mixup was invisible on the standard recipes.
        attn_tp_size = get_attention_tp_size()
        max_tokens_per_rank = ceil_div(max(global_num_tokens), attn_tp_size)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if max_tokens_per_rank > cap:
        # main falls back to the normal path here, but that relies on its
        # FIX_MEGA_MOE_MEMORY weight sharing where the normal runner was
        # adapted to consume the interleaved layout. On this branch
        # build_mega_moe_experts_weights aliases the mega-transformed tensors
        # into w13_weight/w2_weight in place, so the normal path would read
        # the wrong layout and silently corrupt outputs — raise instead.
        raise RuntimeError(
            f"mega MoE: forward carries up to {max_tokens_per_rank} tokens per "
            f"rank, exceeding "
            f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK={cap}. "
            f"Raise the env var or lower the prefill / decode batch bounds."
        )
    return True


def forward_mega_moe(
    moe,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        getattr(moe, "alt_stream", None) is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )

    # Quantize with the same kernel as the DeepEP path (not DeepGEMM's
    # per_token_cast_to_fp8, whose scale math differs slightly) to keep
    # megamoe bit-identical to DeepEP.
    if num_tokens > 0:
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        x_fp8, x_sf = sglang_per_token_group_quant_fp8(
            hidden_states,
            128,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_sf)
        buf.topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
        buf.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    # Allocate at least one row so y has a non-null CUDA data_ptr.
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    # Read weights via the params, not cached refs: SGL_CPU_QUANTIZATION's
    # loader swaps param.data after build, orphaning any tensor cached there.
    deep_gemm.fp8_mega_moe(
        y,
        (moe.experts.w13_weight.data, moe.experts.w13_weight_scale_inv.data),
        (moe.experts.w2_weight.data, moe.experts.w2_weight_scale_inv.data),
        buf,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=swiglu_limit,
        # match sglang's -use_fast_math SwiGLU PTX for bit-exactness
        fast_math=True,
    )
    y = y[:num_tokens]

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import transform_weights_for_mega_moe_sm90

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf = experts.w2_weight_scale_inv.data

    (l1_w, l1_sf), (l2_w, l2_sf) = transform_weights_for_mega_moe_sm90(
        (w13, w13_sf), (w2, w2_sf)
    )

    # Alias every transformed tensor back into its source param so that EPLB's
    # in-place `.copy_()` rebalance updates reach the kernel (and cuda graph
    # keeps capturing the same buffers). No-op when the transform returns the
    # source tensor itself; the shape asserts fail loudly at load time if the
    # DeepGEMM fork ever changes a tensor's layout. Do not cache the tensors in
    # extra attributes: SGL_CPU_QUANTIZATION swaps param.data after this runs.
    for transformed, param_data, name in (
        (l1_w, w13, "L1 weight"),
        (l1_sf, w13_sf, "L1 scale"),
        (l2_w, w2, "L2 weight"),
        (l2_sf, w2_sf, "L2 scale"),
    ):
        assert (
            transformed.shape == param_data.shape
        ), f"mega MoE: transformed {name} shape changed, cannot alias for EPLB"
    experts.w13_weight.data = l1_w
    experts.w13_weight_scale_inv.data = l1_sf
    experts.w2_weight.data = l2_w
    experts.w2_weight_scale_inv.data = l2_sf

    experts._mega_moe_weights_built = True
