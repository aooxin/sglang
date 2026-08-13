# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    parse_dflash_draft_config,
)
from sglang.srt.utils.common import add_prefix

logger = logging.getLogger(__name__)


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.v_head_dim = int(getattr(config, "v_head_dim", head_dim))
        self.q_size = self.num_heads * head_dim
        self.k_size = self.num_kv_heads * head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim
        # Keep the old name aliased so existing callers that read .kv_size keep
        # working when k_size == v_size (the common case).
        self.kv_size = self.k_size

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            v_head_size=self.v_head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=(f"{prefix}.qkv_proj" if prefix else "qkv_proj"),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=(f"{prefix}.o_proj" if prefix else "o_proj"),
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Parse dflash_config early so RoPE / SWA / sink / v_scale all use it
        # consistently.
        draft_cfg = parse_dflash_draft_config(draft_hf_config=config)

        # RoPE base: read directly from config.rope_theta. The HF ckpt's top-level
        # rope_theta is what the draft was TRAINED with — for SWA-trained drafts
        # this is the SWA base (e.g. 10000), NOT the backbone base. The DFlash
        # commit f765d9de7 (in the old repo) explicitly REMOVED an override that
        # had been incorrectly substituting the backbone base.
        # transformers v5+ moves top-level rope_theta / rope_scaling into a
        # config.rope_parameters dict and hides the legacy attributes. Read the
        # dict first; if absent, fall back to the legacy top-level attributes.
        # Mirrors the pattern used in models/qwen3.py.
        if (
            hasattr(config, "rope_parameters")
            and config.rope_parameters
            and "rope_theta" in config.rope_parameters
        ):
            rope_theta = float(config.rope_parameters["rope_theta"])
            rope_scaling = config.rope_parameters
        else:
            rope_theta = float(getattr(config, "rope_theta", 1000000))
            rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))

        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
            partial_rotary_factor=partial_rotary_factor,
        )

        # SWA window for the draft block. Prefer dflash_config.swa_window_size
        # (training-time setting), fall back to top-level sliding_window.
        swa_window = None
        if draft_cfg.use_swa:
            swa_window = draft_cfg.swa_window_size
            if swa_window is None:
                swa_window = getattr(config, "sliding_window", None)

        self.scaling = head_dim**-0.5
        # DFlash uses non-causal attention over the draft block.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            v_head_dim=self.v_head_dim,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            sliding_window_size=int(swa_window) if swa_window else -1,
        )

        # Optional: per-layer attention_sink_bias parameter + value scale read
        # from dflash_config. Defaults preserve old checkpoints behavior.
        self.v_scale: Optional[float] = draft_cfg.attention_value_scale
        self.attention_sink_bias = (
            nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if draft_cfg.attention_sink_bias
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        if self.v_scale is not None:
            v = v * self.v_scale
        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + self.k_size + self.v_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.k_size, self.v_size], dim=-1)
            if self.v_scale is not None:
                v = v * self.v_scale
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        if self.v_scale is not None:
            v = v * self.v_scale
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Use a minimal dummy query (1 head) to avoid doing full-Q work.
        dummy_q = k.new_empty((k.shape[0], self.head_dim))
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=(f"{prefix}.self_attn" if prefix else "self_attn"),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(
            config=config,
            quant_config=quant_config,
            prefix=(f"{prefix}.mlp" if prefix else "mlp"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization

    The token embedding weight is not in the checkpoint; under dp attention
    the worker populates `self.embed_tokens.weight` from the target's embed
    weight (resharded to the attention-tp group), so draft embed lookup is
    isolated to the active dp attention group and doesn't collide with idle
    peer ranks.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        # Embed aliased to the target's embed weight by the worker; not
        # loaded from the draft checkpoint (checkpoint carries no embed).
        # `config.vocab_size` here is ALREADY the target's vocab size — the
        # worker overrides it via json_model_override_args before this model
        # is constructed. It must match the target exactly: the shard indices
        # computed from it drive input masking and the mask-embedding row
        # lookup against the aliased target weight.
        # `use_attn_tp_group=is_dp_attention_enabled()` makes the runtime
        # all_reduce inside `VocabParallelEmbedding.forward` route to the
        # attention-tp group directly, so we don't rely on the caller
        # patching `_TP` first.
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=(f"{prefix}.layers.{i}" if prefix else f"layers.{i}"),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            # Merged QKV (per-rank chunked FP8 layout): both .weight (fp8) and
            # .weight_scale_inv (fp32) are sharded by chunk(attn_tp_size, dim=0).
            # Falls through to stacked_params_mapping for legacy ckpts that store
            # q_proj/k_proj/v_proj separately.
            if (
                ".self_attn.qkv_proj.weight" in name
                or ".self_attn.qkv_proj.weight_scale_inv" in name
            ):
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                attn_tp_size = get_attention_tp_size()
                attn_tp_rank = get_attention_tp_rank()
                sharded = loaded_weight.chunk(attn_tp_size, dim=0)[attn_tp_rank]
                default_weight_loader(param, sharded)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("attention_sink_bias"):
                    # Slice the full-head sink_bias tensor per attn_tp_rank to
                    # match the per-rank parameter shape (= num_heads).
                    start = get_attention_tp_rank() * param.numel()
                    sliced = loaded_weight[start : start + param.numel()]
                    param.data.copy_(sliced)
                    continue
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = DFlashDraftModel
