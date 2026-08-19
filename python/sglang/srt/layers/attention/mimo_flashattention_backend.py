"""MiMo FlashAttention backend with an FP8-quantized paged KV cache for full
attention layers.

Layer routing:
- SWA layers: inherited FA3 paths over the plain BF16 MHA sub-pool of
  MiMoSWAKVPool.
- Full layers: FP8-quantized paged cache (MiMoTokenToKVPool, page_size=64).
  - Write: MiMoTokenToKVPool.set_kv_buffer quantizes k_nope and v per token
    (k_rope stays BF16) and packs them into page blocks.
  - Decode and chain target-verify: the fused ez_kernel_kit kernel
    (mimo_v2_gqa_decode_fp8kv) reads the quantized pages directly. This is
    the only decode path for full layers; unsupported configurations raise
    instead of dequantizing the pool.
  - Prefill/extend: dequantize only the tokens attended by the batch into a
    per-layer workspace, then run flash_attn_varlen_func. The intermediate
    BF16 footprint is O(attended tokens of the batch), independent of the
    pool size and the number of layers.

Page block layout (ez_kernel_kit contract; the pool exposes 3D byte views
[num_pages, h_kv, page_dim] and flatten(1) gives the 2D kernel shape):
- k_buffer [num_pages, h_kv, 16640] uint8:
  [64 tokens, h_kv, 256B (128B fp8e4m3 k_nope | 64 x bf16 k_rope)]
  + [h_kv, 64] float32 k_scale tail
- v_buffer [num_pages, h_kv, 8448] uint8:
  [64 tokens, h_kv, 128B fp8e4m3 v] + [h_kv, 64] float32 v_scale tail

All tensor interfaces (q of the decode kernel, k of the pool ops) use the
model head order [rope 64 | nope 128], i.e. the order produced by partial
rotary embedding. Only the packed in-page K record is [nope | rope]; the
reordering happens inside the kernels (ez_kernel_kit >= 0.3.0.dev89), so no
Python-side permute is needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from ez_kernel_kit.ez_kernel_kit_cpp import mimo_v2_gqa_decode_fp8kv as ekk_decode

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# The fused kernel takes q [b, s_q, h_q, d] with s_q <= 8 and an in-window
# causal mask (query i attends [0, seqlens - s_q + i]).
_EKK_DECODE_MAX_SQ = 8


class _MiMoDequantWorkspace:
    """Persistent scratch buffers for layer-by-layer KV dequantization.

    All full layers of a forward pass share one workspace: it is fully
    overwritten before every attention call and grows lazily to the largest
    number of attended tokens seen so far. The peak intermediate memory is
    therefore O(max attended tokens of a batch) instead of O(pool size) per
    layer.
    """

    def __init__(self, device, head_num: int, head_dim: int, v_head_dim: int):
        self.device = device
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def get(self, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_tokens > 0
        capacity = self._k.shape[0] if self._k is not None else 0
        if num_tokens > capacity:
            self._k = torch.empty(
                (num_tokens, self.head_num, self.head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._v = torch.empty(
                (num_tokens, self.head_num, self.v_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
        return self._k[:num_tokens], self._v[:num_tokens]


class MiMoFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend for MiMoV2 hybrid (full + SWA) attention.

    Full layers run on the FP8-quantized paged KV cache (fused kernel for
    decode/target-verify, dequant + varlen for extend); SWA layers use the
    inherited FA3 implementation.
    """

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        super().__init__(model_runner, **kwargs)

        from sglang.srt.mem_cache.swa_memory_pool import MiMoSWAKVPool

        pool = model_runner.token_to_kv_pool
        assert isinstance(pool, MiMoSWAKVPool), (
            f"MiMoFlashAttentionBackend requires a MiMoSWAKVPool token_to_kv_pool, "
            f"got {type(pool).__name__}."
        )

        # Dequant workspace for the extend path, shared by all full layers of
        # a forward pass and overwritten per layer.
        full_pool = pool.full_kv_pool
        self._dequant_workspace = _MiMoDequantWorkspace(
            device=self.device,
            head_num=full_pool.head_num,
            head_dim=full_pool.head_dim,
            v_head_dim=full_pool.v_head_dim,
        )
        # Token slot ids attended by the current extend batch; built once per
        # forward pass and reused by every full layer.
        self._ragged_token_ids: Optional[torch.Tensor] = None

        logger.info("MiMoFlashAttentionBackend initialized")

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        # The ragged token ids are per batch; invalidate the cache before the
        # parent builds the new metadata.
        self._ragged_token_ids = None
        return super().init_forward_metadata(forward_batch)

    def _get_ragged_token_ids(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Return the token slot ids attended by this batch, concatenated per
        request in the order described by metadata.cu_seqlens_k.

        Built once per forward pass and reused by every full layer. The
        construction is sync-free: the output size comes from seq_lens_cpu
        and the gather uses repeat_interleave(output_size=...), so no
        dynamic-shaped op forces a hidden device-to-host sync.
        """
        if self._ragged_token_ids is None:
            metadata = self.forward_metadata
            seq_lens = forward_batch.seq_lens
            bs = seq_lens.numel()
            device = seq_lens.device
            total_k = int(forward_batch.seq_lens_cpu[:bs].sum().item())
            if total_k == 0:
                self._ragged_token_ids = torch.empty(
                    0, dtype=torch.int64, device=device
                )
                return self._ragged_token_ids

            # out[cu_seqlens_k[i] + j] = req_to_token[req_pool_indices[i], j]
            # for j in [0, seq_lens[i]).
            pos = torch.arange(total_k, device=device, dtype=torch.int64)
            row = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=torch.int64),
                seq_lens.to(torch.int64),
                output_size=total_k,
            )
            col = pos - metadata.cu_seqlens_k[row].to(torch.int64)
            self._ragged_token_ids = self.req_to_token[
                forward_batch.req_pool_indices.to(torch.int64)[row], col
            ].to(torch.int64)

            # Debug-only cross-check; adds one device-to-host sync.
            if envs.SGLANG_MIMO_CHECK_RAGGED_TOKEN_IDS.get():
                assert total_k == int(metadata.cu_seqlens_k[-1].item()), (
                    f"ragged token count {total_k} (from seq_lens_cpu) does "
                    f"not match cu_seqlens_k total "
                    f"{int(metadata.cu_seqlens_k[-1].item())}"
                )
        return self._ragged_token_ids

    @staticmethod
    def _is_swa_layer(layer: RadixAttention) -> bool:
        return layer.sliding_window_size is not None and layer.sliding_window_size > -1

    @staticmethod
    def _write_kv_cache(
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache: bool,
    ):
        """Quantize and pack the new K/V into the paged cache."""
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

    def _run_fused_decode_kernel(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        s_q: int,
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the fused FP8 kernel directly on the quantized pages.

        q is viewed as (b, s_q, h_q, d) in the model head order [rope | nope],
        which the kernel takes directly. metadata.page_table holds page
        indices (page_size > 1) and cache_seqlens_int32 the kernel's
        in-window seqlens. All kernel arguments are positional; a None output
        lets the kernel allocate it, and sinks is the attention sink bias
        (one raw logit per q head, FA3 semantics).
        """
        metadata = self.forward_metadata
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_quant_kv_buffer(
            layer.layer_id
        )
        o = ekk_decode(
            q.contiguous().view(-1, s_q, layer.tp_q_head_num, layer.head_dim),
            # flatten(1): zero-copy view to the kernel's 2D page-block shape.
            k_cache.flatten(1),
            v_cache.flatten(1),
            metadata.page_table,
            metadata.cache_seqlens_int32,
            float(layer.scaling),
            int(self.num_splits),
            sinks,
            None,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _use_fused_decode(
        self,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        q_rope: Optional[torch.Tensor],
    ) -> bool:
        """Return True if the fused FP8 decode kernel can serve this call.

        This is the only decode path for full layers; forward_decode raises
        when it returns False. Attention sinks are passed through to the
        kernel and do not disable this path.
        """
        return (
            not self._is_swa_layer(layer)
            and not layer.is_cross_attention
            and forward_batch.spec_info is None
            and q_rope is None
            and not layer.logit_cap
        )

    def _use_ragged_extend(
        self,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        q_rope: Optional[torch.Tensor],
    ) -> bool:
        """Return True if the dequant + varlen path can serve this extend
        call on a full layer.

        The path builds dynamic-shaped ragged token ids, so it is eager-only;
        prefill is never CUDA-graph captured. Target-verify batches are graph
        captured and are served by _forward_target_verify_fused instead.
        Attention sinks are supported by flash_attn_varlen_func.
        """
        metadata = self.forward_metadata
        is_cp_mode = (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        )
        return (
            not self._is_swa_layer(layer)
            and not layer.is_cross_attention
            and layer.attn_type == AttentionType.DECODER
            and forward_batch.spec_info is None
            and not is_cp_mode
            and q_rope is None
            and metadata is not None
            and metadata.cu_seqlens_k is not None
        )

    def _use_fused_target_verify(
        self,
        layer: RadixAttention,
        q_rope: Optional[torch.Tensor],
    ) -> bool:
        """Return True if the fused kernel can serve a target-verify call on
        a full layer.

        Chain (topk == 1) verification matches the kernel's in-window causal
        mask: q has s_q = speculative_num_draft_tokens (<= 8) tokens per
        request, seqlens include the draft tokens, and query i attends
        [0, seqlens - s_q + i]. Tree verification (topk > 1) needs a custom
        mask and is not supported.
        """
        return (
            self.topk <= 1
            and self.speculative_num_draft_tokens <= _EKK_DECODE_MAX_SQ
            and not layer.is_cross_attention
            and layer.attn_type == AttentionType.DECODER
            and q_rope is None
            and not layer.logit_cap
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # SWA layers use the inherited FA3 path over the BF16 MHA sub-pool.
        is_swa_layer = self._is_swa_layer(layer)
        if is_swa_layer:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        # Full layer: the fused kernel is the only decode path. Fail loudly
        # instead of silently dequantizing the whole pool.
        if not self._use_fused_decode(layer, forward_batch, q_rope):
            logger.error(
                "mimo_flash_attn fused decode cannot serve this call "
                "(layer_id=%s, spec_info=%s, cross_attention=%s, q_rope=%s, "
                "logit_cap=%s); there is no decode fallback for the "
                "quantized full-layer KV cache.",
                layer.layer_id,
                forward_batch.spec_info is not None,
                layer.is_cross_attention,
                q_rope is not None,
                layer.logit_cap,
            )
            raise RuntimeError(
                "mimo_flash_attn: unsupported decode configuration for the "
                "fused FP8 kernel (see error log above)."
            )

        # Quantize and pack the new K/V into the paged cache.
        self._write_kv_cache(layer, forward_batch, k, v, save_kv_cache)

        if q.numel() == 0:
            # Idle batch (e.g. DP attention idle round).
            return q.new_zeros((0, layer.tp_q_head_num * layer.v_head_dim))

        # Decode has one token per request (s_q = 1).
        return self._run_fused_decode_kernel(q, layer, forward_batch, 1, sinks)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if (
            not self._is_swa_layer(layer)
            and forward_batch.forward_mode.is_target_verify()
        ):
            # Full-layer target-verify (MTP / EAGLE) must run on the fused
            # kernel: the quantized pool has no BF16 buffers, and the ragged
            # dequant path is not CUDA-graph capturable while target-verify
            # is graph captured, so there is no fallback.
            if not self._use_fused_target_verify(layer, q_rope):
                logger.error(
                    "mimo_flash_attn fused target-verify cannot serve this call "
                    "(layer_id=%s, topk=%s, num_draft_tokens=%s, "
                    "cross_attention=%s, q_rope=%s, logit_cap=%s); the fused "
                    "kernel requires chain verification (topk=1) with at most "
                    "%s draft tokens and there is no fallback on the quantized "
                    "full-layer KV cache.",
                    layer.layer_id,
                    self.topk,
                    self.speculative_num_draft_tokens,
                    layer.is_cross_attention,
                    q_rope is not None,
                    layer.logit_cap,
                    _EKK_DECODE_MAX_SQ,
                )
                raise RuntimeError(
                    "mimo_flash_attn: unsupported target-verify configuration "
                    "for the fused FP8 kernel (see error log above)."
                )
            return self._forward_target_verify_fused(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        # SWA layers (and other unsupported combinations) go through the
        # inherited FA3 path over the BF16 sub-pool.
        if not self._use_ragged_extend(layer, forward_batch, q_rope):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        # Full layer: bounded dequant + varlen attention.
        # Write the current chunk into the paged cache first; the ragged
        # token ids below include the chunk's slots.
        self._write_kv_cache(layer, forward_batch, k, v, save_kv_cache)

        # Dequantize only the tokens attended by this batch (prefix + current
        # chunk) into the shared workspace, overwritten for every layer.
        metadata = self.forward_metadata
        token_ids = self._get_ragged_token_ids(forward_batch)
        total_k = token_ids.numel()
        if total_k == 0 or q.numel() == 0:
            return q.new_zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))

        k_ws, v_ws = self._dequant_workspace.get(total_k)
        # dequant_tokens returns K in the model head order [rope | nope], so
        # q and k line up without a permute.
        forward_batch.token_to_kv_pool.dequant_tokens(
            layer.layer_id, token_ids, k_out=k_ws, v_out=v_ws
        )

        # Packed varlen attention over the workspace (no page table): k_ws /
        # v_ws hold the per-request concatenation described by cu_seqlens_k,
        # like the parent's MLA gather-then-varlen path. FA3's causal mask is
        # bottom-right aligned, so the current chunk attends to the whole
        # prefix plus its own causal part, as chunked prefill requires.
        varlen_kwargs = {}
        if sinks is not None:
            varlen_kwargs["sinks"] = sinks
        result = self.flash_attn_varlen_func(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k=k_ws,
            v=v_ws,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_k,
            softmax_scale=layer.scaling,
            causal=True,
            softcap=layer.logit_cap,
            **varlen_kwargs,
        )
        o = result[0] if isinstance(result, tuple) else result
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_target_verify_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: "ForwardBatch",
        save_kv_cache: bool,
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run chain (topk == 1) target-verify on the quantized pages.

        The FA3 target-verify metadata already matches the kernel contract:
        cache_seqlens_int32 = seq_lens + s_q (draft tokens included, in both
        the eager and CUDA-graph replay paths) and a page-index page_table.
        All shapes are fixed for a given batch size and the metadata tensors
        are preallocated, so this path is CUDA-graph safe, unlike the ragged
        dequant extend path.
        """
        # Quantize and pack the draft tokens' K/V into the paged cache.
        self._write_kv_cache(layer, forward_batch, k, v, save_kv_cache)

        if q.numel() == 0:
            # Idle batch (e.g. DP attention idle round).
            return q.new_zeros((0, layer.tp_q_head_num * layer.v_head_dim))

        # s_q = speculative_num_draft_tokens tokens per request; the
        # target-verify metadata provides cache_seqlens_int32 = seq_lens + s_q.
        return self._run_fused_decode_kernel(
            q, layer, forward_batch, self.speculative_num_draft_tokens, sinks
        )
