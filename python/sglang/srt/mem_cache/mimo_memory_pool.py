"""FP8-quantized paged KV cache pool for MiMoV2 full attention layers.

The page layout is derived from the model config (config.json):
- head_dim: K head dim (rope + nope parts).
- v_head_dim: V head dim.
- partial_rotary_factor: rope_dim = int(head_dim * partial_rotary_factor),
  the same truncation get_rope() applies for partial rotary embedding; the
  remaining nope_dim = head_dim - rope_dim.

Per-token K records keep the rotated rope dims in BF16 and quantize the nope
dims to FP8; V is fully quantized to FP8. Quantization is per token per head:
scale = amax / 448, original = fp8_value * scale.

The page-block byte content matches the ez_kernel_kit fp8kv kernel contract
(see ez_kernel_kit/ops/mimo_v2_gqa.py). The kernels are compiled for exactly
one layout (rope 64 / nope 128 / v 128 / page 64, i.e. the MiMoV2 config
head_dim=192, v_head_dim=128, partial_rotary_factor=0.334); the pool derives
its layout from the config and rejects anything the kernels do not support.

Buffers are kept as 3D views [num_pages, h_kv, page_dim]; a page block is
contiguous, so buf.flatten(1) yields the [num_pages, h_kv * page_dim] kernel
shape at zero cost. Note that for h_kv > 1, dim 1 of the 3D view is not the
head index: the data region is token-outer and the scale tail is head-outer.

k_buffer page block (h_kv * k_page_bytes bytes):
- [0, h_kv * k_data_bytes): data [page_size tokens, h_kv, k_token_bytes];
  each K record is [nope_dim x fp8e4m3 | rope_dim x bf16]
- scale tail: [h_kv, page_size] float32 k_scale

v_buffer page block (h_kv * v_page_bytes bytes):
- [0, h_kv * v_data_bytes): data [page_size tokens, h_kv, v_head_dim bytes];
  each record is v_head_dim x fp8e4m3
- scale tail: [h_kv, page_size] float32 v_scale

Head-order contract (ez_kernel_kit >= 0.3.0.dev89): every tensor interface of
this pool and of the kernels uses the model head order [rope | nope], the
order produced by partial rotary embedding. Only the packed in-page K record
is [nope | rope]; the reordering happens inside the kernels.

The hot read/write paths go through the ez_kernel_kit companion ops
(mimo_v2_gqa_quant_store_kv / mimo_v2_gqa_dequant_gather_kv), which are
bit-identical to the fused decode kernel's in-kernel dequantization and are
CUDA-graph safe. The pure-torch byte gather/scatter helpers serve the
record-granular maintenance paths: move_kv_cache and the CPU offload copies
(get_cpu_copy / load_cpu_copy).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import ez_kernel_kit.ez_kernel_kit_cpp as ekk_cpp
import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

# Per-token per-head dequantization scale (float32).
SCALE_BYTES = 4

# The only (rope_dim, nope_dim, v_head_dim, page_size) the ez_kernel_kit
# mimo_v2_gqa fp8kv kernels are compiled for; this is the full layout
# contract. Derived values from the model config must match exactly.
EKK_SUPPORTED_ROPE_DIM = 64
EKK_SUPPORTED_NOPE_DIM = 128
EKK_SUPPORTED_V_HEAD_DIM = 128
EKK_SUPPORTED_PAGE_SIZE = 64

@dataclass(frozen=True)
class MiMoKVLayout:
    """Byte layout of the quantized KV page blocks, derived from the model
    config via derive_mimo_kv_layout()."""

    page_size: int
    head_dim: int
    v_head_dim: int
    rope_dim: int  # rotated K dims, kept in BF16
    nope_dim: int  # non-rotated K dims, FP8 quantized

    @property
    def k_token_bytes(self) -> int:
        # fp8 nope (1 byte per element) + bf16 rope (2 bytes per element).
        return self.nope_dim + self.rope_dim * 2

    @property
    def k_data_bytes(self) -> int:
        # Per page per head.
        return self.page_size * self.k_token_bytes

    @property
    def v_token_bytes(self) -> int:
        # fp8 (1 byte per element).
        return self.v_head_dim

    @property
    def v_data_bytes(self) -> int:
        # Per page per head.
        return self.page_size * self.v_token_bytes

    @property
    def page_scale_bytes(self) -> int:
        # Scale tail per page per head.
        return self.page_size * SCALE_BYTES

    @property
    def k_page_bytes(self) -> int:
        # Page block dim 2 of k_buffer; one page block is h_kv * this bytes.
        return self.k_data_bytes + self.page_scale_bytes

    @property
    def v_page_bytes(self) -> int:
        # Page block dim 2 of v_buffer.
        return self.v_data_bytes + self.page_scale_bytes

    @property
    def quant_bytes_per_token_per_head(self) -> int:
        # Exact quantized KV bytes per token per head, used by the model
        # runner for exact KV capacity profiling.
        return (self.k_token_bytes + SCALE_BYTES) + (self.v_token_bytes + SCALE_BYTES)


def derive_mimo_kv_layout(
    head_dim: int,
    v_head_dim: int,
    partial_rotary_factor: float,
    page_size: int,
) -> MiMoKVLayout:
    """Derive the quantized page layout from model config values.

    rope_dim uses the same truncation as the model's partial rotary embedding
    (get_rope): int(head_dim * partial_rotary_factor). For the MiMoV2 config
    (head_dim=192, v_head_dim=128, partial_rotary_factor=0.334) this yields
    rope 64 / nope 128 and the 16640 / 8448 page dims of the kernel contract.
    """
    rope_dim = int(head_dim * partial_rotary_factor)
    nope_dim = head_dim - rope_dim
    assert 0 < rope_dim < head_dim, (
        f"Invalid rope_dim={rope_dim} derived from head_dim={head_dim} and "
        f"partial_rotary_factor={partial_rotary_factor}; the quantized MiMo "
        f"KV layout requires a partial rotary embedding (0 < rope_dim < "
        f"head_dim)."
    )
    # Reject layouts the ez_kernel_kit fp8kv kernels are not compiled for.
    got = (rope_dim, nope_dim, v_head_dim, page_size)
    expected = (
        EKK_SUPPORTED_ROPE_DIM,
        EKK_SUPPORTED_NOPE_DIM,
        EKK_SUPPORTED_V_HEAD_DIM,
        EKK_SUPPORTED_PAGE_SIZE,
    )
    assert got == expected, (
        f"The ez_kernel_kit mimo_v2_gqa fp8kv kernels only support "
        f"(rope_dim, nope_dim, v_head_dim, page_size)={expected}, "
        f"but the model config derives {got} "
        f"(head_dim={head_dim}, v_head_dim={v_head_dim}). "
        f"Check head_dim / v_head_dim / partial_rotary_factor in config.json "
        f"and the --page-size argument."
    )
    return MiMoKVLayout(
        page_size=page_size,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )


def get_mimo_partial_rotary_factor(model_config) -> float:
    """Return partial_rotary_factor from a sglang ModelConfig, defaulting to
    1.0 like get_rope() does when the config omits it."""
    return getattr(model_config.hf_text_config, "partial_rotary_factor", 1.0)


def derive_mimo_kv_layout_from_config(
    model_config, page_size: int
) -> MiMoKVLayout:
    """Derive the quantized page layout from a sglang ModelConfig (the parsed
    config.json). See derive_mimo_kv_layout for the derivation rules."""
    return derive_mimo_kv_layout(
        head_dim=model_config.head_dim,
        v_head_dim=model_config.v_head_dim,
        partial_rotary_factor=get_mimo_partial_rotary_factor(model_config),
        page_size=page_size,
    )


class MiMoTokenToKVPool(MHATokenToKVPool):
    """FP8-quantized paged KV cache pool for MiMoV2 full attention layers.

    The byte layout (self.kv_layout) is derived from the model config; see
    the module docstring. Per-layer buffers, with the current kernel
    contract's dims (flatten(1) gives the 2D shape the kernels take):
    - k_buffer: [num_pages, h_kv, 16640] uint8
    - v_buffer: [num_pages, h_kv, 8448] uint8

    All public read/write methods use the model head order [rope | nope].

    Write path: set_kv_buffer quantizes and scatters via
    mimo_v2_gqa_quant_store_kv (CUDA-graph safe).
    Read paths:
    - get_quant_kv_buffer: raw quantized pages, consumed directly by the
      fused decode kernel.
    - dequant_tokens: gather and dequantize selected token slots to BF16,
      used by the attention backend's extend path.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        partial_rotary_factor: Optional[float] = None,
    ):
        assert partial_rotary_factor is not None, (
            "MiMoTokenToKVPool requires partial_rotary_factor (from the model "
            "config) to derive the rope/nope split of the quantized K layout."
        )
        # Derive the byte layout from the model config values; the derivation
        # validates it against the fused kernel contract.
        self.kv_layout = derive_mimo_kv_layout(
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            partial_rotary_factor=partial_rotary_factor,
            page_size=page_size,
        )

        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            # The tiled token-record copy kernel does not understand the
            # packed page layout; move_kv_cache is overridden below.
            enable_kv_cache_copy=False,
        )

    def _create_buffers(self):
        layout = self.kv_layout
        # The base class provisions size + page_size token slots (page 0 is
        # the padding page); cover them all with whole pages.
        self.num_pages = (self.size + 2 * self.page_size - 1) // self.page_size

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # 3D byte views of the kernel page blocks; see the module
                # docstring for the in-page layout.
                self.k_buffer = [
                    torch.zeros(
                        (self.num_pages, self.head_num, layout.k_page_bytes),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.num_pages, self.head_num, layout.v_page_bytes),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        # Bytes per page: dim 0 of the buffers is pages, not tokens.
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    # _clear_buffers is inherited from MHATokenToKVPool; it deletes k_buffer
    # and v_buffer, which matches the buffer names here.

    def _k_cols(self, slot: torch.Tensor) -> torch.Tensor:
        """Return the byte columns of the K record + scale for tokens at
        in-page offsets `slot`, per kv head; shape [N, H, k_token_bytes + 4].

        The data region is token-outer ([page_size, H, k_token_bytes]) and
        the scale tail is head-outer ([H, page_size] float32) starting at
        H * k_data_bytes. Index tensors are built on demand; this only runs
        on the record-granular maintenance paths (move_kv_cache / CPU
        offload), not on the per-forward hot paths.
        """
        layout = self.kv_layout
        h = torch.arange(self.head_num, dtype=torch.int64, device=self.device)  # [H]
        data = (
            slot[:, None, None] * self.head_num + h[None, :, None]
        ) * layout.k_token_bytes
        data = data + torch.arange(
            layout.k_token_bytes, dtype=torch.int64, device=self.device
        )  # [N, H, k_token_bytes]
        scale = (
            self.head_num * layout.k_data_bytes
            + (h[None, :, None] * self.page_size + slot[:, None, None]) * SCALE_BYTES
            + torch.arange(SCALE_BYTES, dtype=torch.int64, device=self.device)
        )  # [N, H, 4]
        return torch.cat([data, scale], dim=-1)

    def _v_cols(self, slot: torch.Tensor) -> torch.Tensor:
        """Return the byte columns of the V record + scale; shape
        [N, H, v_token_bytes + 4]."""
        layout = self.kv_layout
        h = torch.arange(self.head_num, dtype=torch.int64, device=self.device)
        data = (
            slot[:, None, None] * self.head_num + h[None, :, None]
        ) * layout.v_token_bytes
        data = data + torch.arange(
            layout.v_token_bytes, dtype=torch.int64, device=self.device
        )  # [N, H, v_token_bytes]
        scale = (
            self.head_num * layout.v_data_bytes
            + (h[None, :, None] * self.page_size + slot[:, None, None]) * SCALE_BYTES
            + torch.arange(SCALE_BYTES, dtype=torch.int64, device=self.device)
        )  # [N, H, 4]
        return torch.cat([data, scale], dim=-1)

    def _scatter_records(
        self,
        buf: torch.Tensor,  # [num_pages, H, page_dim] uint8
        page_idx: torch.Tensor,  # [N] int64
        cols: torch.Tensor,  # [N, H, C] int64, byte offsets within a page block
        vals: torch.Tensor,  # [N, H, C] uint8
    ):
        # flatten(1) is a zero-copy view of the contiguous page block.
        buf.flatten(1)[page_idx[:, None, None], cols] = vals

    def _gather_records(
        self,
        buf: torch.Tensor,  # [num_pages, H, page_dim] uint8
        page_idx: torch.Tensor,  # [N] int64
        cols: torch.Tensor,  # [N, H, C] int64, byte offsets within a page block
    ) -> torch.Tensor:  # [N, H, C] uint8
        return buf.flatten(1)[page_idx[:, None, None], cols]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Quantize and pack the K/V of `loc` tokens into the paged buffers.

        Args:
            layer: RadixAttention providing layer_id; may be None when
                layer_id_override is given.
            loc: [B] token slot indices in the pool (page = loc // page_size).
            cache_k: [B, h_kv, head_dim] BF16 K after rotary embedding, model
                head order ([..., :rope_dim] rope, [..., rope_dim:] nope).
            cache_v: [B, h_kv, v_head_dim] BF16 V.
            k_scale, v_scale: unused; quantization is per-token dynamic.
        """
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        lid = layer_id - self.start_layer

        if loc.numel() == 0:
            return

        B, H, D = cache_k.shape
        assert D == self.head_dim, f"Expected K head_dim={self.head_dim}, got {D}"
        assert (
            cache_v.shape[-1] == self.v_head_dim
        ), f"Expected v_head_dim={self.v_head_dim}, got {cache_v.shape[-1]}"
        assert H == self.head_num

        assert (
            cache_k.dtype == torch.bfloat16
        ), f"Expected BF16 cache_k, got {cache_k.dtype}"
        assert (
            cache_v.dtype == torch.bfloat16
        ), f"Expected BF16 cache_v, got {cache_v.dtype}"

        loc = loc.view(-1).to(dtype=torch.int64, device=self.device)

        # Fused quantize + scatter, bit-identical to the decode kernel's
        # dequantization and CUDA-graph safe (no alloc or sync inside the
        # op). Takes k in the model head order [rope | nope].
        ekk_cpp.mimo_v2_gqa_quant_store_kv(
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.k_buffer[lid].flatten(1),
            self.v_buffer[lid].flatten(1),
        )

    def _maybe_wait_layer(self, layer_id: int, global_layer_id: Optional[int] = None):
        # HiCache layer-wise loading synchronizes on the global layer id.
        if self.layer_transfer_counter is not None:
            wait_id = (
                global_layer_id if global_layer_id is not None else layer_id
            ) - self.start_layer
            self.layer_transfer_counter.wait_until(wait_id)

    def get_quant_kv_buffer(
        self, layer_id: int, global_layer_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the raw quantized page buffers for the fused decode kernel.

        Returns:
            k_buffer: [num_pages, h_kv, k_page_bytes] uint8
            v_buffer: [num_pages, h_kv, v_page_bytes] uint8
            (3D byte views; the attention backend flattens dim 1 before
            passing them to the kernel.)
        """
        self._maybe_wait_layer(layer_id, global_layer_id)
        lid = layer_id - self.start_layer
        return self.k_buffer[lid], self.v_buffer[lid]

    def dequant_tokens(
        self,
        layer_id: int,
        token_ids: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        global_layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the given token slots into caller-owned BF16 K/V.

        Only the requested tokens are dequantized into the caller-owned
        workspace (overwritten per layer), so the peak extra memory is
        bounded by the batch size rather than the pool size. K is returned in
        the model head order [rope | nope]; the in-page [nope | rope]
        reordering is undone inside the gather op.

        Args:
            layer_id: pool-local layer id (MiMoSWAKVPool passes the mapped id
                plus global_layer_id).
            token_ids: [N] token slot indices (page = id // page_size).
            k_out: [N, h_kv, head_dim] BF16 destination workspace.
            v_out: [N, h_kv, v_head_dim] BF16 destination workspace.
            global_layer_id: global layer id for HiCache layer-wise waiting.

        Returns:
            k: [N, h_kv, head_dim] BF16 in the model head order [rope | nope].
            v: [N, h_kv, v_head_dim] BF16.
        """
        self._maybe_wait_layer(layer_id, global_layer_id)
        lid = layer_id - self.start_layer

        loc = token_ids.view(-1).to(dtype=torch.int64, device=self.device)
        N = loc.numel()
        H = self.head_num

        assert k_out.shape == (N, H, self.head_dim) and k_out.dtype == torch.bfloat16
        assert v_out.shape == (N, H, self.v_head_dim) and v_out.dtype == torch.bfloat16
        if N == 0:
            return k_out, v_out

        ekk_cpp.mimo_v2_gqa_dequant_gather_kv(
            self.k_buffer[lid].flatten(1),
            self.v_buffer[lid].flatten(1),
            loc,
            k_out,
            v_out,
        )
        return k_out, v_out

    def _get_key_buffer(self, layer_id: int):
        raise NotImplementedError(
            "MiMoTokenToKVPool stores quantized pages. Use get_quant_kv_buffer() for "
            "raw pages or dequant_tokens() for a dequantized copy."
        )

    def _get_value_buffer(self, layer_id: int):
        raise NotImplementedError(
            "MiMoTokenToKVPool stores quantized pages. Use get_quant_kv_buffer() for "
            "raw pages or dequant_tokens() for a dequantized copy."
        )

    def get_kv_size_bytes(self):
        k_size = sum(buf.nbytes for buf in self.k_buffer)
        v_size = sum(buf.nbytes for buf in self.v_buffer)
        return k_size, v_size

    def get_contiguous_buf_infos(self):
        # dim 0 of the buffers is pages, so the per-item length (one page
        # block) is buf[0].nbytes.
        kv_data_ptrs = [buf.data_ptr() for buf in self.k_buffer] + [
            buf.data_ptr() for buf in self.v_buffer
        ]
        kv_data_lens = [buf.nbytes for buf in self.k_buffer] + [
            buf.nbytes for buf in self.v_buffer
        ]
        kv_item_lens = [buf[0].nbytes for buf in self.k_buffer] + [
            buf[0].nbytes for buf in self.v_buffer
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError("MiMoTokenToKVPool does not support move_kv_cache.")

    def get_cpu_copy(self, indices):
        """Copy the quantized records (data + scales) of the given tokens to
        CPU; layout mirrors MHATokenToKVPool.get_cpu_copy (per layer, per
        chunk, [k, v] pairs)."""
        torch.cuda.synchronize()
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.int64)
        indices = indices.view(-1).to(dtype=torch.int64, device=self.device)

        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for lid in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                pages, slots = chunk // self.page_size, chunk % self.page_size
                k_rec = self._gather_records(
                    self.k_buffer[lid], pages, self._k_cols(slots)
                ).to("cpu", non_blocking=True)
                v_rec = self._gather_records(
                    self.v_buffer[lid], pages, self._v_cols(slots)
                ).to("cpu", non_blocking=True)
                kv_cache_cpu[-1].append([k_rec, v_rec])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        """Load quantized records produced by get_cpu_copy back into the
        pool at the (possibly different) given token slots."""
        torch.cuda.synchronize()
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.int64)
        indices = indices.view(-1).to(dtype=torch.int64, device=self.device)

        chunk_size = self.cpu_offloading_chunk_size
        for lid in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                k_rec, v_rec = kv_cache_cpu[lid][i // chunk_size]
                assert k_rec.shape[0] == v_rec.shape[0] == len(chunk)
                pages, slots = chunk // self.page_size, chunk % self.page_size
                self._scatter_records(
                    self.k_buffer[lid],
                    pages,
                    self._k_cols(slots),
                    k_rec.to(self.device, non_blocking=True),
                )
                self._scatter_records(
                    self.v_buffer[lid],
                    pages,
                    self._v_cols(slots),
                    v_rec.to(self.device, non_blocking=True),
                )
        torch.cuda.synchronize()
