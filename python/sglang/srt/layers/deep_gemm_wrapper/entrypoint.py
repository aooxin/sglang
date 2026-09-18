import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = get_bool_env_var("SGLANG_DEEPGEMM_SANITY_CHECK")

# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_f8fp4bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    gran_k_a: int = 128,
    gran_k_b: int = 32,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    masked_m_max_hint: Optional[int] = None,
    active_groups_hint: Optional[int] = None,
):
    if overlap_args is not None:
        raise RuntimeError("SM90 FP8xFP4 masked DeepGEMM does not support overlap yet")

    kernel = getattr(
        deep_gemm,
        "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma",
        None,
    )
    if kernel is None:
        raise RuntimeError(
            "DeepGEMM does not expose "
            "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma; "
            "install DeepGEMM with deepseek-ai/DeepGEMM#332 or newer."
        )
    if recipe_a is not None:
        gran_k_a = recipe_a[1]
    if recipe_b is not None:
        gran_k_b = recipe_b[1]

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return kernel(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            gran_k=gran_k_a,
            gran_k_a=gran_k_a,
            gran_k_b=gran_k_b,
            masked_m_max_hint=masked_m_max_hint,
            active_groups_hint=active_groups_hint,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(lhs, rhs, out, m_indices)


def grouped_gemm_nt_f8fp4bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    gran_k_a: int = 128,
    gran_k_b: int = 32,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    kernel = getattr(
        deep_gemm,
        "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma",
        None,
    )
    if kernel is None:
        raise RuntimeError(
            "DeepGEMM does not expose "
            "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma; "
            "install DeepGEMM with SM90 contiguous FP8xFP4 gran_k_b support."
        )
    if recipe_a is not None:
        gran_k_a = recipe_a[1]
    if recipe_b is not None:
        gran_k_b = recipe_b[1]

    # SFA is always per-128 (the kernel hard-asserts gran_k_a == 128); only the
    # weight (B) scale granularity is configurable. The optimized SM90 contiguous
    # binding dropped the explicit `gran_k_a` kwarg, so we pass only `gran_k_b`.
    # gran_k_b == 32 selects the E8M0/uint8 SFB fast path (B scale is raw uint8).
    assert gran_k_a == 128, f"SM90 FP8xFP4 contiguous requires gran_k_a==128, got {gran_k_a}"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        kernel(
            lhs,
            rhs,
            out,
            m_indices,
            gran_k_b=gran_k_b,
            compiled_dims="nk",
        )


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    compile_utils.update_deep_gemm_config(gpu_id, server_args)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
