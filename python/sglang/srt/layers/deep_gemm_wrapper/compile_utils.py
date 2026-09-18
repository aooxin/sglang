import logging
import os
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div, get_available_gpu_memory

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()

# FP4 W4A8 masked decode: max expected_m for which the DeepGEMM kernel's uint8
# E8M0 SFB fast path (scale_b_direct_load) is valid. At/below this the runner
# feeds the MN-major uint8 weight scale; above it the kernel host-asserts on a
# uint8 SFB, so the runner switches to the fp32 companion. The two SFB dtypes are
# different kernel instantiations, so the masked warmup executor below must
# exercise the same dtype the runner will pick for each m -- otherwise the decode
# cubin is not precompiled here and its first (lazy) compile lands inside CUDA
# graph capture's warmup forward.
# MUST match the kernel gate `expected_m <= kW4MaxDecodeM` in
# sm90_fp8_fp4_gemm_1d2d.hpp (DeepGEMM guanwenfei @026bcf7, kW4MaxDecodeM=96).
# Deploying a wheel whose gate is lower than this value would host-assert on the
# uint8 SFB; keep this constant and the wheel version in lockstep (see
# docker/Dockerfile.xiaomi, which must point at the 026bcf7 wheel).
FP4_MASKED_UINT8_SFB_MAX_EXPECTED_M = 96
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
_FAST_WARMUP = envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()

# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE

    _BUILTIN_M_LIST = []

    if _FAST_WARMUP:
        # In fast warmup mode, only compile a small set of typical Ms

        # First cover all the small bs to ensure decode performance
        _BUILTIN_M_LIST += list(range(1, 1025))

        # Then cover larger batch sizes with gradually increasing steps
        # For example, when chunekd prefill size is 16384
        # The sampled Ms would be:
        #   1024, 1026, ... 2046 (step 2)
        #   2048, 2052, ... 4092 (step 4)
        #   4096, 5004, ... 8184 (step 8)
        #   8192, 9008, ... 16384 (step 16)
        # Totally 1024 + 1024 / 2 + 2048 / 4 + 4096 / 8 + 8192 / 16 = 3072 kernels
        next_m, sample_step = 1024, 2
        max_prefill_bs = (
            min(server_args.chunked_prefill_size, 32 * 1024)
            if server_args.chunked_prefill_size >= 1
            else 16 * 1024
        )
        while next_m < max_prefill_bs:
            _BUILTIN_M_LIST += list(range(next_m, 2 * next_m, sample_step))
            next_m = next_m * 2
            sample_step = sample_step * 2
        _BUILTIN_M_LIST.append(max_prefill_bs)
        _BUILTIN_M_LIST = sorted(list(set(_BUILTIN_M_LIST)))
    else:
        # When fast warmup isn't enabled, generate m_max and compile all the covered Ms.
        m_max = 1024 * 16
        if server_args.chunked_prefill_size < 1:
            m_max = 1024 * 64
        elif server_args.chunked_prefill_size > 8192:
            m_max = server_args.chunked_prefill_size * 2
        m_max = min(1024 * 128, m_max)
        _BUILTIN_M_LIST += list(range(1, m_max + 1))

    # A PD-disaggregation *decode* node never runs prefill: its per-forward token
    # count M is bounded by the decode batch, not chunked_prefill_size. Compiling
    # (and, worse, running the warmup executor) up to a prefill-scale m_max makes
    # the warmup allocate an `out = (max_m, n)` buffer of several GB for the large
    # lm_head/dense GEMMs (e.g. m_max=32768, n=32768 -> 2 GB in bf16). Because the
    # first DeepGEMM call happens inside CUDA graph capture's warmup forward, that
    # transient scratch collides with the graph memory pool and OOMs at the same
    # --mem-fraction-static that fits the graphs otherwise. Bound m_max to the
    # decode ceiling so the warmup only touches the M values decode will ever see.
    # This is memory-neutral (KV cache untouched) and only skips compiling M values
    # this node can never hit (they would JIT lazily and cheaply if ever needed).
    if getattr(server_args, "disaggregation_mode", "null") == "decode":
        num_draft_tokens = getattr(server_args, "speculative_num_draft_tokens", None) or 1
        max_running = server_args.max_running_requests or 0
        cuda_graph_max_bs = server_args.cuda_graph_max_bs or 0
        decode_bs_ceiling = max(max_running, cuda_graph_max_bs)
        if decode_bs_ceiling > 0:
            # +1 page of headroom; covers spec draft-token expansion.
            decode_m_max = (decode_bs_ceiling + 1) * num_draft_tokens
            prev_max = _BUILTIN_M_LIST[-1] if _BUILTIN_M_LIST else 0
            if decode_m_max < prev_max:
                _BUILTIN_M_LIST = [m for m in _BUILTIN_M_LIST if m <= decode_m_max]
                if not _BUILTIN_M_LIST or _BUILTIN_M_LIST[-1] != decode_m_max:
                    _BUILTIN_M_LIST.append(decode_m_max)
                logger.info(
                    f"Decode disaggregation node: capping DeepGEMM warmup m_max "
                    f"from {prev_max} to {decode_m_max} "
                    f"(decode_bs_ceiling={decode_bs_ceiling}, "
                    f"num_draft_tokens={num_draft_tokens}) to avoid oversized "
                    f"warmup scratch during CUDA graph capture."
                )

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8FP4BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GROUPED_GEMM_NT_F8FP4BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()
    GEMM_NT_BF16BF16F32 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


# TODO improve code
def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        # TODO maybe improve logs
        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type in (
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_CONTIG,
        ):
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))

        # Here the precompilation is only run on the first rank, so gpu_id should be 0
        memory_budget = get_available_gpu_memory(device="cuda", gpu_id=0)

        # If the memory budget is less memory requirement, we need to reduce max_m to avoid out of memory, which might further cause hanging during warmup
        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:
            # TODO: Maybe compute the max_m based on the memory budget
            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 4096
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]

        # Need some methods to estimate needed memory for warmup
        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        # Newer DeepGEMM removed get/set_compile_mode.
        has_compile_mode = hasattr(deep_gemm, "get_compile_mode") and hasattr(
            deep_gemm, "set_compile_mode"
        )
        old_compile_mode = deep_gemm.get_compile_mode() if has_compile_mode else None
        if has_compile_mode:
            deep_gemm.set_compile_mode(1)
        # TODO can use multi thread
        for m in tqdm(m_list, desc=f"DeepGEMM warmup"):
            executor.execute(m=m)
        if has_compile_mode:
            deep_gemm.set_compile_mode(old_compile_mode)

        # clean up input buffers
        torch.cuda.current_stream().synchronize()
        del executor
        torch.cuda.empty_cache()
    finally:
        # Restore symmetric memory context
        restore_symmetric_memory_context(saved_context)


class _BaseWarmupExecutor:
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_CONTIG: _GroupedContFp8Fp4WarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_MASKED: _GroupedMaskedFp8Fp4WarmupExecutor,
            DeepGemmKernelType.GEMM_NT_BF16BF16F32: _BF16F32WarmupExecutor,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        # Return the required memory space in GB for warmup executor
        _GB = 1 << 30
        if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
            return (max_m * k + n * k + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_CONTIG:
            return (
                max_m * k
                + num_groups * n * ceil_div(k, 2)
                # contiguous SFB is uint8 E8M0 (1 byte), not fp32
                + num_groups * n * ceil_div(k, 32) * 1
                + max_m * 4
                + max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8FP4BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * ceil_div(k, 2)
                # both SFB variants are allocated: uint8 E8M0 + fp32 companion
                + num_groups * n * ceil_div(k, 32) * (1 + 4)
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        elif kernel_type == DeepGemmKernelType.GEMM_NT_BF16BF16F32:
            # bf16 lhs + bf16 rhs + fp32 out
            return (max_m * k * 2 + n * k * 2 + max_m * n * 4) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _NormalWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        # Pass m_indices positionally; newer DeepGEMM renames the kwarg.
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            self.m_indices[:m],
        )


class _GroupedContFp8Fp4WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q = torch.empty(
            (num_groups, n, ceil_div(k, 2)), device="cuda", dtype=torch.int8
        )
        # gran_k_b=32 contiguous path consumes uint8 E8M0 SFB that must be
        # MN-major (stride(-1) != 1): allocate [G, K/32, N] then transpose so the
        # logical shape is [G, N, K/32] with N as the contiguous dim.
        self.rhs_s = torch.empty(
            (num_groups, ceil_div(k, 32), n), device="cuda", dtype=torch.uint8
        ).transpose(-1, -2)
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            self.m_indices[:m],
            gran_k_b=32,
            compiled_dims="nk",
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedMaskedFp8Fp4WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q = torch.empty(
            (num_groups, n, ceil_div(k, 2)), device="cuda", dtype=torch.int8
        )
        # Two SFB variants, matching the runner's per-expected_m selection in
        # DeepGemmRunnerCore._run_masked_gemm:
        #   * expected_m <= FP4_MASKED_UINT8_SFB_MAX_EXPECTED_M: MN-major uint8
        #     E8M0 (the same tensor the contiguous path reads), i.e. logical
        #     [G, N, K/32] with N as the contiguous dim.
        #   * expected_m >  FP4_MASKED_UINT8_SFB_MAX_EXPECTED_M: K-major fp32
        #     companion.
        # They are distinct kernel instantiations, so both must be warmed here.
        self.rhs_s_u8 = torch.empty(
            (num_groups, ceil_div(k, 32), n), device="cuda", dtype=torch.uint8
        ).transpose(-1, -2)
        self.rhs_s_fp32 = torch.empty(
            (num_groups, n, ceil_div(k, 32)), device="cuda", dtype=torch.float32
        )
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        rhs_s = (
            self.rhs_s_u8
            if m <= FP4_MASKED_UINT8_SFB_MAX_EXPECTED_M
            else self.rhs_s_fp32
        )
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, rhs_s),
            self.out,
            masked_m=self.masked_m,
            expected_m=m,
            gran_k=128,
            gran_k_a=128,
            gran_k_b=32,
        )


class _BF16F32WarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = torch.empty((max_m, k), device="cuda", dtype=torch.bfloat16)
        self.rhs = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.float32)

    def execute(self, m):
        deep_gemm.bf16_gemm_nt(self.lhs[:m], self.rhs, self.out[:m])


@contextmanager
def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
