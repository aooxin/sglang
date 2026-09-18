from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    silu_and_mul_masked_post_quant_fwd,
)
from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_tuning import (
    HUMMING_FINE_TUNING_RANGES,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_HUMMING_EXPERT_LAYOUT_CACHE = {}
_HUMMING_WARNED_REASONS = set()


@contextlib.contextmanager
def _stage_timer(timings: dict[str, float], key: str, device: torch.device):
    """Accumulate wall time for one repack stage into ``timings[key]``.

    Synchronizes the device on both sides so the number reflects GPU
    completion rather than kernel-launch overhead. Load-time one-shot
    profiling only; the sync overhead is negligible here.
    """
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        timings[key] = time.perf_counter() - start


@dataclass
class HummingMxfp4W4A8Weight:
    layer: Any
    n: int
    k: int
    num_experts: int
    contig_compute_config: dict[str, Any]
    contig_tuning_config: Any
    masked_compute_config: dict[str, Any]
    masked_tuning_config: Any
    ep_size: int | None = None


def _log_humming_warning_once(reason: str, message: str) -> None:
    if reason in _HUMMING_WARNED_REASONS:
        return
    _HUMMING_WARNED_REASONS.add(reason)
    logger.warning(message)


def _import_humming():
    try:
        from humming.config import GemmType
        from humming.layer import HummingLayer
        from humming.tune import get_heuristics_config
    except ImportError as exc:
        raise RuntimeError(
            "The mxfp4_w4a8 backend requires the patched humming-kernels wheel "
            "built from commit 7669649 or a compatible descendant."
        ) from exc
    return GemmType, HummingLayer, get_heuristics_config


# All Hopper (sm_90) silicon in this fleet is H20-class and reports 132 SMs, and
# it MUST run Humming's Sm90H20Heuristics (block_m clamped to <=64). The generic
# Sm90Heuristics emits large grouped tiles (block_m up to 176; warp_m == block_m
# in the warp-spec mainloop) that (a) overflow ptxas registers at
# num_ctas_per_sm>=2 -> compile fails at cuda-graph capture, and (b) hang the
# grouped GEMM at runtime even when forced to compile.
#
# We route by SM count, NOT device name: the same H20 silicon reports several
# product names (H20Z / M403 / L20X / ...) and Humming's own router matches only
# the substring "H20", so every other name falls through to the crashing generic
# path. SM count is name-independent and covers all aliases at once.
#
# NOTE: 132 SMs is not unique to H20 -- genuine H100-SXM / H200 also report 132,
# so this clamps them too. That is intentional and safe: this fleet runs only H20,
# and a clamped tile on full-fat Hopper would merely be slower, never wrong.
# (H100-PCIe is 114 SMs and is not matched; a differently-sized H20 variant would
# need this constant extended.)
_HUMMING_H20_SM_COUNT = 132


def _humming_should_force_h20_heuristics() -> bool:
    """True when the current CUDA device must use Humming's Sm90H20Heuristics.

    Routes by sm_90 + SM count rather than product name, so every H20 alias
    (H20Z / M403 / L20X / ...) is covered without maintaining a device-name list.
    """
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count == _HUMMING_H20_SM_COUNT


def _humming_heuristics_config(
    get_heuristics_config, *, meta, use_f16_accum, gemm_type
):
    """Wrap Humming's get_heuristics_config, forcing Sm90H20Heuristics on the
    132-SM H20-class devices that Humming's name-based router would otherwise send
    to the crashing/hanging generic Sm90Heuristics path (its product names lack the
    "H20" substring the router keys on). Falls back to the stock router (and stock
    behavior) on every other device."""
    if _humming_should_force_h20_heuristics():
        from humming.tune.sm90_h20 import Sm90H20Heuristics

        return Sm90H20Heuristics.get_configs(
            meta=meta,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=False,
            gemm_type=gemm_type,
        )
    return get_heuristics_config(
        meta=meta,
        use_f16_accum=use_f16_accum,
        gemm_type=gemm_type,
    )


_HUMMING_TUNED_MOE_SHAPES = frozenset(
    {
        (4096, 6144),
        (6144, 2048),
        (4096, 4096),
        (4096, 2048),
    }
)


def _humming_prefill_tuning_overrides(
    n: int,
    k: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    # Large-M (M >= 4096) prefill tile for the W13/W2 MoE GEMMs.
    #
    # C1 (2026-07, H20): the original override forced block_m=48/stages=3/ctas=2
    # here, which is ~10-12% slower than block_m=64 at the real prefill sizes
    # (M ~= 18176..19456). Measured (tests/probe_c1_tiles.py, CUDA-event steady
    # state, M=19200, warp_n=32, stream_k):
    #   block_m=48 s3 c2 (old): W13 2128us / W2 1180us  (baseline)
    #   block_m=64 s4 c2 (new): W13 1900us / W2 1075us  (~1.11x faster, M-robust)
    #   block_m=64 s{3,5} c2  : identical to s4 (stages count not sensitive here)
    #   block_m=64 *  c3      : 3044us / 1566us  (0.70x -- ctas 3 over-subscribes
    #                           the big tile; fused-E8M0 config tuning caps it at 2)
    #   block_m=128           : >=2x slower (too few CTAs to fill 132 SMs)
    # Small-M chunk buckets (M < 4096) are left to the heuristic and are unchanged
    # by this override.
    #
    # C2 (2026-08-29, H20Z): re-tuned stream_k OFF + stages=5 for this large-M
    # contiguous tile. At prefill M the grid already has far more than 132 tiles,
    # so the stream-k split just adds a cross-CTA reduction (locks + partial-sum
    # epilogue) with no occupancy benefit. Exhaustive sweep on pro_ep8 (W13
    # 4096x6144, W2 6144x2048) over block_m{16..96} x block_k{128,256} x
    # stages{3-6} x stream_k{0,1} x ctas{1,2,3}, all 6 sizes per-expert 384..2048
    # (total_m 18k..98k), triton.do_bench warm:
    #   - block_m=64 and stream_k=OFF are the per-size optimum at EVERY size
    #     (larger block_m needs warp_m>MmaShape::M=64 which the WGMMA path forbids;
    #     smaller block_m under-fills the tile: bm32 +73%, bm96 +38% vs bm64).
    #   - vs the old sk1/s4 override this is ~+4.4pp faster (Humming +28.7% ->
    #     ~+24% slower than DeepGEMM's f8fp4bf16 contiguous kernel, avg W13+W2).
    #   - stages: within a single config, s5 is the best all-size choice (avg only
    #     +0.7% / worst +2.0% behind the per-size optimum, beating s4's +1.0%/+3.2%
    #     and s3/s6). A per-M-bucket stage table would recover the last ~1% but is
    #     not worth the complexity.
    # Humming still trails DeepGEMM at large-M contiguous (structural: WGMMA M-atom
    # forces warp_m==64, and the warp-spec/TMA/multicast paths are unviable for the
    # fused-E8M0 mxfp4 contiguous case), so prefill should prefer the deep_gemm
    # backend; this override only makes the Humming fallback as fast as it tunes.
    tuned_warp_n32_bm64 = {
        "block_shape": (64, 128, 128),
        "warp_shape": (64, 32, 128),
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_sms": 132,
        "num_stages": 5,
        "num_ctas_per_sm": 2,
    }
    if (n, k) in _HUMMING_TUNED_MOE_SHAPES:
        return [(4096, 1 << 30, tuned_warp_n32_bm64)]
    return []


_HUMMING_DYNAMIC_MASKED_SHAPES = _HUMMING_TUNED_MOE_SHAPES
# Coarse fallback buckets for supported MoE shapes. The fine-grained table below
# covers the measured Flash/Pro EP8/16/32 decode range; these buckets remain the
# compatibility path for missing EP metadata and for M beyond a fine table when
# a coarse bucket is available.
_HUMMING_DYNAMIC_MASKED_TUNING_SPECS = (
    (8, 8, 4),
    (16, 16, 4),
    (24, 24, 4),
    (32, 32, 4),
    (40, 40, 4),
    (48, 48, 4),
    (56, 56, 4),
    (64, 64, 4),
    (72, 72, 4),
    (80, 80, 4),
    (88, 88, 3),
)


def _make_humming_masked_tuning_config(
    block_m: int,
    num_stages: int,
    stream_k: bool,
) -> dict[str, Any]:
    return {
        "block_shape": (block_m, 128, 128),
        "warp_shape": (block_m, 32, 128),
        "use_stream_k": stream_k,
        "use_f16_accum": False,
        "num_sms": 132,
        "num_stages": num_stages,
        "num_ctas_per_sm": 2,
    }


# Generic fallback buckets, reached only when no measured fine table applies --
# i.e. a supported (N, K) shape with missing/unknown EP metadata (ep_size=None)
# or an EP with no HUMMING_FINE_TUNING_RANGES entry. Every measured EP's fine
# table now reaches at least this bucket's top (em=88), and coarse retention
# keeps only buckets strictly above the fine table's top, so none of them fall
# through here. stream_k=True mirrors what the deep-K measured tables use in em81-88
# 1-tile region; it is not optimal below em81 (measured SK0 wins there), so any
# shape that actually runs hot in em65-80 should get its own fine table rather
# than lean on these buckets.
_HUMMING_DYNAMIC_MASKED_TUNING_CONFIGS = tuple(
    (max_expected_m, _make_humming_masked_tuning_config(block_m, num_stages, True))
    for max_expected_m, block_m, num_stages in _HUMMING_DYNAMIC_MASKED_TUNING_SPECS
)


def _humming_masked_fine_tuning_configs(
    entry: HummingMxfp4W4A8Weight,
    ep_size: int | None = None,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    if ep_size is None:
        ep_size = getattr(entry, "ep_size", None)
    if ep_size is None:
        return ()
    ranges = HUMMING_FINE_TUNING_RANGES.get((entry.n, entry.k, ep_size), ())
    # stream_k is measured per row and differs between W13 and W2 (e.g. at EP32
    # W13 wants it throughout the low-em range while W2 never does, because
    # W13's K is deeper), so it must come from the table, not a global default.
    return tuple(
        (
            end_expected_m,
            _make_humming_masked_tuning_config(block_m, num_stages, stream_k),
        )
        for _, end_expected_m, block_m, num_stages, stream_k in ranges
    )


def _humming_masked_tuning_override(
    tuning_config: Any,
    n: int,
    k: int,
) -> Any:
    """Decode (masked) tuning override for the W13/W2 MoE GEMMs.

    For the measured MiMo W13/W2 shapes, replace Humming's per-M list heuristic
    with a single-dict BM64 fallback; the live low-latency path overrides it with
    an expected_m bucket from the tuning table. Any other shape keeps Humming's
    own heuristic, which is the automatic fallback for shapes we never measured.
    """
    if (n, k) not in _HUMMING_TUNED_MOE_SHAPES:
        return tuning_config
    return _make_humming_masked_tuning_config(64, 4, True)


def _humming_masked_dynamic_tuning_configs(
    entry: HummingMxfp4W4A8Weight,
    ep_size: int | None = None,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    fine_configs = _humming_masked_fine_tuning_configs(entry, ep_size)
    if fine_configs:
        fine_table_end = fine_configs[-1][0]
        coarse_configs = tuple(
            (max_expected_m, tuning_config)
            for max_expected_m, tuning_config in _HUMMING_DYNAMIC_MASKED_TUNING_CONFIGS
            if max_expected_m > fine_table_end
        )
        return (*fine_configs, *coarse_configs)
    if (entry.n, entry.k) not in _HUMMING_DYNAMIC_MASKED_SHAPES:
        return ()
    return _HUMMING_DYNAMIC_MASKED_TUNING_CONFIGS


def _select_humming_masked_tuning_config(
    entry: HummingMxfp4W4A8Weight,
    expected_m: int,
    ep_size: int | None = None,
) -> Any:
    # Try the measured per-(N, K, ep_size) table first. Scanning it before
    # building the concatenated list is equivalent -- the fine ranges are
    # monotonic and every retained coarse bucket sits above the fine table's top
    # -- and it skips materializing the coarse configs in the common case.
    fine_configs = _humming_masked_fine_tuning_configs(entry, ep_size)
    for max_expected_m, tuning_config in fine_configs:
        if expected_m <= max_expected_m:
            return tuning_config

    # No measured entry applies, so the coarse buckets or the single-config
    # fallback take over and the calibration for this shape is silently lost.
    # Surface it once per (N, K, ep_size) -- a decode config that outgrew its
    # table looks identical to a tuned one from the outside.
    resolved_ep_size = (
        ep_size if ep_size is not None else getattr(entry, "ep_size", None)
    )
    _log_humming_warning_once(
        f"masked_tuning_fallback_{entry.n}_{entry.k}_{resolved_ep_size}",
        f"Humming masked tuning fell back for (n={entry.n}, k={entry.k}, "
        f"ep_size={resolved_ep_size}) at expected_m={expected_m}: "
        + (
            f"beyond the measured table (top expected_m={fine_configs[-1][0]})"
            if fine_configs
            else "no measured table for this shape/EP combination"
        )
        + ". Decode is running an uncalibrated tile config; add a "
        "HUMMING_FINE_TUNING_RANGES entry to recover it.",
    )

    for max_expected_m, tuning_config in _humming_masked_dynamic_tuning_configs(
        entry, ep_size
    ):
        if expected_m <= max_expected_m:
            return tuning_config
    return entry.masked_tuning_config


def _humming_masked_warmup_tuning_configs(
    entry: HummingMxfp4W4A8Weight,
    ep_size: int | None = None,
) -> tuple[Any, ...]:
    dynamic_configs = tuple(
        tuning_config
        for _, tuning_config in _humming_masked_dynamic_tuning_configs(entry, ep_size)
    )
    dynamic_configs = tuple(
        tuning_config
        for index, tuning_config in enumerate(dynamic_configs)
        if tuning_config not in dynamic_configs[:index]
    )
    if not dynamic_configs:
        return (entry.masked_tuning_config,)
    if entry.masked_tuning_config in dynamic_configs:
        return dynamic_configs
    return (*dynamic_configs, entry.masked_tuning_config)


# num_ctas_per_sm bounds for the fused-E8M0 MoE GEMM path, measured on H20.
# Humming's grouped heuristic mis-sizes num_ctas_per_sm at the two extremes of
# block_m:
#
#  * Smallest tile (block_m == 8, m[0,4) region) gets num_ctas_per_sm == 1. On the
#    E8M0 path that runs ~1.6x slower than == 2 (0.040 vs 0.024ms at M=64),
#    because a single CTA/SM cannot hide the per-tile launch + scale-load overhead.
#    -> floor at 2.
#  * Largest tile (block_m >= 64) gets num_ctas_per_sm == 3.
#    A big tile's register/SMEM footprint makes 3 CTAs/SM over-subscribe and drop
#    occupancy: fused block_m 64 runs 0.037ms at ctas 3 vs 0.025ms at ctas 2
#    (1.48x, measured).
#    -> cap at 2 once block_m >= _HUMMING_E8M0_LARGE_BLOCK_M.
#
# Both bounds converge to exactly 2 for the tiles they touch; the floor only
# bites the smallest tile and the cap only the largest, so mid tiles (16/32/48,
# heuristic ctas 2-3) are left as-is below the large-block threshold.
_HUMMING_E8M0_MIN_CTAS_PER_SM = 2
_HUMMING_E8M0_LARGE_BLOCK_M = 64
_HUMMING_E8M0_LARGE_BLOCK_MAX_CTAS = 2


def _tune_humming_fused_e8m0_configs(tuning_config: Any) -> Any:
    """Apply measured CTA bounds to fused-E8M0 Humming configs.

    Keeps at least two CTAs per SM for the smallest tile and caps large tiles at
    two CTAs per SM. Returns a new object and supports both a single config dict
    and Humming's ``[m_start, m_end, config]`` list form.

    """

    def fix_cfg(cfg: dict) -> dict:
        block_m = cfg.get("block_shape", (0,))[0]
        ctas = cfg.get("num_ctas_per_sm", 1)
        new_ctas = max(ctas, _HUMMING_E8M0_MIN_CTAS_PER_SM)
        if block_m >= _HUMMING_E8M0_LARGE_BLOCK_M:
            new_ctas = min(new_ctas, _HUMMING_E8M0_LARGE_BLOCK_MAX_CTAS)
        if new_ctas == ctas:
            return cfg
        new_cfg = dict(cfg)
        new_cfg["num_ctas_per_sm"] = new_ctas
        return new_cfg

    if isinstance(tuning_config, dict):
        return fix_cfg(tuning_config)
    if isinstance(tuning_config, list):
        sanitized: list[Any] = []
        for entry in tuning_config:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 3
                and isinstance(entry[2], dict)
            ):
                sanitized.append([entry[0], entry[1], fix_cfg(entry[2])])
            else:
                sanitized.append(entry)
        return sanitized
    return tuning_config


def _apply_humming_prefill_tuning(
    tuning_config: Any,
    n: int,
    k: int,
) -> Any:
    overrides = _humming_prefill_tuning_overrides(n, k)
    if not overrides:
        return tuning_config

    if isinstance(tuning_config, dict):
        return overrides[-1][2]
    if not isinstance(tuning_config, list):
        return tuning_config

    min_override_m = min(start for start, _, _ in overrides)
    tuned_config = [
        entry
        for entry in tuning_config
        if len(entry) == 3 and entry[1] <= min_override_m
    ]
    # The heuristic's own large-M bucket (end far above min_override_m) is dropped
    # by the filter above, but the retained small-M buckets often stop well below
    # min_override_m -- e.g. at 2048 for 24-local-expert shapes, 1024 for 12. Left
    # as-is that leaves (m0, min_override_m) with no entry, and Humming hard-errors
    # ("shape_m is not within any range defined in configs") on any contiguous M
    # that lands there, which is reachable at ordinary chunked-prefill sizes.
    # Bridge it by extending the lowest override entry down to the last retained
    # bucket end (or to 0 if the heuristic contributed no bucket at or below
    # min_override_m).
    bridged = [list(item) for item in overrides]
    m0 = max((entry[1] for entry in tuned_config), default=0)
    if m0 < min_override_m:
        for item in bridged:
            if item[0] == min_override_m:
                item[0] = m0
                break
    tuned_config.extend(bridged)
    return tuned_config


def _build_humming_weight_entry(
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    n: int,
    k: int,
    expert_major_input_scale: bool = False,
    ep_size: int | None = None,
) -> HummingMxfp4W4A8Weight:
    # expert_major_input_scale: only W13 receives its FP8 activation scale in the
    # DeepEP [E,G,T] dispatch layout. W2's scale is produced fresh (row-major
    # [E,T,G]) by the SiLU+quant kernel, so W2 must stay on the standard loader.
    GemmType, HummingLayer, get_heuristics_config = _import_humming()
    if expert_major_input_scale:
        from humming.config import ComputeConfig

        if "use_expert_major_input_scale" not in getattr(
            ComputeConfig, "__dataclass_fields__", {}
        ):
            raise RuntimeError(
                "Humming expert-major input scale requires commit 7669649 or "
                "newer. Upgrade the humming-kernels wheel instead of running the "
                "older d91089e interface, which cannot consume DeepEP [E,G,T] "
                "scales."
            )

    # The checkpoint stores per-32 weight scales as native E8M0 exponents. Feed
    # them directly to Humming's fused-E8M0 kernel without an fp32 detour.
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        raise RuntimeError(
            "Humming MXFP4 W4A8 path requires torch.float8_e8m0fnu support."
        )
    if b_scale.dtype == e8m0_dtype:
        b_scale_e8m0 = b_scale.contiguous()
    elif b_scale.dtype == torch.uint8:
        b_scale_e8m0 = b_scale.contiguous().view(e8m0_dtype)
    else:
        raise TypeError(
            "Humming MXFP4 W4A8 path expects E8M0 (uint8/float8_e8m0fnu) weight "
            f"scale, got {b_scale.dtype}"
        )
    if not b_packed.is_contiguous():
        raise ValueError("Humming MXFP4 W4A8 path requires contiguous weight tensors.")

    num_experts = b_packed.shape[0]
    with torch.cuda.device(b_packed.device):
        layer = HummingLayer(
            shape_n=n,
            shape_k=k,
            num_experts=num_experts,
            weight_config={
                "dtype": "float4e2m1",
                "group_size": 32,
                "scale_dtype": "float8e8m0",
            },
            input_config={"dtype": "float8e4m3", "group_size": 128},
            torch_dtype=torch.bfloat16,
        )
    layer.locks = torch.zeros((1024), dtype=torch.int32, device=b_packed.device)
    for name in ("weight", "weight_scale"):
        if hasattr(layer, name):
            delattr(layer, name)
    layer.weight = torch.nn.Parameter(b_packed.view(torch.int32), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(b_scale_e8m0, requires_grad=False)
    layer.transform()

    meta = layer.humming_metas[""]
    if not getattr(meta, "use_fused_e8m0_scale", False):
        raise RuntimeError(
            "The mxfp4_w4a8 backend requires Humming fused-E8M0 support. "
            "Use Humming commit 7669649 or a compatible descendant."
        )
    contig_compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_CONTIGUOUS.value,
    }
    masked_compute_config = {
        "use_f16_accum": False,
        "gemm_type": GemmType.GROUPED_MASKED.value,
    }
    if expert_major_input_scale:
        # W13 only: consume the DeepEP low-latency dispatch scale layout [E, G, T]
        # (per-expert [G, T] column-major) natively in the AS scale loader, so we
        # no longer materialize a row-major [E*T, G] copy (direct_copy_kernel_cuda,
        # ~0.8ms/step). Requires humming >= expert-major build (g7669649).
        masked_compute_config["use_expert_major_input_scale"] = True
    return HummingMxfp4W4A8Weight(
        layer=layer,
        n=n,
        k=k,
        num_experts=num_experts,
        ep_size=ep_size,
        contig_compute_config=contig_compute_config,
        contig_tuning_config=_tune_humming_fused_e8m0_configs(
            _apply_humming_prefill_tuning(
                _humming_heuristics_config(
                    get_heuristics_config,
                    meta=meta,
                    use_f16_accum=False,
                    gemm_type=GemmType.GROUPED_CONTIGUOUS,
                ),
                n,
                k,
            )
        ),
        masked_compute_config=masked_compute_config,
        masked_tuning_config=_tune_humming_fused_e8m0_configs(
            _humming_masked_tuning_override(
                _humming_heuristics_config(
                    get_heuristics_config,
                    meta=meta,
                    use_f16_accum=False,
                    gemm_type=GemmType.GROUPED_MASKED,
                ),
                n,
                k,
            )
        ),
    )


def _launch_grouped_gemm_contig_humming_entry(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    entry: HummingMxfp4W4A8Weight,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    a_scale_group_size: int,
) -> torch.Tensor:
    if a_scale_group_size != 128:
        raise ValueError(
            f"Humming MXFP4 W4A8 expects activation scale group size 128, "
            f"got {a_scale_group_size}"
        )
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        raise TypeError(
            "Humming MXFP4 W4A8 expects FP8 E4M3 activations and float32 scales."
        )
    if entry.k != a.shape[1]:
        raise ValueError(f"Humming K mismatch: entry={entry.k}, a={a.shape[1]}")

    layout_key = (
        expert_start.device.index,
        torch.cuda.current_stream(expert_start.device).cuda_stream,
        "torch.int32",
        num_tokens_per_expert.shape[0],
    )
    expert_layout = _HUMMING_EXPERT_LAYOUT_CACHE.get(layout_key)
    if expert_layout is None:
        expert_layout = torch.empty(
            (num_tokens_per_expert.shape[0] + 1,),
            device=expert_start.device,
            dtype=torch.int32,
        )
        _HUMMING_EXPERT_LAYOUT_CACHE[layout_key] = expert_layout
    expert_layout[:-1].copy_(expert_start)
    # Write the total-token sentinel with an on-device fill instead of
    # ``expert_layout[-1] = a.shape[0]`` (which assigns a host Python int to a CUDA
    # element -> a CPU->CUDA ``copy_``). This is a harmless, defensive choice, not a
    # capture enabler: the DeepEP NORMAL (contiguous) path is not CUDA-graph
    # capturable regardless, because upstream ``_mxfp4_w4a8_deepep_normal`` derives
    # ``all_tokens = sum(num_recv_tokens_per_expert)`` from a host list and
    # allocates tensors of that data-dependent shape (and pins+copies the counts to
    # device). ``fill_`` merely keeps this call free of host<->device copies; making
    # the path truly capturable would require fixing that host sum and copy, not
    # this sentinel.
    expert_layout[-1:].fill_(a.shape[0])
    return entry.layer(
        inputs=a,
        input_scale=a_scale,
        expert_layout=expert_layout,
        compute_config=entry.contig_compute_config,
        tuning_config=entry.contig_tuning_config,
        top_k=1,
    )


def _launch_grouped_gemm_masked_humming_entry(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    entry: HummingMxfp4W4A8Weight,
    masked_m: torch.Tensor,
    tuning_config: Any | None = None,
) -> torch.Tensor:
    e, m, k = a.shape
    if e != entry.num_experts:
        raise ValueError(
            f"Humming expert mismatch: entry={entry.num_experts}, input={e}"
        )
    if k != entry.k:
        raise ValueError(f"Humming K mismatch: entry={entry.k}, input={k}")
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        raise TypeError(
            "Humming MXFP4 W4A8 expects FP8 E4M3 activations and float32 scales."
        )
    if a_scale.shape[:2] != (e, m):
        raise ValueError(
            f"activation scale must start with {(e, m)}, got {tuple(a_scale.shape)}"
        )
    a_scale_group_size = k // a_scale.shape[-1]
    if a_scale_group_size != 128:
        raise ValueError(
            f"Humming MXFP4 W4A8 expects activation scale group size 128, "
            f"got {a_scale_group_size}"
        )
    if masked_m.shape[0] != e:
        raise ValueError(f"masked_m shape {tuple(masked_m.shape)} does not match {e=}")

    # W13 uses expert-major scale (DeepEP [E,G,T] layout, no copy); W2 does not
    # (its scale is a fresh row-major [E,T,G] from the SiLU+quant kernel). The
    # entry's masked_compute_config carries the flag set at build time.
    use_expert_major = bool(
        entry.masked_compute_config.get("use_expert_major_input_scale")
    )
    if use_expert_major:
        # `a_scale` should be DeepEP's transpose(1,2) VIEW of the physical
        # [E, G, T] tensor: logical [E, T, G] with stride (G*T, 1, T).
        # transpose(1,2) recovers the original contiguous [E, G, T] as a pure
        # metadata op (no kernel, no copy) -- replacing a_scale.reshape(e*m, G)
        # that PyTorch materialized as direct_copy_kernel_cuda (~0.8ms/step).
        num_groups = a_scale.shape[-1]
        candidate = a_scale.transpose(1, 2)
        if candidate.is_contiguous() and tuple(candidate.shape) == (e, num_groups, m):
            input_scale = candidate  # zero-copy fast path
        else:
            # DeepEP layout wasn't the expected transpose view. Fall back to the
            # standard loader with an explicit copy so we stay correct (the
            # compute_config still requests expert-major, so materialize the
            # [E, G, T] contiguous tensor the loader expects instead).
            _log_humming_warning_once(
                "expert_major_scale_fallback",
                "expert-major scale expected DeepEP's [E,T,G] transpose view "
                f"(stride (G*T,1,T)); got shape={tuple(a_scale.shape)} "
                f"stride={tuple(a_scale.stride())}. Falling back to a contiguous "
                "[E,G,T] copy (reintroduces the direct_copy this optimization "
                "removes).",
            )
            input_scale = a_scale.transpose(1, 2).contiguous()
    else:
        # W2 (and any non-expert-major path): standard row-major [E*T, G].
        input_scale = a_scale.reshape(e * m, a_scale.shape[-1])

    output = entry.layer(
        inputs=a.reshape(e * m, k),
        input_scale=input_scale,
        expert_layout=masked_m.contiguous(),
        compute_config=entry.masked_compute_config,
        tuning_config=(
            entry.masked_tuning_config if tuning_config is None else tuning_config
        ),
        top_k=1,
    )
    return output.view(e, m, entry.n)


# Eagerly load Humming's masked-GEMM cubins at weight-load time, outside CUDA
# graph capture.
#
# The masked (low-latency / decode) kernel's cubin is loaded lazily on its FIRST
# invocation. Under PD-disaggregation decode (deepep low_latency), that first
# invocation lands inside CUDA graph capture's warmup forward. In an 8-rank +
# DeepEP setup this deadlocks asymmetrically: the ranks that hit a cold cubin
# load stall in cuModuleLoad (GPU busy-spinning) while the ranks that already
# captured wait forever at the post-capture custom-all-reduce barrier. Forcing
# local dummy masked calls here -- one per reachable tile config -- loads those
# cubins before capture begins, so capture no longer performs any lazy
# cuModuleLoad. This warmup is purely local (no cross-rank collective), so it is
# deadlock-free.
#
# This is not optional and has no switch: the cubins it compiles are exactly the
# ones decode will need, so skipping it does not save the compilation, it only
# moves it into the window where it deadlocks.


def _warmup_humming_masked_entry(entry: HummingMxfp4W4A8Weight) -> None:
    """Force-load every reachable masked GEMM cubin for ``entry``.

    Raises on any failure. A cubin that cannot be compiled/loaded here cannot be
    compiled at decode time either; letting the process continue would only move
    the failure into CUDA graph capture, where on a multi-rank DeepEP server it
    surfaces as the asymmetric cuModuleLoad deadlock described above rather than
    as an error. Failing during weight loading is clean: the scheduler subprocess
    exits before signalling readiness and the launcher tears the server down.
    """
    device = entry.layer.weight.device
    # Minimal masked input: 1 token routed to expert 0, none elsewhere. The
    # loop below replays it once per reachable tuning config, so every cubin
    # the decode path may select is compiled and loaded here.
    m = 128
    a = torch.zeros(
        (entry.num_experts, m, entry.k),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    num_groups = entry.k // 128
    if entry.masked_compute_config.get("use_expert_major_input_scale"):
        # Match the real W13 input: DeepEP's transpose(1,2) view of a physical
        # [E, G, T] tensor -> logical [E, m, G], stride (G*m, 1, m). Build it
        # from a contiguous [E, G, m] and transpose so the wrapper's
        # transpose(1,2) recovers a contiguous [E, G, m] (no copy).
        a_scale = torch.ones(
            (entry.num_experts, num_groups, m),
            device=device,
            dtype=torch.float32,
        ).transpose(1, 2)
    else:
        a_scale = torch.ones(
            (entry.num_experts, m, num_groups),
            device=device,
            dtype=torch.float32,
        )
    masked_m = torch.zeros((entry.num_experts,), device=device, dtype=torch.int32)
    masked_m[0] = 1
    for tuning_config in _humming_masked_warmup_tuning_configs(entry):
        try:
            _launch_grouped_gemm_masked_humming_entry(
                a,
                a_scale,
                entry,
                masked_m,
                tuning_config=tuning_config,
            )
        except Exception as exc:
            raise RuntimeError(
                "Humming masked-GEMM cubin warmup failed for "
                f"(n={entry.n}, k={entry.k}, num_experts={entry.num_experts}, "
                f"ep_size={entry.ep_size}) with tuning_config={tuning_config!r}. "
                "This config is reachable by the decode path, so it would fail "
                "there too -- inside CUDA graph capture, where a lazy cubin load "
                "deadlocks multi-rank DeepEP servers. Fix the config/table or the "
                "humming wheel instead of continuing."
            ) from exc
    # Surface asynchronous launch errors here, attributed to the warmup, rather
    # than at some unrelated later sync point.
    torch.cuda.synchronize(device)


def replace_mxfp4_w4a8_weights_with_humming(
    layer: torch.nn.Module, prefix: str = ""
) -> None:
    gateup_size = layer.w13_weight.shape[1]
    hidden_size = layer.w13_weight.shape[2] * 2
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    w2_weight_shape = tuple(layer.w2_weight.shape)
    if (
        layer.w2_weight.shape[1] != hidden_size
        or layer.w2_weight.shape[2] * 2 != intermediate_size
    ):
        raise ValueError(
            f"w2 shape mismatch for Humming replacement: {w2_weight_shape=}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    device = layer.w13_weight.device
    timings: dict[str, float] = {}

    with _stage_timer(timings, "w13_build", device):
        w13_entry = _build_humming_weight_entry(
            layer.w13_weight,
            layer.w13_weight_scale_inv,
            n=gateup_size,
            k=hidden_size,
            expert_major_input_scale=True,
            ep_size=getattr(layer, "mxfp4_w4a8_ep_size", None),
        )
    # The entries are plain dataclasses, so register the Humming layers explicitly
    # to keep their parameters visible to PyTorch module traversal and state handling.
    # NOTE: deliberately no gc.collect()/empty_cache() here. Per-layer, they
    # dominated the repack cost (~0.27s of ~0.6s): a full GC
    # pass over the whole module graph plus cudaFree of the freed weight
    # blocks back to the driver, which defeats the caching allocator and
    # forces a fresh cudaMalloc for the next layer's identical-size
    # allocations. The allocator reuses the freed blocks for the next layer
    # instead, keeping the memory peak flat at one extra layer.
    with _stage_timer(timings, "w13_install", device):
        layer.add_module("w13_humming_layer", w13_entry.layer)
        layer.w13_humming_weight = w13_entry
        delattr(layer, "w13_weight")
        delattr(layer, "w13_weight_scale_inv")

    with _stage_timer(timings, "w2_build", device):
        w2_entry = _build_humming_weight_entry(
            layer.w2_weight,
            layer.w2_weight_scale_inv,
            n=hidden_size,
            k=intermediate_size,
            ep_size=getattr(layer, "mxfp4_w4a8_ep_size", None),
        )
    with _stage_timer(timings, "w2_install", device):
        layer.add_module("w2_humming_layer", w2_entry.layer)
        layer.w2_humming_weight = w2_entry
        delattr(layer, "w2_weight")
        delattr(layer, "w2_weight_scale_inv")

    # Load the masked-GEMM cubins now (outside CUDA graph capture) to avoid a
    # lazy cuModuleLoad deadlock during capture on multi-rank decode servers.
    with _stage_timer(timings, "warmup", device):
        _warmup_humming_masked_entry(w13_entry)
        _warmup_humming_masked_entry(w2_entry)

    breakdown = " ".join(f"{k}={v:.3f}s" for k, v in timings.items())
    log_info_on_rank0(
        logger,
        f"Humming repack for MoE layer {prefix}: {breakdown} "
        f"total={sum(timings.values()):.3f}s",
    )


def mxfp4_w4a8_deepep_ll_humming(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    w13_weight: HummingMxfp4W4A8Weight,
    w2_weight: HummingMxfp4W4A8Weight,
) -> torch.Tensor:
    expected_m = int(expected_m)
    if expected_m <= 0:
        raise ValueError(f"expected_m must be positive, got {expected_m}")
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    if not hidden_states.is_cuda or not hidden_states_scale.is_cuda:
        raise ValueError("hidden_states and hidden_states_scale must be CUDA tensors")

    num_experts, padded_m, hidden_size = hidden_states.shape
    if expected_m > padded_m:
        raise ValueError(
            f"expected_m {expected_m} exceeds padded M dimension {padded_m}"
        )
    if masked_m.shape[0] != num_experts:
        raise ValueError(
            f"masked_m shape {tuple(masked_m.shape)} does not match {num_experts=}"
        )
    if w13_weight.num_experts != num_experts or w2_weight.num_experts != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")
    if w13_weight.k != hidden_size:
        raise ValueError(f"w13 K mismatch: weight={w13_weight.k}, {hidden_size=}")
    gateup_size = w13_weight.n
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w2_weight.n != hidden_size or w2_weight.k != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: n={w2_weight.n}, k={w2_weight.k}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    gateup_output = _launch_grouped_gemm_masked_humming_entry(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        masked_m,
        tuning_config=_select_humming_masked_tuning_config(w13_weight, expected_m),
    )

    down_input = torch.empty(
        (num_experts, padded_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale = torch.empty(
        (num_experts, padded_m, intermediate_size // 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        128,
        masked_m,
    )
    del gateup_output

    return _launch_grouped_gemm_masked_humming_entry(
        down_input,
        down_input_scale,
        w2_weight,
        masked_m,
        tuning_config=_select_humming_masked_tuning_config(w2_weight, expected_m),
    )


def mxfp4_w4a8_deepep_normal_humming(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    expert_start: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    w13_weight: HummingMxfp4W4A8Weight,
    w2_weight: HummingMxfp4W4A8Weight,
) -> torch.Tensor:
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected FP8 hidden_states, got {hidden_states.dtype}")
    if hidden_states_scale.dtype != torch.float32:
        raise TypeError(
            f"expected float32 hidden_states_scale, got {hidden_states_scale.dtype}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("hidden_states_scale", hidden_states_scale),
        ("expert_start", expert_start),
        ("num_tokens_per_expert", num_tokens_per_expert),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")

    total_m, hidden_size = hidden_states.shape
    if hidden_states_scale.shape[0] != total_m:
        raise ValueError(
            f"hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
            f"does not match {tuple(hidden_states.shape)}"
        )
    num_experts = num_tokens_per_expert.shape[0]
    if expert_start.shape[0] != num_experts:
        raise ValueError("expert_start and num_tokens_per_expert shape mismatch")
    if w13_weight.num_experts != num_experts or w2_weight.num_experts != num_experts:
        raise ValueError("weight expert dimension does not match DeepEP dispatch")
    if w13_weight.k != hidden_size:
        raise ValueError(f"w13 K mismatch: weight={w13_weight.k}, {hidden_size=}")
    gateup_size = w13_weight.n
    intermediate_size = gateup_size // 2
    if gateup_size % 2 != 0:
        raise ValueError(f"w13 gate/up dimension must be even, got {gateup_size}")
    if intermediate_size % 128 != 0:
        raise ValueError(
            f"intermediate size must be divisible by FP8 group size 128, "
            f"got {intermediate_size}"
        )
    if w2_weight.n != hidden_size or w2_weight.k != intermediate_size:
        raise ValueError(
            f"w2 shape mismatch: n={w2_weight.n}, k={w2_weight.k}, "
            f"{hidden_size=}, {intermediate_size=}"
        )

    a_scale_group_size = hidden_size // hidden_states_scale.shape[-1]
    gateup_output = _launch_grouped_gemm_contig_humming_entry(
        hidden_states,
        hidden_states_scale,
        w13_weight,
        expert_start,
        num_tokens_per_expert,
        a_scale_group_size,
    )

    # Activation + requantization feeding the W2 GEMM. Use the same pair as the
    # DeepGEMM contiguous path (silu_and_mul, then per-token-group quant) rather
    # than the masked SiLU kernel. That kernel grids as
    # (hidden_blocks, BLOCK_NUM_PER_EXPERT, num_experts) with num_warps=1, and
    # BLOCK_NUM_PER_EXPERT is 64 when num_experts < 4, so feeding it one
    # synthetic expert leaves 64 blocks on the token axis each looping
    # total_m/64 times -- far too little parallelism at prefill-sized total_m.
    # Both compute the scale as absmax/fp8_max over each group of 128 along the
    # row, so the result is numerically equivalent.
    from sgl_kernel import silu_and_mul

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    down_input = torch.empty(
        (total_m, intermediate_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    silu_and_mul(gateup_output.view(-1, gateup_size), down_input)
    del gateup_output

    down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(down_input, 128)
    del down_input

    a_scale_group_size = intermediate_size // down_input_scale.shape[-1]
    return _launch_grouped_gemm_contig_humming_entry(
        down_input_fp8,
        down_input_scale,
        w2_weight,
        expert_start,
        num_tokens_per_expert,
        a_scale_group_size,
    )
