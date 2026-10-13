"""Numerical and CUDA graph checks for the private MiMo SM90 W4A8 kernels.

Requires the DeepGEMM build used by sglang-private/v0.5.10-xm, with matching
PyTorch and GLIBC. Run from the repository root, for example::

    PYTHONPATH=python SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 python \
        test/manual/layers/moe/verify_mimo_deepgemm_sm90.py \
        --models flash pro --benchmark --json-output /tmp/mimo-sm90.json

Use --binding direct to check the wheel before importing SGLang. Timings are
individual GEMMs with synthetic inputs, not end-to-end model throughput.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
from pathlib import Path

import torch


def runtime_info():
    import deep_gemm

    libc = ctypes.CDLL("libc.so.6")
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    return {
        "glibc": libc.gnu_get_libc_version().decode(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deep_gemm": deep_gemm.__version__,
        "deep_gemm_path": deep_gemm.__file__,
    }


def _spawn_probe(queue):
    queue.put(runtime_info())


def check_spawn():
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawn_probe, args=(queue,))
    proc.start()
    proc.join(30)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise RuntimeError("Spawned Python did not finish the runtime check")
    if proc.exitcode != 0:
        raise RuntimeError(f"Spawned Python failed with exit code {proc.exitcode}")
    result = queue.get(timeout=5)
    queue.close()
    return result


def make_weights(experts, n, k):
    packed = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda"
    )
    scales_u8 = torch.randint(
        122, 127, (experts, n, k // 32), dtype=torch.uint8, device="cuda"
    )
    scales_fp32 = scales_u8.view(torch.float8_e8m0fnu).float()
    scales_mn = scales_u8.transpose(-1, -2).contiguous().transpose(-1, -2)
    # Decode the checkpoint format independently of DeepGEMM's quantizer.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    reference = lut[codes] * scales_fp32.repeat_interleave(32, dim=-1)
    return packed.view(torch.int8), scales_mn, scales_fp32, reference


def make_activations(shape, column_major):
    import deep_gemm

    k = shape[-1]
    values = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quant, scales = deep_gemm.per_token_cast_to_fp8(values.reshape(-1, k), False)
    quant = quant.reshape(shape)
    scales = scales.reshape(*shape[:-1], k // 128)
    if column_major:
        scales = scales.transpose(-1, -2).contiguous().transpose(-1, -2)
    reference = quant.float() * scales.repeat_interleave(128, dim=-1)
    return quant, scales, reference


def make_call(binding, masked, lhs, rhs, output, layout, expected_m):
    if binding == "sglang":
        from sglang.srt.layers import deep_gemm_wrapper as wrapper

        if not wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError("SGLang could not enable DeepGEMM in this environment")
        if masked:
            return lambda: wrapper.grouped_gemm_nt_f8fp4bf16_masked(
                lhs, rhs, output, layout, expected_m,
                recipe_a=(1, 128), recipe_b=(1, 32),
            )
        return lambda: wrapper.grouped_gemm_nt_f8fp4bf16_contig(
            lhs, rhs, output, layout, recipe_a=(1, 128), recipe_b=(1, 32)
        )

    import deep_gemm

    if masked:
        return lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma(
            lhs, rhs, output, layout, expected_m, gran_k_a=128, gran_k_b=32
        )
    return lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma(
        lhs, rhs, output, layout, gran_k_b=32
    )


def check_output(output, reference):
    if not torch.isfinite(output).all():
        raise AssertionError("GEMM produced NaN or Inf")
    relative_l2 = (
        (output.float() - reference).norm() / reference.norm().clamp_min(1e-8)
    ).item()
    if relative_l2 >= 0.01:
        raise AssertionError(f"Relative L2 error {relative_l2:.6f} exceeds 0.01")
    return relative_l2


def graph_check_and_bench(call, benchmark):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    unroll = 20 if benchmark else 1
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            call()
    graph.replay()
    torch.cuda.synchronize()
    if not benchmark:
        return None
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (10 * unroll)


def verify_shape(args, model, projection, n, k):
    experts = args.experts
    weight, scales_mn, scales_fp32, weight_ref = make_weights(experts, n, k)
    cases = [(False, 128)] + [(True, m) for m in (1, 96, 97, 128)]
    for masked, expected_m in cases:
        for column_major in (False, True):
            shape = (experts, 128, k) if masked else (experts * 128, k)
            quant, scales, activation_ref = make_activations(shape, column_major)
            if masked:
                counts = [max(0, expected_m - i * 3) for i in range(experts)]
                layout = torch.tensor(counts, dtype=torch.int32, device="cuda")
                weight_scales = scales_mn if expected_m <= 96 else scales_fp32
                output = torch.empty(experts, 128, n, dtype=torch.bfloat16, device="cuda")
                reference = torch.cat(
                    [activation_ref[i, :c] @ weight_ref[i].t() for i, c in enumerate(counts)]
                )

                def valid_output():
                    return torch.cat([output[i, :c] for i, c in enumerate(counts)])

            else:
                layout = torch.arange(experts, dtype=torch.int32, device="cuda").repeat_interleave(128)
                weight_scales = scales_mn
                output = torch.empty(experts * 128, n, dtype=torch.bfloat16, device="cuda")
                reference = torch.cat(
                    [activation_ref[i * 128 : (i + 1) * 128] @ weight_ref[i].t() for i in range(experts)]
                )

                def valid_output():
                    return output

            call = make_call(
                args.binding, masked, (quant, scales), (weight, weight_scales),
                output, layout, expected_m,
            )
            call()
            error = check_output(valid_output(), reference)
            latency_us = graph_check_and_bench(call, args.benchmark)
            graph_error = check_output(valid_output(), reference)
            result = {
                "model": model, "projection": projection, "n": n, "k": k,
                "experts": experts, "masked": masked, "expected_m": expected_m,
                "activation_scales": "column_major" if column_major else "row_major",
                "relative_l2": error, "graph_relative_l2": graph_error,
                "latency_us": latency_us,
            }
            print(json.dumps(result), flush=True)
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("smoke", "flash", "pro"), nargs="+", default=["flash", "pro"])
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--binding", choices=("direct", "sglang"), default="sglang")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.experts < 1:
        parser.error("--experts must be positive")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    parent, child = runtime_info(), check_spawn()
    if parent != child:
        raise AssertionError(f"Spawned Python uses a different runtime: {child}")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This validator targets SM90 (H20/H100/H200)")
    parent.update(gpu=torch.cuda.get_device_name(), binding=args.binding)
    print(json.dumps({"runtime": parent, "spawn": "passed"}), flush=True)
    torch.manual_seed(20260929)
    results = []
    for model in args.models:
        hidden, intermediate = {"smoke": (256, 256), "flash": (4096, 2048), "pro": (6144, 2048)}[model]
        for projection, n, k in (("gate_up", intermediate * 2, hidden), ("down", hidden, intermediate)):
            results.extend(verify_shape(args, model, projection, n, k))
    report = {"runtime": parent, "spawn_passed": True, "cases": results}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(results)} numerical and CUDA graph cases.", flush=True)


if __name__ == "__main__":
    main()
