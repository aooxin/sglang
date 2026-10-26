"""Force-compare ATen, cudaMemcpyBatchAsync, and cudaMemcpy3D H2D copies.

The 800 MiB cases use the real SGLang layer-first/page-first pools and the
allocator sizes from the previous unit test. The 200 MiB case keeps the
historical benchmark's small shapes. Both the best (single joint run) and worst
(one page per joint run) destination orders are measured.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LAYERS = 10
K_BYTES = 192 * 2
V_BYTES = 128 * 2
TOKEN_BYTES = (K_BYTES + V_BYTES) * LAYERS
REAL_CAPACITY = 10_345_984

CASES = {
    "800MiB-ps64": dict(tokens=131_200, page_size=64, capacity=REAL_CAPACITY, real_pool=True),
    "800MiB-ps128": dict(tokens=131_200, page_size=128, capacity=REAL_CAPACITY, real_pool=True),
    "200MiB-ps64": dict(tokens=32_768, page_size=64, capacity=None, real_pool=False),
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def configure_cpu():
    allowed = os.sched_getaffinity(0)
    node = Path("/sys/devices/system/node/node0/cpulist")
    if node.exists():
        selected = set()
        for item in node.read_text().strip().split(","):
            parts = [int(x) for x in item.split("-")]
            selected.update(range(parts[0], parts[-1] + 1))
        selected &= allowed
        if selected:
            os.sched_setaffinity(0, selected)
    torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))


def environment():
    import importlib.metadata

    packages = {}
    for name in ("torch", "transfer-kernel", "sglang-kernel"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "time": time.time(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "gpu": torch.cuda.get_device_properties(0).__repr__(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
    }


def page_tokens(pages, page_size):
    pages = torch.as_tensor(pages, dtype=torch.int64)
    return (pages[:, None] * page_size + torch.arange(page_size)).flatten()


def joint_runs(host_indices, dst_indices, page_size):
    host = host_indices.reshape(-1, page_size)[:, 0]
    dst = dst_indices.reshape(-1, page_size)[:, 0]
    count = 1
    for i in range(1, len(host)):
        if host[i] != host[i - 1] + page_size or dst[i] != dst[i - 1] + page_size:
            count += 1
    return count


def layer_groups(src, dst):
    groups = []
    begin = 0
    while begin < len(src):
        row_bytes = src[begin].numel() // src[begin].size(0) * src[begin].element_size()
        end = begin + 1
        src_stride = None
        dst_stride = None
        if end < len(src):
            src_stride = src[end].data_ptr() - src[begin].data_ptr()
            dst_stride = dst[end].data_ptr() - dst[begin].data_ptr()
            while (
                end < len(src)
                and src[end].numel() // src[end].size(0) * src[end].element_size()
                == row_bytes
                and src[end].data_ptr() - src[end - 1].data_ptr() == src_stride
                and dst[end].data_ptr() - dst[end - 1].data_ptr() == dst_stride
            ):
                end += 1
        groups.append(
            {
                "start": begin,
                "end": end,
                "layers": end - begin,
                "row_bytes": row_bytes,
                "src_stride": src_stride,
                "dst_stride": dst_stride,
            }
        )
        begin = end
    return groups


def buffer_info(src, dst):
    return {
        "src_layers": len(src),
        "dst_layers": len(dst),
        "src_storages": len({x.untyped_storage().data_ptr() for x in src}),
        "dst_storages": len({x.untyped_storage().data_ptr() for x in dst}),
        "src_pinned": all(x.is_pinned() for x in src),
        "src_contiguous": all(x.is_contiguous() for x in src),
        "dst_contiguous": all(x.is_contiguous() for x in dst),
        "src_total_bytes": sum(x.numel() * x.element_size() for x in src),
        "dst_total_bytes": sum(x.numel() * x.element_size() for x in dst),
        "layer_groups": layer_groups(src, dst),
    }


def allocate_historical(case, layout, pattern):
    page_size = case["page_size"]
    pages = case["tokens"] // page_size
    host_pages = torch.arange(pages, dtype=torch.int64)
    if pattern == "contiguous":
        dst_pages = torch.arange(pages, 2 * pages, dtype=torch.int64)
        device_pages = 2 * pages
    else:
        dst_pages = torch.arange(4 * pages - 2, pages * 2 - 2, -2, dtype=torch.int64)
        device_pages = 4 * pages - 1
    host_indices = page_tokens(host_pages, page_size)
    dst_indices = page_tokens(dst_pages, page_size)
    host_tokens = pages * page_size
    device_tokens = device_pages * page_size
    if layout == "LF":
        src = [
            torch.empty(host_tokens, width, dtype=torch.uint8, pin_memory=True)
            for width in [K_BYTES] * LAYERS + [V_BYTES] * LAYERS
        ]
        dst = [
            torch.empty(device_tokens, width, dtype=torch.uint8, device="cuda")
            for width in [K_BYTES] * LAYERS + [V_BYTES] * LAYERS
        ]
    else:
        src = [torch.empty(host_tokens, TOKEN_BYTES, dtype=torch.uint8, pin_memory=True)]
        dst = [torch.empty(device_tokens, TOKEN_BYTES, dtype=torch.uint8, device="cuda")]
    return src, dst, host_indices, dst_indices, 0


def allocate_real_pool(case, layout):
    from sglang.srt.mem_cache.memory_pool import (
        MHAPageFirstTokenToKVPool,
        MHATokenToKVPool,
    )
    from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

    page_size = case["page_size"]
    tokens = case["tokens"]
    pages = tokens // page_size
    host_indices = torch.arange(page_size, page_size + tokens, dtype=torch.int64)
    patterns = {
        "contiguous": torch.arange(1, pages + 1, dtype=torch.int64),
        "fragmented": torch.arange(pages, 0, -1, dtype=torch.int64),
    }
    indices = {
        name: page_tokens(device_pages, page_size) for name, device_pages in patterns.items()
    }
    cls = MHATokenToKVPool if layout == "LF" else MHAPageFirstTokenToKVPool
    device_pool = cls(
        size=case["capacity"],
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=192,
        v_head_dim=128,
        layer_num=LAYERS,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    host_pool = MHATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=page_size,
        layout=device_pool.kv_transfer_layout,
    )
    if layout == "LF":
        src = host_pool.k_data_refs + host_pool.v_data_refs
        dst = device_pool.k_buffer + device_pool.v_buffer
    else:
        src = [host_pool.k_buffer, host_pool.v_buffer]
        dst = list(device_pool.get_page_first_raw_buffers())
    return src, dst, host_indices, indices, page_size, host_pool, device_pool


def fill_sources(src, start, tokens):
    torch.manual_seed(20261026)
    for tensor in src:
        tensor[start : start + tokens].random_(1, 128)
    torch.cuda.synchronize()


def check_copy(src, dst, host_indices, dst_indices, page_size, mode):
    transfer = __import__("transfer_kernel.kvcacheio", fromlist=["transfer_kv_direct_mode"])
    dst_gpu = dst_indices.cuda()
    for target in dst:
        target[0:page_size].zero_()
        target.index_fill_(0, dst_gpu, 254)
    transfer.transfer_kv_direct_mode(src, dst, host_indices, dst_indices, page_size, mode)
    torch.cuda.synchronize()
    for source, target in zip(src, dst):
        got = target.index_select(0, dst_gpu).cpu()
        want = source.index_select(0, host_indices)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert bool((target[0] == 0).all()), "padding row changed"


def measure(transfer, src, dst, host_indices, dst_indices, page_size, mode, stream, warmup, iters):
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            transfer(src, dst, host_indices, dst_indices, page_size, mode)
    torch.cuda.synchronize()
    samples = []
    cpu_samples = []
    with torch.cuda.stream(stream):
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            began = time.perf_counter()
            stats = transfer(src, dst, host_indices, dst_indices, page_size, mode)
            cpu_samples.append((time.perf_counter() - began) * 1000)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples), statistics.median(cpu_samples), stats, samples, cpu_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--layouts", nargs="+", choices=["LF", "PF"], default=["LF", "PF"])
    parser.add_argument("--modes", nargs="+", choices=["plain", "batch", "3d"], default=["plain", "batch", "3d"])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["contiguous", "fragmented"],
        default=["contiguous", "fragmented"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configure_cpu()
    torch.cuda.set_device(0)
    save(out / "environment.json", environment())
    from transfer_kernel.kvcacheio import transfer_kv_direct_mode

    rows = []
    checks = []
    unsupported = []
    status = {"state": "starting", "completed": 0}
    save(out / "status.json", status)

    def run_layout(case_name, case, layout, src, dst, host_indices, index_sets, page_size):
        nonlocal status
        start_token = page_size if case["real_pool"] else 0
        fill_sources(src, start_token, case["tokens"])
        info = buffer_info(src, dst)
        save(out / f"buffers-{case_name}-{layout}.json", info)
        print(f"H2D3D_BUFFERS {case_name} {layout} {json.dumps(info)}", flush=True)
        stream = torch.cuda.Stream()
        active_modes = list(args.modes)
        patterns = list(index_sets)
        for round_id in range(1, args.rounds + 1):
            modes = active_modes if round_id % 2 else list(reversed(active_modes))
            for mode in modes:
                if mode in {x["mode"] for x in unsupported if x["case"] == case_name and x["layout"] == layout}:
                    continue
                for pattern in patterns:
                    dst_indices = index_sets[pattern]
                    key = (case_name, layout, mode, pattern)
                    try:
                        if round_id == 1:
                            check_copy(src, dst, host_indices, dst_indices, page_size, mode)
                            checks.append(
                                {
                                    "case": case_name,
                                    "layout": layout,
                                    "mode": mode,
                                    "pattern": pattern,
                                    "passed": True,
                                }
                            )
                            save(out / "correctness.json", checks)
                        median_ms, cpu_ms, stats, samples, cpu_samples = measure(
                            transfer_kv_direct_mode,
                            src,
                            dst,
                            host_indices,
                            dst_indices,
                            page_size,
                            mode,
                            stream,
                            args.warmup,
                            args.iters,
                        )
                    except RuntimeError as error:
                        torch.cuda.synchronize()
                        unsupported.append(
                            {
                                "case": case_name,
                                "layout": layout,
                                "mode": mode,
                                "pattern": pattern,
                                "error": str(error),
                            }
                        )
                        save(out / "unsupported.json", unsupported)
                        print(f"H2D3D_UNSUPPORTED {case_name} {layout} {mode} {error}", flush=True)
                        break
                    copied_bytes = case["tokens"] * TOKEN_BYTES
                    row = {
                        "case": case_name,
                        "layout": layout,
                        "mode": mode,
                        "pattern": pattern,
                        "round": round_id,
                        "runs": joint_runs(host_indices, dst_indices, page_size),
                        "median_ms": median_ms,
                        "cpu_median_ms": cpu_ms,
                        "bandwidth_gb_s": copied_bytes / median_ms / 1e6,
                        "copied_bytes": copied_bytes,
                        **stats,
                    }
                    rows.append(row)
                    with (out / "samples.jsonl").open("a") as handle:
                        handle.write(
                            json.dumps(
                                dict(row, samples_ms=samples, cpu_samples_ms=cpu_samples)
                            )
                            + "\n"
                        )
                    save(out / "results.json", rows)
                    status = {"state": "benchmarking", "completed": len(rows), "latest": row}
                    save(out / "status.json", status)
                    print(f"H2D3D_ROW {json.dumps(row)}", flush=True)

    for case_name in args.cases:
        case = CASES[case_name]
        if case["real_pool"]:
            for layout in args.layouts:
                src, dst, host_indices, index_sets, page_size, host_pool, device_pool = allocate_real_pool(
                    case, layout
                )
                try:
                    run_layout(case_name, case, layout, src, dst, host_indices, index_sets, page_size)
                finally:
                    torch.cuda.synchronize()
                    del src, dst, host_indices, index_sets, host_pool, device_pool
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            for layout in args.layouts:
                for pattern in args.patterns:
                    src, dst, host_indices, dst_indices, page_size = allocate_historical(
                        case, layout, pattern
                    )
                    run_layout(
                        case_name,
                        case,
                        layout,
                        src,
                        dst,
                        host_indices,
                        {pattern: dst_indices},
                        page_size,
                    )
                    del src, dst, host_indices, dst_indices
                    gc.collect()
                    torch.cuda.empty_cache()

    summary = []
    keys = sorted({(x["case"], x["layout"], x["mode"], x["pattern"]) for x in rows})
    for key in keys:
        selected = [x for x in rows if (x["case"], x["layout"], x["mode"], x["pattern"]) == key]
        sample_medians = [x["median_ms"] for x in selected]
        cpu_medians = [x["cpu_median_ms"] for x in selected]
        summary.append(
            {
                "case": key[0],
                "layout": key[1],
                "mode": key[2],
                "pattern": key[3],
                "rounds": len(selected),
                "median_ms": statistics.median(sample_medians),
                "cpu_median_ms": statistics.median(cpu_medians),
                "bandwidth_gb_s": selected[0]["copied_bytes"] / statistics.median(sample_medians) / 1e6,
                "runs": selected[0]["runs"],
                "api_calls": selected[0]["api_calls"],
                "copy_segments": selected[0]["copy_segments"],
                "layer_groups": selected[0]["layer_groups"],
            }
        )
    save(out / "summary.json", summary)
    save(
        out / "status.json",
        {
            "state": "complete",
            "formal_cases": len(rows),
            "checks": len(checks),
            "unsupported": len(unsupported),
            "files": sorted(x.name for x in out.iterdir() if x.is_file()),
        },
    )
    for row in summary:
        print(f"H2D3D_SUMMARY {json.dumps(row)}", flush=True)
    print("H2D3D_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
