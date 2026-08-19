"""Benchmark: FA3 vs MiMo forward_extend — attention + KV pool ops (GQA).

Compares the full per-layer, per-chunk GPU cost of the two production
forward_extend paths, using the REAL memory-pool classes so the measured
set/get logic is exactly what runs in serving:

  fa3 path (FlashAttentionBackend + MHATokenToKVPool):
      fa3_set_kv    — MHATokenToKVPool.set_kv_buffer: scatter the chunk's
                      BF16 K/V into the pool (fused store_cache kernel when
                      head_dim == v_head_dim, otherwise two indexed scatters)
      fa3_get_kv    — MHATokenToKVPool.get_kv_buffer: returns tensor VIEWS of
                      the pool buffers; zero GPU work (measured to prove it —
                      the paged attention kernel reads the pool directly)
      paged_attn    — flash_attn_with_kvcache on the paged BF16 pool

  mimo path (MiMoFlashAttentionBackend + MiMoTokenToKVPool):
      mimo_set_kv   — MiMoTokenToKVPool.set_kv_buffer: per-token FP8 quantize
                      + pack the chunk into quantized pages
                      (ekk mimo_v2_gqa_quant_store_kv)
      mimo_get_kv   — MiMoTokenToKVPool.dequant_tokens: gather + dequantize
                      ALL attended tokens (prefix + chunk) into the BF16
                      workspace, paid EVERY layer
                      (ekk mimo_v2_gqa_dequant_gather_kv)
      varlen_attn   — flash_attn_varlen_func on the packed BF16 workspace

Timing methodology (GPU time only, reported as mean ± std):
  - Attention kernels (ms-scale): per-iteration CUDA events with an L2 flush
    between iterations; CPU launch overhead is negligible at this scale.
  - Pool ops (us~ms-scale): CPU launch gaps would pollute per-iteration event
    timing, so each op is captured N times into ONE CUDA graph (interleaved
    with L2 flushes), the replay is event-timed, and a flush-only graph is
    subtracted:  t_op = (t(graph[op+flush]) - t(graph[flush])) / N.
    Graph replay has no per-kernel CPU launch cost -> pure GPU time.

The MiMo pool implements the fixed MiMoV2 full-layer layout: head_dim = 192
(64 rope BF16 + 128 nope FP8), v_head_dim = 128, page_size = 64. The mimo
path is benchmarked only under exactly this configuration (and with
ez_kernel_kit installed); otherwise only the fa3 path runs.

Usage:
    python3 bench_fa3_paged_vs_varlen_extend.py \
        --seq-lens 65536 131072 300000 --page-size 64 \
        --num-q-heads 16 --num-kv-heads 1 --head-dim 192 --v-head-dim 128 \
        --extend-len 16384
"""

import argparse
import functools
import itertools
import math
import statistics

import torch
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils import get_device_capability
from sglang.utils import is_in_ci

# Flush every print immediately: the ekk C++ ops can hard-crash the process
# (abort/segfault), which would otherwise swallow buffered output and make
# the failure location impossible to see.
print = functools.partial(print, flush=True)

IS_CI = is_in_ci()

try:
    # Importing the pool module also imports ez_kernel_kit (hard dependency).
    from sglang.srt.mem_cache.mimo_memory_pool import MiMoTokenToKVPool

    HAS_MIMO = True
    MIMO_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 — .so load failures are not ImportError
    MiMoTokenToKVPool = None
    HAS_MIMO = False
    MIMO_IMPORT_ERROR = repr(e)

# MiMo fixed full-layer layout (see mimo_memory_pool.py).
MIMO_HEAD_DIM = 192  # 64 rope (BF16) + 128 nope (FP8)
MIMO_V_DIM = 128
MIMO_PAGE_SIZE = 64
# MiMoV2 config.json value; rope_dim = int(head_dim * factor) = 64.
MIMO_PARTIAL_ROTARY_FACTOR = 0.334

L2_FLUSH_BYTES = 128 * 1024 * 1024


def _cdiv(a, b):
    return (a + b - 1) // b


# Case construction


def _build_case(
    batch_size: int,
    seq_len: int,
    extend_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Build one benchmark case around real pool objects.

    Per request: kv_len = seq_len, q_len = extend_len (the last chunk).
    The whole sequence's K/V are written into both pools up front (as earlier
    chunked-prefill forwards would have done); the measured set ops then
    re-write the current chunk, exactly like one production forward_extend.
    """
    q_len = extend_len
    total_q = batch_size * q_len
    total_k = batch_size * seq_len

    q = torch.randn(total_q, num_q_heads, head_dim, dtype=dtype, device=device)
    k_packed = torch.randn(total_k, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_packed = torch.randn(
        total_k, num_kv_heads, v_head_dim, dtype=dtype, device=device
    )

    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    # Randomly permuted pages: realistic scattered radix-cache layout.
    pages_per_seq = _cdiv(seq_len, page_size)
    num_pages = batch_size * pages_per_seq
    perm = torch.randperm(num_pages, dtype=torch.int32, device=device)
    page_table = perm.view(batch_size, pages_per_seq)

    # Token slot ids consistent with the page table:
    # token j of request i -> slot page_table[i, j // ps] * ps + j % ps.
    # `slots` = all attended tokens (MiMo _get_ragged_token_ids);
    # `slots_chunk` = the chunk's out_cache_loc (set_kv_buffer targets).
    j = torch.arange(seq_len, device=device)
    slots = (
        page_table.to(torch.int64)[:, j // page_size] * page_size + j % page_size
    ).view(-1)
    chunk_mask = (j >= seq_len - q_len).repeat(batch_size)
    slots_chunk = slots[chunk_mask]
    k_chunk = k_packed[chunk_mask].contiguous()
    v_chunk = v_packed[chunk_mask].contiguous()

    # Real FA3 pool: MHATokenToKVPool (BF16, paged view for the kernel).
    mha_pool = MHATokenToKVPool(
        size=num_pages * page_size,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        v_head_dim=v_head_dim,
        start_layer=0,
        end_layer=0,
        enable_alt_stream=False,
    )
    # Populate the whole sequence (prefix + chunk) through the real API.
    mha_pool.set_kv_buffer(None, slots, k_packed, v_packed, layer_id_override=0)
    key_buf, value_buf = mha_pool.get_kv_buffer(0)
    key_cache = key_buf.view(-1, page_size, num_kv_heads, head_dim)
    value_cache = value_buf.view(-1, page_size, num_kv_heads, v_head_dim)

    case = dict(
        q=q,
        k_packed=k_packed,
        v_packed=v_packed,
        k_chunk=k_chunk,
        v_chunk=v_chunk,
        slots=slots,
        slots_chunk=slots_chunk,
        mha_pool=mha_pool,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_len,
        max_seqlen_k=seq_len,
        softmax_scale=head_dim**-0.5,
    )

    # Real MiMo pool: MiMoTokenToKVPool (FP8 pages) + dequant workspace.
    mimo_ok = (
        HAS_MIMO
        and head_dim == MIMO_HEAD_DIM
        and v_head_dim == MIMO_V_DIM
        and page_size == MIMO_PAGE_SIZE
    )
    if mimo_ok:
        print("  building MiMo quantized pool ...")
        mimo_pool = MiMoTokenToKVPool(
            size=num_pages * page_size,
            page_size=page_size,
            dtype=dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            v_head_dim=v_head_dim,
            start_layer=0,
            end_layer=0,
            enable_alt_stream=False,
            partial_rotary_factor=MIMO_PARTIAL_ROTARY_FACTOR,
        )
        mimo_pool.set_kv_buffer(None, slots, k_packed, v_packed, layer_id_override=0)
        # Surface async CUDA errors here instead of at a random later point.
        torch.cuda.synchronize()
        # Persistent dequant workspace (backend _MiMoDequantWorkspace).
        k_ws = torch.empty(
            total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        v_ws = torch.empty(
            total_k, num_kv_heads, v_head_dim, dtype=torch.bfloat16, device=device
        )
        case.update(mimo_pool=mimo_pool, k_ws=k_ws, v_ws=v_ws)
    case["mimo_ok"] = mimo_ok
    return case


# Measured ops, one production forward_extend step each


def _fa3_set_kv(case):
    """MHATokenToKVPool.set_kv_buffer: write the chunk's BF16 K/V."""
    case["mha_pool"].set_kv_buffer(
        None, case["slots_chunk"], case["k_chunk"], case["v_chunk"],
        layer_id_override=0,
    )  # fmt: skip


def _fa3_get_kv(case):
    """MHATokenToKVPool.get_kv_buffer: pure view return, no GPU work."""
    case["mha_pool"].get_kv_buffer(0)


def _fa3_paged_attn(case):
    """FlashAttentionBackend.forward_extend attention call (paged KV)."""
    return flash_attn_with_kvcache(
        q=case["q"],
        k_cache=case["key_cache"],
        v_cache=case["value_cache"],
        page_table=case["page_table"],
        cache_seqlens=case["cache_seqlens"],
        cu_seqlens_q=case["cu_seqlens_q"],
        cu_seqlens_k_new=case["cu_seqlens_k"],
        max_seqlen_q=case["max_seqlen_q"],
        softmax_scale=case["softmax_scale"],
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        k_descale=None,
        v_descale=None,
        return_softmax_lse=False,
        num_splits=0,
    )


def _mimo_set_kv(case):
    """MiMoTokenToKVPool.set_kv_buffer: FP8 quantize + pack the chunk."""
    case["mimo_pool"].set_kv_buffer(
        None, case["slots_chunk"], case["k_chunk"], case["v_chunk"],
        layer_id_override=0,
    )  # fmt: skip


def _mimo_get_kv(case):
    """MiMoTokenToKVPool.dequant_tokens: dequant ALL attended tokens."""
    case["mimo_pool"].dequant_tokens(
        0, case["slots"], k_out=case["k_ws"], v_out=case["v_ws"], global_layer_id=0
    )


def _mimo_varlen_attn(case, use_ws=True):
    """MiMoFlashAttentionBackend.forward_extend attention call (packed)."""
    use_ws = use_ws and case["mimo_ok"]
    return flash_attn_varlen_func(
        q=case["q"],
        k=case["k_ws"] if use_ws else case["k_packed"],
        v=case["v_ws"] if use_ws else case["v_packed"],
        cu_seqlens_q=case["cu_seqlens_q"],
        cu_seqlens_k=case["cu_seqlens_k"],
        max_seqlen_q=case["max_seqlen_q"],
        max_seqlen_k=case["max_seqlen_k"],
        softmax_scale=case["softmax_scale"],
        causal=True,
        softcap=0.0,
    )


# Timing helpers (GPU time only, mean +- std)


def _summarize(times):
    return dict(
        mean=statistics.fmean(times),
        std=statistics.stdev(times) if len(times) > 1 else 0.0,
        median=statistics.median(times),
        min=min(times),
        max=max(times),
        n=len(times),
    )


def _bench_event_stats(fn, warmup, max_iters, min_iters, max_bench_time_s):
    """Per-iteration CUDA-event timing with L2 flush between iterations.

    Suitable for ms-scale kernels where CPU launch overhead is negligible.
    Iteration count adapts to the kernel duration (total timed wall clock
    ~ max_bench_time_s, clamped to [min_iters, max_iters]).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    est_ms = max(start.elapsed_time(end), 1e-3)
    iters = int(max(min_iters, min(max_iters, max_bench_time_s * 1000.0 / est_ms)))

    cache = torch.empty(L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    del cache
    return _summarize(times)


def _time_graph_replays(graph, replays):
    times = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _bench_graph_stats(fn, warmup, replays, target_replay_ms=25.0, max_inner=200):
    """CUDA-graph timing for short kernels — pure GPU time, no launch gaps.

    `fn` is captured `inner` times into one graph, interleaved with L2-flush
    (cache.zero_()) nodes so every instance runs cold-cache. A flush-only
    graph with the same number of flush nodes is timed separately and
    subtracted:

        t_op = (t(op_graph) - t(flush_graph)) / inner

    Replay timing uses CUDA events around graph.replay(); a graph replay has
    no per-kernel CPU launch cost, so the result is GPU execution time only.
    Stats (mean ± std) are computed over `replays` replays; std is propagated
    through the subtraction as sqrt(var_op + var_flush) / inner.

    An op that launches no kernels (e.g. get_kv_buffer, a pure view return)
    yields t_op ~ 0 by construction.
    """
    cache = torch.empty(L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")

    # Warmup on a side stream (required before graph capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # One-shot estimate to pick `inner` so a replay lasts ~target_replay_ms.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    est_ms = max(start.elapsed_time(end), 1e-3)
    flush_ms = L2_FLUSH_BYTES / 2.0e12 * 1e3  # rough write-bw guess, only a hint
    inner = int(max(1, min(max_inner, target_replay_ms / (est_ms + flush_ms))))

    op_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(op_graph):
        for _ in range(inner):
            cache.zero_()
            fn()
    flush_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flush_graph):
        for _ in range(inner):
            cache.zero_()

    # Graph warmup replays.
    op_graph.replay()
    flush_graph.replay()
    torch.cuda.synchronize()

    t_op = _time_graph_replays(op_graph, replays)
    t_flush = _time_graph_replays(flush_graph, replays)

    mean = (statistics.fmean(t_op) - statistics.fmean(t_flush)) / inner
    var_op = statistics.variance(t_op) if len(t_op) > 1 else 0.0
    var_flush = statistics.variance(t_flush) if len(t_flush) > 1 else 0.0
    std = math.sqrt(var_op + var_flush) / inner

    del op_graph, flush_graph, cache
    return dict(
        mean=mean,
        std=std,
        median=(statistics.median(t_op) - statistics.median(t_flush)) / inner,
        min=(min(t_op) - statistics.fmean(t_flush)) / inner,
        max=(max(t_op) - statistics.fmean(t_flush)) / inner,
        n=len(t_op),
        inner=inner,
    )


# Checks and reporting


def _check_consistency(case, atol=2e-2):
    """Same BF16 data through both attention kernels must match closely."""
    o_paged = _fa3_paged_attn(case)
    o_varlen = _mimo_varlen_attn(case, use_ws=False)
    o_varlen = o_varlen[0] if isinstance(o_varlen, tuple) else o_varlen
    max_diff = (o_paged.float() - o_varlen.float()).abs().max().item()
    assert max_diff < atol, f"paged vs varlen mismatch: max_diff={max_diff}"
    return max_diff


def _check_fp8_roundtrip(case):
    """Attention on dequantized-FP8 KV vs original BF16 KV (info only)."""
    _mimo_get_kv(case)
    o_bf16 = _mimo_varlen_attn(case, use_ws=False)
    o_fp8 = _mimo_varlen_attn(case, use_ws=True)
    o_bf16 = o_bf16[0] if isinstance(o_bf16, tuple) else o_bf16
    o_fp8 = o_fp8[0] if isinstance(o_fp8, tuple) else o_fp8
    return (o_bf16.float() - o_fp8.float()).abs().max().item()


def _attn_tflops(batch_size, seq_len, q_len, num_q_heads, head_dim, v_head_dim, ms):
    # Bottom-right aligned causal: query i attends kv_len - q_len + i + 1 keys.
    attended = q_len * (seq_len - q_len) + q_len * (q_len + 1) // 2
    flops = 2 * batch_size * num_q_heads * (head_dim + v_head_dim) * attended
    return flops / (ms * 1e-3) / 1e12


def _fmt(stats, extra=""):
    s = (
        f"{stats['mean']:9.3f} ± {stats['std']:6.3f} ms  "
        f"(median={stats['median']:.3f}, min={stats['min']:.3f}, "
        f"max={stats['max']:.3f}, n={stats['n']}"
    )
    if "inner" in stats:
        s += f", inner={stats['inner']}"
    if extra:
        s += f", {extra}"
    return s + ")"


def _combine(*stats_list):
    """Sum of independent measurements: means add, variances add."""
    return dict(
        mean=sum(s["mean"] for s in stats_list),
        std=math.sqrt(sum(s["std"] ** 2 for s in stats_list)),
    )


# Main loop


def run(args):
    v_head_dim = args.v_head_dim or args.head_dim
    if IS_CI:
        configs = [(1, 1024)]
    else:
        configs = list(itertools.product(args.batch_sizes, args.seq_lens))

    print(
        f"device={torch.cuda.get_device_name()} "
        f"capability={torch.cuda.get_device_capability()} "
        f"torch={torch.__version__}"
    )
    print(f"mimo pool: {'OK' if HAS_MIMO else f'UNAVAILABLE ({MIMO_IMPORT_ERROR})'}")
    print(f"IS_CI={IS_CI}, {len(configs)} config(s): {configs}")

    ev_kwargs = dict(
        warmup=args.warmup,
        max_iters=args.iters,
        min_iters=args.min_iters,
        max_bench_time_s=args.max_bench_time,
    )
    gr_kwargs = dict(warmup=args.warmup, replays=args.graph_replays)
    summary = []

    for batch_size, seq_len in configs:
        extend_len = min(args.extend_len or seq_len, seq_len)
        print(
            f"\n=== bs={batch_size} seq_len={seq_len} q_len={extend_len} "
            f"(h_q={args.num_q_heads}, h_kv={args.num_kv_heads}, "
            f"d={args.head_dim}/{v_head_dim}, page={args.page_size}) ==="
        )
        print("  building inputs and pools ...")
        case = _build_case(
            batch_size=batch_size,
            seq_len=seq_len,
            extend_len=extend_len,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            v_head_dim=v_head_dim,
            page_size=args.page_size,
            dtype=torch.bfloat16,
        )

        max_diff = _check_consistency(case)
        print(f"[check] paged vs varlen (same BF16 data): max_diff={max_diff:.4f}")
        if case["mimo_ok"]:
            fp8_diff = _check_fp8_roundtrip(case)
            print(f"[check] FP8 quant->dequant attention output diff: {fp8_diff:.4f}")

        row = {"bs": batch_size, "seq_len": seq_len, "q_len": extend_len}

        # fa3 path
        st = _bench_graph_stats(lambda: _fa3_set_kv(case), **gr_kwargs)
        print(f"  fa3_set_kv      : {_fmt(st)}")
        row["fa3_set_kv"] = st

        st = _bench_graph_stats(lambda: _fa3_get_kv(case), **gr_kwargs)
        print(f"  fa3_get_kv      : {_fmt(st, 'pure view, expect ~0')}")
        row["fa3_get_kv"] = st

        st = _bench_event_stats(lambda: _fa3_paged_attn(case), **ev_kwargs)
        tflops = _attn_tflops(
            batch_size, seq_len, extend_len,
            args.num_q_heads, args.head_dim, v_head_dim, st["mean"],
        )  # fmt: skip
        print(f"  paged_attn      : {_fmt(st, f'{tflops:.0f} TFLOPS')}")
        row["paged_attn"] = st

        fa3_total = _combine(row["fa3_set_kv"], row["fa3_get_kv"], row["paged_attn"])
        print(
            f"  fa3_total       : {fa3_total['mean']:9.3f} ± "
            f"{fa3_total['std']:6.3f} ms  (set_kv + get_kv + paged_attn)"
        )
        row["fa3_total"] = fa3_total

        # mimo path
        if case["mimo_ok"]:
            st = _bench_graph_stats(lambda: _mimo_set_kv(case), **gr_kwargs)
            print(f"  mimo_set_kv     : {_fmt(st, 'FP8 quant_store')}")
            row["mimo_set_kv"] = st

            st = _bench_graph_stats(lambda: _mimo_get_kv(case), **gr_kwargs)
            print(f"  mimo_get_kv     : {_fmt(st, 'dequant_gather, per layer')}")
            row["mimo_get_kv"] = st

            st = _bench_event_stats(lambda: _mimo_varlen_attn(case), **ev_kwargs)
            tflops = _attn_tflops(
                batch_size, seq_len, extend_len,
                args.num_q_heads, args.head_dim, v_head_dim, st["mean"],
            )  # fmt: skip
            print(f"  varlen_attn     : {_fmt(st, f'{tflops:.0f} TFLOPS')}")
            row["varlen_attn"] = st

            mimo_total = _combine(
                row["mimo_set_kv"], row["mimo_get_kv"], row["varlen_attn"]
            )
            print(
                f"  mimo_total      : {mimo_total['mean']:9.3f} ± "
                f"{mimo_total['std']:6.3f} ms  (set_kv + get_kv + varlen_attn)"
            )
            row["mimo_total"] = mimo_total

            speedup = fa3_total["mean"] / mimo_total["mean"]
            print(f"  fa3_total / mimo_total = {speedup:.3f}x")
        else:
            reason = (
                f"mimo pool unavailable ({MIMO_IMPORT_ERROR})"
                if not HAS_MIMO
                else f"requires head_dim={MIMO_HEAD_DIM}, v_head_dim={MIMO_V_DIM}, "
                f"page_size={MIMO_PAGE_SIZE}"
            )
            print(f"  mimo path       : skipped ({reason})")

        summary.append(row)
        del case
        torch.cuda.empty_cache()

    # Final summary table (means, ± std).
    cols = [
        "fa3_set_kv",
        "fa3_get_kv",
        "paged_attn",
        "fa3_total",
        "mimo_set_kv",
        "mimo_get_kv",
        "varlen_attn",
        "mimo_total",
    ]
    print("\n=== Summary (mean ± std, ms) ===")
    header = f"{'bs':>4} {'seq_len':>8} {'q_len':>8}"
    for c in cols:
        header += f" {c:>22}"
    print(header)
    for row in summary:
        line = f"{row['bs']:>4} {row['seq_len']:>8} {row['q_len']:>8}"
        for c in cols:
            if c in row:
                line += f" {row[c]['mean']:>13.3f} ± {row[c]['std']:<6.3f}"
            else:
                line += f" {'-':>22}"
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FA3 vs MiMo forward_extend: attention + real "
        "KV pool set/get ops (GQA)"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1], help="Batch sizes"
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[1024, 4096, 8192, 16384, 32768, 65536, 131072],
        help="Per-request KV sequence lengths",
    )
    parser.add_argument(
        "--extend-len",
        type=int,
        default=None,
        help="Extend (query) length per request; default = seq_len (full "
        "prefill). Set a smaller value to simulate chunked prefill over a "
        "long prefix.",
    )
    parser.add_argument("--num-q-heads", type=int, default=32, help="Query heads (GQA)")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="KV heads (GQA)")
    parser.add_argument("--head-dim", type=int, default=128, help="Q/K head dim")
    parser.add_argument(
        "--v-head-dim", type=int, default=None, help="V head dim; default = --head-dim"
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1,
        help="KV cache page size for the paged path (sglang FA3 default: 1; "
        "the MiMo pool requires 64)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Untimed warmup iterations per op"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Max timed iterations per attention config (adaptively reduced "
        "for long kernels, see --max-bench-time)",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=10,
        help="Min timed iterations per attention config",
    )
    parser.add_argument(
        "--max-bench-time",
        type=float,
        default=10.0,
        help="Target total timed seconds per attention config",
    )
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=30,
        help="Timed CUDA-graph replays for pool set/get ops",
    )
    args = parser.parse_args()

    major, minor = get_device_capability()
    print(f"starting benchmark, device capability: {major}.{minor}")
    if major is None or (major, minor) < (9, 0):
        print(f"Skipping: FA3 extend benchmark requires SM90+, found {major}.{minor}")
    else:
        assert args.num_q_heads % args.num_kv_heads == 0, "GQA requires h_q % h_kv == 0"
        run(args)
        print("\nBenchmark finished!")
