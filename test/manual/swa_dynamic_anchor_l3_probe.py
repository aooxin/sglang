"""Manual probe: isolate the dynamic SWA reuse anchor's contribution.

Unlike an automated CI test, this script does NOT launch a server. You start
the server yourself with your own known-good launch config, then point this
probe at it. That keeps the fragile model/parallelism launch out of the probe.

--- What it measures ------------------------------------------------------

The dynamic anchor protects the SWA window ending at an observed full-KV
branch point, on top of the *static* reserved points. With the production
default reserved points (512,1024,2048,4096,8192,12288,16384,24576,32768) the
anchor at a branch like 16384 overlaps a static point and its net contribution
is ~1 page -- invisible. To isolate the anchor, launch the server with a SPARSE
reserved-point set that leaves the branch point in a clean zone, e.g.

    SGLANG_SIMPLE_SWA_RESERVED_POINTS=1024,98304   # nothing between 1k and 98k

Then a dynamic anchor planted at 16384 protects a full sliding window that no
static point would, and a later request reusing that prefix stays SWA-safe at
16384 (ON) instead of collapsing to the nearest static point 1024 (OFF).

--- Recipe (A / B / C, all share prefix P, divergent tails) ---------------

  P = shared prefix (len = --prefix-len, must exceed the sliding window)
  A = P + tail_A   populate; A's live window ends past P, so P's SWA is evicted
  B = P + tail_B   full-KV hit reaches len(P) but SWA-safe is trimmed far below
                   -> B registers a dynamic anchor at floor_align(len(P)); B's
                   live window is *past* P, so [P-window, P) is mid-request and
                   survives B's eviction only because the anchor protects it
  C = P + tail_C   reuses SWA-safe up to len(P) via the anchor (ON), or only to
                   the nearest static point (OFF)

The tails MUST be divergent and push each request's live window past P; if B
ended exactly at P then [P-window, P) would be B's natural tail window and would
survive with or without the anchor -- no isolation.

--- Server you must start first -------------------------------------------

Launch your model with your usual args PLUS:

    --enable-hierarchical-cache
    --hicache-write-policy write_through
    --page-size 128                       # must match --page-size below
    --enable-cache-report
    (--hicache-storage-backend file|gcache for the L3 / --flush variant)

and env:

    SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS=1                # 0 for the control run
    SGLANG_SIMPLE_SWA_RESERVED_POINTS=1024,98304       # sparse, isolates anchor

--- Then run the probe ----------------------------------------------------

    # PD-disaggregated: /generate -> gateway, /flush_cache + /metrics -> prefill node
    python3 test/manual/swa_dynamic_anchor_l3_probe.py \
        --base-url  http://<gateway> \
        --metrics-url http://<prefill-ip>:<port> \
        --flush-url   http://<prefill-ip>:<port> \
        --page-size 128 --prefix-len 16384 --tail-len 8192

Run once with SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS=1 and once with =0; compare
C's cached_tokens and the swa_dynamic_anchor_protected_tokens_total delta.
Add --flush to clear L1/L2 (on the prefill node) between B and C and force C's
prefix to come from L3 (watch cache_source="storage_..." and storage_swa_hit).
"""

import argparse
import json
import random
import sys
import time
from typing import Dict, List, Optional

import requests

# P-node Prometheus counters we diff around the flow. Cumulative; we snapshot
# before A and after C and report the delta. See metrics_collector.py.
METRICS = {
    "anchor_protected": "sglang:swa_dynamic_anchor_protected_tokens_total",
    "storage_swa_hit": "sglang:storage_swa_hit_tokens_total",
    "storage_full_hit": "sglang:storage_full_hit_tokens_total",
    "raw_cached": "sglang:raw_cached_tokens_total",
}


def rand_ids(n: int, lo: int = 1000, hi: int = 100000) -> List[int]:
    return [random.randint(lo, hi) for _ in range(n)]


def generate(base_url: str, input_ids: List[int], gen_tokens: int) -> Dict:
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": gen_tokens,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def cached_tokens(resp: Dict) -> int:
    return int(resp.get("meta_info", {}).get("cached_tokens", 0))


def cached_details(resp: Dict) -> Dict:
    d = resp.get("meta_info", {}).get("cached_tokens_details")
    return d if isinstance(d, dict) else {}


def _fmt(resp: Dict) -> str:
    d = cached_details(resp)
    if d:
        return (
            f"cached_tokens={cached_tokens(resp)} "
            f"details(device={d.get('device')}, host={d.get('host')}, "
            f"storage={d.get('storage')}, backend={d.get('storage_backend')})"
        )
    return f"cached_tokens={cached_tokens(resp)} (no details)"


def _meta(resp: Dict) -> str:
    """Full server meta_info exactly as returned (cached_tokens, its _details,
    prompt/completion token counts, finish_reason, ...) so the raw response is
    visible for cross-checking against the P-node metrics."""
    return json.dumps(resp.get("meta_info", {}), ensure_ascii=False, sort_keys=True)


def scrape_metrics(metrics_url: str) -> str:
    resp = requests.get(f"{metrics_url}/metrics", timeout=30)
    resp.raise_for_status()
    return resp.text


def metric_max(text: str, name: str, label_substr: Optional[str] = None) -> float:
    """Max value across all label sets of a Prometheus counter.

    Ranks that handled the request report the same cumulative value (tensor
    parallel processes the same sequence), so max() gives the true per-request
    cumulative without double counting across tp ranks; an aggregated single
    line also works (max of one). Returns 0.0 if the metric is absent.
    """
    best = 0.0
    found = False
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        # The char after the name must be '{' or whitespace, so a metric name
        # that is a prefix of another (e.g. *_total vs *_total_sum) never leaks.
        rest = line[len(name):]
        if rest and rest[0] not in "{ \t":
            continue
        if label_substr is not None and label_substr not in line:
            continue
        try:
            best = max(best, float(line.rsplit(" ", 1)[1]))
            found = True
        except (IndexError, ValueError):
            continue
    return best if found else 0.0


def snapshot(metrics_url: str) -> Dict[str, float]:
    text = scrape_metrics(metrics_url)
    snap = {k: metric_max(text, v) for k, v in METRICS.items()}
    snap["cached_device"] = metric_max(
        text, "sglang:cached_tokens_total", 'cache_source="device"'
    )
    snap["cached_storage"] = metric_max(
        text, "sglang:cached_tokens_total", "cache_source=\"storage"
    )
    return snap


def diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:30000",
                   help="Where /generate goes. In PD mode this is the gateway.")
    p.add_argument("--metrics-url", default=None,
                   help="Where GET /metrics is read. In PD mode set to the "
                        "prefill node ip:port. Defaults to --base-url.")
    p.add_argument("--flush-url", default=None,
                   help="Where POST /flush_cache goes (only with --flush). In PD "
                        "mode set to the prefill node ip:port. Defaults to "
                        "--base-url.")
    p.add_argument("--page-size", type=int, default=128,
                   help="Must match the server's --page-size.")
    p.add_argument("--prefix-len", type=int, default=16384,
                   help="Shared prefix P; must exceed the sliding window and, to "
                        "isolate the anchor, sit in a zone with no static point.")
    p.add_argument("--tail-len", type=int, default=8192,
                   help="Divergent per-request tail; must be >0 so each request's "
                        "live window ends past P.")
    p.add_argument("--gen-tokens", type=int, default=8)
    p.add_argument("--flush", action="store_true",
                   help="Flush L1/L2 on the prefill node between B and C so C's "
                        "prefix can only come from L3 (tests storage recovery).")
    p.add_argument("--settle-secs", type=float, default=8.0,
                   help="Wait before flush for write_through to drain to L3.")
    args = p.parse_args()

    metrics_url = args.metrics_url or args.base_url
    flush_url = args.flush_url or args.base_url

    prefix_len = args.prefix_len
    tail_len = args.tail_len
    if tail_len <= 0:
        print("ERROR: --tail-len must be > 0 (each request must extend past P).")
        return 2
    if prefix_len % args.page_size != 0:
        print(f"WARNING: --prefix-len {prefix_len} not a multiple of page-size "
              f"{args.page_size}; the anchor is floor-aligned to a page.")

    prefix = rand_ids(prefix_len)
    tail_a, tail_b, tail_c = (rand_ids(tail_len) for _ in range(3))

    print(f"base_url={args.base_url}\nmetrics_url={metrics_url} flush_url={flush_url}")
    print(f"page_size={args.page_size} prefix_len={prefix_len} tail_len={tail_len} "
          f"flush={args.flush}")
    # No random.seed(): prefix/tails differ every run. P is shared by A/B/C;
    # only the tails diverge. Print the first 10 ids so the shared-prefix /
    # divergent-tail structure is visible and cross-checkable.
    print(f"prefix[:10]={prefix[:10]}")
    print(f"tail_a[:10]={tail_a[:10]}")
    print(f"tail_b[:10]={tail_b[:10]}")
    print(f"tail_c[:10]={tail_c[:10]}")

    base = snapshot(metrics_url)
    print(f"\n[metrics baseline] {base}")

    print("\n[A] P + tail_A (populate; A's window ends past P) ...")
    ra = generate(args.base_url, prefix + tail_a, args.gen_tokens)
    print(f"    A {_fmt(ra)}")
    print(f"      meta_info: {_meta(ra)}")

    print("[B] P + tail_B (registers dynamic anchor at floor_align(P)) ...")
    rb = generate(args.base_url, prefix + tail_b, args.gen_tokens)
    print(f"    B {_fmt(rb)}")
    print(f"      meta_info: {_meta(rb)}")

    after_b = snapshot(metrics_url)
    print(f"    [metrics after B] anchor_protected +"
          f"{after_b['anchor_protected'] - base['anchor_protected']:.0f}")

    if args.flush:
        print(f"[flush] wait {args.settle_secs}s for write_through -> L3, then "
              f"flush {flush_url} (clears L1+L2, keeps L3) ...")
        time.sleep(args.settle_secs)
        r = requests.post(f"{flush_url}/flush_cache", params={"timeout": 60}, timeout=120)
        r.raise_for_status()
        time.sleep(1.0)

    print("[C] P + tail_C (reuse SWA-safe at P via anchor) ...")
    rc = generate(args.base_url, prefix + tail_c, args.gen_tokens)
    print(f"    C {_fmt(rc)}")
    print(f"      meta_info: {_meta(rc)}")

    after_c = snapshot(metrics_url)
    d = diff(base, after_c)

    print("\n==================== RESULT ====================")
    print(f"C cached_tokens = {cached_tokens(rc)}  (prefix_len={prefix_len})")
    print(f"metric deltas over the whole flow:")
    for k in ("anchor_protected", "storage_swa_hit", "storage_full_hit",
              "raw_cached", "cached_device", "cached_storage"):
        print(f"    {k:18s} +{d.get(k, 0.0):.0f}")
    print("------------------------------------------------")
    print("Interpretation:")
    print(f"  * anchor_protected delta > 0 => the dynamic anchor withheld SWA")
    print(f"    beyond the static points (0 in the ANCHORS=0 control).")
    print(f"  * C cached_tokens ~ prefix_len ({prefix_len}) with anchors ON vs")
    print(f"    ~ nearest static point with anchors OFF isolates the anchor's")
    print(f"    reuse gain. With --flush, a nonzero cached_storage/storage_swa_hit")
    print(f"    delta on C proves the recovery came from L3.")
    print("================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
