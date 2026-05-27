#!/usr/bin/env python3
"""Summarize bench_serving JSONL result files into a compact table."""
import argparse
import glob
import json

HEADERS = [
    # identifiers — direction=None marks them as non-metric ID columns;
    # they always render first.
    ("Scenario", "_scenario", None, None),
    ("Input Len", "random_input_len", 0, None),
    ("Output Len", "random_output_len", 0, None),
    ("Rate", "request_rate", 2, None),
    # metrics — order here is the *fallback* render order when --compare
    # is not given; with --compare, listed keys are promoted to the front
    # in the order they were given (see reorder_headers).
    ("Accept Length=3.0", "accept_length", 2, "up"),
    ("Duration(s)", "duration", 2, "down"),
    ("Req/s", "request_throughput", 2, "up"),
    # Wider header labels for the high-volume metrics so that the
    # color-emoji prefix on a compared cell still fits on one line.
    ("Input tok/s", "input_throughput", 2, "up"),
    ("Output tok/s", "output_throughput", 2, "up"),
    ("Mean TTFT(ms)", "mean_ttft_ms", 2, "down"),
    ("Mean TPOT(ms)", "mean_tpot_ms", 2, "down"),
    ("Reqs", "completed", 0, "up"),
    ("P99 TTFT(ms)", "p99_ttft_ms", 2, "down"),
    ("P99 TPOT(ms)", "p99_tpot_ms", 2, "down"),
]

LEGEND = (
    "Legend: \U0001F7E2 ≥+10% (excellent) · "
    "\U0001F7E9 ≥+3% (better) · "
    "⬜ ±<3% (same) · "
    "\U0001F7E7 ≤-3% (worse) · "
    "\U0001F7E5 ≤-10% (much worse)"
)


def classify_delta(result, baseline, direction):
    if direction is None or result is None or baseline is None:
        return None
    try:
        result = float(result)
        baseline = float(baseline)
    except (TypeError, ValueError):
        return None
    if baseline == 0:
        return None
    change = (result - baseline) / baseline
    if direction == "down":
        change = -change
    pct = change * 100
    if pct >= 10:
        return "\U0001F7E2"
    if pct >= 3:
        return "\U0001F7E9"
    if pct > -3:
        return "⬜"
    if pct > -10:
        return "\U0001F7E7"
    return "\U0001F7E5"

def load_results(patterns):
    results = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        r["_scenario"] = (
                            path.replace("benchmark_results_", "")
                            .replace(".json", "")
                        )
                        results.append(r)
    return results

def load_baseline(path):
    if not path:
        return {}
    baselines = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = r.get("_scenario")
                if key:
                    baselines[key] = r
    except FileNotFoundError:
        return {}
    return baselines

def fmt(v, precision=2):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def drop_empty_metric_cols(headers, results, compare_keys):
    """Hide metric columns that no result row populates.

    ID columns (direction=None) and any column listed in --compare are always
    kept, so explicit user intent wins over auto-pruning.
    """
    keep_keys = set(compare_keys or [])
    out = []
    for h in headers:
        _, key, _, direction = h
        if direction is None or key in keep_keys:
            out.append(h)
            continue
        if any(r.get(key) is not None for r in results):
            out.append(h)
    return out


def reorder_headers(headers, compare_keys):
    """Promote --compare keys (in given order) right after the ID columns.

    ID columns (direction=None) keep their order at the front; --compare
    metrics follow in the order they were listed; remaining metrics trail
    in their original HEADERS order. Unknown compare keys are ignored.
    """
    if not compare_keys:
        return list(headers)
    id_cols = [h for h in headers if h[3] is None]
    metric_cols = [h for h in headers if h[3] is not None]
    by_key = {h[1]: h for h in metric_cols}
    promoted = [by_key[k] for k in compare_keys if k in by_key]
    moved = {k for k in compare_keys if k in by_key}
    trailing = [h for h in metric_cols if h[1] not in moved]
    return id_cols + promoted + trailing


def build_rows(results, headers, baselines=None, compare_keys=None):
    baselines = baselines or {}
    rows = []
    for r in results:
        scenario = r.get("_scenario")
        base = baselines.get(scenario)
        row = []
        for _, key, prec, direction in headers:
            value = r.get(key)
            cell = fmt(value, prec)
            if (
                base is not None
                and direction is not None
                and (compare_keys is None or key in compare_keys)
            ):
                marker = classify_delta(value, base.get(key), direction)
                if marker is not None:
                    # Use non-breaking space so renderers don't wrap between
                    # the emoji marker and the value (and so column auto-sizes
                    # to fit both on one line).
                    cell = f"{marker} {cell}"
            row.append(cell)
        if base is not None:
            base_row = dict(base)
            base_row["_scenario"] = f"baseline:{scenario}"
            rows.append(
                ([fmt(base_row.get(key), prec) for _, key, prec, _ in headers], True)
            )
        rows.append((row, False))
    return rows


def format_ascii_table(rows, headers):
    header_names = [h for h, _, _, _ in headers]
    widths = [len(h) for h in header_names]
    for row, _ in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(header_names, widths)) + " |"

    lines = [sep, hdr, sep]
    for row, _ in rows:
        lines.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_markdown_table(rows, headers):
    header_names = [h for h, _, _, _ in headers]
    lines = ["| " + " | ".join(header_names) + " |"]
    lines.append("| " + " | ".join("---" for _ in header_names) + " |")
    for row, is_baseline in rows:
        if is_baseline:
            cells = [f"*{c}*" if c else c for c in row]
        else:
            cells = row
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("patterns", nargs="*", default=["benchmark_results_*.json"])
    parser.add_argument("--output", help="Write markdown summary to file (for MR note)")
    parser.add_argument("--title", default="Benchmark Summary", help="Summary title")
    parser.add_argument("--baseline", help="JSONL baseline file; matching _scenario rows are appended under each result")
    parser.add_argument("--compare", help="Comma-separated header keys to color-rate against baseline AND promote to the front of the table (in the given order)")
    args = parser.parse_args()

    compare_keys = None
    if args.compare:
        compare_keys = [k.strip() for k in args.compare.split(",") if k.strip()]

    results = load_results(args.patterns)
    if not results:
        print("No benchmark results found.")
        return

    baselines = load_baseline(args.baseline)
    headers = reorder_headers(HEADERS, compare_keys)
    headers = drop_empty_metric_cols(headers, results, compare_keys)
    rows = build_rows(results, headers, baselines, compare_keys)

    print()
    print("=" * 80)
    print(f"  {args.title}")
    print("=" * 80)
    print()
    print(format_ascii_table(rows, headers))
    print()

    if args.output:
        md = f"### {args.title}\n\n{format_markdown_table(rows, headers)}\n"
        if baselines:
            md += f"\n{LEGEND}\n"
        with open(args.output, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
