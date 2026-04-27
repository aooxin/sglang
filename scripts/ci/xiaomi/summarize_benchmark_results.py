#!/usr/bin/env python3
"""Summarize bench_serving JSONL result files into a compact table."""
import argparse
import glob
import json

HEADERS = [
    ("Scenario", "_scenario", None),
    ("Input Len", "random_input_len", 0),
    ("Output Len", "random_output_len", 0),
    ("Rate", "request_rate", 2),
    ("Reqs", "completed", 0),
    ("Duration(s)", "duration", 2),
    ("Req/s", "request_throughput", 2),
    ("In tok/s", "input_throughput", 2),
    ("Out tok/s", "output_throughput", 2),
    ("Mean TTFT(ms)", "mean_ttft_ms", 2),
    ("P99 TTFT(ms)", "p99_ttft_ms", 2),
    ("Mean TPOT(ms)", "mean_tpot_ms", 2),
    ("P99 TPOT(ms)", "p99_tpot_ms", 2),
]


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


def fmt(v, precision=2):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def build_rows(results):
    rows = []
    for r in results:
        rows.append([fmt(r.get(key), prec) for _, key, prec in HEADERS])
    return rows


def format_ascii_table(rows):
    header_names = [h for h, _, _ in HEADERS]
    widths = [len(h) for h in header_names]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(header_names, widths)) + " |"

    lines = [sep, hdr, sep]
    for row in rows:
        lines.append(
            "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_markdown_table(rows):
    header_names = [h for h, _, _ in HEADERS]
    lines = ["| " + " | ".join(header_names) + " |"]
    lines.append("| " + " | ".join("---" for _ in header_names) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("patterns", nargs="*", default=["benchmark_results_*.json"])
    parser.add_argument("--output", help="Write markdown summary to file (for MR note)")
    parser.add_argument("--title", default="Benchmark Summary", help="Summary title")
    args = parser.parse_args()

    results = load_results(args.patterns)
    if not results:
        print("No benchmark results found.")
        return

    rows = build_rows(results)

    print()
    print("=" * 80)
    print(f"  {args.title}")
    print("=" * 80)
    print()
    print(format_ascii_table(rows))
    print()

    if args.output:
        md = f"### {args.title}\n\n{format_markdown_table(rows)}\n"
        with open(args.output, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
