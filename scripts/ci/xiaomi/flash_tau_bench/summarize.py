#!/usr/bin/env python3
"""Summarize tau-bench results into per-run + average reward, plus MR markdown.

The image's /root/tau-bench/run.sh runs `tau2 run` three times in a row over
the same domain (telecom). Each run prints a final line like:

    master-0 >> │ 🏆 Average Reward: 0.8509

We grep those lines from the captured run log, extract the floats, average
them, and write a markdown table to $EVAL_SUMMARY_MD for the MR note.

Usage:
    summarize.py <run_log_path>

Reads env:
    EVAL_LABEL                  display label (e.g. "FLASH-TAU")
    CREATED                     image tag id
    OPENAI_COMPATIBLE_API_URL   deployment endpoint
    EVAL_SUMMARY_MD             output markdown path

Exit codes:
    0   parsed at least one Average Reward
    2   no Average Reward found / log missing
"""
import os
import re
import sys
from pathlib import Path

# MiMo-V2-Flash tau-bench (telecom) baseline (team reference, intentionally
# loose — refine after the first stable green run).
BASELINE_AVG_REWARD = 0.90

# Match lines like:
#   master-0 >> │ 🏆 Average Reward: 0.8509
#   Average Reward: 0.85
# Tolerant of leading rich box-drawing / emoji / prefix decoration.
REWARD_RE = re.compile(r"Average Reward:\s*([0-9]*\.?[0-9]+)")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run_log_path>", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"ERROR: run log does not exist: {log_path}", file=sys.stderr)
        return 2

    rewards: list[float] = []
    # Read line-by-line; tau-bench logs can be 100+ MB with rich progress
    # output, but each Average Reward line is a single short line.
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = REWARD_RE.search(line)
            if m:
                try:
                    rewards.append(float(m.group(1)))
                except ValueError:
                    pass

    avg = sum(rewards) / len(rewards) if rewards else 0.0

    print("=" * 72)
    print("tau-bench evaluation summary")
    print(f"  log path:      {log_path}")
    print(f"  runs found:    {len(rewards)}")
    for i, r in enumerate(rewards, start=1):
        print(f"  run {i}:         {r:.4f}")
    print(f"  average:       {avg:.4f}")
    print(f"  baseline:      {BASELINE_AVG_REWARD:.4f}")
    print("=" * 72)

    md_path = os.environ.get("EVAL_SUMMARY_MD", "").strip()
    if md_path:
        label = os.environ.get("EVAL_LABEL", "FLASH-TAU")
        created = os.environ.get("CREATED", "")
        endpoint = os.environ.get("OPENAI_COMPATIBLE_API_URL", "")

        # Build per-run table dynamically — usually 3 rows, but keep the table
        # honest if a run died and we only have 1 or 2.
        run_rows = [
            f"| Run {i} | {r:.4f} |"
            for i, r in enumerate(rewards, start=1)
        ]
        run_table = (
            ["| Metric | Score |", "| --- | ---: |"]
            + run_rows
            + [
                f"| **Average** | **{avg:.4f}** |",
                f"| Baseline | {BASELINE_AVG_REWARD:.4f} |",
            ]
        )

        lines = [
            f"### {label} tau-bench Evaluation (telecom × {len(rewards)} runs)",
            "",
            f"- **Image:** `{created}`",
            f"- **Endpoint:** `{endpoint}`",
            "",
            *run_table,
            "",
        ]

        if not rewards:
            lines.append("❌ **No `Average Reward:` lines found in run log**")
        elif len(rewards) < 3:
            lines.append(
                f"⚠️ **Only {len(rewards)}/3 runs reported a reward — "
                "some runs may have failed mid-eval. Check artifacts.**"
            )
        elif avg >= BASELINE_AVG_REWARD:
            lines.append("✅ **Run completed (avg ≥ baseline)**")
        else:
            lines.append(
                f"⚠️ **Run completed but avg ({avg:.4f}) < baseline "
                f"({BASELINE_AVG_REWARD:.4f})**"
            )
        lines.append("")
        Path(md_path).write_text("\n".join(lines))

    return 0 if rewards else 2


if __name__ == "__main__":
    sys.exit(main())
