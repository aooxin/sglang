#!/usr/bin/env python3
"""Summarize mini-swe-agent SWE-bench results into pass@1 + MR markdown.

Walks <results_dir>/<instance>/*.traj.json and counts entries whose
info.exit_status contains "pass" (matching exp_stats.py's rule). Writes a
markdown table to $EVAL_SUMMARY_MD for the MR note.

Usage:
    summarize.py <results_dir>

Reads env:
    EVAL_LABEL                  display label (e.g. "FLASH-SWE")
    CREATED                     image tag id
    OPENAI_COMPATIBLE_API_URL   deployment endpoint
    EVAL_SUMMARY_MD             output markdown path

Exit codes:
    0   results_dir had instances and was parsed
    2   no instances found / results_dir missing
"""
import json
import os
import sys
from pathlib import Path

# MiMo-V2-Flash SWE-bench baseline (team reference).
BASELINE_PASS_AT_1 = 72.0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        return 2
    results_dir = Path(sys.argv[1])

    if not results_dir.is_dir():
        print(f"ERROR: results dir does not exist: {results_dir}", file=sys.stderr)
        return 2

    # Same scoring rule as scripts/exp_stats.py: info.exit_status containing
    # "pass" (case-insensitive) counts as a pass. Each instance has exactly
    # one *.traj.json under its subdir.
    passed = 0
    total = 0
    parse_errors: list[tuple[str, str]] = []
    for traj_path in sorted(results_dir.glob("*/*.traj.json")):
        try:
            data = json.loads(traj_path.read_text())
            status = (data.get("info", {}) or {}).get("exit_status") or ""
            if isinstance(status, str) and "pass" in status.lower():
                passed += 1
            total += 1
        except Exception as exc:
            parse_errors.append((str(traj_path), str(exc)))

    pass_at_1 = (100.0 * passed / total) if total else 0.0

    print("=" * 72)
    print("SWE-bench evaluation summary")
    print(f"  results dir:    {results_dir}")
    print(f"  instances:      {total}")
    print(f"  passed:         {passed}")
    print(f"  pass@1:         {pass_at_1:.2f}%")
    if parse_errors:
        print(f"  parse errors:   {len(parse_errors)}")
        for path, err in parse_errors[:5]:
            print(f"    - {path}: {err}")
    print("=" * 72)

    md_path = os.environ.get("EVAL_SUMMARY_MD", "").strip()
    if md_path:
        label = os.environ.get("EVAL_LABEL", "FLASH-SWE")
        created = os.environ.get("CREATED", "")
        endpoint = os.environ.get("OPENAI_COMPATIBLE_API_URL", "")
        bl_str = f"{BASELINE_PASS_AT_1:.2f}" if BASELINE_PASS_AT_1 is not None else "--"
        lines = [
            f"### {label} SWE-bench Evaluation",
            "",
            f"- **Image:** `{created}`",
            f"- **Endpoint:** `{endpoint}`",
            f"- **Instances:** {passed}/{total}",
            "",
            "| Metric | Score | Baseline |",
            "| --- | ---: | ---: |",
            f"| pass@1 | {pass_at_1:.2f} | {bl_str} |",
            "",
        ]
        if parse_errors:
            lines.append(f"⚠️ **Parse errors:** {len(parse_errors)} traj file(s) unreadable")
        elif total == 0:
            lines.append("❌ **No instances found**")
        else:
            lines.append("✅ **Run completed**")
        lines.append("")
        Path(md_path).write_text("\n".join(lines))

    return 0 if total > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
