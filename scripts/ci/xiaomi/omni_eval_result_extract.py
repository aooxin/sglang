#!/usr/bin/env python3
"""Summarize omni evaluation scores.

Reads task result files under <results_dir> (one subdirectory per task, each
containing dated subdirectories), extracts scores, compares against
baselines, and writes:
  - aligned plain-text scores to <scores_path> (also printed to stdout)
  - markdown table rows to <scores_md_path>

Tasks whose score regresses by more than 1 point are listed at the end with
their full result file dumps.

Usage:
    summarize_scores.py <results_dir> [scores_path] [scores_md_path]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, Optional

REGRESSION_THRESHOLD = -1


# ---------------------------------------------------------------------------
# Task definitions
#
# Each entry describes how to extract one or more scores from a task's latest
# date directory. ``filename`` is either an exact filename or a suffix glob
# (e.g. ``*_summary.json``). ``path`` is the dotted JSON path to the score.
# ``subfields``, when set, fans out one row per subfield (the path is then
# treated as the parent and each subfield is appended).
# ``lower_is_better`` flips the diff sign (used for WER).
# ---------------------------------------------------------------------------
TASKS = {
    # IMAGE tasks (each task lives in a *_results.json under "results.<task>")
    "countbench_boxed_mimo": {
        "filename": "*_results.json",
        "path": ["results", "countbench_boxed_mimo", "accuracy,flexible-extract"],
        "baseline": 93.9,
    },
    "osworld_g_rec_test_rel_mimo": {
        "filename": "*_results.json",
        "path": ["results", "osworld_g_rec_test_rel_mimo", "osworld_g_Center_ACC,none"],
        "baseline": 64.9,
    },
    "refcoco+_bbox_rec_val_rel_mimo": {
        "filename": "*_results.json",
        "path": ["results", "refcoco+_bbox_rec_val_rel_mimo", "refcoco_ACC@0.5,none"],
        "baseline": 83.4,
    },
    "screenspot_pro_en_rec_test_rel_mimo": {
        "filename": "*_results.json",
        "path": ["results", "screenspot_pro_en_rec_test_rel_mimo", "screenspot_Center_ACC,none"],
        "baseline": 60.2,
    },
    "mmmu_val_boxed_gpt": {
        "filename": "*_results.json",
        "path": ["results", "mmmu_val_boxed_gpt", "mmmu_judge_acc,none"],
        "baseline": 82,
    },
    "mmmu_pro_standard_boxed_gpt": {
        "filename": "*_results.json",
        "path": ["results", "mmmu_pro_standard_boxed_gpt", "mmmu_acc,none"],
        "baseline": 76.2,
    },
    "charxiv_boxed_mimo": {
        "filename": "*_results.json",
        "path": ["results", "charxiv_boxed_mimo", "reasoning,none"],
        "baseline": 79.9,
    },
    # AUDIO tasks
    "mmau_mmau": {
        "filename": "mmau_scores.json",
        "path": [],
        "subfields": ("overall", "music", "sound", "speech"),
        "baseline": {"overall": 78.8, "music": 74.9, "sound": 79.3, "speech": 82.3},
    },
    "asr_aishell1_test": {"filename": "asr_results.json", "path": ["summary", "avg_wer"], "baseline": 8.41, "lower_is_better": True},
    "asr_librispeech_test_clean": {"filename": "asr_results.json", "path": ["summary", "avg_wer"], "baseline": 1.67, "lower_is_better": True},
    "asr_librispeech_test_other": {"filename": "asr_results.json", "path": ["summary", "avg_wer"], "baseline": 4.09, "lower_is_better": True},
    "asr_wenetspeech_test_net": {"filename": "asr_results.json", "path": ["summary", "avg_wer"], "baseline": 15.02, "lower_is_better": True},
    "asr_wenetspeech_test_meeting": {"filename": "asr_results.json", "path": ["summary", "avg_wer"], "baseline": 14.94, "lower_is_better": True},
    # VIDEO tasks
    "dailyomni": {"filename": "*_summary.json", "path": ["summary", "accuracy"], "baseline": 83.5},
    "videomme": {"filename": "*_summary.json", "path": ["summary", "accuracy"], "baseline": 87.2},
    "videommev2": {"filename": "*_summary.json", "path": ["summary", "accuracy"], "baseline": 40.2},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dig(obj, keys):
    """Walk a JSON path given as a list of keys. Empty list returns the object itself."""
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
        if obj is None:
            return None
    return obj


def latest_date_dir(task_path: str) -> Optional[str]:
    dates = sorted(d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d)))
    return os.path.join(task_path, dates[-1]) if dates else None


def resolve_file(date_dir: str, filename: str) -> Optional[str]:
    """Resolve filename: exact match or '*<suffix>' glob -> first sorted match."""
    if filename.startswith("*"):
        suffix = filename[1:]
        for f in sorted(os.listdir(date_dir)):
            if f.endswith(suffix):
                return os.path.join(date_dir, f)
        return None
    full = os.path.join(date_dir, filename)
    return full if os.path.isfile(full) else None


def fmt(score: Optional[float], baseline, lower_is_better: bool):
    score_str = f"{score:.2f}" if score is not None else "N/A"
    baseline_str = str(baseline) if baseline is not None else "N/A"
    diff_str = "N/A"
    diff = None
    if score is not None and baseline is not None:
        diff = baseline - score if lower_is_better else score - baseline
        diff_str = f"{diff:+.2f}"
    return score_str, baseline_str, diff_str, diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [scores_path] [scores_md_path]", file=sys.stderr)
        return 2

    results_dir = sys.argv[1]
    scores_path = sys.argv[2] if len(sys.argv) > 2 else None
    scores_md_path = sys.argv[3] if len(sys.argv) > 3 else None

    txt_fp = open(scores_path, "w") if scores_path else None
    md_fp = open(scores_md_path, "w") if scores_md_path else None

    def write_txt(line: str) -> None:
        print(line)
        if txt_fp:
            txt_fp.write(line + "\n")

    def write_md(line: str) -> None:
        if md_fp:
            md_fp.write(line + "\n")

    def emit_row(task: str, score: str, baseline: str, diff: str) -> None:
        write_txt(f"  {task:<40} {baseline:>10} {score:>10} {diff:>10}")
        write_md(f"| {task} | {baseline} | {score} | {diff} |")

    # Header
    write_txt(f"  {'Task':<40} {'Baseline':>10} {'Score':>10} {'Diff':>10}")
    write_txt(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    write_md("| Task | Baseline | Score | Diff |")
    write_md("| --- | ---: | ---: | ---: |")

    regressed: list[tuple[str, str]] = []  # (display_name, date_dir)

    for task_dir in sorted(os.listdir(results_dir)):
        spec = TASKS.get(task_dir)
        task_path = os.path.join(results_dir, task_dir)
        if spec is None or not os.path.isdir(task_path):
            continue

        date_dir = latest_date_dir(task_path)
        if date_dir is None:
            continue

        filepath = resolve_file(date_dir, spec["filename"])
        if filepath is None:
            print(f"  [WARN] no {spec['filename']} under {date_dir}", file=sys.stderr)
            continue

        try:
            data = json.load(open(filepath))
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(filepath)}: {e}", file=sys.stderr)
            continue

        lower = bool(spec.get("lower_is_better", False))
        subfields: Iterable[Optional[str]] = spec.get("subfields") or (None,)
        for sub in subfields:
            path = list(spec["path"]) + ([sub] if sub else [])
            raw = dig(data, path)
            score_val = raw * 100 if isinstance(raw, (int, float)) else None
            baseline_val = spec["baseline"][sub] if sub else spec["baseline"]
            score_str, baseline_str, diff_str, diff = fmt(score_val, baseline_val, lower)
            display = f"{task_dir}({sub})" if sub else task_dir
            if diff is not None and diff < REGRESSION_THRESHOLD:
                regressed.append((display, date_dir))
            emit_row(display, score_str, baseline_str, diff_str)

    # Dump details for regressed tasks
    if regressed:
        print()
        print("=" * 70)
        print("REGRESSED TASKS (score - baseline < -1):")
        print("=" * 70)
        for name, date_dir in regressed:
            print(f"\n--- {name} ---")
            for f in sorted(os.listdir(date_dir)):
                fpath = os.path.join(date_dir, f)
                try:
                    if f.endswith(".json"):
                        print(f"\n[FULL] {f}:")
                        print(open(fpath).read())
                    elif f.endswith(".jsonl"):
                        print(f"\n[FIRST 10 LINES] {f}:")
                        with open(fpath) as fp:
                            for i, line in enumerate(fp):
                                if i >= 10:
                                    print("  ... (truncated)")
                                    break
                                print(line.rstrip())
                except Exception as e:
                    print(f"  [ERROR reading {f}: {e}]")

    if txt_fp:
        txt_fp.close()
    if md_fp:
        md_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
