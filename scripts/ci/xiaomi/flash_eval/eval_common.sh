#!/usr/bin/env bash
# Common setup for MiMo-V2-Flash mini_eval accuracy evaluation.
# Source this file, do not execute directly.
#
# The caller's positional parameters are forwarded to deploy_pd.sh, so the
# entry script should be invoked like:
#   run_eval.sh --deploy-yaml <path>
#
# Required env vars (set by the GitLab job + release-docker artifact):
#   SGLANG_IMAGE_URL           - sglang image to deploy on CloudML
#   CLOUDML_WORKSPACE_ID       - CloudML workspace id
#   CLOUDML_SERVICE_ID         - CloudML service id
#   CLOUDML_ACCESS_KEY         - CloudML AK (masked CI var)
#   CLOUDML_SECRET_KEY         - CloudML SK (masked CI var)
#   CLOUDML_MAAS_HOME          - path to cloudml-service-deploy-skill in image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PD="${SCRIPT_DIR}/../deploy_pd.sh"

# Forward "$@" (from the entry script) so --deploy-yaml flows through.
# deploy_pd.sh registers an EXIT trap that stops the CloudML deployment when
# this shell exits, so cleanup happens automatically on both success and failure.
# shellcheck source=../deploy_pd.sh disable=SC1091
source "${DEPLOY_PD}" "$@"

: "${DEPLOYMENT_ENDPOINT:?ERROR: DEPLOYMENT_ENDPOINT is empty after deploy_pd.sh}"
export OPENAI_COMPATIBLE_API_URL="${DEPLOYMENT_ENDPOINT}"
# mini_eval uses the openai python client whose base_url should point at the
# OpenAI-compatible /v1 path. CloudML DEPLOYMENT_ENDPOINT is just scheme://host,
# so append /v1 (trim any trailing slash first to avoid double slashes).
export LOCAL_API_BASE="${OPENAI_COMPATIBLE_API_URL%/}/v1"

# File-safe run id derived from the image tag (used as MODEL_NAME + output subdir).
_IMAGE_TAG="${SGLANG_IMAGE_URL:+${SGLANG_IMAGE_URL##*/}}"
_IMAGE_TAG="${_IMAGE_TAG:-unknown}"
CREATED="${_IMAGE_TAG//:/_}"
CREATED="${CREATED//\//_}"
export CREATED

export MINI_EVAL_DIR="${MINI_EVAL_DIR:-/app/mini_eval}"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${MINI_EVAL_DIR}/output}"
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-128}"
export EVAL_LABEL="${EVAL_LABEL:-FLASH}"

FAILED_TASKS=()

# Pull in shared smoke-test helper. eval_smoke_test() handles the actual POST.
# shellcheck source=../eval_smoke.sh disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/eval_smoke.sh"

# Thin wrapper: keeps the existing call site (`wait_for_service_ready`) working
# while delegating to the shared helper.
wait_for_service_ready() {
    eval_smoke_test "${OPENAI_COMPATIBLE_API_URL%/}/v1/chat/completions" "${CREATED}"
}

run_task() {
    local task="$1"
    local script="${MINI_EVAL_DIR}/eval_${task}.py"
    echo ""
    echo "================================================================"
    echo "=== [${EVAL_LABEL}] ${task}"
    echo "================================================================"
    if [[ ! -f "${script}" ]]; then
        echo "=== [${EVAL_LABEL}] ${task} - FAILED: ${script} not found ==="
        FAILED_TASKS+=("${task}")
        return 0
    fi
    if (cd "${MINI_EVAL_DIR}" && MODEL_NAME="${CREATED}" python3 "${script}"); then
        echo "=== [${EVAL_LABEL}] ${task} - PASSED ==="
    else
        echo "=== [${EVAL_LABEL}] ${task} - FAILED (continuing) ==="
        FAILED_TASKS+=("${task}")
    fi
}

print_summary() {
    local label="${1:-${EVAL_LABEL}}"
    local results_dir="${OUTPUT_BASE_DIR}/${CREATED}"
    # Markdown summary for MR note, written relative to CI_PROJECT_DIR so
    # artifacts: can pick it up and post_mr_note.sh can read it.
    local md_name="eval_summary_$(echo "${label}" | tr '[:upper:]' '[:lower:]').md"
    export EVAL_SUMMARY_MD="${CI_PROJECT_DIR:-.}/${md_name}"

    echo ""
    echo "######################################################################"
    echo "# ${label} ACCURACY EVALUATION SUMMARY (mini_eval)"
    echo "######################################################################"
    echo "Target:   ${OPENAI_COMPATIBLE_API_URL}"
    echo "Run ID:   ${CREATED}"
    echo "Output:   ${results_dir}"
    echo "Tasks:    ${EVAL_TASKS:-<all>}"
    echo ""

    if [[ ! -d "${results_dir}" ]]; then
        echo "WARNING: results dir does not exist: ${results_dir}"
        if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
            echo "FAILED TASKS (${#FAILED_TASKS[@]}): ${FAILED_TASKS[*]}"
        fi
        return 1
    fi

    # Baselines (accuracy in percent, 0-100) for MiMo-V2-Flash. Set None for
    # any task without a published baseline so it shows '--' instead of a
    # spurious comparison.
    local rc=0
    python3 - "${results_dir}" <<'PYEOF' || rc=$?
import json
import os
import sys
from pathlib import Path

BASELINE = {
    "aime":         89.5,
    "cmmlu":        87.8,
    "minerva_math": 95.0,
    "mmlu_pro":     78.2,
    "mmlu_redux":   91.3,
    "supergpqa":    50.7,
}

results_dir = Path(sys.argv[1])

# Only check the tasks that were actually requested (EVAL_TASKS env var),
# so subset runs don't produce spurious MISSING warnings for tasks that
# were never supposed to run. Fallback to all baselines if unset.
expected = os.environ.get("EVAL_TASKS", "").split() or sorted(BASELINE.keys())

# Keep the latest <task>_summary_<yyyymmdd_hhmmss>.json per task,
# using the embedded filename timestamp (lexicographic sort).
latest = {}
for path in results_dir.glob("*_summary_*.json"):
    task, sep, stamp = path.name.rpartition("_summary_")
    if not sep:
        continue
    if task not in latest or stamp > latest[task][0]:
        latest[task] = (stamp, path)

# Build rows once, use for both console table and markdown note.
rows = []
missing = []
for task in expected:
    entry = latest.get(task)
    if entry is None:
        score_str = "MISSING"
        missing.append(task)
    else:
        try:
            data = json.loads(entry[1].read_text())
            acc = float(data["accuracy"]) * 100
            score_str = f"{acc:.2f}"
        except Exception as exc:
            score_str = "ERR"
            missing.append(task)
            print(f"  [WARN] {entry[1].name}: {exc}", file=sys.stderr)
    bl = BASELINE.get(task)
    bl_str = f"{float(bl):.2f}" if bl is not None else "--"
    rows.append((task, score_str, bl_str))

print("SCORES:")
print(f"  {'Task':<20} {'Score':>8}   {'Baseline':>8}")
print(f"  {'-'*20} {'-'*8}   {'-'*8}")
for task, s, b in rows:
    print(f"  {task:<20} {s:>8}   {b:>8}")

print("")
if missing:
    print(f"MISSING/FAILED: {', '.join(missing)}")
else:
    print("ALL REQUESTED TASKS PASSED")

# Also write a markdown version for the MR note.
md_path = os.environ.get("EVAL_SUMMARY_MD", "").strip()
if md_path:
    label = os.environ.get("EVAL_LABEL", "EVAL")
    created = os.environ.get("CREATED", "")
    endpoint = os.environ.get("OPENAI_COMPATIBLE_API_URL", "")
    tasks_str = " ".join(expected)
    md_lines = [
        f"### {label} Accuracy Evaluation (mini_eval)",
        "",
        f"- **Image:** `{created}`",
        f"- **Endpoint:** `{endpoint}`",
        f"- **Tasks:** `{tasks_str}`",
        "",
        "| Task | Score | Baseline |",
        "| --- | ---: | ---: |",
    ]
    for task, s, b in rows:
        md_lines.append(f"| {task} | {s} | {b} |")
    md_lines.append("")
    if missing:
        md_lines.append(f"❌ **MISSING/FAILED:** {', '.join(missing)}")
    else:
        md_lines.append("✅ **ALL REQUESTED TASKS PASSED**")
    md_lines.append("")
    Path(md_path).write_text("\n".join(md_lines))

if missing:
    sys.exit(2)
PYEOF

    if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
        echo ""
        echo "FAILED TASKS (${#FAILED_TASKS[@]}):"
        for t in "${FAILED_TASKS[@]}"; do
            echo "  - ${t}"
        done
        echo "NOTE: Some tasks failed but the job continues (allow_failure)."
    fi

    return "${rc}"
}
