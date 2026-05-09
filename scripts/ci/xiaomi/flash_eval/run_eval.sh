#!/usr/bin/env bash
# MiMo-V2-Flash accuracy evaluation.
# Runs the 6 mini_eval text tasks against a CloudML-deployed sglang server.
#
# Usage:
#   run_eval.sh --deploy-yaml <path>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export EVAL_LABEL="FLASH"
# Forwards --deploy-yaml (and any other positional args) to deploy_pd.sh via eval_common.sh.
# shellcheck source=./eval_common.sh disable=SC1091
source "${SCRIPT_DIR}/eval_common.sh"

# Task list: default all 6; override via EVAL_TASKS env var (space-separated), e.g.
#   EVAL_TASKS="aime cmmlu" → only those two tasks run and get checked in the summary
DEFAULT_TASKS="aime cmmlu minerva_math mmlu_pro mmlu_redux supergpqa"
export EVAL_TASKS="${EVAL_TASKS:-${DEFAULT_TASKS}}"
read -ra TASKS <<<"${EVAL_TASKS}"

echo "=== ${EVAL_LABEL} Accuracy Evaluation (mini_eval) ==="
echo "Target:     ${OPENAI_COMPATIBLE_API_URL}"
echo "Run ID:     ${CREATED}"
echo "Output:     ${OUTPUT_BASE_DIR}"
echo "Tasks:      ${TASKS[*]}"

# K8s probes on each role (startupProbe + readinessProbe in the deploy
# YAML) already gate pod readiness at the infrastructure layer. This is
# just a final end-to-end smoke test to catch the brief window where K8s
# says Ready but the full Router → Prefill → Decode chain isn't really
# returning valid completions yet.
wait_for_service_ready

for task in "${TASKS[@]}"; do
    run_task "${task}"
done

# Copy results into CI_PROJECT_DIR so `artifacts:` can pick them up
# (GitLab artifacts only accept paths relative to CI_PROJECT_DIR).
ART_DIR="${CI_PROJECT_DIR:-.}/mini_eval_output"
mkdir -p "${ART_DIR}"
if [[ -d "${OUTPUT_BASE_DIR}/${CREATED}" ]]; then
    cp -r "${OUTPUT_BASE_DIR}/${CREATED}" "${ART_DIR}/"
    echo "=== artifacts copied to ${ART_DIR}/${CREATED} ==="
else
    echo "WARNING: no output dir at ${OUTPUT_BASE_DIR}/${CREATED}" >&2
fi

SUMMARY_RC=0
print_summary "${EVAL_LABEL}" || SUMMARY_RC=$?

# Post the markdown summary as a GitLab MR note (same pattern as the benchmark
# and codereview jobs). post_mr_note.sh gracefully skips when the job isn't
# part of a merge request or GITLAB_API_TOKEN is unset; don't let a note
# failure override the evaluation's own exit code.
if [[ -s "${EVAL_SUMMARY_MD:-}" ]]; then
    bash "${SCRIPT_DIR}/../post_mr_note.sh" "${EVAL_SUMMARY_MD}" || true
fi

echo "=== ${EVAL_LABEL} evaluation complete ==="
exit "${SUMMARY_RC}"
