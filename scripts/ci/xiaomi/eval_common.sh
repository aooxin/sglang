#!/usr/bin/env bash
# Common setup for omni evaluation scripts.
# Source this file, do not execute directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PD="${SCRIPT_DIR}/deploy_pd.sh"
source "${DEPLOY_PD}" --deployment-name "gitlab-ci-omni-single-${CI_JOB_ID:-local}" "$@"

: "${DEPLOYMENT_ENDPOINT:?ERROR: DEPLOYMENT_ENDPOINT is empty, deployment may have failed}"
export OPENAI_COMPATIBLE_API_URL="${DEPLOYMENT_ENDPOINT}/v1"
export OUTPUT_BASE_DIR="/tmp/eval_result"

# remove eval output
trap 'rc=$?; rm -rf "${OUTPUT_BASE_DIR}"; cleanup; exit $rc' EXIT

IMAGE_TAG="${SGLANG_IMAGE_URL:+${SGLANG_IMAGE_URL##*/}}"
IMAGE_TAG="${IMAGE_TAG:-unknown}"
export CREATED="${IMAGE_TAG}"
mkdir -p "${OUTPUT_BASE_DIR}"

FAILED_TASKS=()

run_task() {
    local modality="$1"
    local task="$2"
    local script="$3"
    echo ""
    echo "================================================================"
    echo "=== [${modality}] ${task}"
    echo "================================================================"
    if bash "${script}" "${task}" "${CREATED}"; then
        echo "=== [${modality}] ${task} - PASSED ==="
    else
        echo "=== [${modality}] ${task} - FAILED (continuing) ==="
        FAILED_TASKS+=("${modality}:${task}")
    fi
}

print_summary() {
    local modality="$1"
    echo ""
    echo "######################################################################"
    echo "# ${modality} EVALUATION SUMMARY"
    echo "######################################################################"
    echo "Run ID:  ${CREATED}"
    echo "Output:  ${OUTPUT_BASE_DIR}"
    echo ""

    # Extract scores from _results.json
    # Directory structure: OUTPUT_BASE_DIR/CREATED/TASK/DATE_STR/*_results.json
    local results_dir="${OUTPUT_BASE_DIR}/${CREATED}"
    local scores_file="/tmp/omni_scores_${modality}.txt"
    local scores_md_file="/tmp/omni_scores_${modality}.md"
    : > "${scores_file}"
    : > "${scores_md_file}"
    if [[ -d "${results_dir}" ]]; then
        echo "SCORES:"
        python3 "${SCRIPT_DIR}/omni_eval_result_extract.py" \
            "${results_dir}" "${scores_file}" "${scores_md_file}"
        echo ""
    fi

    if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
        echo "FAILED TASKS (${#FAILED_TASKS[@]}):"
        for t in "${FAILED_TASKS[@]}"; do
            echo "  - ${t}"
        done
        echo ""
        echo "NOTE: Some tasks failed but the job continues (allow_failure)."
    else
        echo "ALL TASKS PASSED"
    fi

    # Post scores to MR note (skipped silently when not running on an MR or when
    # GITLAB_API_TOKEN is unset).
    local post_script
    post_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/post_mr_note.sh"
    if [[ -s "${scores_md_file}" && -x "${post_script}" ]]; then
        local note_file="/tmp/omni_note_${modality}.md"
        {
            echo "## Omni Eval (${modality}) Result"
            echo
            cat "${scores_md_file}"
        } > "${note_file}"
        bash "${post_script}" "${note_file}" || true
    fi
}
