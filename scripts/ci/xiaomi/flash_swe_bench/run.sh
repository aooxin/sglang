#!/usr/bin/env bash
# MiMo-V2-Flash SWE-bench accuracy evaluation.
#
# Deploys an sglang Flash service to CloudML, then runs the mini-swe-agent
# SWE-bench harness inside this CI image against that endpoint, parses the
# resulting trajectories for pass@1, and posts a markdown summary as an MR
# note.
#
# Usage:
#   run.sh --deploy-yaml <path>
#
# The image baked at /root/mini-swe-agent/scripts/run_ci_swe.sh is the
# contract surface: this script only sets the env vars it expects (see that
# file for the full list).
#
# IMPORTANT — runner-side prerequisite:
#   environment_class=docker (the Flash yaml's default) requires the runner pod
#   to expose a working docker daemon (host /var/run/docker.sock mounted, or
#   privileged + dind). If unavailable, set SWE_ENV_CLASS=singularity and use
#   an image with apptainer installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_PD="${SCRIPT_DIR}/../deploy_pd.sh"
POST_MR_NOTE="${SCRIPT_DIR}/../post_mr_note.sh"

# 1) Deploy CloudML; deploy_pd.sh registers an EXIT trap to stop the
#    deployment, so cleanup happens on both success and failure.
# shellcheck source=../deploy_pd.sh disable=SC1091
source "${DEPLOY_PD}" "$@"

: "${DEPLOYMENT_ENDPOINT:?ERROR: DEPLOYMENT_ENDPOINT is empty after deploy_pd.sh}"
export OPENAI_COMPATIBLE_API_URL="${DEPLOYMENT_ENDPOINT}"
export LOCAL_API_BASE="${OPENAI_COMPATIBLE_API_URL%/}/v1"

# Persist the CloudML deployment id to a known path so that ci_runner.sh's
# outer cleanup tier (running on the runner host, outside this container) can
# `docker cp` it out and call stop-deployment as a fallback if this shell's
# EXIT trap doesn't get to run (SIGKILL, force-rm, GitLab cancel, OOM, ...).
# DEPLOYMENT_ID is set inside deploy_pd.sh, accessible because we sourced it.
if [[ -n "${DEPLOYMENT_ID:-}" && -n "${CI_PROJECT_DIR:-}" ]]; then
    echo "${DEPLOYMENT_ID}" > "${CI_PROJECT_DIR}/.swe-deployment-id" 2>/dev/null || true
fi

# Image-tag-derived run id; same convention as flash_eval's CREATED. Reused as
# EXP_NAME so the output dir name matches the sglang image being evaluated.
_IMAGE_TAG="${SGLANG_IMAGE_URL:+${SGLANG_IMAGE_URL##*/}}"
_IMAGE_TAG="${_IMAGE_TAG:-unknown}"
CREATED="${_IMAGE_TAG//:/_}"
CREATED="${CREATED//\//_}"
export CREATED

export EVAL_LABEL="${EVAL_LABEL:-FLASH-SWE}"

# 2) End-to-end smoke against /v1/chat/completions to confirm the full
#    Router → Prefill → Decode chain returns content before firing thousands
#    of agent requests. K8s probes already gate pod readiness; this just
#    closes the brief window where K8s says Ready but the chain isn't.
# shellcheck source=../eval_smoke.sh disable=SC1091
source "${SCRIPT_DIR}/../eval_smoke.sh"
eval_smoke_test "${LOCAL_API_BASE%/}/chat/completions" "${CREATED}"

# 3) Run mini-swe-agent SWE-bench via the image-baked wrapper. The wrapper
#    translates these env vars into a --override-config JSON merged on top of
#    example_configs/swe_flash.yaml.
export OPENAI_API_BASE="${LOCAL_API_BASE}"
export MODEL_NAME="${SWE_MODEL_NAME:-mimo-v2-flash}"
export EXP_NAME="${CREATED}"
export WORKERS="${SWE_WORKERS:-50}"
export MAX_SLICE="${SWE_MAX_SLICE:-500}"
# Only set ENV_CLASS if user explicitly wants to override yaml default
# (yaml is "docker"). Setting it empty would still override.
if [[ -n "${SWE_ENV_CLASS:-}" ]]; then
    export ENV_CLASS="${SWE_ENV_CLASS}"
fi

WRAPPER="/root/mini-swe-agent/scripts/run_ci_swe.sh"
# We invoke as `bash "${WRAPPER}"` below, so the x bit isn't required —
# only check existence. -x would false-positive if image's COPY/chmod
# layer failed to preserve the executable bit.
if [[ ! -f "${WRAPPER}" ]]; then
    echo "ERROR: ${WRAPPER} not found in image" >&2
    exit 1
fi

echo "=== ${EVAL_LABEL} SWE-bench Evaluation ==="
echo "Endpoint:   ${OPENAI_COMPATIBLE_API_URL}"
echo "Model:      ${MODEL_NAME}"
echo "Run ID:     ${CREATED}"
echo "Workers:    ${WORKERS}"
echo "MaxSlice:   ${MAX_SLICE}"
echo "EnvClass:   ${ENV_CLASS:-<from yaml>}"

# Background heartbeat: print one short line every 60s with traj progress.
# Defends against "no log output" idle timeouts on GitLab side AND gives the
# operator a visible heartbeat without rich's ANSI-escape spam (which inflates
# the CI log to 100+ MB and gets jobs throttled / 403'd by GitLab).
#
# IMPORTANT: do NOT register an EXIT trap here — deploy_pd.sh already
# installed one (stop-deployment), and `trap ... EXIT` would overwrite it.
# The heartbeat is killed explicitly after the wrapper returns; if the
# parent shell dies mid-run, the heartbeat receives SIGHUP via job control
# and exits on its own.
OUTPUTS_DIR_INSIDE="/root/mini-swe-agent/outputs/${EXP_NAME}"
START_TS=$(date +%s)
heartbeat_loop() {
    while sleep 60; do
        local count elapsed_min elapsed_sec
        count=$(find "${OUTPUTS_DIR_INSIDE}" -maxdepth 2 -name "*.traj.json" 2>/dev/null | wc -l)
        elapsed_sec=$(( $(date +%s) - START_TS ))
        elapsed_min=$(( elapsed_sec / 60 ))
        echo "[heartbeat $(date '+%H:%M:%S')] traj.json=${count}/${MAX_SLICE} elapsed=${elapsed_min}m"
    done
}
heartbeat_loop &
HEARTBEAT_PID=$!

WRAPPER_RC=0
bash "${WRAPPER}" || WRAPPER_RC=$?

kill "${HEARTBEAT_PID}" 2>/dev/null || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true

# 4) Copy results into CI_PROJECT_DIR so artifacts: can pick them up.
ART_DIR="${CI_PROJECT_DIR:-.}/swe_bench_output"
mkdir -p "${ART_DIR}"
SRC_DIR="/root/mini-swe-agent/outputs/${EXP_NAME}"
if [[ -d "${SRC_DIR}" ]]; then
    cp -r "${SRC_DIR}" "${ART_DIR}/"
    echo "=== artifacts copied to ${ART_DIR}/${EXP_NAME} ==="
else
    echo "WARNING: no output dir at ${SRC_DIR}" >&2
fi

# 5) Compute pass@1 and write markdown summary.
md_name="eval_summary_$(echo "${EVAL_LABEL}" | tr '[:upper:]' '[:lower:]').md"
export EVAL_SUMMARY_MD="${CI_PROJECT_DIR:-.}/${md_name}"

SUMMARY_RC=0
python3 "${SCRIPT_DIR}/summarize.py" "${ART_DIR}/${EXP_NAME}" || SUMMARY_RC=$?

# 6) Post markdown to MR (skipped silently outside MR / without token).
if [[ -s "${EVAL_SUMMARY_MD:-}" ]]; then
    bash "${POST_MR_NOTE}" "${EVAL_SUMMARY_MD}" || true
fi

echo "=== ${EVAL_LABEL} evaluation complete ==="

# Surface the most informative non-zero exit code: wrapper failure > stats failure.
if [[ "${WRAPPER_RC}" -ne 0 ]]; then
    exit "${WRAPPER_RC}"
fi
exit "${SUMMARY_RC}"
