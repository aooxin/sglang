#!/usr/bin/env bash
# MiMo-V2-Flash tau-bench (telecom domain) accuracy evaluation.
#
# Deploys an sglang Flash service to CloudML, then runs the tau-bench harness
# inside this CI image against that endpoint. The image bakes /root/tau2-bench/
# with a run.sh that invokes `tau2 run` three times in a row (intentional —
# the team's eval protocol takes the average of three independent runs as the
# headline number while still surfacing per-run scores for variance).
#
# Usage:
#   run.sh --deploy-yaml <path>
#
# Required env (from .gitlab-ci.yml):
#   CLOUDML_*           consumed by deploy_pd.sh
#   SGLANG_IMAGE_URL    consumed by deploy_pd.sh
#   OPENROUTER_API_KEY  user-simulator LLM (qwen3-235b on openrouter); the
#                       tau-bench paper protocol uses a strong user model and
#                       openrouter is the cheapest path to qwen3-235b. Hard
#                       requirement — fail-fast if missing.
#
# The image at /root/tau2-bench/run.sh is the contract surface. We only set the
# four env vars it expects (OPENAI_API_BASE_AGENT/KEY/USER + OPENAI_API_KEY_USER)
# and let it run; everything model-name / domain / num-trials related is baked
# into that file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_PD="${SCRIPT_DIR}/../deploy_pd.sh"
POST_MR_NOTE="${SCRIPT_DIR}/../post_mr_note.sh"

# Image-baked tau-bench install. WORKDIR matters: the in-image run.sh assumes
# cwd == /root/tau2-bench so tau2 picks up the bundled data/ via TAU2_DATA_DIR.
TAU_DIR="/root/tau2-bench"
IMAGE_RUN_SH="${TAU_DIR}/run.sh"

# Fixed model name baked into the image's run.sh (--agent-llm openai/mimo_v2_flash_sft).
# We don't control it from here — keep this in sync with what the image actually
# requests. Used only for smoke-test routing; the sglang server doesn't gate on
# model name (any served-model-name is accepted), so name mismatch won't break
# the eval but would mask a real "model not loaded" smoke failure.
SMOKE_MODEL="mimo_v2_flash_sft"

# 1) Validate OpenRouter key BEFORE deploying — failing here saves ~5 min of
#    CloudML deploy + teardown cycle on a misconfiguration.
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2
    echo "  tau-bench's user simulator runs against openrouter (qwen3-235b)." >&2
    echo "  Add OPENROUTER_API_KEY as a masked CI variable in GitLab settings." >&2
    exit 1
fi

# 2) Deploy CloudML; deploy_pd.sh registers an EXIT trap to stop+delete the
#    deployment, so cleanup runs on success and failure paths alike.
# shellcheck source=../deploy_pd.sh disable=SC1091
source "${DEPLOY_PD}" "$@"

: "${DEPLOYMENT_ENDPOINT:?ERROR: DEPLOYMENT_ENDPOINT is empty after deploy_pd.sh}"
export OPENAI_COMPATIBLE_API_URL="${DEPLOYMENT_ENDPOINT}"
LOCAL_API_BASE="${OPENAI_COMPATIBLE_API_URL%/}/v1"

# Image-tag-derived run id; same convention as flash_swe_bench.
_IMAGE_TAG="${SGLANG_IMAGE_URL:+${SGLANG_IMAGE_URL##*/}}"
_IMAGE_TAG="${_IMAGE_TAG:-unknown}"
CREATED="${_IMAGE_TAG//:/_}"
CREATED="${CREATED//\//_}"
export CREATED

export EVAL_LABEL="${EVAL_LABEL:-FLASH-TAU}"

# 3) Smoke test the deployed endpoint with a chat completion that mirrors the
#    same chat_template_kwargs the eval uses (enable_thinking=true). Catches
#    "endpoint up but rejects sglang-specific kwargs" before launching the
#    long-running eval.
# shellcheck source=../eval_smoke.sh disable=SC1091
source "${SCRIPT_DIR}/../eval_smoke.sh"
eval_smoke_test \
    "${LOCAL_API_BASE%/}/chat/completions" \
    "${SMOKE_MODEL}" \
    '"chat_template_kwargs":{"enable_thinking":true,"keep_all_reasoning":true},"temperature":0.3'

# 4) Wire up the four env vars that /root/tau2-bench/run.sh's `${VAR:-default}`
#    indirection looks for. Agent points at our CloudML deployment; user points
#    at openrouter (qwen3-235b is the team's user-sim of choice).
export OPENAI_API_BASE_AGENT="${LOCAL_API_BASE}"
export OPENAI_API_KEY_AGENT="EMPTY"
export OPENAI_API_BASE_USER="https://openrouter.ai/api/v1"
export OPENAI_API_KEY_USER="${OPENROUTER_API_KEY}"

if [[ ! -f "${IMAGE_RUN_SH}" ]]; then
    echo "ERROR: ${IMAGE_RUN_SH} not found in image" >&2
    exit 1
fi

echo "=== ${EVAL_LABEL} tau-bench Evaluation ==="
echo "Endpoint:     ${LOCAL_API_BASE}"
echo "SmokeModel:   ${SMOKE_MODEL}"
echo "Run ID:       ${CREATED}"
echo "TauDir:       ${TAU_DIR}"
echo "ImageRunSh:   ${IMAGE_RUN_SH}"

# 5) Run the image's run.sh with stdout captured to a log. Two reasons:
#   - tau-bench's "Average Reward: <float>" line is the source of truth for
#     scoring; summarize.py greps it from this log.
#   - GitLab artifacts: pick up the log so failures can be triaged offline.
ART_DIR="${CI_PROJECT_DIR:-.}/tau_bench_output"
mkdir -p "${ART_DIR}"
RUN_LOG="${ART_DIR}/tau_eval.log"

# Background heartbeat: same defense as flash_swe_bench against GitLab idle
# log-throttling. Counts the number of "Average Reward:" lines in the log to
# show eval progress (expect 3 by the end). No EXIT trap — deploy_pd.sh owns
# that; we kill the heartbeat explicitly after the wrapper returns.
START_TS=$(date +%s)
heartbeat_loop() {
    while sleep 60; do
        local rewards elapsed_min elapsed_sec
        rewards=$(grep -c "Average Reward:" "${RUN_LOG}" 2>/dev/null || echo 0)
        elapsed_sec=$(( $(date +%s) - START_TS ))
        elapsed_min=$(( elapsed_sec / 60 ))
        echo "[heartbeat $(date '+%H:%M:%S')] runs_done=${rewards}/3 elapsed=${elapsed_min}m"
    done
}
heartbeat_loop &
HEARTBEAT_PID=$!

# cd into tau-bench so relative paths in the image's run.sh (likely
# `data/tau2/simulations/...`) resolve correctly.
#
# `tee` always returns 0, so the pipeline's $? reflects tee, not the wrapper.
# We capture PIPESTATUS[0] (LHS exit) directly the line after the pipeline —
# any intervening simple command (incl. `|| RUN_RC=$?`) would reset it.
( cd "${TAU_DIR}" && bash "${IMAGE_RUN_SH}" ) 2>&1 | tee "${RUN_LOG}"
RUN_RC="${PIPESTATUS[0]}"

kill "${HEARTBEAT_PID}" 2>/dev/null || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true

# 6) Best-effort: copy tau2's per-task simulation JSON for offline inspection.
#    Path is convention-driven (tau2 README says data/tau2/simulations/); we
#    don't fail if it's empty since stdout grep is the primary metric path.
SIM_SRC="${TAU_DIR}/data/tau2/simulations"
if [[ -d "${SIM_SRC}" ]]; then
    cp -r "${SIM_SRC}" "${ART_DIR}/simulations" || true
    echo "=== copied ${SIM_SRC} -> ${ART_DIR}/simulations ==="
else
    echo "WARNING: no simulations dir at ${SIM_SRC} (not fatal — log-grep is primary)"
fi

# 7) Compute average reward from the run log + write markdown summary.
md_name="eval_summary_$(echo "${EVAL_LABEL}" | tr '[:upper:]' '[:lower:]').md"
export EVAL_SUMMARY_MD="${CI_PROJECT_DIR:-.}/${md_name}"

SUMMARY_RC=0
python3 "${SCRIPT_DIR}/summarize.py" "${RUN_LOG}" || SUMMARY_RC=$?

# 8) Post markdown to MR (skipped silently outside MR / without token).
if [[ -s "${EVAL_SUMMARY_MD:-}" ]]; then
    bash "${POST_MR_NOTE}" "${EVAL_SUMMARY_MD}" || true
fi

echo "=== ${EVAL_LABEL} evaluation complete ==="

# Surface the most informative non-zero exit code (eval failure > stats failure).
if [[ "${RUN_RC}" -ne 0 ]]; then
    exit "${RUN_RC}"
fi
exit "${SUMMARY_RC}"
