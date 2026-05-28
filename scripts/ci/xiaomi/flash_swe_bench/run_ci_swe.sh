#!/usr/bin/env bash
# CI entry for mini-swe-agent SWE-bench evaluation against an OpenAI-compatible
# server. Reads env vars set by the CI job, translates them into a JSON
# --override-config blob so the upstream yaml stays untouched, runs batch.py,
# then prints a human-readable summary via exp_stats.py.
#
# This script is BAKED INTO THE SWE-bench CI IMAGE at
#     /root/mini-swe-agent/scripts/run_ci_swe.sh
# The source of truth lives here in sglang-private so reviewers can trace what
# runs inside the image; the image Dockerfile copies this file in.
#
# Required env (caller MUST set):
#   OPENAI_API_BASE   /v1 URL of the model server, e.g.
#                     https://flash-xxx.cloudml-hd-pri-ksytjv.alb.xiaomi.srv/v1
#                     The "/v1" suffix is REQUIRED — the OpenAI client appends
#                     "/chat/completions" to base_url verbatim.
#
# Optional env (fall back to yaml defaults if unset):
#   MODEL_NAME        served-model-name (overrides config.environment.env.model)
#   ENV_CLASS         "docker" | "singularity" | ... (overrides config.environment.environment_class)
#   EXP_NAME          run id used for output dir (default: swe_<timestamp>)
#   WORKERS           concurrent agents (default: 50)
#   MAX_SLICE         max instances (default: 500)
#   CONFIG_FILE       relative yaml path (default: example_configs/swe_flash.yaml)
#   DATASET           parquet path (default: dataset/swebench.parquet)
#   OUTPUT_DIR        output base dir (default: ${REPO_DIR}/outputs)
#   FILTER_REGEX      --filter regex (default: skip sphinx-doc__sphinx-9320)
#   REPO_DIR          mini-swe-agent install path (default: /root/mini-swe-agent)
#
# Output (consumed by CI's summarize.py):
#   ${OUTPUT_DIR}/${EXP_NAME}/<instance>/*.traj.json   <-- one per instance
set -euo pipefail

: "${OPENAI_API_BASE:?ERROR: OPENAI_API_BASE not set}"

REPO_DIR="${REPO_DIR:-/root/mini-swe-agent}"
EXP_NAME="${EXP_NAME:-swe_$(date +%Y%m%d%H%M%S)}"
WORKERS="${WORKERS:-50}"
MAX_SLICE="${MAX_SLICE:-500}"
CONFIG_FILE="${CONFIG_FILE:-example_configs/swe_flash.yaml}"
DATASET="${DATASET:-dataset/swebench.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs}"
FILTER_REGEX="${FILTER_REGEX:-^(?!sphinx-doc__sphinx-9320$).*}"

# Build override JSON: include only keys the caller explicitly set, so omitted
# vars fall back to the yaml's defaults (deep_update on the python side).
OVERRIDE_JSON=$(python3 - <<'PY'
import json, os
o = {}
if os.environ.get("OPENAI_API_BASE"):
    o.setdefault("model", {}).setdefault("model_kwargs", {})["base_url"] = os.environ["OPENAI_API_BASE"]
if os.environ.get("MODEL_NAME"):
    o.setdefault("environment", {}).setdefault("env", {})["model"] = os.environ["MODEL_NAME"]
if os.environ.get("ENV_CLASS"):
    o.setdefault("environment", {})["environment_class"] = os.environ["ENV_CLASS"]
print(json.dumps(o))
PY
)

cd "${REPO_DIR}"

echo "=== SWE-bench CI run ==="
echo "  REPO_DIR:    ${REPO_DIR}"
echo "  EXP_NAME:    ${EXP_NAME}"
echo "  CONFIG:      ${CONFIG_FILE}"
echo "  DATASET:     ${DATASET}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}/${EXP_NAME}"
echo "  WORKERS:     ${WORKERS}"
echo "  MAX_SLICE:   ${MAX_SLICE}"
echo "  ENDPOINT:    ${OPENAI_API_BASE}"
echo "  MODEL_NAME:  ${MODEL_NAME:-<from yaml>}"
echo "  ENV_CLASS:   ${ENV_CLASS:-<from yaml>}"
echo "  OVERRIDE:    ${OVERRIDE_JSON}"
echo "========================"

batch_rc=0
python src/minisweagent/run/extra/batch.py \
    --output "${OUTPUT_DIR}/${EXP_NAME}" \
    --config "${CONFIG_FILE}" \
    --workers "${WORKERS}" \
    --dataset "${DATASET}" \
    --split test \
    --slice "0:${MAX_SLICE}" \
    --shuffle \
    --filter "${FILTER_REGEX}" \
    --override-config "${OVERRIDE_JSON}" || batch_rc=$?

# Human-readable summary in the CI log; never fail the job because of stats.
# exp_stats.py's find_experiments() looks under a hardcoded relative
# Path("outputs") and matches by name.startswith(prefix), so it only accepts
# a prefix like "swe_20260511155327" or "outputs/swe_..." — NOT an absolute
# path. We're already cd'd into REPO_DIR above, so the bare EXP_NAME is right.
echo ""
echo "=== exp_stats.py ==="
python scripts/exp_stats.py "${EXP_NAME}" || true

exit "${batch_rc}"
