#!/usr/bin/env bash
# VIDEO modality evaluation against a deployed SGLang server.
# Runs inside the omni_eval_all:v1 Docker image.
#
# Required env vars:
#   OPENAI_COMPATIBLE_API_URL  - OpenAI-compatible API endpoint
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

echo "=== VIDEO Evaluation ==="
echo "Target:     ${OPENAI_COMPATIBLE_API_URL}"
echo "Run ID:     ${CREATED}"
echo "Output:     ${OUTPUT_BASE_DIR}"
echo ""

echo "######################################################################"
echo "# VIDEO EVALUATION"
echo "######################################################################"

cd /app/mimo-video-eval

export API_NUM_WORKERS=96
run_task "VIDEO" "dailyomni" "run_all.sh"

export API_NUM_WORKERS=32
for task in \
    videomme \
    videommev2; do
    run_task "VIDEO" "${task}" "run_all.sh"
done

print_summary "VIDEO"
echo "=== VIDEO evaluation complete ==="
