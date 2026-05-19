#!/usr/bin/env bash
# IMAGE modality evaluation against a deployed SGLang server.
# Runs inside the omni_eval_all:v1 Docker image.
#
# Required env vars:
#   OPENAI_COMPATIBLE_API_URL  - OpenAI-compatible API endpoint
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

echo "=== IMAGE Evaluation ==="
echo "Target:     ${OPENAI_COMPATIBLE_API_URL}"
echo "Run ID:     ${CREATED}"
echo "Output:     ${OUTPUT_BASE_DIR}"
echo ""

echo "######################################################################"
echo "# IMAGE EVALUATION"
echo "######################################################################"

mkdir -p /root/.cache/huggingface/omni_dataset/image/
wget --progress=dot:giga -O - http://wlcb-cloudml-starfs.api.xiaomi.net/aie-shared-models-n-datasets-wlcb/datasets/omni/image_eval/image.tar.gz | tar -xz -C /root/.cache/huggingface/omni_dataset/image/
ls -la /root/.cache/huggingface || true
ls -la /root/.cache/huggingface/omni_dataset/image || true

export API_NUM_WORKERS=64
export ENABLE_THINKING=1
export GEN_KWARGS="--gen_kwargs max_new_tokens=65536,temperature=1,top_p=0.95"
export LMMS_EVAL_DATA_HOME="/root/.cache/huggingface/omni_dataset/image/"
export HF_HOME="/root/.cache/huggingface/omni_dataset/image/.cache/"

cd /app/lmms-eval
for task in \
    countbench_boxed_mimo \
    mmmu_val_boxed_gpt \
    mmmu_pro_standard_boxed_gpt \
    charxiv_boxed_mimo; do
    run_task "IMAGE" "${task}" "eval_script/api/run_ci.sh"
done

export QWEN_RESIZE_MAX_PIXELS=0
export RELATIVE_COORD_SCALE=999
export API_NUM_WORKERS=32

# TODO: The concurrency needs to be modified after code optimization
# task:screenspot_pro_en_rec_test_rel_mimo score may be distorted due to D oom
for task in \
    osworld_g_rec_test_rel_mimo \
    refcoco+_bbox_rec_val_rel_mimo \
    screenspot_pro_en_rec_test_rel_mimo; do
    run_task "IMAGE" "${task}" "eval_script/api/run_ci.sh"
done

print_summary "IMAGE"
echo "=== IMAGE evaluation complete ==="
