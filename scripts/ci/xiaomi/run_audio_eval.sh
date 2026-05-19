#!/usr/bin/env bash
# AUDIO modality evaluation against a deployed SGLang server.
# Runs inside the omni_eval_all:v1 Docker image.
#
# Required env vars:
#   OPENAI_COMPATIBLE_API_URL  - OpenAI-compatible API endpoint
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

echo "=== AUDIO Evaluation ==="
echo "Target:     ${OPENAI_COMPATIBLE_API_URL}"
echo "Run ID:     ${CREATED}"
echo "Output:     ${OUTPUT_BASE_DIR}"
echo ""

echo "######################################################################"
echo "# AUDIO EVALUATION"
echo "######################################################################"

mkdir -p /root/.cache/huggingface/omni_dataset/audio/
wget --progress=dot:giga -O - http://wlcb-cloudml-starfs.api.xiaomi.net/aie-shared-models-n-datasets-wlcb/datasets/omni/audio_data_15000/audio.tar.gz | tar -xz -C /root/.cache/huggingface/omni_dataset/audio/
ls -la /root/.cache/huggingface || true
ls -la /root/.cache/huggingface/omni_dataset/audio || true

cd /app/mimo-audio-eval

export API_NUM_WORKERS=96
for task in \
    asr_wenetspeech_test_net \
    asr_wenetspeech_test_meeting \
    asr_aishell1_test \
    asr_librispeech_test_clean \
    asr_librispeech_test_other; do
    run_task "AUDIO" "${task}" "run_api.sh"
done

export API_NUM_WORKERS=1
run_task "AUDIO" "mmau_mmau" "run_api.sh"

print_summary "AUDIO"
echo "=== AUDIO evaluation complete ==="
