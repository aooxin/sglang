#!/bin/bash
set -euo pipefail
export SGLANG_USE_FUSED_MOE_GATE=1
export SGLANG_ENABLE_SPEC_V2=1

MODEL_PATH="${MODEL_PATH:-/mnt/aie-shared-models-n-datasets/models/mimo-v2.5-int8-qat-2400-450-0824-v0}"

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --max-total-tokens 262144 \
    --decode-log-interval 1 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --quantization w8a8_int8 \
    --tp-size 8 \
    --max-running-requests 128 \
    --mem-fraction-static 0.8 \
    --enable-multi-layer-eagle \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 16384 \
    --attention-backend fa3 \
    --deepep-mode auto \
    --reasoning-parser qwen3 \
    --tool-call-parser mimo
