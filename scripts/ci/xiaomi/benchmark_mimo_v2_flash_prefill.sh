#!/usr/bin/env bash
# Prefill benchmark for mimo-v2-flash.
# Single scenario: 16k input, output-len=1 (prefill-dominated, baseline-comparable).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-XiaomiMiMo/MiMo-V2-Flash}"
SERVED_NAME="${SERVED_NAME:-mimo-v2-flash-1207-sft}"
PORT="${PORT:-9001}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
LOG_FILE="mimo_v2_flash_server_$(date +%Y%m%d_%H%M%S).log"

setup_common_env

echo "=== launching mimo-v2-flash (model=${MODEL_PATH}) ==="
setsid python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --enable-cache-report \
    --pp-size 1 \
    --dp-size 2 \
    --enable-dp-attention \
    --tp-size 8 \
    --moe-a2a-backend deepep \
    --init-expert-location trivial \
    --ep-num-redundant-experts 0 \
    --eplb-algorithm deepseek \
    --decode-log-interval 1 \
    --page-size 1 \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.75 \
    --max-running-requests 128 \
    --chunked-prefill-size 65536 \
    --reasoning-parser qwen3 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --collect-tokens-histogram \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --enable-metrics \
    --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
    --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
    --ep-dispatch-algorithm static \
    --prefill-round-robin-balance \
    --attention-backend fa3 \
    --enable-metrics-for-all-schedulers \
    >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "=== server pid=${SERVER_PID}, log=${LOG_FILE} ==="

trap cleanup_bench_server EXIT INT TERM

wait_for_server_ready

echo "=== smoke test: POST /v1/chat/completions ==="
curl -fsS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
    | tee smoke_response.json
echo
grep -q '"content"' smoke_response.json


BENCH_COMMON_ARGS=(
    --backend sglang
    --model "${SERVED_NAME}"
    --tokenizer "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --dataset-name random
    --random-output-len 1
    --random-range-ratio 1.0
    --seed 12345
    --flush-cache
    --warmup-requests 100
)

# (input_len, label, rate, num_prompts)
BENCH_CONFIGS=(
    "16384 16k 4.65 1000"   # ttft <= 1s, max throughput, expected input tok/s ~ 75277
)

for cfg in "${BENCH_CONFIGS[@]}"; do
    read -r input_len label rate num_prompts <<< "$cfg"
    echo "=== benchmark: ${label} input, output=1, rate=${rate}, reqs=${num_prompts} ==="
    python3 -m sglang.bench_serving \
        "${BENCH_COMMON_ARGS[@]}" \
        --random-input-len "$input_len" \
        --request-rate "$rate" \
        --num-prompts "$num_prompts" \
        --output-file "benchmark_results_prefill-${label}-r${rate}.json"
done

echo "=== prefill benchmark complete ==="
