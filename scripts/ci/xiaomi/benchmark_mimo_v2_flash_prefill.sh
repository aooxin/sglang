#!/usr/bin/env bash
# Prefill benchmark for mimo-v2-flash.
# Single scenario: 16k input, output-len=1 (prefill-dominated, baseline-comparable).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-XiaomiMiMo/MiMo-V2-Flash}"
SERVED_NAME="${SERVED_NAME:-mimo-v2-flash-1207-sft}"
PORT="${PORT:-9001}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
LOG_FILE="mimo_v2_flash_server_$(date +%Y%m%d_%H%M%S).log"

if [[ -d /mnt/deepgemm-cache ]]; then
    export SGLANG_DG_CACHE_DIR=/mnt/deepgemm-cache
    export CLOUDML_DG_CACHE_DIR=/mnt/deepgemm-cache
fi
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_USE_FUSED_MOE_GATE=true
export SGLANG_USE_UPDATE_LOCAL_ATTN_METADATA_CUDA=true

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
SERVER_PGID=$(ps -o pgid= "${SERVER_PID}" | tr -d ' ')
echo "=== server pid=${SERVER_PID} pgid=${SERVER_PGID}, log=${LOG_FILE} ==="

cleanup() {
    local rc=$?
    echo "=== cleaning up server (pgid=${SERVER_PGID}, exit=${rc}) ==="
    if [[ -n "${SERVER_PGID:-}" ]]; then
        kill -TERM -"${SERVER_PGID}" 2>/dev/null || true
        sleep 10
        kill -KILL -"${SERVER_PGID}" 2>/dev/null || true
    fi
    echo "=== tail of server log ==="
    tail -n 200 "${LOG_FILE}" || true
    exit "${rc}"
}
trap cleanup EXIT INT TERM

echo "=== waiting for /health_generate (timeout=${STARTUP_TIMEOUT}s) ==="
DEADLINE=$(( $(date +%s) + STARTUP_TIMEOUT ))
until curl -fs -o /dev/null "${BASE_URL}/health_generate" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "server process died during startup"
        exit 1
    fi
    if (( $(date +%s) > DEADLINE )); then
        echo "server did not become ready within ${STARTUP_TIMEOUT}s"
        exit 1
    fi
    sleep 10
done
echo "=== server ready ==="

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
