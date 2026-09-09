#!/usr/bin/env bash
set -euo pipefail

DG_CACHE_NAME=mimo-v2.6-flash-deepgemm-ep8-cache-0510-0909-1054
DG_CACHE_DIR=/mnt/deepgemm-cache/${DG_CACHE_NAME}
if [ ! -d "${DG_CACHE_DIR}" ]; then
    wget -c "https://cnbj1-fds.api.xiaomi.net/aie-shared-objects/${DG_CACHE_NAME}.tar.gz"
    mkdir -p "${DG_CACHE_DIR}"
    tar -zxf "${DG_CACHE_NAME}.tar.gz" -C "${DG_CACHE_DIR}"
fi
export SGLANG_DG_CACHE_DIR=${DG_CACHE_DIR}

export SGLANG_HEALTH_CHECK_TIMEOUT=60
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=10
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=48
export SGLANG_DISAGGREGATION_QUEUE_SIZE=8
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=6
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_MOE_GATE=1
export SGLANG_SWA_EVICT_LENGTH_PER_STEP=256
export SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_MIN_CONTENT_LENGTH=16384
export SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_NGRAM_SIZE=300
export SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_WINDOW_SIZE=8300
export SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_MIN_REPEAT=15
export SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_DEBUG_SKIP_TRUNCATION=1
export SGLANG_FORWARD_UNKNOWN_TOOLS=1
export SGLANG_IO_WORKERS=16
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1024
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export SGLANG_DISCRETIZE_USER_TOOLCALL_TAGS=1
export SGLANG_REQUIRE_SPECIAL_TOOLCALL_TOKEN=1
export SGLANG_REMOTE_INSTANCE_SEED_RETRIES=3
export SGLANG_REMOTE_INSTANCE_SEED_SLEEP=2
export SGLANG_REMOTE_INSTANCE_SEED_STRATEGY=random
export REQUEST_TIMEOUT=300
export SGLANG_VIDEO_DECODE_NUM_THREADS=32
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_FP8_IGNORED_LAYERS=self_attn.o_proj
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_HUMMING_DISABLE_FUSED_E8M0=0
export SGLANG_MXFP4_W4A8_DG_SILU=0
export SGLANG_MIMO_V2_BF16_MOE_GATE=1

MODEL_PATH="${MODEL_PATH:-/preset-models}"
MODEL_NAME="${MODEL_NAME:-mimo-v2.6-flash-preview}"

DFLASH_MODEL_PATH="${DFLASH_MODEL_PATH:-/mnt/mify-gw-model-alicn3/models/dflash_model/iter-004000-hf/}"

MAX_BS=128

sglang serve \
    --model-path ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --disaggregation-ib-device ${HPN_IB_HCA} \
    --disaggregation-mode decode \
    --enable-cache-report \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-runner-backend mxfp4_w4a8 \
    --init-expert-location trivial \
    --ep-num-redundant-experts 0 \
    --eplb-algorithm deepseek \
    --decode-log-interval 1 \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.75 \
    --max-running-requests $((${MAX_BS}*2)) \
    --cuda-graph-max-bs ${MAX_BS} \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --context-length 1048576 \
    --collect-tokens-histogram \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --enable-metrics \
    --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
    --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
    --ep-dispatch-algorithm static \
    --attention-backend mimo_flash_attn \
    --allow-auto-truncate \
    --enable-metrics-for-all-schedulers \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DFLASH_MODEL_PATH} \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-attention-backend fa3 \
    --disable-tokenizer-batch-decode \
    --mm-enable-dp-encoder \
    --mm-attention-backend fa3 \
    --tokenizer-worker-num 8 \
    --tokenizer-backend fastokens \
    --swa-full-tokens-ratio 0.0166 \
    --load-balance-method round_robin \
    --enable-custom-logit-processor \
    --default-custom-logit-processor MiMoV2FlashNoRepeatNGramLogitProcessor \
    --enable-fork-tokenizer-workers
