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
export SGLANG_USE_FUSED_MOE_GATE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_FORWARD_UNKNOWN_TOOLS=1
export SGLANG_HICACHE_KEY_REFRESH_LRU_MAXSIZE=10485760
export SGLANG_HICACHE_KEY_REFRESH_INTERVAL=180
export SGLANG_RESERVE_SWA_SIZE_BEFORE_PREFETCHING=4096
export SGLANG_SIMPLE_SWA_RESERVED_POINTS="1024,2048,3072,4096,6144,8192,10240,12288,14336,16384,20480,24576,28672,32768,36864,40960,49152,57344,65536,73728,81920,90112,98304"
export SGLANG_SIMPLE_SWA_RESERVED_CHUNK=4096
export SGLANG_SIMPLE_SWA_RESERVED_MAX_POINT=614400
export GCACHE_MASTER_ADDR=bj9-cloudml-ali2.juicefs.srv
export GCACHE_CACHE_GROUP=gcache-alibj-mimo-online-2
export SGLANG_ENCODER_MAX_BATCH_SIZE=8
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export SGLANG_DISCRETIZE_USER_TOOLCALL_TAGS=1
export SGLANG_REQUIRE_SPECIAL_TOOLCALL_TOKEN=1
export SGLANG_REMOTE_INSTANCE_SEED_RETRIES=3
export SGLANG_REMOTE_INSTANCE_SEED_SLEEP=2
export SGLANG_REMOTE_INSTANCE_SEED_STRATEGY=random
export SGLANG_HICACHE_H2D_BLOCK_QUOTA=8
export SGLANG_HICACHE_D2H_BLOCK_QUOTA=8
export REQUEST_TIMEOUT=300
export SGLANG_PREFILL_FLOPS_COEFF=92207
export SGLANG_PREFILL_FLOPS_LINEAR_TERM=112187
export SGLANG_ENCODER_RECV_TIMEOUT=300
export SGLANG_FP8_IGNORED_LAYERS=self_attn.o_proj
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_MIMO_V2_BF16_MOE_GATE=1

MODEL_PATH="${MODEL_PATH:-/preset-models}"
MODEL_NAME="${MODEL_NAME:-mimo-v2.6-flash-preview}"

DFLASH_MODEL_PATH="${DFLASH_MODEL_PATH:-/mnt/mify-gw-model-alicn3/models/dflash_model/iter-004000-hf/}"

/home/work/gcache/wait-gcache-server.sh
sglang serve \
    --model-path ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --disaggregation-ib-device ${HPN_IB_HCA} \
    --disaggregation-mode prefill \
    --log-level-http warning \
    --enable-cache-report \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --moe-runner-backend deep_gemm \
    --decode-log-interval 1 \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 32768 \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --context-length 1048576 \
    --collect-tokens-histogram \
    --enable-metrics \
    --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
    --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --load-balance-method round_robin \
    --attention-backend mimo_flash_attn \
    --allow-auto-truncate \
    --enable-metrics-for-all-schedulers \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --enable-hierarchical-cache \
    --hicache-ratio 1.65 \
    --hicache-mem-layout page_first \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DFLASH_MODEL_PATH} \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-attention-backend fa3 \
    --disable-tokenizer-batch-decode \
    --mm-enable-dp-encoder \
    --mm-attention-backend fa3 \
    --tokenizer-worker-num 8 \
    --tokenizer-backend fastokens \
    --swa-full-tokens-ratio 0.1158 \
    --enable-encoder-endpoint-discovery \
    --encoder-urls http://${CLOUDML_MULTI_ROLE_SERVICE_ENCODER}-cip:9001 \
    --encoder-transfer-backend zmq_to_tokenizer \
    --language-only \
    --numa-node 0 0 0 0 1 1 1 1 \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend gcache \
    --hicache-storage-backend-extra-config '{"hicache_storage_pass_prefix_keys": true}' \
    --bucket-bandwidth 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 10 12 16 24 32 \
    --bucket-io 1 5 10 50 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 \
    --enable-fork-tokenizer-workers \
    --disable-piecewise-cuda-graph
