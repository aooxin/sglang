#!/usr/bin/env bash
# DFlash + PD 分离：Prefill 节点 (公开示例)
#
# Required PD args:
#   --disaggregation-mode prefill
#   --disaggregation-ib-device <NIC>          # must be reachable from the decode side
#   --disaggregation-transfer-backend mooncake
#
# DFlash-specific notes:
#   - Do NOT pass --speculative-eagle-topk / --speculative-num-steps; sglang's
#     server_args._handle_speculative_decoding_args forces num_steps=1, topk=1,
#     disable-overlap-schedule=true once --speculative-algorithm DFLASH is set.
#   - --speculative-num-draft-tokens should match draft config's
#     dflash_config.block_size.
#   - DP attention is supported. To enable, add:
#     --enable-dp-attention --enable-dp-lm-head --dp-size N

set -euo pipefail

# Point PYTHONPATH at your sglang checkout that contains the DFLASH branch.
export SGLANG_PATH=/sgl-workspace/sglang/python
export PYTHONPATH=$SGLANG_PATH:${PYTHONPATH:-}

unset NCCL_DEBUG_SUBSYS && unset NCCL_DEBUG && unset NVSHMEM_DEBUG_SUBSYS && unset NVSHMEM_DEBUG && export GLOG_minloglevel=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DEBUG_MEMORY_POOL=1
export SGLANG_DEBUG_SWA_INVARIANT=1
export SGLANG_DFLASH_DRAFT_SWA_RESERVED_POINTS=1024,2048,4096,8192,12288,16384,24576,32768
export SGLANG_DFLASH_DRAFT_SWA_RESERVED_CHUNK=14336
export SGLANG_DFLASH_DRAFT_SWA_RESERVED_MAX_POINT=131072
export GCACHE_MASTER_ADDR=wlcb-cloudml.juicefs.srv
export GCACHE_CACHE_GROUP=gcache-wlcb-idc-framework-test
export SGLANG_TEST_GCACHE_FI_TRANSFER_PROB=0.05
export SGLANG_TEST_GCACHE_FI_EXISTS_PROB=0.05
export SGLANG_TEST_GCACHE_FI_TIMEOUT_PROB=0.05
export SGLANG_TEST_GCACHE_FI_TIMEOUT_SLEEP_S=1.8
export ORG_ID=3002791
export CLOUDML_DEPLOY_ID=s-20260824110953-406sk-hsbd9

# ========== 按实际路径修改 ==========
TARGET_MODEL=/mnt/aie-shared-models-n-datasets-wlcb/models/mimo-v25-serve-0811-dflash/
DRAFT_MODEL=/mnt/aie-shared-models-n-datasets-wlcb/models/mimo-v25-serve-0811-dflash/dflash/
# ====================================

python3 -m sglang.launch_server \
    --model-path ${TARGET_MODEL} \
    --served-model-name mimo-flash-test-dflash \
    --disaggregation-mode prefill \
    --disaggregation-ib-device ${HPN_IB_HCA} \
    --disaggregation-transfer-backend mooncake \
    --log-level-http warning \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.75 \
    --chunked-prefill-size 32768 \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --quantization fp8 \
    --attention-backend fa3 \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --enable-metrics \
    --decode-log-interval 1 \
    --enable-cache-report \
    --collect-tokens-histogram \
    --enable-metrics-for-all-schedulers \
    --load-balance-method round_robin \
    --allow-auto-truncate \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DRAFT_MODEL} \
    --speculative-num-draft-tokens 8 \
    --disable-tokenizer-batch-decode \
    --tokenizer-worker-num 8 \
    --swa-full-tokens-ratio 0.022 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.2 \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend gcache \
    --numa-node 0 0 0 0 1 1 1 1 \
    --hicache-storage-backend-extra-config '{"hicache_storage_pass_prefix_keys": true}' \
    --disaggregation-two-phase-prealloc \
    --enable-trace \
    --otlp-traces-endpoint "http://preview-micollector-llm.api.xiaomi.net:80/v1/traces" \
    --otlp-traces-resource-attrs "{\"tracing.service.name\": \"${ORG_ID}_${CLOUDML_DEPLOY_ID}\", \"langfuse.project.name\": \"${ORG_ID}_MiMo-V2.5\"}"
