#!/usr/bin/env bash
# DFlash + PD 分离：Decode 节点 (公开示例)
#
# Required PD args:
#   --disaggregation-mode decode
#   --disaggregation-ib-device <NIC>
#   --disaggregation-transfer-backend mooncake
#
# DFlash-specific notes:
#   - --disable-radix-cache is forced on by sglang anyway for decode role; we
#     keep it explicit.
#   - Do NOT pass --disaggregation-decode-tp / --disaggregation-decode-dp;
#     server_args asserts they must be unset on the decode role.
#   - When dp_size > 1, --prefill-round-robin-balance may be required by sglang.
#   - DP attention is supported. To enable, add:
#     --enable-dp-attention --enable-dp-lm-head --dp-size N

set -euo pipefail

# Point PYTHONPATH at your sglang checkout that contains the DFLASH branch.
export SGLANG_PATH=/sgl-workspace/sglang/python
export PYTHONPATH=$SGLANG_PATH:${PYTHONPATH:-}

unset NCCL_DEBUG_SUBSYS && unset NCCL_DEBUG && unset NVSHMEM_DEBUG_SUBSYS && unset NVSHMEM_DEBUG && export GLOG_minloglevel=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export NCCL_GRAPH_MIXING_SUPPORT=0
export SGLANG_HICACHE_H2D_BLOCK_QUOTA=4
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=96

# ========== 按实际路径修改 ==========
TARGET_MODEL=/mnt/aie-shared-models-n-datasets-wlcb/models/mimo-v25-serve-0811-dflash/
DRAFT_MODEL=/mnt/aie-shared-models-n-datasets-wlcb/models/mimo-v25-serve-0811-dflash/dflash/
# ====================================

python3 -m sglang.launch_server \
    --model-path ${TARGET_MODEL} \
    --served-model-name mimo-flash-test-dflash \
    --disaggregation-mode decode \
    --disaggregation-ib-device ${HPN_IB_HCA} \
    --disaggregation-transfer-backend mooncake \
    --log-level-http warning \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --enable-dp-attention \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --max-running-requests 128 \
    --cuda-graph-max-bs 64 \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --quantization fp8 \
    --attention-backend fa3 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
    --disable-radix-cache \
    --enable-metrics \
    --skip-server-warmup \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DRAFT_MODEL} \
    --speculative-num-draft-tokens 8 \
    --enable-cache-report \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --init-expert-location trivial \
    --ep-num-redundant-experts 0 \
    --eplb-algorithm deepseek \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --collect-tokens-histogram \
    --ep-dispatch-algorithm static \
    --allow-auto-truncate \
    --enable-metrics-for-all-schedulers \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --disable-tokenizer-batch-decode \
    --tokenizer-worker-num 8 \
    --swa-full-tokens-ratio 0.3 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --enable-pd-direct2host \
    --pd-direct2host-host-ratio 3.2 \
    --hicache-io-backend kernel \
    --deepep-ll-num-sms 76 \
    --disaggregation-two-phase-prealloc
