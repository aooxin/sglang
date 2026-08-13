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
export SGLANG_PATH=/path/to/sglang/python
export PYTHONPATH=$SGLANG_PATH:${PYTHONPATH:-}

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

# ========== 按实际路径修改 ==========
TARGET_MODEL=/path/to/target_model
DRAFT_MODEL=/path/to/dflash_draft_model
# ====================================

python3 -m sglang.launch_server \
    --model-path ${TARGET_MODEL} \
    --served-model-name mimo-v2-pro \
    --disaggregation-mode decode \
    --disaggregation-ib-device ${NCCL_IB_HCA_NEW} \
    --disaggregation-transfer-backend mooncake \
    --pp-size 1 \
    --tp-size 8 \
    --page-size 128 \
    --host 0.0.0.0 \
    --port 29999 \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --max-running-requests 128 \
    --cuda-graph-max-bs 64 \
    --reasoning-parser qwen3 \
    --tool-call-parser mimo \
    --quantization fp8 \
    --dtype bfloat16 \
    --attention-backend fa4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
    --disable-radix-cache \
    --enable-metrics \
    --skip-server-warmup \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DRAFT_MODEL} \
    --speculative-num-draft-tokens 8 \
    --context-length 262144
