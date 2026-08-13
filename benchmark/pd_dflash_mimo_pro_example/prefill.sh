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
    --disaggregation-mode prefill \
    --disaggregation-ib-device ${NCCL_IB_HCA_NEW} \
    --disaggregation-transfer-backend mooncake \
    --pp-size 1 \
    --tp-size 8 \
    --page-size 128 \
    --host 0.0.0.0 \
    --port 29999 \
    --trust-remote-code \
    --mem-fraction-static 0.75 \
    --max-running-requests 128 \
    --chunked-prefill-size 32768 \
    --reasoning-parser qwen3 \
    --tool-call-parser mimo \
    --quantization fp8 \
    --dtype bfloat16 \
    --attention-backend fa4 \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}' \
    --enable-metrics \
    --skip-server-warmup \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ${DRAFT_MODEL} \
    --speculative-num-draft-tokens 8 \
    --context-length 262144
