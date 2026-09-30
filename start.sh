#!/usr/bin/env bash

# MiMo Pro on 8 x GM504; Prefill benchmark based on document A's v2 settings.
# Reference: docs/504/MiMo-V2.5-Pro-FP4-DFlash  504测试.md (v2, commit 2dd80321).
# Mount the target checkpoint at /preset-models/ and DFLASH at /extra-models/.
# Keep DP2: TP8 / DP2 gives attention TP4 for this fused-QKV Pro checkpoint.
# Keep draft window 1024 for this run. The legacy BF16-router and
# pre-generated fake-top-k patches are deferred, so this differs from v2 + fake.
export SGLANG_DG_CACHE_DIR=/model/cache/deep_gemm
export SGLANG_CACHE_DIR=/model/cache/sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_FORWARD_UNKNOWN_TOOLS=1
export NCCL_TIMEOUT=1200
export SGLANG_WARMUP_TIMEOUT=1200
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1

exec python3 -m sglang.launch_server \
    --model-path /preset-models/ \
    --served-model-name mimo-v2.6-pro \
    --log-level-http warning \
    --enable-cache-report \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --ep-size 8 \
    --enable-dp-attention \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --decode-log-interval 1 \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 9001 \
    --trust-remote-code \
    --watchdog-timeout 1000000 \
    --mem-fraction-static 0.7 \
    --swa-full-tokens-ratio 0.08 \
    --chunked-prefill-size 49152 \
    --max-prefill-tokens 65536 \
    --reasoning-parser qwen3 \
    --tool-call-parser mimo \
    --enable-metrics \
    --quantization fp8 \
    --load-balance-method round_robin \
    --attention-backend fa4 \
    --allow-auto-truncate \
    --enable-metrics-for-all-schedulers \
    --skip-server-warmup \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /extra-models/ \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-window-size 1024 \
    --disable-tokenizer-batch-decode \
    --context-length 1048576 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-backend-decode disabled \
    --deepep-config '{"normal_dispatch": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 32, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 12, "num_max_rdma_chunked_recv_tokens": 128}, "normal_combine": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 16, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 12, "num_max_rdma_chunked_recv_tokens": 128}}' \
    --tokenizer-worker-num 8 \
    --disable-radix-cache
