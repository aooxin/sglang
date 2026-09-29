rm -f ./sglang_server.log

unset NCCL_DEBUG_SUBSYS && unset NCCL_DEBUG && unset NVSHMEM_DEBUG_SUBSYS && unset NVSHMEM_DEBUG
unset SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
# DFLASH verify: 96 requests * 8 tokens / (TP 8 / DP 2) = 192 tokens per rank.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

python3 -m sglang.launch_server \
              --model-path /preset-models/ \
              --served-model-name mimo-v2.6-pro-0520 \
              --disaggregation-ib-device ${HPN_IB_HCA} \
              --disaggregation-mode decode \
              --enable-cache-report \
              --pp-size 1 \
              --dp-size 2 \
              --tp-size 8 \
              --enable-dp-attention \
              --moe-a2a-backend deepep \
              --deepep-mode low_latency \
              --moe-runner-backend deep_gemm \
              --quantization fp8 \
              --init-expert-location trivial \
              --ep-num-redundant-experts 0 \
              --eplb-algorithm deepseek \
              --decode-log-interval 1 \
              --page-size 128 \
              --host 0.0.0.0 \
              --port 9001 \
              --trust-remote-code \
              --watchdog-timeout 1000000 \
              --dist-timeout 1800 \
              --mem-fraction-static 0.7 \
              --max-running-requests 192 \
              --cuda-graph-max-bs-decode 96 \
              --cuda-graph-bs-decode 96 \
              --reasoning-parser mimo \
              --tool-call-parser mimo \
              --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
              --enable-metrics \
              --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
              --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
              --ep-dispatch-algorithm static \
              --attention-backend fa3 \
              --allow-auto-truncate \
              --disable-radix-cache \
              --enable-metrics-for-all-schedulers \
              --speculative-algorithm DFLASH \
              --speculative-draft-model-path /extra-models/ \
              --speculative-draft-model-quantization fp8 \
              --speculative-draft-window-size 1024 \
              --speculative-num-draft-tokens 8 \
              --speculative-draft-attention-backend fa3 \
              --moe-dense-tp-size 1 \
              --enable-dp-lm-head \
              --dist-init-addr ${LWS_LEADER_IP}:20000 \
              --node-rank ${LWS_WORKER_INDEX} \
              --nnodes ${LWS_GROUP_SIZE} \
              --disable-tokenizer-batch-decode \
              --context-length 1048576 \
              --load-balance-method round_robin \
              --tokenizer-worker-num 64 --swa-full-tokens-ratio 0.13 \
              --disable-prefill-cuda-graph \
              2>&1 | tee ./sglang_server.log
