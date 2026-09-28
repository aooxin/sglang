rm -f ./sglang_server.log

export GCACHE_MASTER_ADDR="${GCACHE_MASTER_ADDR:-bj9-cloudml.juicefs.srv}"
export GCACHE_CACHE_GROUP="${GCACHE_CACHE_GROUP:-gcache-bj9-bcecn-dev-test}"

unset NCCL_DEBUG_SUBSYS && unset NCCL_DEBUG && unset NVSHMEM_DEBUG_SUBSYS && unset NVSHMEM_DEBUG
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
# Compile actual shapes on demand instead of sweeping all M values in a request.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

python3 -m sglang.launch_server \
              --model-path /preset-models/ \
              --served-model-name mimo-v2-pro-0520 \
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
              --decode-log-interval 1 \
              --page-size 64 \
              --host 0.0.0.0 \
              --port 9001 \
              --trust-remote-code \
              --watchdog-timeout 1000000 \
              --dist-timeout 1800 \
              --mem-fraction-static 0.80 \
              --chunked-prefill-size 32768 \
              --reasoning-parser mimo \
              --tool-call-parser mimo \
              --enable-metrics \
              --bucket-time-to-first-token 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 800 1200 1800 2400 3600 5400 7200 \
              --bucket-e2e-request-latency 0.1 0.2 0.4 0.6 0.8 1 2 4 6 8 10 20 40 60 80 100 200 400 600 1200 1800 2400 3600 5400 7200 \
              --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
              --load-balance-method round_robin \
              --attention-backend fa3 \
              --allow-auto-truncate \
              --enable-metrics-for-all-schedulers \
              --skip-server-warmup \
              --moe-dense-tp-size 1 \
              --enable-dp-lm-head \
              --enable-hierarchical-cache \
              --speculative-algorithm DFLASH \
              --speculative-draft-model-path /preset-models/dflash/ \
              --speculative-num-draft-tokens 8 \
              --speculative-draft-attention-backend fa3 \
              --hicache-ratio 2.5 \
              --hicache-mem-layout page_first \
              --hicache-write-policy write_through \
              --hicache-storage-prefetch-policy wait_complete \
              --hicache-storage-backend gcache \
              --hicache-storage-backend-extra-config '{"prefetch_threshold": 16}' \
              --dist-init-addr ${LWS_LEADER_IP}:20000 \
              --node-rank ${LWS_WORKER_INDEX} \
              --nnodes ${LWS_GROUP_SIZE} \
              --disable-tokenizer-batch-decode \
              --context-length 262144 \
              --numa-node 0 0 0 0 1 1 1 1 \
              --tokenizer-worker-num 64 --swa-full-tokens-ratio 0.3 \
              2>&1 | tee ./sglang_server.log
