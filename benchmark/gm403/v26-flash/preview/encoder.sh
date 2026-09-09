#!/usr/bin/env bash
set -euo pipefail

export TORCH_CUDNN_V8_API_DISABLED=1
export NCCL_IB_DISABLE=0
export MC_TE_METRIC=true
export SGLANG_ENCODER_MAX_BATCH_SIZE=8
export SGLANG_VLM_CACHE_SIZE_MB=921600
export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1
export SGLANG_IMAGE_LOADER_NUM_WORKERS=32
export SGLANG_MM_CACHE_SHM=1
export SGLANG_IGNORE_CKPT_TP_CHECK=1
export SGLANG_ENCODER_VIDEO_DECODE_NUM_THREADS=32
export SGLANG_MM_CACHE_URL_SHA1_ONLY=1
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export REQUEST_TIMEOUT=300
export SGLANG_ENCODER_SEND_TIMEOUT=300
export SGLANG_ENCODER_REQ_TIMEOUT=300
export SGLANG_FP8_IGNORED_LAYERS=self_attn.o_proj
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

MODEL_PATH="${MODEL_PATH:-/preset-models}"

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 9001 \
    --dp-size 8 \
    --encoder-only \
    --log-level info \
    --encoder-transfer-backend zmq_to_tokenizer \
    --disaggregation-ib-device ${HPN_IB_HCA} \
    --trust-remote-code \
    --enable-metrics \
    --enable-prefix-mm-cache \
    --tokenizer-backend fastokens
