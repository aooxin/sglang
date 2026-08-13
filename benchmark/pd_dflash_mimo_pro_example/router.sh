#!/usr/bin/env bash
# DFlash + PD 分离：Router (公开示例)
#
# Router 不感知 spec 算法，仅负责按策略路由请求到 prefill / decode 服务。
# Set PREFILL_HOST / DECODE_HOST to the addresses where prefill.sh / decode.sh
# are running.

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"

python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://${PREFILL_HOST}:29999 \
    --decode http://${DECODE_HOST}:29999 \
    --prefill-policy cache_aware \
    --decode-policy round_robin \
    --host 0.0.0.0 \
    --port 29999
