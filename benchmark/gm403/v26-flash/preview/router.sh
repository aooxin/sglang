#!/usr/bin/env bash
set -euo pipefail

export LANGFUSE_HOST=cnbj1-micollector-llm.api.xiaomi.net
export NACOS_ADDR=cnbj1-nacos.api.xiaomi.net:80
export NACOS_NAMESPACE_ID=llm_router_prod
export NACOS_GROUP=common
export NACOS_USERNAME=llm_router_prod
export NACOS_PASSWORD="${NACOS_PASSWORD:-}"
export SERVICE_MODEL=MiMo-V26
export TRACING_SERVICE_NAME=${ORG_ID}_${CLOUDML_DEPLOY_ID}
export LANGFUSE_PROJECT_NAME=${ORG_ID}_MiMo-V2.6

ARGS=(
    -n ns-${ORG_ID}
    --sglangVersion 0.5.10
    --prefillService ${CLOUDML_MULTI_ROLE_SERVICE_PREFILL}
    --prefillPort 9001
    --decoderService ${CLOUDML_MULTI_ROLE_SERVICE_DECODE}
    --decoderPort 9001
    --inferMode 1
    -e alybj3
    --routerPort 9001
    --framework SGLang
    --traceIDKey X-Model-Request-Id
    --prefillRouter kv_cache
    --hashFromModel=false
    --redisExpire 3600
    --sliceNum 64
    --cacheThreshold 0.3
    --PrefillMetricsUri v1/loads
    --kvCacheStorageMode hybrid
    --decoderRouter load_balance
    --DecodeMetricsUri v1/loads
    --enableDPRouter=true
    --enableMultiNode=true
    --metricPullInterval 5s
)
./manager "${ARGS[@]}"
