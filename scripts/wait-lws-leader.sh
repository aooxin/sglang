#!/bin/bash

set -e

INTERVAL=${LWS_WAIT_INTERVAL:-5}
ELAPSED=0

echo "Waiting for LWS_LEADER_ADDRESS..."

while true; do
    if [ -n "${LWS_LEADER_ADDRESS}" ]; then
        echo "LWS_LEADER_ADDRESS is set (addr: ${LWS_LEADER_ADDRESS}), trying to resolve..."

        # 使用 getent hosts 解析域名，提取 IP
        RESOLVED_IP=$(getent hosts "${LWS_LEADER_ADDRESS}" | awk '{print $1}' | head -1)

        if [ -n "${RESOLVED_IP}" ]; then
            echo "Successfully resolved: ${LWS_LEADER_ADDRESS} -> ${RESOLVED_IP} (elapsed: ${ELAPSED}s)"
            exit 0
        else
            echo "getent failed to resolve ${LWS_LEADER_ADDRESS}, still waiting... (${ELAPSED}s elapsed)"
        fi
    else
        echo "Still waiting... (${ELAPSED}s elapsed)"
    fi

    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
done
