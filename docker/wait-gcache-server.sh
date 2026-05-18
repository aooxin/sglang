#!/bin/bash

set -e

: ${GCACHE_MASTER_ADDR:?"Missing GCACHE_MASTER_ADDR"}
: ${GCACHE_CACHE_GROUP:?"Missing GCACHE_CACHE_GROUP"}

INTERVAL=${GCACHE_WAIT_INTERVAL:-10}
ELAPSED=0

PEERS_URL="http://${GCACHE_MASTER_ADDR}/groupcache/getPeers?group=${GCACHE_CACHE_GROUP}"
echo "Waiting for gcache-server..."

while true; do
    ADDR=$(curl -sf "${PEERS_URL}" 2>/dev/null \
        | jq -r '.data.Peers[]? | select(.addr != "") | .addr' 2>/dev/null \
        | head -1)
    if [ -n "${ADDR}" ]; then
        echo "gcache-server registered with master (addr: ${ADDR})"
        exit 0
    fi

    echo "Still waiting... (${ELAPSED}s elapsed)"
    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
done
