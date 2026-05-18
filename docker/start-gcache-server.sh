#!/bin/bash

set -e

cd /home/work/gcache

# Check ENV
: ${GCACHE_MASTER_ADDR:?"Missing GCACHE_MASTER_ADDR"}
: ${GCACHE_CACHE_GROUP:?"Missing GCACHE_CACHE_GROUP"}
: ${GCACHE_CACHE_TYPE:?"Missing GCACHE_CACHE_TYPE"}
: ${GCACHE_SERVER_THREADS:?"Missing GCACHE_SERVER_THREADS"}
: ${GCACHE_SERVER_IFNAME:?"Missing GCACHE_SERVER_IFNAME"}

if [ "${GCACHE_CACHE_TYPE}" = "block" ]; then
    : ${GCACHE_SERVER_DISK_CAP_GB:?"Missing GCACHE_SERVER_DISK_CAP_GB (required for block cache)"}
    : ${CLOUDML_MODEL_PATH:?"Missing CLOUDML_MODEL_PATH (required for block cache)"}
else
    : ${GCACHE_SERVER_MEM_CAP_GB:?"Missing GCACHE_SERVER_MEM_CAP_GB (required for memory cache)"}
fi

GCACHE_SERVER_PREALLOC_MEM=${GCACHE_SERVER_PREALLOC_MEM:-false}

case "$GCACHE_SERVER_PREALLOC_MEM" in
  true|True|TRUE|1)
    if [ "${GCACHE_CACHE_TYPE}" = "memory" ]; then
      PREALLOC_SIZE=${GCACHE_SERVER_MEM_CAP_GB}
    else
      PREALLOC_SIZE=10
    fi
    ;;
  *)
    PREALLOC_SIZE=10
    ;;
esac

if [ "${GCACHE_CACHE_TYPE}" = "memory" ]; then
    CACHE_DIR_PREFIX="data"
else
    CACHE_DIR_PREFIX="${CLOUDML_MODEL_PATH}"
fi

CACHE_DIRS=$(printf "${CACHE_DIR_PREFIX}/%d," $(seq 1 ${GCACHE_SERVER_THREADS}))
CACHE_DIRS="${CACHE_DIRS%,}"    # Example: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16


if [ -n "${GCACHE_SERVER_MEM_CAP_GB}" ]; then
    MEM_CAP_PER_THREAD=$((GCACHE_SERVER_MEM_CAP_GB / GCACHE_SERVER_THREADS))
else
    MEM_CAP_PER_THREAD=1
fi
if [ -n "${GCACHE_SERVER_DISK_CAP_GB}" ]; then
    DISK_CAP_PER_THREAD=$((GCACHE_SERVER_DISK_CAP_GB / GCACHE_SERVER_THREADS))
else
    DISK_CAP_PER_THREAD=1
fi

nohup ./gcache-server \
    --master_addr=${GCACHE_MASTER_ADDR} \
    --cache_group=${GCACHE_CACHE_GROUP} \
    --cache-dirs=${CACHE_DIRS} \
    --cache_type=${GCACHE_CACHE_TYPE} \
    --mem_cache_capacity_GB=${MEM_CAP_PER_THREAD} \
    --disk_cache_capacity_GB=${DISK_CAP_PER_THREAD} \
    --log_level=1 \
    --rpc_socket_type=rdma \
    --use_byterpc=true \
    --server_ifname=${GCACHE_SERVER_IFNAME} \
    --cache_write_mode=write_only \
    --metric_exposer_addr=0.0.0.0:19090 \
    --mem_pool_prealloc_size_GB=${PREALLOC_SIZE} \
    --set_max_open_fds=false \
    --memory_cache_entry_expected_size_KB=8 \
    &> /dev/null &

while true; do
    if ss -tulnp state listening | grep ':19090' &> /dev/null; then
        break
    fi
    echo "Waiting gcache-server..."
    sleep 1
done
