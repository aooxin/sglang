#!/usr/bin/env bash
# Shared helpers for mimo-v2-flash bench scripts (prefill / decode / mtp_decode).
# Source at the top of a bench script. Callers must set the relevant globals
# before invoking each helper:
#   setup_common_env       — no inputs
#   wait_for_server_ready  — SERVER_PID, BASE_URL, STARTUP_TIMEOUT
#   cleanup_bench_server   — LOG_FILE  (registered via trap)

setup_common_env() {
    if [[ -d /mnt/deepgemm-cache ]]; then
        export SGLANG_DG_CACHE_DIR=/mnt/deepgemm-cache
        export CLOUDML_DG_CACHE_DIR=/mnt/deepgemm-cache
    fi
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
    export SGLANG_USE_FUSED_MOE_GATE=true
    export SGLANG_USE_UPDATE_LOCAL_ATTN_METADATA_CUDA=true
}

wait_for_server_ready() {
    echo "=== waiting for /health_generate (timeout=${STARTUP_TIMEOUT}s) ==="
    local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
    until curl -fs -o /dev/null "${BASE_URL}/health_generate" 2>/dev/null; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server process died during startup"
            exit 1
        fi
        if (( $(date +%s) > deadline )); then
            echo "server did not become ready within ${STARTUP_TIMEOUT}s"
            exit 1
        fi
        sleep 10
    done
    echo "=== server ready ==="
}

cleanup_bench_server() {
    local rc=$?
    echo "=== cleaning up server (exit=${rc}) ==="
    # PGID kill first: deterministic, gives workers grace to release shm.
    local pgid
    pgid=$(ps -o pgid= "${SERVER_PID:-0}" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" && "$pgid" != "0" ]]; then
        kill -TERM -"$pgid" 2>/dev/null || true
        sleep 10
        kill -KILL -"$pgid" 2>/dev/null || true
    fi
    # Backstop for processes that escaped the PGID; ignore exit.
    python3 python/sglang/cli/killall.py || true
    cleanup_shm_leaks
    echo "=== tail of server log ==="
    tail -n 200 "${LOG_FILE}" || true
    exit "${rc}"
}

# Remove /dev/shm artifacts left behind when sglang workers are SIGKILL'd
# before they can shm_unlink/sem_unlink themselves. Without this, leaked
# CUDA IPC handles break NCCL bootstrap on later runs and loky semaphores
# accumulate until reboot.
cleanup_shm_leaks() {
    find /dev/shm -maxdepth 1 -user "$(id -u)" -mmin -180 \
        \( -name 'cuda.shm.*' -o -name 'nccl-*' -o -name 'torch_*' \
           -o -name 'pymp-*' -o -name 'sem.loky-*' \) \
        -delete 2>/dev/null || true
}
