#!/usr/bin/env bash
# Shell-executor entry for the flash-swe-bench GitLab job.
#
# This script runs ON THE RUNNER HOST (not inside the eval image). It exists
# because the sglang-A800-builder runner's docker CLI points at a REMOTE
# daemon (tj5-llm-g8a800-node05.kscn): bind-mounting ${CI_PROJECT_DIR} into
# the eval container fails because the host path isn't visible on the daemon
# side.
#
# Strategy: `docker build` streams a tar of our scripts to the daemon (path-
# independent), producing an ephemeral image that inherits mini-swe-agent-ci
# and has /work/scripts baked in. We then `docker run` (named, not --rm) and
# `docker cp` the artifacts out via the Docker API after the container exits.
#
# Required env (from GitLab CI):
#   CI_PROJECT_DIR          runner-side checkout path
#   SWE_EVAL_IMAGE          base image (micr.../mini-swe-agent-ci:V1)
#   SGLANG_IMAGE_URL        from release-docker-cu129 dotenv artifact
#   CLOUDML_ACCESS_KEY/SECRET_KEY/MAAS_HOME/WORKSPACE_ID/SERVICE_ID/OWNER
#   CI_JOB_ID / CI_API_V4_URL / CI_PROJECT_ID / CI_MERGE_REQUEST_IID
#   GITLAB_API_TOKEN
# Optional:
#   SWE_WORKERS / SWE_MAX_SLICE / SWE_MODEL_NAME / SWE_ENV_CLASS
set -euo pipefail

: "${CI_PROJECT_DIR:?CI_PROJECT_DIR is not set}"
: "${SWE_EVAL_IMAGE:?SWE_EVAL_IMAGE is not set (expected from .gitlab-ci.yml variables)}"

JOB_ID="${CI_JOB_ID:-local$$}"
CONTAINER="swe-eval-${JOB_ID}"
EPHEMERAL_IMAGE="swe-eval:${JOB_ID}"
DEPLOY_YAML="scripts/ci/xiaomi/flash_eval/flash-1p1d-gm403-hd.yaml"
# Must match run.sh's md filename convention: eval_summary_<lowercased EVAL_LABEL>.md
EVAL_SUMMARY_BASENAME="eval_summary_flash-swe.md"

DEPLOYMENT_ID_FILE="${CI_PROJECT_DIR}/.swe-deployment-id"

# If our eval container died before its own EXIT trap could call
# stop-deployment (SIGKILL, GitLab cancel, OOM, etc.), the CloudML deployment
# is still running and burning quota. run.sh persists the deployment id to
# /work/.swe-deployment-id inside the container; we extract it here via
# docker cp (works on stopped containers, and against remote daemons) and
# issue stop-deployment ourselves as a fallback.
stop_dangling_deployment() {
    docker cp "${CONTAINER}:/work/.swe-deployment-id" "${DEPLOYMENT_ID_FILE}" 2>/dev/null || true
    if [[ ! -s "${DEPLOYMENT_ID_FILE}" ]]; then
        return 0
    fi
    local dep_id
    dep_id=$(tr -d '[:space:]' < "${DEPLOYMENT_ID_FILE}")
    rm -f "${DEPLOYMENT_ID_FILE}"
    if [[ -z "${dep_id}" ]]; then
        return 0
    fi
    echo "=== fallback stop-deployment for ${dep_id} ==="
    # Spawn a short-lived helper container just for the cloudml_api call.
    # The fresh container has no /root/tmp/cloudml_aksk.json (each container
    # has its own private filesystem), so save-aksk MUST run first before
    # stop-deployment can authenticate. If the in-container trap already ran
    # stop-deployment this is still safe — stop on an already-stopped
    # deployment is a benign no-op.
    docker run --rm \
        --network host \
        -e CLOUDML_ACCESS_KEY -e CLOUDML_SECRET_KEY \
        -e CLOUDML_MAAS_HOME \
        "${SWE_EVAL_IMAGE}" \
        bash -c "
            python3 \${CLOUDML_MAAS_HOME}/scripts/cloudml_api.py save-aksk \
                --access-key \${CLOUDML_ACCESS_KEY} \
                --secret-key \${CLOUDML_SECRET_KEY} \
                --owner ci-fallback >/dev/null 2>&1 \
            && python3 \${CLOUDML_MAAS_HOME}/scripts/cloudml_api.py stop-deployment \
                --workspace-id ${CLOUDML_WORKSPACE_ID:-10042} \
                --service-id ${CLOUDML_SERVICE_ID} \
                --deployment-id ${dep_id}
        " 2>&1 | head -30 || true
}

cleanup() {
    # Order matters: extract deployment id BEFORE rm -f; cp on a removed
    # container fails silently.
    stop_dangling_deployment
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    docker rmi "${EPHEMERAL_IMAGE}" >/dev/null 2>&1 || true
    # Reclaim disk used by SWE-bench instance containers/images that
    # mini-swe-agent spawned. The runner host accumulates ~hundreds of GB
    # of `swebench/sweb.eval.*` images per run and they aren't reused
    # across SWE-bench versions, so we drop them after the eval finishes.
    # Filtering by "until=2h" keeps anything still in use by a concurrent
    # job alone.
    docker ps -aq --filter "name=minisweagent-" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
    docker container prune -f --filter "until=2h" >/dev/null 2>&1 || true
    docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null \
        | grep -iE 'swebench|sweb\.eval' \
        | awk '{print $2}' \
        | xargs -r docker rmi -f >/dev/null 2>&1 || true
    docker image prune -f >/dev/null 2>&1 || true
    echo "=== post-job docker disk usage ==="
    docker system df 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Pulling SWE eval base image ${SWE_EVAL_IMAGE} ==="
docker pull "${SWE_EVAL_IMAGE}"

echo "=== Building ephemeral image ${EPHEMERAL_IMAGE} (inherits ${SWE_EVAL_IMAGE}, bakes in scripts/) ==="
# Context = ${CI_PROJECT_DIR}/scripts (small, ~few MB). Using a stdin
# Dockerfile avoids touching the repo. WORKDIR /work lets run.sh resolve
# relative paths (scripts/ci/xiaomi/...).
docker build -t "${EPHEMERAL_IMAGE}" -f - "${CI_PROJECT_DIR}/scripts" <<DOCKERFILE
FROM ${SWE_EVAL_IMAGE}
COPY . /work/scripts
WORKDIR /work
DOCKERFILE

echo "=== Running SWE-bench evaluation ==="
# Named (not --rm) so artifacts can be extracted via docker cp after exit.
# --network host: share daemon-host's network stack so SSL fetches to
# raw.githubusercontent.com (done by mini-swe-agent during SWE-bench env
# setup) use the host's configured route / proxy / DNS — the default
# bridge NAT drops whatever custom networking the host has.
# /var/run/docker.sock is bind-mounted FROM the daemon host — inside the
# container it's the daemon's own socket, which is exactly what mini-swe-agent
# needs to spawn SWE instance containers (they'll be siblings on the same daemon).
# Intentionally NO `-t`, FORCE_COLOR, or RICH_FORCE_TERMINAL: a previous job
# with those settings produced 100+ MB of GitLab CI log filled with ANSI
# cursor escape sequences from rich's live progress display, which made
# GitLab throttle the runner (update-interval 3s → 1m) and eventually 403
# the job mid-eval. Without them, rich falls back to writing only the final
# summary table; run.sh emits its own once-a-minute heartbeat for visibility.
RUN_RC=0
docker run --name "${CONTAINER}" \
    --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e CLOUDML_ACCESS_KEY -e CLOUDML_SECRET_KEY \
    -e CLOUDML_MAAS_HOME \
    -e CLOUDML_WORKSPACE_ID -e CLOUDML_SERVICE_ID \
    -e CLOUDML_OWNER \
    -e SGLANG_IMAGE_URL \
    -e CI_JOB_ID -e CI_API_V4_URL -e CI_PROJECT_ID \
    -e CI_MERGE_REQUEST_IID -e GITLAB_API_TOKEN \
    -e CI_PROJECT_DIR=/work \
    -e SWE_WORKERS -e SWE_MAX_SLICE -e SWE_MODEL_NAME -e SWE_ENV_CLASS \
    -e EVAL_URL \
    -e PYTHONUNBUFFERED=1 \
    "${EPHEMERAL_IMAGE}" \
    bash scripts/ci/xiaomi/flash_swe_bench/run.sh --deploy-yaml "${DEPLOY_YAML}" \
    || RUN_RC=$?

echo "=== Extracting artifacts (docker cp uses API, works with remote daemon) ==="
# swe_bench_output/<EXP_NAME>/<instance>/... — the whole dir
docker cp "${CONTAINER}:/work/swe_bench_output" "${CI_PROJECT_DIR}/" 2>/dev/null || \
    echo "WARNING: no swe_bench_output to extract"
# Markdown summary (single known filename)
docker cp "${CONTAINER}:/work/${EVAL_SUMMARY_BASENAME}" "${CI_PROJECT_DIR}/" 2>/dev/null || \
    echo "WARNING: no ${EVAL_SUMMARY_BASENAME} to extract"

# cleanup trap handles container + ephemeral image removal.
exit "${RUN_RC}"
