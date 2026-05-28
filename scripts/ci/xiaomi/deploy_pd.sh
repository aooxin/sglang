#!/usr/bin/env bash
# Deploy a PD-disaggregation service to CloudML.
#
# Usage:
#   deploy_pd.sh --deploy-yaml <path> [--deployment-name <name>]
#
# --deploy-yaml      (required) path to the deployment YAML config
# --deployment-name  defaults to "gitlab-ci-flash-1p1d-<CI_JOB_ID>"
#
# BYPASS MODE — skip CloudML deploy and point at a pre-deployed endpoint:
#   Set EVAL_URL=<bare-base-url> in the CI variables. The script
#   exports DEPLOYMENT_ENDPOINT from that URL and returns immediately —
#   no save-aksk, no create-deployment, no EXIT trap. The caller manages
#   the deployment's lifecycle.
#
#   Useful when (a) the workspace blocks OpenAPI writes, (b) you want to
#   iterate on eval logic against a long-lived test deployment without
#   paying the ~30 min deploy/teardown cycle each run.
#
#   The URL must be the bare base URL WITHOUT a trailing /v1 — runners
#   append /v1 themselves. Trailing slash and trailing /v1 are both
#   stripped defensively below, so any of these work:
#     https://flash-xxx.cloudml-hd-pri-ksytjv.alb.xiaomi.srv
#     https://flash-xxx.cloudml-hd-pri-ksytjv.alb.xiaomi.srv/
#     https://flash-xxx.cloudml-hd-pri-ksytjv.alb.xiaomi.srv/v1
set -euo pipefail

# --- Bypass: use a pre-deployed endpoint instead of creating one ---
# Must run BEFORE the CLOUDML_API line — that line expands ${CLOUDML_MAAS_HOME}
# under `set -u`, so unset CLOUDML envs would error out before we get a chance
# to skip them. Bypass mode is the whole point of letting CLOUDML envs be unset.
if [[ -n "${EVAL_URL:-}" ]]; then
    DEPLOYMENT_ENDPOINT="${EVAL_URL%/}"
    DEPLOYMENT_ENDPOINT="${DEPLOYMENT_ENDPOINT%/v1}"
    DEPLOYMENT_ID=""    # empty: any subsequent stop-deployment is a no-op
    echo "=== EVAL_URL set — bypassing CloudML deploy ==="
    echo "    Endpoint: ${DEPLOYMENT_ENDPOINT}"
    # Script is sourced; `return` ends sourcing without exiting the parent shell.
    return 0
fi

CLOUDML_API="${CLOUDML_MAAS_HOME}/scripts/cloudml_api.py"

# --- Parse arguments ---
DEPLOYMENT_NAME=""
DEPLOY_YAML=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deployment-name) DEPLOYMENT_NAME="$2"; shift 2 ;;
        --deploy-yaml)     DEPLOY_YAML="$2";     shift 2 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Validate required variables ---
REQUIRED_VARS=(
    CLOUDML_MAAS_HOME
    CLOUDML_WORKSPACE_ID
    CLOUDML_SERVICE_ID
    CLOUDML_ACCESS_KEY
    CLOUDML_SECRET_KEY
    SGLANG_IMAGE_URL
)

MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        MISSING+=("$var")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required variables: ${MISSING[*]}"
    exit 1
fi

# --- Validate arguments ---
if [[ -z "${DEPLOY_YAML}" ]]; then
    echo "ERROR: --deploy-yaml is required"
    exit 1
fi

DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-gitlab-ci-flash-1p1d-${CI_JOB_ID:-local}}"

# --- Temp files (cleaned up on exit) ---
RENDERED_DEPLOY_YAML=$(mktemp /tmp/pd-roles-config.XXXXXX.yaml)
DEPLOY_JSON=$(mktemp /tmp/pd-deployment.XXXXXX.json)
DEPLOYMENT_ID=""

wait_for_stopped() {
    local deployment_id="$1"
    local timeout=600
    local deadline=$(( $(date +%s) + timeout ))
    local state
    while (( $(date +%s) < deadline )); do
        state=$(python3 "${CLOUDML_API}" deployment-detail \
            --workspace-id "${CLOUDML_WORKSPACE_ID}" \
            --service-id "${CLOUDML_SERVICE_ID}" \
            --deployment-id "${deployment_id}" 2>/dev/null \
            | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("state",""))' 2>/dev/null || true)
        echo "  state: ${state:-<unknown>}"
        [[ "${state}" == "stopped" ]] && return 0
        sleep 10
    done
    echo "  timeout waiting for stop; proceeding to delete anyway"
    return 0
}

cleanup() {
    rm -f "${RENDERED_DEPLOY_YAML}" "${DEPLOY_JSON}"
    if [[ -n "${DEPLOYMENT_ID}" ]]; then
        echo "=== stopping deployment ${DEPLOYMENT_ID} ==="
        python3 "${CLOUDML_API}" stop-deployment \
            --workspace-id "${CLOUDML_WORKSPACE_ID}" \
            --service-id "${CLOUDML_SERVICE_ID}" \
            --deployment-id "${DEPLOYMENT_ID}" || true
        echo "=== waiting for deployment ${DEPLOYMENT_ID} to reach stopped state ==="
        wait_for_stopped "${DEPLOYMENT_ID}"
        echo "=== deleting deployment ${DEPLOYMENT_ID} ==="
        python3 "${CLOUDML_API}" delete-deployment \
            --workspace-id "${CLOUDML_WORKSPACE_ID}" \
            --service-id "${CLOUDML_SERVICE_ID}" \
            --deployment-id "${DEPLOYMENT_ID}" || true
    fi
}
trap cleanup EXIT

echo "=== CloudML PD Deployment ==="
echo "Workspace:  ${CLOUDML_WORKSPACE_ID}"
echo "Service:    ${CLOUDML_SERVICE_ID}"
echo "Deployment: ${DEPLOYMENT_NAME}"
echo "YAML:       ${DEPLOY_YAML}"

# --- Step 1: Save AK/SK ---
echo "=== saving credentials ==="
python3 "${CLOUDML_API}" save-aksk \
    --access-key "${CLOUDML_ACCESS_KEY}" \
    --secret-key "${CLOUDML_SECRET_KEY}" \
    --owner "${CLOUDML_OWNER:-ci-deploy}"

# --- Step 2: Render deploy YAML with image URL ---
ESCAPED_URL=$(printf '%s\n' "${SGLANG_IMAGE_URL}" | sed 's/[\\&|]/\\&/g')
sed "s|imageUrl: micr\.cloud\.mioffice\.cn/llm-infra/sglang:.*|imageUrl: ${ESCAPED_URL}|" \
    "${DEPLOY_YAML}" > "${RENDERED_DEPLOY_YAML}"

if [[ -n "${PREFILL_NUM:-}" ]]; then
    sed -i -E "s|^(  num: )[0-9]+([[:space:]]+# __PREFILL_NUM__)$|\1${PREFILL_NUM}\2|" "${RENDERED_DEPLOY_YAML}"
fi

if [[ -n "${DECODE_NUM:-}" ]]; then
    sed -i -E "s|^(  num: )[0-9]+([[:space:]]+# __DECODE_NUM__)$|\1${DECODE_NUM}\2|" "${RENDERED_DEPLOY_YAML}"
fi

if [[ -n "${ENCODER_NUM:-}" ]]; then
    sed -i -E "s|^(  num: )[0-9]+([[:space:]]+# __ENCODER_NUM__)$|\1${ENCODER_NUM}\2|" "${RENDERED_DEPLOY_YAML}"
fi

# --- Step 3: Create deployment ---
echo "=== creating multi-role deployment ==="
python3 "${CLOUDML_API}" create-multi-role-deployment \
    --deployment-name "${DEPLOYMENT_NAME}" \
    --workspace-id "${CLOUDML_WORKSPACE_ID}" \
    --service-id "${CLOUDML_SERVICE_ID}" \
    --from-file "@${RENDERED_DEPLOY_YAML}" | tee "${DEPLOY_JSON}"

read -r DEPLOYMENT_ID DEPLOYMENT_ENDPOINT < <(python3 -c "import json; d=json.load(open('${DEPLOY_JSON}')); print(d['deploymentId'], d.get('endpoint',''))")
echo "=== deployment created: ${DEPLOYMENT_ID} endpoint: ${DEPLOYMENT_ENDPOINT} ==="

# --- Step 4: Wait for deployment to be ready ---
echo "=== waiting for deployment to be ready ==="
python3 "${CLOUDML_API}" wait-for-deployment-ready \
    --workspace-id "${CLOUDML_WORKSPACE_ID}" \
    --service-id "${CLOUDML_SERVICE_ID}" \
    --deployment-id "${DEPLOYMENT_ID}" \
    --timeout 3600

echo "=== deployment is ready ==="

# --- Step 5: run the benchmark ---
echo "=== running benchmark ==="
# TODO: run the benchmark

# cleanup trap handles stop-deployment on both success and failure
