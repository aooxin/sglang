#!/usr/bin/env bash
# Deploy a PD-disaggregation service to CloudML.
# Uses flash-1p1d-gm403.yaml as the roles config, replacing imageUrl for
# Prefill/Decode roles with SGLANG_IMAGE_URL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLOUDML_API="${SCRIPT_DIR}/cloudml_api.py"
ROLES_CONFIG="${SCRIPT_DIR}/flash-1p1d-gm403.yaml"
MODEL_REPO_CONFIG="${SCRIPT_DIR}/flash-model-repo.yaml"

# --- Validate required variables ---
REQUIRED_VARS=(
    CLOUDML_WORKSPACE_ID
    CLOUDML_SERVICE_ID
    CLOUDML_AREA
    CLOUDML_QUEUE_NAME
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

CLOUDML_DEPLOYMENT_NAME="gitlab-ci-flash-1p1d-${CI_JOB_ID:-local}"

echo "=== CloudML PD Deployment ==="
echo "Workspace:  ${CLOUDML_WORKSPACE_ID}"
echo "Service:    ${CLOUDML_SERVICE_ID}"
echo "Deployment: ${CLOUDML_DEPLOYMENT_NAME}"
echo "Area:       ${CLOUDML_AREA}"
echo "Queue:      ${CLOUDML_QUEUE_NAME}"
echo "Image:      ${SGLANG_IMAGE_URL}"

# --- Step 1: Save AK/SK ---
echo "=== saving credentials ==="
python3 "${CLOUDML_API}" save-aksk \
    --access-key "${CLOUDML_ACCESS_KEY}" \
    --secret-key "${CLOUDML_SECRET_KEY}" \
    --owner "${CLOUDML_OWNER:-ci-deploy}"

# --- Step 2: Replace SGLang imageUrl (all non-Router roles) ---
RENDERED="/tmp/pd-roles-config.yaml"
ESCAPED_URL=$(printf '%s\n' "${SGLANG_IMAGE_URL}" | sed 's/[\\&/]/\\&/g')
sed "s|imageUrl: micr\.cloud\.mioffice\.cn/llm-infra/sglang:.*|imageUrl: ${ESCAPED_URL}|" \
    "${ROLES_CONFIG}" > "${RENDERED}"

# --- Step 3: Create deployment and capture deployment ID ---
echo "=== creating multi-role deployment ==="
python3 "${CLOUDML_API}" create-multi-role-deployment \
    --workspace-id "${CLOUDML_WORKSPACE_ID}" \
    --service-id "${CLOUDML_SERVICE_ID}" \
    --deployment-name "${CLOUDML_DEPLOYMENT_NAME}" \
    --area "${CLOUDML_AREA}" \
    --queue-name "${CLOUDML_QUEUE_NAME}" \
    --roles-config "@${RENDERED}" \
    --model-repo-config "@${MODEL_REPO_CONFIG}" | tee /tmp/deployment.json
read -r DEPLOYMENT_ID DEPLOYMENT_ENDPOINT < <(python3 -c "import json; d=json.load(open('/tmp/deployment.json')); print(d['deploymentId'], d.get('endpoint',''))")
echo "=== deployment created: ${DEPLOYMENT_ID} endpoint: ${DEPLOYMENT_ENDPOINT} ==="

# --- Step 4: Wait for deployment to be ready ---
echo "=== waiting for deployment to be ready ==="
python3 "${CLOUDML_API}" wait-for-deployment-ready \
    --workspace-id "${CLOUDML_WORKSPACE_ID}" \
    --service-id "${CLOUDML_SERVICE_ID}" \
    --deployment-id "${DEPLOYMENT_ID}" \
    --timeout 1200

echo "=== deployment is ready ==="
# --- Step 5: run the benchmark ---
echo "=== running benchmark ==="
# TODO: run the benchmark

# --- Step 6: delete the deployment ---
echo "=== deleting deployment ==="
python3 "${CLOUDML_API}" delete-deployment \
    --workspace-id "${CLOUDML_WORKSPACE_ID}" \
    --service-id "${CLOUDML_SERVICE_ID}" \
    --deployment-id "${DEPLOYMENT_ID}"
