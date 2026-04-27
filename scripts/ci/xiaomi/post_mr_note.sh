#!/usr/bin/env bash
# Post a markdown file as a GitLab MR note.
# Usage: post_mr_note.sh <markdown_file>
# Requires CI variables: GITLAB_API_TOKEN, CI_API_V4_URL, CI_PROJECT_ID, CI_MERGE_REQUEST_IID
set -euo pipefail

NOTE_FILE="${1:?Usage: post_mr_note.sh <markdown_file>}"

if [ -z "${GITLAB_API_TOKEN:-}" ]; then
    echo "GITLAB_API_TOKEN not set; skipping MR note."
    exit 0
fi
if [ -z "${CI_MERGE_REQUEST_IID:-}" ]; then
    echo "CI_MERGE_REQUEST_IID not set; skipping MR note."
    exit 0
fi
if [ ! -s "${NOTE_FILE}" ]; then
    echo "Note file '${NOTE_FILE}' is empty or missing; skipping MR note."
    exit 0
fi

python3 -c 'import json,sys; print(json.dumps({"body": sys.stdin.read()}))' \
    < "${NOTE_FILE}" > /tmp/mr_note_payload.json

API_URL="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/notes"
HTTP_CODE=$(curl -sS -o /tmp/mr_note_resp.json -w "%{http_code}" \
    --header "PRIVATE-TOKEN: ${GITLAB_API_TOKEN}" \
    --header "Content-Type: application/json" \
    --data @/tmp/mr_note_payload.json \
    "$API_URL")
echo "POST $API_URL -> HTTP $HTTP_CODE"

if [ "${HTTP_CODE:-000}" -ge 400 ]; then
    echo "Failed to post MR note (HTTP $HTTP_CODE):"
    cat /tmp/mr_note_resp.json || true
    echo "Note content was:"
    cat "${NOTE_FILE}"
    exit 1
fi
