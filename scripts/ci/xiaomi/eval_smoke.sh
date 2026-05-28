#!/usr/bin/env bash
# Common smoke-test helper for eval runners. Source this file, do not execute.
#
# Provides a single function `eval_smoke_test` that POSTs one chat-completion
# against a deployed endpoint and exits non-zero if the response lacks a
# "content" field. It exists to close the brief window where K8s
# readiness/startup probes say a pod is Ready but the full
# Router → Prefill → Decode chain isn't actually returning valid completions
# yet (or where the endpoint silently rejects sglang-specific kwargs the eval
# is about to use heavily, e.g. chat_template_kwargs).
#
# Usage:
#   source eval_smoke.sh
#   eval_smoke_test <chat_completions_url> <model_name> [extra_kwargs_json]
#
# Args:
#   <chat_completions_url>  full URL ending in /chat/completions
#   <model_name>            value to put in the request body's "model" field
#   [extra_kwargs_json]     optional fragment to merge into the JSON body, e.g.
#                           '"chat_template_kwargs":{"enable_thinking":true},
#                            "temperature":0.3'
#                           (no leading or trailing comma — caller passes only
#                            the additional fields)
#
# Exit codes:
#   0  smoke succeeded (response had a "content" field)
#   1  HTTP request failed OR response had no "content" field

eval_smoke_test() {
    local url="$1"
    local model="$2"
    local extra_kwargs="${3:-}"

    if [[ -z "${url}" || -z "${model}" ]]; then
        echo "ERROR: eval_smoke_test requires <url> <model> [extra_kwargs_json]" >&2
        return 1
    fi

    echo "=== smoke test: POST ${url} ==="

    # Build request body. We inline-construct JSON; if extra_kwargs is empty we
    # need to skip the leading comma to keep the JSON valid.
    local body
    if [[ -n "${extra_kwargs}" ]]; then
        body="{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],${extra_kwargs}}"
    else
        body="{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
    fi

    local smoke_resp
    smoke_resp="$(mktemp /tmp/eval-smoke.XXXXXX.json)"
    if ! curl -fsS "${url}" \
            -H "Content-Type: application/json" \
            -d "${body}" \
            | tee "${smoke_resp}"; then
        echo ""
        echo "ERROR: smoke test request failed" >&2
        rm -f "${smoke_resp}"
        return 1
    fi
    echo ""
    if ! grep -q '"content"' "${smoke_resp}"; then
        echo "ERROR: smoke response has no 'content' field" >&2
        rm -f "${smoke_resp}"
        return 1
    fi
    rm -f "${smoke_resp}"
    echo "=== smoke test passed ==="
}
