#!/bin/bash
# Usage: ./hello.sh [base_url]
#   base_url: OpenAI API server address (default: http://localhost:9001)
#   Sends a simple chat completion request to verify the service is healthy.

BASE_URL="${1:-http://localhost:9001}"

curl -f -X POST "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model": "base_model",
          "messages": [{"role": "user",
                        "content": "hello"}],
          "max_tokens": 1024,
          "stream": false}'
