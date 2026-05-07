#!/bin/bash
# Usage: ./torch_profiler.sh [base_url] [duration]
#   base_url: sglang server address (default: http://localhost:9001)
#   duration: seconds to profile (default: 10)

BASE_URL="${1:-http://localhost:9001}"
DURATION="${2:-10}"

curl -f -X POST "${BASE_URL}/start_profile" -H "Content-Type: application/json" || { echo "Failed to start profile"; exit 1; }
echo "Profiling started. Capturing for ${DURATION}s..."
sleep "$DURATION"
curl -f -X POST "${BASE_URL}/stop_profile" -H "Content-Type: application/json" || { echo "Failed to stop profile"; exit 1; }
echo ""
echo "Profile files are saved under /tmp by default:"
ls -al /tmp/*.trace.json.gz 2>/dev/null || echo "  No trace files found in /tmp"
