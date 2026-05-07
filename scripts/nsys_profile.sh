#!/bin/bash
# Usage: ./nsys_profile.sh [duration]
#   duration: seconds to profile (default: 10)
#   Lists active nsys sessions, prompts for a session ID, then captures a profile.

DURATION="${1:-10}"

nsys sessions list
read -p "Enter session ID: " SESSION_ID
[[ -z "$SESSION_ID" ]] && { echo "No session ID provided"; exit 1; }
nsys start --session "$SESSION_ID" || { echo "Failed to start nsys session"; exit 1; }
echo "Profiling session '$SESSION_ID' started. Capturing for ${DURATION}s..."
sleep "$DURATION"
nsys stop --session "$SESSION_ID"
echo ""
echo "Profile files are saved in the current working directory by default:"
ls -al ./*.nsys-rep 2>/dev/null || echo "  No .nsys-rep files found in $(pwd)"
