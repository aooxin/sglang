#!/usr/bin/env bash
# Validate MR title format in GitLab CI.
# Requires CI_MERGE_REQUEST_TITLE (set automatically in MR pipelines).
#
# Accepted formats:
#   [Category] description        e.g. [Fix] Resolve OOM on large batches
#   type: description             e.g. fix: resolve OOM on large batches
#   type(scope): description      e.g. fix(runtime): resolve OOM on large batches
#
# Also allows an optional "Draft: " prefix.

set -euo pipefail

TITLE="${CI_MERGE_REQUEST_TITLE:-}"

if [ -z "$TITLE" ]; then
  echo "CI_MERGE_REQUEST_TITLE is not set — skipping MR title check."
  exit 0
fi

echo "MR title: ${TITLE}"

# Strip optional Draft prefix for validation
clean_title="${TITLE#Draft: }"

# Pattern 1: [Category] description  (bracket prefix)
BRACKET_RE='^\[.+\] .+'

# Pattern 2: conventional commits — type(scope): desc  or  type: desc
CONVENTIONAL_RE='^(feat|fix|docs|chore|refactor|test|style|ci|perf|build|revert|model|diffusion)(\(.+\))?: .+'

if [[ "$clean_title" =~ $BRACKET_RE ]] || [[ "$clean_title" =~ $CONVENTIONAL_RE ]]; then
  echo "MR title format OK."
  exit 0
fi

echo ""
echo "ERROR: MR title does not match the required format."
echo ""
echo "Accepted formats:"
echo "  [Category] description        e.g. [Fix] Resolve OOM on large batches"
echo "  type: description             e.g. fix: resolve OOM on large batches"
echo "  type(scope): description      e.g. fix(runtime): resolve OOM on large batches"
echo ""
echo "Conventional commit types: feat, fix, docs, chore, refactor, test, style, ci, perf, build, revert, model, diffusion"
echo ""
echo "Please update your MR title and re-run the pipeline."
exit 1
