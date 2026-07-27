#!/usr/bin/env bash
# Source this script before starting SGLang to discover another running peer
# pod in the same deployment and export REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP.
#
# Usage:
#   REMOTE_INSTANCE_SEED_FORCE=exact-service.ns-xxx.svc.cluster.local      # optional forced FQDN
#   REMOTE_INSTANCE_SEED_FALLBACK=other-service.ns-xxx.svc.cluster.local  # optional fallback FQDN
#   source scripts/resolve_remote_instance_seed_ip.sh prefill  # or: decode
#   remote_seed_args=()
#   if [ -n "${REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP:-}" ]; then
#     remote_seed_args=(--remote-instance-weight-loader-seed-instance-ip "$REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP")
#   fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "[resolve-remote-seed] this script must be sourced, not executed" >&2
  echo "[resolve-remote-seed] usage: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

log() {
  echo "[resolve-remote-seed] $*" >&2
}

is_ipv4() {
  case "$1" in
    *[!0-9.]*|""|.*|*.) return 1 ;;
  esac
  local old_ifs=$IFS
  IFS=.
  # shellcheck disable=SC2086
  set -- $1
  IFS=$old_ifs
  [ "$#" -eq 4 ] || return 1
  local octet
  for octet in "$@"; do
    [ -n "$octet" ] || return 1
    [ "$octet" -ge 0 ] 2>/dev/null || return 1
    [ "$octet" -le 255 ] 2>/dev/null || return 1
  done
  return 0
}

REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP=""
export REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP

if ! command -v getent >/dev/null 2>&1; then
  log "getent is not available; leave seed IP empty for local fallback"
  return 0
fi

role="${1:-${SGLANG_REMOTE_INSTANCE_SEED_ROLE:-}}"
if [ -z "$role" ]; then
  if [ -n "${REMOTE_INSTANCE_SEED_FORCE:-}" ]; then
    role="force"
  else
    log "role is not set; use 'source ${BASH_SOURCE[0]} prefill' or 'decode'. leave seed IP empty for local fallback"
    return 0
  fi
fi
case "$role" in
  force)
    service_name=""
    ;;
  prefill)
    service_name="${CLOUDML_MULTI_ROLE_SERVICE_PREFILL:-}"
    ;;
  decode)
    service_name="${CLOUDML_MULTI_ROLE_SERVICE_DECODE:-}"
    ;;
  *)
    log "unknown SGLANG_REMOTE_INSTANCE_SEED_ROLE=${role}; leave seed IP empty for local fallback"
    return 0
    ;;
esac

if [ "$role" != "force" ] && { [ -z "$service_name" ] || [ -z "${ORG_ID:-}" ]; }; then
  log "service env for role=${role} or ORG_ID is missing; leave seed IP empty for local fallback"
  return 0
fi

if [ "$role" != "force" ]; then
  service_fqdn="${service_name}.ns-${ORG_ID}.svc.cluster.local"
else
  service_fqdn=""
fi
current_pod_ip="${POD_IP:-${MY_POD_IP:-}}"
retries="${SGLANG_REMOTE_INSTANCE_SEED_RETRIES:-10}"
sleep_sec="${SGLANG_REMOTE_INSTANCE_SEED_SLEEP:-2}"
strategy="${SGLANG_REMOTE_INSTANCE_SEED_STRATEGY:-random}"
case "$strategy" in
  first|random) ;;
  *)
    log "unknown SGLANG_REMOTE_INSTANCE_SEED_STRATEGY=${strategy}; use random"
    strategy="random"
    ;;
esac

if [ -n "$current_pod_ip" ]; then
  log "excluding current pod IP ${current_pod_ip}"
else
  log "POD_IP/MY_POD_IP is not set; cannot exclude current pod"
fi

select_seed_from_fqdn() {
  local target_fqdn="$1"
  local label="$2"
  local allow_retries="$3"
  local max_attempts=1
  local attempt=1
  local ip
  local selected_index

  if [ "$allow_retries" = "true" ]; then
    max_attempts="$retries"
  fi

  log "resolving ${target_fqdn} (${label}, role=${role})"
  while [ "$attempt" -le "$max_attempts" ]; do
    local candidate_ips=()
    while IFS= read -r line; do
      ip="$(printf '%s\n' "$line" | awk '{print $1}')"
      if ! is_ipv4 "$ip"; then
        continue
      fi
      if [ -n "$current_pod_ip" ] && [ "$ip" = "$current_pod_ip" ]; then
        continue
      fi
      candidate_ips+=("$ip")
    done < <(getent hosts "$target_fqdn" 2>/dev/null || true)

    if [ "${#candidate_ips[@]}" -gt 0 ]; then
      if [ "$strategy" = "random" ]; then
        selected_index=$((RANDOM % ${#candidate_ips[@]}))
      else
        selected_index=0
      fi
      REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP="${candidate_ips[$selected_index]}"
      export REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP
      log "selected seed IP ${REMOTE_INSTANCE_WEIGHT_LOADER_SEED_INSTANCE_IP} from ${label} (strategy=${strategy}, candidates=${#candidate_ips[@]})"
      return 0
    fi

    if [ "$attempt" -lt "$max_attempts" ]; then
      log "no seed found from ${label} on attempt ${attempt}/${max_attempts}; retrying in ${sleep_sec}s"
      sleep "$sleep_sec"
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

if [ -n "${REMOTE_INSTANCE_SEED_FORCE:-}" ]; then
  log "REMOTE_INSTANCE_SEED_FORCE is set; only trying ${REMOTE_INSTANCE_SEED_FORCE}"
  if select_seed_from_fqdn "$REMOTE_INSTANCE_SEED_FORCE" "forced-deployment" "false"; then
    return 0
  fi
  log "forced seed target failed; leave seed IP empty for local fallback"
  return 0
fi

if select_seed_from_fqdn "$service_fqdn" "current-deployment" "true"; then
  return 0
fi

if [ -n "${REMOTE_INSTANCE_SEED_FALLBACK:-}" ]; then
  log "no seed found in current deployment; trying REMOTE_INSTANCE_SEED_FALLBACK=${REMOTE_INSTANCE_SEED_FALLBACK}"
  if select_seed_from_fqdn "$REMOTE_INSTANCE_SEED_FALLBACK" "fallback-deployment" "false"; then
    return 0
  fi
fi

log "no seed found; leave seed IP empty for local fallback"
return 0
