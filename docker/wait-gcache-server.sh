#!/usr/bin/env bash
set -euo pipefail

: "${GCACHE_MASTER_ADDR:?Missing GCACHE_MASTER_ADDR}"
: "${GCACHE_CACHE_GROUP:?Missing GCACHE_CACHE_GROUP}"

# Only wait for master discovery. RDMA readiness is checked by the SDK at init.
# Use the image's Python so this helper needs neither curl nor jq.
exec python3 - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request

master = os.environ["GCACHE_MASTER_ADDR"].strip()
group = os.environ["GCACHE_CACHE_GROUP"].strip()
interval = float(os.environ.get("GCACHE_WAIT_INTERVAL", "10"))
if not master or not group or interval <= 0:
    raise SystemExit("GCache master/group must be non-empty and wait interval positive")
url = f"http://{master}/groupcache/getPeers?" + urllib.parse.urlencode({"group": group})
print("Waiting for gcache-server discovery...", flush=True)
while True:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            peers = json.load(response).get("data", {}).get("Peers", [])
        if any(peer.get("addr") for peer in peers):
            print("gcache-server registered with master", flush=True)
            break
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        print(f"GCache discovery unavailable: {exc}", flush=True)
    time.sleep(interval)
PY
