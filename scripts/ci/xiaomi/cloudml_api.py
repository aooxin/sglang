#!/usr/bin/env python3
"""CloudML Service Deploy API — Programmatic Tool Calling (AK/SK Auth)

High-level commands that encapsulate entire workflows.
The LLM decides WHAT to do; this script handles HOW.

Usage:
  py scripts/cloudml_api.py save-aksk --access-key AK... --secret-key SK... --owner sunzongyu
  py scripts/cloudml_api.py gateways --workspace-id 10001
  py scripts/cloudml_api.py areas --workspace-id 10001
  py scripts/cloudml_api.py resolve-gateway --workspace-id 10001 --keyword "青海"
  py scripts/cloudml_api.py queue-resources --workspace-id 10001 --area chn-zhangjiakou
  py scripts/cloudml_api.py create-service --workspace-id 10001 --service-name test --profile staging --area chn-zhangjiakou
  py scripts/cloudml_api.py create-multi-role-deployment --workspace-id 10001 --service-id s-xxx --deployment-name deploy1 --area chn-zhangjiakou --queue-name myqueue --roles-config '[{"roleName":"Prefill","entryPoint":false,"imageUrl":"img:1.0","imageCommand":"python app.py","resourceInstance":"cloudml.gpu-8","num":4},{"roleName":"Encode","entryPoint":false,"imageUrl":"img:1.0","imageCommand":"python encode.py","resourceInstance":"cloudml.gpu-8","num":2},{"roleName":"Router","entryPoint":true,"routerType":"LLM-Router","imageUrl":"router:1.0","imageCommand":"./manager","resourceInstance":"cloudml.cpu-1","num":2}]'
  py scripts/cloudml_api.py patch-role --workspace-id 10001 --service-id s-xxx --deployment-id s-xxx-yyy --role-name Prefill --image-command "sleep 100"
  py scripts/cloudml_api.py patch-deployment --workspace-id 10001 --service-id s-xxx --deployment-id s-xxx-yyy --changes-config '{"Encode":{"num":3},"Router":{"imageCommand":"./new-manager"}}'
  py scripts/cloudml_api.py deployment-detail --workspace-id 10001 --service-id s-xxx --deployment-id s-xxx-yyy
  py scripts/cloudml_api.py service-detail --workspace-id 10001 --service-id s-xxx
  py scripts/cloudml_api.py update-service --workspace-id 10001 --service-id s-xxx --service-name new-name --gateway-read-timeout 600 --hash-header sessionid --tag-names '["标签A","标签B"]'
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("CLOUDML_BASE_URL", "https://open-api.cloudml.mioffice.cn")
AKSK_FILE = os.path.expanduser("~/.cloudml-cli/cloudml_aksk.json")
VALID_PROFILES = ["staging", "prod", "preview"]
VALID_ROUTER_TYPES = ["LLM-Router", "SGLang-Router"]
BUILTIN_ROLE_TYPES = {"prefill", "decode", "router"}
_aksk_cache = None


def is_error(result):
    return isinstance(result, dict) and ("http_error" in result or "error" in result)


def parse_json_arg(value, field_name):
    """Parse a CLI argument as JSON/YAML. Accepts a JSON string or @filepath (.json/.yaml/.yml)."""
    if value is None:
        return None
    if value.startswith("@"):
        path = value[1:]
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    import yaml
                    return yaml.safe_load(f)
                return json.load(f)
        except Exception as e:
            fail(f"{field_name}: 无法读取文件 {path}: {e}")
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        fail(f"{field_name}: JSON 解析失败: {e}", raw=value)


def out(data):
    """Print JSON output."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def fail(msg, **extra):
    """Print error and exit."""
    out({"error": msg, **extra})
    sys.exit(1)


def load_aksk():
    global _aksk_cache
    if _aksk_cache is not None:
        return _aksk_cache
    if not os.path.exists(AKSK_FILE):
        fail("AK/SK 文件不存在，请先执行 save-aksk", path=AKSK_FILE)
    with open(AKSK_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            fail("AK/SK 文件格式错误，需为合法 JSON", path=AKSK_FILE)
    for key in ("access_key", "secret_key", "owner"):
        if not data.get(key):
            fail(f"AK/SK 文件缺少 '{key}' 字段", path=AKSK_FILE)
    _aksk_cache = data
    return data


def save_aksk(access_key, secret_key, owner):
    """Save AK/SK credentials to config file."""
    data = {"access_key": access_key, "secret_key": secret_key, "owner": owner}
    os.makedirs(os.path.dirname(AKSK_FILE), exist_ok=True)
    # Skip write if unchanged
    if os.path.exists(AKSK_FILE):
        with open(AKSK_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        if existing == data:
            out({"success": True, "path": AKSK_FILE, "unchanged": True, "owner": owner})
            return
    with open(AKSK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    out({"success": True, "path": AKSK_FILE, "owner": owner})


def resolve_owner(args_owner):
    """Return explicit --owner if given, else read from AK/SK config."""
    if args_owner:
        return args_owner
    aksk = load_aksk()
    return aksk["owner"]


def _build_signature(path, body_bytes):
    """Build HMAC-SHA1 signature headers for AK/SK auth.

    Signing algorithm (matches ApiAuthFilter.verifySignature):
      stringToSign = path + "\\n" + timestamp + "\\n" + contentMd5 + "\\n"
      signature = Base64( HMAC-SHA1(secretKey, stringToSign) )
      Authorization: "Galaxy V3 " + signature
    """
    aksk = load_aksk()
    timestamp = str(int(time.time() * 1000))
    content_md5 = hashlib.md5(body_bytes or b"").hexdigest()
    # Sign only the URI path (no query string) — matches request.getRequestURI()
    sign_path = path.split("?", 1)[0]
    string_to_sign = f"{sign_path}\n{timestamp}\n{content_md5}\n"

    sig_bytes = hmac.new(
        aksk["secret_key"].encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.b64encode(sig_bytes).decode("ascii")

    # Extract workspace_id from URL path for Workspace-Id header
    m = re.search(r"/workspaces/(\d+)", path)
    workspace_id = m.group(1) if m else ""

    return {
        "Auth-Type": "key",
        "X-Xiaomi-Secret-Key-Id": aksk["access_key"],
        "X-Xiaomi-Timestamp": timestamp,
        "X-Xiaomi-Content-Md5": content_md5,
        "Authorization": f"Galaxy V3 {signature}",
        "Workspace-Id": workspace_id,
        "Content-Type": "application/json;charset=utf-8",
        "From": "cli",
    }


def api_request(method, path, body=None):
    url = f"{BASE_URL}{path}"
    body_bytes = json.dumps(body).encode("utf-8") if body is not None else None
    headers = _build_signature(path, body_bytes)
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            error_json = json.loads(error_body)
        except json.JSONDecodeError:
            error_json = {"raw": error_body}
        return {"http_error": e.code, "reason": e.reason, "body": error_json}
    except urllib.error.URLError as e:
        return {"error": f"连接失败: {e.reason}"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def fetch_gateways(workspace_id, area=None):
    """Fetch and summarize gateway list."""
    path = f"/api/openapi/v2/workspaces/{workspace_id}/services/gateways"
    if area:
        path += f"?area={urllib.parse.quote(area, safe='')}"
    result = api_request("GET", path)
    if is_error(result):
        return result
    return [
        {
            "area": gw.get("area"),
            "domainName": gw.get("domainName"),
            "domain": gw.get("domain"),
            "gatewayId": gw.get("albId"),
            "gatewayDomain": gw.get("albName"),
            "cluster": gw.get("cluster"),
        }
        for gw in result.get("data", [])
    ]


def fetch_areas(workspace_id):
    """Fetch available areas (area code + Chinese name) for a workspace."""
    path = f"/api/openapi/v2/workspaces/{workspace_id}/areas"
    result = api_request("GET", path)
    if is_error(result):
        return result
    return result.get("data", [])


def _resolve_area_keyword(workspace_id, keyword, gateways):
    """Try to resolve a Chinese area keyword to an area code via the areas API.

    Returns the area code if the keyword matches a chineseName, otherwise
    returns the original keyword unchanged.
    """
    areas = fetch_areas(workspace_id)
    if is_error(areas):
        return keyword  # areas API failed, keep original keyword
    for a in areas:
        if keyword == a.get("chineseName", ""):
            return a["name"]  # exact match: "青海" → "chn-qinghai"
    # Also try substring match: "青海" in "chn-qinghai-xxx"
    for a in areas:
        if keyword in (a.get("chineseName") or ""):
            return a["name"]
    return keyword


def _match_gateways(gateways, keyword):
    kw = keyword.lower()
    return [gw for gw in gateways
            if kw in (gw.get("domainName") or "").lower()
            or kw in (gw.get("area") or "").lower()
            or kw in (gw.get("domain") or "").lower()]


def resolve_gateway_by_keyword(workspace_id, keyword):
    """Find a gateway matching keyword (fuzzy match on domainName/area/domain).

    Supports Chinese area names (e.g. "青海") by resolving them to area codes
    via the areas API when direct matching fails.
    """
    gateways = fetch_gateways(workspace_id)
    if is_error(gateways):
        return gateways
    matches = _match_gateways(gateways, keyword)
    if not matches:
        resolved = _resolve_area_keyword(workspace_id, keyword, gateways)
        if resolved != keyword:
            matches = _match_gateways(gateways, resolved)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return {"ambiguous": True, "matches": matches, "hint": "请用更精确的关键词"}
    else:
        return {"error": f"未找到匹配 '{keyword}' 的网关", "available": [gw["domainName"] for gw in gateways]}


def fetch_queue_resources(workspace_id, area, resolve_area=True):
    """Fetch and summarize queues + resource specs for an area."""
    if resolve_area:
        area = _resolve_area_keyword(workspace_id, area, [])
    encoded_area = urllib.parse.quote(area, safe='')
    path = f"/api/openapi/v2/workspaces/{workspace_id}/queueInfos?filterType=in&withResourceUseDetails=true&area={encoded_area}&availableScope=MODEL_SERVICE"
    result = api_request("GET", path)
    if is_error(result):
        return result
    summary = []
    for q in result.get("data", []):
        resources = []
        for rd in q.get("resourceUseDetails", []):
            d = rd.get("resourceDetail", {})
            resources.append({
                "resourceInstance": d.get("resourceInstance"),
                "resourcePriority": d.get("resourcePriority"),
                "type": d.get("resourceType"),
                "gpuType": d.get("gpuType") or None,
                "cpu": d.get("cpuLimit"),
                "mem": d.get("memoryLimit"),
                "gpu": d.get("gpuLimit"),
            })
        summary.append({
            "queueId": q.get("queueId"),
            "name": q.get("name"),
            "region": q.get("region"),
            "cluster": q.get("cluster"),
            "resources": resources,
        })
    return summary


def resolve_resource_priority(queue_resources, resource_instance):
    """Determine the best resourcePriority for a given resourceInstance in a queue.

    Priority preference: GUARANTEED > GUARANTEED_PUBLIC > others (e.g. BEST_EFFORT).
    Returns the priority string, or "GUARANTEED" as a safe default.
    """
    priorities = [
        r["resourcePriority"] for r in queue_resources
        if r.get("resourceInstance") == resource_instance and r.get("resourcePriority")
    ]
    if not priorities:
        return "GUARANTEED"
    # Prefer GUARANTEED, then GUARANTEED_PUBLIC, then first available
    for preferred in ["GUARANTEED", "GUARANTEED_PUBLIC"]:
        if preferred in priorities:
            return preferred
    return priorities[0]


def fetch_deployment_detail(workspace_id, service_id, deployment_id, enable_instances=False):
    """Fetch deployment detail and return structured summary."""
    path = f"/api/openapi/v2/workspaces/{workspace_id}/services/{service_id}/deployments/{deployment_id}"
    if enable_instances:
        path += "?enableInstances=true"
    result = api_request("GET", path)
    if is_error(result):
        return result
    data = result.get("data", {})
    dc = data.get("deploymentConfig", {})
    gw = data.get("gatewayConfig", {})
    msb = data.get("modelServiceBase", {})
    roles = []
    for role in data.get("roleInfoList", []):
        rdc = role.get("deploymentConfig", {})
        role_info = {
            "deploymentId": role.get("deploymentId"),
            "roleId": role.get("roleId"),
            "roleName": role.get("roleName"),
            "roleType": rdc.get("roleType"),
            "entryPoint": rdc.get("entryPoint", False),
            "routerType": rdc.get("routerType"),
            "num": rdc.get("num"),
            "imageUrl": role.get("imageConfig", {}).get("imageUrl"),
            "imageCommand": role.get("imageConfig", {}).get("imageCommand"),
            "modelFrame": rdc.get("modelFrame"),
            "resourceInstance": (role.get("resourceConfigs") or [{}])[0].get("resourceInstance"),
            "trafficPorts": role.get("trafficPorts", []),
            "envConfigs": role.get("envConfigs", []),
            "healthCheck": role.get("healthCheck"),
            "externalStorageMountConfigs": role.get("externalStorageMountConfigs", []),
        }
        if enable_instances and role.get("instances"):
            role_info["instances"] = role["instances"]
        roles.append(role_info)
    model_repo = data.get("modelRepoConfig") or dc.get("modelRepoConfig")
    return {
        "serviceId": dc.get("serviceId"),
        "deployGroupId": dc.get("deploymentId"),
        "deploymentName": dc.get("deploymentName"),
        "state": dc.get("state"),
        "areaCode": dc.get("areaCode"),
        "profile": msb.get("profile"),
        "queueId": dc.get("queueId"),
        "cluster": dc.get("cluster"),
        "gatewayConfig": gw,
        "modelRepoConfig": model_repo if model_repo else None,
        "roles": roles,
        "trafficPorts": dc.get("trafficPorts", []),
        "_raw": data,  # Keep raw for update body construction
    }


def fetch_service_detail(workspace_id, service_id):
    """Fetch service detail (includes deployments, queue info, etc.)."""
    path = f"/api/openapi/v2/workspaces/{workspace_id}/services/{service_id}"
    result = api_request("GET", path)
    if is_error(result):
        return result
    data = result.get("data")
    if not data:
        return {"error": f"服务 {service_id} 不存在"}
    return data


# ---------------------------------------------------------------------------
# Body builders — encapsulate the massive JSON templates
# ---------------------------------------------------------------------------

def build_service_body(gateway, **kwargs):
    """Build create-service request body."""
    gw_mode = kwargs.get("gateway_mode", "wrr")
    gw_read_timeout = kwargs.get("gateway_read_timeout", 300)

    body = {
        "serviceName": kwargs["service_name"],
        "owner": kwargs["owner"],
        "profile": kwargs["profile"],
        "areaCode": gateway["area"],
        "gatewayConfig": {
            "loadbalancerLayer": "APPLICATION",
            "gatewayDomain": gateway["gatewayDomain"],
            "domain": gateway["domain"],
            "mode": gw_mode,
            "readTimeout": gw_read_timeout,
            "gatewayId": gateway["gatewayId"],
        },
        "trafficPorts": kwargs.get("traffic_ports", [{"port": 80, "protocol": "HTTP", "path": "/"}]),
    }

    # Optional fields — only include if explicitly provided
    if kwargs.get("description"):
        body["description"] = kwargs["description"]
    if kwargs.get("custom_domain"):
        body["customDomain"] = kwargs["custom_domain"]
    if kwargs.get("hash_header"):
        body["gatewayConfig"]["header"] = kwargs["hash_header"]
    if kwargs.get("tag_names"):
        body["tagNames"] = kwargs["tag_names"]

    return body


def _build_role_from_config(role_config, queue_info, gateway, deployment_name, owner, area, num_per_instance=1):
    """Build a single role entry from a role config dict.

    Required keys: roleName, imageUrl, imageCommand, resourceInstance, num
    Optional keys: entryPoint (bool), routerType, roleType, modelFrame,
                   graceShutdownTime, preStopCommand, weight, description,
                   spreadLimit, rollingUpdate, envs, healthCheck, storage,
                   resourcePriority
    """
    role_name = role_config["roleName"]
    # roleType: backend only accepts "prefill", "decode", "router", "default"
    # Custom role names (e.g. Encode) must use "default" as roleType
    default_role_type = role_name.lower() if role_name.lower() in BUILTIN_ROLE_TYPES else "default"
    role_type = role_config.get("roleType", default_role_type)
    entry_point = role_config.get("entryPoint", False)
    router_type = role_config.get("routerType")
    num = role_config["num"]
    model_frame = role_config.get("modelFrame", "other")
    grace_shutdown = role_config.get("graceShutdownTime", 300)
    description = role_config.get("description", "")
    weight = role_config.get("weight")
    spread_limit = role_config.get("spreadLimit")
    rolling_update = role_config.get("rollingUpdate", {"maxSurge": 25, "maxUnavailable": 1})
    resource_instance = role_config["resourceInstance"]
    resource_priority = role_config.get("resourcePriority", "GUARANTEED")
    image_url = role_config["imageUrl"]
    image_command = role_config["imageCommand"]
    pre_stop_command = role_config.get("preStopCommand", "")
    envs = role_config.get("envs", [])
    health_check = role_config.get("healthCheck")
    storage = role_config.get("storage", [])

    dc = {
        "deploymentName": deployment_name,
        "owner": owner,
        "areaCode": area,
        "cluster": queue_info["cluster"],
        "queueId": queue_info["queueId"],
        "modelFrame": model_frame,
        "creator": owner,
        "num": num,
        "graceShutdownTime": grace_shutdown,
        "rollingUpdate": rolling_update,
        "roleName": role_name,
        "entryPoint": entry_point,
        "deploymentGroupModel": "multiRole",
        "roleType": role_type,
    }
    if router_type:
        dc["routerType"] = router_type
    if description:
        dc["description"] = description
    if weight is not None:
        dc["weight"] = weight
    if spread_limit is not None:
        dc["spreadLimit"] = spread_limit

    is_preemptible = resource_priority != "GUARANTEED"
    role_entry = {
        "roleName": role_name,
        "deploymentConfig": dc,
        "envConfigs": envs,
        "externalStorageMountConfigs": storage,
        "resourceConfigs": [{
            "num": num,
            "resourcePriority": resource_priority,
            "resourceInstance": resource_instance,
            "numPerInstance": num_per_instance,
            "resource": {
                "resourcePriority": resource_priority,
                "resourceInstance": resource_instance,
                "numPerInstance": num_per_instance,
            },
            "numPerReplica": 1,
            "elastic": False,
            "maxNum": 0,
            "minNum": 0,
            "preemptible": is_preemptible,
        }],
        "imageConfig": {
            "imageType": "CUSTOM",
            "imageUrl": image_url,
            "imageSecretName": "system-created-global-image-secret",
            "imageCommand": image_command,
            "preStopCommand": pre_stop_command,
        },
        "gatewayConfig": {
            "gatewayId": gateway["gatewayId"],
            "gatewayDomain": gateway["gatewayDomain"],
            "loadbalancerLayer": "APPLICATION",
            "mode": "wrr",
            "enabled": True,
            "domain": gateway["domain"],
            "readTimeout": 300,
        },
        "trafficPorts": [{"port": 80, "path": "/", "type": 1, "protocol": "HTTP"}],
    }
    if health_check:
        role_entry["healthCheck"] = health_check
    return role_entry


def build_deployment_body(gateway, queue_info, roles_config, **kwargs):
    """Build create-multi-role-deployment request body.

    roles_config: list of role config dicts.  Each must have at minimum:
        roleName, imageUrl, imageCommand, resourceInstance, num.
    Exactly one role should have entryPoint=True.
    """
    entry_point_role = next(
        (r["roleName"] for r in roles_config if r.get("entryPoint")),
        roles_config[-1]["roleName"],
    )
    dc = {
        "deploymentName": kwargs["deployment_name"],
        "owner": kwargs["owner"],
        "areaCode": kwargs["area"],
        "deploymentGroupModel": "multiRole",
        "queueId": queue_info["queueId"],
        "entryPointRoleName": entry_point_role,
    }
    model_repo_config = kwargs.get("model_repo_config")
    if model_repo_config:
        dc["modelRepoConfig"] = model_repo_config
    return {
        "deploymentConfig": dc,
        "gatewayConfig": {
            "loadbalancerLayer": "APPLICATION",
            "gatewayDomain": gateway["gatewayDomain"],
            "domain": gateway["domain"],
            "mode": "wrr",
        },
        "queueRegion": queue_info["region"],
        "queueCluster": queue_info["cluster"],
        "roleInfoList": [
            _build_role_from_config(
                rc, queue_info, gateway,
                deployment_name=kwargs["deployment_name"],
                owner=kwargs["owner"],
                area=kwargs["area"],
                num_per_instance=kwargs.get("num_per_instance", 1),
            )
            for rc in roles_config
        ],
    }


def build_update_body_from_detail(detail, changes):
    """Build update body by merging changes into existing deployment detail.

    `detail` is the output of fetch_deployment_detail().
    `changes` is a dict like: {"Prefill": {"imageCommand": "sleep 100"}, "Decode": {"num": 2}}
    """
    raw = detail["_raw"]
    gw = raw.get("gatewayConfig", {})
    dc = raw.get("deploymentConfig", {})
    deploy_group_id = dc.get("deploymentId")

    role_info_list = []
    for role_raw in raw.get("roleInfoList", []):
        role_name = role_raw.get("roleName")
        role_changes = changes.get(role_name, {})
        role_dc = dict(role_raw.get("deploymentConfig", {}))
        role_id = role_raw.get("roleId")

        # Apply imageConfig changes
        image_keys = {"imageCommand", "imageUrl", "preStopCommand"}
        image_changes = {k: role_changes[k] for k in image_keys if k in role_changes}
        if image_changes:
            ic = dict(role_raw.get("imageConfig", {}))
            ic.update(image_changes)
            role_raw = dict(role_raw)
            role_raw["imageConfig"] = ic
        if "num" in role_changes:
            role_dc["num"] = role_changes["num"]
            rcs = role_raw.get("resourceConfigs", [{}])
            if rcs:
                rcs = [dict(rcs[0])]
                rcs[0]["num"] = role_changes["num"]
                role_raw = dict(role_raw)
                role_raw["resourceConfigs"] = rcs
        if "resourceInstance" in role_changes:
            rcs = role_raw.get("resourceConfigs", [{}])
            if rcs:
                rcs = [dict(rcs[0])]
                rcs[0]["resourceInstance"] = role_changes["resourceInstance"]
                if "resource" in rcs[0]:
                    rcs[0]["resource"] = dict(rcs[0]["resource"])
                    rcs[0]["resource"]["resourceInstance"] = role_changes["resourceInstance"]
                role_raw = dict(role_raw)
                role_raw["resourceConfigs"] = rcs
        # DeploymentConfig simple fields
        for key in ["modelFrame", "graceShutdownTime", "weight", "description", "spreadLimit"]:
            if key in role_changes:
                role_dc[key] = role_changes[key]
        # Full-replace fields (envs, healthCheck, storage)
        if "envConfigs" in role_changes:
            role_raw = dict(role_raw)
            role_raw["envConfigs"] = role_changes["envConfigs"]
        if "healthCheck" in role_changes:
            role_raw = dict(role_raw)
            role_raw["healthCheck"] = role_changes["healthCheck"]
        if "externalStorageMountConfigs" in role_changes:
            role_raw = dict(role_raw)
            role_raw["externalStorageMountConfigs"] = role_changes["externalStorageMountConfigs"]

        # Strip runtime-only fields from deploymentConfig
        for key in ["state", "stateCname", "reason", "createTime", "updateTime",
                     "deploymentDomain", "weight", "enableAutoscale", "lvsWeight",
                     "podMuted", "accessibleInstanceCount", "teamId", "orgId",
                     "queueName", "queueDeleted"]:
            role_dc.pop(key, None)

        role_entry = {
            "roleName": role_name,
            "deploymentId": deploy_group_id,
            "roleId": role_id,
            "deploymentConfig": {**role_dc, "deploymentId": role_dc.get("deploymentId", role_id), "roleId": role_id},
            "envConfigs": role_raw.get("envConfigs", []),
            "externalStorageMountConfigs": role_raw.get("externalStorageMountConfigs", []),
            "resourceConfigs": role_raw.get("resourceConfigs", []),
            "imageConfig": role_raw.get("imageConfig", {}),
            "gatewayConfig": role_raw.get("gatewayConfig", gw),
            "healthCheck": role_raw.get("healthCheck", {}),
            "trafficPorts": role_raw.get("trafficPorts", []),
        }
        role_info_list.append(role_entry)

    return {
        "deploymentConfig": {
            "deploymentId": deploy_group_id,
            "deploymentName": dc.get("deploymentName"),
            "owner": dc.get("owner"),
            "areaCode": dc.get("areaCode"),
            "deploymentGroupModel": "multiRole",
            "queueId": dc.get("queueId"),
            "entryPointRoleName": dc.get("entryPointRoleName", "Router"),
        },
        "gatewayConfig": gw,
        "queueRegion": dc.get("areaCode"),
        "queueCluster": dc.get("cluster"),
        "roleInfoList": role_info_list,
    }


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_gateways(args):
    gateways = fetch_gateways(args.workspace_id, args.area)
    out(gateways)


def cmd_areas(args):
    areas = fetch_areas(args.workspace_id)
    out(areas)


def cmd_resolve_gateway(args):
    result = resolve_gateway_by_keyword(args.workspace_id, args.keyword)
    out(result)


def cmd_queue_resources(args):
    result = fetch_queue_resources(args.workspace_id, args.area)
    out(result)


def _deployment_to_roles_yaml(result):
    """Convert deployment-detail result to roles-config YAML format."""
    try:
        import yaml
    except ImportError:
        fail("PyYAML is required: pip install pyyaml")
    roles_config = []
    for role in result.get("roles", []):
        rc = {"roleName": role["roleName"]}
        rc["entryPoint"] = role.get("entryPoint", False)
        if role.get("routerType"):
            rc["routerType"] = role["routerType"]
        rc["imageUrl"] = role.get("imageUrl", "")
        rc["imageCommand"] = (role.get("imageCommand") or "").rstrip()
        rc["resourceInstance"] = role.get("resourceInstance", "")
        rc["num"] = role.get("num", 1)
        if role.get("modelFrame") and role["modelFrame"] != "other":
            rc["modelFrame"] = role["modelFrame"]
        else:
            rc["modelFrame"] = "other"
        envs = role.get("envConfigs", [])
        if envs:
            rc["envs"] = [{"name": e["name"], "value": e["value"]} for e in envs
                          if "name" in e and "value" in e]
        hc = role.get("healthCheck")
        if hc:
            rc["healthCheck"] = hc
        storage = role.get("externalStorageMountConfigs", [])
        if storage:
            rc["storage"] = storage
        roles_config.append(rc)

    class _YamlDumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        if data.lower() in ("true", "false", "null", "yes", "no"):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _YamlDumper.add_representer(str, _str_representer)
    return yaml.dump(roles_config, Dumper=_YamlDumper,
                     default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_deployment_endpoint(service_domain, service_id, deployment_id, roles):
    """Build deployment endpoint: replace service_id in serviceDomain with deployment_id, add entryPoint port."""
    if not service_domain or not deployment_id:
        return ""
    domain = service_domain.replace(service_id, deployment_id, 1)
    port = ""
    for r in roles:
        if r.get("entryPoint"):
            tp = r.get("trafficPorts") or []
            if tp:
                port = str(tp[0].get("port", ""))
            break
    if port:
        return f"http://{domain}:{port}"
    return f"http://{domain}"


def _fetch_deployment_endpoint(workspace_id, service_id, deployment_id, roles):
    svc = fetch_service_detail(workspace_id, service_id)
    service_domain = svc.get("serviceDomain", "") if not is_error(svc) else ""
    return _build_deployment_endpoint(service_domain, service_id, deployment_id, roles)


def cmd_deployment_detail(args):
    enable_instances = args.enable_instances
    result = fetch_deployment_detail(args.workspace_id, args.service_id, args.deployment_id, enable_instances)
    if is_error(result):
        fail("查询部署详情失败", detail=result)

    endpoint = _fetch_deployment_endpoint(args.workspace_id, args.service_id, args.deployment_id, result.get("roles", []))
    if endpoint:
        print(f"# endpoint: {endpoint}")
    model_repo = result.get("modelRepoConfig")
    if model_repo:
        print(f"# modelRepoConfig: {json.dumps(model_repo, ensure_ascii=False)}")
    print(_deployment_to_roles_yaml(result))


READY_STATE = "running"
WAIT_STATES = {"enqueued", "deploying", "updating", "creating"}


def cmd_wait_for_deployment_ready(args):
    """Poll deployment state until READY_STATE, fail on unexpected states."""
    deployment_id = args.deployment_id
    timeout = args.timeout
    interval = 30
    deadline = time.time() + timeout
    print(f"Waiting for deployment {deployment_id} to be ready (timeout={timeout}s) ...")
    while True:
        result = fetch_deployment_detail(args.workspace_id, args.service_id, deployment_id)
        if is_error(result):
            fail("查询部署详情失败", detail=result)
        state = result.get("state", "")
        print(f"  state: {state}")
        if state == READY_STATE:
            print("Deployment is ready.")
            return
        if state not in WAIT_STATES:
            fail(f"部署进入异常状态: {state}，停止等待")
        if time.time() >= deadline:
            fail(f"部署在 {timeout}s 内未就绪，当前状态: {state}")
        remaining = max(0, deadline - time.time())
        time.sleep(min(interval, remaining))


def cmd_delete_deployment(args):
    """Delete a deployment by ID."""
    deployment_id = args.deployment_id
    print(f"Deleting deployment {deployment_id} ...")
    result = api_request(
        "DELETE",
        f"/api/openapi/v2/workspaces/{args.workspace_id}/services/{args.service_id}/deployments/{deployment_id}",
    )
    if is_error(result):
        fail("删除部署失败", detail=result)
    out({"deleted": deployment_id})


def cmd_create_service(args):
    profile = args.profile
    if profile not in VALID_PROFILES:
        fail(f"profile 必须是 {VALID_PROFILES} 之一，收到: '{profile}'")

    owner = resolve_owner(args.owner)

    # Resolve gateway
    gateway = resolve_gateway_by_keyword(args.workspace_id, args.area)
    if isinstance(gateway, dict) and (is_error(gateway) or "ambiguous" in gateway):
        out({"error": "无法自动匹配网关，请手动指定", "detail": gateway})
        sys.exit(1)

    # Parse optional JSON args
    traffic_ports = parse_json_arg(args.traffic_ports, "--traffic-ports")
    tag_names = parse_json_arg(args.tag_names, "--tag-names")

    build_kwargs = dict(
        service_name=args.service_name,
        owner=owner,
        profile=profile,
    )
    # Optional simple fields
    if args.description is not None:
        build_kwargs["description"] = args.description
    if args.custom_domain is not None:
        build_kwargs["custom_domain"] = args.custom_domain
    if args.gateway_read_timeout is not None:
        build_kwargs["gateway_read_timeout"] = args.gateway_read_timeout
    # Gateway mode + hash header
    if args.hash_header:
        build_kwargs["gateway_mode"] = "hash"
        build_kwargs["hash_header"] = args.hash_header
        if args.gateway_mode and args.gateway_mode != "hash":
            fail("--hash-header 与 --gateway-mode wrr 冲突，hash-header 仅在 hash 模式下有效")
    elif args.gateway_mode == "hash":
        fail("--gateway-mode hash 需要指定 --hash-header，如 --hash-header sessionid")
    elif args.gateway_mode:
        build_kwargs["gateway_mode"] = args.gateway_mode
    # Optional JSON fields
    if traffic_ports is not None:
        build_kwargs["traffic_ports"] = traffic_ports
    if tag_names is not None:
        build_kwargs["tag_names"] = tag_names

    body = build_service_body(gateway, **build_kwargs)
    result = api_request("POST", f"/api/openapi/v2/workspaces/{args.workspace_id}/services", body)
    out({"result": result, "gateway_used": gateway, "body_sent": body, "owner": owner})


def cmd_create_multi_role_deployment(args):
    owner = resolve_owner(args.owner)

    # Resolve gateway
    gateway = resolve_gateway_by_keyword(args.workspace_id, args.area or args.service_id)
    if isinstance(gateway, dict) and (is_error(gateway) or "ambiguous" in gateway):
        fail("无法解析网关，请通过 --area 指定区域关键词", detail=gateway)

    # Resolve queue
    queues = fetch_queue_resources(args.workspace_id, gateway["area"], resolve_area=False)
    if is_error(queues):
        fail("查询队列失败", detail=queues)
    if not queues:
        fail(f"区域 {gateway['area']} 下没有可用队列")

    queue_info = None
    kw = args.queue_name.lower()
    for q in queues:
        if kw in (q.get("name") or "").lower():
            queue_info = q
            break
    if not queue_info:
        fail(f"未找到名称匹配 '{args.queue_name}' 的队列",
             available=[q["name"] for q in queues])

    q_resources = queue_info.get("resources", [])
    q_instances = {r.get("resourceInstance") for r in q_resources}

    # Parse --roles-config (required)
    roles_config = parse_json_arg(args.roles_config, "--roles-config")
    if not isinstance(roles_config, list) or not roles_config:
        fail("--roles-config 必须是非空 JSON 数组")
    REQUIRED_FIELDS = ["roleName", "imageUrl", "imageCommand", "resourceInstance", "num"]
    for i, rc in enumerate(roles_config):
        for field in REQUIRED_FIELDS:
            if field not in rc or rc[field] is None:
                fail(f"角色 #{i+1} 缺少必填字段 '{field}'", role=rc.get("roleName", f"#{i+1}"))
        ri = rc["resourceInstance"]
        if ri not in q_instances:
            fail(f"队列 '{queue_info['name']}' 不包含角色 '{rc['roleName']}' 的资源规格 '{ri}'",
                 available_in_queue=sorted(q_instances))
        if "resourcePriority" not in rc:
            rc["resourcePriority"] = resolve_resource_priority(q_resources, ri)
        if rc.get("routerType") and rc["routerType"] not in VALID_ROUTER_TYPES:
            fail(f"角色 '{rc['roleName']}' 的 routerType 必须是 {VALID_ROUTER_TYPES} 之一")
    if not any(rc.get("entryPoint") for rc in roles_config):
        fail("roles-config 中至少需要一个角色设置 entryPoint: true")

    model_repo_config = parse_json_arg(args.model_repo_config, "--model-repo-config")
    body = build_deployment_body(
        gateway, queue_info, roles_config,
        deployment_name=args.deployment_name,
        owner=owner,
        area=gateway["area"],
        num_per_instance=args.num_per_instance,
        model_repo_config=model_repo_config,
    )
    result = api_request("POST", f"/api/openapi/v2/workspaces/{args.workspace_id}/services/{args.service_id}/deployments", body)
    if is_error(result):
        fail("创建部署失败", detail=result)

    deployment_id = result.get("data")
    # Summarize
    role_summary = {}
    for rc in roles_config:
        name = rc["roleName"]
        summary = {
            "image_url": rc.get("imageUrl"),
            "image_command": rc.get("imageCommand"),
            "resource_instance": rc.get("resourceInstance"),
            "num": rc.get("num"),
            "entry_point": rc.get("entryPoint", False),
        }
        if rc.get("routerType"):
            summary["router_type"] = rc["routerType"]
        if rc.get("envs"):
            summary["envs"] = "provided"
        if rc.get("healthCheck"):
            summary["health_check"] = "provided"
        if rc.get("storage"):
            summary["storage"] = "provided"
        role_summary[name] = summary
    endpoint = _fetch_deployment_endpoint(args.workspace_id, args.service_id, deployment_id, roles_config)
    out({"deploymentId": deployment_id, "endpoint": endpoint,
         "gateway": gateway["domainName"], "queue": queue_info["name"], "roles": role_summary})


def cmd_patch_role(args):
    """Query existing deployment, apply changes to specified role, send update."""
    detail = fetch_deployment_detail(args.workspace_id, args.service_id, args.deployment_id)
    if is_error(detail):
        fail("查询部署详情失败", detail=detail)

    # Build changes dict
    changes = {}
    role_changes = {}
    if args.image_command is not None:
        role_changes["imageCommand"] = args.image_command
    if args.image_url is not None:
        role_changes["imageUrl"] = args.image_url
    if args.num is not None:
        role_changes["num"] = args.num
    if args.resource_instance is not None:
        role_changes["resourceInstance"] = args.resource_instance
    if args.model_frame is not None:
        role_changes["modelFrame"] = args.model_frame
    if args.grace_shutdown_time is not None:
        role_changes["graceShutdownTime"] = args.grace_shutdown_time
    if args.pre_stop_command is not None:
        role_changes["preStopCommand"] = args.pre_stop_command
    if args.weight is not None:
        role_changes["weight"] = args.weight
    if args.description is not None:
        role_changes["description"] = args.description
    if args.spread_limit is not None:
        role_changes["spreadLimit"] = args.spread_limit
    if args.envs is not None:
        role_changes["envConfigs"] = parse_json_arg(args.envs, "--envs")
    if args.health_check is not None:
        role_changes["healthCheck"] = parse_json_arg(args.health_check, "--health-check")
    if args.storage is not None:
        role_changes["externalStorageMountConfigs"] = parse_json_arg(args.storage, "--storage")

    if not role_changes:
        fail("未指定任何变更字段")

    # Verify role exists
    role_names = [r["roleName"] for r in detail.get("roles", [])]
    if args.role_name not in role_names:
        fail(f"角色 '{args.role_name}' 不存在", available=role_names)

    changes[args.role_name] = role_changes

    body = build_update_body_from_detail(detail, changes)
    result = api_request(
        "PUT",
        f"/api/openapi/v2/workspaces/{args.workspace_id}/services/{args.service_id}/deployments/{args.deployment_id}",
        body,
    )
    out({"result": result, "changes": changes})


def cmd_service_detail(args):
    """Fetch service detail including deployments."""
    data = fetch_service_detail(args.workspace_id, args.service_id)
    if is_error(data):
        fail("查询服务详情失败", detail=data)
    # Summarize key fields
    gw = data.get("gatewayConfig", {}) or {}
    summary = {
        "serviceId": data.get("serviceId"),
        "serviceName": data.get("serviceName"),
        "description": data.get("description"),
        "owner": data.get("owner"),
        "state": data.get("state"),
        "profile": data.get("profile"),
        "areaCode": data.get("areaCode"),
        "serviceDomain": data.get("serviceDomain"),
        "customDomain": data.get("customDomain"),
        "gatewayConfig": gw,
        "trafficPorts": data.get("trafficPorts", []),
        "customTags": data.get("customTags", []),
        "deploymentNum": data.get("deploymentNum", 0),
        "createTime": data.get("createTime"),
        "updateTime": data.get("updateTime"),
    }
    deployments = data.get("deployments", []) or []
    if deployments:
        summary["deployments"] = [
            {
                "deploymentId": d.get("deploymentId"),
                "deploymentName": d.get("deploymentName"),
                "state": d.get("state"),
                "queueName": d.get("queueName"),
            }
            for d in deployments
        ]
    out(summary)


def cmd_update_service(args):
    """Fetch current service, merge user changes, send PUT update."""
    # Step 1: Fetch current service detail to get existing gatewayConfig etc.
    data = fetch_service_detail(args.workspace_id, args.service_id)
    if is_error(data):
        fail("查询服务详情失败", detail=data)

    existing_gw = data.get("gatewayConfig", {}) or {}

    # Step 2: Build update body by merging changes on top of existing state
    body = {
        "serviceName": data.get("serviceName"),
        "owner": data.get("owner"),
        "description": data.get("description", ""),
        "gatewayConfig": existing_gw,
        "trafficPorts": data.get("trafficPorts", []),
    }

    # Apply user-specified changes
    changes = {}
    if args.service_name is not None:
        body["serviceName"] = args.service_name
        changes["serviceName"] = args.service_name
    if args.owner is not None:
        body["owner"] = args.owner
        changes["owner"] = args.owner
    if args.description is not None:
        body["description"] = args.description
        changes["description"] = args.description
    if args.gateway_read_timeout is not None:
        body["gatewayConfig"]["readTimeout"] = args.gateway_read_timeout
        changes["gatewayConfig.readTimeout"] = args.gateway_read_timeout
    # Gateway mode + hash header
    if args.hash_header is not None:
        if args.gateway_mode and args.gateway_mode != "hash":
            fail("--hash-header 与 --gateway-mode wrr 冲突")
        body["gatewayConfig"]["mode"] = "hash"
        body["gatewayConfig"]["header"] = args.hash_header
        changes["gatewayConfig.mode"] = "hash"
        changes["gatewayConfig.header"] = args.hash_header
    elif args.gateway_mode == "hash":
        fail("--gateway-mode hash 需要指定 --hash-header，如 --hash-header sessionid")
    elif args.gateway_mode is not None:
        body["gatewayConfig"]["mode"] = args.gateway_mode
        changes["gatewayConfig.mode"] = args.gateway_mode
        if args.gateway_mode == "wrr":
            body["gatewayConfig"].pop("header", None)

    traffic_ports = parse_json_arg(args.traffic_ports, "--traffic-ports")
    if traffic_ports is not None:
        body["trafficPorts"] = traffic_ports
        changes["trafficPorts"] = traffic_ports
    tag_names = parse_json_arg(args.tag_names, "--tag-names")
    if tag_names is not None:
        body["tagNames"] = tag_names
        changes["tagNames"] = tag_names

    if not changes:
        fail("未指定任何变更字段")

    # Step 3: Send update
    result = api_request(
        "PUT",
        f"/api/openapi/v2/workspaces/{args.workspace_id}/services/{args.service_id}",
        body,
    )
    out({"result": result, "changes": changes})


def cmd_patch_deployment(args):
    """Query existing deployment, apply per-role changes via --changes-config, send single update."""
    detail = fetch_deployment_detail(args.workspace_id, args.service_id, args.deployment_id)
    if is_error(detail):
        fail("查询部署详情失败", detail=detail)

    role_names = [r["roleName"] for r in detail.get("roles", [])]

    changes = parse_json_arg(args.changes_config, "--changes-config")
    if not isinstance(changes, dict):
        fail("--changes-config 必须是 JSON 对象，如 {\"Prefill\": {\"imageCommand\": \"...\"}}")
    for role_name in changes:
        if role_name not in role_names:
            fail(f"角色 '{role_name}' 不存在于当前部署", available=role_names)
    if not changes:
        fail("--changes-config 为空，未指定任何变更")

    body = build_update_body_from_detail(detail, changes)
    result = api_request(
        "PUT",
        f"/api/openapi/v2/workspaces/{args.workspace_id}/services/{args.service_id}/deployments/{args.deployment_id}",
        body,
    )
    out({"result": result, "changes": changes})


def main():
    parser = argparse.ArgumentParser(description="CloudML Service Deploy API")
    sub = parser.add_subparsers(dest="command")

    # save-aksk
    p = sub.add_parser("save-aksk")
    p.add_argument("--access-key", required=True, help="个人 Access Key ID")
    p.add_argument("--secret-key", required=True, help="个人 Secret Key")
    p.add_argument("--owner", required=True, help="用户名（用于 request body 中的 owner 字段）")

    # gateways
    p = sub.add_parser("gateways")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--area", default=None, help="按区域过滤（可选）")

    # areas
    p = sub.add_parser("areas")
    p.add_argument("--workspace-id", required=True)

    # resolve-gateway
    p = sub.add_parser("resolve-gateway")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--keyword", required=True, help="网关关键词（如 '张家口阿里' / '青海' / 'chn-qinghai'）")

    # queue-resources
    p = sub.add_parser("queue-resources")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--area", required=True)

    # deployment-detail
    p = sub.add_parser("deployment-detail")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-id", required=True)
    p.add_argument("--enable-instances", action="store_true", default=False, help="是否返回实例列表")

    # wait-for-deployment-ready
    p = sub.add_parser("wait-for-deployment-ready")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-id", required=True, help="部署 ID")
    p.add_argument("--timeout", type=int, default=1200, help="超时时间秒（默认 1200）")

    # delete-deployment
    p = sub.add_parser("delete-deployment")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-id", required=True, help="部署 ID")

    # service-detail
    p = sub.add_parser("service-detail")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)

    # update-service
    p = sub.add_parser("update-service")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--service-name", default=None, help="服务名称")
    p.add_argument("--owner", default=None, help="负责人")
    p.add_argument("--description", default=None, help="服务描述")
    p.add_argument("--gateway-read-timeout", type=int, default=None, help="网关超时时间秒")
    p.add_argument("--gateway-mode", default=None, help="网关负载均衡模式")
    p.add_argument("--traffic-ports", default=None,
                   help='流量端口 JSON，如 \'[{"port":80,"protocol":"HTTP","path":"/"}]\'，支持 @filepath（.json/.yaml/.yml）')
    p.add_argument("--hash-header", default=None,
                   help='一致性哈希的 HTTP Header 名称（如 sessionid），设置后自动切换网关模式为 hash')
    p.add_argument("--tag-names", default=None,
                   help='标签名称列表 JSON，如 \'["标签A","标签B"]\'，支持 @filepath（.json/.yaml/.yml）')

    # create-service (HIGH-LEVEL: resolves gateway internally)
    p = sub.add_parser("create-service")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-name", required=True)
    p.add_argument("--owner", default=None, help="服务负责人（不填则从 AK/SK 配置读取）")
    p.add_argument("--profile", required=True, choices=VALID_PROFILES)
    p.add_argument("--area", required=True, help="区域关键词，自动匹配网关")
    # Optional fields
    p.add_argument("--description", default=None, help="服务描述")
    p.add_argument("--custom-domain", default=None, help="自定义域名")
    p.add_argument("--gateway-mode", default=None, help="网关负载均衡模式（默认 wrr）")
    p.add_argument("--gateway-read-timeout", type=int, default=None, help="网关超时时间秒（默认 300）")
    p.add_argument("--traffic-ports", default=None,
                   help='流量端口 JSON，如 \'[{"port":80,"protocol":"HTTP","path":"/"}]\'，支持 @filepath（.json/.yaml/.yml）')
    p.add_argument("--hash-header", default=None,
                   help='一致性哈希的 HTTP Header 名称（如 sessionid），设置后自动切换网关模式为 hash')
    p.add_argument("--tag-names", default=None,
                   help='标签名称列表 JSON，如 \'["标签A","标签B"]\'，支持 @filepath（.json/.yaml/.yml）')

    # create-multi-role-deployment (HIGH-LEVEL: resolves gateway + queue internally)
    p = sub.add_parser("create-multi-role-deployment")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-name", required=True)
    p.add_argument("--owner", default=None, help="服务负责人（不填则从 AK/SK 配置读取）")
    p.add_argument("--area", required=True, help="区域关键词")
    p.add_argument("--num-per-instance", type=int, default=1)
    p.add_argument("--queue-name", required=True, help="指定队列名称（模糊匹配）")
    p.add_argument("--roles-config", required=True,
                   help='角色配置 JSON 数组或 @filepath（.json/.yaml/.yml）。每个角色: '
                        '{"roleName":"Encode","entryPoint":false,"imageUrl":"...","imageCommand":"...",'
                        '"resourceInstance":"...","num":2,...}')
    p.add_argument("--model-repo-config", default=None,
                   help='模型仓库配置 JSON 或 @filepath，如 '
                        '{"modelType":"custom","modelUri":"AI_AIE/Mimo-V2-Flash/1","modelName":"Mimo-V2-Flash"}')

    # patch-role (HIGH-LEVEL: single role change)
    p = sub.add_parser("patch-role")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-id", required=True)
    p.add_argument("--role-name", required=True, help="角色名: Prefill / Decode / Router")
    p.add_argument("--image-command", default=None)
    p.add_argument("--image-url", default=None)
    p.add_argument("--num", type=int, default=None)
    p.add_argument("--resource-instance", default=None)
    p.add_argument("--model-frame", default=None, help="模型框架 (SGLang/vLLM/other)")
    p.add_argument("--grace-shutdown-time", type=int, default=None, help="优雅停机时间秒")
    p.add_argument("--pre-stop-command", default=None, help="停机前钩子命令")
    p.add_argument("--weight", type=int, default=None, help="权重")
    p.add_argument("--description", default=None, help="部署描述")
    p.add_argument("--spread-limit", type=int, default=None, help="容灾打散上限")
    p.add_argument("--envs", default=None, help='环境变量 JSON/YAML: [{"name":"K","value":"V"}]，支持 @filepath')
    p.add_argument("--health-check", default=None, help="健康检查 JSON/YAML，支持 @filepath")
    p.add_argument("--storage", default=None, help="存储挂载 JSON/YAML，支持 @filepath")

    # patch-deployment (HIGH-LEVEL: multi-role changes in one call)
    p = sub.add_parser("patch-deployment")
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--service-id", required=True)
    p.add_argument("--deployment-id", required=True)
    p.add_argument("--changes-config", required=True,
                   help='变更配置 JSON 对象或 @filepath（.json/.yaml/.yml），key 为角色名，value 为变更字段。'
                        '如 \'{"Encode":{"num":3},"Router":{"imageCommand":"..."}}\'')


    args = parser.parse_args()
    commands = {
        "save-aksk": lambda: save_aksk(args.access_key, args.secret_key, args.owner),
        "gateways": lambda: cmd_gateways(args),
        "areas": lambda: cmd_areas(args),
        "resolve-gateway": lambda: cmd_resolve_gateway(args),
        "queue-resources": lambda: cmd_queue_resources(args),
        "deployment-detail": lambda: cmd_deployment_detail(args),
        "wait-for-deployment-ready": lambda: cmd_wait_for_deployment_ready(args),
        "delete-deployment": lambda: cmd_delete_deployment(args),
        "service-detail": lambda: cmd_service_detail(args),
        "update-service": lambda: cmd_update_service(args),
        "create-service": lambda: cmd_create_service(args),
        "create-multi-role-deployment": lambda: cmd_create_multi_role_deployment(args),
        "patch-role": lambda: cmd_patch_role(args),
        "patch-deployment": lambda: cmd_patch_deployment(args),
    }
    fn = commands.get(args.command)
    if fn:
        fn()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
