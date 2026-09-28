"""An out-of-process, in-memory KV server for the GCache backend tests.

The test transport is HTTP/JSON, not GCache's native byte-RPC/RDMA protocol.
NetworkGCacheClient implements the SDK's pointer/size API but sends actual
bytes over a socket; it never owns the KV dictionary. No production fallback
or dependency on the native SDK is introduced.

Run manually: python python/sglang/test/gcache_mock_server.py --port 18080
"""

import argparse
import base64
import ctypes
import json
import os
import select
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _encode(value):
    return base64.b64encode(value).decode("ascii")


def _decode(value):
    return base64.b64decode(value, validate=True)


def _request(address, operation, payload, timeout=5):
    request = urllib.request.Request(
        f"http://{address}/{operation}",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    # Loopback traffic must not depend on the developer machine's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = json.loads(exc.read()).get("error", str(exc))
        raise RuntimeError(detail) from exc


class _KVServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, _Handler)
        self.lock = threading.Lock()
        self.objects = {}
        self.calls = []
        self.registrations = {}
        self.faults = {}
        self.gates = {}

    def control(self, request):
        with self.lock:
            action = request["action"]
            group = request.get("group", "test")
            if action == "snapshot":
                return {
                    "pid": os.getpid(),
                    "objects": {
                        key: _encode(value)
                        for (stored_group, key), value in self.objects.items()
                        if stored_group == group
                    },
                    "calls": list(self.calls),
                    "registrations": dict(self.registrations),
                }
            if action == "configure":
                op = request["operation"]
                self.faults[op] = request["fault"]
                gate = self.gates.setdefault(op, threading.Event())
                if request["fault"].get("hold"):
                    gate.clear()
                else:
                    gate.set()
            elif action == "release":
                self.gates[request["operation"]].set()
            elif action == "clear_calls":
                self.calls.clear()
            elif action == "put":
                for key, value in request["objects"].items():
                    self.objects[group, key] = _decode(value)
            elif action == "delete":
                for key in request["keys"]:
                    self.objects.pop((group, key), None)
            else:
                raise ValueError(f"Unknown mockserver control action: {action}")
            return {}

    def execute(self, operation, request):
        with self.lock:
            fault = dict(self.faults.get(operation, {}))
            gate = self.gates.get(operation)
            if operation in ("exists", "get", "set"):
                self.calls.append(
                    {
                        "operation": operation,
                        "keys": request["keys"],
                        "sizes": request["sizes"],
                        "for_read": request.get("for_read"),
                        "client": request["client"],
                    }
                )
        if gate is not None and not gate.wait(timeout=10):
            raise TimeoutError("Mockserver held request timed out")
        if fault.get("delay"):
            time.sleep(fault["delay"])
        if fault.get("error"):
            raise RuntimeError(fault["error"])

        with self.lock:
            if operation == "init":
                return {"pid": os.getpid()}
            if operation == "register":
                regions = self.registrations.setdefault(request["client"], [])
                region = [request["ptr"], request["size"]]
                if region in regions:
                    raise ValueError("Duplicate SDK registration")
                # Fault injection can fail the second physical allocation.
                if fault.get("fail_after") == len(regions):
                    raise RuntimeError("Injected registration failure")
                regions.append(region)
                return {"result": fault.get("result")}
            if operation == "close":
                return {}

            keys, sizes = request["keys"], request["sizes"]
            if len(keys) != len(sizes):
                raise ValueError("Mismatched object and size lists")
            group = request["group"]
            failed = set(fault.get("fail_keys", []))
            results, values = [], []
            for i, (key, size) in enumerate(zip(keys, sizes)):
                value = self.objects.get((group, key))
                if operation == "set":
                    value = _decode(request["values"][i])
                    ok = key not in failed and len(value) == size
                    if ok:
                        self.objects[group, key] = value
                elif operation in ("get", "exists"):
                    ok = key not in failed and value is not None and len(value) == size
                else:
                    raise ValueError(f"Unknown mockserver operation: {operation}")
                results.append(ok)
                if operation == "get":
                    values.append(_encode(value) if ok else None)
            return {"result": fault.get("result", results), "values": values}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            operation = self.path.removeprefix("/")
            result = (
                self.server.control(request)
                if operation == "control"
                else self.server.execute(operation, request)
            )
            self._respond(200, result)
        except Exception as exc:
            self._respond(503, {"error": str(exc)})

    def _respond(self, status, payload):
        data = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when a client deliberately times out.
            pass


class GCacheMockServer:
    """Start/stop a real server process and control faults through its HTTP API."""

    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            if not select.select([self.process.stdout], [], [], 10)[0]:
                raise TimeoutError("GCache mockserver did not start")
            message = json.loads(self.process.stdout.readline())
            self.address = message["address"]
            self.pid = message["pid"]
        except BaseException:
            self.close()
            raise

    def control(self, action, **kwargs):
        return _request(self.address, "control", {"action": action, **kwargs})

    def snapshot(self):
        snapshot = self.control("snapshot")
        snapshot["objects"] = {
            key: _decode(value) for key, value in snapshot["objects"].items()
        }
        return snapshot

    @property
    def objects(self):
        return self.snapshot()["objects"]

    @property
    def calls(self):
        return self.snapshot()["calls"]

    def configure(self, operation, **fault):
        self.control("configure", operation=operation, fault=fault)

    def put(self, objects):
        self.control(
            "put", objects={key: _encode(value) for key, value in objects.items()}
        )

    def delete(self, *keys):
        self.control("delete", keys=keys)

    def clear_calls(self):
        self.control("clear_calls")

    def wait_for_call(self, operation, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(call["operation"] == operation for call in self.calls):
                return
            time.sleep(0.01)
        raise TimeoutError(f"No {operation} request reached the mockserver")

    def release(self, operation):
        self.control("release", operation=operation)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NetworkGCacheClient:
    """SDK-shaped client using real network I/O and registered CPU addresses."""

    def __init__(self, *, conf_file, timeout_us=0):
        self.options = dict(conf_file=conf_file, timeout_us=timeout_us)
        flags = {}
        for line in Path(conf_file).read_text().splitlines():
            if line.startswith("--") and "=" in line:
                key, value = line[2:].split("=", 1)
                flags[key] = value
        self.address = flags["master_addr"]
        self.group = flags["cache_group"]
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout_us / 1e6 if timeout_us else 5
        self.registrations = []
        self.closed = False
        self._call("init")

    def _call(self, endpoint, **payload):
        if self.closed:
            raise RuntimeError("Client is closed")
        return _request(
            self.address,
            endpoint,
            {"client": self.client_id, "group": self.group, **payload},
            timeout=self.timeout,
        )

    def register_memory(self, ptr, size):
        if ptr <= 0 or size <= 0:
            raise ValueError("Invalid registration range")
        result = self._call("register", ptr=ptr, size=size)["result"]
        self.registrations.append((ptr, size))
        return result

    def _check_ranges(self, keys, ptrs, sizes):
        if not len(keys) == len(ptrs) == len(sizes):
            raise ValueError("Mismatched object, pointer and size lists")
        for ptr, size in zip(ptrs, sizes):
            if size <= 0 or not any(
                base <= ptr and ptr + size <= base + length
                for base, length in self.registrations
            ):
                raise ValueError("SDK I/O outside registered memory")

    def batch_exists(self, keys, for_read=True, size_list=None):
        return self._call("exists", keys=keys, sizes=size_list, for_read=for_read)[
            "result"
        ]

    def batch_set(self, keys, ptrs, sizes):
        self._check_ranges(keys, ptrs, sizes)
        values = [
            _encode(ctypes.string_at(ptr, size)) for ptr, size in zip(ptrs, sizes)
        ]
        return self._call("set", keys=keys, sizes=sizes, values=values)["result"]

    def batch_get(self, keys, ptrs, sizes):
        self._check_ranges(keys, ptrs, sizes)
        response = self._call("get", keys=keys, sizes=sizes)
        # Malformed status is intentionally left to the backend to reject. Only
        # complete payloads may touch a destination, even on a faulty response.
        for ptr, size, value in zip(ptrs, sizes, response["values"]):
            if value is not None:
                data = _decode(value)
                if len(data) != size:
                    raise ValueError("Short mockserver read")
                ctypes.memmove(ptr, data, size)
        return response["result"]

    def close(self):
        self._call("close")
        self.closed = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    with _KVServer(("127.0.0.1", args.port)) as server:
        address = "{}:{}".format(*server.server_address)
        print(json.dumps({"address": address, "pid": os.getpid()}), flush=True)
        server.serve_forever(poll_interval=0.05)
