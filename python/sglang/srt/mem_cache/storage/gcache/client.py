"""Own the pinned SDK singleton and its registrations for the process lifetime."""

import atexit
import os
import threading
from weakref import WeakKeyDictionary

import torch

from sglang.srt.mem_cache.storage.gcache.config import GCacheClientConfig

_client_lock = threading.Lock()
_runtimes: WeakKeyDictionary = WeakKeyDictionary()


class GCacheClientRuntime:
    def __init__(self, client, config_path, signature):
        self.client = client
        self.config_path = config_path
        self.signature = signature
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._registered: set[tuple[int, int]] = set()
        # SDK 2.1.0+g5978d0b2 has no unregister_memory. Native registrations
        # outlive backend detach, so the allocations must do so as well.
        self._allocations: dict[tuple[int, int], torch.UntypedStorage] = {}

    def register(self, allocations: dict[tuple[int, int], torch.UntypedStorage]):
        with self._lock:
            for region, storage in allocations.items():
                if region in self._registered:
                    continue
                # Keep even a failed registration's allocation alive: the SDK
                # may have made progress before reporting a native error.
                self._allocations[region] = storage
                result = self.client.register_memory(*region)
                if result is not None and not (type(result) is int and result == 0):
                    raise RuntimeError(f"GCache register_memory failed: {result}")
                self._registered.add(region)


def get_client_runtime(
    config: GCacheClientConfig, dp_rank: int, tp_rank: int
) -> GCacheClientRuntime:
    try:
        from gcache_sdk import GCacheClient
    except ImportError as exc:
        raise ImportError(
            "The gcache backend requires the optional gcache-sdk package; "
            "see mem_cache/storage/gcache/README.md for the supported wheel."
        ) from exc

    contents = config.render()
    signature = (contents, config.timeout_us)
    with _client_lock:
        runtime = _runtimes.get(GCacheClient)
        if runtime is not None:
            if runtime.pid != os.getpid():
                raise RuntimeError(
                    "GCache must be initialized after worker processes are created"
                )
            if runtime.signature != signature:
                raise ValueError(
                    "GCache connection settings cannot change in the same process; "
                    "restart the worker to change the SDK configuration"
                )
            return runtime

        config_path = config.write_flags(dp_rank, tp_rank, contents=contents)
        try:
            client = GCacheClient(
                conf_file=str(config_path), timeout_us=config.timeout_us
            )
        except BaseException:
            config_path.unlink(missing_ok=True)
            raise
        runtime = GCacheClientRuntime(client, config_path, signature)
        _runtimes[GCacheClient] = runtime
        # One flags file per client, reused across attach/detach.
        atexit.register(config_path.unlink, missing_ok=True)
        return runtime
