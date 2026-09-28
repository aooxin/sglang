"""GCache timing and byte counters, shared across backend attach/detach."""

import threading
from weakref import WeakKeyDictionary

_collectors = WeakKeyDictionary()
_collector_lock = threading.Lock()
_SECONDS = (0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30)


class GCacheMetrics:
    def __init__(self, tp_rank: int, dp_rank: int):
        from prometheus_client import REGISTRY, Counter, Histogram

        with _collector_lock:
            collectors = _collectors.get(REGISTRY)
            if collectors is None:
                collectors = {}
                labels = ("storage_backend", "tp_rank", "dp_rank")
                for direction in ("prefetch", "backup"):
                    prefix = f"sglang:gcache_{direction}"
                    collectors[direction, "bytes"] = Counter(
                        f"{prefix}_bytes_total",
                        "Bytes successfully transferred by the SDK, excluding existing writes.",
                        labels,
                        registry=REGISTRY,
                    )
                    for stage in ("preprocess", "rdma", "postprocess", "idle"):
                        collectors[direction, stage] = Histogram(
                            f"{prefix}_{stage}_seconds",
                            {
                                "preprocess": "Time building and validating host page metadata.",
                                "rdma": "Time in GCache SDK calls, including network and server work.",
                                "postprocess": "Time folding SDK object results into page results.",
                                "idle": "Gap between consecutive backend calls in this direction.",
                            }[stage],
                            labels,
                            buckets=_SECONDS,
                            registry=REGISTRY,
                        )
                    collectors[direction, "bandwidth"] = Histogram(
                        f"{prefix}_rdma_bandwidth",
                        "Successful bytes / SDK call duration, in GiB/s.",
                        labels,
                        buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320),
                        registry=REGISTRY,
                    )
                    collectors[direction, "pages"] = Histogram(
                        f"{prefix}_batch_keys",
                        "Logical pages per pool per batch (two K/V objects per page).",
                        labels,
                        buckets=(1, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                        registry=REGISTRY,
                    )
                _collectors[REGISTRY] = collectors
        self._metrics = {
            key: collector.labels("gcache", str(tp_rank), str(dp_rank))
            for key, collector in collectors.items()
        }
        self._lock = threading.Lock()
        self._last_end = {}

    def observe(self, *, is_set, pages, transferred_bytes, times):
        started, prepared, transferred, finished = times
        direction = "backup" if is_set else "prefetch"
        sdk_seconds = max(transferred - prepared, 0)
        self._metrics[direction, "bytes"].inc(transferred_bytes)
        self._metrics[direction, "pages"].observe(pages)
        self._metrics[direction, "preprocess"].observe(max(prepared - started, 0))
        self._metrics[direction, "rdma"].observe(sdk_seconds)
        self._metrics[direction, "postprocess"].observe(max(finished - transferred, 0))
        self._metrics[direction, "bandwidth"].observe(
            transferred_bytes / (1 << 30) / max(sdk_seconds, 1e-9)
        )
        with self._lock:
            previous = self._last_end.get(direction)
            if previous is not None and started >= previous:
                self._metrics[direction, "idle"].observe(started - previous)
            self._last_end[direction] = max(previous or 0, finished)
