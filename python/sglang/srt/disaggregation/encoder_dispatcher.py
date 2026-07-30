import asyncio
import hashlib
import logging
import time
from bisect import bisect
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_SA_NS_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

_HEX = frozenset("0123456789abcdefABCDEF")


def _key_to_pos(key: str) -> int:
    if len(key) == 40 and all(c in _HEX for c in key):
        return int(key[:16], 16)
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")


class HashRing:
    def __init__(self, endpoints: Iterable[str], vnodes: int = 160):
        unique = list(dict.fromkeys(endpoints))
        ring = sorted(
            (
                int.from_bytes(hashlib.sha1(f"{ep}#{i}".encode()).digest()[:8], "big"),
                ep,
            )
            for ep in unique
            for i in range(vnodes)
        )
        self._positions = [pos for pos, _ in ring]
        self._endpoints = [ep for _, ep in ring]
        self._unique = sorted(unique)
        self._vnodes = vnodes

    @property
    def endpoints(self) -> List[str]:
        return list(self._unique)

    def get(self, key: str, exclude: Optional[Set[str]] = None) -> Optional[str]:
        n = len(self._positions)
        if n == 0:
            return None
        pos = _key_to_pos(key)
        idx = bisect(self._positions, pos) % n
        for _ in range(n):
            ep = self._endpoints[idx]
            if not exclude or ep not in exclude:
                return ep
            idx = (idx + 1) % n
        return None


class EndpointResolver:
    def __init__(
        self,
        service_url: str,
        vnodes: int = 160,
        watch_timeout: int = 180,
        backoff_max: float = 30.0,
    ):
        self._svc, self._ns, self._port = self._parse_url(service_url)
        self._vnodes = vnodes
        self._watch_timeout = watch_timeout
        self._backoff_max = backoff_max

        self._ready_ips: Set[str] = set()
        self._last_urls: List[str] = []
        self._ring: Optional[HashRing] = None
        self._last_event_at: float = 0.0

        self._stop = asyncio.Event()
        self._initial_ready = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @staticmethod
    def _parse_url(url: str) -> Tuple[str, str, int]:
        u = urlparse(url)
        host = u.hostname or ""
        parts = host.split(".")
        svc = parts[0]
        if len(parts) >= 3 and parts[2] == "svc":
            ns = parts[1]
        else:
            # Outside k8s the SA namespace file does not exist. Constructing
            # the dispatcher must never crash the server: leave ns empty and
            # let start() disable discovery (pick() falls back to the VIP).
            try:
                with open(_SA_NS_FILE) as f:
                    ns = f.read().strip()
            except OSError:
                logger.warning(
                    "cannot determine k8s namespace for %s (no %s); encoder "
                    "endpoint discovery will be disabled, falling back to VIP",
                    url,
                    _SA_NS_FILE,
                )
                ns = ""
        port = u.port or 80
        return svc, ns, port

    @property
    def ring(self) -> Optional[HashRing]:
        return self._ring

    @property
    def last_event_at(self) -> float:
        return self._last_event_at

    async def start(self) -> None:
        # Any failure here (unknown namespace, missing kubernetes_asyncio, no
        # kubeconfig, running outside a cluster) must not propagate into
        # request handling: leave the ring empty so pick() falls back to VIP.
        if not self._ns:
            return
        try:
            from kubernetes_asyncio import config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                await config.load_kube_config()
        except Exception as e:
            logger.warning(
                "k8s config load failed (%s: %s); encoder endpoint discovery "
                "disabled, falling back to VIP",
                type(e).__name__,
                e,
            )
            return

        self._task = asyncio.create_task(self._watch_loop())
        try:
            await asyncio.wait_for(self._initial_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                "endpoint resolver initial sync timed out (10s), "
                "proceeding with empty ring — will fallback to VIP"
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch_loop(self) -> None:
        from kubernetes_asyncio import client
        from kubernetes_asyncio import watch as kwatch

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with client.ApiClient() as api:
                    v1 = client.CoreV1Api(api)
                    w = kwatch.Watch()
                    async for event in w.stream(
                        v1.list_namespaced_endpoints,
                        namespace=self._ns,
                        field_selector=f"metadata.name={self._svc}",
                        timeout_seconds=self._watch_timeout,
                    ):
                        self._apply_event(event["type"], event["object"])
                        self._last_event_at = time.monotonic()
                        backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Empty TimeoutError = K8s watch timeout_seconds reached, normal close.
                expected_close = isinstance(e, asyncio.TimeoutError) and not str(e)
                if expected_close:
                    logger.debug(
                        "endpoints watch for %s/%s reached timeout, reconnecting",
                        self._ns,
                        self._svc,
                    )
                else:
                    logger.warning(
                        "endpoints watch for %s/%s disconnected: %s: %r; retry in %.1fs",
                        self._ns,
                        self._svc,
                        type(e).__name__,
                        str(e),
                        backoff,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._backoff_max)

    def _apply_event(self, event_type: str, ep_obj) -> None:
        if event_type == "DELETED":
            self._ready_ips = set()
        else:
            self._ready_ips = {
                addr.ip
                for subset in ep_obj.subsets or []
                for addr in subset.addresses or []
                if getattr(addr, "ip", None)
            }
            # Don't signal ready on a leading DELETE (would leave ring empty).
            if not self._initial_ready.is_set():
                self._initial_ready.set()
        self._rebuild_ring()

    def _rebuild_ring(self) -> None:
        all_ips = sorted(self._ready_ips)
        new_urls = [f"http://{ip}:{self._port}" for ip in all_ips]
        if new_urls == self._last_urls:
            return
        added = sorted(set(new_urls) - set(self._last_urls))
        removed = sorted(set(self._last_urls) - set(new_urls))
        self._last_urls = new_urls

        self._ring = HashRing(new_urls, vnodes=self._vnodes) if new_urls else None

        logger.info(
            "encoder endpoints updated: svc=%s/%s count=%d added=%s removed=%s",
            self._ns,
            self._svc,
            len(new_urls),
            added,
            removed,
        )


class LocalHashRingDispatcher:
    def __init__(
        self,
        service_url: str,
        vnodes: int = 160,
        max_stale_seconds: float = 300.0,
    ):
        self._resolver = EndpointResolver(service_url, vnodes=vnodes)
        self._fallback_url = service_url
        self._max_stale = max_stale_seconds
        self._ready = asyncio.Event()
        self._starting = False

    async def start(self) -> None:
        if self._ready.is_set():
            return
        if self._starting:
            # Concurrent caller: wait for the in-flight start to finish.
            await self._ready.wait()
            return
        self._starting = True
        try:
            await self._resolver.start()
        finally:
            # Set even on failure to unblock waiters; retries won't reinit.
            self._ready.set()

    async def stop(self) -> None:
        await self._resolver.stop()

    def pick(self, mm_item_key: str, exclude: Optional[Set[str]] = None) -> str:
        ring = self._resolver.ring
        stale = (time.monotonic() - self._resolver.last_event_at) > self._max_stale
        if ring is None or stale:
            return self._fallback_url
        return ring.get(mm_item_key, exclude=exclude) or self._fallback_url
