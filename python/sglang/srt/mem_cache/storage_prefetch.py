from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey


class PrefetchTiming(msgspec.Struct):
    """time.monotonic() stamps of one L3 prefetch; 0.0 means not reached.

    Each stamp is written by one thread and read by the scheduler thread after
    the operation has passed through the queue that orders the two.
    """

    start: float = 0.0
    start_wall: float = 0.0
    # Prefetch thread.
    query_start: float = 0.0
    query_finish: float = 0.0
    resolve_finish: float = 0.0
    # Scheduler thread: hit drained from the queue, then host staging allocated.
    hit_drained: float = 0.0
    io_enqueued: float = 0.0
    # IO thread.
    io_start: float = 0.0
    kv_finish: float = 0.0
    io_finish: float = 0.0
    # Sync thread: completion agreed across ranks.
    ack_reduced: float = 0.0
    # Scheduler thread: committed into the tree, or revoked.
    commit: float = 0.0

    @classmethod
    def started(cls) -> PrefetchTiming:
        return cls(start=time.monotonic(), start_wall=time.time())

    def stage_durations(self) -> dict[str, float]:
        """Seconds per stage, using 0510's stage names; stages are contiguous."""
        return self._segments(
            ("queue", self.query_start),
            ("query", self.query_finish),
            ("resolve", self.resolve_finish),
            ("io_queue", self.io_start),
            ("io", self.io_finish),
            ("commit", self.commit),
        )

    def timeline_ms(self) -> str:
        """Finer split of the stages for logs: thread hand-offs are separate."""
        segments = self._segments(
            ("queue", self.query_start),
            ("query", self.query_finish),
            ("sync", self.resolve_finish),
            ("hit_wait", self.hit_drained),
            ("alloc", self.io_enqueued),
            ("io_wait", self.io_start),
            ("kv", self.kv_finish),
            ("sidecar", self.io_finish),
            ("ack_sync", self.ack_reduced),
            ("ack_wait", self.commit),
        )
        parts = [f"{name}={seconds * 1000:.1f}" for name, seconds in segments.items()]
        return " ".join(parts + [f"total={sum(segments.values()) * 1000:.1f}"])

    def _segments(self, *stamps: tuple[str, float]) -> dict[str, float]:
        # A stage runs from the latest stamp reached before it; unreached stages
        # are omitted, so a revoked prefetch reports no IO stages.
        segments: dict[str, float] = {}
        previous = self.start
        for name, stamp in stamps:
            if stamp > 0.0:
                segments[name] = stamp - previous
                previous = stamp
        return segments


@dataclass
class _StoragePrefetchRetry:
    immediate: bool
    storage_hit_end: Optional[int] = None
    due_step: Optional[int] = None


class StoragePrefetchRetries:
    """New L3 attempts only (parked IO and retained staging retry at their owner);
    step deadlines keep TP ranks in lockstep, retries follow queue order, and a
    request past its re-issue budget is admitted with what the device holds."""

    def __init__(self):
        self._pending: dict[str, _StoragePrefetchRetry] = {}
        self._step = 0

    def poll_miss(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(False, storage_hit_end)

    def refetch(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(True, storage_hit_end)

    def cancel(self, req_id: str) -> None:
        self._pending.pop(req_id, None)

    def clear(self) -> None:
        self._pending.clear()

    def pop_ready(
        self, waiting_queue: list[Req], interval: int, max_attempts: int
    ) -> list[tuple[Req, Optional[int]]]:
        self._step += 1
        if not waiting_queue or not self._pending:
            return []
        # A speculative miss must never delay the queue head.
        head_id = waiting_queue[0].rid
        head_retry = self._pending.get(head_id)
        if head_retry is not None and not head_retry.immediate:
            self.cancel(head_id)
        if not any(
            retry.due_step is None or retry.due_step <= self._step
            for retry in self._pending.values()
        ):
            return []

        ready = []
        for req in waiting_queue:
            retry = self._pending.get(req.rid)
            if retry is None:
                continue
            if req.storage_prefetch_retry_attempts >= max_attempts:
                self.cancel(req.rid)
                continue
            if not retry.immediate:
                if interval <= 0:
                    self.cancel(req.rid)
                    continue
                if retry.due_step is None:
                    retry.due_step = self._step + interval
                if retry.due_step > self._step:
                    continue
            self.cancel(req.rid)
            ready.append((req, retry.storage_hit_end))
        return ready


@dataclass(frozen=True)
class StagedPrefetchPlan:
    """One admission pass, computed before promoting the joint FULL/SWA match."""

    operation_id: int
    key: RadixKey
    device_prefix_len: int
    full_tokens: int
    swa_tokens: int
