import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import PrefetchStatus

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.prefetch_rate_limited_rounds = 0
        self.metadata_buffer_index = -1  # asserted to stay -1: buffer never allocated
        self.time_stats = MagicMock()
        # Real float: the drop path reads this to compute pending-queue dwell
        # time. MagicMock's auto-attr would break the numeric comparison.
        self.time_stats.prefill_bootstrap_queue_entry_time = 0.0


class TestSchedulerPrefetchPending(unittest.TestCase):
    def _make_prefill_scheduler(self, *, max_rounds=3):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.enable_hicache_storage = True
        s.prefetch_pending_queue = []
        s.model_config = SimpleNamespace(num_key_value_heads=8)
        s.tree_cache = MagicMock()
        s.tree_cache.prefetch_max_retry_rounds = max_rounds
        s.disagg_prefill_bootstrap_queue = MagicMock()
        s.send_to_tokenizer = MagicMock()
        return s

    def test_prefetch_kvcache_returns_skipped_when_storage_off(self):
        s = self._make_prefill_scheduler()
        s.enable_hicache_storage = False
        req = _FakeReq("r0")
        status = Scheduler._prefetch_kvcache(s, req)
        self.assertEqual(status, PrefetchStatus.SKIPPED)

    def test_drain_promote_started(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        req = _FakeReq("r0")
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.STARTED)
        Scheduler._drain_prefetch_pending_queue(s)
        self.assertEqual(s.prefetch_pending_queue, [])
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)

    # RATE_LIMITED and HOST_ALLOC_FAILED share the same retry/requeue/drop path;
    # the only observable difference is the Prometheus reason label. Parametrize
    # over both so the two statuses can't silently diverge.
    _RETRYABLE = [
        (PrefetchStatus.RATE_LIMITED, "rate_limited"),
        (PrefetchStatus.HOST_ALLOC_FAILED, "host_alloc_failed"),
    ]

    def test_drain_keep_and_count(self):
        # Retryable status under budget: requeue + increment the round + log a
        # retry with the status's own label (no bootstrap this pass).
        for status, label in self._RETRYABLE:
            with self.subTest(status=status):
                s = self._make_prefill_scheduler(max_rounds=3)
                req = _FakeReq("r0")
                s.prefetch_pending_queue = [req]
                s._prefetch_kvcache = MagicMock(return_value=status)
                Scheduler._drain_prefetch_pending_queue(s)
                self.assertEqual(s.prefetch_pending_queue, [req])
                self.assertEqual(req.prefetch_rate_limited_rounds, 1)
                s.disagg_prefill_bootstrap_queue.add.assert_not_called()
                # Scheduler passes the enum itself; the collector derives the
                # label. Assert the enum here, and cross-check the mapping.
                s.tree_cache.storage_metrics_collector.log_prefetch_retry.assert_called_once_with(
                    status
                )
                self.assertEqual(status.name.lower(), label)

    def test_drain_429_when_exhausted(self):
        # Retryable status past the budget is dropped with 429 (released + the
        # status's own drop label), never silently recomputed.
        for status, label in self._RETRYABLE:
            with self.subTest(status=status):
                s = self._make_prefill_scheduler(max_rounds=2)
                req = _FakeReq("r0")
                req.prefetch_rate_limited_rounds = 2  # already at bound
                s.prefetch_pending_queue = [req]
                s._prefetch_kvcache = MagicMock(return_value=status)
                Scheduler._drain_prefetch_pending_queue(s)
                self.assertEqual(s.prefetch_pending_queue, [])
                s.tree_cache.release_aborted_request.assert_called_once_with("r0")
                sent_obj = s.send_to_tokenizer.send_output.call_args[0][0]
                self.assertEqual(
                    sent_obj.finished_reason["status_code"],
                    HTTPStatus.TOO_MANY_REQUESTS,
                )
                dropped = s.tree_cache.storage_metrics_collector.log_prefetch_dropped
                dropped.assert_called_once()
                self.assertEqual(dropped.call_args[0][0], status)
                self.assertEqual(status.name.lower(), label)

    def test_drop_records_pending_dwell_seconds(self):
        # When a parked req is dropped, log_prefetch_dropped receives the time
        # it spent in the pending queue (now - park entry timestamp), > 0.
        import time as _time

        s = self._make_prefill_scheduler(max_rounds=2)
        req = _FakeReq("r0")
        req.prefetch_rate_limited_rounds = 2  # already at bound -> 429 this pass
        req.time_stats.prefill_bootstrap_queue_entry_time = _time.perf_counter() - 1.5
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.RATE_LIMITED)
        Scheduler._drain_prefetch_pending_queue(s)
        dropped = s.tree_cache.storage_metrics_collector.log_prefetch_dropped
        dropped.assert_called_once()
        reason, pending_seconds = dropped.call_args[0]
        self.assertEqual(reason, PrefetchStatus.RATE_LIMITED)
        self.assertIsNotNone(pending_seconds)
        self.assertGreaterEqual(pending_seconds, 1.5)

    def test_drop_pending_seconds_none_when_no_entry_time(self):
        # No park timestamp (entry == 0.0) -> dwell time is None, not a bogus
        # huge value; the counter still increments.
        s = self._make_prefill_scheduler(max_rounds=2)
        req = _FakeReq("r0")
        req.prefetch_rate_limited_rounds = 2
        req.time_stats.prefill_bootstrap_queue_entry_time = 0.0
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.RATE_LIMITED)
        Scheduler._drain_prefetch_pending_queue(s)
        dropped = s.tree_cache.storage_metrics_collector.log_prefetch_dropped
        dropped.assert_called_once()
        reason, pending_seconds = dropped.call_args[0]
        self.assertEqual(reason, PrefetchStatus.RATE_LIMITED)
        self.assertIsNone(pending_seconds)

    def test_park_when_limited(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        s.tree_cache.cache_controller.prefetch_rate_limited.return_value = True
        s._prefetch_kvcache = MagicMock()
        req = _FakeReq("r0")
        Scheduler._add_request_to_queue(s, req)
        self.assertIn(req, s.prefetch_pending_queue)
        s._prefetch_kvcache.assert_not_called()
        s.disagg_prefill_bootstrap_queue.add.assert_not_called()
        self.assertEqual(req.metadata_buffer_index, -1)

    def test_passthrough_when_not_limited(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        s.tree_cache.cache_controller.prefetch_rate_limited.return_value = False
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.STARTED)
        req = _FakeReq("r0")
        Scheduler._add_request_to_queue(s, req)
        self.assertNotIn(req, s.prefetch_pending_queue)
        s._prefetch_kvcache.assert_called_once_with(req)
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)

    def test_add_parks_when_host_alloc_failed(self):
        # Not occupancy-limited, so we fire the prefetch — but it comes back
        # HOST_ALLOC_FAILED (host pool transiently full). Park & retry rather than
        # bootstrap/recompute now. Unlike the rate-limited path, the prefetch IS
        # fired (that's how we learn the alloc failed), and it self-released host.
        s = self._make_prefill_scheduler(max_rounds=3)
        s.tree_cache.cache_controller.prefetch_rate_limited.return_value = False
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.HOST_ALLOC_FAILED)
        req = _FakeReq("r0")
        Scheduler._add_request_to_queue(s, req)
        self.assertIn(req, s.prefetch_pending_queue)
        s._prefetch_kvcache.assert_called_once_with(req)
        s.disagg_prefill_bootstrap_queue.add.assert_not_called()

    def test_add_hostfail_bootstraps_when_disabled(self):
        # Feature OFF: HOST_ALLOC_FAILED must fall straight through to bootstrap
        # (bit-identical to trunk), never parked.
        s = self._make_prefill_scheduler(max_rounds=0)
        s.tree_cache.cache_controller.prefetch_rate_limited.return_value = False
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.HOST_ALLOC_FAILED)
        req = _FakeReq("r0")
        Scheduler._add_request_to_queue(s, req)
        self.assertEqual(s.prefetch_pending_queue, [])
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)

    def test_disabled_is_bit_identical(self):
        s = self._make_prefill_scheduler(max_rounds=0)
        s.tree_cache.cache_controller.prefetch_rate_limited.return_value = True
        s._prefetch_kvcache = MagicMock()
        req = _FakeReq("r0")
        Scheduler._add_request_to_queue(s, req)
        self.assertEqual(s.prefetch_pending_queue, [])
        s._prefetch_kvcache.assert_called_once_with(req)
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)

    def test_abort_request_clears_pending(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        # Other queues abort_request touches — all empty.
        s.waiting_queue = []
        s.grammar_manager = MagicMock()
        s.disagg_prefill_bootstrap_queue.queue = []
        s.disagg_prefill_inflight_queue = []
        s.running_batch = SimpleNamespace(reqs=[])
        s.cur_batch = None
        # Stubbed so we can prove the pending scan never touches it.
        s.req_to_metadata_buffer_idx_allocator = MagicMock()
        req = _FakeReq("r0")
        s.prefetch_pending_queue = [req]

        Scheduler.abort_request(s, AbortReq(rid="r0"))

        self.assertEqual(s.prefetch_pending_queue, [])
        sent_obj = s.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(sent_obj.rid, "r0")
        # metadata buffer must NOT be released (pending never allocated one)
        s.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

    def test_disabled_drain_is_noop(self):
        s = self._make_prefill_scheduler(max_rounds=0)
        s.prefetch_pending_queue = []  # never populated when disabled
        s._prefetch_kvcache = MagicMock()
        Scheduler._drain_prefetch_pending_queue(s)
        s._prefetch_kvcache.assert_not_called()

    def test_drain_passthrough_skipped(self):
        # SKIPPED has nothing worth prefetching (below threshold / not backuped),
        # so it falls through to bootstrap/compute — no retry, no requeue.
        s = self._make_prefill_scheduler(max_rounds=3)
        req = _FakeReq("r0")
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.SKIPPED)
        Scheduler._drain_prefetch_pending_queue(s)
        self.assertEqual(s.prefetch_pending_queue, [])
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)

    def test_fifo_and_mixed(self):
        s = self._make_prefill_scheduler(max_rounds=2)
        a, b, c = _FakeReq("a"), _FakeReq("b"), _FakeReq("c")
        c.prefetch_rate_limited_rounds = 2  # c already at bound → 429 this pass
        s.prefetch_pending_queue = [a, b, c]
        s._prefetch_kvcache = MagicMock(
            side_effect=[
                PrefetchStatus.STARTED,  # a → bootstrap
                PrefetchStatus.RATE_LIMITED,  # b → keep
                PrefetchStatus.RATE_LIMITED,  # c → 429
            ]
        )
        Scheduler._drain_prefetch_pending_queue(s)
        self.assertEqual(s.prefetch_pending_queue, [b])  # order preserved
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(a, 8)
        # c aborted: released + 429 sent
        s.tree_cache.release_aborted_request.assert_called_once_with("c")

    def test_abort_pending_hicache_off(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        s.enable_hicache_storage = False
        req = _FakeReq("r0")
        Scheduler._abort_pending_prefetch(s, req, PrefetchStatus.RATE_LIMITED)
        s.tree_cache.release_aborted_request.assert_not_called()
        s.send_to_tokenizer.send_output.assert_called_once()
        sent_obj = s.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(
            sent_obj.finished_reason["status_code"], HTTPStatus.TOO_MANY_REQUESTS
        )

    def test_abort_pending_no_collector(self):
        s = self._make_prefill_scheduler(max_rounds=2)
        s.tree_cache.storage_metrics_collector = None  # metrics disabled
        req = _FakeReq("r0")
        req.prefetch_rate_limited_rounds = 2
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.RATE_LIMITED)
        # Must not raise despite collector being None.
        Scheduler._drain_prefetch_pending_queue(s)
        self.assertEqual(s.prefetch_pending_queue, [])

    def test_drain_keeps_req_exactly_at_boundary(self):
        # Post-increment rounds == max_rounds must KEEP (operator is strict >).
        # A `>`→`>=` off-by-one would abort here one pass early; this pins it.
        s = self._make_prefill_scheduler(max_rounds=2)
        req = _FakeReq("r0")
        req.prefetch_rate_limited_rounds = 1  # → 2 after this pass, == max_rounds
        s.prefetch_pending_queue = [req]
        s._prefetch_kvcache = MagicMock(return_value=PrefetchStatus.RATE_LIMITED)
        Scheduler._drain_prefetch_pending_queue(s)
        self.assertEqual(s.prefetch_pending_queue, [req])  # kept, not 429'd
        self.assertEqual(req.prefetch_rate_limited_rounds, 2)
        s.tree_cache.release_aborted_request.assert_not_called()
        s.tree_cache.storage_metrics_collector.log_prefetch_retry.assert_called_once_with(
            PrefetchStatus.RATE_LIMITED
        )

    def test_abort_request_retains_non_matching_pending(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        s.waiting_queue = []
        s.grammar_manager = MagicMock()
        s.disagg_prefill_bootstrap_queue.queue = []
        s.disagg_prefill_inflight_queue = []
        s.running_batch = SimpleNamespace(reqs=[])
        s.cur_batch = None
        s.req_to_metadata_buffer_idx_allocator = MagicMock()
        keep, drop = _FakeReq("keep"), _FakeReq("drop")
        s.prefetch_pending_queue = [keep, drop]

        Scheduler.abort_request(s, AbortReq(rid="drop"))

        self.assertEqual(s.prefetch_pending_queue, [keep])
        sent_obj = s.send_to_tokenizer.send_output.call_args[0][0]
        self.assertEqual(sent_obj.rid, "drop")

    def test_abort_request_all_clears_pending(self):
        s = self._make_prefill_scheduler(max_rounds=3)
        s.waiting_queue = []
        s.grammar_manager = MagicMock()
        s.disagg_prefill_bootstrap_queue.queue = []
        s.disagg_prefill_inflight_queue = []
        s.running_batch = SimpleNamespace(reqs=[])
        s.cur_batch = None
        s.req_to_metadata_buffer_idx_allocator = MagicMock()
        a, b = _FakeReq("a"), _FakeReq("b")
        s.prefetch_pending_queue = [a, b]

        Scheduler.abort_request(s, AbortReq(abort_all=True))

        self.assertEqual(s.prefetch_pending_queue, [])
        self.assertEqual(s.send_to_tokenizer.send_output.call_count, 2)

    def test_validate_config_rejects_feature_under_pp(self):
        # Feature ON (max_rounds>0) under pp_size>1 must be rejected at startup:
        # the PP disagg-prefill loop has no drain, so parked reqs would strand.
        s = self._make_prefill_scheduler(max_rounds=3)
        s.pp_size = 2
        with self.assertRaises(ValueError):
            Scheduler._validate_prefetch_pending_config(s)

    def test_validate_config_allows_feature_off_under_pp(self):
        # Feature OFF (max_rounds==0) under pp_size>1 is fine — bit-identical to trunk.
        s = self._make_prefill_scheduler(max_rounds=0)
        s.pp_size = 2
        Scheduler._validate_prefetch_pending_config(s)  # must not raise

    def test_validate_config_allows_feature_on_without_pp(self):
        # Feature ON with pp_size==1 is the supported config — must not raise.
        s = self._make_prefill_scheduler(max_rounds=3)
        s.pp_size = 1
        Scheduler._validate_prefetch_pending_config(s)  # must not raise


if __name__ == "__main__":
    unittest.main()
