"""Regression test: report_prefill_stats must populate the prefetch pending-queue
metric from the live queue.

The bug this pins: report_prefill_stats (the per-batch stats publisher for the
disagg PREFILL scheduler) refreshed the sibling prefill queue gauges but NEVER
assigned stats.num_prefetch_pending_queue_reqs. Its only other writer,
check_memory (scheduler_runtime_checker_mixin.py), is commented out in
self_check_during_idle and never runs in the disagg-prefill loop. So the gauge
stayed pinned at its dataclass default 0 while requests were actually parked and
429-dropped. The parked reqs were also omitted from num_prefill_total_reqs.

The existing collector test (test_scheduler_metrics_prefetch_pending.py) only
proves the stats->gauge plumbing; it sets the field by hand. This test exercises
the population step that was missing.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.observability.metrics_collector import QueueCount, SchedulerStats

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def _prefill_stats():
    # Minimal PrefillStats stand-in: only the fields report_prefill_stats reads.
    return SimpleNamespace(
        new_token_ratio=0.0,
        num_new_seqs=0,
        log_input_tokens=0,
        log_hit_tokens=0,
        log_host_hit_tokens=0,
        num_running_reqs=QueueCount(total=0),
        total_flops=0,
    )


class TestReportPrefillStatsPendingQueue(unittest.TestCase):
    def _make_scheduler(self, *, pending, bootstrap=0, inflight=0, waiting=0):
        """A PREFILL scheduler stub wired with just enough state for
        report_prefill_stats to run its metrics-enabled path. All heavy
        collaborators are mocked; only the queue-accounting inputs are real."""
        s = Scheduler.__new__(Scheduler)

        # --- gates: take the metrics-enabled, non-logging, prefill path ---
        s.current_scheduler_metrics_enabled = True
        s.is_stats_logging_rank = False
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.is_hybrid_swa = False
        s.is_hybrid_ssm = False
        s.enable_metrics = False  # skip two-phase commit-wait sampling
        s.enable_mfu_metrics = False
        s.enable_priority_scheduling = False
        s.device = "cpu"
        s.forward_ct = 0

        # --- throughput bookkeeping (perf_counter - 0.0 => positive gap) ---
        s.last_prefill_stats_tic = 0.0
        s.last_prefill_tokens = 0
        s.last_input_throughput = 0.0
        s.last_tree_cache_info_time = 0.0

        # --- server args the prefill path reads ---
        s.server_args = SimpleNamespace(
            disaggregation_two_phase_prealloc=False,
            language_only=False,
            encoder_transfer_backend="none",
        )

        # --- the queues under test ---
        s.prefetch_pending_queue = [SimpleNamespace(rid=f"p{i}") for i in range(pending)]
        s.waiting_queue = [SimpleNamespace(rid=f"w{i}") for i in range(waiting)]
        s.disagg_prefill_inflight_queue = [
            SimpleNamespace(rid=f"i{i}") for i in range(inflight)
        ]
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(
            queue=[SimpleNamespace(rid=f"b{i}") for i in range(bootstrap)]
        )

        # grammar manager: len(...) and .grammar_queue are both read.
        gm = MagicMock()
        gm.__len__.return_value = 0
        gm.grammar_queue = []
        s.grammar_manager = gm

        # --- token pool: real numbers feed a division ---
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 100,
            req_to_token=SimpleNamespace(shape=[100]),
        )
        s._get_token_info = MagicMock(return_value=(5, 0.5, 0, 0))
        s.max_total_num_tokens = 1000
        s.num_retracted_reqs = 0
        s.num_paused_reqs = 0

        # kv-transfer telemetry (copied verbatim into stats)
        s.kv_transfer_speed_gb_s = 0.0
        s.kv_transfer_latency_ms = 0.0
        s.kv_transfer_rdma_speed_gb_s = 0.0
        s.kv_transfer_rdma_latency_ms = 0.0

        # --- mocked-out collaborators (side effects only) ---
        s.tree_cache = MagicMock()
        s.metrics_collector = MagicMock()
        s.calculate_utilization = MagicMock()
        s.update_lora_metrics = MagicMock()
        s._log_hicache_stats = MagicMock()
        s._emit_kv_metrics = MagicMock()
        s._publish_kv_events = MagicMock()

        s.stats = SchedulerStats()
        return s

    def test_pending_depth_reported(self):
        s = self._make_scheduler(pending=3)
        Scheduler.report_prefill_stats(
            s, prefill_stats=_prefill_stats(), can_run_cuda_graph=False
        )
        # The bug left this at the default 0 despite 3 parked reqs.
        self.assertEqual(s.stats.num_prefetch_pending_queue_reqs, 3)

    def test_pending_included_in_total(self):
        # bootstrap=2, inflight=1, waiting=4, running=0, grammar=0, pending=3 => total 10
        s = self._make_scheduler(pending=3, bootstrap=2, inflight=1, waiting=4)
        Scheduler.report_prefill_stats(
            s, prefill_stats=_prefill_stats(), can_run_cuda_graph=False
        )
        self.assertEqual(s.stats.num_prefetch_pending_queue_reqs, 3)
        # num_prefill_total_reqs must count parked reqs too (was undercounting).
        self.assertEqual(s.stats.num_prefill_total_reqs, 2 + 1 + 4 + 0 + 0 + 3)

    def test_empty_pending_reports_zero(self):
        s = self._make_scheduler(pending=0, bootstrap=1)
        Scheduler.report_prefill_stats(
            s, prefill_stats=_prefill_stats(), can_run_cuda_graph=False
        )
        self.assertEqual(s.stats.num_prefetch_pending_queue_reqs, 0)
        self.assertEqual(s.stats.num_prefill_total_reqs, 1)


if __name__ == "__main__":
    unittest.main()
