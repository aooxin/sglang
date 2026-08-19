import unittest

from prometheus_client import REGISTRY

from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSchedulerMetricsPrefetchPending(unittest.TestCase):
    def setUp(self):
        # SchedulerMetricsCollector reads server args (histogram buckets, delayer
        # config) and registers many named metrics into the process-global
        # Prometheus registry. Prometheus forbids two collectors sharing a metric
        # NAME, so clear the registry before each construction to keep it fresh.
        self.server_args = ServerArgs(model_path="dummy")
        set_global_server_args_for_scheduler(self.server_args)
        for collector in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(collector)

    def _collector(self):
        # Mirror the label set production builds (scheduler_metrics_mixin):
        # __init__ indexes labels["moe_ep_rank"], so all keys must be present.
        labels = {
            "model_name": "dummy",
            "engine_type": "prefill",
            "tp_rank": 0,
            "pp_rank": 0,
            "moe_ep_rank": 0,
        }
        return SchedulerMetricsCollector(labels=labels, server_args=self.server_args)

    def _pending_sample(self, collector):
        return collector.num_prefetch_pending_queue_reqs.labels(
            **collector.labels
        )._value.get()

    def test_gauge_registered_with_expected_name(self):
        collector = self._collector()
        # The gauge exists on the scheduler collector (not the storage one).
        self.assertTrue(hasattr(collector, "num_prefetch_pending_queue_reqs"))

    def test_reports_pending_queue_length(self):
        collector = self._collector()
        stats = SchedulerStats()
        stats.num_prefetch_pending_queue_reqs = 7
        collector.log_stats(stats)
        self.assertEqual(self._pending_sample(collector), 7.0)

    def test_defaults_to_zero(self):
        # Fresh stats (no PREFILL parking) report 0 — bit-identical to trunk
        # behavior on non-prefill / feature-off nodes.
        collector = self._collector()
        collector.log_stats(SchedulerStats())
        self.assertEqual(self._pending_sample(collector), 0.0)


if __name__ == "__main__":
    unittest.main()
