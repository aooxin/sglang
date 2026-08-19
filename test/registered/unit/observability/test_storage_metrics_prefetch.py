import unittest

from prometheus_client import REGISTRY

from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestStorageMetricsPrefetch(unittest.TestCase):
    def setUp(self):
        # StorageMetricsCollector.__init__ reads the global server args for
        # histogram bucket configuration, so it must be set before construction.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        # StorageMetricsCollector registers ~15 named metrics into the process-
        # global Prometheus registry. Prometheus forbids two collectors sharing a
        # metric NAME (label keys are irrelevant to the clash), so constructing a
        # second collector in the same process raises "Duplicated timeseries".
        # Clear the registry before each test so every construction is fresh.
        for collector in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(collector)

    def _collector(self):
        return StorageMetricsCollector(labels={"storage_backend": "test"})

    def test_log_prefetch_retry_increments(self):
        collector = self._collector()
        # Must not raise; counter exists and is callable.
        collector.log_prefetch_retry()
        sample = collector.prefetch_retry_total.labels(**collector.labels)._value.get()
        self.assertEqual(sample, 1.0)

    def test_log_prefetch_dropped_increments(self):
        collector = self._collector()
        collector.log_prefetch_dropped()
        sample = (
            collector.prefetch_dropped_total.labels(**collector.labels)._value.get()
        )
        self.assertEqual(sample, 1.0)

    def test_log_prefetch_dropped_observes_pending_seconds(self):
        collector = self._collector()
        collector.log_prefetch_dropped(pending_seconds=2.5)
        hist = collector.histogram_prefetch_pending_dropped_seconds.labels(
            **collector.labels
        )
        # One observation recorded, summing to 2.5s.
        self.assertEqual(hist._sum.get(), 2.5)

    def test_log_prefetch_dropped_skips_histogram_when_seconds_none(self):
        collector = self._collector()
        collector.log_prefetch_dropped(pending_seconds=None)
        hist = collector.histogram_prefetch_pending_dropped_seconds.labels(
            **collector.labels
        )
        # Counter still incremented, but no histogram observation.
        self.assertEqual(
            collector.prefetch_dropped_total.labels(**collector.labels)._value.get(),
            1.0,
        )
        self.assertEqual(hist._sum.get(), 0.0)


if __name__ == "__main__":
    unittest.main()
