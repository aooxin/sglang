import unittest

from prometheus_client import REGISTRY

from sglang.srt.mem_cache.base_prefix_cache import PrefetchStatus
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
        # Pass the enum; the collector derives the reason label internally
        # (.name.lower()). The lookup below by the *string* label proves the
        # enum-to-label derivation lands on the expected time series.
        collector.log_prefetch_retry(PrefetchStatus.RATE_LIMITED)
        sample = collector.prefetch_retry_total.labels(
            **collector.labels, reason="rate_limited"
        )._value.get()
        self.assertEqual(sample, 1.0)

    def test_prefetch_retry_labeled_by_reason(self):
        # The reason label must keep rate_limited and host_alloc_failed as
        # separate time series so the two retry causes can be attributed apart.
        collector = self._collector()
        collector.log_prefetch_retry(PrefetchStatus.RATE_LIMITED)
        collector.log_prefetch_retry(PrefetchStatus.HOST_ALLOC_FAILED)
        collector.log_prefetch_retry(PrefetchStatus.HOST_ALLOC_FAILED)
        rl = collector.prefetch_retry_total.labels(
            **collector.labels, reason="rate_limited"
        )._value.get()
        haf = collector.prefetch_retry_total.labels(
            **collector.labels, reason="host_alloc_failed"
        )._value.get()
        self.assertEqual(rl, 1.0)
        self.assertEqual(haf, 2.0)

    def test_log_prefetch_dropped_increments(self):
        collector = self._collector()
        collector.log_prefetch_dropped(PrefetchStatus.HOST_ALLOC_FAILED)
        sample = collector.prefetch_dropped_total.labels(
            **collector.labels, reason="host_alloc_failed"
        )._value.get()
        self.assertEqual(sample, 1.0)

    def test_prefetch_dropped_labeled_by_reason(self):
        collector = self._collector()
        collector.log_prefetch_dropped(PrefetchStatus.RATE_LIMITED)
        collector.log_prefetch_dropped(PrefetchStatus.HOST_ALLOC_FAILED)
        rl = collector.prefetch_dropped_total.labels(
            **collector.labels, reason="rate_limited"
        )._value.get()
        haf = collector.prefetch_dropped_total.labels(
            **collector.labels, reason="host_alloc_failed"
        )._value.get()
        self.assertEqual(rl, 1.0)
        self.assertEqual(haf, 1.0)

    def test_log_prefetch_dropped_observes_pending_seconds(self):
        collector = self._collector()
        collector.log_prefetch_dropped(PrefetchStatus.RATE_LIMITED, pending_seconds=2.5)
        # The pending-dwell histogram is not labeled by reason.
        hist = collector.histogram_prefetch_pending_dropped_seconds.labels(
            **collector.labels
        )
        # One observation recorded, summing to 2.5s.
        self.assertEqual(hist._sum.get(), 2.5)

    def test_log_prefetch_dropped_skips_histogram_when_seconds_none(self):
        collector = self._collector()
        collector.log_prefetch_dropped(
            PrefetchStatus.RATE_LIMITED, pending_seconds=None
        )
        hist = collector.histogram_prefetch_pending_dropped_seconds.labels(
            **collector.labels
        )
        # Counter still incremented, but no histogram observation.
        self.assertEqual(
            collector.prefetch_dropped_total.labels(
                **collector.labels, reason="rate_limited"
            )._value.get(),
            1.0,
        )
        self.assertEqual(hist._sum.get(), 0.0)


if __name__ == "__main__":
    unittest.main()
