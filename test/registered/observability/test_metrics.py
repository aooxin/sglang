import unittest
from types import SimpleNamespace
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GetLoadsReqInput
from sglang.srt.observability.metrics_collector import (
    ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
    compute_routing_key_stats,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=95, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=32, suite="stage-b-test-1-gpu-small-amd")

_MODEL_NAME = "Qwen/Qwen3-0.6B"


class TestEnableMetrics(CustomTestCase):
    def test_metrics_1gpu(self):
        """Test that metrics endpoint returns data when enabled"""
        self._execute_core(
            other_args=[],
            verify_metrics_extra=None,
            expect_mfu_metrics=True,
            enable_mfu_metrics=True,
        )

    def test_mfu_metrics_gate_disabled(self):
        """MFU metrics should not be emitted when the gate is disabled."""
        self._execute_core(
            other_args=[],
            verify_metrics_extra=None,
            expect_mfu_metrics=False,
            enable_mfu_metrics=False,
        )

    def test_metrics_2gpu(self):
        # TODO enable when we have 2-gpu runner in nightly CI
        if is_in_ci():
            print("Skip test_metrics_2gpu since in 1-gpu CI")
            return

        def _verify_metrics_extra(metrics):
            metrics_to_check = [
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "prefill_compute"},
                ),
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "decode"},
                ),
                (
                    "sglang:dp_cooperation_gpu_execution_seconds_total",
                    {"category": "forward_extend"},
                ),
                (
                    "sglang:dp_cooperation_gpu_execution_seconds_total",
                    {"category": "forward_decode"},
                ),
            ]
            _check_metrics_positive(self, metrics, metrics_to_check)

            num_prefill_ranks_values = {
                s.labels["num_prefill_ranks"]
                for s in metrics["sglang:dp_cooperation_realtime_tokens_total"]
            }
            self.assertIn("0", num_prefill_ranks_values)
            self.assertIn("1", num_prefill_ranks_values)

        self._execute_core(
            other_args=["--tp", "2", "--dp", "2", "--enable-dp-attention"],
            verify_metrics_extra=_verify_metrics_extra,
            expect_mfu_metrics=True,
            enable_mfu_metrics=True,
        )

    def _execute_core(
        self,
        other_args,
        verify_metrics_extra,
        expect_mfu_metrics: bool,
        enable_mfu_metrics: bool,
    ):
        with (
            envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.override(True),
            envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.override(True),
            envs.SGLANG_TEST_RETRACT.override(True),
        ):
            launch_args = ["--enable-metrics", "--cuda-graph-max-bs", 2, *other_args]
            if enable_mfu_metrics:
                launch_args.insert(1, "--enable-mfu-metrics")
            process = popen_launch_server(
                _MODEL_NAME,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

        try:
            # Make some requests to generate some metrics
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": ["The capital of France is"] * 20,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 50,
                    },
                    "stream": True,
                    "ignore_eos": True,
                },
                stream=True,
            )
            for _ in response.iter_lines(decode_unicode=False):
                pass

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                },
                headers={"x-smg-routing-key": "test-key"},
            )
            self.assertEqual(response.status_code, 200)

            # Get metrics
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_text = metrics_response.text

            print(f"metrics_text=\n{metrics_text}")

            metrics = _parse_prometheus_metrics(metrics_text)
            self._verify_metrics_common(metrics_text, metrics, expect_mfu_metrics)
            if verify_metrics_extra is not None:
                verify_metrics_extra(metrics)
        finally:
            kill_process_tree(process.pid)

    def _verify_metrics_common(self, metrics_text, metrics, expect_mfu_metrics: bool):
        essential_metrics = [
            "sglang:num_running_reqs",
            "sglang:num_used_tokens",
            "sglang:token_usage",
            "sglang:gen_throughput",
            "sglang:num_queue_reqs",
            "sglang:num_grammar_queue_reqs",
            "sglang:cache_hit_rate",
            "sglang:spec_accept_length",
            "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total",
            "sglang:cached_tokens_total",
            "sglang:num_requests_total",
            "sglang:time_to_first_token_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:e2e_request_latency_seconds",
            "sglang:http_requests_active",
            "sglang:routing_keys_active",
            "sglang:num_unique_running_routing_keys",
            "sglang:routing_key_running_req_count",
            "sglang:routing_key_all_req_count",
        ]
        mfu_metrics = [
            "sglang:estimated_flops_per_gpu_total",
            "sglang:estimated_read_bytes_per_gpu_total",
            "sglang:estimated_write_bytes_per_gpu_total",
        ]
        if expect_mfu_metrics:
            essential_metrics.extend(mfu_metrics)
        for metric in essential_metrics:
            self.assertIn(metric, metrics_text, f"Missing metric: {metric}")

        # Verify routing key GaugeHistogram buckets
        expected_buckets = len(ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS) + 1
        for metric_name in [
            "sglang:routing_key_running_req_count",
            "sglang:routing_key_all_req_count",
        ]:
            gt_le_pairs = set()
            for sample in metrics.get(metric_name, []):
                gt_le_pairs.add((sample.labels.get("gt"), sample.labels.get("le")))
            self.assertEqual(
                len(gt_le_pairs),
                expected_buckets,
                f"{metric_name}: Expected {expected_buckets} buckets, got {len(gt_le_pairs)}",
            )

        self.assertIn(f'model_name="{_MODEL_NAME}"', metrics_text)
        self.assertIn("_sum{", metrics_text)
        self.assertIn("_count{", metrics_text)
        self.assertIn("_bucket{", metrics_text)

        metrics_to_check = [
            ("sglang:realtime_tokens_total", {"mode": "prefill_compute"}),
            ("sglang:realtime_tokens_total", {"mode": "decode"}),
            ("sglang:gpu_execution_seconds_total", {"category": "forward_extend"}),
            ("sglang:gpu_execution_seconds_total", {"category": "forward_decode"}),
            ("sglang:process_cpu_seconds_total", {"component": "tokenizer"}),
        ]
        _check_metrics_positive(self, metrics, metrics_to_check)

        if expect_mfu_metrics:
            # Estimated perf metrics may have multiple series (e.g., by rank). Ensure
            # that at least one series for this model has a positive accumulated value.
            for metric_name in mfu_metrics:
                values = [
                    sample.value
                    for sample in metrics.get(metric_name, [])
                    if sample.labels.get("model_name") == _MODEL_NAME
                ]
                self.assertTrue(
                    values, f"{metric_name}: no samples for model {_MODEL_NAME}"
                )
                self.assertGreater(
                    sum(values),
                    0,
                    f"{metric_name}: expected positive total for model {_MODEL_NAME}",
                )
        else:
            # With only --enable-metrics (without --enable-mfu-metrics), MFU
            # counters should not emit positive values.
            for metric_name in mfu_metrics:
                values = [
                    sample.value
                    for sample in metrics.get(metric_name, [])
                    if sample.labels.get("model_name") == _MODEL_NAME
                ]
                if values:
                    self.assertEqual(
                        sum(values),
                        0,
                        f"{metric_name}: expected no positive samples with MFU metrics gate disabled",
                    )


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


def _get_sample_value_by_labels(samples: List[Sample], labels: Dict[str, str]) -> float:
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    raise KeyError(f"No sample found with labels {labels}")


def _check_metrics_positive(test_case, metrics, metrics_to_check):
    for metric_name, labels in metrics_to_check:
        value = _get_sample_value_by_labels(metrics[metric_name], labels)
        test_case.assertGreater(value, 0, f"{metric_name} {labels}")


class TestComputeRoutingKeyStats(unittest.TestCase):
    def test_empty(self):
        num_unique, req_counts = compute_routing_key_stats([])
        self.assertEqual(num_unique, 0)
        self.assertEqual(req_counts, [])

    def test_all_none(self):
        num_unique, req_counts = compute_routing_key_stats([None, None, None])
        self.assertEqual(num_unique, 0)
        self.assertEqual(req_counts, [])

    def test_with_none(self):
        num_unique, req_counts = compute_routing_key_stats([None, "key1", None])
        self.assertEqual(num_unique, 1)
        self.assertEqual(req_counts, [1])

    def test_single_key_multiple_reqs(self):
        num_unique, req_counts = compute_routing_key_stats(["key1"] * 5)
        self.assertEqual(num_unique, 1)
        self.assertEqual(req_counts, [5])

    def test_distribution(self):
        routing_keys = ["key1"] * 5 + ["key2"] * 1 + ["key3"] * 15 + ["key4"] * 250
        num_unique, req_counts = compute_routing_key_stats(routing_keys)
        self.assertEqual(num_unique, 4)
        self.assertEqual(sorted(req_counts), [1, 5, 15, 250])


class TestGetLoadsNumUsedTokens(unittest.TestCase):
    """Unit tests for SchedulerMetricsMixin.get_loads token accounting.

    num_used_tokens (reported to the router) must include queued tokens so a
    long backlog is not underestimated, while token_usage must stay based only
    on KV-cache occupancy so it remains normalized in [0, 1].
    """

    MAX_TOTAL = 1000
    KV_USED = 100  # KV-cache occupancy from _get_token_info()

    def _make_scheduler(self, waiting_queue):
        """Build a minimal mock scheduler for the non-hybrid `core`-only path."""
        stats = SimpleNamespace(
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=0.0,
        )
        return SimpleNamespace(
            dp_rank=0,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            _get_token_info=lambda: (self.KV_USED, 0, 0, 0),
            waiting_queue=waiting_queue,
            disaggregation_mode=DisaggregationMode.NULL,
            running_batch=SimpleNamespace(reqs=[]),
            max_total_num_tokens=self.MAX_TOTAL,
            max_running_requests=256,
            stats=stats,
        )

    def _call(self, waiting_queue):
        # Imported lazily via the managers.scheduler package first: importing
        # scheduler_metrics_mixin directly triggers a circular import
        # (mixin <-> managers.scheduler) at collection time. Loading the
        # scheduler package fully resolves the cycle before we grab the mixin.
        import sglang.srt.managers.scheduler  # noqa: F401
        from sglang.srt.observability.scheduler_metrics_mixin import (
            SchedulerMetricsMixin,
        )

        scheduler = self._make_scheduler(waiting_queue)
        req = GetLoadsReqInput(include=["core"])
        return SchedulerMetricsMixin.get_loads(scheduler, req)

    def test_empty_waiting_queue(self):
        out = self._call([])
        self.assertEqual(out.num_used_tokens, self.KV_USED)
        self.assertEqual(out.num_waiting_reqs, 0)
        self.assertEqual(out.token_usage, round(self.KV_USED / self.MAX_TOTAL, 4))

    def test_waiting_queue_counted_in_num_used_tokens(self):
        waiting = [SimpleNamespace(seqlen=30), SimpleNamespace(seqlen=70)]
        out = self._call(waiting)

        # Reported token count includes queued tokens (100 + 30 + 70).
        self.assertEqual(out.num_used_tokens, self.KV_USED + 100)
        self.assertEqual(out.num_waiting_reqs, 2)

    def test_token_usage_excludes_waiting_queue(self):
        # A backlog far larger than the KV cache must not push token_usage > 1.
        waiting = [SimpleNamespace(seqlen=self.MAX_TOTAL * 5)]
        out = self._call(waiting)

        # token_usage stays based only on KV occupancy → normalized in [0, 1].
        self.assertEqual(out.token_usage, round(self.KV_USED / self.MAX_TOTAL, 4))
        self.assertLessEqual(out.token_usage, 1.0)
        # ...but the raw count still reflects the backlog.
        self.assertEqual(out.num_used_tokens, self.KV_USED + self.MAX_TOTAL * 5)


if __name__ == "__main__":
    unittest.main()
