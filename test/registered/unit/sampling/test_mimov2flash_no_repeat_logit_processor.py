"""Unit tests for MiMoV2FlashNoRepeatNGramLogitProcessor."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import FINISH_REPEAT_TRUNCATION
from sglang.srt.sampling.custom_logit_processor import (
    MiMoV2FlashNoRepeatNGramLogitProcessor,
)
from sglang.test.test_utils import CustomTestCase


class DummyReq:
    def __init__(self, output_ids, rid="req"):
        self.output_ids = list(output_ids)
        self.repetition_detected = False
        self.to_finish = None
        self.rid = rid


class DummySamplingBatchInfo:
    def __init__(
        self,
        custom_params,
        custom_logit_processor,
        custom_logit_processor_params=None,
        custom_logit_processor_indices=None,
        custom_logit_processor_indices_device=None,
    ):
        self.custom_params = custom_params
        self.custom_logit_processor = custom_logit_processor
        self.custom_logit_processor_params = custom_logit_processor_params
        self.custom_logit_processor_indices = custom_logit_processor_indices
        self.custom_logit_processor_indices_device = (
            custom_logit_processor_indices_device
        )

    def __len__(self):
        return len(self.custom_params)


class TestMiMoV2FlashNoRepeatNGramLogitProcessor(CustomTestCase):
    vocab_size = 128

    def _base_overrides(self):
        return dict(
            NGRAM_SIZE=3,
            WINDOW_SIZE=8,
            MIN_CONTENT_LENGTH=3,
            MIN_REPEAT=2,
            EOS_TOKEN_ID=99,
            DEBUG_SKIP_TRUNCATION=False,
        )

    def test_default_min_repeat_requires_an_actual_repeat(self):
        self.assertGreaterEqual(MiMoV2FlashNoRepeatNGramLogitProcessor.MIN_REPEAT, 2)

    def _run_processor(self, req, logits, overrides, req_count=1):
        params = {
            MiMoV2FlashNoRepeatNGramLogitProcessor.REQ_PARAM_KEY: req,
        }
        params_list = [params] * req_count
        with patch.multiple(MiMoV2FlashNoRepeatNGramLogitProcessor, **overrides):
            processor = MiMoV2FlashNoRepeatNGramLogitProcessor()
            return processor(logits, params_list)

    def _assert_logits_unchanged(self, original, result):
        """Assert that logits remain unchanged."""
        self.assertTrue(torch.equal(original, result))

    def _assert_skip_conditions(
        self, output_ids, overrides=None, expected_detected=False, expected_finish=None
    ):
        """Assert that processing is skipped (logits unchanged)."""
        if overrides is None:
            overrides = self._base_overrides()
        logits = torch.randn((1, self.vocab_size))
        req = DummyReq(output_ids)
        output = self._run_processor(req, logits.clone(), overrides)
        self._assert_logits_unchanged(logits, output)
        self.assertEqual(req.repetition_detected, expected_detected)
        self.assertEqual(req.to_finish, expected_finish)

    def _assert_force_stop(self, output_ids, overrides=None):
        """Assert that repetition is detected and EOS token is forced."""
        if overrides is None:
            overrides = self._base_overrides()
        logits = torch.zeros((1, self.vocab_size))
        req = DummyReq(output_ids)
        output = self._run_processor(req, logits, overrides)

        self.assertTrue(req.repetition_detected)
        self.assertIsInstance(req.to_finish, FINISH_REPEAT_TRUNCATION)
        eos_id = overrides["EOS_TOKEN_ID"]
        self.assertEqual(output[0, eos_id].item(), 0.0)
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        mask[eos_id] = False
        self.assertTrue(torch.isinf(output[0][mask]).all())
        self.assertTrue((output[0][mask] < 0).all())

    def test_invalid_config_skips(self):
        """Test that invalid configurations (NGRAM_SIZE <= 1 or WINDOW_SIZE <= 0) skip processing."""
        self._assert_skip_conditions(
            [1, 2, 3, 1, 2, 3],
            overrides={**self._base_overrides(), "NGRAM_SIZE": 1, "WINDOW_SIZE": 0},
        )

    def test_window_smaller_than_ngram_skips(self):
        """Test that WINDOW_SIZE < NGRAM_SIZE skips processing."""
        self._assert_skip_conditions(
            [1, 2, 3, 4, 1, 2, 3, 4],
            overrides={**self._base_overrides(), "NGRAM_SIZE": 4, "WINDOW_SIZE": 3},
        )

    def test_min_content_length_skip(self):
        """Test that sequences shorter than MIN_CONTENT_LENGTH are skipped."""
        self._assert_skip_conditions(
            [1, 2, 3, 4, 5, 6],
            overrides={**self._base_overrides(), "MIN_CONTENT_LENGTH": 10},
        )

    def test_seq_len_shorter_than_ngram(self):
        """Test that sequences shorter than NGRAM_SIZE are skipped."""
        self._assert_skip_conditions(
            [1, 2, 3],
            overrides={
                **self._base_overrides(),
                "MIN_CONTENT_LENGTH": 1,
                "NGRAM_SIZE": 4,
            },
        )

    def test_min_repeat_threshold(self):
        """Test that repetitions below MIN_REPEAT threshold are not triggered."""
        self._assert_skip_conditions(
            [1, 2, 3, 1, 2, 3, 4, 5],
            overrides={**self._base_overrides(), "MIN_REPEAT": 3},
        )

    def test_repeat_outside_window_skips(self):
        """Test that repetitions outside the sliding window are ignored."""
        self._assert_skip_conditions(
            [1, 2, 3, 7, 8, 9, 1, 2, 3],
            overrides={**self._base_overrides(), "WINDOW_SIZE": 3, "MIN_REPEAT": 2},
        )

    def test_debug_skip_truncation_on_ngram_repeat(self):
        """Test that DEBUG_SKIP_TRUNCATION=True detects repetition but does not truncate."""
        overrides = self._base_overrides()
        overrides["DEBUG_SKIP_TRUNCATION"] = True
        logits = torch.randn((1, self.vocab_size))
        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5])
        output = self._run_processor(req, logits.clone(), overrides)
        self._assert_logits_unchanged(logits, output)
        self.assertTrue(req.repetition_detected)
        self.assertIsNone(req.to_finish)

    def test_cache_reset_on_seq_shrink(self):
        """Test that cache is properly reset when sequence length decreases."""
        overrides = {
            **self._base_overrides(),
            "DEBUG_SKIP_TRUNCATION": True,
            "MIN_REPEAT": 3,
        }
        logits = torch.randn((1, self.vocab_size))
        req = DummyReq([1, 2, 3, 4, 5, 6, 7])
        output = self._run_processor(req, logits.clone(), overrides)
        self._assert_logits_unchanged(logits, output)
        self.assertFalse(req.repetition_detected)

        # Shrink the sequence to trigger cache reset.
        req.output_ids = [1, 2, 3]
        output = self._run_processor(req, logits.clone(), overrides)
        self._assert_logits_unchanged(logits, output)
        self.assertFalse(req.repetition_detected)

    def test_cached_action_reuse_on_ngram_repeat(self):
        """Test that cached action is reused when the same request appears multiple times in a batch."""
        overrides = self._base_overrides()
        req_count = 2
        logits = torch.zeros((req_count, self.vocab_size))
        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5])
        output = self._run_processor(req, logits.clone(), overrides, req_count)

        self.assertTrue(req.repetition_detected)
        self.assertIsInstance(req.to_finish, FINISH_REPEAT_TRUNCATION)
        eos_id = overrides["EOS_TOKEN_ID"]
        for row in range(req_count):
            self.assertEqual(output[row, eos_id].item(), 0.0)
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            mask[eos_id] = False
            self.assertTrue(torch.isinf(output[row][mask]).all())

    def test_force_stop_on_ngram_repeat(self):
        """Test that n-gram repetition triggers force stop with EOS token."""
        self._assert_force_stop([1, 2, 3, 1, 2, 3, 4, 5])

    def test_sampler_compact_params_with_spec_tokens(self):
        """Test sampler passes compact params aligned with masked logits rows."""
        overrides = self._base_overrides()
        draft_token_num = 2
        batch_size = 3
        logits = torch.arange(
            batch_size * draft_token_num * self.vocab_size, dtype=torch.float
        ).reshape(batch_size * draft_token_num, self.vocab_size)
        original = logits.clone()

        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5], rid="masked-req")
        custom_params = [
            None,
            {MiMoV2FlashNoRepeatNGramLogitProcessor.REQ_PARAM_KEY: req},
            None,
        ]
        batch_mask = torch.tensor([False, True, False])

        with patch.multiple(MiMoV2FlashNoRepeatNGramLogitProcessor, **overrides):
            processor = MiMoV2FlashNoRepeatNGramLogitProcessor()
            sampling_info = DummySamplingBatchInfo(
                custom_params=custom_params,
                custom_logit_processor={1: (processor, batch_mask)},
            )
            apply_custom_logit_processor(
                logits, sampling_info, num_tokens_in_batch=draft_token_num
            )

        changed_rows = [2, 3]
        unchanged_rows = [0, 1, 4, 5]
        self.assertTrue(torch.equal(logits[unchanged_rows], original[unchanged_rows]))
        eos_id = overrides["EOS_TOKEN_ID"]
        for row in changed_rows:
            self.assertEqual(logits[row, eos_id].item(), 0.0)
            mask = torch.ones(self.vocab_size, dtype=torch.bool)
            mask[eos_id] = False
            self.assertTrue(torch.isinf(logits[row][mask]).all())
            self.assertTrue((logits[row][mask] < 0).all())

    def test_sampler_uses_cached_processor_params_without_nonzero(self):
        """Test sampler avoids mask.nonzero() when compact params metadata exists."""
        overrides = self._base_overrides()
        draft_token_num = 2
        batch_size = 3
        logits = torch.arange(
            batch_size * draft_token_num * self.vocab_size, dtype=torch.float
        ).reshape(batch_size * draft_token_num, self.vocab_size)
        original = logits.clone()

        req = DummyReq([1, 2, 3, 1, 2, 3, 4, 5], rid="masked-req")
        params = {MiMoV2FlashNoRepeatNGramLogitProcessor.REQ_PARAM_KEY: req}
        custom_params = [None, params, None]
        batch_mask = torch.tensor([False, True, False])

        with patch.multiple(MiMoV2FlashNoRepeatNGramLogitProcessor, **overrides):
            processor = MiMoV2FlashNoRepeatNGramLogitProcessor()
            sampling_info = DummySamplingBatchInfo(
                custom_params=custom_params,
                custom_logit_processor={1: (processor, batch_mask)},
                custom_logit_processor_params={1: [params]},
                custom_logit_processor_indices={1: [1]},
                custom_logit_processor_indices_device={1: torch.tensor([1])},
            )
            with patch.object(
                torch.Tensor,
                "nonzero",
                side_effect=AssertionError("unexpected nonzero"),
            ):
                apply_custom_logit_processor(
                    logits, sampling_info, num_tokens_in_batch=draft_token_num
                )

        self.assertTrue(torch.equal(logits[[0, 1, 4, 5]], original[[0, 1, 4, 5]]))
        eos_id = overrides["EOS_TOKEN_ID"]
        for row in [2, 3]:
            self.assertEqual(logits[row, eos_id].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
