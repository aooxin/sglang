"""Unit tests for the tower-free decode node's multimodal gate — no server.

A decode node on --enable-mm-context-forward loads no vision/audio tower and
never contacts the encoder: it takes the expanded multimodal input length from
the prefill node. Anything short of a hit is a request it cannot serve, because
there is no tower to preprocess with locally, so it must fail loudly rather than
fall back to work it cannot do.

Drives ``TokenizerManager._tokenize_one_request`` on a bare instance (built via
``__new__``) so no model, tokenizer, or ZMQ is required.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


def _make_tm(
    *,
    disaggregation_mode,
    mm_context_forward=False,
    language_only=False,
    backend="zmq_to_scheduler",
):
    """Bare TokenizerManager exposing only what _tokenize_one_request reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock(
        language_only=language_only,
        enable_mm_context_forward=mm_context_forward,
        encoder_transfer_backend=backend,
        disable_radix_cache=False,
        limit_mm_data_per_request=False,
        # Mirrors ServerArgs.has_local_vision_tower for the shapes under test.
        has_local_vision_tower=not (
            language_only or (mm_context_forward and disaggregation_mode == "decode")
        ),
    )
    tm.mm_processor = MagicMock()
    tm.disaggregation_mode = DisaggregationMode(disaggregation_mode)
    tm._enable_mm_context_forward = mm_context_forward
    # Mirrors TokenizerManager.init_disaggregation.
    tm._no_local_vision_tower = (
        tm.disaggregation_mode == DisaggregationMode.DECODE
        and not tm.server_args.has_local_vision_tower
    )
    tm.rid_to_state = {}
    tm._mm_context_tasks = set()
    tm.max_req_input_len = 4096
    tm._validate_one_request = MagicMock()
    tm._create_tokenized_object = MagicMock(return_value="tokenized")
    return tm


def _mm_request(*, bootstrap=True, input_ids=None):
    return GenerateReqInput(
        text="describe this image",
        image_data=["http://img/1.jpg"],
        input_ids=input_ids,
        bootstrap_host="D-host" if bootstrap else None,
        bootstrap_port=8998 if bootstrap else None,
        bootstrap_room=7 if bootstrap else None,
    )


class TestTowerFreeDecodeGate(unittest.IsolatedAsyncioTestCase):
    def _decode(self, **overrides):
        return _make_tm(
            disaggregation_mode="decode", mm_context_forward=True, **overrides
        )

    async def test_miss_raises(self):
        """No tower and no local fallback, so a miss cannot be served at all.

        Pinned across SGLANG_MM_CONTEXT_FORWARD_FALLBACK: a tower-free node has
        nothing to fall back to, so the env is no longer consulted. Its kept for
        compatibility only.
        """
        for fallback in (True, False):
            with self.subTest(SGLANG_MM_CONTEXT_FORWARD_FALLBACK=fallback):
                tm = self._decode()
                tm._fetch_mm_context = AsyncMock(return_value=None)
                with patch(
                    "sglang.srt.managers.tokenizer_manager."
                    "_MM_CONTEXT_FORWARD_FALLBACK",
                    fallback,
                ):
                    with self.assertRaisesRegex(ValueError, "no local vision tower"):
                        await tm._tokenize_one_request(_mm_request())

                tm.mm_processor.process_mm_data_async.assert_not_called()

    async def test_hit_uses_forwarded_length_without_local_work(self):
        tm = self._decode()
        tm._fetch_mm_context = AsyncMock(return_value=128)

        self.assertEqual(await tm._tokenize_one_request(_mm_request()), "tokenized")

        # input_ids is the placeholder of the forwarded length; nothing is loaded.
        _, input_ids = tm._validate_one_request.call_args[0]
        self.assertEqual(len(input_ids), 128)
        self.assertEqual(set(input_ids), {0})
        tm.mm_processor.process_mm_data_async.assert_not_called()

    async def test_missing_bootstrap_routing_is_also_rejected(self):
        """No bootstrap_host means no context lookup, so the request is unservable."""
        tm = self._decode()
        tm._fetch_mm_context = AsyncMock(return_value=None)

        with self.assertRaisesRegex(ValueError, "no local vision tower"):
            await tm._tokenize_one_request(
                _mm_request(bootstrap=False, input_ids=[1, 2, 3])
            )

    async def test_plain_decode_still_falls_back_locally(self):
        """Without the flag the tower is present, so behavior is unchanged."""
        tm = _make_tm(disaggregation_mode="decode")
        tm._fetch_mm_context = AsyncMock(return_value=None)
        tm.mm_processor.process_mm_data_async = AsyncMock(
            return_value=MagicMock(input_ids=[5, 6, 7], token_type_ids=None)
        )

        self.assertEqual(
            await tm._tokenize_one_request(
                _mm_request(bootstrap=False, input_ids=[1, 2, 3])
            ),
            "tokenized",
        )

        tm.mm_processor.process_mm_data_async.assert_awaited_once()

    async def test_prefill_language_only_is_not_gated(self):
        """Prefill owns the mm work; only the tower-free decode node loses it."""
        tm = _make_tm(
            disaggregation_mode="prefill",
            language_only=True,
            mm_context_forward=True,
            backend="zmq_to_tokenizer",
        )
        tm.mm_receiver = MagicMock()
        tm.mm_receiver.recv_mm_data = AsyncMock(
            return_value=MagicMock(input_ids=[9, 9], token_type_ids=None)
        )
        tm._push_mm_context = AsyncMock()
        obj = _mm_request(input_ids=[1, 2, 3])
        obj.need_wait_for_mm_inputs = True

        self.assertEqual(await tm._tokenize_one_request(obj), "tokenized")

        tm.mm_receiver.recv_mm_data.assert_awaited_once()
        # Pushed as a fire-and-forget task, so assert the call rather than the await.
        tm._push_mm_context.assert_called_once()
        await asyncio.sleep(0)  # let the created task finish before teardown


if __name__ == "__main__":
    unittest.main()
