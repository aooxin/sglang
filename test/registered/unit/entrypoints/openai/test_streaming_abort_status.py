"""Scheduler rejection before the first chunk must be an HTTP error for PD routers."""

import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.responses import StreamingResponse

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def output(text="", finish_reason=None):
    return {
        "text": text,
        "output_ids": [],
        "meta_info": {
            "id": "test-request",
            "prompt_tokens": 10,
            "completion_tokens": int(bool(text)),
            "finish_reason": finish_reason,
            "input_token_logprobs": [],
            "output_token_logprobs": [],
            "output_token_logprobs_length": 0,
        },
    }


class TestStreamingAbortStatus(unittest.IsolatedAsyncioTestCase):
    def serving(self, kind, responses, logprobs=False, stream=True):
        if kind == "chat":
            handler = OpenAIServingChat.__new__(OpenAIServingChat)
            request = ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hello"}],
                stream=stream,
                logprobs=logprobs,
            )
            handler.tool_call_parser = None
            handler.reasoning_parser = None
            handler._toolcall_gate_active = Mock(return_value=False)
            handler._toolcall_special_id = None
        else:
            handler = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
            request = CompletionRequest(
                model="test",
                prompt="hello",
                stream=stream,
                logprobs=0 if logprobs else None,
            )

        async def generate():
            for response in responses:
                if isinstance(response, Exception):
                    raise response
                yield response

        handler.tokenizer_manager = SimpleNamespace(
            # Keep a reference so GC cannot hide missing explicit closure.
            generate_request=Mock(return_value=generate()),
            create_abort_task=Mock(return_value=AsyncMock()),
            request_logger=SimpleNamespace(log_requests=False),
            server_args=SimpleNamespace(
                enable_cache_report=False,
                incremental_streaming_output=False,
                stream_response_default_include_usage=False,
            ),
        )
        handler._validate_request = Mock(return_value=None)
        handler._convert_to_internal_request = Mock(
            return_value=(GenerateReqInput(stream=stream), request)
        )
        return handler, request

    async def test_initial_abort_returns_http_error_and_cleans_up(self):
        for kind in ("chat", "completion"):
            for code in (400, HTTPStatus.BAD_REQUEST, 503, None, "missing"):
                for logprobs in (False, True):
                    with self.subTest(kind=kind, code=code, logprobs=logprobs):
                        reason = {
                            "type": "abort",
                            "message": "Multimodal prompt is too long after expanding multimodal tokens.",
                        }
                        if code != "missing":
                            reason["status_code"] = code
                        content = output(finish_reason=reason)
                        # Scheduler validation failures may have no logprob data.
                        for key in list(content["meta_info"]):
                            if "logprob" in key:
                                del content["meta_info"][key]
                        handler, request = self.serving(
                            kind, [content], logprobs=logprobs
                        )
                        response = await handler.handle_request(request, None)
                        expected = 500 if code in (None, "missing") else int(code)
                        self.assertEqual(response.status_code, expected)
                        body = json.loads(response.body)
                        self.assertEqual(body["code"], expected)
                        self.assertEqual(body["message"], reason["message"])

                        manager = handler.tokenizer_manager
                        self.assertIsNone(
                            manager.generate_request.return_value.ag_frame
                        )
                        adapted = handler._convert_to_internal_request.return_value[0]
                        manager.create_abort_task.assert_called_once_with(adapted)
                        cleanup = manager.create_abort_task.return_value
                        cleanup.assert_not_awaited()
                        send = AsyncMock()
                        await response({"type": "http"}, AsyncMock(), send)
                        self.assertEqual(
                            send.call_args_list[0].args[0]["status"], expected
                        )
                        cleanup.assert_awaited_once()

    async def test_abort_after_first_chunk_remains_sse(self):
        for kind in ("chat", "completion"):
            with self.subTest(kind=kind):
                handler, request = self.serving(
                    kind,
                    [
                        output("hello"),
                        output(
                            "hello",
                            {
                                "type": "abort",
                                "status_code": 400,
                                "message": "cancelled",
                            },
                        ),
                    ],
                )
                response = await handler.handle_request(request, None)
                self.assertIsInstance(response, StreamingResponse)
                self.assertEqual(response.status_code, 200)
                chunks = [chunk async for chunk in response.body_iterator]
                events = [
                    json.loads(chunk.removeprefix("data: "))
                    for chunk in chunks
                    if chunk != "data: [DONE]\n\n"
                ]
                errors = [event["error"] for event in events if "error" in event]
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["code"], 400)
                self.assertEqual(errors[0]["message"], "cancelled")
                self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    async def test_tokenizer_validation_error_returns_http_400(self):
        for kind in ("chat", "completion"):
            for stream in (False, True):
                with self.subTest(kind=kind, stream=stream):
                    handler, request = self.serving(
                        kind, [ValueError("input too long")], stream=stream
                    )
                    response = await handler.handle_request(request, None)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        json.loads(response.body)["message"], "input too long"
                    )

    async def test_normal_stream_still_completes(self):
        for kind in ("chat", "completion"):
            with self.subTest(kind=kind):
                handler, request = self.serving(
                    kind,
                    [
                        output("hello"),
                        output("hello", {"type": "length", "length": 1}),
                    ],
                )
                response = await handler.handle_request(request, None)
                self.assertEqual(response.status_code, 200)
                chunks = [chunk async for chunk in response.body_iterator]
                self.assertIn("hello", "".join(chunks))
                self.assertNotIn('"error"', "".join(chunks))
                self.assertEqual(chunks[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
