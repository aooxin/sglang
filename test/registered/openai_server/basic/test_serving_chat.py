"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import json
import unittest
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )
        # Mock hf_config for _use_dpsk_v32_encoding check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_jinja_uses_openai_tool_schema_first(self):
        """Ensure Jinja chat templates receive OpenAI-shaped tools by default."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        expected_tools = [tool.model_dump() for tool in req.tools]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], expected_tools)

    def test_jinja_tool_schema_fallback_to_flat_function(self):
        """Fallback to function-only schema when template rejects OpenAI wrapper."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.tm.tokenizer.apply_chat_template.side_effect = [
            RuntimeError("template expects flat tools format"),
            [1, 2, 3],
        ]

        self.chat._process_messages(req, is_multimodal=False)

        first_tools = self.tm.tokenizer.apply_chat_template.call_args_list[0].kwargs[
            "tools"
        ]
        second_tools = self.tm.tokenizer.apply_chat_template.call_args_list[1].kwargs[
            "tools"
        ]
        self.assertEqual(first_tools, [tool.model_dump() for tool in req.tools])
        self.assertEqual(
            second_tools, [tool.function.model_dump() for tool in req.tools]
        )

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")

        # Parse the result to verify content
        self.assertTrue(result.startswith("data: "))
        chunk = json.loads(result[6:])
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        arguments = tool_calls[0]["function"]["arguments"]
        self.assertIn(', "unit": "celsius"}', arguments)

        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since there's no parser data
        self.assertIsNone(
            result, "Should return None when parser has no tool call data"
        )

    # ------------- kimi_k2 tool_call_id formatting -------------
    def test_kimi_k2_non_streaming_tool_call_id_format(self):
        """Ensure non-streaming tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Paris"}'
            call_info.tool_index = 0

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            tool_calls, remaining_text, finish_reason = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:0")
            self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_format(self):
        """Ensure streaming first chunk tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tools
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:0")

    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA and Tokyo?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=False,
        )

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Loa Angeles"}'
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            call_info.tool_index = 0

            call_info2 = Mock()
            call_info2.name = "get_weather"
            call_info2.parameters = '{"city":"Tokyo"}'
            call_info2.tool_index = 1

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = (
                "",
                [call_info, call_info2],
            )

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            history_tool_calls_cnt = self.chat._get_history_tool_calls_cnt(req)
            tool_calls, remaining_text, _ = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
                history_tool_calls_cnt=history_tool_calls_cnt,
            )

            self.assertEqual(history_tool_calls_cnt, 1)
            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:1")
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(tool_calls[1].id, "functions.get_weather:2")
            self.assertEqual(tool_calls[1].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_with_history(self):
        """Ensure streaming first chunk tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:1")

    def test_dpsk_v32_encoding_path(self):
        """Test DeepSeek V3.2 encoding path detection and application."""
        from sglang.srt.managers.template_manager import TemplateManager
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(model_path="deepseek-ai/DeepSeek-V3.2")
        port_args = PortArgs.init_new(server_args)

        # Use mocks for TokenizerManager components to avoid full initialization
        with patch(
            "sglang.srt.managers.tokenizer_manager.TokenizerManager"
        ) as MockTokenizerManager:
            tokenizer_manager = MockTokenizerManager(server_args, port_args)
            tokenizer_manager.server_args = server_args
            tokenizer_manager.model_config = Mock()
            tokenizer_manager.model_config.get_default_sampling_params.return_value = (
                None
            )

            # Mock hf_config
            mock_hf_config = Mock()
            mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]

            tokenizer_manager.model_config.hf_config = mock_hf_config

            # Case 1: No chat template in tokenizer -> should use dpsk encoding
            tokenizer_manager.tokenizer = Mock()
            tokenizer_manager.tokenizer.chat_template = None

            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertTrue(serving_chat.use_dpsk_v32_encoding)

            # Case 2: Chat template exists -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = "some template"
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

            # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = None
            mock_hf_config.architectures = ["LlamaForCausalLM"]
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort():
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate_abort()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                try:
                    async for chunk in self.chat._generate_chat_stream(
                        adapted_request, req, self.fastapi_request
                    ):
                        chunks.append(chunk)
                except Exception as e:
                    print(f"Error during stream iteration: {e}")
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk
        self.assertEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    # ------------- X-Data-Parallel-Rank header tests -------------
    def test_extract_routed_dp_rank_from_header_no_header(self):
        """Test that None is returned when no header is present."""
        self.fastapi_request.headers = {}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertIsNone(result)

    def test_extract_routed_dp_rank_from_header_with_header(self):
        """Test that header value is extracted correctly."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "2"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertEqual(result, 2)

    def test_extract_routed_dp_rank_header_overrides_body(self):
        """Test that header value has higher priority than body."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "3"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=1
        )
        self.assertEqual(result, 3)  # header wins

    def test_extract_routed_dp_rank_from_header_invalid(self):
        """Test that invalid header value raises HTTPException."""
        from fastapi import HTTPException

        self.fastapi_request.headers = {"x-data-parallel-rank": "abc"}
        with self.assertRaises(HTTPException) as context:
            self.chat.extract_routed_dp_rank_from_header(
                self.fastapi_request, body_routed_dp_rank=None
            )
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be an integer", context.exception.detail)


class _FakeSpecialAwareTokenizer:
    """Base fake tokenizer that models the added-token fast-path.

    Subclasses provide `_SPECIALS` (str -> single id). Behavior:
    - an already-present special-token *string* collapses into its single id
      (the fast path that discretization must bypass);
    - any other text -> one id per character (deterministic, reversible);
    - decode reverses everything, so round-trips can assert losslessness.
    """

    _SPECIALS: dict = {}
    _BASE = 256  # char ids offset so they never collide with special ids
    chat_template = None

    @property
    def _id_to_str(self):
        return {v: k for k, v in self._SPECIALS.items()}

    def encode(self, text, add_special_tokens=True):
        # Note: an already-present special-token string collapses REGARDLESS of
        # add_special_tokens (that flag only controls auto-added BOS/EOS) --
        # exactly why the fix discretizes via "<" + inner + ">".
        ids = []
        i = 0
        n = len(text)
        while i < n:
            hit = next((s for s in self._SPECIALS if text.startswith(s, i)), None)
            if hit:
                ids.append(self._SPECIALS[hit])
                i += len(hit)
            else:
                ids.append(self._BASE + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, **kwargs):
        id_to_str = self._id_to_str
        out = []
        for tid in ids:
            if tid in id_to_str:
                out.append(id_to_str[tid])
            else:
                out.append(chr(tid - self._BASE))
        return "".join(out)

    def convert_tokens_to_ids(self, tok):
        return self._SPECIALS.get(tok, -1)

    def convert_ids_to_tokens(self, tid):
        return self._id_to_str.get(tid)


class _FakeToolCallTokenizer(_FakeSpecialAwareTokenizer):
    """Models the <tool_call>/</tool_call> added-token fast-path."""

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"
    OPEN_ID = 151657
    CLOSE_ID = 151658
    _SPECIALS = {OPEN: OPEN_ID, CLOSE: CLOSE_ID}


class DiscretizeUserToolCallTagsTestCase(unittest.TestCase):
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.tm.tokenizer = _FakeToolCallTokenizer()
        self.template_manager = _MockTemplateManager()
        # These tests exercise tool-call tag discretization, so construct the
        # handler with the tool-call switch ON. The switch is read once in
        # __init__ and the target set is cached there, so the override must wrap
        # construction.
        with envs.SGLANG_DISCRETIZE_USER_TOOLCALL_TAGS.override(True):
            self.chat = OpenAIServingChat(self.tm, self.template_manager)

    def _count(self, ids, token_id):
        return ids.count(token_id)

    def test_segment_discretizes_open_and_close_tags(self):
        seg = "<tool_call>x</tool_call>"
        ids = self.chat._encode_segment_discretizing_tags(seg)
        self.assertNotIn(_FakeToolCallTokenizer.OPEN_ID, ids)
        self.assertNotIn(_FakeToolCallTokenizer.CLOSE_ID, ids)
        # lossless: decode reproduces the original segment
        self.assertEqual(self.tm.tokenizer.decode(ids), seg)

    def test_segment_without_tags_is_unchanged(self):
        seg = "plain text no tags"
        ids = self.chat._encode_segment_discretizing_tags(seg)
        self.assertEqual(ids, self.tm.tokenizer.encode(seg))

    def test_locate_user_spans_only_user_role(self):
        text = "SYS<tool_call>USERPART tail"
        messages = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "USERPART"},
        ]
        spans = self.chat._locate_user_content_spans(text, messages)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(text[start:end], "USERPART")

    def test_locate_user_spans_missing_content_skipped(self):
        # content not present verbatim (e.g. multimodal rewrite) -> skipped
        text = "rendered without the raw user content"
        messages = [{"role": "user", "content": "NOT_IN_TEXT"}]
        self.assertEqual(self.chat._locate_user_content_spans(text, messages), [])

    def test_end_to_end_only_user_tag_discretized(self):
        # Structure carries one <tool_call> (kept as special), user carries one
        # (must be discretized). Result: exactly one special id remains.
        text = "STRUCT<tool_call>MID<tool_call>END"
        messages = [{"role": "user", "content": "<tool_call>"}]
        # user content "<tool_call>" first appears at index 6 (after STRUCT).
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        # one structural <tool_call> survives, the user one is discretized
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 1)
        # lossless overall
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_no_user_message_encodes_whole_string(self):
        text = "STRUCT<tool_call>END"
        messages = [{"role": "system", "content": "s"}]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        # no user span -> whole string encoded normally -> special preserved
        self.assertEqual(ids, self.tm.tokenizer.encode(text))
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 1)

    def test_render_switch_off_uses_apply_chat_template(self):
        self.tm.tokenizer.apply_chat_template = Mock(return_value=[9, 9, 9])
        self.chat._discretize_user_toolcall_tags = False
        # The target set is cached in __init__; recompute after mutating the
        # switch so the render path reflects the new state.
        self.chat._cached_targets = self.chat._compute_discretize_targets()
        out = self.chat._render_prompt_ids(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            template_kwargs={},
        )
        self.assertEqual(out, [9, 9, 9])
        _, kwargs = self.tm.tokenizer.apply_chat_template.call_args
        self.assertTrue(kwargs["tokenize"])

    def test_render_switch_on_renders_then_discretizes(self):
        self.tm.tokenizer.apply_chat_template = Mock(return_value="AA<tool_call>BB")
        self.chat._discretize_user_toolcall_tags = True
        # The target set is cached in __init__; recompute after mutating the
        # switch so the render path reflects the new state.
        self.chat._cached_targets = self.chat._compute_discretize_targets()
        out = self.chat._render_prompt_ids(
            messages=[{"role": "user", "content": "<tool_call>"}],
            tools=None,
            template_kwargs={},
        )
        # rendered as string (tokenize=False), then user tag discretized
        _, kwargs = self.tm.tokenizer.apply_chat_template.call_args
        self.assertFalse(kwargs["tokenize"])
        self.assertNotIn(_FakeToolCallTokenizer.OPEN_ID, out)
        self.assertEqual(self.tm.tokenizer.decode(out), "AA<tool_call>BB")

    def test_locate_user_spans_multi_turn_returns_each_user_in_order(self):
        text = "sys u1=<tool_call> a1 u2=<tool_call> tail"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1=<tool_call>"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2=<tool_call>"},
        ]
        spans = self.chat._locate_user_content_spans(text, messages)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0] : spans[0][1]], "u1=<tool_call>")
        self.assertEqual(text[spans[1][0] : spans[1][1]], "u2=<tool_call>")
        self.assertLess(spans[0][1], spans[1][0])

    def test_multi_turn_all_user_tags_discretized_assistant_kept(self):
        # Two user turns each carry <tool_call> (both must be discretized); an
        # assistant turn in between also carries <tool_call> (must be kept as the
        # special id, since discretization only targets user content).
        text = "<tool_call>ASSISTANT<tool_call>"
        messages = [
            {"role": "user", "content": "<tool_call>"},
            {"role": "assistant", "content": "ASSISTANT<tool_call>"},
        ]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        # exactly the assistant-turn <tool_call> survives as a single special id
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 1)
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_multi_turn_two_user_turns_both_discretized(self):
        # Both user turns must be discretized; no user <tool_call> stays special.
        text = "<tool_call>MID<tool_call>"
        messages = [
            {"role": "user", "content": "<tool_call>"},
            {"role": "user", "content": "<tool_call>"},
        ]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 0)
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_multi_turn_identical_user_content_located_separately(self):
        # Identical repeated user content: search_from advances so the second
        # occurrence is located separately (not collapsed onto the first).
        text = "SAME between SAME"
        messages = [
            {"role": "user", "content": "SAME"},
            {"role": "user", "content": "SAME"},
        ]
        spans = self.chat._locate_user_content_spans(text, messages)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0], (0, 4))
        self.assertEqual(spans[1], (13, 17))

    def test_locate_spans_include_tool_role(self):
        text = "SYS<tool_call>TOOLRESULT tail"
        messages = [
            {"role": "system", "content": "ignored"},
            {"role": "tool", "content": "TOOLRESULT"},
        ]
        spans = self.chat._locate_user_content_spans(text, messages)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(text[start:end], "TOOLRESULT")

    def test_end_to_end_tool_tag_discretized(self):
        # Structure carries one <tool_call> (kept as special); a tool result
        # carries one (must be discretized). Result: exactly one special remains.
        text = "STRUCT<tool_call>MID<tool_call>END"
        messages = [{"role": "tool", "content": "<tool_call>"}]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 1)
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_locate_spans_mixed_user_and_tool_in_order(self):
        text = "sys u=<tool_call> a1 t=<tool_call> tail"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u=<tool_call>"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t=<tool_call>"},
        ]
        spans = self.chat._locate_user_content_spans(text, messages)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0] : spans[0][1]], "u=<tool_call>")
        self.assertEqual(text[spans[1][0] : spans[1][1]], "t=<tool_call>")
        self.assertLess(spans[0][1], spans[1][0])

    def test_mixed_user_and_tool_tags_discretized_assistant_kept(self):
        # User and tool turns each carry <tool_call> (both must be discretized);
        # an assistant turn in between also carries <tool_call> (kept special,
        # since discretization only targets untrusted-input roles).
        text = "<tool_call>ASSISTANT<tool_call>TOOL<tool_call>"
        messages = [
            {"role": "user", "content": "<tool_call>"},
            {"role": "assistant", "content": "ASSISTANT<tool_call>"},
            {"role": "tool", "content": "TOOL<tool_call>"},
        ]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        # only the assistant-turn <tool_call> survives as a single special id
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.OPEN_ID), 1)
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_tool_close_tag_discretized(self):
        text = "STRUCT</tool_call>TOOL</tool_call>END"
        messages = [{"role": "tool", "content": "</tool_call>"}]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        self.assertEqual(self._count(ids, _FakeToolCallTokenizer.CLOSE_ID), 1)
        self.assertEqual(self.tm.tokenizer.decode(ids), text)


class ToolCallSpecialTokenGateTestCase(unittest.TestCase):
    OPEN_ID = _FakeToolCallTokenizer.OPEN_ID

    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.tm.server_args.tool_call_parser = "mimo"
        self.tm.tokenizer.convert_tokens_to_ids = Mock(return_value=self.OPEN_ID)
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

    @contextmanager
    def _enable(self, on):
        # The switch is read once in __init__ and cached, so reconstruct the
        # instance under the overridden env var to exercise the real read path.
        with envs.SGLANG_REQUIRE_SPECIAL_TOOLCALL_TOKEN.override(on):
            prev = self.chat
            self.chat = OpenAIServingChat(self.tm, self.template_manager)
            try:
                yield
            finally:
                self.chat = prev

    def test_special_id_resolved_for_mimo(self):
        self.assertEqual(self.chat._toolcall_special_id, self.OPEN_ID)

    def test_special_id_none_for_non_mimo(self):
        self.tm.server_args.tool_call_parser = "qwen25"
        chat = OpenAIServingChat(self.tm, self.template_manager)
        self.assertIsNone(chat._toolcall_special_id)

    def test_gate_off_never_blocks(self):
        with self._enable(False):
            self.assertFalse(self.chat._toolcall_gate_blocks([1, 2, 3]))

    def test_gate_on_blocks_when_special_id_absent(self):
        # discrete tokens (no 151657) -> gate blocks parsing
        with self._enable(True):
            self.assertTrue(self.chat._toolcall_gate_blocks([27, 14172, 13429, 29]))

    def test_gate_on_allows_when_special_id_present(self):
        with self._enable(True):
            self.assertFalse(self.chat._toolcall_gate_blocks([1, self.OPEN_ID, 2]))

    def test_gate_on_fails_open_when_no_output_ids(self):
        with self._enable(True):
            self.assertFalse(self.chat._toolcall_gate_blocks(None))
            self.assertFalse(self.chat._toolcall_gate_blocks([]))

    def test_gate_on_fails_open_for_non_mimo(self):
        self.tm.server_args.tool_call_parser = "qwen25"
        with self._enable(True):
            # id unavailable -> cannot judge -> fail open
            self.assertFalse(self.chat._toolcall_gate_blocks([27, 14172]))

    def test_gate_blocks_injected_toolcall_but_allows_genuine(self):
        # The non-streaming path gates BEFORE calling _process_tool_calls, so
        # the blocking decision lives entirely in _toolcall_gate_blocks.
        with self._enable(True):
            # injected discrete ids (no genuine OPEN_ID) -> blocked
            self.assertTrue(self.chat._toolcall_gate_blocks([27, 14172, 13429, 29]))
            # genuine call carries the single OPEN_ID -> allowed
            self.assertFalse(self.chat._toolcall_gate_blocks([27, self.OPEN_ID, 29]))


class _FakeSpecialTokenTokenizer(_FakeSpecialAwareTokenizer):
    """Models the <|im_end|>/<|im_start|> added-token fast-path (MiMo/Qwen)."""

    IM_END = "<|im_end|>"
    IM_START = "<|im_start|>"
    IM_END_ID = 151645
    IM_START_ID = 151644
    _SPECIALS = {IM_END: IM_END_ID, IM_START: IM_START_ID}


class DiscretizeUserSpecialTokensTestCase(unittest.TestCase):
    """Covers SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS: end-of-turn/structural
    special tokens inside user content must not be collapsed into their EOS id."""

    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.tm.tokenizer = _FakeSpecialTokenTokenizer()
        # Model EOS set includes <|im_end|> so it is reverse-mapped and activated.
        self.tm.model_config.hf_eos_token_id = {
            _FakeSpecialTokenTokenizer.IM_END_ID
        }
        self.template_manager = _MockTemplateManager()
        with envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS.override(True):
            self.chat = OpenAIServingChat(self.tm, self.template_manager)

    def _count(self, ids, token_id):
        return ids.count(token_id)

    def test_special_tokens_activated_at_init(self):
        # Both the candidate list and the model EOS reverse-map should activate
        # <|im_end|> and <|im_start|>.
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_END, self.chat._special_token_strs
        )
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_START, self.chat._special_token_strs
        )

    def test_segment_discretizes_im_end(self):
        seg = "hello <|im_end|> world"
        ids = self.chat._encode_segment_discretizing_tags(seg)
        self.assertNotIn(_FakeSpecialTokenTokenizer.IM_END_ID, ids)
        # lossless: decode reproduces the original segment verbatim
        self.assertEqual(self.tm.tokenizer.decode(ids), seg)

    def test_end_to_end_user_special_discretized_structure_kept(self):
        # Template structure carries a real <|im_end|> (kept as the EOS id); the
        # user content carries one (must be discretized). Exactly one survives.
        text = "STRUCT<|im_end|>MID<|im_end|>END"
        messages = [{"role": "user", "content": "<|im_end|>"}]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        self.assertEqual(
            self._count(ids, _FakeSpecialTokenTokenizer.IM_END_ID), 1
        )
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_content_after_injected_special_is_preserved(self):
        # Regression for the reported bug: user text "A<|im_end|>C" must keep C
        # (no early cut), and no user-side EOS id remains.
        text = "A<|im_end|>C"
        messages = [{"role": "user", "content": "A<|im_end|>C"}]
        ids = self.chat._encode_with_user_tags_discretized(text, messages)
        self.assertEqual(
            self._count(ids, _FakeSpecialTokenTokenizer.IM_END_ID), 0
        )
        self.assertEqual(self.tm.tokenizer.decode(ids), text)

    def test_switch_off_leaves_special_id(self):
        # With the switch off, no special tokens are activated and the direct
        # segment encoder collapses <|im_end|> into its EOS id (baseline bug).
        with envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS.override(False):
            chat = OpenAIServingChat(self.tm, self.template_manager)
        self.assertEqual(chat._special_token_strs, ())

    def test_render_switch_on_special_tokens_discretizes(self):
        self.tm.tokenizer.apply_chat_template = Mock(
            return_value="AA<|im_end|>BB"
        )
        out = self.chat._render_prompt_ids(
            messages=[{"role": "user", "content": "<|im_end|>"}],
            tools=None,
            template_kwargs={},
        )
        _, kwargs = self.tm.tokenizer.apply_chat_template.call_args
        self.assertFalse(kwargs["tokenize"])
        self.assertNotIn(_FakeSpecialTokenTokenizer.IM_END_ID, out)
        self.assertEqual(self.tm.tokenizer.decode(out), "AA<|im_end|>BB")

    def test_candidate_list_defaults_to_env_default(self):
        # With the default candidate env var, both structural tokens activate.
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_END, self.chat._special_token_strs
        )
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_START, self.chat._special_token_strs
        )

    def test_candidate_list_configurable_via_env(self):
        # A custom comma-separated list is honored: only listed tokens (that the
        # tokenizer maps to a single id) get activated. Here we drop <|im_start|>
        # from the candidate list (model EOS only contributes <|im_end|>).
        with envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS.override(True), (
            envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKEN_STRS.override("<|im_end|>")
        ):
            chat = OpenAIServingChat(self.tm, self.template_manager)
        # <|im_end|> is still activated (single special id) ...
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_END, chat._special_token_strs
        )
        # ... but <|im_start|> is NOT, since it was excluded from the list.
        self.assertNotIn(
            _FakeSpecialTokenTokenizer.IM_START, chat._special_token_strs
        )

    def test_empty_candidate_list_still_covers_model_eos(self):
        # Even with an empty candidate env value, the model's own EOS tokens
        # (reverse-mapped from hf_eos_token_id) are always included, so the EOS
        # token still gets discretized inside user content.
        with envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS.override(True), (
            envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKEN_STRS.override("")
        ):
            chat = OpenAIServingChat(self.tm, self.template_manager)
        # <|im_end|> is the model EOS here, so it is still activated;
        # <|im_start|> is not (not a model EOS and not in the empty list).
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_END, chat._special_token_strs
        )
        self.assertNotIn(
            _FakeSpecialTokenTokenizer.IM_START, chat._special_token_strs
        )

    def test_targets_cached_and_nonempty(self):
        # The target set is precomputed once in __init__ and non-empty here
        # (special-token discretization is on with model EOS activated).
        self.assertTrue(self.chat._cached_targets)
        self.assertIn(
            _FakeSpecialTokenTokenizer.IM_END, self.chat._cached_targets
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
