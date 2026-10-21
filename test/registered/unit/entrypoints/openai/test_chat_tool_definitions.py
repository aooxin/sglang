"""Tool definitions in chat completion requests.

- Verbatim passthrough: chat templates render tool definitions with `tojson`, so
  a tool must serialize back to exactly the JSON the client sent (key order,
  unknown keys, no injected defaults), including custom (freeform) tools.
- Message-level tools: `tools` attached to individual non-assistant messages.
  Chat templates such as MiMo-V2 render a message's own `tools` inline (e.g. tool
  definitions discovered by tool search, attached to the tool result), so
  tool-call parsing, validation and constraints must see them alongside
  request-level tools.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _tool(name, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


BASH = _tool("bash", {"cmd": {"type": "string"}})
CALC = _tool("calc", {"expr": {"type": "string"}})
WEATHER = _tool(
    "get_weather", {"city": {"type": "string"}, "days": {"type": "integer"}}
)

WEATHER_CALL = (
    "<tool_call><function=get_weather><parameter=city>Beijing</parameter>"
    "<parameter=days>3</parameter></function></tool_call>"
)

# Keys deliberately not in pydantic field order, plus fields Function does not
# declare; "strict" and "description" are omitted to catch injected defaults.
VERBATIM_FUNCTION = {
    "type": "function",
    "function": {
        "name": "Bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        "defer_loading": True,
    },
    "cache_control": {"type": "ephemeral"},
}
VERBATIM_CUSTOM = {
    "type": "custom",
    "custom": {
        "name": "exec",
        "description": "Run JavaScript.",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
    },
}


def _dumps(obj):
    # Same serialization as the template's tojson: key order matters.
    return json.dumps(obj, ensure_ascii=False)


def _request(tools=(BASH,), **kwargs):
    return ChatCompletionRequest(
        model="x",
        tools=list(tools) or None,
        messages=[
            {"role": "user", "content": "weather?", "tools": [CALC]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd": "find"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "found",
                "tools": [WEATHER],
            },
        ],
        **kwargs,
    )


class TestToolDefinitionPassthrough(unittest.TestCase):
    def test_function_tool_serializes_verbatim(self):
        tool = Tool.model_validate(VERBATIM_FUNCTION)
        self.assertEqual(_dumps(tool.model_dump()), _dumps(VERBATIM_FUNCTION))
        # Parsed fields stay available, with defaults for omitted ones.
        self.assertEqual(tool.function.name, "Bash")
        self.assertFalse(tool.function.strict)
        self.assertIsNone(tool.function.description)

    def test_custom_tool_accepted_and_serializes_verbatim(self):
        tool = Tool.model_validate(VERBATIM_CUSTOM)
        self.assertEqual(_dumps(tool.model_dump()), _dumps(VERBATIM_CUSTOM))
        self.assertEqual(tool.type, "custom")
        self.assertEqual(tool.function.name, "exec")
        self.assertIsNone(tool.function.parameters)

    def test_request_and_message_tools_serialize_verbatim(self):
        req = ChatCompletionRequest(
            model="x",
            tools=[VERBATIM_FUNCTION, VERBATIM_CUSTOM],
            messages=[
                {"role": "user", "content": "hi", "tools": [VERBATIM_CUSTOM]},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "ok",
                    "tools": [VERBATIM_FUNCTION],
                },
            ],
        )
        dumped = req.model_dump()
        self.assertEqual(
            _dumps(dumped["tools"]), _dumps([VERBATIM_FUNCTION, VERBATIM_CUSTOM])
        )
        self.assertEqual(
            _dumps(dumped["messages"][0]["tools"]), _dumps([VERBATIM_CUSTOM])
        )
        self.assertEqual(
            _dumps(dumped["messages"][1]["tools"]), _dumps([VERBATIM_FUNCTION])
        )


class TestMessageLevelToolsProtocol(unittest.TestCase):
    def test_message_tools_keep_chat_completions_shape(self):
        req = _request()
        for msg in (req.messages[0], req.messages[2]):
            self.assertIsInstance(msg.tools[0], Tool)
        dumped = req.messages[2].model_dump()["tools"][0]
        self.assertEqual(dumped["type"], "function")
        self.assertEqual(dumped["function"]["name"], "get_weather")

    def test_tool_choice_defaults_to_auto_with_message_tools_only(self):
        self.assertEqual(_request(tools=()).tool_choice, "auto")
        plain = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertEqual(plain.tool_choice, "none")

    def test_get_all_tools_merges_in_order(self):
        req = _request()
        self.assertEqual(
            [t.function.name for t in req.get_all_tools()],
            ["bash", "calc", "get_weather"],
        )
        plain = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}], tools=[BASH]
        )
        self.assertIs(plain.get_all_tools(), plain.tools)

    def test_get_all_tools_later_definition_wins(self):
        redefined = _tool(
            "bash", {"cmd": {"type": "string"}, "timeout": {"type": "integer"}}
        )
        req = ChatCompletionRequest(
            model="x",
            tools=[BASH],
            messages=[{"role": "user", "content": "hi", "tools": [redefined]}],
        )
        (tool,) = req.get_all_tools()
        self.assertIn("timeout", tool.function.parameters["properties"])

    def test_assistant_message_tools_ignored(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok", "tools": [CALC]},
            ],
        )
        self.assertEqual(req.tool_choice, "none")
        self.assertIsNone(req.get_all_tools())


class TestToolDefinitionsServing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tm = Mock()
        tm.server_args = SimpleNamespace(
            tool_call_parser="mimo",
            reasoning_parser=None,
            context_length=None,
            allow_auto_truncate=False,
            enable_cache_report=False,
        )
        tm.model_config.get_default_sampling_params.return_value = None
        tm.model_config.hf_config.architectures = ["MiMoV2FlashForCausalLM"]
        tm.tokenizer.encode.return_value = []
        tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        template_manager = SimpleNamespace(
            chat_template_name=None, jinja_template_content_format="string"
        )
        self.chat = OpenAIServingChat(tm, template_manager)
        self.apply_chat_template = tm.tokenizer.apply_chat_template

    def test_template_receives_verbatim_tool_json(self):
        req = ChatCompletionRequest(
            model="x",
            tools=[VERBATIM_FUNCTION, VERBATIM_CUSTOM],
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "ok",
                    "tools": [VERBATIM_FUNCTION],
                },
            ],
        )
        self.assertIsNone(self.chat._validate_request(req))
        self.chat._process_messages(req, is_multimodal=False)

        call = self.apply_chat_template.call_args
        self.assertEqual(
            _dumps(call.kwargs["tools"]), _dumps([VERBATIM_FUNCTION, VERBATIM_CUSTOM])
        )
        self.assertEqual(_dumps(call.args[0][1]["tools"]), _dumps([VERBATIM_FUNCTION]))

    def test_custom_tool_call_parsed(self):
        req = ChatCompletionRequest(
            model="x",
            tools=[VERBATIM_CUSTOM],
            messages=[{"role": "user", "content": "hi"}],
        )
        finish_reason = {"type": "stop", "matched": None}
        code = "const r = await tools.Bash({command: 'ls'});"
        tool_calls, _, _ = self.chat._process_tool_calls(
            f"<tool_call><function=exec>{code}</function></tool_call>",
            req.get_all_tools(),
            finish_reason,
            "auto",
        )
        (call,) = tool_calls
        self.assertEqual((call.function.name, call.function.arguments), ("exec", code))

        # Custom tools have no parameters schema; parameter tags must not crash.
        tool_calls, _, _ = self.chat._process_tool_calls(
            "<tool_call><function=exec><parameter=code>1</parameter></function>"
            "</tool_call>",
            req.get_all_tools(),
            finish_reason,
            "auto",
        )
        self.assertEqual(json.loads(tool_calls[0].function.arguments), {"code": "1"})

    def test_template_gets_request_tools_and_messages_keep_theirs(self):
        req = _request()
        result = self.chat._process_messages(req, is_multimodal=False)

        call = self.apply_chat_template.call_args
        self.assertEqual(call.kwargs["tools"], [t.model_dump() for t in req.tools])
        messages = call.args[0]
        self.assertEqual(messages[0]["tools"][0]["function"]["name"], "calc")
        self.assertEqual(messages[2]["tools"][0]["function"]["name"], "get_weather")
        self.assertFalse(req.skip_special_tokens)
        self.assertIsNone(result.tool_call_constraint)

    def test_message_tools_only(self):
        req = _request(tools=())
        self.assertIsNone(self.chat._validate_request(req))
        self.chat._process_messages(req, is_multimodal=False)

        self.assertIsNone(self.apply_chat_template.call_args.kwargs["tools"])
        self.assertFalse(req.skip_special_tokens)

    def test_named_tool_choice_on_message_level_tool(self):
        req = _request(
            tool_choice={"type": "function", "function": {"name": "get_weather"}}
        )
        self.assertIsNone(self.chat._validate_request(req))
        result = self.chat._process_messages(req, is_multimodal=False)

        constraint_type, schema = result.tool_call_constraint
        self.assertEqual(constraint_type, "json_schema")
        self.assertIn("get_weather", json.dumps(schema))
        # Only the named tool may be rendered via the `tools` argument.
        self.assertIsNone(self.apply_chat_template.call_args.kwargs["tools"])

        missing = _request(
            tool_choice={"type": "function", "function": {"name": "missing"}}
        )
        self.assertIn("not found", self.chat._validate_request(missing))

    def test_invalid_message_level_tool_schema_rejected(self):
        bad = _tool("bad", {"x": {"type": "not-a-type"}})
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi", "tools": [bad]}],
        )
        self.assertIn("invalid 'parameters' schema", self.chat._validate_request(req))

    def test_non_stream_parses_message_level_tool_call(self):
        req = _request()
        ret = [
            {
                "text": WEATHER_CALL,
                "meta_info": {
                    "id": "chatcmpl-x",
                    "finish_reason": {"type": "stop", "matched": None},
                    "prompt_tokens": 3,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "weight_version": "default",
                },
            }
        ]
        response = self.chat._build_chat_response(req, ret, created=0)

        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        (call,) = choice.message.tool_calls
        self.assertEqual(call.function.name, "get_weather")
        self.assertEqual(
            json.loads(call.function.arguments), {"city": "Beijing", "days": 3}
        )

    async def test_stream_parses_message_level_tool_call(self):
        req = _request(stream=True)
        parser_dict, has_tool_calls, deltas = {}, {}, []
        for i in range(0, len(WEATHER_CALL), 7):
            async for chunk in self.chat._process_tool_call_stream(
                index=0,
                delta=WEATHER_CALL[i : i + 7],
                parser_dict=parser_dict,
                content={"meta_info": {"id": "chatcmpl-x"}},
                request=req,
                has_tool_calls=has_tool_calls,
            ):
                deltas.append(json.loads(chunk[len("data: ") :])["choices"][0]["delta"])

        self.assertFalse(any(d.get("content") for d in deltas))
        calls = [tc for d in deltas for tc in d.get("tool_calls") or []]
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        arguments = "".join(tc["function"].get("arguments") or "" for tc in calls)
        self.assertEqual(json.loads(arguments), {"city": "Beijing", "days": 3})


if __name__ == "__main__":
    unittest.main()
