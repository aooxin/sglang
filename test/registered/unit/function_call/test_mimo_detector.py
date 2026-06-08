import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name, properties):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"Tool {name}",
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        ),
    )


class TestMiMoDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool(
                "get_current_weather",
                {
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "unit": {"type": "string"},
                },
            ),
            _make_tool(
                "calculate_area",
                {
                    "shape": {"type": "string"},
                    "dimensions": {"type": "object"},
                    "precision": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
            ),
        ]
        self.detector = MiMoDetector()

    def _collect_streaming_result(self, chunks, tools=None):
        tools = self.tools if tools is None else tools
        normal_text = ""
        tool_calls = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.tool_index is None:
                    continue
                if call.tool_index not in tool_calls:
                    tool_calls[call.tool_index] = {"name": "", "parameters": ""}
                if call.name:
                    tool_calls[call.tool_index]["name"] = call.name
                if call.parameters:
                    tool_calls[call.tool_index]["parameters"] += call.parameters
        return normal_text, tool_calls

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call("<tool_call>test</tool_call>"))
        self.assertFalse(self.detector.has_tool_call("No tool call here"))

    def test_detect_and_parse_no_tools(self):
        model_output = "This is a response without tool calls."

        result = self.detector.detect_and_parse(model_output, tools=[])

        self.assertEqual(result.normal_text, model_output)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_tool(self):
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>Dallas</parameter>
<parameter=state>TX</parameter>
<parameter=unit>fahrenheit</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Dallas", "state": "TX", "unit": "fahrenheit"},
        )

    def test_detect_and_parse_with_surrounding_content(self):
        model_output = """Sure, checking now.<tool_call>
<function=get_current_weather>
<parameter=city>Dallas</parameter>
<parameter=state>TX</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(result.normal_text, "Sure, checking now.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

    def test_detect_and_parse_type_conversion(self):
        model_output = """<tool_call>
<function=calculate_area>
<parameter=shape>rectangle</parameter>
<parameter=dimensions>{"width": 10, "height": 20}</parameter>
<parameter=precision>2</parameter>
<parameter=enabled>true</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "calculate_area")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "shape": "rectangle",
                "dimensions": {"width": 10, "height": 20},
                "precision": 2,
                "enabled": True,
            },
        )

    def test_detect_and_parse_multiple_tools(self):
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>Dallas</parameter>
<parameter=state>TX</parameter>
</function>
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>Orlando</parameter>
<parameter=state>FL</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Dallas")
        self.assertEqual(json.loads(result.calls[1].parameters)["city"], "Orlando")

    def test_detect_and_parse_no_parameters(self):
        model_output = """<tool_call>
<function=get_current_weather>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_detect_and_parse_multiline_parameter(self):
        write_tool = _make_tool(
            "write_file",
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        )
        model_output = """<tool_call>
<function=write_file>
<parameter=path>/tmp/report.txt</parameter>
<parameter=content>line 1
line 2
line 3</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, [write_tool])

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "write_file")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"path": "/tmp/report.txt", "content": "line 1\nline 2\nline 3"},
        )

    def test_detect_and_parse_special_chars_in_value(self):
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>Dallas->TX</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Dallas->TX")

    def test_html_entities_are_preserved(self):
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>Dallas &amp; Fort Worth &current</parameter>
<parameter=state>TX</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Dallas &amp; Fort Worth &current")

    def test_streaming_complete_tool_call(self):
        chunks = [
            "Let me check.",
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n<parameter=city>Dallas</parameter>",
            "\n<parameter=state>TX</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]

        normal_text, tool_calls = self._collect_streaming_result(chunks)

        self.assertEqual(normal_text, "Let me check.")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(json.loads(tool_calls[0]["parameters"])["city"], "Dallas")

    def test_streaming_incomplete_tool_call_streams_completed_parts(self):
        chunks = [
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n<parameter=city>Dallas</parameter>",
            "\n<parameter=state>",
        ]

        normal_text, tool_calls = self._collect_streaming_result(chunks)

        self.assertEqual(normal_text, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Dallas", "state": ')

        result = self.detector.parse_streaming_increment(
            "TX</parameter>\n</function>\n</tool_call>", self.tools
        )

        self.assertEqual(result.normal_text, "")
        for call in result.calls:
            if call.parameters:
                tool_calls[call.tool_index]["parameters"] += call.parameters
        self.assertEqual(
            json.loads(tool_calls[0]["parameters"]),
            {"city": "Dallas", "state": "TX"},
        )

    def test_streaming_multiple_tool_calls_with_interleaved_text(self):
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>Dallas</parameter>
<parameter=state>TX</parameter>
</function>
</tool_call>
Some text in between.
<tool_call>
<function=calculate_area>
<parameter=shape>circle</parameter>
<parameter=dimensions>{"radius": 5}</parameter>
<parameter=precision>2</parameter>
<parameter=enabled>false</parameter>
</function>
</tool_call>"""
        chunks = [model_output[i : i + 20] for i in range(0, len(model_output), 20)]

        normal_text, tool_calls = self._collect_streaming_result(chunks)

        self.assertIn("Some text in between.", normal_text)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(tool_calls[1]["name"], "calculate_area")
        self.assertEqual(
            json.loads(tool_calls[0]["parameters"]),
            {"city": "Dallas", "state": "TX"},
        )
        self.assertEqual(
            json.loads(tool_calls[1]["parameters"]),
            {
                "shape": "circle",
                "dimensions": {"radius": 5},
                "precision": 2,
                "enabled": False,
            },
        )


class TestMiMoReasoningToolCalls(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_current_weather", {"city": {"type": "string"}})]

    def test_reasoning_text_tool_call_fallback(self):
        try:
            from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        except ImportError as exc:
            self.skipTest(f"serving_chat dependencies unavailable: {exc}")

        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        chat.tool_call_parser = "mimo"
        finish_reason = {"type": "stop", "matched": "</s>"}
        reasoning_text = """Need weather data.<tool_call>
<function=get_current_weather>
<parameter=city>Dallas</parameter>
</function>
</tool_call>"""

        tool_calls, text, reasoning_text, finish_reason = (
            chat._process_tool_calls_with_reasoning_fallback(
                "",
                reasoning_text,
                self.tools,
                finish_reason,
                "auto",
                0,
            )
        )

        self.assertEqual(text, "")
        self.assertEqual(reasoning_text, "Need weather data.")
        self.assertEqual(finish_reason["type"], "tool_calls")
        self.assertIsNone(finish_reason["matched"])
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments), {"city": "Dallas"}
        )


if __name__ == "__main__":
    unittest.main()
