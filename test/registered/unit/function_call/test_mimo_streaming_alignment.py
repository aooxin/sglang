"""
Alignment test: verify that streaming parse_streaming_increment produces
identical results to non-streaming detect_and_parse for the same input.

For each test case:
  1. Run detect_and_parse on the full text → get ground-truth tool calls
  2. Simulate streaming by splitting text into random-sized chunks
  3. Feed each chunk to parse_streaming_increment → concatenate results
  4. Assert: function names match, accumulated arguments match (JSON
     semantically for regular calls, verbatim for freeform calls)
  5. Repeat with multiple random chunk splits for robustness
"""

import json
import random
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(10, "stage-a-test-cpu")


def make_tool(name, params):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"Tool {name}",
            parameters={"properties": params, "required": list(params.keys())},
        ),
    )


TOOLS = [
    make_tool(
        "write",
        {"file_path": {"type": "string"}, "content": {"type": "string"}},
    ),
    make_tool(
        "bash",
        {"command": {"type": "string"}, "description": {"type": "string"}},
    ),
    make_tool(
        "edit",
        {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
    ),
    make_tool(
        "calc",
        {
            "shape": {"type": "string"},
            "dimensions": {"type": "object"},
            "precision": {"type": "integer"},
            "enabled": {"type": "boolean"},
        },
    ),
    make_tool(
        "search",
        {"query": {"type": "string"}, "tags": {"type": "array"}},
    ),
    # Freeform (custom) tool: arguments is a raw string, no schema
    make_tool("exec", {}),
]

# Test corpus: various model outputs covering edge cases.
# Primary cases use the current chat template format (no separators between
# tags); the "Legacy" group keeps the old newline-separated format, which
# must remain accepted.
TEST_CASES = [
    # 1. Simple single tool call
    (
        "Simple single tool",
        "<tool_call><function=bash><parameter=command>ls -la /tmp</parameter><parameter=description>list files</parameter></function></tool_call>",
    ),
    # 2. Tool call with preceding text
    (
        "Text before tool call",
        "Let me check the files for you.<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>",
    ),
    # 3. HTML content with < > in string parameter
    (
        "HTML content with angle brackets",
        '<tool_call><function=write><parameter=file_path>/tmp/test.html</parameter><parameter=content><!DOCTYPE html>\n<html lang="zh-CN">\n<head><title>Test</title></head>\n<body>\n<div class="nav">Navigation</div>\n<div class="card">\n<h2>Title</h2>\n<p>Content here</p>\n</div>\n</body>\n</html></parameter></function></tool_call>',
    ),
    # 4. Content with &copy; and &amp; (should NOT be unescaped)
    (
        "HTML entities preserved",
        "<tool_call><function=write><parameter=file_path>/tmp/entity.html</parameter><parameter=content>&copy; 2026 Xiaomi &amp; Partners\n&lt;div&gt; escaped tags &lt;/div&gt;</parameter></function></tool_call>",
    ),
    # 5. Content with quotes and backslashes (JSON escape stress)
    (
        "JSON escape stress test",
        "<tool_call><function=write><parameter=file_path>/tmp/escape.txt</parameter><parameter=content>He said \"hello\" and she said 'goodbye'\nLine 2 with\\backslash\nLine 3\twith\ttabs</parameter></function></tool_call>",
    ),
    # 6. Multiple tool calls (back-to-back, no separator)
    (
        "Multiple tool calls",
        "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call><tool_call><function=write><parameter=file_path>/tmp/out.txt</parameter><parameter=content>hello world</parameter></function></tool_call>",
    ),
    # 7. Multiple tool calls with text in between
    (
        "Multiple tool calls with interleaving text",
        "First task.<tool_call><function=bash><parameter=command>date</parameter></function></tool_call>Now the second task.<tool_call><function=write><parameter=file_path>/tmp/date.txt</parameter><parameter=content>today</parameter></function></tool_call>",
    ),
    # 8. Object type parameter (deferred parsing)
    (
        "Object type parameter",
        '<tool_call><function=calc><parameter=shape>rectangle</parameter><parameter=dimensions>{"width": 10, "height": 20}</parameter><parameter=precision>3</parameter><parameter=enabled>true</parameter></function></tool_call>',
    ),
    # 9. Array type parameter
    (
        "Array type parameter",
        '<tool_call><function=search><parameter=query>sglang tool call</parameter><parameter=tags>["python", "ml", "inference"]</parameter></function></tool_call>',
    ),
    # 10. Edit tool with 3 string params
    (
        "Edit with multi-line strings",
        '<tool_call><function=edit><parameter=file_path>/tmp/code.py</parameter><parameter=old_string>def hello():\n    print("old")</parameter><parameter=new_string>def hello():\n    print("new")\n    return True</parameter></function></tool_call>',
    ),
    # 11. Large HTML content (~2000 chars)
    (
        "Large HTML content",
        "<tool_call><function=write><parameter=file_path>/tmp/large.html</parameter><parameter=content>"
        + "<!DOCTYPE html><html><body>"
        + "".join(
            f'<div class="item-{i}"><h3>Item {i}</h3><p>Description for item {i} with special chars: <>&"\'</p></div>\n'
            for i in range(50)
        )
        + "</body></html>"
        + "</parameter></function></tool_call>",
    ),
    # 12. Unicode content
    (
        "Unicode Chinese content",
        "<tool_call><function=write><parameter=file_path>/tmp/cn.txt</parameter><parameter=content>你好世界！这是一段中文测试。\n包含特殊符号：《》「」【】\n以及 emoji 😀🎉\nEnd.</parameter></function></tool_call>",
    ),
    # 13. Empty tool call (no parameters)
    (
        "Empty parameters",
        "<tool_call><function=bash></function></tool_call>",
    ),
    # 14. Content that looks like XML tags (partial tags that don't match real closers)
    (
        "Content with partial XML-like text",
        "<tool_call><function=write><parameter=file_path>/tmp/xml_like.txt</parameter><parameter=content>Some text with </para and <param=fake> tags\nAlso <bold> and </bold> like strings\nBut these are just content</parameter></function></tool_call>",
    ),
    # NOTE: Content containing literal </function> or </tool_call> is a known
    # limitation of BOTH streaming and non-streaming paths. The regex-based
    # detect_and_parse is truncated by the inner </tool_call>, and the SAX
    # streaming path actually handles it better. Models never generate these
    # sequences as content because they are trained as structural tokens.
    # 15. Three consecutive tool calls
    (
        "Three consecutive tool calls",
        "<tool_call><function=bash><parameter=command>echo 1</parameter></function></tool_call>"
        "<tool_call><function=bash><parameter=command>echo 2</parameter></function></tool_call>"
        "<tool_call><function=write><parameter=file_path>/tmp/3.txt</parameter><parameter=content>three</parameter></function></tool_call>",
    ),
    # 15b. String param whose value is the literal text "null" (must stay a
    # string in both paths; regression for the null-before-string-type check)
    (
        "String param with literal null",
        "<tool_call><function=bash><parameter=command>null</parameter></function></tool_call>",
    ),
    # 16. Freeform (custom) tool call: no <parameter=...> tags, body is the
    # raw arguments string
    (
        "Freeform simple",
        '<tool_call><function=exec>await tools.run({cmd:"ls -la"});</function></tool_call>',
    ),
    # 17. Freeform multi-line code with quotes, backslashes and shell syntax
    (
        "Freeform multi-line code",
        "<tool_call><function=exec>const results = await Promise.all([\n"
        "  tools.exec_command({cmd:\"pwd && rg --files -g '!*node_modules*' | sed -n '1,200p'\",workdir:\"/workspace\"}),\n"
        '  tools.exec_command({cmd:"find . -type f \\\\( -name \'*.py\' \\\\) -print",workdir:"/workspace"})\n'
        "]);\nfor (const r of results) text(r.output);\n</function></tool_call>",
    ),
    # 18. Freeform with angle brackets and HTML-ish content
    (
        "Freeform with angle brackets",
        '<tool_call><function=exec>if (a < b && c > d) { render("<div class=\'x\'>hi</div>"); }</function></tool_call>',
    ),
    # 19. Freeform with leading and trailing newlines (part of the arguments)
    (
        "Freeform leading/trailing newlines",
        "<tool_call><function=exec>\nconst x = 1;\nreturn x;\n</function></tool_call>",
    ),
    # 20. Freeform followed by a regular call, with interleaving text
    (
        "Freeform then regular call",
        "Run this first.<tool_call><function=exec>await run();</function></tool_call>Then write.<tool_call><function=write><parameter=file_path>/tmp/a.txt</parameter><parameter=content>done</parameter></function></tool_call>",
    ),
    # --- Legacy format: newlines between tags (old chat template) ---
    # 21. Legacy simple single tool
    (
        "Legacy simple single tool",
        "<tool_call>\n<function=bash>\n<parameter=command>ls -la /tmp</parameter>\n<parameter=description>list files</parameter>\n</function>\n</tool_call>",
    ),
    # 22. Legacy multi-line string values
    (
        "Legacy edit with multi-line strings",
        '<tool_call>\n<function=edit>\n<parameter=file_path>/tmp/code.py</parameter>\n<parameter=old_string>def hello():\n    print("old")</parameter>\n<parameter=new_string>def hello():\n    print("new")\n    return True</parameter>\n</function>\n</tool_call>',
    ),
    # 23. Legacy empty tool call (whitespace-only body is not freeform)
    (
        "Legacy empty parameters",
        "<tool_call>\n<function=bash>\n</function>\n</tool_call>",
    ),
    # 24. Legacy multiple tool calls with text in between
    (
        "Legacy multiple tool calls with interleaving text",
        "First task.<tool_call>\n<function=bash>\n<parameter=command>date</parameter>\n</function>\n</tool_call>\nNow the second task.<tool_call>\n<function=write>\n<parameter=file_path>/tmp/date.txt</parameter>\n<parameter=content>today</parameter>\n</function>\n</tool_call>",
    ),
    # 25. Legacy object type parameter
    (
        "Legacy object type parameter",
        '<tool_call>\n<function=calc>\n<parameter=shape>rectangle</parameter>\n<parameter=dimensions>{"width": 10, "height": 20}</parameter>\n<parameter=precision>3</parameter>\n<parameter=enabled>true</parameter>\n</function>\n</tool_call>',
    ),
]


def run_non_streaming(text, tools):
    """Ground truth: one-shot parse."""
    detector = MiMoDetector()
    result = detector.detect_and_parse(text, tools)
    calls = []
    for c in result.calls:
        calls.append({"name": c.name, "params": c.parameters})
    return result.normal_text, calls


def run_streaming(text, tools, chunk_sizes):
    """Simulate streaming with given chunk sizes."""
    detector = MiMoDetector()
    normal_text = ""
    tool_calls = {}

    pos = 0
    for cs in chunk_sizes:
        chunk = text[pos : pos + cs]
        if not chunk:
            break
        pos += cs
        result = detector.parse_streaming_increment(chunk, tools)
        normal_text += result.normal_text
        for call in result.calls:
            idx = call.tool_index
            if idx is not None:
                if idx not in tool_calls:
                    tool_calls[idx] = {"name": "", "params": ""}
                if call.name:
                    tool_calls[idx]["name"] = call.name
                if call.parameters:
                    tool_calls[idx]["params"] += call.parameters

    # Feed remaining
    if pos < len(text):
        chunk = text[pos:]
        result = detector.parse_streaming_increment(chunk, tools)
        normal_text += result.normal_text
        for call in result.calls:
            idx = call.tool_index
            if idx is not None:
                if idx not in tool_calls:
                    tool_calls[idx] = {"name": "", "params": ""}
                if call.name:
                    tool_calls[idx]["name"] = call.name
                if call.parameters:
                    tool_calls[idx]["params"] += call.parameters

    calls = [tool_calls[i] for i in sorted(tool_calls.keys())]
    return normal_text, calls


def make_random_chunks(text_len, min_size=1, max_size=30, seed=None):
    """Generate random chunk sizes that cover the full text."""
    rng = random.Random(seed)
    sizes = []
    remaining = text_len
    while remaining > 0:
        upper = min(max_size, remaining)
        lower = min(min_size, upper)
        cs = rng.randint(lower, upper)
        sizes.append(cs)
        remaining -= cs
    return sizes


class TestMiMoStreamingAlignment(unittest.TestCase):
    """Verify streaming output matches non-streaming for every test case."""

    def _assert_aligned(
        self, label, text, chunk_desc, ns_text, ns_calls, st_text, st_calls
    ):
        # Normal text: streaming may include inter-tool-call text that
        # detect_and_parse discards, so we check that detect_and_parse's
        # normal_text is a substring of streaming's normal_text.
        self.assertIn(
            ns_text.strip(),
            st_text.strip(),
            f"[{label}] [{chunk_desc}] normal_text: non-streaming text not found in streaming text:\n  non-streaming: {ns_text!r}\n  streaming:     {st_text!r}",
        )

        # Compare number of tool calls
        self.assertEqual(
            len(ns_calls),
            len(st_calls),
            f"[{label}] [{chunk_desc}] tool call count mismatch: "
            f"non-streaming={len(ns_calls)} streaming={len(st_calls)}",
        )

        # Compare each tool call
        for i, (ns, st) in enumerate(zip(ns_calls, st_calls)):
            self.assertEqual(
                ns["name"],
                st["name"],
                f"[{label}] [{chunk_desc}] tool[{i}] name mismatch",
            )
            if ns["params"] and st["params"]:
                # Regular calls: params is JSON, compare semantically.
                # Freeform calls: params is a raw string, compare verbatim.
                try:
                    ns_parsed = json.loads(ns["params"])
                    st_parsed = json.loads(st["params"])
                except json.JSONDecodeError:
                    self.assertEqual(
                        ns["params"],
                        st["params"],
                        f"[{label}] [{chunk_desc}] tool[{i}] freeform params mismatch:\n"
                        f"  non-streaming: {ns['params']!r}\n"
                        f"  streaming:     {st['params']!r}",
                    )
                else:
                    self.assertEqual(
                        ns_parsed,
                        st_parsed,
                        f"[{label}] [{chunk_desc}] tool[{i}] params mismatch:\n"
                        f"  non-streaming: {json.dumps(ns_parsed, ensure_ascii=False)}\n"
                        f"  streaming:     {json.dumps(st_parsed, ensure_ascii=False)}",
                    )

    def test_alignment_char_by_char(self):
        """Stream one character at a time — worst case for boundary handling."""
        for label, text in TEST_CASES:
            ns_text, ns_calls = run_non_streaming(text, TOOLS)
            chunks = [1] * len(text)
            st_text, st_calls = run_streaming(text, TOOLS, chunks)
            self._assert_aligned(
                label, text, "char-by-char", ns_text, ns_calls, st_text, st_calls
            )

    def test_alignment_token_sized(self):
        """Stream in 5-20 char chunks, simulating real token sizes."""
        for label, text in TEST_CASES:
            ns_text, ns_calls = run_non_streaming(text, TOOLS)
            chunks = make_random_chunks(len(text), min_size=5, max_size=20, seed=42)
            st_text, st_calls = run_streaming(text, TOOLS, chunks)
            self._assert_aligned(
                label, text, "token-sized", ns_text, ns_calls, st_text, st_calls
            )

    def test_alignment_large_chunks(self):
        """Stream in large chunks (50-200 chars)."""
        for label, text in TEST_CASES:
            ns_text, ns_calls = run_non_streaming(text, TOOLS)
            chunks = make_random_chunks(len(text), min_size=50, max_size=200, seed=99)
            st_text, st_calls = run_streaming(text, TOOLS, chunks)
            self._assert_aligned(
                label, text, "large-chunks", ns_text, ns_calls, st_text, st_calls
            )

    def test_alignment_single_shot(self):
        """Stream the entire text as one chunk — should behave like detect_and_parse."""
        for label, text in TEST_CASES:
            ns_text, ns_calls = run_non_streaming(text, TOOLS)
            st_text, st_calls = run_streaming(text, TOOLS, [len(text)])
            self._assert_aligned(
                label, text, "single-shot", ns_text, ns_calls, st_text, st_calls
            )

    def test_alignment_random_splits_x10(self):
        """10 different random chunk splits per test case."""
        for label, text in TEST_CASES:
            ns_text, ns_calls = run_non_streaming(text, TOOLS)
            for seed in range(10):
                chunks = make_random_chunks(
                    len(text), min_size=1, max_size=50, seed=seed
                )
                st_text, st_calls = run_streaming(text, TOOLS, chunks)
                self._assert_aligned(
                    label,
                    text,
                    f"random-seed-{seed}",
                    ns_text,
                    ns_calls,
                    st_text,
                    st_calls,
                )


class TestMiMoStreamingFailureModes(unittest.TestCase):
    """
    Streaming-only behaviors that have no direct alignment counterpart:
      - unknown tool names must not surface to the client (mirror of
        detect_and_parse's SGLANG_FORWARD_UNKNOWN_TOOLS=False handling)
      - SAX parse errors must fail-closed (no half-baked tool emitted)
    """

    def test_unknown_tool_matches_nonstreaming(self):
        """Unknown function name: streaming should not emit a tool_call,
        and the consumed bytes should appear in normal_text just like
        detect_and_parse does."""
        text = (
            "prefix <tool_call><function=evil_unknown>"
            "<parameter=command>rm -rf /</parameter></function></tool_call>"
        )

        ns_text, ns_calls = run_non_streaming(text, TOOLS)
        st_text, st_calls = run_streaming(text, TOOLS, [1] * len(text))

        self.assertEqual(ns_calls, [], "non-streaming must drop unknown tool")
        self.assertEqual(st_calls, [], "streaming must drop unknown tool")
        # Streaming normal_text must be a superset of non-streaming's --
        # they should both contain the raw <tool_call>...</tool_call>.
        self.assertIn("evil_unknown", st_text)
        self.assertIn("evil_unknown", ns_text)

    def test_unknown_freeform_tool_matches_nonstreaming(self):
        """Unknown freeform tool: same fail-closed behavior."""
        text = (
            "prefix <tool_call><function=evil_unknown>"
            "await doBadThings();</function></tool_call>"
        )

        ns_text, ns_calls = run_non_streaming(text, TOOLS)
        st_text, st_calls = run_streaming(text, TOOLS, [1] * len(text))

        self.assertEqual(ns_calls, [], "non-streaming must drop unknown tool")
        self.assertEqual(st_calls, [], "streaming must drop unknown tool")
        self.assertIn("evil_unknown", st_text)
        self.assertIn("evil_unknown", ns_text)

    def test_streaming_fail_closed_on_parse_error(self):
        """If the underlying expat parser raises, streaming must not emit
        any partially-formed tool_call. The SAX should mark itself
        completed+suppressed so the detector treats consumed bytes as
        normal text."""
        from sglang.srt.function_call.mimo_detector import _MiMoStreamingSAX

        class _AlwaysFailExpat:
            def Parse(self, s, end):
                import xml.parsers.expat as _ex

                raise _ex.ExpatError("synthetic parse failure")

        sax = _MiMoStreamingSAX(TOOLS)
        sax._parser = _AlwaysFailExpat()

        calls = sax.feed_chunk("<tool_call><function=bash><parameter=command>")
        self.assertEqual(calls, [], "no tool_call may be emitted on parse error")
        self.assertTrue(sax.tool_call_completed, "SAX must hand control back")
        self.assertTrue(
            sax._tool_call_suppressed,
            "SAX must mark suppressed so detector converts raw to normal_text",
        )


if __name__ == "__main__":
    unittest.main()
