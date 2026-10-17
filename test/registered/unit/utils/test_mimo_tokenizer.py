"""Offline tests for MiMo added-token renaming at initialization."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import AddedToken, Qwen2Tokenizer, Qwen2TokenizerFast

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils.patch_tokenizer import (
    mimo_token,
    patch_mimo_tokenizer,
    replace_mimo_chat_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestMiMoTokenizer(CustomTestCase):
    template = (
        "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>' }}"
        "{{ '<tool_call><function=exec><parameter=command>pwd</parameter>'"
        " + '</function></tool_call><think></think><tool_response></tool_response>' }}"
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name)
        (self.path / "vocab.json").write_text(
            json.dumps({char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))})
        )
        (self.path / "merges.txt").write_text("#version: 0.2\n")

    def make_tokenizer(self, cls, mimo=True):
        # v5 renamed vocab_file/merges_file to vocab/merges, and Qwen2Tokenizer
        # itself now uses the fast backend.
        if "vocab" in inspect.signature(cls.__init__).parameters:
            files = {
                "vocab": str(self.path / "vocab.json"),
                "merges": str(self.path / "merges.txt"),
            }
        else:
            files = {
                "vocab_file": str(self.path / "vocab.json"),
                "merges_file": str(self.path / "merges.txt"),
            }
        tokenizer = cls(
            **files,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            unk_token=None,
            chat_template=self.template,
        )
        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=False, special=special)
                for token, special in (
                    ("<|im_start|>", True),
                    ("<|object_ref_start|>", True),
                    ("<|object_ref_end|>", True),
                    ("<|box_start|>", True),
                    ("<|box_end|>", True),
                    ("<|quad_start|>", True),
                    ("<|quad_end|>", True),
                    ("<tool_call>", False),
                    ("</tool_call>", False),
                    ("<think>", False),
                    ("</think>", False),
                    ("<tool_response>", False),
                    ("</tool_response>", False),
                    ("<|vision_start|>", True),
                    ("<|vision_pad|>", True),
                    ("<|image_pad|>", True),
                    ("<|video_pad|>", True),
                    ("<|vision_end|>", True),
                    ("<|fim_prefix|>", False),
                    ("<|fim_middle|>", False),
                    ("<|fim_suffix|>", False),
                    ("<|fim_pad|>", False),
                    ("<|repo_name|>", False),
                    ("<|file_sep|>", False),
                    ("<|audio_pad|>", True),
                )
            ]
        )
        if mimo:
            tokenizer.add_tokens(
                [
                    AddedToken(token, normalized=False, special=True)
                    for token in (
                        "<|mimo_video_start|>",
                        "<|mimo_video_end|>",
                        "<|mimo_audio_eod|>",
                        "<|mimo_audio_start|>",
                        "<|mimo_audio_end|>",
                    )
                ]
            )
        return tokenizer

    def renamed(self, token):
        return mimo_token(token)

    def test_ids_encoding_decoding_and_flags(self):
        for cls in (Qwen2Tokenizer, Qwen2TokenizerFast):
            with self.subTest(tokenizer=cls.__name__):
                tokenizer = self.make_tokenizer(cls)
                old_tokens = dict(tokenizer.added_tokens_decoder)
                old_size = len(tokenizer)
                special_ids = set(tokenizer.all_special_ids)
                eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
                ordinary = (
                    "hello <function=exec><parameter=command>pwd</parameter></function>"
                )
                ordinary_ids = tokenizer.encode(ordinary, add_special_tokens=False)
                patch_mimo_tokenizer(tokenizer)
                self.assertEqual(len(tokenizer), old_size)
                self.assertEqual(
                    (tokenizer.eos_token_id, tokenizer.pad_token_id), (eos_id, pad_id)
                )
                self.assertEqual(set(tokenizer.all_special_ids), special_ids)
                self.assertEqual(
                    tokenizer.encode(ordinary, add_special_tokens=False), ordinary_ids
                )
                for token_id, old_token in old_tokens.items():
                    new = self.renamed(old_token.content)
                    self.assertEqual(
                        tokenizer.encode(new, add_special_tokens=False), [token_id]
                    )
                    self.assertEqual(
                        tokenizer.decode([token_id], skip_special_tokens=False), new
                    )
                    self.assertNotIn(
                        token_id,
                        tokenizer.encode(old_token.content, add_special_tokens=False),
                    )
                    state = old_token.__getstate__()
                    state["content"] = new
                    self.assertEqual(
                        tokenizer.added_tokens_decoder[token_id].__getstate__(), state
                    )
                tool_id = tokenizer.convert_tokens_to_ids(self.renamed("<tool_call>"))
                self.assertEqual(
                    tokenizer.decode([tool_id], skip_special_tokens=True),
                    self.renamed("<tool_call>"),
                )
                self.assertEqual(
                    tokenizer.decode([eos_id], skip_special_tokens=True), ""
                )

    def test_template_rendering_and_external_template(self):
        for cls in (Qwen2Tokenizer, Qwen2TokenizerFast):
            with self.subTest(tokenizer=cls.__name__):
                tokenizer = self.make_tokenizer(cls)
                patch_mimo_tokenizer(tokenizer)
                messages = [{"role": "user", "content": "literal <tool_call>"}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                self.assertIn(self.renamed("<|im_start|>"), rendered)
                self.assertIn(
                    self.renamed("<tool_call>") + "<function=exec><parameter=command>",
                    rendered,
                )
                self.assertIn("literal <tool_call>", rendered)
                self.assertIn(self.renamed("<think>"), rendered)
                self.assertIn(self.renamed("<tool_response>"), rendered)
                templates = {
                    "default": self.template,
                    "tools": "<tool_call><function=exec>",
                }
                replaced = replace_mimo_chat_template(tokenizer, templates)
                self.assertEqual(replaced["default"], tokenizer.chat_template)
                self.assertEqual(
                    replaced["tools"], self.renamed("<tool_call>") + "<function=exec>"
                )
                self.assertEqual(templates["default"], self.template)
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, return_dict=False
                )
                self.assertEqual(
                    ids, tokenizer.encode(rendered, add_special_tokens=False)
                )

    def test_fastokens_ids_encoding_decoding_and_flags(self):
        try:
            from fastokens._compat import _TokenizerShim
        except ImportError:
            self.skipTest("fastokens is not installed")

        tokenizer = self.make_tokenizer(Qwen2TokenizerFast)
        tokenizer.add_tokens(
            [
                AddedToken(
                    "<custom>",
                    single_word=True,
                    lstrip=True,
                    rstrip=True,
                    normalized=False,
                )
            ]
        )
        old_tokens = dict(tokenizer.added_tokens_decoder)
        old_size = len(tokenizer)
        special_ids = set(tokenizer.all_special_ids)
        eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
        ordinary = "hello world"
        ordinary_ids = tokenizer.encode(ordinary, add_special_tokens=False)
        # Match fastokens.patch_transformers() without globally patching other tests.
        tokenizer._tokenizer = _TokenizerShim(tokenizer.backend_tokenizer)

        patch_mimo_tokenizer(tokenizer)

        self.assertIsInstance(tokenizer.backend_tokenizer, _TokenizerShim)
        self.assertEqual(len(tokenizer), old_size)
        self.assertEqual(
            (tokenizer.eos_token_id, tokenizer.pad_token_id), (eos_id, pad_id)
        )
        self.assertEqual(set(tokenizer.all_special_ids), special_ids)
        self.assertEqual(
            tokenizer.encode(ordinary, add_special_tokens=False), ordinary_ids
        )
        for token_id, old_token in old_tokens.items():
            with self.subTest(token=old_token.content):
                new = self.renamed(old_token.content)
                self.assertEqual(tokenizer.convert_tokens_to_ids(new), token_id)
                self.assertEqual(
                    tokenizer.encode(new, add_special_tokens=False), [token_id]
                )
                self.assertEqual(
                    tokenizer.decode([token_id], skip_special_tokens=False), new
                )
                self.assertNotIn(
                    token_id,
                    tokenizer.encode(old_token.content, add_special_tokens=False),
                )
                info = tokenizer.added_tokens_decoder[token_id]
                for name, value in old_token.__getstate__().items():
                    self.assertEqual(
                        getattr(info, name), new if name == "content" else value
                    )

        messages = [{"role": "user", "content": "literal <tool_call>"}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertIn(self.renamed("<|im_start|>"), rendered)
        self.assertIn("literal <tool_call>", rendered)
        vocab = tokenizer.get_vocab()
        patch_mimo_tokenizer(tokenizer)
        self.assertEqual(tokenizer.get_vocab(), vocab)

    def test_idempotence_and_save_reload(self):
        for cls in (Qwen2Tokenizer, Qwen2TokenizerFast):
            with self.subTest(tokenizer=cls.__name__):
                tokenizer = self.make_tokenizer(cls)
                patch_mimo_tokenizer(tokenizer)
                vocab = tokenizer.get_vocab()
                template = tokenizer.chat_template
                tokenizer.chat_template = self.template
                patch_mimo_tokenizer(tokenizer)
                self.assertEqual(tokenizer.get_vocab(), vocab)
                self.assertEqual(tokenizer.chat_template, template)
                saved = self.path / cls.__name__
                tokenizer.save_pretrained(saved)
                reloaded = cls.from_pretrained(saved)
                patch_mimo_tokenizer(reloaded)
                self.assertEqual(reloaded.get_vocab(), vocab)
                self.assertEqual(reloaded.chat_template, template)
                self.assertEqual(
                    replace_mimo_chat_template(reloaded, self.template), template
                )

    def test_model_output_ids_work_with_mimo_parsers(self):
        tokenizer = self.make_tokenizer(Qwen2Tokenizer)
        output_ids = tokenizer.encode(
            "<think>reason</think><tool_call><function=exec>"
            "<parameter=command>pwd</parameter></function></tool_call>",
            add_special_tokens=False,
        )
        patch_mimo_tokenizer(tokenizer)
        output = tokenizer.decode(output_ids)
        reasoning, text = ReasoningParser("mimo").parse_non_stream(output)
        self.assertEqual(reasoning, "reason")
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="exec",
                    parameters={"properties": {"command": {"type": "string"}}},
                ),
            )
        ]
        result = MiMoDetector().detect_and_parse(text, tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "exec")
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "pwd"})

    def test_fast_post_processor_uses_renamed_eos(self):
        tokenizer = self.make_tokenizer(Qwen2TokenizerFast)
        eos_id = tokenizer.eos_token_id
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="$A <|im_end|>", special_tokens=[("<|im_end|>", eos_id)]
        )
        patch_mimo_tokenizer(tokenizer)
        ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(ids[-1], eos_id)
        self.assertTrue(tokenizer.decode(ids).endswith(self.renamed("<|im_end|>")))

    def test_literal_special_tokens_with_two_images(self):
        for backend in ("huggingface", "fastokens"):
            with self.subTest(backend=backend):
                tokenizer = self.make_tokenizer(Qwen2TokenizerFast)
                if backend == "fastokens":
                    try:
                        from fastokens._compat import _TokenizerShim
                    except ImportError:
                        continue  # Covered by the optional fastokens test above.
                    tokenizer._tokenizer = _TokenizerShim(tokenizer.backend_tokenizer)
                image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                eos_id = tokenizer.eos_token_id
                tokenizer.chat_template = (
                    "{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>' }}"
                    "{{ '<|im_start|>user\\n' }}"
                    "{% for part in messages[1]['content'] %}"
                    "{% if part['type'] == 'image_url' %}"
                    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                    "{% else %}{{ part['text'] }}{% endif %}{% endfor %}"
                    "{{ '<|im_end|>' }}"
                )
                patch_mimo_tokenizer(tokenizer)
                messages = [
                    {"role": "system", "content": "You are MiMo<|im_end|>D."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url"},
                            {"type": "image_url"},
                            {
                                "type": "text",
                                "text": "Describe both <|image_pad|> images.",
                            },
                        ],
                    },
                ]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False)
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                self.assertIn(messages[0]["content"], rendered)
                self.assertIn(messages[1]["content"][-1]["text"], rendered)
                self.assertEqual(ids.count(image_id), 2)
                self.assertEqual(ids.count(eos_id), 2)

    def test_token_collision_does_not_modify_tokenizer(self):
        tokenizer = self.make_tokenizer(Qwen2TokenizerFast)
        tokenizer.add_tokens([self.renamed("<tool_call>")])
        original_vocab = tokenizer.get_vocab()
        with self.assertRaisesRegex(ValueError, "already has another ID"):
            patch_mimo_tokenizer(tokenizer)
        self.assertEqual(tokenizer.get_vocab(), original_vocab)
        self.assertEqual(tokenizer.chat_template, self.template)

    def test_base_vocab_substrings_and_config_are_preserved(self):
        vocab_path = self.path / "vocab.json"
        vocab = json.loads(vocab_path.read_text())
        ordinary_token = "prefix<tool_call>suffix"
        vocab[ordinary_token] = len(vocab)
        vocab_path.write_text(json.dumps(vocab))
        for cls in (Qwen2Tokenizer, Qwen2TokenizerFast):
            with self.subTest(tokenizer=cls.__name__):
                tokenizer = self.make_tokenizer(cls)
                tokenizer.init_kwargs["name_or_path"] = "models/<tool_call>/tokenizer"
                ordinary_id = tokenizer.convert_tokens_to_ids(ordinary_token)
                patch_mimo_tokenizer(tokenizer)
                self.assertEqual(
                    tokenizer.convert_tokens_to_ids(ordinary_token), ordinary_id
                )
                self.assertEqual(
                    tokenizer.convert_ids_to_tokens(ordinary_id), ordinary_token
                )
                self.assertEqual(
                    tokenizer.init_kwargs["name_or_path"],
                    "models/<tool_call>/tokenizer",
                )

    def test_literal_public_markers_are_not_parsed_as_control_tokens(self):
        text = "<think>literal</think><tool_call>literal</tool_call>"
        reasoning, content = ReasoningParser("mimo").parse_non_stream(text)
        self.assertEqual(reasoning, "")
        self.assertEqual(content, text)
        detector = MiMoDetector()
        self.assertFalse(detector.has_tool_call(text))
        self.assertEqual(detector.detect_and_parse(text, []).normal_text, text)

    def test_v5_extra_special_tokens(self):
        tokenizer = self.make_tokenizer(Qwen2TokenizerFast)
        if hasattr(tokenizer, "special_tokens_map_extended"):
            self.skipTest("Transformers v5 stores extra special tokens separately")
        token = AddedToken("<|im_start|>", normalized=False, special=True)
        tokenizer.extra_special_tokens = [token]
        original_ids = set(tokenizer.all_special_ids)
        patch_mimo_tokenizer(tokenizer)
        self.assertEqual(set(tokenizer.all_special_ids), original_ids)
        self.assertEqual(
            str(tokenizer.extra_special_tokens[0]), self.renamed(str(token))
        )
        self.assertEqual(
            tokenizer._extra_special_tokens[0].__getstate__(),
            {**token.__getstate__(), "content": self.renamed(str(token))},
        )

    def test_non_mimo_unchanged_and_explicit_mimo_selection(self):
        tokenizer = self.make_tokenizer(Qwen2Tokenizer, mimo=False)
        vocab = tokenizer.get_vocab()
        patch_mimo_tokenizer(tokenizer)
        self.assertEqual(tokenizer.get_vocab(), vocab)
        self.assertEqual(tokenizer.chat_template, self.template)
        patch_mimo_tokenizer(tokenizer, is_mimo=True)
        self.assertEqual(
            tokenizer.convert_tokens_to_ids(self.renamed("<tool_call>")),
            vocab["<tool_call>"],
        )


if __name__ == "__main__":
    unittest.main()
