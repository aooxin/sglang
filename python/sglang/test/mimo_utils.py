"""Readable MiMo parser fixtures using the runtime control-token spellings."""

from sglang.srt.utils.patch_tokenizer import mimo_token


def mimo_tool_call_text(text: str) -> str:
    """Convert complete fixture text before splitting it into streaming chunks."""
    for token in ("<tool_call>", "</tool_call>"):
        text = text.replace(token, mimo_token(token))
    return text
