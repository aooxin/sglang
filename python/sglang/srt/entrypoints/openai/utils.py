import logging
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
    StreamOptions,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states


def should_include_usage(
    stream_options: StreamOptions | None, stream_response_default_include_usage: bool
) -> tuple[bool, bool]:
    # When stream_options are specified in the request
    if stream_options:
        include_usage = (
            stream_options.include_usage or stream_response_default_include_usage
        )
        continuous_usage_stats = bool(stream_options.continuous_usage_stats)
    else:
        include_usage, continuous_usage_stats = (
            stream_response_default_include_usage,
            False,
        )
    return include_usage, continuous_usage_stats


def process_routed_experts_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[str]:
    """Process routed experts from a ret item in non-streaming response."""
    if not getattr(request, "return_routed_experts", False):
        return None
    return ret_item["meta_info"].get("routed_experts", None)


def process_cached_tokens_details_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[CachedTokensDetails]:
    """Process cached tokens details from a ret item in non-streaming response."""
    if not getattr(request, "return_cached_tokens_details", False):
        return None

    details = ret_item["meta_info"].get("cached_tokens_details", None)
    if details is None:
        return None

    # Check if L3 storage fields are present
    if "storage" in details:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
            storage=details.get("storage", 0),
            storage_backend=details.get("storage_backend"),
        )
    else:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
        )


def tokenize_or_passthrough(
    tokenizer,
    prompts: Union[str, List[int], List[str], List[List[int]]],
    max_model_len: int,
) -> Dict[str, Any]:
    """Tokenize prompts or pass through pre-tokenized input_ids.

    Returns a dict with "tokens", "count", and "max_model_len" suitable for
    return_tokens_only responses. Handles both single and batch inputs.
    """
    if isinstance(prompts, str):
        tokens = tokenizer.encode(prompts)
        return {"tokens": tokens, "count": len(tokens), "max_model_len": max_model_len}

    if not isinstance(prompts, list):
        raise ValueError(f"prompts must be str or list, got {type(prompts).__name__}")

    # Empty batch
    if len(prompts) == 0:
        return {"tokens": [], "count": [], "max_model_len": max_model_len}

    # Single pre-tokenized sequence: List[int]
    is_single = isinstance(prompts[0], int)
    if is_single:
        return {
            "tokens": prompts,
            "count": len(prompts),
            "max_model_len": max_model_len,
        }

    # Batch: List[str] or List[List[int]]
    all_tokens = []
    for p in prompts:
        if isinstance(p, str):
            all_tokens.append(tokenizer.encode(p))
        elif isinstance(p, list):
            all_tokens.append(p)
        else:
            raise ValueError(
                f"Each prompt must be str or list of int, got {type(p).__name__}"
            )
    return {
        "tokens": all_tokens,
        "count": [len(t) for t in all_tokens],
        "max_model_len": max_model_len,
    }
