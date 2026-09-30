from __future__ import annotations

import contextlib
import copy
import json
import logging
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import jinja2
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    SglExt,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    should_include_usage,
    to_openai_style_logprobs,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False

    _TOOLCALL_TAGS = ("<tool_call>", "</tool_call>")
    # Untrusted-input roles whose <tool_call> tags get discretized: end-user
    # turns and tool results fed back in can both carry injected tags.
    _DISCRETIZE_ROLES = ("user", "tool")

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # Check if the model is a GPT-OSS model
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        self.use_dpsk_v32_encoding = self._use_dpsk_v32_encoding()

        # If encode("") returns specials, the discretized-encode site must pass
        # add_special_tokens=False to avoid a double BOS.
        try:
            self._tokenizer_auto_adds_specials = (
                self.tokenizer_manager.tokenizer is not None
                and len(self.tokenizer_manager.tokenizer.encode("")) > 0
            )
        except Exception as e:
            logger.warning(
                f"Failed to probe tokenizer for auto-added special tokens; "
                f"assuming it does: {e}"
            )
            self._tokenizer_auto_adds_specials = True

        # Cached "<" / ">" sub-token ids, computed once and reused for all tags.
        self._open_ids: Optional[List[int]] = None
        self._close_ids: Optional[List[int]] = None
        # Precompute the discretized "<" + inner + ">" id list for each fixed tag.
        self._discretized_tag_ids: Dict[str, List[int]] = {}
        try:
            for tag in self._TOOLCALL_TAGS:
                ids = self._precompute_discretized_ids(tag)
                if ids is not None:
                    self._discretized_tag_ids[tag] = ids
        except Exception as e:
            logger.warning(
                f"Failed to precompute discretized tool-call tag ids; "
                f"will encode them lazily: {e}"
            )
            self._discretized_tag_ids = {}

        # Used by the token-level gate; None when unavailable or parser is not mimo.
        self._toolcall_special_id: Optional[int] = None
        if self.tool_call_parser == "mimo":
            try:
                tokenizer = self.tokenizer_manager.tokenizer
                if tokenizer is not None:
                    tid = tokenizer.convert_tokens_to_ids("<tool_call>")
                    if isinstance(tid, int) and tid >= 0:
                        self._toolcall_special_id = tid
            except Exception as e:
                logger.warning(
                    f"Failed to resolve the mimo '<tool_call>' special token id; "
                    f"token-level tool-call gate will be inactive: {e}"
                )
                self._toolcall_special_id = None

        self._discretize_user_toolcall_tags = (
            envs.SGLANG_DISCRETIZE_USER_TOOLCALL_TAGS.get()
        )
        self._require_special_toolcall_token = (
            envs.SGLANG_REQUIRE_SPECIAL_TOOLCALL_TOKEN.get()
        )
        self._discretize_user_special_tokens = (
            envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKENS.get()
        )

        # Build the set of end-of-turn / structural special-token strings that
        # must be discretized when they appear inside untrusted user content, and
        # precompute their discretized ("<" + inner + ">") id lists. A string is
        # only activated if the tokenizer maps it to a single added-token id
        # (otherwise it is already treated as ordinary text and needs no fix).
        self._special_token_strs: tuple = ()
        if self._discretize_user_special_tokens:
            self._build_discretized_special_tokens()

        # All discretization switches/lists are fixed after __init__, so compute
        # the target tag/special-token set once and reuse it per request.
        self._cached_targets = self._compute_discretize_targets()

    def _precompute_discretized_ids(self, tag: str) -> Optional[List[int]]:
        """Return the discretized id list for a bracketed tag, i.e.
        encode("<") + encode(inner) + encode(">"), all with add_special_tokens
        =False so the single added-token fast path is bypassed. decode() of the
        result reconstructs the identical string. Returns None if the tokenizer
        is unavailable. "<" / ">" ids are cached across calls.
        """
        tokenizer = self.tokenizer_manager.tokenizer
        if tokenizer is None:
            return None
        if self._open_ids is None:
            self._open_ids = tokenizer.encode("<", add_special_tokens=False)
        if self._close_ids is None:
            self._close_ids = tokenizer.encode(">", add_special_tokens=False)
        inner_ids = tokenizer.encode(tag[1:-1], add_special_tokens=False)
        return self._open_ids + inner_ids + self._close_ids

    def _build_discretized_special_tokens(self) -> None:
        """Resolve which special-token strings must be discretized inside user
        content, and precompute each one's "<" + inner + ">" id list.

        Candidates come from two sources:
          1. env var SGLANG_DISCRETIZE_USER_SPECIAL_TOKEN_STRS (comma-separated).
          2. Model EOS tokens (model_config.hf_eos_token_id), reverse-mapped to
             their token strings so any model-specific end token is covered.

        A candidate is activated only if the tokenizer maps it to a *single*
        added-token id; otherwise it is already plain text and needs no fix.
        """
        tokenizer = self.tokenizer_manager.tokenizer
        if tokenizer is None:
            self._special_token_strs = ()
            return

        # Configurable candidate list (comma-separated env var); this env var's
        # default is the single source of truth for the built-in candidates.
        candidates: List[str] = list(
            envs.SGLANG_DISCRETIZE_USER_SPECIAL_TOKEN_STRS.get()
        )

        # Add model EOS token strings (reverse-mapped from ids) so we catch
        # model-specific end tokens like "<EOS>" / "<|I_am_end|>".
        try:
            eos_ids = getattr(
                self.tokenizer_manager.model_config, "hf_eos_token_id", None
            )
            if eos_ids:
                for eos_id in eos_ids:
                    try:
                        tok_str = tokenizer.convert_ids_to_tokens(eos_id)
                    except Exception:
                        tok_str = None
                    if isinstance(tok_str, str) and tok_str:
                        candidates.append(tok_str)
        except Exception as e:
            logger.warning(f"Failed to collect model EOS token strings: {e}")

        activated: List[str] = []
        for tok in dict.fromkeys(candidates):  # dedup, preserve order
            # Only discretize strings that (a) look like a bracketed tag and
            # (b) the tokenizer collapses into a single added-token id.
            if not (len(tok) >= 2 and tok.startswith("<") and tok.endswith(">")):
                continue
            try:
                ids = tokenizer.encode(tok, add_special_tokens=False)
            except Exception:
                continue
            if len(ids) != 1:
                # Already tokenized as ordinary sub-tokens; no injection risk.
                continue
            try:
                discretized = self._precompute_discretized_ids(tok)
            except Exception:
                continue
            if discretized is None:
                continue
            self._discretized_tag_ids[tok] = discretized
            activated.append(tok)

        self._special_token_strs = tuple(activated)
        if activated:
            logger.info(
                f"User-content special-token discretization active for: "
                f"{activated}"
            )

    def _compute_discretize_targets(self) -> tuple:
        """Compute all tag/special-token strings to discretize inside user
        content. Called once in __init__ since the inputs are fixed afterwards.

        Longest-first so that overlapping prefixes (e.g. "<tool_call>" vs "<")
        match the most specific tag first.
        """
        targets = set()
        if self._discretize_user_toolcall_tags:
            targets.update(self._TOOLCALL_TAGS)
        if self._discretize_user_special_tokens:
            targets.update(self._special_token_strs)
        return tuple(sorted(targets, key=len, reverse=True))

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatCompletionRequest with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _use_dpsk_v32_encoding(self) -> bool:
        has_chat_template = (
            self.tokenizer_manager.tokenizer is not None
            and self.tokenizer_manager.tokenizer.chat_template is not None
        )
        architectures = self.tokenizer_manager.model_config.hf_config.architectures
        is_dpsk_v32 = "DeepseekV3" in architectures[0] if architectures else False
        return not has_chat_template and is_dpsk_v32

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not request.tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in request.tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        # Validate tool definitions
        if not request.return_tokens_only:
            for i, tool in enumerate(request.tools or []):
                if tool.function.parameters is None:
                    continue
                try:
                    Draft202012Validator.check_schema(tool.function.parameters)
                except SchemaError as e:
                    return (
                        f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
                    )

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)

        if request.return_tokens_only:
            has_mm_data = (
                processed_messages.image_data
                or processed_messages.video_data
                or processed_messages.audio_data
            )
            if has_mm_data and is_multimodal:
                adapted_request = GenerateReqInput(
                    text=processed_messages.prompt,
                    image_data=processed_messages.image_data,
                    video_data=processed_messages.video_data,
                    audio_data=processed_messages.audio_data,
                    modalities=processed_messages.modalities,
                    return_tokens_only=True,
                )
            else:
                adapted_request = GenerateReqInput(
                    input_ids=processed_messages.prompt_ids,
                )
            return adapted_request, request

        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # Handle single vs multiple requests
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            rid=request.rid,
            extra_key=self._compute_extra_key(request),
            require_reasoning=self._get_reasoning_from_request(request),
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
        )

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss:
            request.skip_special_tokens = False

        self._patch_mistral_skip_special_tokens(request)

        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
            # Handle JSON schema constraint directly for required or named tool choice
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(
                request, tools, is_multimodal,
                return_tokens_only=getattr(request, "return_tokens_only", False),
            )
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _render_prompt_ids(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        template_kwargs: Dict,
        return_tokens_only: bool = False,
    ) -> List[int]:
        # Non-discretize path keeps the branch's normal single-pass encoding
        # (apply_chat_template tokenize=True), so default behavior is unchanged.
        # A non-empty target set is the single source of truth for whether any
        # discretization applies (it already encodes both switches and whether
        # any special tokens were actually activated).
        tokenizer = self.tokenizer_manager.tokenizer
        discretize_enabled = bool(self._cached_targets)
        if not discretize_enabled:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                return_dict=False,
                **template_kwargs,
            )

        # The discretize path renders to a string first, then encodes segment by
        # segment so that targeted tags/special tokens (e.g. <tool_call>,
        # <|im_end|>) appearing inside USER and TOOL content are split into
        # ordinary sub-tokens instead of the single added-token/EOS id, while
        # template structure and tool definitions keep their normal encoding.
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            **template_kwargs,
        )
        if return_tokens_only and self.tokenizer_manager.model_config.is_multimodal:
            # tokenize_proxy path for multimodal models: use plain encode to match
            # the inference path (which re-tokenizes via _tokenize_texts).
            result = tokenizer.encode(rendered, add_special_tokens=False)
        else:
            result = self._encode_with_user_tags_discretized(rendered, messages)
        return result

    def _encode_with_user_tags_discretized(
        self, text: str, messages: List[Dict]
    ) -> List[int]:
        tokenizer = self.tokenizer_manager.tokenizer
        # The rendered string already contains any template-emitted specials
        # (e.g. BOS as text). Segments are encoded independently and concatenated,
        # so specials must never be injected per-segment. The single exception is
        # a tokenizer that does NOT auto-add specials on encode(): to match the
        # non-discretize path, the leading chunk prepends specials once.
        lead_add_specials = not self._tokenizer_auto_adds_specials

        spans = self._locate_user_content_spans(text, messages)
        if not spans:
            return tokenizer.encode(text, add_special_tokens=lead_add_specials)

        ids: List[int] = []
        cursor = 0
        first_chunk = True
        for start, end in spans:
            if start > cursor:
                ids += tokenizer.encode(
                    text[cursor:start],
                    add_special_tokens=(lead_add_specials and first_chunk),
                )
            ids += self._encode_segment_discretizing_tags(text[start:end])
            first_chunk = False
            cursor = end
        if cursor < len(text):
            ids += tokenizer.encode(
                text[cursor:],
                add_special_tokens=(lead_add_specials and first_chunk),
            )
        return ids

    def _locate_user_content_spans(
        self, text: str, messages: List[Dict]
    ) -> List[tuple]:
        # Locate each untrusted-role message's raw content verbatim inside the
        # rendered prompt. If a content string is not found (e.g. multimodal
        # content the template rewrote), that message is skipped and encoded
        # normally.
        spans = []
        search_from = 0
        for msg in messages:
            if msg.get("role") not in self._DISCRETIZE_ROLES:
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            idx = text.find(content, search_from)
            if idx == -1:
                continue
            spans.append((idx, idx + len(content)))
            search_from = idx + len(content)
        return spans

    def _encode_segment_discretizing_tags(self, seg: str) -> List[int]:
        tokenizer = self.tokenizer_manager.tokenizer
        # Targets are the precomputed union of tool-call tags and end-of-turn/
        # structural special tokens (whichever discretization switches are on),
        # ordered longest-first so the most specific tag wins on overlapping
        # prefixes. If empty, the loop below encodes the segment verbatim -- no
        # implicit fallback, so nothing is discretized unless explicitly enabled.
        targets = self._cached_targets
        ids: List[int] = []
        i = 0
        n = len(seg)
        while i < n:
            hit = next((t for t in targets if seg.startswith(t, i)), None)
            if hit:
                # Encode e.g. "<tool_call>" as "<" + "tool_call" + ">" and
                # "<|im_end|>" as "<" + "|im_end|" + ">" so the added-token
                # fast-path is bypassed. decode() reconstructs the identical
                # string, so the model sees the same text (but as ordinary
                # sub-tokens rather than a single special/EOS id).
                hit_ids = self._discretized_tag_ids.get(hit)
                if hit_ids is None:
                    inner = hit[1:-1]
                    hit_ids = (
                        tokenizer.encode("<", add_special_tokens=False)
                        + tokenizer.encode(inner, add_special_tokens=False)
                        + tokenizer.encode(">", add_special_tokens=False)
                    )
                    self._discretized_tag_ids[hit] = hit_ids
                ids += hit_ids
                i += len(hit)
                continue
            next_positions = [
                pos for pos in (seg.find(t, i) for t in targets) if pos != -1
            ]
            nxt = min(next_positions) if next_positions else n
            ids += tokenizer.encode(seg[i:nxt], add_special_tokens=False)
            i = nxt
        return ids

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
        return_tokens_only: bool = False,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        if self.use_dpsk_v32_encoding:
            thinking_mode = (
                "thinking"
                if (request.chat_template_kwargs or {}).get("thinking")
                else "chat"
            )
            messages = request.messages
            messages = [msg.model_dump() for msg in messages]

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.use_dpsk_v32_encoding,
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]
            real_input = encode_messages(messages, thinking_mode=thinking_mode)
            prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for message in request.messages:
                if message.content is None:
                    message.content = ""
                msg_dict = message.model_dump()

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                # per the Transformers docs & maintainers, tool call arguments in
                # assistant-role messages with tool_calls need to be dicts not JSON str -
                # this is how tool-use chat templates will expect them moving forwards
                # so, for messages that have tool_calls, parse the string (which we get
                # from openAI format) to dict
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            try:
                                parsed_args = orjson.loads(
                                    item["function"]["arguments"]
                                )
                            except orjson.JSONDecodeError:
                                # Freeform (custom) tool calls carry a raw
                                # string (e.g. code), not JSON; chat templates
                                # render string arguments verbatim.
                                continue
                            if isinstance(parsed_args, dict):
                                item["function"]["arguments"] = parsed_args

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            try:
                prompt_ids = self._render_prompt_ids(
                    openai_compatible_messages,
                    tools,
                    extra_template_kwargs,
                    return_tokens_only=return_tokens_only,
                )
            except Exception as e:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    prompt_ids = self._render_prompt_ids(
                        openai_compatible_messages,
                        tools,
                        extra_template_kwargs,
                        return_tokens_only=return_tokens_only,
                    )
                except jinja2.TemplateError as template_error:
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # should be treated as client errors (400 BadRequest)
                    raise ValueError(str(template_error)) from template_error
                except Exception as e:
                    # Any other failure to apply the chat template is most likely
                    # caused by an invalid format of the input messages, so treat it
                    # as a client error (400 BadRequest) rather than a server error.
                    raise ValueError(
                        f"Failed to apply chat template to the request: {str(e)}. "
                        "This is likely due to an invalid format of the input messages. "
                        "Please check your input."
                    ) from e

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_reasoning_from_request(
                request
            ) and self.reasoning_parser not in ["qwen3", "qwen3-thinking", "glm4"]:
                # qwen3 and glm4 think internally without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse, dict]:
        """Handle streaming chat completion request"""
        if request.return_tokens_only:
            from sglang.srt.entrypoints.openai.utils import _truncate_tokens

            tm = self.tokenizer_manager
            if adapted_request.contains_mm_input():
                async with contextlib.aclosing(
                    tm.generate_request(adapted_request, raw_request)
                ) as gen:
                    ret = await gen.__anext__()
                prompt_ids = ret["input_ids"]
            else:
                prompt_ids = adapted_request.input_ids
            if tm.server_args.allow_auto_truncate and tm.max_req_input_len is not None:
                prompt_ids = _truncate_tokens(prompt_ids, tm.max_req_input_len)
            return {"tokens": prompt_ids, "count": len(prompt_ids)}

        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}
        reasoning_tool_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}
        seen_special_toolcall = {}
        toolcall_gate_active = self._toolcall_gate_active()
        toolcall_special_token_id = self._toolcall_special_id
        toolcall_incremental_output = (
            self.tokenizer_manager.server_args.incremental_streaming_output
        )
        toolcall_scanned_len = {}

        # Usage tracking
        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        image_tokens = {}
        audio_tokens = {}
        video_tokens = {}

        stream_started = False
        try:
            enable_cache_report = self.tokenizer_manager.server_args.enable_cache_report
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )
            should_parse_tool_calls = bool(
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
            )
            history_tool_calls_cnt = (
                self._get_history_tool_calls_cnt(request)
                if should_parse_tool_calls
                else 0
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)
                image_tokens[index] = content["meta_info"].get("image_tokens", 0)
                audio_tokens[index] = content["meta_info"].get("audio_tokens", 0)
                video_tokens[index] = content["meta_info"].get("video_tokens", 0)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        )
                    n_prev_tokens[index] = total_output_logprobs

                # Attach logprobs to the first chunk emitted this step (reasoning,
                # tool-call, or content) so they aren't dropped when a parser is active
                # nor duplicated across chunks; flush any leftover at the end.
                remaining_logprobs = choice_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    # If the abort is from scheduler.
                    if finish_reason_type == "abort":
                        # FINISH_ABORT may carry status_code=None; .get()'s default only covers a missing key.
                        code = (
                            finish_reason.get("status_code")
                            or HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    else:
                        finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )

                    if continuous_usage_stats:
                        chunk.usage = UsageProcessor.calculate_token_usage(
                            prompt_tokens=prompt_tokens.get(index, 0),
                            completion_tokens=0,
                            cached_tokens=UsageProcessor.cached_details_if_enabled(
                                cached_tokens.get(index, 0), enable_cache_report
                            ),
                        )

                    yield f"data: {chunk.model_dump_json()}\n\n"
                    stream_started = True

                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta

                if toolcall_gate_active and not seen_special_toolcall.get(index, False):
                    output_ids = content.get("output_ids") or []
                    if toolcall_incremental_output:
                        tail = output_ids
                    else:
                        prev = toolcall_scanned_len.get(index, 0)
                        tail = output_ids[prev:]
                        toolcall_scanned_len[index] = len(output_ids)
                    if toolcall_special_token_id in tail:
                        seen_special_toolcall[index] = True

                # Apply on BOTH the reasoning and main paths so an injected
                # <tool_call> cannot slip through the reasoning parser.
                gate_blocked = toolcall_gate_active and not seen_special_toolcall.get(
                    index, False
                )

                # Handle reasoning content
                if self.reasoning_parser and request.separate_reasoning:
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text:
                        if should_parse_tool_calls:
                            async for chunk in self._process_tool_call_stream(
                                index,
                                reasoning_text,
                                reasoning_tool_parser_dict,
                                content,
                                request,
                                has_tool_calls,
                                continuous_usage_stats,
                                emit_text_as_reasoning=True,
                                history_tool_calls_cnt=history_tool_calls_cnt,
                                gate_blocked=gate_blocked,
                            ):
                                if chunk:
                                    yield chunk
                        else:
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(reasoning_content=reasoning_text),
                                finish_reason=None,
                                logprobs=remaining_logprobs,
                            )
                            remaining_logprobs = None
                            chunk = ChatCompletionStreamResponse(
                                id=content["meta_info"]["id"],
                                created=int(time.time()),
                                choices=[choice_data],
                                model=request.model,
                            )

                            # Add usage stats if continuous_usage_stats is enabled
                            if continuous_usage_stats:
                                chunk.usage = UsageProcessor.calculate_token_usage(
                                    prompt_tokens=prompt_tokens.get(index, 0),
                                    reasoning_tokens=reasoning_tokens.get(index, 0),
                                    completion_tokens=completion_tokens.get(index, 0),
                                    cached_tokens=UsageProcessor.cached_details_if_enabled(
                                        cached_tokens.get(index, 0),
                                        enable_cache_report,
                                    ),
                                )

                            yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle tool calls
                if should_parse_tool_calls:
                    if delta:
                        async for chunk in self._process_tool_call_stream(
                            index,
                            delta,
                            parser_dict,
                            content,
                            request,
                            has_tool_calls,
                            continuous_usage_stats,
                            history_tool_calls_cnt=history_tool_calls_cnt,
                            gate_blocked=gate_blocked,
                        ):
                            if chunk:
                                yield chunk

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type is not None:
                        for parser_pool in (parser_dict, reasoning_tool_parser_dict):
                            if index not in parser_pool:
                                continue
                            parser = parser_pool[index]
                            remaining_chunk = self._check_for_unstreamed_tool_args(
                                parser, content, request, index
                            )
                            if remaining_chunk:
                                yield remaining_chunk

                else:
                    # Regular content
                    if delta:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=remaining_logprobs,
                        )
                        remaining_logprobs = None
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # Add usage stats if continuous_usage_stats is enabled
                        if continuous_usage_stats:
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                                cached_tokens=UsageProcessor.cached_details_if_enabled(
                                    cached_tokens.get(index, 0), enable_cache_report
                                ),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

                # Flush logprobs still unattached this step — only when a parser is
                # active, since _process_tool_call_stream may consume the delta and
                # emit no content chunk. On the plain path an empty-delta step has
                # no chunk to attach to either way, and a standalone empty-delta
                # logprobs chunk is not a shape clients expect.
                if remaining_logprobs is not None and (
                    self.reasoning_parser or self.tool_call_parser
                ):
                    flush_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(),
                                finish_reason=None,
                                logprobs=remaining_logprobs,
                            )
                        ],
                        model=request.model,
                    )
                    if continuous_usage_stats:
                        flush_chunk.usage = UsageProcessor.calculate_token_usage(
                            prompt_tokens=prompt_tokens.get(index, 0),
                            reasoning_tokens=reasoning_tokens.get(index, 0),
                            completion_tokens=completion_tokens.get(index, 0),
                            cached_tokens=UsageProcessor.cached_details_if_enabled(
                                cached_tokens.get(index, 0), enable_cache_report
                            ),
                        )
                    yield f"data: {flush_chunk.model_dump_json()}\n\n"
                    remaining_logprobs = None

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            if request.return_routed_experts and routed_experts:
                # Get first non-None routed_experts value
                first_routed_experts = next(
                    (v for v in routed_experts.values() if v is not None), None
                )
                if first_routed_experts is not None:
                    routed_experts_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],  # sglext is at response level
                        model=request.model,
                        sglext=SglExt(routed_experts=first_routed_experts),
                    )
                    yield f"data: {routed_experts_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if include_usage:
                total_image_tokens = sum(
                    tok for idx, tok in image_tokens.items() if idx % request.n == 0
                )
                total_audio_tokens = sum(
                    tok for idx, tok in audio_tokens.items() if idx % request.n == 0
                )
                total_video_tokens = sum(
                    tok for idx, tok in video_tokens.items() if idx % request.n == 0
                )
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=enable_cache_report,
                    image_tokens=total_image_tokens,
                    audio_tokens=total_audio_tokens,
                    video_tokens=total_video_tokens,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse, dict]:
        """Handle non-streaming chat completion request"""
        if request.return_tokens_only:
            from sglang.srt.entrypoints.openai.utils import _truncate_tokens

            tm = self.tokenizer_manager
            if adapted_request.contains_mm_input():
                async with contextlib.aclosing(
                    tm.generate_request(adapted_request, raw_request)
                ) as gen:
                    ret = await gen.__anext__()
                prompt_ids = ret["input_ids"]
            else:
                prompt_ids = adapted_request.input_ids
            if tm.server_args.allow_auto_truncate and tm.max_req_input_len is not None:
                prompt_ids = _truncate_tokens(prompt_ids, tm.max_req_input_len)
            return {"tokens": prompt_ids, "count": len(prompt_ids)}

        try:
            async with contextlib.aclosing(
                self.tokenizer_manager.generate_request(
                    adapted_request, raw_request
                )
            ) as gen:
                ret = await gen.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if (
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
                and not self._toolcall_gate_blocks(ret_item.get("output_ids"))
            ):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                (
                    tool_calls,
                    text,
                    reasoning_text,
                    finish_reason,
                ) = self._process_tool_calls_with_reasoning_fallback(
                    text,
                    reasoning_text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage (multimodal token counts are added below)
        image_tokens = sum(
            ret[i]["meta_info"].get("image_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        audio_tokens = sum(
            ret[i]["meta_info"].get("audio_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        video_tokens = sum(
            ret[i]["meta_info"].get("video_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                for top_token, top_logprob in logprobs.top_logprobs[token_idx].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        if self.tool_call_parser != "kimi_k2":
            # A simple uuid is sufficient for all models except for Kimi-K2.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        else:
            # Align with Kimi-K2 format: functions.{name}:{index}
            # Kimi-K2 allows multiple tool_calls in one message; SGLang sets call_item.tool_index to the *local* position inside that message.
            # Therefore, the index must be corrected by using `history_tool_calls_cnt + call_item.tool_index` to ensure globally unique and properly ordered.
            tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt+call_item.tool_index}"
            logger.debug(
                f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
            )
            return tool_call_id

    def _toolcall_gate_active(self) -> bool:
        return (
            self._require_special_toolcall_token
            and self._toolcall_special_id is not None
        )

    def _toolcall_gate_blocks(self, output_ids: Optional[List[int]]) -> bool:
        # Token-level gate (mimo only): when enabled, a tool_call is honored only
        # if the output contains the single "<tool_call>" token id. A tool_call
        # spelled out as discrete tokens (e.g. a repeated/injected string) lacks
        # that id and is treated as normal text. Global judgement: presence of
        # the id anywhere means "genuine call present". Fails open (does not
        # block) when the switch is off, parser is not mimo, ids/id unavailable.
        if not self._toolcall_gate_active():
            return False
        if not output_ids:
            return False
        return self._toolcall_special_id not in output_ids

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response"""

        # Handle required or named tool choice
        if tool_choice == "required" or (
            isinstance(tool_choice, ToolChoice) and tool_choice.type == "function"
        ):
            # Set finish reason to tool_calls since we're processing tool calls
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                # For required tool choice, we expect a JSON array of tool calls
                tool_call_data = orjson.loads(text)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    # Create a ToolCallItem from the JSON data
                    call_info = ToolCallItem(
                        tool_index=i,  # Use the loop index as tool_index
                        name=tool["name"],
                        parameters=json.dumps(tool["parameters"], ensure_ascii=False),
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=json.dumps(
                                    tool["parameters"], ensure_ascii=False
                                ),
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except json.JSONDecodeError as e:
                logger.error(f"Tool call parsing error: {e}")
                return ToolCallProcessingResult(None, text, finish_reason)

        # Use parser since output is not constrained by JSON schema
        parser = FunctionCallParser(tools, self.tool_call_parser)
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=getattr(call_info, "tool_index", None),
                            function=FunctionResponse(
                                name=call_info.name, arguments=call_info.parameters
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, text, finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                # Return error but don't fail the whole request
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_tool_calls_with_reasoning_fallback(
        self,
        text: str,
        reasoning_text: Optional[str],
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> tuple[Optional[List[ToolCall]], str, Optional[str], Dict[str, Any]]:
        tool_calls, text, finish_reason = self._process_tool_calls(
            text,
            tools,
            finish_reason,
            tool_choice,
            history_tool_calls_cnt,
        )
        if not tool_calls and reasoning_text:
            reasoning_tool_calls, cleaned_reasoning, finish_reason = (
                self._process_tool_calls(
                    reasoning_text,
                    tools,
                    finish_reason,
                    tool_choice,
                    history_tool_calls_cnt,
                )
            )
            if reasoning_tool_calls:
                tool_calls = reasoning_tool_calls
                reasoning_text = cleaned_reasoning

        return tool_calls, text, reasoning_text, finish_reason

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:total_output_logprobs
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:total_output_logprobs
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
        return idx

    def _patch_mistral_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Mistral uses special tokens ([THINK]/[/THINK]) for reasoning markers,
        which get stripped when skip_special_tokens=True."""
        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Judge whether the request needs reasoning for hybrid reasoning models
        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False
        if self.reasoning_parser in ["deepseek-v3"]:
            # Models that require explicit enable thinking (thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("thinking") is True
            )
        if self.reasoning_parser in ["kimi_k2"]:
            # Models that thinking by default, and can be disabled by setting thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("thinking") is not False
            )
        if self.reasoning_parser in [
            "qwen3",
            "glm45",
            "nemotron_3",
            "interns1",
            "mimo",
        ]:
            # Models that thinking by default, and can be disabled by setting enable_thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("enable_thinking") is not False
            )
        if self.reasoning_parser in ["mistral"]:
            # Mistral models only reason when reasoning_effort is explicitly
            # set to a value other than None/"none" (typically "high").
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )
        return True  # default

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
        emit_text_as_reasoning: bool = False,
        history_tool_calls_cnt: Optional[int] = None,
        gate_blocked: bool = False,
    ):
        """Process tool calls in streaming response"""
        enable_cache_report = self.tokenizer_manager.server_args.enable_cache_report
        # Gate blocked (see _toolcall_gate_blocks): emit delta as text, not a call.
        if gate_blocked:
            if delta:
                delta_message = (
                    DeltaMessage(reasoning_content=delta)
                    if emit_text_as_reasoning
                    else DeltaMessage(content=delta)
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=index,
                    delta=delta_message,
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[choice_data],
                    model=request.model,
                )

                # Add usage stats if continuous_usage_stats is enabled
                if continuous_usage_stats:
                    prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                    completion_tokens = content["meta_info"].get("completion_tokens", 0)
                    reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                    chunk.usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        reasoning_tokens=reasoning_tokens,
                        cached_tokens=UsageProcessor.cached_details_if_enabled(
                            content["meta_info"].get("cached_tokens", 0),
                            enable_cache_report,
                        ),
                    )

                yield f"data: {chunk.model_dump_json()}\n\n"
            return

        if index not in parser_dict:
            # Use JSON detector directly for required or named tool choice
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=request.tools,
                    tool_call_parser=self.tool_call_parser,
                )

        parser = parser_dict[index]

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, request.tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)

        # Yield normal text
        if normal_text:
            delta_message = (
                DeltaMessage(reasoning_content=normal_text)
                if emit_text_as_reasoning
                else DeltaMessage(content=normal_text)
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=delta_message,
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=UsageProcessor.cached_details_if_enabled(
                        content["meta_info"].get("cached_tokens", 0),
                        enable_cache_report,
                    ),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        if history_tool_calls_cnt is None:
            history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=UsageProcessor.cached_details_if_enabled(
                        content["meta_info"].get("cached_tokens", 0),
                        enable_cache_report,
                    ),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        # Freeform (custom) tool calls store arguments as a raw string that
        # is streamed verbatim, so it must not be JSON-encoded here.
        expected_call = (
            expected_args
            if isinstance(expected_args, str)
            else json.dumps(expected_args, ensure_ascii=False)
        )
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
