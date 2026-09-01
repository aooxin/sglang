import asyncio
import logging
import os
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, List, Union

from fastapi import HTTPException, Request
from pydantic import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.json_response import orjson_response

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIServingTokenize(OpenAIServingBase):
    """Handler for /v1/tokenize requests"""

    def _request_id_prefix(self) -> str:
        return "tok-"

    def _convert_to_internal_request(
        self, request: TokenizeRequest, raw_request: Request
    ) -> tuple[TokenizeRequest, TokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: TokenizeRequest,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse]:
        try:
            tokenizer = self.tokenizer_manager.tokenizer
            max_model_len = getattr(tokenizer, "model_max_length", -1)

            if isinstance(request.prompt, str):
                token_ids = tokenizer.encode(
                    request.prompt,
                    add_special_tokens=request.add_special_tokens,
                )
                tokens = token_ids
                count = len(token_ids)
            elif isinstance(request.prompt, list):
                token_ids_list = [
                    tokenizer.encode(
                        text, add_special_tokens=request.add_special_tokens
                    )
                    for text in request.prompt
                ]
                tokens = token_ids_list
                count = [len(ids) for ids in token_ids_list]
            else:
                return self.create_error_response(
                    f"Invalid prompt type: {type(request.prompt)}. Expected str or List[str]."
                )

            return TokenizeResponse(
                tokens=tokens, count=count, max_model_len=max_model_len
            )
        except Exception as e:
            logger.error("Error during tokenization", exc_info=True)
            return self.create_error_response(
                f"Internal server error during tokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


class OpenAIServingDetokenize(OpenAIServingBase):
    """Handler for /v1/detokenize requests"""

    def _request_id_prefix(self) -> str:
        return "detok-"

    def _convert_to_internal_request(
        self, request: DetokenizeRequest, raw_request: Request
    ) -> tuple[DetokenizeRequest, DetokenizeRequest]:
        return request, request

    async def _handle_non_streaming_request(
        self,
        adapted_request: DetokenizeRequest,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> Union[DetokenizeResponse, ErrorResponse]:
        try:
            tokenizer = self.tokenizer_manager.tokenizer

            if (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], int)
            ):
                if not all(isinstance(t, int) for t in request.tokens):
                    return self.create_error_response(
                        "Invalid input: 'tokens' must be a list of integers."
                    )
                tokens_to_decode = [int(t) for t in request.tokens]
                text = tokenizer.decode(
                    tokens_to_decode, skip_special_tokens=request.skip_special_tokens
                )
                text_out: Union[str, List[str]] = text
            elif (
                isinstance(request.tokens, list)
                and request.tokens
                and isinstance(request.tokens[0], list)
            ):
                texts: List[str] = []
                for token_list in request.tokens:
                    if not all(isinstance(t, int) for t in token_list):
                        return self.create_error_response(
                            f"Invalid input: Sublist in 'tokens' must contain only integers. Found: {token_list}"
                        )
                    decoded_text = tokenizer.decode(
                        [int(t) for t in token_list],
                        skip_special_tokens=request.skip_special_tokens,
                    )
                    texts.append(decoded_text)
                text_out = texts
            elif isinstance(request.tokens, list) and not request.tokens:
                text_out = ""
            else:
                return self.create_error_response(
                    f"Invalid tokens type: {type(request.tokens)}. Expected List[int] or List[List[int]]."
                )

            return DetokenizeResponse(text=text_out)
        except Exception as e:
            logger.error("Error during detokenization", exc_info=True)
            if "decode" in str(e).lower():
                return self.create_error_response(
                    f"Error decoding tokens: {e}. Input tokens might be invalid for the model.",
                    err_type="DecodeError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            return self.create_error_response(
                f"Internal server error during detokenization: {e}",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


class OpenAIServingTokenizeProxy:
    """Handler for /v1/tokenize_proxy.

    Proxies tokenize requests to inference endpoints via return_tokens_only,
    ensuring 100% consistency with the tokenization used during inference.
    """

    def __init__(
        self,
        tokenizer_manager: "TokenizerManager",
        openai_serving_chat: Any,
        openai_serving_completion: Any,
        generate_handler: Callable,
    ):
        self.tokenizer_manager = tokenizer_manager
        self.openai_serving_chat = openai_serving_chat
        self.openai_serving_completion = openai_serving_completion
        self.generate_handler = generate_handler

    async def handle_request(self, raw_request: Request):
        t_start = time.perf_counter()

        body = await raw_request.json()
        t_parse = time.perf_counter()

        body["return_tokens_only"] = True
        need_hash = body.pop("return_hash_values", False)
        rid = body.get("rid") or f"tokproxy-{uuid.uuid4().hex[:8]}"

        # Determine endpoint: prefer X-Original-Path header, fall back to body fields.
        original_path = raw_request.headers.get("x-original-path", "")
        if original_path.startswith("/v1/chat"):
            endpoint = "chat"
        elif original_path.startswith("/v1/completions"):
            endpoint = "completions"
        elif original_path.startswith("/generate") or original_path.startswith(
            "/v1/generate"
        ):
            endpoint = "generate"
        elif "messages" in body:
            endpoint = "chat"
        elif "prompt" in body:
            endpoint = "completions"
        elif "text" in body:
            endpoint = "generate"
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine endpoint: set 'X-Original-Path' header to one of /v1/chat/completions, /v1/completions, /generate; or include 'messages', 'prompt', or 'text' in the request body",
            )

        try:
            if endpoint == "chat":
                body.setdefault("max_tokens", 1)
                request = ChatCompletionRequest(**body)
                t_validate = time.perf_counter()
                result = await self.openai_serving_chat.handle_request(
                    request, raw_request
                )
            elif endpoint == "completions":
                body.setdefault("max_tokens", 1)
                request = CompletionRequest(**body)
                t_validate = time.perf_counter()
                result = await self.openai_serving_completion.handle_request(
                    request, raw_request
                )
            else:
                request = GenerateReqInput(**body)
                t_validate = time.perf_counter()
                result = await self.generate_handler(request, raw_request)
        except HTTPException:
            raise
        except (TypeError, ValueError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        t_tokenize = time.perf_counter()

        if not isinstance(result, dict):
            logger.debug(
                "[tokenize_proxy] rid=%s endpoint=%s error_response "
                "parse=%.1fms validate=%.1fms tokenize=%.1fms total=%.1fms",
                rid,
                endpoint,
                (t_parse - t_start) * 1000,
                (t_validate - t_parse) * 1000,
                (t_tokenize - t_validate) * 1000,
                (t_tokenize - t_start) * 1000,
            )
            return result

        # If auto_truncate is not enabled and tokens exceed max_req_input_len,
        # return empty tokens without computing hash.
        # Uses the same condition as scheduler's validate_input_length:
        # len(input_ids) >= max_req_input_len
        if not self.tokenizer_manager.server_args.allow_auto_truncate:
            max_len = self.tokenizer_manager.max_req_input_len
            tokens = result["tokens"]
            is_batch = isinstance(tokens, list) and tokens and isinstance(tokens[0], list)
            exceeded = (
                any(len(t) >= max_len for t in tokens)
                if is_batch
                else len(tokens) >= max_len
            )
            if exceeded:
                n = len(tokens) if is_batch else 0
                result["tokens"] = [[] for _ in range(n)] if is_batch else []
                result["count"] = [0] * n if is_batch else 0
                if need_hash:
                    result["hash_values"] = [[] for _ in range(n)] if is_batch else []
                result["max_model_len"] = self.tokenizer_manager.model_config.context_len
                t_end = time.perf_counter()
                logger.debug(
                    "[tokenize_proxy] pid=%d rid=%s endpoint=%s tokens=0 "
                    "exceeded_max_req_input_len=%d "
                    "parse=%.1fms validate=%.1fms tokenize=%.1fms total=%.1fms",
                    os.getpid(),
                    rid,
                    endpoint,
                    max_len,
                    (t_parse - t_start) * 1000,
                    (t_validate - t_parse) * 1000,
                    (t_tokenize - t_validate) * 1000,
                    (t_end - t_start) * 1000,
                )
                return orjson_response(result)

        # Compute hash values if requested
        if need_hash:
            from sglang.srt.mem_cache.hicache_storage import compute_hash_values
            from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

            server_args = self.tokenizer_manager.server_args
            page_size = server_args.page_size
            spec_algorithm = SpeculativeAlgorithm.from_string(
                server_args.speculative_algorithm
            )
            if spec_algorithm.is_eagle():
                step = (
                    server_args.speculative_num_steps
                    if server_args.enable_multi_layer_eagle
                    else 1
                )
            else:
                step = 0

            tokens = result["tokens"]
            if not tokens:
                result["hash_values"] = []
            elif isinstance(tokens[0], list):
                result["hash_values"] = await asyncio.to_thread(
                    lambda: [
                        compute_hash_values(t, page_size, step=step) for t in tokens
                    ]
                )
            else:
                result["hash_values"] = await asyncio.to_thread(
                    compute_hash_values, tokens, page_size, step=step
                )

        t_hash = time.perf_counter()

        result["max_model_len"] = self.tokenizer_manager.model_config.context_len
        t_end = time.perf_counter()

        token_count = result["count"] if isinstance(result["count"], int) else sum(result["count"])
        logger.debug(
            "[tokenize_proxy] pid=%d rid=%s endpoint=%s tokens=%d "
            "parse=%.1fms validate=%.1fms tokenize=%.1fms hash=%.1fms total=%.1fms",
            os.getpid(),
            rid,
            endpoint,
            token_count,
            (t_parse - t_start) * 1000,
            (t_validate - t_parse) * 1000,
            (t_tokenize - t_validate) * 1000,
            (t_hash - t_tokenize) * 1000,
            (t_end - t_start) * 1000,
        )
        return orjson_response(result)
