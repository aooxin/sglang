import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


@lru_cache(maxsize=1)
def _get_logit_processor_collector_impl():
    from sglang.srt.observability.metrics_collector import LogitProcessorCollector

    return LogitProcessorCollector()


def _is_logit_processor_metrics_rank() -> bool:
    try:
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        return get_attention_tp_rank() == 0
    except (AssertionError, ValueError):
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank() == 0
        except (AssertionError, ValueError):
            return True


def _get_logit_processor_collector():
    try:
        from sglang.srt.server_args import get_global_server_args

        if not get_global_server_args().enable_metrics:
            return None
    except ValueError:
        return None
    if not _is_logit_processor_metrics_rank():
        return None
    return _get_logit_processor_collector_impl()


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence: List[int] = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = tuple(sequence[-(ngram_size - 1) :])
            else:
                current_prefix = tuple()

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits


class MiMoV2FlashNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for MiMoV2Flash outputs."""

    EOS_TOKEN_ID: int = 151645
    NGRAM_SIZE: int = envs.SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_NGRAM_SIZE.get()
    WINDOW_SIZE: int = envs.SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_WINDOW_SIZE.get()
    MIN_CONTENT_LENGTH: int = (
        envs.SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_MIN_CONTENT_LENGTH.get()
    )
    MIN_REPEAT: int = max(
        2, envs.SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_MIN_REPEAT.get()
    )
    DEBUG_SKIP_TRUNCATION: bool = (
        envs.SGLANG_MIMO_NGRAM_NO_REPEAT_PROCESSOR_DEBUG_SKIP_TRUNCATION.get()
    )
    CACHE_ATTR: str = "_mimov2flash_ngram_no_repeat_cache"
    DETECT_FLAG_ATTR: str = "_mimov2flash_no_repeat_detected"
    REQ_PARAM_KEY: str = "__req__"
    ACTION_SKIP: str = "skip"
    ACTION_FORCE_STOP: str = "force_stop"

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        if (
            self.NGRAM_SIZE <= 1
            or self.WINDOW_SIZE <= 0
            or self.WINDOW_SIZE < self.NGRAM_SIZE
        ):
            logger.warning(
                f"MiMoV2Flash no-repeat processor disabled due to invalid config: window_size={self.WINDOW_SIZE}, ngram_size={self.NGRAM_SIZE}"
            )
            return logits

        collector = _get_logit_processor_collector()
        # Per-request decision cache for this batch (spec decoding can repeat requests).
        per_req_action = {}

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get(self.REQ_PARAM_KEY)
            if req is None:
                continue

            if req.repetition_detected and not self.DEBUG_SKIP_TRUNCATION:
                assert req.to_finish is not None
                logits[batch_idx, :] = -float("inf")
                logits[batch_idx, self.EOS_TOKEN_ID] = 0.0
                continue

            action = per_req_action.get(req)
            if action is not None:
                # Same request appears again: reuse the decision but apply it to this row.
                logger.debug(
                    "MiMoV2Flash no-repeat reuse cached action: rid=%s action=%s output_ids_len=%s output_ids=%s",
                    getattr(req, "rid", None),
                    action,
                    len(req.output_ids),
                    req.output_ids,
                )
                if action == self.ACTION_SKIP:
                    continue

                assert action == self.ACTION_FORCE_STOP
                assert req.to_finish is not None
                logits[batch_idx, :] = -float("inf")
                logits[batch_idx, self.EOS_TOKEN_ID] = 0.0
                continue

            sequence: List[int] = req.output_ids
            seq_len = len(sequence)
            if seq_len < self.MIN_CONTENT_LENGTH:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if seq_len < self.NGRAM_SIZE:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if collector is not None and not getattr(req, self.DETECT_FLAG_ATTR, False):
                collector.no_repeat_processor_detected_count.inc()
                setattr(req, self.DETECT_FLAG_ATTR, True)

            # Cache tracks a sliding window of n-gram occurrences for this request.
            # It lets us update counts incrementally instead of rescanning the entire
            # sequence on every step.
            #
            # Diagram (NGRAM_SIZE=3):
            #   sequence: [t0, t1, t2, t3, t4, t5]
            #   n-grams :  (t0,t1,t2), (t1,t2,t3), (t2,t3,t4), (t3,t4,t5)
            #   queue   :  [(t0,t1,t2), (t1,t2,t3), (t2,t3,t4), (t3,t4,t5)]
            #   ngram_counts: {(t0,t1,t2):1, (t1,t2,t3):1, ...}
            cache = getattr(req, self.CACHE_ATTR, None)
            if cache is None:
                # Sliding window cache of n-gram occurrences for this request.
                cache = {
                    "last_len": 0,
                    "queue_start": 0,
                    "queue": deque(),
                    "ngram_counts": {},
                }
                setattr(req, self.CACHE_ATTR, cache)

            # last_len tracks the output length processed into the cache so far.
            # It helps us append only newly formed n-grams when the sequence grows.
            last_len = cache["last_len"]
            if seq_len < last_len:
                # Sequence can shrink on speculative decoding retraction; reset cache.
                cache["queue"].clear()
                cache["ngram_counts"].clear()
                cache["queue_start"] = 0
                last_len = 0

            if seq_len != last_len:
                start_i = last_len - self.NGRAM_SIZE + 1
                if start_i < 0:
                    start_i = 0
                end_i = seq_len - self.NGRAM_SIZE
                if end_i >= start_i:
                    queue = cache["queue"]
                    ngram_counts = cache["ngram_counts"]
                    if not queue:
                        cache["queue_start"] = start_i
                    for i in range(start_i, end_i + 1):
                        ngram = tuple(sequence[i : i + self.NGRAM_SIZE])
                        queue.append(ngram)
                        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                cache["last_len"] = seq_len

                # Trim the queue to keep only the last WINDOW_SIZE tokens worth of
                # n-grams. This maintains a moving window over the output sequence.
                #
                # Diagram (WINDOW_SIZE=5, NGRAM_SIZE=3):
                #   keep n-grams starting at indices [seq_len-5, ..., seq_len-3]
                #   drop older entries from queue and ngram_counts.
                min_start = seq_len - self.WINDOW_SIZE
                if min_start < 0:
                    min_start = 0
                queue = cache["queue"]
                ngram_counts = cache["ngram_counts"]
                # queue_start is the absolute start index (in sequence) for the
                # first element stored in the queue; it advances as we pop items.
                queue_start = cache["queue_start"]
                while queue and queue_start < min_start:
                    ngram = queue.popleft()
                    count = ngram_counts[ngram] - 1
                    if count:
                        ngram_counts[ngram] = count
                    else:
                        del ngram_counts[ngram]
                    queue_start += 1
                cache["queue_start"] = queue_start

            if not cache["ngram_counts"]:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if not any(
                count >= self.MIN_REPEAT for count in cache["ngram_counts"].values()
            ):
                per_req_action[req] = self.ACTION_SKIP
                continue

            # Found repetition in the request's output ids.
            if collector is not None and req.repetition_detected is False:
                collector.no_repeat_processor_truncated_count.inc()

            req.repetition_detected = True
            if self.DEBUG_SKIP_TRUNCATION:
                per_req_action[req] = self.ACTION_SKIP
                continue

            if req.to_finish is None:
                from sglang.srt.managers.schedule_batch import FINISH_REPEAT_TRUNCATION

                req.to_finish = FINISH_REPEAT_TRUNCATION()

            # Assign the highest probability to the EOS token
            logits[batch_idx, :] = -float("inf")
            logits[batch_idx, self.EOS_TOKEN_ID] = 0.0
            per_req_action[req] = self.ACTION_FORCE_STOP

        return logits
