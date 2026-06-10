from __future__ import annotations

"""
SimpleSWARadixCache: RadixCache with SWA (Sliding Window Attention) memory optimization.

Designed for **prefill nodes only** in a PD (Prefill–Decode) disaggregation setup.
Decode nodes use a different cache type; no decode code-path is needed here.

Actively frees SWA KV cache for tokens outside the sliding window during chunked
prefill and request completion. This reduces SWA pool memory usage from O(seq_len)
to O(sliding_window_size) per request.

Design:
    - Inherits from RadixCache, reusing tree management, insert, match_prefix, evict, etc.
    - Overrides cache_unfinished_req and cache_finished_req to free out-of-window SWA KV.
    - Performs SWA shrinking only at chunk / request lifecycle boundaries, after
      a request-local req_to_token row already exists.
    - Does NOT use tombstone mechanism or dual LRU lists (unlike the existing SWARadixCache).
    - Tree insertion happens ONLY at request completion (cache_finished_req), NOT during
      intermediate scheduling rounds (cache_unfinished_req). This simplifies ownership
      tracking and avoids double-free / memory-leak issues from mid-flight tree sharing.

Safety:
    - free_swa() in the allocator filters out already-freed entries (mapping == 0),
      so double-free is safe.
    - translate_loc_from_full_to_swa() returns 0 for freed entries (padding slot),
      which is safe because flash_attn's window_size parameter prevents accessing
      out-of-window KV entries.
"""

import heapq
import logging
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.nvtx_utils import nvtx_annotated_method
from sglang.srt.utils.common import ceil_align, floor_align, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def compute_swa_safe_len(swa_indices: torch.Tensor, sliding_window_size: int) -> int:
    """
    Find the longest prefix whose SWA tail window has all non-zero indices.

    For a candidate prefix length ``L`` (i.e. reuse tokens ``[0, L)``),
    the sliding window tail ``[max(0, L - sliding_window_size), L)`` must all
    be non-zero. A zero slot means the SWA state is freed/invalid.

    Args:
        swa_indices: 1-D tensor of SWA index mappings (>= 0).
            Non-zero = valid; zero = freed/invalid.
        sliding_window_size: The SWA window size.

    Returns:
        The maximum SWA-safe prefix length (0 means nothing is reusable,
        ``len(swa_indices)`` means the entire prefix is safe).
    """
    matched_len = len(swa_indices)
    if matched_len == 0:
        return 0

    # TODO(yingchun): this assert has D2H op
    assert (swa_indices >= 0).all(), "swa_indices must be non-negative"

    # --- Fast path: if the tail window is all valid, the entire prefix is safe.
    tail_start = max(0, matched_len - sliding_window_size)
    if swa_indices[tail_start:].all():
        return matched_len

    # --- Slow path: some slots are freed; find the longest safe prefix. ---
    bad = (swa_indices == 0).to(torch.int32)
    prefix_bad = torch.nn.functional.pad(torch.cumsum(bad, dim=0), (1, 0))

    lens = torch.arange(1, matched_len + 1, device=swa_indices.device)
    starts = torch.clamp(lens - sliding_window_size, min=0)
    bad_count = prefix_bad[lens] - prefix_bad[starts]
    valid = bad_count == 0

    if not torch.any(valid):
        return 0

    return int(torch.nonzero(valid, as_tuple=False)[-1].item() + 1)


class SimpleSWARadixCache(RadixCache):
    """RadixCache with active SWA KV cache memory optimization (prefill-only).

    Used exclusively on prefill nodes in PD disaggregation. Decode nodes use
    a separate cache type; no decode code-path is needed here.

    During chunked prefill and request completion, actively frees SWA KV cache
    for tokens outside the sliding window. The full attention KV cache is
    unaffected and managed normally by RadixCache.

    Args:
        sliding_window_size: The sliding window size for SWA layers. Only the
            last ``sliding_window_size`` tokens' SWA KV are kept alive.
    """

    @classmethod
    def _parse_reserved_points(cls, raw_points: str) -> list[int]:
        raw_points = raw_points.strip()
        if not raw_points:
            return []

        try:
            points = [int(point.strip()) for point in raw_points.split(",")]
        except ValueError as exc:
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_POINTS must be a comma-separated integer list"
            ) from exc

        if any(point <= 0 for point in points):
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_POINTS must contain positive integers"
            )
        if len(points) != len(set(points)):
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_POINTS must not contain duplicates"
            )
        if points != sorted(points):
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_POINTS must be sorted in ascending order"
            )

        return points

    @classmethod
    def _load_swa_reserved_config(
        cls, sliding_window_size: int, page_size: int
    ) -> tuple[list[int], int, int]:
        reserved_points = cls._parse_reserved_points(
            envs.SGLANG_SIMPLE_SWA_RESERVED_POINTS.get()
        )
        reserved_chunk = envs.SGLANG_SIMPLE_SWA_RESERVED_CHUNK.get()
        reserved_max_point = envs.SGLANG_SIMPLE_SWA_RESERVED_MAX_POINT.get()

        if reserved_chunk <= 0:
            raise ValueError("SGLANG_SIMPLE_SWA_RESERVED_CHUNK must be positive")
        if reserved_chunk % page_size != 0:
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_CHUNK must be a multiple of page_size"
            )
        if reserved_max_point < 0:
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_MAX_POINT must be non-negative"
            )
        if reserved_max_point % page_size != 0:
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_MAX_POINT must be a multiple of page_size"
            )
        if any(point % sliding_window_size != 0 for point in reserved_points):
            raise ValueError(
                "SGLANG_SIMPLE_SWA_RESERVED_POINTS must be multiples of "
                "sliding_window_size"
            )

        return reserved_points, reserved_chunk, reserved_max_point

    def __init__(
        self,
        params: CacheInitParams,
    ):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator), (
            f"SimpleSWARadixCache requires SWATokenToKVPoolAllocator, "
            f"got {type(params.token_to_kv_pool_allocator)}"
        )
        assert (
            params.sliding_window_size is not None and params.sliding_window_size > 0
        ), f"sliding_window_size must be positive, got {params.sliding_window_size}"

        super().__init__(params)
        self.sliding_window_size = params.sliding_window_size
        (
            self.swa_reserved_points,
            self.swa_reserved_chunk,
            self.swa_reserved_max_point,
        ) = self._load_swa_reserved_config(self.sliding_window_size, params.page_size)

        # Cumulative counters for metrics.
        # Updated in evict_swa() and evict(); read by the scheduler to compute
        # per-interval deltas for Prometheus counters.
        self.swa_active_freed_tokens: int = 0
        self.swa_evicted_from_tree: int = 0
        self.full_evicted_from_tree: int = 0
        assert self.sliding_window_size % self.page_size == 0
        logger.info(
            f"SimpleSWARadixCache initialized with sliding_window_size={self.sliding_window_size}"
        )

        # Cached length of the most recent raw prefix-tree match (before SWA
        # safety trim) used by callers for diagnostics. Updated in match_prefix().
        self._last_raw_match_len: int = 0

        # Incrementally-maintained live SWA token counts, partitioned by
        # ownership. Updated on insert / evict / lock_ref changes; evict_swa
        # is a no-op here because it only frees request-private SWA slots
        # (tree-owned slots are protected by the tree_prefix_len lower bound).
        self._swa_evictable_size: int = 0
        self._swa_protected_size: int = 0

        # Invariant check sampling counter. Triggered every
        # SGLANG_DEBUG_SWA_INVARIANT_INTERVAL queries when
        # SGLANG_DEBUG_SWA_INVARIANT is enabled. Logs (does NOT raise) on drift.
        self._swa_invariant_query_counter: int = 0

    ##### Public API #####

    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SimpleSWARadixCache"
        return True

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = self.full_evictable_size()
        swa_evictable_size = self.swa_evictable_size()
        # 对 MR review/线上报错定位很重要：
        # full 看起来能回收很多，并不代表 swa 也能回收很多。
        # 尤其在 SimpleSWARadixCache 下，窗口外 SWA 早就被清空了，真正决定
        # 能否继续调度的往往是“剩余 live SWA + 可再驱逐的 live SWA”。
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
        )

    def _filter_swa_free_slots_before_free(
        self,
        free_slots: torch.Tensor,  # read-only here
        effective_start: int,
        swa_free_end: int,
        tree_prefix_len: int,
    ) -> List[torch.Tensor]:
        """
        Filter out SWA pages to free that are explicit reserved
        """

        # 如果 swa_free_end>>effective_start 且 effective_start>>tree_prefix_len，则保留effective_start附近的swa
        if (
            swa_free_end - effective_start >= self.swa_reserved_chunk
            and effective_start - tree_prefix_len >= self.swa_reserved_chunk
            and effective_start + self.sliding_window_size < self.swa_reserved_max_point
        ):
            effective_start += self.sliding_window_size
            free_slots = free_slots[self.sliding_window_size :]

        if not self.swa_reserved_points:
            return [free_slots]

        # 如果和 reserved points 没有交集
        if (
            effective_start >= self.swa_reserved_points[-1]
            or swa_free_end <= self.swa_reserved_points[0] - self.sliding_window_size
        ):
            return [free_slots]

        result = []
        last_wend = effective_start
        # 如果一个point的起点和free_slots有交集，则以它为分割点，添加last_wend到point起点之前的片段
        # 特殊情况：如果swa_free_end在第一个point的窗口内，则不满足任何条件，全部可free
        for wend in self.swa_reserved_points:
            wbegin = wend - self.sliding_window_size
            if wbegin <= swa_free_end and last_wend < wbegin:
                result.append(
                    free_slots[last_wend - effective_start : wbegin - effective_start]
                )
                last_wend = wend
        # 添加最后剩下的片段
        if last_wend < swa_free_end:
            result.append(free_slots[last_wend - effective_start :])

        return result

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
        tree_prefix_len: int = 0,
    ) -> int:
        """Free SWA KV cache for tokens outside the sliding window.

        SimpleSWARadixCache performs real SWA shrinking only after a prefill
        forward has completed, when the request still owns a request-local
        req_to_token row. Unlike SWAChunkCache, alloc_for_extend() does not call
        into this method before allocation.

        Args:
            req: The request whose SWA KV to partially free.
            prelen: Current number of tokens that have KV computed.
            attention_chunk_size: Unused, kept for interface compatibility with
                SWAChunkCache.evict_swa().
            tree_prefix_len: Number of leading positions in req_to_token that
                are shared with the radix tree and must not have their SWA
                mapping freed here.
        """
        assert req.req_pool_idx is not None, (
            "SimpleSWARadixCache.evict_swa() requires a request-local "
            "req_to_token row and should only run after a prefill chunk has "
            "completed."
        )
        # Keep SWA eviction page-aligned to avoid freeing a partial page.
        # PagedTokenToKVPoolAllocator.free() converts token indices to page
        # indices and frees *entire* pages.  If the free range slices through
        # the middle of a page, the whole page is returned to the free-list
        # while tokens on the remaining portion still hold non-zero SWA
        # mappings.  A later free of those tokens would then trigger a
        # double-free.  Rounding the end down and the start up guarantees
        # only fully-covered pages are freed.
        swa_free_end = floor_align(
            max(0, prelen - self.sliding_window_size), self.page_size
        )
        # effective_start 取两者的最大值：
        # 1. req.swa_evicted_seqlen：避免重复 free 同一段请求本地 SWA；
        # 2. tree_prefix_len：保护已经共享给 radix tree 的前缀，不允许请求本地
        #    的缩窗逻辑把共享前缀误释放。
        #
        # 例子：请求命中了 6 个前缀 token，当前 prelen=12、window=4。
        # 理论上窗口外可回收到位置 8；但前 6 个 token 已是树共享前缀，因此只能
        # 从 max(swa_evicted_seqlen, 6) 开始回收，而不能动 [0, 6)。
        effective_start = ceil_align(
            max(req.swa_evicted_seqlen, tree_prefix_len), self.page_size
        )
        freed_swa_size = 0
        if swa_free_end > effective_start:
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                effective_start:swa_free_end,
            ]
            # freed_swa_size = self.token_to_kv_pool_allocator.free_swa(free_slots)
            freed_swa_size = 0
            filtered_free_slots = self._filter_swa_free_slots_before_free(
                free_slots, effective_start, swa_free_end, tree_prefix_len
            )
            # TODO(yingchun): 先 torch.cat 合并再统一释放
            for one_free_slots in filtered_free_slots:
                freed_swa_size += self.token_to_kv_pool_allocator.free_swa(
                    one_free_slots
                )
            self.swa_active_freed_tokens += freed_swa_size
        if swa_free_end > req.swa_evicted_seqlen:
            req.swa_evicted_seqlen = swa_free_end
        return freed_swa_size

    # TODO(yingchun): 在节点上缓存 SWA 数量（注意在 splice/free 时更新），避免频繁的 GPU sync
    def _count_live_swa_tokens(self, value: torch.Tensor) -> int:
        assert (
            value is not None and value.numel() > 0
        ), "value must be a non-empty tensor"
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        return int(torch.count_nonzero(mapping[value]).item())

    def _batch_count_live_swa(self, values: list[torch.Tensor]) -> int:
        """Count live SWA tokens across multiple value tensors in one GPU op."""
        if not values:
            return 0
        all_indices = torch.cat(values)
        if all_indices.numel() == 0:
            return 0
        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        return int(torch.count_nonzero(mapping[all_indices]).item())

    def _collect_swa_sizes_recomputed(self) -> tuple[int, int]:
        """Full tree traversal to recompute live SWA sizes. Used as a
        reference by tests / debug invariant checks; production queries use
        the incrementally-maintained counters."""
        evictable_values: list[torch.Tensor] = []
        protected_values: list[torch.Tensor] = []
        stack = [self.root_node]

        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.value is not None:
                    if child.lock_ref > 0:
                        protected_values.append(child.value)
                    else:
                        evictable_values.append(child.value)
                stack.append(child)

        return (
            self._batch_count_live_swa(evictable_values),
            self._batch_count_live_swa(protected_values),
        )

    def _check_swa_invariant_if_due(self) -> None:
        if not envs.SGLANG_DEBUG_SWA_INVARIANT.get():
            return
        self._swa_invariant_query_counter += 1
        interval = envs.SGLANG_DEBUG_SWA_INVARIANT_INTERVAL.get()
        if interval <= 0 or self._swa_invariant_query_counter % interval != 0:
            return
        try:
            real_evictable, real_protected = self._collect_swa_sizes_recomputed()
        except Exception as e:
            logger.warning(f"SWA invariant: tree traversal failed: {e}")
            return
        evictable_drift = self._swa_evictable_size - real_evictable
        protected_drift = self._swa_protected_size - real_protected
        if evictable_drift != 0 or protected_drift != 0:
            logger.warning(
                "SWA counter drift detected: "
                f"_swa_evictable_size={self._swa_evictable_size} (real={real_evictable}, drift={evictable_drift}); "
                f"_swa_protected_size={self._swa_protected_size} (real={real_protected}, drift={protected_drift}); "
                f"sampled_at_query={self._swa_invariant_query_counter}"
            )

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def swa_evictable_size(self) -> int:
        self._check_swa_invariant_if_due()
        return self._swa_evictable_size

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_protected_size(self) -> int:
        return self._swa_protected_size

    def _on_node_becomes_protected(self, node: TreeNode) -> None:
        swa_live = self._count_live_swa_tokens(node.value)
        self._swa_evictable_size -= swa_live
        self._swa_protected_size += swa_live

    def _on_node_becomes_evictable(self, node: TreeNode) -> None:
        swa_live = self._count_live_swa_tokens(node.value)
        self._swa_evictable_size += swa_live
        self._swa_protected_size -= swa_live

    def _on_tree_value_added(self, node: TreeNode) -> None:
        self._swa_evictable_size += self._count_live_swa_tokens(node.value)

    def _on_tree_value_removed(self, node: TreeNode) -> None:
        # Called before node.value is released; if the node is protected
        # (lock_ref > 0) the SWA tokens were in _swa_protected_size, otherwise
        # in _swa_evictable_size. Deduct from the right bucket.
        swa_live = self._count_live_swa_tokens(node.value)
        if swa_live == 0:
            return
        if node.lock_ref > 0:
            self._swa_protected_size -= swa_live
        else:
            self._swa_evictable_size -= swa_live

    def _on_node_swa_mapping_increased(
        self, node: TreeNode, num_transferred: int
    ) -> None:
        if num_transferred <= 0:
            return
        if node.lock_ref > 0:
            self._swa_protected_size += num_transferred
        else:
            self._swa_evictable_size += num_transferred

    def reset(self):
        super().reset()
        self._swa_evictable_size = 0
        self._swa_protected_size = 0

    def evict(self, params: EvictParams) -> EvictResult:
        """Evict cache entries to satisfy both full and SWA shortages.

        For SimpleSWARadixCache, a leaf may have fewer reclaimable SWA slots than
        full slots because some SWA mappings were proactively freed earlier.
        Therefore we track full/SWA reclaimed sizes separately.
        """
        if self.disable:
            return EvictResult()
        full_num_tokens = params.num_tokens
        swa_num_tokens = params.swa_num_tokens
        if full_num_tokens <= 0 and swa_num_tokens <= 0:
            return EvictResult()

        start_time = time.perf_counter()
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        full_num_evicted = 0
        swa_num_evicted = 0

        while (
            full_num_evicted < full_num_tokens or swa_num_evicted < swa_num_tokens
        ) and len(eviction_heap) > 0:
            priority, x = heapq.heappop(eviction_heap)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            value = x.value
            reclaimed_swa = self._count_live_swa_tokens(value)

            self._on_tree_value_removed(x)
            self.token_to_kv_pool_allocator.free(value)
            full_num_evicted += len(value)
            swa_num_evicted += reclaimed_swa
            self._delete_leaf(x)

            # TODO(yingchun): 需要像 RadixCache 一样增加 x.parent.lock_ref == 0 条件吗？
            if len(x.parent.children) == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)
        self.swa_evicted_from_tree += swa_num_evicted
        self.full_evicted_from_tree += full_num_evicted

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)
        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            swa_num_tokens_evicted=swa_num_evicted,
        )

    def _compute_safe_prefix_len(
        self, full_matched_result: MatchResult
    ) -> tuple[int, Optional[int]]:
        """Compute the maximum device prefix length that is SWA-safe.

        Looks up SWA indices from the device allocator's mapping and delegates
        to the shared ``compute_swa_safe_len()`` for the actual check.

        Returns:
            (safe_len, None): safe_len is the longest SWA-safe device prefix
            length. The second element is always None (no host portion).
        """
        swa_indices = self.token_to_kv_pool_allocator.full_to_swa_index_mapping[
            full_matched_result.device_indices
        ]
        assert len(swa_indices) == len(full_matched_result.device_indices)
        safe_len = compute_swa_safe_len(swa_indices, self.sliding_window_size)
        return safe_len, None

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """
        Find the longest prefix whose SWA tail window is valid.

        `RadixCache.match_prefix()` only matches token path in the tree and is
        unaware of SWA indices already freed to 0. For hybrid SWA, we must make
        sure the matched tail window still maps to live SWA slots.
        """
        # TODO(yingchun): 需要保证/能保证 page size > 1 的情况吗？
        key = params.key
        # Use a copy to avoid mutating caller-provided key object.
        raw_params = MatchPrefixParams(
            key=RadixKey(list(key.token_ids), key.extra_key),
            req=params.req,
        )
        full_matched_result = super().match_prefix(raw_params)

        full_matched_indices = full_matched_result.device_indices
        full_matched_len = len(full_matched_indices)
        self._last_raw_match_len = full_matched_len
        if full_matched_len == 0:
            return full_matched_result

        # 仅有 token path 命中还不够。对于 hybrid SWA，前缀想真正可复用，还必须
        # 满足"这个前缀尾部窗口里的 SWA mapping 仍然活着"。
        # 否则 prefill 虽然能在树上找到 full index，但一旦要访问 SWA state
        # 就会发现对应 mapping 已经被置 0。
        swa_matched_len, _ = self._compute_safe_prefix_len(full_matched_result)
        if swa_matched_len >= full_matched_len:
            return full_matched_result

        if self.is_eagle:
            safe_token_len = (
                swa_matched_len + self.speculative_num_steps
                if swa_matched_len > 0
                else 0
            )
        else:
            safe_token_len = swa_matched_len
        safe_params = MatchPrefixParams(
            key=RadixKey(list(key.token_ids)[:safe_token_len], key.extra_key),
            req=params.req,
        )
        return super().match_prefix(safe_params)

    @nvtx_annotated_method("swa_radix.cache_unfinished_req")
    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished, freeing out-of-window SWA KV.

        Only frees SWA KV outside the sliding window and saves kv_indices as
        prefix_indices for the next scheduling round. Does NOT insert KV into
        the radix tree — tree insertion happens only at request completion
        (cache_finished_req). This keeps ownership tracking simple and avoids
        double-free / memory-leak issues from mid-flight tree sharing.
        """
        if self.disable:
            return

        seq_len = len(req.fill_ids)

        if chunked:
            # TODO(yingchun): 以下重复代码可简化
            # Free SWA KV outside the sliding window.  tree_prefix_len protects
            # positions shared with the radix tree (from the initial match) so
            # they are not freed here.
            freed_swa_size = self.evict_swa(
                req,
                seq_len,
                self.sliding_window_size,
                tree_prefix_len=req.cache_protected_len,
            )
            if self.debug_mode and freed_swa_size > 0:
                log_info_on_rank0(
                    logger, f"cache_unfinished_req chunked: free SWA: {freed_swa_size}"
                )

            # Save kv_indices as prefix_indices for the next chunk / scheduling
            # round.  No tree insertion, lock-ref update, or req_to_token rewrite
            # is needed — the request keeps its own private KV slots until finish.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ].to(dtype=torch.int64, copy=True)
            req.prefix_indices = kv_indices
            return

        # Compute actual KV length (accounts for speculative/EAGLE tail tokens
        # that do not have cacheable KV entries).
        if self.is_eagle:
            actual_kv_len = max(0, seq_len - self.speculative_num_steps)
        else:
            actual_kv_len = seq_len

        # Free SWA KV outside the sliding window.  tree_prefix_len protects
        # positions shared with the radix tree (from the initial match) so
        # they are not freed here.
        freed_swa_size = self.evict_swa(
            req,
            actual_kv_len,
            self.sliding_window_size,
            tree_prefix_len=req.cache_protected_len,
        )
        if self.debug_mode and freed_swa_size > 0:
            log_info_on_rank0(
                logger, f"cache_unfinished_req unchunked: free SWA: {freed_swa_size}"
            )

        # Save kv_indices as prefix_indices for the next chunk / scheduling
        # round.  No tree insertion, lock-ref update, or req_to_token rewrite
        # is needed — the request keeps its own private KV slots until finish.
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].to(
            dtype=torch.int64, copy=True
        )
        req.prefix_indices = kv_indices

    @nvtx_annotated_method("swa_radix.cache_finished_req")
    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes, keeping only swa_size SWA KV.

        1. Free SWA KV outside the sliding window.
        2. Delegate to RadixCache.cache_finished_req() for tree insertion
           and cleanup.

        The parent's free() calls go through free_swa() which filters out
        already-freed entries (mapping == 0), so double-free is safe.
        """
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        actual_kv_len = (
            max(0, all_token_len - self.speculative_num_steps)
            if self.is_eagle
            else all_token_len
        )
        page_aligned_len = floor_align(actual_kv_len, self.page_size)

        tree_prefix_len = req.cache_protected_len

        freed_swa_size = self.evict_swa(
            req,
            page_aligned_len,
            self.sliding_window_size,
            tree_prefix_len=tree_prefix_len,
        )
        if self.debug_mode and freed_swa_size > 0:
            log_info_on_rank0(logger, f"cache_finished_req: free SWA: {freed_swa_size}")

        super().cache_finished_req(req, is_insert)
