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
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


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

    def __init__(
        self,
        params: CacheInitParams,
        # TODO(yingchun): consider move speculative_num_steps to CacheInitParams
        speculative_num_steps: Optional[int] = None,
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
        self.speculative_num_steps = speculative_num_steps

        # Cumulative counters for metrics.
        # Updated in evict_swa() and evict(); read by the scheduler to compute
        # per-interval deltas for Prometheus counters.
        self.swa_active_freed_tokens: int = 0
        self.full_evicted_from_tree: int = 0

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

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
        tree_prefix_len: int = 0,
    ):
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
        swa_free_end = max(0, prelen - self.sliding_window_size)
        if self.page_size > 1:
            # Keep SWA eviction page-aligned to avoid freeing a partial page.
            # PagedTokenToKVPoolAllocator.free() converts token indices to page
            # indices and frees *entire* pages.  If the free range slices through
            # the middle of a page, the whole page is returned to the free-list
            # while tokens on the remaining portion still hold non-zero SWA
            # mappings.  A later free of those tokens would then trigger a
            # double-free.  Rounding the end down and the start up guarantees
            # only fully-covered pages are freed.
            swa_free_end = (swa_free_end // self.page_size) * self.page_size
        # effective_start 取两者的最大值：
        # 1. req.swa_evicted_seqlen：避免重复 free 同一段请求本地 SWA；
        # 2. tree_prefix_len：保护已经共享给 radix tree 的前缀，不允许请求本地
        #    的缩窗逻辑把共享前缀误释放。
        #
        # 例子：请求命中了 6 个前缀 token，当前 prelen=12、window=4。
        # 理论上窗口外可回收到位置 8；但前 6 个 token 已是树共享前缀，因此只能
        # 从 max(swa_evicted_seqlen, 6) 开始回收，而不能动 [0, 6)。
        effective_start = max(req.swa_evicted_seqlen, tree_prefix_len)
        if self.page_size > 1:
            effective_start = (
                (effective_start + self.page_size - 1) // self.page_size
            ) * self.page_size
        if swa_free_end > effective_start:
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx,
                effective_start:swa_free_end,
            ]
            freed_swa_size = int(
                torch.count_nonzero(
                    self.token_to_kv_pool_allocator.full_to_swa_index_mapping[
                        free_slots
                    ]
                ).item()
            )
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            self.swa_active_freed_tokens += freed_swa_size
        if swa_free_end > req.swa_evicted_seqlen:
            req.swa_evicted_seqlen = swa_free_end

    def _count_live_swa_tokens(self, value: torch.Tensor | None) -> int:
        if value is None or value.numel() == 0:
            return 0

        mapping = self.token_to_kv_pool_allocator.full_to_swa_index_mapping
        return int(torch.count_nonzero(mapping[value]).item())

    # TODO(yingchun): 每次请求 admission 触发多次全树遍历+GPU sync，后续会优化
    def _collect_swa_sizes(self) -> tuple[int, int]:
        """Return live SWA sizes split by evictable/protected ownership."""
        swa_evictable = 0
        swa_protected = 0
        stack = [self.root_node]

        while stack:
            node = stack.pop()
            for child in node.children.values():
                live_swa = self._count_live_swa_tokens(child.value)
                if child.lock_ref > 0:
                    swa_protected += live_swa
                else:
                    swa_evictable += live_swa
                stack.append(child)

        return swa_evictable, swa_protected

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def swa_evictable_size(self) -> int:
        swa_evictable, _ = self._collect_swa_sizes()
        return swa_evictable

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_protected_size(self) -> int:
        _, swa_protected = self._collect_swa_sizes()
        return swa_protected

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

            self.token_to_kv_pool_allocator.free(value)
            full_num_evicted += len(value)
            swa_num_evicted += reclaimed_swa
            self._delete_leaf(x)

            # TODO(yingchun): 需要像 RadixCache 一样增加 x.parent.lock_ref == 0 条件吗？
            if len(x.parent.children) == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(full_num_evicted + swa_num_evicted, start_time)
        self.full_evicted_from_tree += full_num_evicted
        return EvictResult(
            num_tokens_evicted=full_num_evicted,
            swa_num_tokens_evicted=swa_num_evicted,
        )

    def _compute_safe_prefix_len(self, swa_indices: torch.Tensor) -> int:
        """Compute the maximum prefix length that is SWA-safe.

        For a candidate prefix length ``L``, the tail window
        ``[max(0, L-window), L)`` must all have valid SWA indices (> 0).
        """
        matched_len = len(swa_indices)
        if matched_len == 0:
            return 0

        window = self.sliding_window_size
        tail_start = max(0, matched_len - window)
        if torch.all(swa_indices[tail_start:] > 0):
            return matched_len

        bad = (swa_indices <= 0).to(torch.int32)
        prefix_bad = torch.nn.functional.pad(torch.cumsum(bad, dim=0), (1, 0))

        lens = torch.arange(1, matched_len + 1, device=swa_indices.device)
        starts = torch.clamp(lens - window, min=0)
        bad_count = prefix_bad[lens] - prefix_bad[starts]

        valid = bad_count == 0
        if not torch.any(valid):
            return 0
        return int(torch.nonzero(valid, as_tuple=False)[-1].item() + 1)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest prefix whose SWA tail window is valid.

        `RadixCache.match_prefix()` only matches token path in the tree and is
        unaware of SWA indices already freed to 0. For hybrid SWA, we must make
        sure the matched tail window still maps to live SWA slots.
        """
        key = params.key
        # Use a copy to avoid mutating caller-provided key object.
        raw_params = MatchPrefixParams(
            key=RadixKey(list(key.token_ids), key.extra_key),
            req=params.req,
        )
        base_result = super().match_prefix(raw_params)

        matched_indices = base_result.device_indices
        matched_len = len(matched_indices)
        if matched_len == 0:
            return base_result

        # 仅有 token path 命中还不够。对于 hybrid SWA，前缀想真正可复用，还必须
        # 满足“这个前缀尾部窗口里的 SWA mapping 仍然活着”。
        # 否则 decode/prefill 虽然能在树上找到 full index，但一旦要访问 SWA state
        # 就会发现对应 mapping 已经被置 0。
        swa_indices = self.token_to_kv_pool_allocator.full_to_swa_index_mapping[
            matched_indices
        ]
        safe_len = self._compute_safe_prefix_len(swa_indices)
        if safe_len >= matched_len:
            return base_result

        if self.is_eagle:
            safe_token_len = (
                safe_len + self.speculative_num_steps if safe_len > 0 else 0
            )
        else:
            safe_token_len = safe_len
        safe_params = MatchPrefixParams(
            key=RadixKey(list(key.token_ids)[:safe_token_len], key.extra_key),
            req=params.req,
        )
        # TODO(yingchun): 直接对首次结果做切片复用，避免 double walk
        return super().match_prefix(safe_params)

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
            self.evict_swa(
                req,
                seq_len,
                self.sliding_window_size,
                tree_prefix_len=req.cache_protected_len,
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
        self.evict_swa(
            req,
            actual_kv_len,
            self.sliding_window_size,
            tree_prefix_len=req.cache_protected_len,
        )

        # Save kv_indices as prefix_indices for the next chunk / scheduling
        # round.  No tree insertion, lock-ref update, or req_to_token rewrite
        # is needed — the request keeps its own private KV slots until finish.
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].to(
            dtype=torch.int64, copy=True
        )
        req.prefix_indices = kv_indices

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
        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
        else:
            page_aligned_len = actual_kv_len

        tree_prefix_len = req.cache_protected_len

        self.evict_swa(
            req,
            page_aligned_len,
            self.sliding_window_size,
            tree_prefix_len=tree_prefix_len,
        )

        super().cache_finished_req(req, is_insert)
