"""Unit tests for _compute_safe_prefix_len and match_prefix in
SimpleSWARadixCache and HiSimpleSWARadixCache.

Tests the SWA-aware prefix matching logic:
- SimpleSWARadixCache._compute_safe_prefix_len (device-only)
- SimpleSWARadixCache.match_prefix (end-to-end with tree)
- HiSimpleSWARadixCache._compute_safe_prefix_len (device + host)
- HiSimpleSWARadixCache.match_prefix (end-to-end with tree + host eviction)

Coverage targets:
    - _compute_safe_prefix_len:
        - All SWA valid (fast path, returns None for host)
        - Partial SWA invalid in device portion
        - Partial SWA invalid in host portion (HiCache only)
        - All SWA invalid
        - Empty match
        - Device-only match (no host)
        - Host-only match (device empty, but this requires device_len=0 which
          means total=host_hit_length, unusual scenario)
    - match_prefix:
        - Full match with all SWA valid → no trimming
        - Match trimmed due to SWA invalidation
        - Match with host hit (HiCache) trimmed
        - Re-match after device trimming resolves correct last_device_node
        - Empty tree → empty match
        - Partial key overlap
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.simple_swa_radix_cache import SimpleSWARadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-small")


def _skip_if_no_cuda():
    """Helper to skip tests when CUDA is not available."""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


# ====================================================================== #
#  Shared test infrastructure
# ====================================================================== #


def _build_simple_swa_cache(
    sliding_window_size: int = 4,
    kv_size: int = 256,
    kv_size_swa: int = 128,
    max_context_len: int = 64,
    req_size: int = 4,
    page_size: int = 1,
    num_layers: int = 8,
    global_interval: int = 4,
    head_num: int = 4,
    head_dim: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[SimpleSWARadixCache, ReqToTokenPool, SWATokenToKVPoolAllocator]:
    """Build a SimpleSWARadixCache with its dependencies for testing."""
    full_layer_ids = list(range(0, num_layers, global_interval))
    swa_layer_ids = [i for i in range(num_layers) if i not in set(full_layer_ids)]

    req_to_token_pool = ReqToTokenPool(
        size=req_size,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = SimpleSWARadixCache(
        params=CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=sliding_window_size,
        ),
    )
    return cache, req_to_token_pool, allocator


# ====================================================================== #
#  SimpleSWARadixCache._compute_safe_prefix_len tests
# ====================================================================== #


class TestSimpleSWAComputeSafePrefixLen(CustomTestCase):
    """Test SimpleSWARadixCache._compute_safe_prefix_len via MatchResult."""

    def setUp(self):
        _skip_if_no_cuda()
        self.device = "cuda"
        self.sliding_window_size = 4
        self.cache, self.pool, self.allocator = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size,
            device=self.device,
        )

    def _make_match_result(self, device_indices: torch.Tensor) -> MatchResult:
        """Helper to create a MatchResult with the given device_indices."""
        return MatchResult(
            device_indices=device_indices,
            last_device_node=self.cache.root_node,
            last_host_node=self.cache.root_node,
            host_hit_length=0,
        )

    def test_all_swa_valid(self):
        """All SWA mappings valid → (full_len, None)."""
        kv = self.allocator.alloc(8)
        self.assertIsNotNone(kv)
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 8)
        self.assertIsNone(host_len)

    def test_tail_swa_freed(self):
        """Last token's SWA freed → safe_len = len - 1."""
        kv = self.allocator.alloc(8)
        self.assertIsNotNone(kv)
        # Free SWA of the last token
        self.allocator.free_swa(kv[-1:])
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        # The last token is in the window, so safe_len should be 7
        self.assertEqual(safe_len, 7)
        self.assertIsNone(host_len)

    def test_head_swa_freed_outside_window(self):
        """Head SWA freed but outside window → still fully safe."""
        kv = self.allocator.alloc(10)
        self.assertIsNotNone(kv)
        # Free SWA of the first token (outside window for L>=5)
        self.allocator.free_swa(kv[:1])
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        # window=4, len=10. For L=10, window=[6,10), first token outside.
        self.assertEqual(safe_len, 10)
        self.assertIsNone(host_len)

    def test_all_swa_freed(self):
        """All SWA freed → safe_len = 0."""
        kv = self.allocator.alloc(6)
        self.assertIsNotNone(kv)
        self.allocator.free_swa(kv)
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 0)
        self.assertIsNone(host_len)

    def test_middle_swa_freed_within_window(self):
        """Middle SWA freed, within window for some L values.

        kv = [v, v, 0, v, v, v]  (len=6, window=4)
        L=6: w=[2,6) → [0,v,v,v] → bad
        L=5: w=[1,5) → [v,0,v,v] → bad
        L=4: w=[0,4) → [v,v,0,v] → bad
        L=3: w=[0,3) → [v,v,0] → bad
        L=2: w=[0,2) → [v,v] → ok
        safe = 2.
        """
        kv = self.allocator.alloc(6)
        self.assertIsNotNone(kv)
        self.allocator.free_swa(kv[2:3])
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 2)
        self.assertIsNone(host_len)

    def test_gap_then_enough_valid_tail(self):
        """Gap in middle, but enough valid tail to fill the window.

        kv = [0, v, v, v, v, v]  (len=6, window=4)
        L=6: w=[2,6) → [v,v,v,v] → ok!
        safe = 6.
        """
        kv = self.allocator.alloc(6)
        self.assertIsNotNone(kv)
        self.allocator.free_swa(kv[:1])
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 6)
        self.assertIsNone(host_len)

    def test_single_token_valid(self):
        """Single token, SWA valid → safe_len = 1."""
        kv = self.allocator.alloc(1)
        self.assertIsNotNone(kv)
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 1)
        self.assertIsNone(host_len)

    def test_single_token_invalid(self):
        """Single token, SWA freed → safe_len = 0."""
        kv = self.allocator.alloc(1)
        self.assertIsNotNone(kv)
        self.allocator.free_swa(kv)
        result = self._make_match_result(kv)
        safe_len, host_len = self.cache._compute_safe_prefix_len(result)
        self.assertEqual(safe_len, 0)
        self.assertIsNone(host_len)


# ====================================================================== #
#  SimpleSWARadixCache.match_prefix tests
# ====================================================================== #


class TestSimpleSWAMatchPrefix(CustomTestCase):
    """End-to-end match_prefix tests for SimpleSWARadixCache."""

    def setUp(self):
        _skip_if_no_cuda()
        self.device = "cuda"
        self.sliding_window_size = 4

    def test_empty_tree_returns_empty(self):
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(10)))))
        self.assertEqual(len(result.device_indices), 0)

    def test_full_match_all_swa_valid(self):
        """Insert and match — all SWA valid → full match returned."""
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        tokens = list(range(10))
        kv = alloc.alloc(len(tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(tokens), value=kv))

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(result.device_indices), len(tokens))

    def test_match_trimmed_when_tail_swa_freed(self):
        """After freeing SWA of last few tokens, match should be trimmed."""
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        tokens = list(range(10))
        kv = alloc.alloc(len(tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(tokens), value=kv))

        # Free SWA of last 2 tokens
        alloc.free_swa(kv[-2:])

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        # The last 2 tokens have invalid SWA. For the trimmed prefix,
        # the window must be fully valid. Since window=4 and we freed
        # positions 8,9, L=10 bad, L=9 bad, L=8: w=[4,8) all valid → ok.
        self.assertEqual(len(result.device_indices), 8)

    def test_match_trimmed_to_zero_when_all_swa_freed(self):
        """All SWA freed → match returns empty."""
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        tokens = list(range(6))
        kv = alloc.alloc(len(tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(tokens), value=kv))

        alloc.free_swa(kv)

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(result.device_indices), 0)

    def test_match_unaffected_when_head_swa_freed_outside_window(self):
        """SWA freed only at head (outside window) → full match."""
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        tokens = list(range(10))
        kv = alloc.alloc(len(tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(tokens), value=kv))

        # Free SWA of first 3 tokens — they are outside the window for L=10
        alloc.free_swa(kv[:3])

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(result.device_indices), 10)

    def test_partial_key_overlap(self):
        """Query has different suffix → match only the common prefix."""
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        cached_tokens = list(range(10))
        kv = alloc.alloc(len(cached_tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(cached_tokens), value=kv))

        query = list(range(6)) + [100, 101]
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(query)))
        self.assertEqual(len(result.device_indices), 6)

    def test_match_after_evict_swa_via_req(self):
        """After evict_swa on a request, tree match should reflect SWA state.

        Insert tokens into tree, create a request with those tokens,
        evict SWA via the request, then verify match_prefix returns
        trimmed result.
        """
        cache, pool, alloc = _build_simple_swa_cache(
            sliding_window_size=self.sliding_window_size
        )
        tokens = list(range(12))
        kv = alloc.alloc(len(tokens))
        self.assertIsNotNone(kv)
        cache.insert(InsertParams(key=RadixKey(tokens), value=kv))

        # Allocate req slot and write KV to req_to_token_pool
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid="test",
            origin_input_text="",
            origin_input_ids=tokens,
            sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0),
        )
        req_pool_indices = pool.alloc([req])
        self.assertIsNotNone(req_pool_indices)
        req_pool_idx = req_pool_indices[0]
        pool.write((req_pool_idx, slice(0, len(tokens))), kv)

        req.fill_ids = tokens

        cache.evict_swa(req, prelen=len(tokens), attention_chunk_size=0)

        # Now match — the tree shares the same KV indices whose SWA was freed.
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        # After evict_swa(prelen=12, window=4), positions [0,8) have SWA freed.
        # For L=12: w=[8,12) all valid → ok.
        self.assertEqual(len(result.device_indices), 12)

        # For a query of length 8 (where evicted SWA ends):
        # L=8: w=[4,8) → indices [4..7], but those were freed → bad
        # ...need to find the largest L where window is valid.
        result8 = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens[:8])))
        # L=8: w=[4,8) → positions 4-7 freed → bad
        # L=4: w=[0,4) → positions 0-3 freed → bad
        # All L from 1-8 have window containing freed positions → 0
        self.assertEqual(len(result8.device_indices), 0)

    def test_re_match_gives_correct_last_node(self):
        """After trimming, the re-matched result should have a valid last_device_node
        that corresponds to the trimmed prefix in the tree.
        """
        cache, pool, alloc = _build_simple_swa_cache(sliding_window_size=4)
        # Insert two segments: [0,1,2,3,4] and [0,1,2,3,4,5,6,7,8,9]
        tokens_short = list(range(5))
        kv_short = alloc.alloc(len(tokens_short))
        self.assertIsNotNone(kv_short)
        cache.insert(InsertParams(key=RadixKey(tokens_short), value=kv_short))

        tokens_long = list(range(10))
        kv_long = alloc.alloc(len(tokens_long))
        self.assertIsNotNone(kv_long)
        cache.insert(InsertParams(key=RadixKey(tokens_long), value=kv_long))

        # Free SWA of tokens [5..9] (the extension beyond 5)
        alloc.free_swa(kv_long[5:])

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens_long)))
        # L=10: w=[6,10) → freed → bad
        # ... trimmed to L=5: w=[1,5) all valid → ok
        self.assertEqual(len(result.device_indices), 5)
        # last_device_node should NOT be root (should be the node at depth 5)
        self.assertNotEqual(result.last_device_node, cache.root_node)


# ====================================================================== #
#  HiSimpleSWARadixCache._compute_safe_prefix_len tests
#  (Uses mocking to avoid full HiCache infrastructure)
# ====================================================================== #


class TestHiSimpleSWAComputeSafePrefixLen(CustomTestCase):
    """Test HiSimpleSWARadixCache._compute_safe_prefix_len.

    Since HiSimpleSWARadixCache requires a full HiCache setup (host memory,
    cache controller, etc.), we test _compute_safe_prefix_len by:
    1. Creating a minimal mock that provides the required attributes.
    2. Calling the method directly with crafted MatchResult objects.
    """

    def setUp(self):
        _skip_if_no_cuda()
        self.device = "cuda"

    def _make_hi_cache_mock(self, sliding_window_size: int = 4):
        """Create a minimal mock of HiSimpleSWARadixCache for testing
        _compute_safe_prefix_len."""
        from sglang.srt.mem_cache.hiradix_cache import HiSimpleSWARadixCache

        mock = MagicMock(spec=HiSimpleSWARadixCache)
        mock.sliding_window_size = sliding_window_size
        mock.device = self.device
        mock.root_node = TreeNode()

        # Use the real _compute_safe_prefix_len method
        mock._compute_safe_prefix_len = (
            HiSimpleSWARadixCache._compute_safe_prefix_len.__get__(mock)
        )
        return mock

    def _make_evicted_node_chain(
        self,
        host_values: list[torch.Tensor],
        host_swa_values: list[torch.Tensor],
        parent_node: TreeNode,
    ) -> tuple[TreeNode, TreeNode]:
        """Create a chain of evicted nodes with specified host_values.

        Returns (first_non_evicted_ancestor, deepest_evicted_node).
        """
        # parent_node is the first non-evicted ancestor
        current = parent_node
        deepest = parent_node
        for hv, hswa in zip(host_values, host_swa_values):
            child = TreeNode()
            child.parent = current
            child.value = None  # evicted
            child.host_value = hv
            child._test_host_swa = hswa  # stash for mock lookup
            current.children["test"] = child
            current = child
            deepest = child
        return parent_node, deepest

    def test_device_only_all_valid(self):
        """Device-only match, all SWA valid → (device_len, 0)."""
        mock = self._make_hi_cache_mock(sliding_window_size=4)

        # Setup device allocator mapping: all valid
        device_indices = torch.arange(1, 9, device=self.device, dtype=torch.int64)
        mapping = torch.arange(100, 108, device=self.device, dtype=torch.int64)
        mock.token_to_kv_pool_allocator = MagicMock()
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping = torch.zeros(
            200, device=self.device, dtype=torch.int64
        )
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping[device_indices] = (
            mapping
        )

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=mock.root_node,
            last_host_node=mock.root_node,
            host_hit_length=0,
        )
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 8)
        self.assertEqual(safe_host, 0)

    def test_device_only_tail_freed(self):
        """Device-only match, last token's SWA freed."""
        mock = self._make_hi_cache_mock(sliding_window_size=4)

        device_indices = torch.arange(1, 9, device=self.device, dtype=torch.int64)
        swa_mapping = torch.zeros(200, device=self.device, dtype=torch.int64)
        for i in range(8):
            swa_mapping[device_indices[i]] = i + 100
        # Free last token's SWA
        swa_mapping[device_indices[7]] = 0
        mock.token_to_kv_pool_allocator = MagicMock()
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping = swa_mapping

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=mock.root_node,
            last_host_node=mock.root_node,
            host_hit_length=0,
        )
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 7)
        self.assertEqual(safe_host, 0)

    def test_device_plus_host_all_valid(self):
        """Device + host match, all SWA valid → (device_len, host_len)."""
        mock = self._make_hi_cache_mock(sliding_window_size=4)

        device_indices = torch.arange(1, 5, device=self.device, dtype=torch.int64)
        swa_mapping = torch.zeros(200, device=self.device, dtype=torch.int64)
        for i in range(4):
            swa_mapping[device_indices[i]] = i + 100
        mock.token_to_kv_pool_allocator = MagicMock()
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping = swa_mapping

        # Create evicted nodes with valid host SWA
        host_vals = [torch.tensor([10, 11], device="cpu", dtype=torch.int64)]
        host_swa_vals = [torch.tensor([200, 201], device="cpu", dtype=torch.int64)]

        ancestor = TreeNode()
        ancestor.value = torch.tensor([1], dtype=torch.int64)  # non-evicted
        _, deepest = self._make_evicted_node_chain(host_vals, host_swa_vals, ancestor)

        # Mock cache_controller.get_swa_host_indices
        host_swa_mapping = torch.zeros(200, dtype=torch.int64, device=self.device)
        host_swa_mapping[10] = 200
        host_swa_mapping[11] = 201
        mock.cache_controller = MagicMock()
        mock.cache_controller.get_swa_host_indices = lambda x: host_swa_mapping[x]

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=ancestor,
            last_host_node=deepest,
            host_hit_length=2,
        )
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 4)
        self.assertEqual(safe_host, 2)  # entire match is safe, all 2 host tokens usable

    def test_device_plus_host_host_swa_freed(self):
        """Device + host, host portion has freed SWA → trimmed host."""
        mock = self._make_hi_cache_mock(sliding_window_size=3)

        # 4 device tokens, all SWA valid
        device_indices = torch.arange(1, 5, device=self.device, dtype=torch.int64)
        swa_mapping = torch.zeros(200, device=self.device, dtype=torch.int64)
        for i in range(4):
            swa_mapping[device_indices[i]] = i + 100
        mock.token_to_kv_pool_allocator = MagicMock()
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping = swa_mapping

        # 3 host tokens, last one has SWA freed
        host_vals = [torch.tensor([10, 11, 12], dtype=torch.int64, device="cpu")]
        host_swa_vals = [torch.tensor([200, 201, 0], dtype=torch.int64, device="cpu")]

        ancestor = TreeNode()
        ancestor.value = torch.tensor([1], dtype=torch.int64)
        _, deepest = self._make_evicted_node_chain(host_vals, host_swa_vals, ancestor)

        host_swa_mapping = torch.zeros(200, dtype=torch.int64, device=self.device)
        host_swa_mapping[10] = 200
        host_swa_mapping[11] = 201
        host_swa_mapping[12] = 0
        mock.cache_controller = MagicMock()
        mock.cache_controller.get_swa_host_indices = lambda x: host_swa_mapping[x]

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=ancestor,
            last_host_node=deepest,
            host_hit_length=3,
        )
        # total=7, window=3
        # Combined SWA: [100,101,102,103, 200,201,0]
        # L=7: w=[4,7) → [200,201,0] → bad
        # L=6: w=[3,6) → [103,200,201] → ok
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 4)  # all device tokens safe
        self.assertEqual(safe_host, 2)  # 2 host tokens safe

    def test_empty_match(self):
        """Empty match → (0, 0)."""
        mock = self._make_hi_cache_mock(sliding_window_size=4)
        mock.token_to_kv_pool_allocator = MagicMock()

        result = MatchResult(
            device_indices=torch.empty(0, device=self.device, dtype=torch.int64),
            last_device_node=mock.root_node,
            last_host_node=mock.root_node,
            host_hit_length=0,
        )
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 0)
        self.assertEqual(safe_host, 0)

    def test_device_trimmed_means_host_zero(self):
        """When device portion needs trimming, host portion must be 0."""
        mock = self._make_hi_cache_mock(sliding_window_size=3)

        # 4 device tokens, last 2 have SWA freed
        device_indices = torch.arange(1, 5, device=self.device, dtype=torch.int64)
        swa_mapping = torch.zeros(200, device=self.device, dtype=torch.int64)
        swa_mapping[1] = 100
        swa_mapping[2] = 101
        swa_mapping[3] = 0  # freed
        swa_mapping[4] = 0  # freed
        mock.token_to_kv_pool_allocator = MagicMock()
        mock.token_to_kv_pool_allocator.full_to_swa_index_mapping = swa_mapping

        # 2 host tokens, all valid
        host_vals = [torch.tensor([10, 11], dtype=torch.int64, device="cpu")]
        ancestor = TreeNode()
        ancestor.value = torch.tensor([1], dtype=torch.int64)
        _, deepest = self._make_evicted_node_chain(
            host_vals,
            [torch.tensor([200, 201], dtype=torch.int64, device="cpu")],
            ancestor,
        )

        host_swa_mapping = torch.zeros(200, dtype=torch.int64, device=self.device)
        host_swa_mapping[10] = 200
        host_swa_mapping[11] = 201
        mock.cache_controller = MagicMock()
        mock.cache_controller.get_swa_host_indices = lambda x: host_swa_mapping[x]

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=ancestor,
            last_host_node=deepest,
            host_hit_length=2,
        )
        # total=6, window=3
        # Combined SWA: [100,101,0,0,200,201]
        # L=6: w=[3,6) → [0,200,201] bad
        # L=5: w=[2,5) → [0,0,200] bad
        # L=4: w=[1,4) → [101,0,0] bad
        # L=3: w=[0,3) → [100,101,0] bad
        # L=2: w=[0,2) → [100,101] ok
        safe_dev, safe_host = mock._compute_safe_prefix_len(result)
        self.assertEqual(safe_dev, 2)  # trimmed device
        self.assertEqual(safe_host, 0)  # host must be 0 when device is trimmed


# ====================================================================== #
#  _collect_evicted_host_indices tests
# ====================================================================== #


class TestCollectEvictedHostIndices(CustomTestCase):
    """Test _collect_evicted_host_indices (module-level function).

    This function walks UP from last_host_node (deepest evicted) to
    last_device_node (first non-evicted ancestor), collects host_value
    tensors in reverse, then reverses to root-to-leaf order.

    No GPU required — uses CPU tensors only.
    """

    def _make_chain(self, host_values: list[torch.Tensor]) -> tuple[TreeNode, TreeNode]:
        """Build a chain: ancestor → evicted_1 → evicted_2 → ... → evicted_N.

        ancestor has value (non-evicted), all others are evicted with host_value.
        Returns (ancestor, deepest_evicted).
        """
        ancestor = TreeNode()
        ancestor.value = torch.tensor([999], dtype=torch.int64)  # non-evicted
        ancestor.host_value = torch.tensor([888], dtype=torch.int64)  # backuped

        current = ancestor
        for hv in host_values:
            child = TreeNode()
            child.parent = current
            child.value = None  # evicted
            child.host_value = hv
            current.children["child"] = child
            current = child

        return ancestor, current  # (first non-evicted, deepest evicted)

    def test_single_evicted_node(self):
        """One evicted node between ancestor and deepest."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        hv = torch.tensor([10, 11, 12], dtype=torch.int64)
        ancestor, deepest = self._make_chain([hv])

        result = _collect_evicted_host_indices(deepest, ancestor)
        self.assertTrue(torch.equal(result, hv))

    def test_multiple_evicted_nodes(self):
        """Three evicted nodes — result should be concatenated root-to-leaf."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        hv1 = torch.tensor([10, 11], dtype=torch.int64)
        hv2 = torch.tensor([20, 21, 22], dtype=torch.int64)
        hv3 = torch.tensor([30], dtype=torch.int64)
        ancestor, deepest = self._make_chain([hv1, hv2, hv3])

        result = _collect_evicted_host_indices(deepest, ancestor)
        expected = torch.cat([hv1, hv2, hv3])
        self.assertTrue(torch.equal(result, expected))

    def test_order_is_root_to_leaf(self):
        """Verify the order is root-to-leaf, not leaf-to-root."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        hv1 = torch.tensor([1], dtype=torch.int64)
        hv2 = torch.tensor([2], dtype=torch.int64)
        hv3 = torch.tensor([3], dtype=torch.int64)
        ancestor, deepest = self._make_chain([hv1, hv2, hv3])

        result = _collect_evicted_host_indices(deepest, ancestor)
        # Root-to-leaf: hv1 first, hv3 last
        self.assertEqual(result.tolist(), [1, 2, 3])

    def test_last_host_node_equals_last_device_node(self):
        """Edge case: last_host_node IS last_device_node → empty result.

        This shouldn't normally happen (host_hit_length would be 0),
        but the function should handle it gracefully — returns empty cat.
        """
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        node = TreeNode()
        node.value = torch.tensor([1], dtype=torch.int64)  # non-evicted

        # No evicted nodes between them — but the while loop condition
        # `node is not last_device_node` is immediately False.
        # torch.cat([]) would fail, so let's verify behavior.
        # Actually: the loop collects nothing, host_values is empty,
        # torch.cat([]) raises an error. This case shouldn't occur in
        # practice because host_hit_length > 0 is checked before calling.
        # We test that it raises to document the precondition.
        with self.assertRaises(RuntimeError):
            _collect_evicted_host_indices(node, node)

    def test_single_token_host_value(self):
        """Each evicted node has a single-token host_value."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        hv1 = torch.tensor([100], dtype=torch.int64)
        hv2 = torch.tensor([200], dtype=torch.int64)
        ancestor, deepest = self._make_chain([hv1, hv2])

        result = _collect_evicted_host_indices(deepest, ancestor)
        self.assertEqual(result.tolist(), [100, 200])

    def test_large_chain(self):
        """Many evicted nodes in chain."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        num_nodes = 20
        host_values = [
            torch.tensor([i * 10 + j for j in range(3)], dtype=torch.int64)
            for i in range(num_nodes)
        ]
        ancestor, deepest = self._make_chain(host_values)

        result = _collect_evicted_host_indices(deepest, ancestor)
        expected = torch.cat(host_values)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(len(result), num_nodes * 3)

    def test_asserts_on_non_evicted_intermediate(self):
        """If an intermediate node is not evicted, assert should fire.

        Build: ancestor → evicted_1 → NON_EVICTED → evicted_3
        Call with (evicted_3, ancestor) — should assert on NON_EVICTED.
        """
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        ancestor = TreeNode()
        ancestor.value = torch.tensor([999], dtype=torch.int64)

        evicted_1 = TreeNode()
        evicted_1.parent = ancestor
        evicted_1.value = None
        evicted_1.host_value = torch.tensor([10], dtype=torch.int64)
        ancestor.children["c1"] = evicted_1

        non_evicted = TreeNode()
        non_evicted.parent = evicted_1
        non_evicted.value = torch.tensor([50], dtype=torch.int64)  # NOT evicted
        non_evicted.host_value = torch.tensor([51], dtype=torch.int64)
        evicted_1.children["c2"] = non_evicted

        evicted_3 = TreeNode()
        evicted_3.parent = non_evicted
        evicted_3.value = None
        evicted_3.host_value = torch.tensor([30], dtype=torch.int64)
        non_evicted.children["c3"] = evicted_3

        # Walking from evicted_3 up: first node is evicted_3 (evicted, ok),
        # next is non_evicted (not evicted) → assert should fire
        with self.assertRaises(AssertionError):
            _collect_evicted_host_indices(evicted_3, ancestor)

    def test_asserts_on_missing_backup(self):
        """If an evicted node has no host_value (not backuped), assert should fire."""
        from sglang.srt.mem_cache.hiradix_cache import _collect_evicted_host_indices

        ancestor = TreeNode()
        ancestor.value = torch.tensor([999], dtype=torch.int64)

        evicted_no_backup = TreeNode()
        evicted_no_backup.parent = ancestor
        evicted_no_backup.value = None  # evicted
        evicted_no_backup.host_value = None  # NOT backuped!
        ancestor.children["c1"] = evicted_no_backup

        with self.assertRaises(AssertionError):
            _collect_evicted_host_indices(evicted_no_backup, ancestor)


if __name__ == "__main__":
    unittest.main()
