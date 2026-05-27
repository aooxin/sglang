import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.simple_swa_radix_cache import SimpleSWARadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-small")


class TestSimpleSWARadixCache(CustomTestCase):
    device: str = "cuda"
    req_size: int = 0
    max_context_len: int = 0
    kv_size: int = 0
    kv_size_swa: int = 0
    sliding_window_size: int = 0
    head_num: int = 0
    head_dim: int = 0
    num_layers: int = 0
    global_interval: int = 0
    dtype: torch.dtype = torch.bfloat16
    req_to_token_pool: ReqToTokenPool | None = None
    kv_pool: SWAKVPool | None = None
    allocator: SWATokenToKVPoolAllocator | None = None
    cache: SimpleSWARadixCache | None = None

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for SWA cache tests")

        self.device = "cuda"
        self.req_size = 4
        self.max_context_len = 64
        self.kv_size = 128
        self.kv_size_swa = 64
        self.sliding_window_size = 4
        self.head_num = 8
        self.head_dim = 128
        self.num_layers = 16
        self.global_interval = 4
        self.dtype = torch.bfloat16

        full_attention_layer_ids = [
            i for i in range(0, self.num_layers, self.global_interval)
        ]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(self.num_layers) if i not in full_attention_layer_ids_set
        ]

        self.req_to_token_pool = ReqToTokenPool(
            size=self.req_size,
            max_context_len=self.max_context_len,
            device=self.device,
            enable_memory_saver=False,
        )
        self.kv_pool = SWAKVPool(
            size=self.kv_size,
            size_swa=self.kv_size_swa,
            page_size=1,
            dtype=self.dtype,
            head_num=self.head_num,
            head_dim=self.head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=self.kv_size,
            size_swa=self.kv_size_swa,
            page_size=1,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kv_pool,
            need_sort=False,
        )
        self.cache = SimpleSWARadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                sliding_window_size=self.sliding_window_size,
            ),
        )

    def _require_pool(self) -> ReqToTokenPool:
        if self.req_to_token_pool is None:
            self.fail("req_to_token_pool is not initialized")
        return self.req_to_token_pool

    def _require_allocator(self) -> SWATokenToKVPoolAllocator:
        if self.allocator is None:
            self.fail("allocator is not initialized")
        return self.allocator

    def _require_cache(self) -> SimpleSWARadixCache:
        if self.cache is None:
            self.fail("cache is not initialized")
        return self.cache

    def _alloc_req_slot(self) -> int:
        pool = self._require_pool()
        placeholder = Req(
            rid="placeholder",
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0),
        )
        idx_list = pool.alloc([placeholder])
        if not idx_list:
            self.fail("Failed to allocate request slot")
        return idx_list[0]

    def _alloc_kv_indices(self, num_tokens: int) -> torch.Tensor:
        allocator = self._require_allocator()
        kv_indices = allocator.alloc(num_tokens)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        return kv_indices

    def _make_req(self, token_ids: list[int], req_pool_idx: int) -> Req:
        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        req = Req(
            rid="req",
            origin_input_text="",
            origin_input_ids=token_ids,
            sampling_params=sampling_params,
        )
        req.req_pool_idx = req_pool_idx
        req.fill_ids = token_ids
        return req

    def _build_cache(
        self,
        sliding_window_size: int,
        is_eagle: bool = False,
        speculative_num_steps: int | None = None,
        page_size: int = 1,
        kv_size: int | None = None,
        kv_size_swa: int | None = None,
        max_context_len: int | None = None,
        req_size: int | None = None,
    ):
        kv_size = kv_size or self.kv_size
        kv_size_swa = kv_size_swa or self.kv_size_swa
        max_context_len = max_context_len or self.max_context_len
        req_size = req_size or self.req_size

        full_attention_layer_ids = [
            i for i in range(0, self.num_layers, self.global_interval)
        ]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(self.num_layers) if i not in full_attention_layer_ids_set
        ]
        req_to_token_pool = ReqToTokenPool(
            size=req_size,
            max_context_len=max_context_len,
            device=self.device,
            enable_memory_saver=False,
        )
        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=self.dtype,
            head_num=self.head_num,
            head_dim=self.head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=self.device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = SimpleSWARadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                is_eagle=is_eagle,
                sliding_window_size=sliding_window_size,
            ),
            speculative_num_steps=speculative_num_steps,
        )
        return cache, req_to_token_pool, allocator

    # ===== A. evict_swa boundary cases =====

    def test_evict_swa_boundary_cases(self):
        pool = self._require_pool()
        cache = self._require_cache()
        allocator = self._require_allocator()

        cases = [
            ("shorter_than_window", 3, 0),
            ("exactly_at_window", self.sliding_window_size, 0),
            ("one_beyond_window", self.sliding_window_size + 1, 1),
        ]

        for name, seq_len, expected_freed in cases:
            with self.subTest(name=name):
                req_pool_idx = self._alloc_req_slot()
                token_ids = list(range(seq_len))
                kv_indices = self._alloc_kv_indices(seq_len)
                pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
                req = self._make_req(token_ids, req_pool_idx)

                cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

                mapping = allocator.full_to_swa_index_mapping
                self.assertEqual(req.swa_evicted_seqlen, expected_freed)
                self.assertTrue(torch.all(mapping[kv_indices[:expected_freed]] == 0))
                self.assertTrue(torch.all(mapping[kv_indices[expected_freed:]] > 0))

    def test_evict_swa_frees_outside_window_and_idempotent(self):
        seq_len = 10
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)

        req = self._make_req(token_ids, req_pool_idx)

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)
        freed_len = seq_len - self.sliding_window_size
        freed_indices = kv_indices[:freed_len]
        kept_indices = kv_indices[freed_len:]

        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[freed_indices] == 0))
        self.assertTrue(torch.all(mapping[kept_indices] > 0))
        self.assertEqual(req.swa_evicted_seqlen, freed_len)

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)
        self.assertTrue(torch.all(mapping[freed_indices] == 0))
        self.assertEqual(req.swa_evicted_seqlen, freed_len)

    def test_evict_swa_multi_chunk_progression(self):
        cache, pool, allocator = self._build_cache(self.sliding_window_size)
        seq_len = 12
        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]
        token_ids = list(range(seq_len))
        kv_indices = allocator.alloc(seq_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)

        cache.evict_swa(req, prelen=8, attention_chunk_size=0)
        self.assertEqual(req.swa_evicted_seqlen, 4)
        cache.evict_swa(req, prelen=12, attention_chunk_size=0)
        self.assertEqual(req.swa_evicted_seqlen, 8)

        mapping = allocator.full_to_swa_index_mapping
        if mapping is None:
            self.fail("full_to_swa_index_mapping is not initialized")
        self.assertTrue(torch.all(mapping[kv_indices[:8]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[8:]] > 0))

    def test_evict_swa_prelen_zero(self):
        pool = self._require_pool()
        cache = self._require_cache()

        seq_len = 6
        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)

        cache.evict_swa(req, prelen=0, attention_chunk_size=0)
        self.assertEqual(req.swa_evicted_seqlen, 0)

    def test_evict_swa_page_aligned_when_paged(self):
        page_size = 4
        cache, pool, allocator = self._build_cache(
            self.sliding_window_size,
            page_size=page_size,
            kv_size=page_size * 32,
            kv_size_swa=page_size * 32,
            max_context_len=page_size * 8,
        )
        seq_len = 12
        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]
        token_ids = list(range(seq_len))
        kv_indices = allocator.alloc(seq_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)

        cache.evict_swa(req, prelen=10, attention_chunk_size=0)

        mapping = allocator.full_to_swa_index_mapping
        self.assertEqual(req.swa_evicted_seqlen, 4)
        self.assertTrue(torch.all(mapping[kv_indices[:4]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[4:]] > 0))

    def test_evict_swa_respects_tree_prefix_len(self):
        seq_len = 12
        tree_prefix_len = 4
        cache, pool, allocator = self._build_cache(self.sliding_window_size)

        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]
        token_ids = list(range(seq_len))
        kv_indices = allocator.alloc(seq_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)

        cache.evict_swa(
            req,
            prelen=seq_len,
            attention_chunk_size=0,
            tree_prefix_len=tree_prefix_len,
        )

        mapping = allocator.full_to_swa_index_mapping
        self.assertEqual(req.swa_evicted_seqlen, seq_len - self.sliding_window_size)
        self.assertTrue(torch.all(mapping[kv_indices[:tree_prefix_len]] > 0))
        self.assertTrue(
            torch.all(
                mapping[kv_indices[tree_prefix_len : req.swa_evicted_seqlen]] == 0
            )
        )
        self.assertTrue(torch.all(mapping[kv_indices[req.swa_evicted_seqlen :]] > 0))

    # ===== B. _compute_safe_prefix_len edge cases =====

    def test_compute_safe_prefix_len_cases(self):
        cache = self._require_cache()
        cases = [
            ("all_valid", [1, 2, 3, 4, 5, 6, 7, 8], 8),
            ("all_invalid", [0, 0, 0, 0, 0, 0], 0),
            ("empty", [], 0),
            ("partial_tail", [1, 2, 0, 4, 5, 6], 2),
            ("single_valid", [5], 1),
            ("single_invalid", [0], 0),
            ("hole_in_middle", [1, 2, 0, 0, 5, 6, 7, 8], 8),
        ]

        for name, raw_indices, expected in cases:
            with self.subTest(name=name):
                swa_indices = torch.tensor(
                    raw_indices, dtype=torch.int64, device=self.device
                )
                safe_len = cache._compute_safe_prefix_len(swa_indices)
                self.assertEqual(safe_len, expected)

    # ===== C. match_prefix with SWA safety =====

    def test_match_prefix_requires_swa_tail_window(self):
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        token_ids = list(range(10))
        kv_indices = self._alloc_kv_indices(len(token_ids))
        _ = cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        req_pool_idx = self._alloc_req_slot()
        pool.write((req_pool_idx, slice(0, len(token_ids))), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        cache.evict_swa(req, prelen=len(token_ids), attention_chunk_size=0)

        result_full = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(result_full.device_indices), len(token_ids))

        result_short = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids[:6])))
        self.assertEqual(len(result_short.device_indices), 0)

    def test_match_prefix_window_larger_than_sequence(self):
        cache, pool, allocator = self._build_cache(12)
        token_ids = list(range(10))
        kv_indices = allocator.alloc(len(token_ids))
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        _ = cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]
        pool.write((req_pool_idx, slice(0, len(token_ids))), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        cache.evict_swa(req, prelen=len(token_ids), attention_chunk_size=0)

        result_full = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(result_full.device_indices), len(token_ids))

        result_short = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids[:6])))
        self.assertEqual(len(result_short.device_indices), 6)

    def test_match_prefix_empty_tree(self):
        cache = self._require_cache()
        token_ids = list(range(5))
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(result.device_indices), 0)

    def test_match_prefix_no_eviction_yet(self):
        cache = self._require_cache()
        allocator = self._require_allocator()

        token_ids = list(range(8))
        kv_indices = self._alloc_kv_indices(len(token_ids))
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(result.device_indices), len(token_ids))

    def test_match_prefix_partial_token_overlap(self):
        cache = self._require_cache()

        cached_ids = list(range(10))
        kv_indices = self._alloc_kv_indices(len(cached_ids))
        cache.insert(InsertParams(key=RadixKey(cached_ids), value=kv_indices))

        query_ids = list(range(6)) + [100, 101, 102]
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(query_ids)))
        self.assertEqual(len(result.device_indices), 6)

    # ===== D. cache_unfinished_req =====

    def test_cache_unfinished_req_chunked_updates_prefix_and_frees(self):
        seq_len = 12
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)

        req = self._make_req(token_ids, req_pool_idx)
        cache.cache_unfinished_req(req, chunked=True)

        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        self.assertTrue(torch.equal(req.prefix_indices, kv_indices))

        freed_len = seq_len - self.sliding_window_size
        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices[:freed_len]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[freed_len:]] > 0))

    def test_chunked_req_preserves_request_local_prefix_across_rounds(self):
        cache = self._require_cache()
        allocator = self._require_allocator()

        tree_token_ids = [0, 1, 2, 3]
        tree_kv = self._alloc_kv_indices(len(tree_token_ids))
        cache.insert(InsertParams(key=RadixKey(tree_token_ids), value=tree_kv))

        local_tail = self._alloc_kv_indices(4)

        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        req = Req(
            rid="req_chunked_roundtrip",
            origin_input_text="",
            origin_input_ids=list(range(12)),
            sampling_params=sampling_params,
        )
        req.output_ids = []
        req.prefix_indices = torch.cat([tree_kv, local_tail]).to(torch.int64)
        req.cache_protected_len = len(tree_token_ids)
        req.is_chunked = 1

        req.init_next_round_input(cache)

        self.assertEqual(len(req.prefix_indices), 8)
        self.assertTrue(torch.equal(req.prefix_indices[:4], tree_kv))
        self.assertTrue(torch.equal(req.prefix_indices[4:], local_tail))
        self.assertEqual(req.cache_protected_len, len(tree_token_ids))
        self.assertEqual(req.extend_input_len, 4)

    def test_cache_unfinished_req_non_chunked_no_tree_insert(self):
        """Non-chunked cache_unfinished_req should NOT insert into the tree.

        It only frees SWA KV outside the window and saves prefix_indices.
        Tree insertion is deferred to cache_finished_req.
        """
        seq_len = 10
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)

        req = self._make_req(token_ids, req_pool_idx)
        req.last_node = cache.root_node
        cache.inc_lock_ref(req.last_node)

        cache.cache_unfinished_req(req, chunked=False)

        # Tree should remain empty — no insertion during unfinished
        tree_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(tree_result.device_indices), 0)

        # prefix_indices should be saved
        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        self.assertTrue(torch.equal(req.prefix_indices, kv_indices))

        # SWA KV outside the window should be freed
        freed_len = seq_len - self.sliding_window_size
        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices[:freed_len]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[freed_len:]] > 0))

        # last_node and cache_protected_len should be unchanged
        self.assertEqual(req.last_node, cache.root_node)
        self.assertEqual(req.cache_protected_len, 0)

    # ===== E. cache_finished_req =====

    def test_cache_finished_req_frees_swa_and_inserts_tree(self):
        cache, pool, allocator = self._build_cache(self.sliding_window_size)
        seq_len = 10
        output_len = 4

        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]

        origin_input_ids = list(range(seq_len))
        output_ids = list(range(100, 100 + output_len))
        all_token_len = seq_len + max(output_len - 1, 0)
        kv_indices = allocator.alloc(all_token_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, all_token_len)), kv_indices)

        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        req = Req(
            rid="req_finished",
            origin_input_text="",
            origin_input_ids=origin_input_ids,
            sampling_params=sampling_params,
        )
        req.req_pool_idx = req_pool_idx
        req.fill_ids = origin_input_ids + output_ids[: max(output_len - 1, 0)]
        req.output_ids = output_ids
        req.kv_committed_len = all_token_len
        req.last_node = cache.root_node
        cache.inc_lock_ref(req.last_node)

        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices] > 0))

        cache.cache_finished_req(req)

        freed_swa_end = max(0, all_token_len - self.sliding_window_size)
        if freed_swa_end > 0:
            self.assertTrue(torch.all(mapping[kv_indices[:freed_swa_end]] == 0))

        all_token_ids = (origin_input_ids + output_ids)[:all_token_len]
        tree_result = cache.match_prefix(MatchPrefixParams(key=RadixKey(all_token_ids)))
        self.assertGreater(len(tree_result.device_indices), 0)

    def test_cache_finished_req_short_sequence(self):
        cache, pool, allocator = self._build_cache(self.sliding_window_size)

        seq_len = 2
        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]

        origin_input_ids = list(range(seq_len))
        output_ids = [99]
        all_token_len = seq_len + max(len(output_ids) - 1, 0)
        kv_indices = allocator.alloc(all_token_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, all_token_len)), kv_indices)

        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        req = Req(
            rid="req_short",
            origin_input_text="",
            origin_input_ids=origin_input_ids,
            sampling_params=sampling_params,
        )
        req.req_pool_idx = req_pool_idx
        req.fill_ids = origin_input_ids
        req.output_ids = output_ids
        req.kv_committed_len = all_token_len
        req.last_node = cache.root_node
        cache.inc_lock_ref(req.last_node)

        mapping = allocator.full_to_swa_index_mapping
        pre_swa = mapping[kv_indices].clone()

        cache.cache_finished_req(req)

        self.assertTrue(torch.all(mapping[kv_indices[:all_token_len]] >= 0))

    # ===== F. evict (tree-level eviction) =====

    def test_evict_reclaims_full_and_swa_separately(self):
        cache, pool, allocator = self._build_cache(
            self.sliding_window_size, kv_size=256, kv_size_swa=128
        )

        token_ids_a = list(range(10))
        kv_a = allocator.alloc(len(token_ids_a))
        if kv_a is None:
            self.fail("Failed to allocate KV indices for entry A")
        cache.insert(InsertParams(key=RadixKey(token_ids_a), value=kv_a))

        token_ids_b = list(range(20, 30))
        kv_b = allocator.alloc(len(token_ids_b))
        if kv_b is None:
            self.fail("Failed to allocate KV indices for entry B")
        cache.insert(InsertParams(key=RadixKey(token_ids_b), value=kv_b))

        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        cache.evict(EvictParams(num_tokens=10, swa_num_tokens=10))

        full_after = allocator.full_available_size()
        swa_after = allocator.swa_available_size()

        self.assertGreater(full_after, full_before)
        self.assertGreater(swa_after, swa_before)

    def test_evict_skips_locked_nodes(self):
        cache, pool, allocator = self._build_cache(self.sliding_window_size)

        token_ids = list(range(10))
        kv_indices = allocator.alloc(len(token_ids))
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        locked_node = result.last_device_node
        cache.inc_lock_ref(locked_node)

        full_before = allocator.full_available_size()
        cache.evict(EvictParams(num_tokens=10, swa_num_tokens=10))
        full_after = allocator.full_available_size()

        self.assertEqual(full_before, full_after)

        cache.dec_lock_ref(locked_node)

    def test_evict_no_op_when_both_zero(self):
        cache = self._require_cache()
        allocator = self._require_allocator()

        token_ids = list(range(5))
        kv_indices = self._alloc_kv_indices(len(token_ids))
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()

        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=0))

        self.assertEqual(allocator.full_available_size(), full_before)
        self.assertEqual(allocator.swa_available_size(), swa_before)

    # ===== G. EAGLE speculative mode =====

    def test_eagle_mode_match_prefix_adjusts_for_speculative_steps(self):
        speculative_num_steps = 3
        sliding_window = 4
        cache, pool, allocator = self._build_cache(
            sliding_window,
            is_eagle=True,
            speculative_num_steps=speculative_num_steps,
        )

        token_ids = list(range(20))
        kv_indices = allocator.alloc(len(token_ids))
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        # In EAGLE mode the radix tree stores bigram keys, so N tokens -> N-1
        # tree slots. The match length therefore equals len(token_ids) - 1.
        kv_len = len(token_ids) - 1
        result_full = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        self.assertEqual(len(result_full.device_indices), kv_len)

        mapping = allocator.full_to_swa_index_mapping
        gap_start = kv_len - sliding_window - 3
        gap_end = kv_len - sliding_window + 1
        allocator.free_swa(kv_indices[gap_start:gap_end])

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
        matched_len = len(result.device_indices)

        self.assertGreater(matched_len, 0)
        self.assertLess(matched_len, kv_len)


if __name__ == "__main__":
    unittest.main()
