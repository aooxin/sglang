"""Fake PD requests must retain chunk progress without publishing synthetic KV.

Exercise the real admission, allocation, cache and release paths with small KV
heads. No model or remote server is needed. The 64K/48K case reproduces the
original leak; the 960K case also exceeds the request's SWA pool capacity.
"""

import unittest
from array import array

import torch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import SWAChunkCache
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req, release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, restore_context, snapshot_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

CHUNK_SIZE = 49152
FULL_CAPACITY = 1048576
SWA_CAPACITY = 196608
WINDOW_SIZE = 1024


@unittest.skipUnless(torch.cuda.is_available(), "paged allocation needs CUDA")
class TestFakePrefillChunking(unittest.TestCase):
    def setUp(self):
        self.addCleanup(restore_context, snapshot_context())
        self.enterContext(envs.SGLANG_DEBUG_MEMORY_POOL.override(True))
        self.enterContext(
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python")
        )
        publish(
            ServerArgs(
                model_path="dummy",
                attention_backend="triton",
                page_size=128,
                chunked_prefill_size=CHUNK_SIZE,
            ),
            role="scheduler",
        )

    def _cache(self, *, chunk_cache=False, page_size=128):
        pool = ReqToTokenPool(
            size=4,
            max_context_len=FULL_CAPACITY,
            device="cuda",
            enable_memory_saver=False,
        )
        kv_pool = SWAKVPool(
            size=FULL_CAPACITY,
            size_swa=SWA_CAPACITY,
            page_size=page_size,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cuda",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=FULL_CAPACITY,
            size_swa=SWA_CAPACITY,
            page_size=page_size,
            dtype=torch.float16,
            device="cuda",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=chunk_cache,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            chunked_prefill_size=CHUNK_SIZE,
            sliding_window_size=WINDOW_SIZE,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
        return SWAChunkCache(params) if chunk_cache else UnifiedRadixCache(params)

    def _req(self, cache, length, *, bootstrap_host=FAKE_BOOTSTRAP_HOST, rid="test"):
        req = Req(
            rid,
            origin_input_text=None,
            origin_input_ids=array("q", range(length)),
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_host=bootstrap_host,
        )
        req.init_next_round_input(cache)
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        return req

    def _batch(self, cache, reqs, *, overlap=True):
        seq_lens_cpu = torch.tensor([r.extend_range.end for r in reqs])
        return ScheduleBatch(
            reqs=reqs,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            tree_cache=cache,
            model_config=None,
            enable_overlap=overlap,
            forward_mode=ForwardMode.EXTEND,
            device="cuda",
            prefix_lens=[len(r.prefix_indices) for r in reqs],
            extend_lens=[r.extend_range.length for r in reqs],
            seq_lens=seq_lens_cpu.cuda(),
            seq_lens_cpu=seq_lens_cpu,
            extend_num_tokens=sum(r.extend_range.length for r in reqs),
        )

    def _allocate(self, cache, req, *, overlap=True):
        indices, _, _ = alloc_for_extend(self._batch(cache, [req], overlap=overlap))
        # prepare_for_extend advances this after alloc_for_extend returns.
        req.extend_batch_idx += 1
        return indices

    def _next_chunk(self, cache, req, *, overlap=True):
        adder = PrefillAdder(
            page_size=cache.page_size,
            tree_cache=cache,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=CHUNK_SIZE,
            rem_chunk_tokens=CHUNK_SIZE,
        )
        adder.add_chunked_req(req)
        self.assertEqual(len(adder.can_run_list), 1, "chunk stalled on KV capacity")
        self.assertIs(adder.can_run_list[0], req)
        return self._allocate(cache, req, overlap=overlap)

    def _match(self, cache, req):
        return cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids, req.extra_key))
        ).device_indices

    def _assert_all_free(self, cache):
        allocator = cache.token_to_kv_pool_allocator
        self.assertEqual(allocator.full_available_size(), FULL_CAPACITY)
        self.assertEqual(allocator.swa_available_size(), SWA_CAPACITY)
        self.assertEqual(cache.req_to_token_pool.available_size(), 4)
        for pool in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
            self.assertEqual(len(torch.unique(pool.free_pages)), len(pool.free_pages))
        self.assertFalse(bool(allocator.full_to_swa_index_mapping[:-1].any()))

    def test_fake_chunk_progress_keeps_every_allocation_reachable(self):
        for chunk_cache in (False, True):
            for page_size in (1, 128):
                with self.subTest(chunk_cache=chunk_cache, page_size=page_size):
                    cache = self._cache(chunk_cache=chunk_cache, page_size=page_size)
                    req = self._req(cache, 65536)
                    first = self._next_chunk(cache, req)
                    maybe_cache_unfinished_req(req, cache, chunked=True)

                    self.assertEqual(len(req.prefix_indices), 49152)
                    self.assertEqual(req.prefix_indices.dtype, torch.int64)
                    torch.testing.assert_close(req.prefix_indices, first)
                    self.assertEqual(req.kv.cache_protected_len, 0)
                    self.assertEqual(len(self._match(cache, req)), 0)

                    second = self._next_chunk(cache, req)
                    self.assertEqual(
                        (req.extend_range.start, req.extend_range.end), (49152, 65536)
                    )
                    row = cache.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, :65536
                    ]
                    torch.testing.assert_close(row.long(), torch.cat([first, second]))
                    release_kv_cache(req, cache)
                    self._assert_all_free(cache)

    def test_fake_abort_releases_all_chunks(self):
        cache = self._cache()
        req = self._req(cache, 131072)
        self._next_chunk(cache, req)
        maybe_cache_unfinished_req(req, cache, chunked=True)
        self._next_chunk(cache, req)

        req.finished_reason = FINISH_ABORT("test cancellation")
        release_kv_cache(req, cache, is_insert=False)

        self.assertFalse(req.kv.holds_kv)
        self.assertTrue(req.kv.is_kv_released)
        self._assert_all_free(cache)

    def test_fake_tail_at_chunk_and_page_boundaries_releases_once(self):
        for length, expected_ranges in (
            (49151, [(0, 49151)]),
            (49152, [(0, 49152)]),
            (49153, [(0, 49152), (49152, 49153)]),
            (65537, [(0, 49152), (49152, 65537)]),
        ):
            with self.subTest(length=length):
                cache = self._cache()
                req = self._req(cache, length)
                ranges = []
                for _ in expected_ranges:
                    self._next_chunk(cache, req)
                    ranges.append((req.extend_range.start, req.extend_range.end))
                    if req.extend_range.end < length:
                        maybe_cache_unfinished_req(req, cache, chunked=True)

                self.assertEqual(ranges, expected_ranges)
                release_kv_cache(req, cache)
                self._assert_all_free(cache)

    def test_long_fake_prompt_recycles_swa_without_caching(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                cache = self._cache()
                req = self._req(cache, 983040)
                ranges = []
                for _ in range(20):
                    self._next_chunk(cache, req, overlap=overlap)
                    ranges.append((req.extend_range.start, req.extend_range.end))
                    if req.extend_range.end < 983040:
                        maybe_cache_unfinished_req(req, cache, chunked=True)

                self.assertEqual(ranges[0], (0, 49152))
                self.assertEqual(ranges[-1], (933888, 983040))
                self.assertEqual(sum(end - start for start, end in ranges), 983040)
                self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
                self.assertGreater(req.kv.swa_evicted_seqlen, 0)
                self.assertEqual(req.kv.cache_protected_len, 0)
                self.assertEqual(len(self._match(cache, req)), 0)
                release_kv_cache(req, cache)
                self._assert_all_free(cache)

    def test_fake_swa_eviction_preserves_the_inflight_chunk_window(self):
        for overlap, batch_idx, expected_floor in (
            (True, 1, 0),
            (True, 2, 48128),
            (False, 1, 97280),
        ):
            with self.subTest(overlap=overlap, batch_idx=batch_idx):
                cache = self._cache()
                req = self._req(cache, 131072)
                req.set_extend_range(0, 98304)
                indices = self._allocate(cache, req)
                # Fixture with two allocated chunks, independent of the progress helper.
                req.prefix_indices = indices.clone()
                req.extend_batch_idx = batch_idx
                batch = self._batch(cache, [req], overlap=overlap)
                batch.maybe_evict_swa()

                self.assertEqual(req.kv.swa_evicted_seqlen, expected_floor)
                mapping = cache.token_to_kv_pool_allocator.full_to_swa_index_mapping[
                    indices
                ]
                self.assertFalse(bool(mapping[:expected_floor].any()))
                self.assertTrue(bool((mapping[expected_floor:] > 0).all()))
                release_kv_cache(req, cache)
                self._assert_all_free(cache)

    def test_mixed_batch_preserves_normal_requests_kv(self):
        cache = self._cache()
        fake = self._req(cache, 131072, rid="fake")
        normal = self._req(cache, 131072, bootstrap_host="10.0.0.1", rid="normal")
        for req in (fake, normal):
            req.set_extend_range(0, 98304)
            req.prefix_indices = self._allocate(cache, req).clone()
            req.extend_batch_idx = 2
        mapping = cache.token_to_kv_pool_allocator.full_to_swa_index_mapping
        normal_before = mapping[normal.prefix_indices].clone()

        self._batch(cache, [fake, normal]).maybe_evict_swa()

        self.assertEqual(fake.kv.swa_evicted_seqlen, 48128)
        self.assertEqual(normal.kv.swa_evicted_seqlen, 0)
        torch.testing.assert_close(mapping[normal.prefix_indices], normal_before)
        for req in (fake, normal):
            release_kv_cache(req, cache, is_insert=False)
        self._assert_all_free(cache)

    def test_normal_prefill_still_publishes_reusable_kv(self):
        for bootstrap_host in (None, "10.0.0.1"):
            with self.subTest(bootstrap_host=bootstrap_host):
                cache = self._cache()
                req = self._req(cache, 65536, bootstrap_host=bootstrap_host)
                self._next_chunk(cache, req)
                maybe_cache_unfinished_req(req, cache, chunked=True)
                self.assertEqual(req.kv.cache_protected_len, 49152)
                self.assertEqual(len(self._match(cache, req)), 49152)
                self._next_chunk(cache, req)
                release_kv_cache(req, cache)

                self.assertEqual(len(self._match(cache, req)), 65536)
                self.assertEqual(cache.full_evictable_size(), 65536)
                self.assertEqual(cache.full_protected_size(), 0)
                self.assertEqual(cache.swa_protected_size(), 0)
                cache.sanity_check()
                cache.evict(
                    EvictParams(num_tokens=FULL_CAPACITY, swa_num_tokens=SWA_CAPACITY)
                )
                self._assert_all_free(cache)

    def test_fake_request_keeps_a_shared_normal_prefix_intact(self):
        cache = self._cache()
        normal = self._req(cache, 65536, bootstrap_host="10.0.0.1", rid="normal")
        self._next_chunk(cache, normal)
        maybe_cache_unfinished_req(normal, cache, chunked=True)
        shared = normal.prefix_indices.clone()
        mapping = cache.token_to_kv_pool_allocator.full_to_swa_index_mapping
        shared_mapping = mapping[shared].clone()

        fake = self._req(cache, 131072, rid="fake")
        self.assertEqual(fake.kv.cache_protected_len, 49152)
        self._next_chunk(cache, fake)
        maybe_cache_unfinished_req(fake, cache, chunked=True)
        self.assertEqual(len(fake.prefix_indices), 98304)
        self.assertEqual(fake.kv.cache_protected_len, 49152)
        release_kv_cache(fake, cache)

        torch.testing.assert_close(self._match(cache, fake), shared)
        torch.testing.assert_close(mapping[shared], shared_mapping)
        self.assertEqual(cache.full_protected_size(), 49152)
        release_kv_cache(normal, cache, is_insert=False)
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=FULL_CAPACITY, swa_num_tokens=SWA_CAPACITY))
        self._assert_all_free(cache)

    def test_non_chunk_fake_requests_do_not_publish_kv(self):
        cache = self._cache()
        req = self._req(cache, 8192)
        self._next_chunk(cache, req)
        prefix_before = req.prefix_indices
        maybe_cache_unfinished_req(req, cache)
        maybe_cache_unfinished_req(req, cache, chunked=False)

        self.assertIs(req.prefix_indices, prefix_before)
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertEqual(len(self._match(cache, req)), 0)
        release_kv_cache(req, cache)
        self.assertEqual(len(self._match(cache, req)), 0)
        self._assert_all_free(cache)


if __name__ == "__main__":
    unittest.main()
