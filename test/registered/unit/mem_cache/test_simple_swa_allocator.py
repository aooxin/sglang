import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import alloc_for_extend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.simple_swa_radix_cache import SimpleSWARadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")


class TestSimpleSWAAllocator(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for SWA allocator tests")

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.req_size = 4
        self.max_context_len = 64
        self.kv_size = 128
        self.kv_size_swa = 64
        self.sliding_window_size = 4
        self.head_num = 8
        self.head_dim = 128
        self.num_layers = 16
        self.global_interval = 4

    def _attention_layers(self):
        full_attention_layer_ids = [
            i for i in range(0, self.num_layers, self.global_interval)
        ]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(self.num_layers) if i not in full_attention_layer_ids_set
        ]
        return full_attention_layer_ids, swa_attention_layer_ids

    def _build_cache(
        self,
        page_size: int = 1,
        kv_size: int | None = None,
        kv_size_swa: int | None = None,
        max_context_len: int | None = None,
    ):
        kv_size = kv_size or self.kv_size
        kv_size_swa = kv_size_swa or self.kv_size_swa
        max_context_len = max_context_len or self.max_context_len

        full_attention_layer_ids, swa_attention_layer_ids = self._attention_layers()
        req_to_token_pool = ReqToTokenPool(
            size=self.req_size,
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
                sliding_window_size=self.sliding_window_size,
            ),
        )
        return cache, req_to_token_pool, allocator

    def _make_req(self, rid: str, token_ids: list[int]) -> Req:
        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=token_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = token_ids
        req.output_ids = []
        return req

    def test_swa_available_size_after_alloc_and_free(self):
        cache, pool, allocator = self._build_cache()

        full_initial = allocator.full_available_size()
        swa_initial = allocator.swa_available_size()

        num_tokens = 10
        kv_indices = allocator.alloc(num_tokens)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")

        self.assertEqual(allocator.full_available_size(), full_initial - num_tokens)
        self.assertEqual(allocator.swa_available_size(), swa_initial - num_tokens)

        req_pool_idx_list = pool.alloc([Req(rid="placeholder", origin_input_text="", origin_input_ids=[], sampling_params=SamplingParams(max_new_tokens=1, temperature=0.0))])
        if not req_pool_idx_list:
            self.fail("Failed to allocate request slot")
        req_pool_idx = req_pool_idx_list[0]
        pool.write((req_pool_idx, slice(0, num_tokens)), kv_indices)
        req = self._make_req("allocator_resource", list(range(num_tokens)))
        req.req_pool_idx = req_pool_idx
        cache.evict_swa(req, prelen=num_tokens, attention_chunk_size=0)
        freed_count = max(0, num_tokens - self.sliding_window_size)

        self.assertEqual(allocator.full_available_size(), full_initial - num_tokens)
        self.assertEqual(
            allocator.swa_available_size(), swa_initial - num_tokens + freed_count
        )

        allocator.free(kv_indices)

        self.assertEqual(allocator.full_available_size(), full_initial)
        self.assertEqual(allocator.swa_available_size(), swa_initial)

    def test_free_swa_is_idempotent_but_free_rejects_double_free(self):
        _, _, allocator = self._build_cache()

        kv_indices = allocator.alloc(8)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")

        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices] > 0))

        allocator.free_swa(kv_indices[:4])
        self.assertTrue(torch.all(mapping[kv_indices[:4]] == 0))

        allocator.free_swa(kv_indices[:4])
        self.assertTrue(torch.all(mapping[kv_indices[:4]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[4:]] > 0))

        allocator.free(kv_indices)

        with self.assertRaises(AssertionError):
            allocator.free(kv_indices)

    def test_alloc_extend_paged_no_false_oom(self):
        page_size = 4
        bs = 4
        prefix_len_per_req = page_size
        seq_len_per_req = 2 * page_size
        total_pages = bs * 2
        kv_size = total_pages * page_size

        _, _, allocator = self._build_cache(
            page_size=page_size,
            kv_size=kv_size,
            kv_size_swa=kv_size,
            max_context_len=page_size * 4,
        )

        prefix_tokens = bs * prefix_len_per_req
        prefix_indices = allocator.alloc(prefix_tokens)
        self.assertIsNotNone(prefix_indices, "prefix pre-allocation must succeed")

        extend_num_tokens = bs * page_size
        precise_needed = bs * page_size
        old_over_estimate = extend_num_tokens + bs * page_size

        self.assertEqual(allocator.full_available_size(), precise_needed)
        self.assertEqual(allocator.swa_available_size(), precise_needed)
        self.assertGreater(old_over_estimate, precise_needed)

        prefix_lens_cpu = torch.tensor([prefix_len_per_req] * bs, dtype=torch.int64)
        seq_lens_cpu = torch.tensor([seq_len_per_req] * bs, dtype=torch.int64)
        prefix_lens = prefix_lens_cpu.to(self.device)
        seq_lens = seq_lens_cpu.to(self.device)
        last_loc = torch.full((bs,), -1, dtype=torch.int64, device=self.device)

        result = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
        )

        self.assertIsNotNone(result, "alloc_extend should not fail with exact capacity")
        self.assertEqual(result.numel(), extend_num_tokens)

    def test_alloc_for_extend_prefix_hit_without_req_pool_idx_keeps_shared_prefix(self):
        cache, pool, allocator = self._build_cache()

        tree_token_ids = list(range(8))
        tree_kv = allocator.alloc(len(tree_token_ids))
        if tree_kv is None:
            self.fail("Failed to allocate tree KV indices")
        cache.insert(InsertParams(key=RadixKey(tree_token_ids), value=tree_kv))

        req = self._make_req("prefix_hit_without_row", tree_token_ids + [100, 101])
        req.req_pool_idx = None
        req.prefix_indices = tree_kv.to(dtype=torch.int64, copy=True)
        req.cache_protected_len = len(tree_token_ids)
        req.swa_evicted_seqlen = 0

        batch = SimpleNamespace(
            tree_cache=cache,
            reqs=[req],
            prefix_lens=[len(tree_token_ids)],
            model_config=SimpleNamespace(attention_chunk_size=self.sliding_window_size),
            device=self.device,
            extend_lens=[2],
            req_to_token_pool=pool,
            extend_num_tokens=2,
            seq_lens=torch.tensor([10], dtype=torch.int64, device=self.device),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int64),
            maybe_evict_swa=lambda: None,
        )

        out_cache_loc, _, req_pool_indices = alloc_for_extend(batch)

        mapping = allocator.full_to_swa_index_mapping
        self.assertEqual(req.swa_evicted_seqlen, 0)
        self.assertTrue(torch.all(mapping[tree_kv] > 0))
        self.assertEqual(len(req_pool_indices), 1)
        self.assertEqual(out_cache_loc.numel(), 2)
        self.assertTrue(
            torch.equal(
                pool.req_to_token[req_pool_indices[0], : len(tree_token_ids)], tree_kv
            )
        )


if __name__ == "__main__":
    unittest.main()
