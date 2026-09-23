import types
import unittest
from unittest import mock

import torch

# Reuse the SWA cache fixture (pools/allocator/req helpers) from the sibling
# test module. The bare module name is the established pattern for sibling test
# imports in this repo (see test_hicache_storage_*_backend.py,
# test_tracing_disaggregation.py); it resolves both under direct execution
# (python3 <file>.py) and under pytest, which prepend the test file's own
# directory to sys.path. register_cuda_ci is a runtime no-op parsed via AST, so
# importing it has no side effects.
from test_simple_swa_radix_cache import _SWACacheTestBase

from sglang.srt import environ as _environ
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.simple_swa_radix_cache import SimpleSWARadixCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-small")


class _SWADynamicAnchorTestBase(_SWACacheTestBase):
    """Fixture with per-request Full-KV-guided SWA branch-window protection on.

    The base setUp empties SGLANG_SIMPLE_SWA_RESERVED_POINTS so no static point
    interferes; here we additionally turn on SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS
    (read at cache construction). The override is entered before super().setUp()
    builds self.cache and torn down via addCleanup after the test.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for SWA cache tests")

        # Reuse the project's EnvField.override() context manager (same one the
        # sibling tests use inline) instead of hand-managing os.environ, so the
        # env var — and EnvField's internal _set_to_none bookkeeping — is
        # restored correctly. The flag is read at cache construction, so enter
        # the override before super().setUp() builds self.cache.
        cm = _environ.envs.SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS.override("1")
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

        super().setUp()


class TestBuildSwaProtectedPoints(CustomTestCase):
    """Unit tests for SimpleSWARadixCache._build_swa_protected_points.

    The method only reads plain attributes (page_size / sliding_window_size /
    swa_reserved_max_point / swa_reserved_points / swa_dynamic_anchors_enabled)
    and calls the module-level floor_align, so we drive it through a lightweight
    SimpleNamespace stub without a GPU allocator (mirrors TestSegmentFreeSlots).
    """

    def _points(
        self,
        branch_seqlen,
        *,
        enabled=True,
        page_size=1,
        sliding_window_size=4,
        swa_reserved_max_point=128 * 1024,
        swa_reserved_points=(),
    ):
        stub = types.SimpleNamespace(
            swa_dynamic_anchors_enabled=enabled,
            page_size=page_size,
            sliding_window_size=sliding_window_size,
            swa_reserved_max_point=swa_reserved_max_point,
            swa_reserved_points=list(swa_reserved_points),
        )
        return SimpleSWARadixCache._build_swa_protected_points(stub, branch_seqlen)

    def test_none_branch_returns_static_only(self):
        self.assertEqual(self._points(None), [])
        self.assertEqual(
            self._points(None, swa_reserved_points=(8, 16)), [8, 16]
        )

    def test_eligible_branch_added(self):
        # 8 > sliding_window (4) and < reserved_max => protected.
        self.assertEqual(self._points(8), [8])
        self.assertEqual(
            self._points(8, swa_reserved_points=(16, 32)), [8, 16, 32]
        )

    def test_branch_at_or_below_window_rejected(self):
        # No window to protect at/below the sliding window.
        self.assertEqual(self._points(4), [])
        self.assertEqual(self._points(3), [])

    def test_branch_at_or_above_reserved_max_rejected(self):
        # At/above the reserved max point: outside the reserve budget.
        self.assertEqual(self._points(128 * 1024), [])
        self.assertEqual(self._points(128 * 1024 + 8), [])

    def test_branch_page_aligned(self):
        # 13 floors to 12 under page_size=4.
        self.assertEqual(self._points(13, page_size=4), [12])

    def test_branch_dedups_with_static(self):
        # A branch point coinciding with a static reserved point adds nothing.
        self.assertEqual(self._points(8, swa_reserved_points=(8, 16)), [8, 16])

    def test_flag_off_returns_static_only(self):
        # Disabled => the branch point is ignored even when eligible.
        self.assertEqual(self._points(8, enabled=False), [])
        self.assertEqual(
            self._points(8, enabled=False, swa_reserved_points=(16,)), [16]
        )

    def test_reserved_max_zero_disables_upper_bound(self):
        # swa_reserved_max_point == 0 (falsy) means no upper bound is enforced.
        self.assertEqual(self._points(1_000_000, swa_reserved_max_point=0), [1_000_000])


class TestSWADynamicAnchors(_SWADynamicAnchorTestBase, CustomTestCase):
    # ===== A. state initialization =====

    def test_flag_on_state_initialized(self):
        cache = self._require_cache()
        self.assertTrue(cache.swa_dynamic_anchors_enabled)
        self.assertEqual(cache.swa_dynamic_anchor_protected_tokens, 0)
        # The global anchor set / union list were removed in the per-request
        # design; only the static reserved points remain.
        self.assertFalse(hasattr(cache, "_swa_dynamic_anchors"))
        self.assertFalse(hasattr(cache, "_swa_effective_points"))
        self.assertFalse(hasattr(cache, "swa_dynamic_anchor_max"))

    # ===== B. evict_swa protects the request's own branch window =====

    def test_evict_swa_protects_anchor_window(self):
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        seq_len = 20
        window = self.sliding_window_size  # 4
        anchor = 12  # window [8, 12) must survive

        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        # match_prefix would have produced this branch point; set it directly.
        req.swa_branching_seqlen = anchor

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        # swa_free_end = floor_align(20 - 4) = 16.
        # Freed: [0, 8) and [12, 16). Kept: [8, 12) (branch window) and [16, 20).
        mapping = allocator.full_to_swa_index_mapping
        freed = torch.cat([kv_indices[0:8], kv_indices[12:16]])
        kept = torch.cat([kv_indices[8:12], kv_indices[16:20]])
        self.assertTrue(torch.all(mapping[freed] == 0))
        self.assertTrue(torch.all(mapping[kept] > 0))
        # Observability counter attributed the withheld window.
        self.assertEqual(cache.swa_dynamic_anchor_protected_tokens, window)

    def test_evict_swa_protects_static_and_branch_windows(self):
        # Static reserved point 8 AND the request's own branch point 12: both
        # windows must survive, and the counter attributes only the *extra*
        # branch window (effective minus static-only protection).
        with _environ.envs.SGLANG_SIMPLE_SWA_RESERVED_POINTS.override("8"):
            cache, pool, allocator = self._build_cache(self.sliding_window_size)

        self.assertTrue(cache.swa_dynamic_anchors_enabled)
        self.assertEqual(list(cache.swa_reserved_points), [8])

        seq_len = 20
        window = self.sliding_window_size  # 4
        req_pool_idx = self._alloc_req_slot(pool)
        token_ids = list(range(seq_len))
        kv_indices = allocator.alloc(seq_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        req.swa_branching_seqlen = 12

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        # Protected windows: reserved [4, 8) and branch [8, 12).
        # swa_free_end = 16. Freed: [0, 4) and [12, 16). Kept: [4, 12), [16, 20).
        mapping = allocator.full_to_swa_index_mapping
        freed = torch.cat([kv_indices[0:4], kv_indices[12:16]])
        kept = torch.cat([kv_indices[4:12], kv_indices[16:20]])
        self.assertTrue(torch.all(mapping[freed] == 0))
        self.assertTrue(torch.all(mapping[kept] > 0))
        # Only the branch window is "extra" over static-only protection.
        self.assertEqual(cache.swa_dynamic_anchor_protected_tokens, window)

    def test_evict_swa_no_branch_no_anchor(self):
        pool = self._require_pool()
        allocator = self._require_allocator()
        cache = self._require_cache()

        seq_len = 20
        req_pool_idx = self._alloc_req_slot()
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        # No branch point => no extra window protected even with the flag on.
        self.assertIsNone(req.swa_branching_seqlen)

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        # Standard behavior: everything outside the window is freed.
        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices[:16]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[16:]] > 0))
        self.assertEqual(cache.swa_dynamic_anchor_protected_tokens, 0)

    # ===== C. match_prefix becomes SWA-safe at the branch point =====

    def test_match_prefix_swa_safe_at_anchor(self):
        pool = self._require_pool()
        cache = self._require_cache()

        seq_len = 20
        anchor = 12

        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        req_pool_idx = self._alloc_req_slot()
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        req.swa_branching_seqlen = anchor
        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        # At the branch length the tail window [8, 12) is alive => SWA-safe.
        result_anchor = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:anchor]))
        )
        self.assertEqual(len(result_anchor.device_indices), anchor)

        # A shorter length whose tail window falls in the freed region is not
        # SWA-safe and gets trimmed away.
        result_short = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:8]))
        )
        self.assertLess(len(result_short.device_indices), 8)

    def test_match_prefix_records_anchor_metrics(self):
        # match_prefix must count the branch point and observe its position into
        # the RadixCacheMetricsCollector (feeds sglang:swa_dynamic_anchor_created_total
        # and sglang:swa_dynamic_anchor_position). A SWA-safe match records nothing.
        pool = self._require_pool()
        cache = self._require_cache()

        seq_len = 20
        anchor = 12
        token_ids = list(range(seq_len))
        kv_indices = self._alloc_kv_indices(seq_len)
        cache.insert(InsertParams(key=RadixKey(token_ids), value=kv_indices))

        req_pool_idx = self._alloc_req_slot()
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        req.swa_branching_seqlen = anchor
        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        cache.metrics_collector = mock.MagicMock()

        # A short prefix whose tail window was freed => match_prefix detects a
        # full>swa branch point and records both metrics with the branch position.
        result_short = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:8]))
        )
        branch = result_short.swa_branching_seqlen
        self.assertIsNotNone(branch)
        cache.metrics_collector.increment_swa_dynamic_anchor_created.assert_called_once_with()
        cache.metrics_collector.observe_swa_dynamic_anchor_position.assert_called_once_with(
            branch
        )

        # A SWA-safe match (tail window alive, no truncation) records nothing.
        cache.metrics_collector.reset_mock()
        result_safe = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:anchor]))
        )
        self.assertIsNone(result_safe.swa_branching_seqlen)
        cache.metrics_collector.increment_swa_dynamic_anchor_created.assert_not_called()
        cache.metrics_collector.observe_swa_dynamic_anchor_position.assert_not_called()

    # ===== D. flag-off regression =====

    def test_flag_off_ignores_branch_point(self):
        with _environ.envs.SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS.override("0"):
            cache, pool, allocator = self._build_cache(self.sliding_window_size)

        self.assertFalse(cache.swa_dynamic_anchors_enabled)

        seq_len = 20
        req_pool_idx = self._alloc_req_slot(pool)
        token_ids = list(range(seq_len))
        kv_indices = allocator.alloc(seq_len)
        if kv_indices is None:
            self.fail("Failed to allocate KV indices")
        pool.write((req_pool_idx, slice(0, seq_len)), kv_indices)
        req = self._make_req(token_ids, req_pool_idx)
        # Even with a branch point set, the disabled flag ignores it.
        req.swa_branching_seqlen = 12

        cache.evict_swa(req, prelen=seq_len, attention_chunk_size=0)

        mapping = allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[kv_indices[:16]] == 0))
        self.assertTrue(torch.all(mapping[kv_indices[16:]] > 0))
        self.assertEqual(cache.swa_dynamic_anchor_protected_tokens, 0)


if __name__ == "__main__":
    unittest.main()
