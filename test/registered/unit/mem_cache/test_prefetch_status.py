import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.base_prefix_cache import PrefetchStatus
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestPrefetchStatus(unittest.TestCase):
    def test_four_members_with_stable_values(self):
        self.assertEqual(int(PrefetchStatus.STARTED), 0)
        self.assertEqual(int(PrefetchStatus.SKIPPED), 1)
        self.assertEqual(int(PrefetchStatus.RATE_LIMITED), 2)
        self.assertEqual(int(PrefetchStatus.HOST_ALLOC_FAILED), 3)


class TestHiRadixPrefetchStatus(unittest.TestCase):
    def _bare_cache(self, *, enable_storage, threshold, rate_limited):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = enable_storage
        cache.prefetch_threshold = threshold
        cache.page_size = 1
        cache.is_eagle = False
        cache.speculative_num_steps = 0
        cache.cache_controller = SimpleNamespace(
            prefetch_rate_limited=lambda: rate_limited
        )
        cache.ongoing_prefetch = {}
        return cache

    def test_skipped_when_below_threshold(self):
        cache = self._bare_cache(enable_storage=True, threshold=256, rate_limited=False)
        status = HiRadixCache.prefetch_from_storage(
            cache, "rid0", MagicMock(), [1, 2, 3], None, None
        )
        self.assertEqual(status, PrefetchStatus.SKIPPED)
        self.assertEqual(cache.ongoing_prefetch, {})

    def test_rate_limited_leaves_no_side_effects(self):
        cache = self._bare_cache(enable_storage=True, threshold=1, rate_limited=True)
        status = HiRadixCache.prefetch_from_storage(
            cache, "rid1", MagicMock(), list(range(64)), None, None
        )
        self.assertEqual(status, PrefetchStatus.RATE_LIMITED)
        self.assertEqual(cache.ongoing_prefetch, {})


class TestHiMambaPrefetchStatus(unittest.TestCase):
    def _bare_cache(self, *, enable_storage, threshold, rate_limited):
        cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
        cache.enable_storage = enable_storage
        cache.prefetch_threshold = threshold
        cache.page_size = 1
        cache.cache_controller = SimpleNamespace(
            prefetch_rate_limited=lambda: rate_limited
        )
        cache.ongoing_prefetch = {}
        return cache

    def test_skipped_below_threshold(self):
        cache = self._bare_cache(
            enable_storage=True, threshold=256, rate_limited=False
        )
        status = HiMambaRadixCache.prefetch_from_storage(
            cache, "rid0", MagicMock(), [1, 2, 3], None, None
        )
        self.assertEqual(status, PrefetchStatus.SKIPPED)

    def test_rate_limited(self):
        cache = self._bare_cache(
            enable_storage=True, threshold=1, rate_limited=True
        )
        status = HiMambaRadixCache.prefetch_from_storage(
            cache, "rid1", MagicMock(), list(range(64)), None, None
        )
        self.assertEqual(status, PrefetchStatus.RATE_LIMITED)
        self.assertEqual(cache.ongoing_prefetch, {})


if __name__ == "__main__":
    unittest.main()
