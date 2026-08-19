import unittest

from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestPrefetchMaxRetryRoundsParse(unittest.TestCase):
    def _parse(self, cfg):
        cache = HiRadixCache.__new__(HiRadixCache)
        return cache._parse_storage_backend_extra_config(cfg)

    def test_default_is_zero(self):
        result = self._parse(None)
        # last element of the returned tuple is prefetch_max_retry_rounds
        self.assertEqual(result[-1], 0)

    def test_parsed_from_json(self):
        result = self._parse('{"prefetch_max_retry_rounds": 3}')
        self.assertEqual(result[-1], 3)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            self._parse('{"prefetch_max_retry_rounds": -1}')

    def test_bool_rejected(self):
        with self.assertRaises(ValueError):
            self._parse('{"prefetch_max_retry_rounds": true}')


class TestMambaPrefetchMaxRetryRoundsParse(unittest.TestCase):
    def _parse(self, cfg):
        cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
        return cache._parse_storage_backend_extra_config(cfg)

    def test_default_is_zero(self):
        result = self._parse(None)
        self.assertEqual(result[-1], 0)

    def test_parsed_from_json(self):
        result = self._parse('{"prefetch_max_retry_rounds": 3}')
        self.assertEqual(result[-1], 3)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            self._parse('{"prefetch_max_retry_rounds": -1}')

    def test_bool_rejected(self):
        with self.assertRaises(ValueError):
            self._parse('{"prefetch_max_retry_rounds": true}')


if __name__ == "__main__":
    unittest.main()
