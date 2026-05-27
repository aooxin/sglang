import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _f(mask, window):
    return HiCacheController._compute_safe_storage_prefix(
        torch.tensor(mask, dtype=torch.bool), window
    )


class TestComputeSafeStoragePrefix(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_f([], 3), 0)

    def test_all_valid(self):
        self.assertEqual(_f([True] * 10, 3), 10)

    def test_all_invalid(self):
        self.assertEqual(_f([False] * 10, 3), 0)

    def test_tail_window_exact(self):
        mask = [False, False, False, True, True, True]
        self.assertEqual(_f(mask, 3), 6)

    def test_tail_window_insufficient(self):
        mask = [True, True, True, True, False, True]
        self.assertEqual(_f(mask, 3), 4)

    def test_window_1_all_valid(self):
        self.assertEqual(_f([True, True, True], 1), 3)

    def test_window_1_gap_in_middle(self):
        self.assertEqual(_f([True, False, True], 1), 3)

    def test_window_1_starts_invalid(self):
        self.assertEqual(_f([False, True, True], 1), 3)

    def test_window_larger_than_total(self):
        self.assertEqual(_f([True, True, True], 10), 3)

    def test_window_larger_one_invalid(self):
        self.assertEqual(_f([True, False, True], 10), 1)

    def test_realistic_pattern(self):
        mask = [True, True, False, False, True, True, True, True, True, True]
        self.assertEqual(_f(mask, 4), 10)

    def test_realistic_pattern_fail(self):
        mask = [True, True, True, True, True, True, True, True, False, True]
        self.assertEqual(_f(mask, 4), 8)

    def test_single_page_valid(self):
        self.assertEqual(_f([True], 1), 1)

    def test_single_page_invalid(self):
        self.assertEqual(_f([False], 1), 0)

    def test_fast_path_triggered(self):
        mask = [False, True, True, True]
        self.assertEqual(_f(mask, 3), 4)

    def test_fast_path_not_triggered(self):
        mask = [True, True, False, True]
        self.assertEqual(_f(mask, 3), 2)


if __name__ == "__main__":
    unittest.main()
