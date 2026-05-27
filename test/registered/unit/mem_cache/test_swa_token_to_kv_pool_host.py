import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import SWATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


class StubSWATokenToKVPoolHost:
    """Stub that exposes only the bound mapping method without
    requiring the full host pool to be constructed (which needs a real
    device pool / GPU allocation)."""

    def __init__(self, page_size: int, mapping_size: int):
        self.page_size = page_size
        self.full_to_swa_index_mapping = torch.zeros(mapping_size, dtype=torch.int64)

    mapping = SWATokenToKVPoolHost.mapping_full_and_swa_indices


def _make_host(page_size: int, mapping_size: int = 1024):
    return StubSWATokenToKVPoolHost(page_size, mapping_size)


def _assert_mapping(test, host, full_indices, expected_swa):
    actual = host.full_to_swa_index_mapping[full_indices].tolist()
    test.assertEqual(actual, expected_swa)


def _run_and_check(
    test,
    ps,
    full,
    swa,
    mask,
    expected_mask,
    expected_surplus,
    expected_bindings,
):
    host = _make_host(ps)
    updated, surplus = host.mapping(
        torch.tensor(full, dtype=torch.int64),
        torch.tensor(swa, dtype=torch.int64),
        torch.tensor(mask, dtype=torch.bool),
    )
    test.assertEqual(updated.tolist(), list(expected_mask), "mask mismatch")
    if expected_surplus is None:
        test.assertIsNone(surplus, "expected None surplus")
    else:
        test.assertIsNotNone(surplus, "expected non-None surplus")
        test.assertEqual(surplus.tolist(), expected_surplus, "surplus mismatch")
    for full_idx, exp_swa in expected_bindings:
        _assert_mapping(test, host, torch.tensor(full_idx), exp_swa)


class TestMappingFullAndSwaIndices(unittest.TestCase):
    def _check(self, **kwargs):
        _run_and_check(self, **kwargs)

    def test_exact_match(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[False, False, True, True, True],
            expected_mask=[False, False, True, True, True],
            expected_surplus=None,
            expected_bindings=[
                ([30, 31], [100, 101]),
                ([40, 41], [200, 201]),
                ([50, 51], [300, 301]),
                ([10, 11], [0, 0]),
                ([20, 21], [0, 0]),
            ],
        )

    def test_surplus(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[False, False, True],
            expected_mask=[False, False, True],
            expected_surplus=[100, 101, 200, 201],
            expected_bindings=[
                ([30, 31], [300, 301]),
            ],
        )

    def test_insufficient_trims_front(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
            swa=[100, 101, 200, 201],
            mask=[False, False, True, True, True],
            expected_mask=[False, False, False, True, True],
            expected_surplus=None,
            expected_bindings=[
                ([40, 41], [100, 101]),
                ([50, 51], [200, 201]),
                ([30, 31], [0, 0]),
            ],
        )

    def test_no_swa_needed_asserts(self):
        host = _make_host(2)
        with self.assertRaises(AssertionError):
            host.mapping(
                torch.tensor([10, 11, 20, 21]),
                torch.tensor([100, 101, 200, 201]),
                torch.tensor([False, False], dtype=torch.bool),
            )

    def test_all_pages_need_swa(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[True, True, True],
            expected_mask=[True, True, True],
            expected_surplus=None,
            expected_bindings=[
                ([10, 11], [100, 101]),
                ([20, 21], [200, 201]),
                ([30, 31], [300, 301]),
            ],
        )

    def test_zero_swa_available_asserts(self):
        host = _make_host(2)
        with self.assertRaises(AssertionError):
            host.mapping(
                torch.tensor([10, 11, 20, 21]),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([True, True], dtype=torch.bool),
            )

    def test_page_size_4(self):
        self._check(
            ps=4,
            full=[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
            swa=[100, 101, 102, 103, 200, 201, 202, 203],
            mask=[False, True, True],
            expected_mask=[False, True, True],
            expected_surplus=None,
            expected_bindings=[
                ([20, 21, 22, 23], [100, 101, 102, 103]),
                ([30, 31, 32, 33], [200, 201, 202, 203]),
            ],
        )

    def test_single_page(self):
        self._check(
            ps=2,
            full=[10, 11],
            swa=[100, 101],
            mask=[True],
            expected_mask=[True],
            expected_surplus=None,
            expected_bindings=[
                ([10, 11], [100, 101]),
            ],
        )

    def test_non_contiguous_mask_all_true_bound(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31],
            swa=[100, 101, 200, 201],
            mask=[True, False, True],
            expected_mask=[True, False, True],
            expected_surplus=None,
            expected_bindings=[
                ([10, 11], [100, 101]),
                ([30, 31], [200, 201]),
                ([20, 21], [0, 0]),
            ],
        )

    def test_non_contiguous_mask_insufficient(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
            swa=[100, 101],
            mask=[True, False, True, False, True],
            expected_mask=[False, False, False, False, True],
            expected_surplus=None,
            expected_bindings=[
                ([50, 51], [100, 101]),
                ([10, 11], [0, 0]),
                ([30, 31], [0, 0]),
            ],
        )

    def test_page_size_1(self):
        self._check(
            ps=1,
            full=[10, 20, 30],
            swa=[100, 200],
            mask=[True, True, True],
            expected_mask=[False, True, True],
            expected_surplus=None,
            expected_bindings=[
                ([20], [100]),
                ([30], [200]),
                ([10], [0]),
            ],
        )

    def test_page_size_1_surplus(self):
        self._check(
            ps=1,
            full=[10, 20],
            swa=[100, 200, 300],
            mask=[False, True],
            expected_mask=[False, True],
            expected_surplus=[100, 200],
            expected_bindings=[
                ([20], [300]),
                ([10], [0]),
            ],
        )

    def test_page_size_8(self):
        full = list(range(100, 124))
        swa = list(range(200, 216))
        self._check(
            ps=8,
            full=full,
            swa=swa,
            mask=[True, True, True],
            expected_mask=[False, True, True],
            expected_surplus=None,
            expected_bindings=[
                (list(range(108, 116)), list(range(200, 208))),
                (list(range(116, 124)), list(range(208, 216))),
                (list(range(100, 108)), [0] * 8),
            ],
        )

    def test_only_last_page_true(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41],
            swa=[100, 101],
            mask=[False, False, False, True],
            expected_mask=[False, False, False, True],
            expected_surplus=None,
            expected_bindings=[
                ([40, 41], [100, 101]),
                ([10, 11], [0, 0]),
            ],
        )

    def test_only_first_page_true_with_surplus(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[True, False, False],
            expected_mask=[True, False, False],
            expected_surplus=[100, 101, 200, 201],
            expected_bindings=[
                ([10, 11], [300, 301]),
                ([20, 21], [0, 0]),
                ([30, 31], [0, 0]),
            ],
        )

    def test_alternating_mask_exact_fit(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[True, False, True, False, True, False],
            expected_mask=[True, False, True, False, True, False],
            expected_surplus=None,
            expected_bindings=[
                ([10, 11], [100, 101]),
                ([30, 31], [200, 201]),
                ([50, 51], [300, 301]),
                ([20, 21], [0, 0]),
                ([40, 41], [0, 0]),
                ([60, 61], [0, 0]),
            ],
        )

    def test_alternating_mask_with_surplus(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41],
            swa=[100, 101, 200, 201, 300, 301],
            mask=[True, False, True, False],
            expected_mask=[True, False, True, False],
            expected_surplus=[100, 101],
            expected_bindings=[
                ([10, 11], [200, 201]),
                ([30, 31], [300, 301]),
            ],
        )

    def test_many_pages_insufficient(self):
        ps = 2
        num_pages = 20
        full = []
        for i in range(num_pages):
            full.extend([i * 10, i * 10 + 1])
        swa = list(range(500, 506))
        mask = [True] * num_pages

        expected_mask = [False] * 17 + [True] * 3
        self._check(
            ps=ps,
            full=full,
            swa=swa,
            mask=mask,
            expected_mask=expected_mask,
            expected_surplus=None,
            expected_bindings=[
                ([170, 171], [500, 501]),
                ([180, 181], [502, 503]),
                ([190, 191], [504, 505]),
                ([0, 1], [0, 0]),
            ],
        )

    def test_single_true_at_start_insufficient_trims(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41],
            swa=[100, 101],
            mask=[True, True, True, True],
            expected_mask=[False, False, False, True],
            expected_surplus=None,
            expected_bindings=[
                ([40, 41], [100, 101]),
                ([10, 11], [0, 0]),
                ([20, 21], [0, 0]),
                ([30, 31], [0, 0]),
            ],
        )

    def test_surplus_is_exact_head_of_swa(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21],
            swa=[100, 101, 200, 201, 300, 301, 400, 401],
            mask=[False, True],
            expected_mask=[False, True],
            expected_surplus=[100, 101, 200, 201, 300, 301],
            expected_bindings=[
                ([20, 21], [400, 401]),
                ([10, 11], [0, 0]),
            ],
        )

    def test_large_page_size_16_surplus(self):
        ps = 16
        full = list(range(0, 48))
        swa = list(range(1000, 1032))
        self._check(
            ps=ps,
            full=full,
            swa=swa,
            mask=[False, True, False],
            expected_mask=[False, True, False],
            expected_surplus=list(range(1000, 1016)),
            expected_bindings=[
                (list(range(16, 32)), list(range(1016, 1032))),
                (list(range(0, 16)), [0] * 16),
                (list(range(32, 48)), [0] * 16),
            ],
        )

    def test_non_contiguous_trim_keeps_tail_true_pages(self):
        self._check(
            ps=2,
            full=[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
            swa=[100, 101, 200, 201],
            mask=[True, True, False, True, False, True],
            expected_mask=[False, False, False, True, False, True],
            expected_surplus=None,
            expected_bindings=[
                ([40, 41], [100, 101]),
                ([60, 61], [200, 201]),
                ([10, 11], [0, 0]),
                ([20, 21], [0, 0]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
