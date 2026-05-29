"""Unit tests for compute_swa_safe_len().

Tests the core SWA (Sliding Window Attention) safety check algorithm extracted
from SimpleSWARadixCache._compute_safe_prefix_len. The function finds the
longest prefix whose tail window (of size ``sliding_window_size``) contains
only valid (> 0) SWA indices.

Coverage targets:
    - Empty input
    - All valid indices (fast path)
    - All invalid indices
    - Single element (valid / invalid)
    - Window == 1
    - Window > total length
    - Window == total length
    - Invalid at various positions (head, tail, middle, boundary)
    - Mixed valid/invalid patterns
    - Contiguous invalid blocks
    - Large tensors
"""

import unittest

import torch

from sglang.srt.mem_cache.simple_swa_radix_cache import compute_swa_safe_len
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


class TestComputeSwaSafeLen(CustomTestCase):
    """Tests for compute_swa_safe_len(swa_indices, sliding_window_size)."""

    # ------------------------------------------------------------------ #
    #  Empty input
    # ------------------------------------------------------------------ #
    def test_empty_tensor(self):
        """Empty swa_indices → 0."""
        swa = torch.empty(0, dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), 0)

    # ------------------------------------------------------------------ #
    #  All valid (fast path)
    # ------------------------------------------------------------------ #
    def test_all_valid_small(self):
        """All positive → return full length (fast path)."""
        swa = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 5)

    def test_all_valid_window_equals_len(self):
        """Window == len, all valid → full length."""
        swa = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 3)

    def test_all_valid_window_larger_than_len(self):
        """Window > len, all valid → full length."""
        swa = torch.tensor([1, 2], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=10), 2)

    # ------------------------------------------------------------------ #
    #  All invalid
    # ------------------------------------------------------------------ #
    def test_all_zero(self):
        """All zeros → 0."""
        swa = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 0)

    def test_all_negative_rejected(self):
        """Negative indices violate the >= 0 precondition."""
        swa = torch.tensor([-1, -2, -3], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            compute_swa_safe_len(swa, sliding_window_size=2)

    # ------------------------------------------------------------------ #
    #  Single element
    # ------------------------------------------------------------------ #
    def test_single_valid(self):
        """Single valid element → 1."""
        swa = torch.tensor([5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 1)

    def test_single_invalid(self):
        """Single invalid (0) element → 0."""
        swa = torch.tensor([0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 0)

    def test_single_valid_large_window(self):
        """Single valid element, window > 1 → 1."""
        swa = torch.tensor([5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=100), 1)

    # ------------------------------------------------------------------ #
    #  Window size = 1
    # ------------------------------------------------------------------ #
    def test_window_1_all_valid(self):
        """Window=1, all valid → full length."""
        swa = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 4)

    def test_window_1_last_invalid(self):
        """Window=1, last element invalid → length - 1.

        For L=4, window=[3,4), index 3 is 0 → invalid.
        For L=3, window=[2,3), index 2 is 3 → valid.
        So safe len = 3.
        """
        swa = torch.tensor([1, 2, 3, 0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 3)

    def test_window_1_first_invalid(self):
        """Window=1, first element invalid.

        L=1: w=[0,1) → [0] bad
        L=2: w=[1,2) → [1] ok
        L=3: w=[2,3) → [2] ok
        L=4: w=[3,4) → [3] ok
        Safe = 4 (largest L whose window is all valid).
        """
        swa = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 4)

    def test_window_1_middle_invalid(self):
        """Window=1, middle invalid.

        [1, 0, 3, 4]:
        L=1: w=[0,1) → [1] ok
        L=2: w=[1,2) → [0] bad
        L=3: w=[2,3) → [3] ok
        L=4: w=[3,4) → [4] ok
        Safe = 4 (largest L whose window is all valid).
        """
        swa = torch.tensor([1, 0, 3, 4], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 4)

    # ------------------------------------------------------------------ #
    #  Invalid at tail (within window)
    # ------------------------------------------------------------------ #
    def test_invalid_at_tail_within_window(self):
        """Invalid at position 4 (last), window=3.

        Indices: [1, 2, 3, 4, 0]  (len=5, window=3)
        L=5: window=[2,5) → indices [3,4,0] → bad (0 at idx4)
        L=4: window=[1,4) → indices [2,3,4] → all > 0 → safe!
        Safe = 4.
        """
        swa = torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 4)

    # ------------------------------------------------------------------ #
    #  Invalid at head (outside window for most candidates)
    # ------------------------------------------------------------------ #
    def test_invalid_at_head_outside_window(self):
        """Invalid at position 0, window=3, len=6.

        [0, 2, 3, 4, 5, 6]
        L=1: window=[0,1) → idx0=0 → bad
        L=2: window=[0,2) → idx0=0 → bad
        L=3: window=[0,3) → idx0=0 → bad
        L=4: window=[1,4) → [2,3,4] → ok!
        L=5: window=[2,5) → [3,4,5] → ok!
        L=6: window=[3,6) → [4,5,6] → ok!
        Safe = 6.
        """
        swa = torch.tensor([0, 2, 3, 4, 5, 6], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 6)

    def test_invalid_at_head_within_window(self):
        """Invalid at position 0, window=4, len=3.

        [0, 2, 3]
        Window >= len, so for any L, window includes idx 0.
        L=1: window=[0,1) → idx0=0 → bad
        L=2: window=[0,2) → idx0=0 → bad
        L=3: window=[0,3) → idx0=0 → bad
        Safe = 0.
        """
        swa = torch.tensor([0, 2, 3], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), 0)

    # ------------------------------------------------------------------ #
    #  Invalid exactly at window boundary
    # ------------------------------------------------------------------ #
    def test_invalid_at_window_boundary(self):
        """Invalid at position that is exactly at the window left edge for max L.

        [1, 0, 3, 4, 5]  window=3, len=5
        L=5: window=[2,5) → [3,4,5] → ok!
        L=4: window=[1,4) → [0,3,4] → bad (idx1=0)
        L=3: window=[0,3) → [1,0,3] → bad
        L=2: window=[0,2) → [1,0] → bad
        L=1: window=[0,1) → [1] → ok
        Largest valid L with window all ok: L=1 then skip L=2,3,4 (bad), L=5 ok.
        Safe = 5.
        """
        swa = torch.tensor([1, 0, 3, 4, 5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 5)

    def test_invalid_at_window_boundary_exact(self):
        """Invalid at the exact boundary position.

        [1, 2, 0, 4, 5]  window=2, len=5
        L=5: window=[3,5) → [4,5] → ok
        L=4: window=[2,4) → [0,4] → bad
        L=3: window=[1,3) → [2,0] → bad
        L=2: window=[0,2) → [1,2] → ok
        L=1: window=[0,1) → [1] → ok
        Safe = 5 (L=5 is valid).
        """
        swa = torch.tensor([1, 2, 0, 4, 5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 5)

    # ------------------------------------------------------------------ #
    #  Contiguous invalid block in the middle
    # ------------------------------------------------------------------ #
    def test_contiguous_invalid_block(self):
        """Block of zeros in the middle.

        [1, 2, 0, 0, 5, 6]  window=2, len=6
        L=1: w=[0,1) → [1] ok
        L=2: w=[0,2) → [1,2] ok
        L=3: w=[1,3) → [2,0] bad
        L=4: w=[2,4) → [0,0] bad
        L=5: w=[3,5) → [0,5] bad
        L=6: w=[4,6) → [5,6] ok
        Safe = 6.
        """
        swa = torch.tensor([1, 2, 0, 0, 5, 6], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 6)

    def test_contiguous_invalid_block_at_tail(self):
        """Block of zeros at the tail.

        [1, 2, 3, 0, 0]  window=2, len=5
        L=5: w=[3,5) → [0,0] bad
        L=4: w=[2,4) → [3,0] bad
        L=3: w=[1,3) → [2,3] ok
        Safe = 3.
        """
        swa = torch.tensor([1, 2, 3, 0, 0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 3)

    def test_contiguous_invalid_block_larger_than_window(self):
        """Invalid block larger than window.

        [1, 0, 0, 0, 5]  window=2, len=5
        L=5: w=[3,5) → [0,5] bad
        L=4: w=[2,4) → [0,0] bad
        L=3: w=[1,3) → [0,0] bad
        L=2: w=[0,2) → [1,0] bad
        L=1: w=[0,1) → [1] ok
        Safe = 1.
        """
        swa = torch.tensor([1, 0, 0, 0, 5], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 1)

    # ------------------------------------------------------------------ #
    #  Alternating valid/invalid
    # ------------------------------------------------------------------ #
    def test_alternating_window_1(self):
        """Alternating [1, 0, 1, 0, 1], window=1.

        L=1: w=[0,1) → [1] ok
        L=2: w=[1,2) → [0] bad
        L=3: w=[2,3) → [1] ok
        L=4: w=[3,4) → [0] bad
        L=5: w=[4,5) → [1] ok
        Safe = 5 (largest L whose window is all valid).
        """
        swa = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 5)

    def test_alternating_window_2(self):
        """Alternating [1, 0, 1, 0, 1], window=2.

        L=1: w=[0,1) → [1] ok
        L=2: w=[0,2) → [1,0] bad
        L=3: w=[1,3) → [0,1] bad
        L=4: w=[2,4) → [1,0] bad
        L=5: w=[3,5) → [0,1] bad
        Safe = 1.
        """
        swa = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 1)

    # ------------------------------------------------------------------ #
    #  Multiple gaps
    # ------------------------------------------------------------------ #
    def test_two_gaps(self):
        """Two separate gaps.

        [1, 0, 3, 0, 5, 6]  window=2, len=6
        L=1: [1] ok
        L=2: [1,0] bad
        L=3: [0,3] bad
        L=4: [3,0] bad
        L=5: [0,5] bad
        L=6: [5,6] ok
        Safe = 6.
        """
        swa = torch.tensor([1, 0, 3, 0, 5, 6], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=2), 6)

    def test_two_gaps_window_3(self):
        """Two gaps, larger window.

        [1, 0, 3, 0, 5, 6]  window=3
        L=6: w=[3,6) → [0,5,6] bad
        L=5: w=[2,5) → [3,0,5] bad
        L=4: w=[1,4) → [0,3,0] bad
        L=3: w=[0,3) → [1,0,3] bad
        L=2: w=[0,2) → [1,0] bad
        L=1: w=[0,1) → [1] ok
        Safe = 1.
        """
        swa = torch.tensor([1, 0, 3, 0, 5, 6], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 1)

    # ------------------------------------------------------------------ #
    #  Valid tail after gap
    # ------------------------------------------------------------------ #
    def test_valid_tail_after_gap(self):
        """Gap followed by enough valid tokens to fill the window.

        [0, 1, 2, 3]  window=3
        L=4: w=[1,4) → [1,2,3] ok!
        Safe = 4.
        """
        swa = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 4)

    def test_gap_not_enough_valid_tail(self):
        """Gap followed by fewer valid tokens than window.

        [0, 1, 2]  window=3
        L=3: w=[0,3) → [0,1,2] bad
        L=2: w=[0,2) → [0,1] bad
        L=1: w=[0,1) → [0] bad
        Safe = 0.
        """
        swa = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 0)

    # ------------------------------------------------------------------ #
    #  Edge: window = total length
    # ------------------------------------------------------------------ #
    def test_window_equals_len_all_valid(self):
        swa = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), 4)

    def test_window_equals_len_one_invalid(self):
        """Window == len, one invalid at start → 0.

        L=4: w=[0,4) → [0,2,3,4] bad
        L=3: w=[0,3) → [0,2,3] bad
        L=2: w=[0,2) → [0,2] bad
        L=1: w=[0,1) → [0] bad
        Safe = 0.
        """
        swa = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), 0)

    def test_window_equals_len_one_invalid_at_end(self):
        """Window == len, one invalid at end → len-1 (if window allows).

        [1, 2, 3, 0]  window=4
        L=4: w=[0,4) → [1,2,3,0] bad
        L=3: w=[0,3) → [1,2,3] ok
        Safe = 3.
        """
        swa = torch.tensor([1, 2, 3, 0], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), 3)

    # ------------------------------------------------------------------ #
    #  Negative values rejected by precondition
    # ------------------------------------------------------------------ #
    def test_negative_values_rejected(self):
        """Negative indices violate the >= 0 precondition."""
        swa = torch.tensor([1, -1, 3, 4], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            compute_swa_safe_len(swa, sliding_window_size=2)

    # ------------------------------------------------------------------ #
    #  Large tensor (performance sanity check)
    # ------------------------------------------------------------------ #
    def test_large_all_valid(self):
        """Large tensor, all valid → full length (fast path)."""
        n = 10000
        swa = torch.arange(1, n + 1, dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=128), n)

    def test_large_one_invalid_near_end(self):
        """Large tensor with one invalid near end.

        Position n-2 is 0, window=4.
        L=n: w=[n-4, n) → includes n-2 which is 0 → bad
        L=n-1: w=[n-5, n-1) → includes n-2 which is 0 → bad
        L=n-2: w=[n-6, n-2) → all valid → ok
        Safe = n-2.
        """
        n = 10000
        swa = torch.arange(1, n + 1, dtype=torch.int64)
        swa[n - 2] = 0  # invalidate near-end position
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=4), n - 2)

    def test_large_invalid_at_start(self):
        """Large tensor with invalid at position 0, window < len.

        Window = 128. For L > 128, position 0 is outside the window → safe.
        Safe = n.
        """
        n = 10000
        swa = torch.arange(1, n + 1, dtype=torch.int64)
        swa[0] = 0
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=128), n)

    # ------------------------------------------------------------------ #
    #  Exact boundary: invalid at position (total_len - window)
    # ------------------------------------------------------------------ #
    def test_invalid_at_exact_window_left_edge_for_max_L(self):
        """Invalid exactly at position (len - window).

        len=8, window=3.
        Position 5 (= 8 - 3) is invalid.
        [1, 2, 3, 4, 5, 0, 7, 8]
        L=8: w=[5,8) → [0,7,8] bad
        L=7: w=[4,7) → [5,0,7] bad
        L=6: w=[3,6) → [4,5,0] bad
        L=5: w=[2,5) → [3,4,5] ok
        Safe = 5.
        """
        swa = torch.tensor([1, 2, 3, 4, 5, 0, 7, 8], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 5)

    # ------------------------------------------------------------------ #
    #  dtype variations
    # ------------------------------------------------------------------ #
    def test_int32_dtype(self):
        """Should work with int32 too."""
        swa = torch.tensor([1, 2, 3, 0, 5], dtype=torch.int32)
        # L=5: w=[2,5) → [3,0,5] bad
        # L=4: w=[1,4) → [2,3,0] bad
        # L=3: w=[0,3) → [1,2,3] ok
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=3), 3)

    # ------------------------------------------------------------------ #
    #  Window = 1 special cases
    # ------------------------------------------------------------------ #
    def test_window_1_valid_invalid_valid(self):
        """Window=1: [1, 0, 3].
        L=1: w=[0,1) → [1] ok
        L=2: w=[1,2) → [0] bad
        L=3: w=[2,3) → [3] ok
        Safe = 3.
        """
        swa = torch.tensor([1, 0, 3], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 3)

    def test_window_1_all_zeros_then_valid(self):
        """Window=1: [0, 0, 1].
        L=1: w=[0,1) → [0] bad
        L=2: w=[1,2) → [0] bad
        L=3: w=[2,3) → [1] ok
        Safe = 3.
        """
        swa = torch.tensor([0, 0, 1], dtype=torch.int64)
        self.assertEqual(compute_swa_safe_len(swa, sliding_window_size=1), 3)


if __name__ == "__main__":
    unittest.main()
