import unittest

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    _compute_deepep_expected_m,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


class TestDeepEPExpectedM(unittest.TestCase):
    def test_exact_division_does_not_round_up(self):
        self.assertEqual(_compute_deepep_expected_m(256, 4, 8, 256), 32)
        self.assertEqual(_compute_deepep_expected_m(256, 2, 8, 256), 16)
        self.assertEqual(_compute_deepep_expected_m(256, 4, 8, 384), 22)

    def test_non_exact_division_rounds_up(self):
        self.assertEqual(_compute_deepep_expected_m(257, 4, 8, 256), 33)
        self.assertEqual(_compute_deepep_expected_m(193, 2, 8, 256), 13)
        self.assertEqual(_compute_deepep_expected_m(257, 4, 8, 384), 22)

    def test_zero_tokens_floors_at_one(self):
        # DP-attention idle rank running eagerly: the MoE still dispatches with
        # 0 tokens, and the masked GEMMs need a positive expected_m.
        self.assertEqual(_compute_deepep_expected_m(0, 4, 8, 256), 1)
        self.assertEqual(_compute_deepep_expected_m(0, 2, 8, 384), 1)

    def test_common_decode_boundaries(self):
        cases = (
            (160, 2, 8, 256, 10),
            (192, 2, 8, 256, 12),
            (240, 4, 8, 384, 20),
            (256, 4, 8, 384, 22),
        )
        for num_tokens, group_size, top_k, num_experts, expected_m in cases:
            with self.subTest(
                num_tokens=num_tokens,
                group_size=group_size,
                top_k=top_k,
                num_experts=num_experts,
            ):
                self.assertEqual(
                    _compute_deepep_expected_m(
                        num_tokens, group_size, top_k, num_experts
                    ),
                    expected_m,
                )


if __name__ == "__main__":
    unittest.main()
