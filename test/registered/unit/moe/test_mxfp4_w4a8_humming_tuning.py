import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.moe_runner import mxfp4_w4a8_humming as target
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


class TestHummingMaskedTuning(unittest.TestCase):
    def setUp(self):
        self.fallback = target._make_humming_masked_tuning_config(64, 4, True)
        self.w13_entry = SimpleNamespace(
            n=4096,
            k=6144,
            masked_tuning_config=self.fallback,
        )
        self.w2_entry = SimpleNamespace(
            n=6144,
            k=2048,
            masked_tuning_config=self.fallback,
        )
        self.new_w13_entry = SimpleNamespace(
            n=4096,
            k=4096,
            masked_tuning_config=self.fallback,
        )
        self.new_w2_entry = SimpleNamespace(
            n=4096,
            k=2048,
            masked_tuning_config=self.fallback,
        )

    def _assert_selected(self, entry, expected_m, block_m, num_stages):
        config = target._select_humming_masked_tuning_config(entry, expected_m)
        self.assertEqual(config["block_shape"], (block_m, 128, 128))
        self.assertEqual(config["warp_shape"], (block_m, 32, 128))
        self.assertEqual(config["num_stages"], num_stages)
        self.assertEqual(config["num_ctas_per_sm"], 2)
        self.assertEqual(config["num_sms"], 132)

    def test_expected_m_bucket_boundaries(self):
        cases = (
            (1, 8, 4),
            (8, 8, 4),
            (9, 16, 4),
            (16, 16, 4),
            (17, 24, 4),
            (24, 24, 4),
            (25, 32, 4),
            (32, 32, 4),
            (33, 40, 4),
            (40, 40, 4),
            (41, 48, 4),
            (48, 48, 4),
            (49, 56, 4),
            (56, 56, 4),
            (57, 64, 4),
            (64, 64, 4),
            (65, 72, 4),
            (72, 72, 4),
            (73, 80, 4),
            (80, 80, 4),
            (81, 88, 3),
            (88, 88, 3),
        )
        for expected_m, block_m, num_stages in cases:
            with self.subTest(expected_m=expected_m):
                self._assert_selected(self.w13_entry, expected_m, block_m, num_stages)

    def test_all_mimo_shapes_use_dynamic_tuning(self):
        self._assert_selected(self.w13_entry, 32, 32, 4)
        self._assert_selected(self.w2_entry, 72, 72, 4)
        self._assert_selected(self.new_w13_entry, 32, 32, 4)
        self._assert_selected(self.new_w2_entry, 72, 72, 4)

    def test_prefill_overrides_cover_all_mimo_shapes(self):
        supported_shapes = (
            (4096, 6144),
            (6144, 2048),
            (4096, 4096),
            (4096, 2048),
        )
        for shape in supported_shapes:
            with self.subTest(shape=shape):
                overrides = target._humming_prefill_tuning_overrides(*shape)
                self.assertEqual(len(overrides), 1)
                self.assertEqual(overrides[0][2]["block_shape"], (64, 128, 128))

    def test_out_of_range_and_unsupported_shape_fall_back(self):
        unsupported_entry = SimpleNamespace(
            n=8192,
            k=4096,
            masked_tuning_config=self.fallback,
        )
        self.assertIs(
            target._select_humming_masked_tuning_config(self.w13_entry, 89),
            self.fallback,
        )
        self.assertIs(
            target._select_humming_masked_tuning_config(unsupported_entry, 32),
            self.fallback,
        )

    def test_coarse_bucket_fallback_uses_stream_k(self):
        # A supported (N, K) shape with unknown EP metadata (ep_size=None) has no
        # fine table, so it falls to the generic coarse buckets. Those hardcode
        # stream_k=True; assert both block_m AND use_stream_k so the fallback path
        # is actually exercised (the old test used a shape whose fine table still
        # covered the em, so it never reached these buckets and never checked SK).
        entry = SimpleNamespace(
            n=4096, k=6144, ep_size=None, masked_tuning_config=self.fallback
        )
        for expected_m, block_m in ((66, 72), (78, 80), (85, 88)):
            with self.subTest(expected_m=expected_m):
                config = target._select_humming_masked_tuning_config(entry, expected_m)
                self.assertEqual(config["block_shape"], (block_m, 128, 128))
                self.assertTrue(config["use_stream_k"])

    def test_pro_ep8_fine_table_covers_em65_to_88(self):
        # em65-88 is only reached at BS>256, but the Pro EP8 fine table must cover
        # it with the measured configs (SK0 below em81, bm88/s3/SK1 at em81-88)
        # instead of falling through to the SK1 coarse bucket (wrong below em81).
        for n, k in ((4096, 6144), (6144, 2048)):
            entry = SimpleNamespace(
                n=n, k=k, ep_size=8, masked_tuning_config=self.fallback
            )
            for expected_m, block_m, stages, stream_k in (
                (66, 72, 4, False),
                (78, 80, 4, False),
                (85, 88, 3, True),
            ):
                with self.subTest(shape=(n, k), expected_m=expected_m):
                    config = target._select_humming_masked_tuning_config(
                        entry, expected_m
                    )
                    self.assertEqual(config["block_shape"], (block_m, 128, 128))
                    self.assertEqual(config["num_stages"], stages)
                    self.assertEqual(config["use_stream_k"], stream_k)

    def test_fine_tuning_separates_model_ep_and_gemm(self):
        flash_ep8_w13 = SimpleNamespace(
            n=4096, k=4096, ep_size=8, masked_tuning_config=self.fallback
        )
        flash_ep16_w13 = SimpleNamespace(
            n=4096, k=4096, ep_size=16, masked_tuning_config=self.fallback
        )
        pro_ep16_w13 = SimpleNamespace(
            n=4096, k=6144, ep_size=16, masked_tuning_config=self.fallback
        )
        pro_ep32_w13 = SimpleNamespace(
            n=4096, k=6144, ep_size=32, masked_tuning_config=self.fallback
        )
        pro_ep32_w2 = SimpleNamespace(
            n=6144, k=2048, ep_size=32, masked_tuning_config=self.fallback
        )
        cases = (
            (flash_ep8_w13, 65, 72, 4, False),
            (flash_ep16_w13, 81, 48, 4, False),
            (pro_ep16_w13, 80, 80, 4, False),
            (pro_ep16_w13, 81, 88, 3, True),
            (pro_ep32_w13, 80, 88, 3, True),
            (pro_ep32_w2, 80, 48, 4, False),
            (pro_ep32_w2, 121, 64, 4, False),
            (pro_ep32_w2, 161, 80, 4, False),
            (pro_ep32_w2, 169, 64, 4, False),
        )
        for entry, expected_m, block_m, stages, stream_k in cases:
            with self.subTest(
                shape=(entry.n, entry.k),
                ep_size=entry.ep_size,
                expected_m=expected_m,
            ):
                config = target._select_humming_masked_tuning_config(entry, expected_m)
                self.assertEqual(config["block_shape"], (block_m, 128, 128))
                self.assertEqual(config["num_stages"], stages)
                self.assertEqual(config["use_stream_k"], stream_k)

    def test_fine_warmup_deduplicates_repeated_configs(self):
        entry = SimpleNamespace(
            n=6144,
            k=2048,
            ep_size=32,
            masked_tuning_config=self.fallback,
        )
        configs = target._humming_masked_warmup_tuning_configs(entry)
        self.assertEqual(len(configs), len({repr(config) for config in configs}))

    def test_warmup_configs_cover_each_bucket_once(self):
        configs = target._humming_masked_warmup_tuning_configs(self.w13_entry)

        self.assertEqual(len(configs), 11)
        self.assertEqual(
            [config["block_shape"][0] for config in configs],
            [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88],
        )
        self.assertEqual(len({repr(config) for config in configs}), len(configs))

    def test_prefill_tuning_bridges_m_hole(self):
        # _apply_humming_prefill_tuning must not leave an uncovered (m0, 4096) band
        # between the retained small-M heuristic buckets and the >=4096 override --
        # Humming hard-errors ("shape_m is not within any range") on contiguous M
        # that lands in such a gap, reachable at ordinary chunked-prefill sizes.
        def coverage_gaps(ranges):
            segs = sorted((r[0], r[1]) for r in ranges)
            return [(a1, b0) for (a0, a1), (b0, b1) in zip(segs, segs[1:]) if b0 != a1]

        cfg = {"block_shape": (8, 128, 128)}
        # Heuristic whose retained small-M buckets stop at 2048 (24-local-expert
        # shape), plus its own large-M bucket that the <=4096 filter drops.
        heuristic = [[0, 1024, cfg], [1024, 2048, cfg], [2048, 1 << 30, cfg]]
        tuned = target._apply_humming_prefill_tuning(heuristic, 4096, 6144)
        self.assertEqual(coverage_gaps(tuned), [])
        # The >=4096 override was bridged down to m0=2048 to close the hole.
        override = [r for r in tuned if r[1] == (1 << 30)]
        self.assertEqual(len(override), 1)
        self.assertEqual(override[0][0], 2048)

        # A heuristic that already ends exactly at 4096 needs no bridging and stays
        # gapless.
        heuristic2 = [[0, 2048, cfg], [2048, 4096, cfg], [4096, 1 << 30, cfg]]
        tuned2 = target._apply_humming_prefill_tuning(heuristic2, 4096, 6144)
        self.assertEqual(coverage_gaps(tuned2), [])

    def test_fused_e8m0_tuning_preserves_warp_shape(self):
        config = {
            "block_shape": (8, 128, 128),
            "warp_shape": (8, 16, 128),
            "num_ctas_per_sm": 1,
        }

        tuned = target._tune_humming_fused_e8m0_configs(config)

        self.assertEqual(tuned["warp_shape"], (8, 16, 128))
        self.assertEqual(tuned["num_ctas_per_sm"], 2)


if __name__ == "__main__":
    unittest.main()
