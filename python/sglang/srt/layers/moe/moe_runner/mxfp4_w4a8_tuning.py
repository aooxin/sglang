from __future__ import annotations

from typing import Final

TuningRange = tuple[int, int, int, int, bool]

# Flash EP8/EP16 include the measured M=65/129 points from the original search
# envelope. With the corrected DeepEP ceil formula, BS<=256 no longer reaches
# those exact-divisibility points, but they remain valid measured configurations
# for larger workloads. Several high-M block_m decreases below are intentional
# tuner restarts, not typos. Near-tied S3/S4 choices for the same tile are merged
# to avoid needless cubin variants.
HUMMING_FINE_TUNING_RANGES: Final[
    dict[tuple[int, int, int], tuple[TuningRange, ...]]
] = {
    (4096, 4096, 8): (
        (1, 8, 8, 4, False),
        (9, 16, 16, 4, False),
        (17, 24, 24, 4, False),
        (25, 32, 32, 4, False),
        (33, 40, 40, 4, False),
        (41, 48, 48, 4, False),
        (49, 56, 56, 4, False),
        (57, 64, 64, 4, False),
        # High-em extension for BS>256 (dense isolated, 8xH20Z 2026-08-24; W13==W2,
        # all SK0). 1-tile (block_m covers masked_m_max) while max<=80; beyond that
        # block_m=88 is register-inefficient so it flips to 2-tile occupancy.
        (65, 72, 72, 4, False),
        (73, 80, 80, 4, False),
        (81, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
    ),
    (4096, 2048, 8): (
        (1, 8, 8, 4, False),
        (9, 16, 16, 4, False),
        (17, 24, 24, 4, False),
        (25, 32, 32, 4, False),
        (33, 40, 40, 4, False),
        (41, 48, 48, 4, False),
        (49, 56, 56, 4, False),
        (57, 64, 64, 4, False),
        (65, 72, 72, 4, False),
        (73, 80, 80, 4, False),
        (81, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
    ),
    (4096, 4096, 16): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24): block_m sized to
        # per-expert MAX, not mean. Verify4 range (em<=64) shifted down one bucket vs
        # the old balance-tuned table (isolated grid: W13==W2, all SK0). em>=65 uses
        # the dense-isolated high-em extension below (verify8 territory; not part of
        # this max-covering pass).
        (1, 3, 8, 4, False),
        (4, 16, 16, 4, False),
        (17, 19, 24, 4, False),
        (20, 32, 32, 4, False),
        (33, 34, 40, 4, False),
        (35, 48, 48, 4, False),
        (49, 51, 56, 4, False),
        (52, 64, 64, 4, False),
        # High-em extension for BS>256 (dense isolated, 8xH20Z 2026-08-24; W13==W2,
        # all SK0). 1-tile while max<=80, then multi-tile occupancy (block_m steps
        # down as tile-count rises) since 16 local experts need many CTAs to fill
        # 132 SMs. Replaces the old balance-tuned em81-88 BM88/SK1 with BM48/SK0.
        (65, 67, 72, 4, False),
        (68, 80, 80, 4, False),
        (81, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
        (129, 144, 72, 4, False),
        (145, 160, 80, 4, False),
        (161, 165, 56, 4, False),
        (166, 192, 64, 4, False),
        (193, 213, 72, 4, False),
        (214, 240, 80, 4, False),
        (241, 256, 64, 4, False),
    ),
    (4096, 2048, 16): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24); same block_m
        # curve as W13 for Flash EP16 (isolated grid: W13==W2, all SK0), incl the
        # BS>256 high-em extension.
        (1, 3, 8, 4, False),
        (4, 16, 16, 4, False),
        (17, 19, 24, 4, False),
        (20, 32, 32, 4, False),
        (33, 34, 40, 4, False),
        (35, 48, 48, 4, False),
        (49, 51, 56, 4, False),
        (52, 64, 64, 4, False),
        (65, 67, 72, 4, False),
        (68, 80, 80, 4, False),
        (81, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
        (129, 144, 72, 4, False),
        (145, 160, 80, 4, False),
        (161, 165, 56, 4, False),
        (166, 192, 64, 4, False),
        (193, 213, 72, 4, False),
        (214, 240, 80, 4, False),
        (241, 256, 64, 4, False),
    ),
    (4096, 6144, 16): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24): block_m sized
        # to the skewed per-expert MAX (masked_m_max ~= 1.3-1.5x expected_m), not the
        # mean. Boundaries shifted down one bucket vs the old balance-tuned table;
        # recovers up to 46% MoE latency at bucket tops (BS 33-48/113-144/211-238),
        # <=3.5% cost on perfectly-balanced load. em>=49 unreached at verify4.
        (1, 3, 8, 4, False),
        (4, 16, 16, 4, False),
        (17, 19, 24, 4, False),
        (20, 32, 32, 4, False),
        (33, 35, 40, 4, False),
        (36, 48, 48, 4, False),
        (49, 51, 56, 4, False),
        (52, 64, 64, 4, False),
        (65, 67, 72, 4, False),
        (68, 80, 80, 4, False),
        # High-em (verify8 BS>258) measured 8xH20Z 2026-08-26 (isolated w/ SK1+S3).
        # em81-85 BM88/S3/SK1 is genuinely optimal (Pro W13 K=6144 is deep -> stream-k
        # helps, unlike Flash K=4096); at em86 (max>88) it flips to 2-tile occupancy.
        # Top: em167-171 returns to BM88/S3/SK1 like EP32; em161-166 stays
        # BM56/SK0 (measured). W13==W2.
        (81, 85, 88, 3, True),
        (86, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
        (129, 144, 72, 4, False),
        (145, 160, 80, 4, False),
        (161, 166, 56, 4, False),
        (167, 171, 88, 3, True),
    ),
    (6144, 2048, 16): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24); W2 uses the same
        # block_m curve as W13 for Pro EP16 (isolated grid: W13==W2 optimum), incl the
        # verify8 high-em extension (em81-85 & 167-171 BM88/S3/SK1, else SK0;
        # em161-166 is BM56/SK0).
        (1, 3, 8, 4, False),
        (4, 16, 16, 4, False),
        (17, 19, 24, 4, False),
        (20, 32, 32, 4, False),
        (33, 35, 40, 4, False),
        (36, 48, 48, 4, False),
        (49, 51, 56, 4, False),
        (52, 64, 64, 4, False),
        (65, 67, 72, 4, False),
        (68, 80, 80, 4, False),
        (81, 85, 88, 3, True),
        (86, 96, 48, 4, False),
        (97, 112, 56, 4, False),
        (113, 128, 64, 4, False),
        (129, 144, 72, 4, False),
        (145, 160, 80, 4, False),
        (161, 166, 56, 4, False),
        (167, 171, 88, 3, True),
    ),
    (4096, 6144, 32): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24). W13 wants
        # stream-k (SK1) while masked_m_max<=88 (1 tile): +4-12% vs SK0. Above that
        # (em>=83) no config fits the max in one tile, so it becomes an occupancy
        # problem: block_m ~= ceil(masked_m_max/2) (2 tiles => 2x CTAs to fill the
        # 12-expert x 132-SM grid) and SK0 wins, except the very top (em>=163).
        (1, 7, 8, 4, True),
        (8, 13, 16, 4, True),
        (14, 18, 24, 4, True),
        (19, 34, 32, 4, True),
        (35, 45, 48, 4, True),
        (46, 50, 56, 4, True),
        (51, 66, 64, 4, True),
        (67, 69, 72, 4, True),
        (70, 79, 80, 4, True),
        (80, 82, 88, 3, True),
        (83, 98, 48, 4, False),
        (99, 103, 56, 4, False),
        (104, 130, 64, 4, False),
        (131, 135, 72, 4, False),
        (136, 162, 80, 4, False),
        (163, 171, 88, 3, True),
    ),
    (6144, 2048, 32): (
        # fake-topk / max-covering calibration (8xH20Z, 2026-08-24). W2 never wants
        # stream-k (SK0 throughout). block_m covers masked_m_max (1 tile) up to em~79,
        # then ~= ceil(masked_m_max/2) in the occupancy regime (em>=80).
        (1, 5, 8, 4, False),
        (6, 10, 16, 4, False),
        (11, 15, 24, 4, False),
        (16, 34, 32, 4, False),
        (35, 42, 48, 4, False),
        (43, 50, 56, 4, False),
        (51, 66, 64, 4, False),
        (67, 69, 72, 4, False),
        (70, 79, 80, 4, False),
        (80, 98, 48, 4, False),
        (99, 111, 56, 4, False),
        (112, 130, 64, 4, False),
        (131, 141, 72, 4, False),
        (142, 162, 80, 4, False),
        (163, 167, 56, 4, False),
        (168, 171, 64, 4, False),
    ),
    # Pro EP8 (48 local experts): DP1 BS<=256 reaches em<=22 (v4) / <=43 (v8).
    # Measured 8xH20Z 2026-08-26 (diagonal all-BS + isolated): W13==W2, all SK0
    # (48 experts give high occupancy -> no stream-k, unlike EP32), block_m covers
    # masked_m_max in one tile. fake-topk skew is mild here (max ~= mean) so this is
    # the equal-width covering table.
    #
    # em65-88 extension (measured 8xH20Z 2026-08-29): only reachable at BS>256, but
    # without it these rows fell through to the generic SK1 coarse bucket, which is
    # wrong below em81. Measured SK0-vs-SK1 per point (W13 and W2 agree): em65-80
    # SK0 wins (3-7.5% over SK1), em81-88 bm88/s3/SK1 wins (16-19% over SK0; deep-K
    # stream-k, same as EP16/32 -- 1-tile SK1 also beats multi-tile bm48 SK0 here).
    (4096, 6144, 8): (
        (1, 8, 8, 4, False),
        (9, 16, 16, 4, False),
        (17, 24, 24, 4, False),
        (25, 32, 32, 4, False),
        (33, 40, 40, 4, False),
        (41, 48, 48, 4, False),
        (49, 56, 56, 4, False),
        (57, 64, 64, 4, False),
        (65, 72, 72, 4, False),
        (73, 80, 80, 4, False),
        (81, 88, 88, 3, True),
    ),
    (6144, 2048, 8): (
        (1, 8, 8, 4, False),
        (9, 16, 16, 4, False),
        (17, 24, 24, 4, False),
        (25, 32, 32, 4, False),
        (33, 40, 40, 4, False),
        (41, 48, 48, 4, False),
        (49, 56, 56, 4, False),
        (57, 64, 64, 4, False),
        (65, 72, 72, 4, False),
        (73, 80, 80, 4, False),
        (81, 88, 88, 3, True),
    ),
}
