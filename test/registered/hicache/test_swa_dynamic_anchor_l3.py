"""E2E test: dynamic SWA reuse anchors survive an L1/L2 flush via L3 (write_through).

This test exercises the exact gap that dynamic SWA anchors are meant to close:
a full-KV prefix that outran the SWA tail is normally dropped from the sliding
window (and therefore never written to remote storage), so after the device and
host tiers are cleared the prefix can no longer be SWA-safely reused. A dynamic
anchor protects that window, and under the ``write_through`` policy the protected
SWA pages are persisted to L3, so a replay can recover the prefix from storage.

Flow (single rank; the test launches and tears down its own server):

  1. Register a dynamic anchor.
       - Request A = shared prefix ``P`` + divergent tail ``tail_A``. Its SWA
         window ends past ``P`` (around ``len(P)+len(tail_A)``), so A's eviction
         frees the SWA of the ``P`` region.
       - Request B = ``P`` + ``tail_B``. B's full-KV hit reaches ``len(P)`` while
         the SWA-safe match is trimmed far below it, so B carries
         ``swa_branching_seqlen = floor_align(len(P))``. When B finishes,
         ``evict_swa`` registers a dynamic anchor at that seqlen, and B's own
         prefill has repopulated the ``P``-region SWA that the anchor then
         protects. write_through persists the protected SWA to L3.
  2. Clear L1/L2 with ``POST /flush_cache`` (resets device + host, keeps L3).
  3. Replay request C = ``P`` + ``tail_B``. The shared prefix can only come from
     L3 now.
  4. Assert C recovers most of ``P`` from L3, and materially more than the same
     flow with dynamic anchors disabled (control group).

The control group (``SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS=0``) runs the identical
workload in a freshly launched server; without the anchor the ``P``-region SWA
is never persisted, so the replay recovers far fewer cached tokens. The single
test method runs both configurations sequentially so the comparison is direct.

Requirements: a Hopper-class GPU with ``openai/gpt-oss-20b`` weights available
(same model/hardware as test_swa_radix_cache_kl.py). Run it manually with:

    python3 test/registered/hicache/test_swa_dynamic_anchor_l3.py
"""

import os
import random
import shutil
import tempfile
import time
import unittest
from typing import Dict, List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Any hybrid-SWA model works (is_hybrid_swa_model: Llama4, GptOss, MiMoV2,
# Step3p5). The default is openai/gpt-oss-20b — the one hybrid-SWA model wired
# into SGLang's tests (see test_swa_radix_cache_kl.py), sliding window 128.
#
# To run against another checkpoint (e.g. a MiMoV2 model closer to production),
# point the test at it with env vars:
#     SGLANG_SWA_ANCHOR_TEST_MODEL=/path/to/mimo_v2_ckpt \
#     SGLANG_SWA_ANCHOR_TEST_PREFIX_LEN=1024 \
#     python3 test/registered/hicache/test_swa_dynamic_anchor_l3.py
# The only hard constraint is PREFIX_LEN > the model's sliding_window_size, so
# the full-KV hit at the shared prefix outruns the SWA tail and a dynamic anchor
# is registered. Set PREFIX_LEN comfortably above your model's window.
MODEL = os.environ.get("SGLANG_SWA_ANCHOR_TEST_MODEL", "openai/gpt-oss-20b")

register_cuda_ci(est_time=500, suite="stage-b-test-1-gpu-large")

# Must match the server's --page-size (floor_align depends on it). Production
# MiMo runs 128; the gpt-oss default here is 64.
PAGE_SIZE = int(os.environ.get("SGLANG_SWA_ANCHOR_TEST_PAGE_SIZE", "64"))
# Shared prefix P: page-aligned and required to exceed the model's sliding
# window (128 for the gpt-oss default). Override for models with a larger window.
PREFIX_LEN = int(os.environ.get("SGLANG_SWA_ANCHOR_TEST_PREFIX_LEN", "512"))
TAIL_LEN = PREFIX_LEN  # divergent per-request tail; pushes A's SWA window past P
GEN_TOKENS = 8  # keep generation short; it does not affect the branch at P


def _rand_ids(n: int, lo: int = 1000, hi: int = 100000) -> List[int]:
    """A random token-id list. Explicit ids avoid tokenizer round-trip drift so
    the shared prefix matches exactly at the token level across requests."""
    return [random.randint(lo, hi) for _ in range(n)]


class TestSWADynamicAnchorL3(CustomTestCase):
    """Replay-after-flush recovery of a dynamic-anchor-protected SWA prefix."""

    def _launch_server(self, dynamic_anchors: bool):
        temp_dir = tempfile.mkdtemp()
        # Only the flags the test itself depends on are fixed here. Everything
        # model/hardware-specific (tp-size, quantization, attention backend,
        # trust-remote-code, mem-fraction, mem layout, ...) is supplied via
        # SGLANG_SWA_ANCHOR_TEST_EXTRA_ARGS so the test tracks your real launch
        # config without edits. Do NOT repeat the flags below in EXTRA_ARGS.
        other_args = [
            "--enable-hierarchical-cache",
            "--hicache-storage-backend",
            "file",  # local single-node L3 (production may use gcache)
            "--hicache-write-policy",
            "write_through",
            "--page-size",
            str(PAGE_SIZE),
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]
        extra = os.environ.get("SGLANG_SWA_ANCHOR_TEST_EXTRA_ARGS", "").split()
        other_args.extend(extra)
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": temp_dir,
            "SGLANG_SIMPLE_SWA_DYNAMIC_ANCHORS": "1" if dynamic_anchors else "0",
        }
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        return process, temp_dir

    def _generate(self, input_ids: List[int], max_new_tokens: int = GEN_TOKENS) -> Dict:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    @staticmethod
    def _cached_tokens(resp: Dict) -> int:
        return int(resp.get("meta_info", {}).get("cached_tokens", 0))

    def _flush(self):
        # /flush_cache resets device (L1) + host (L2) but leaves the file
        # storage backend (L3) intact; the timeout lets it drain in-flight reqs.
        res = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/flush_cache",
            params={"timeout": 30},
            timeout=60,
        )
        res.raise_for_status()

    def _run_flow(self, dynamic_anchors: bool) -> int:
        """Run the anchor / flush / replay flow; return the replay's cached_tokens."""
        process, temp_dir = self._launch_server(dynamic_anchors)
        try:
            prefix = _rand_ids(PREFIX_LEN)
            tail_a = _rand_ids(TAIL_LEN)
            tail_b = _rand_ids(TAIL_LEN)
            # Distinct tails so A and B diverge exactly at the shared prefix P.
            self.assertNotEqual(tail_a, tail_b)

            # (1) Populate + register the dynamic anchor at floor_align(len(P)).
            self._generate(prefix + tail_a)
            self._generate(prefix + tail_b)

            # Let write_through drain the protected SWA pages to L3 before flush.
            time.sleep(2.0)

            # (2) Clear L1/L2, keep L3.
            self._flush()
            time.sleep(1.0)

            # (3) Replay the B request; its prefix can now only come from L3.
            replay = self._generate(prefix + tail_b)
            cached = self._cached_tokens(replay)
            print(
                f"[dynamic_anchors={dynamic_anchors}] replay cached_tokens={cached} "
                f"(prefix_len={PREFIX_LEN}, page_size={PAGE_SIZE})"
            )
            return cached
        finally:
            kill_process_tree(process.pid)
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Give the port time to free before the next launch.
            time.sleep(3.0)

    def test_l3_recovers_anchor_protected_prefix(self):
        cached_on = self._run_flow(dynamic_anchors=True)
        cached_off = self._run_flow(dynamic_anchors=False)

        print(
            f"cached_tokens: anchors_on={cached_on}, anchors_off={cached_off}, "
            f"delta={cached_on - cached_off}"
        )

        # With anchors the P-region SWA is persisted to L3, so the replay should
        # recover most of the (page-aligned) shared prefix from storage.
        self.assertGreaterEqual(
            cached_on,
            PREFIX_LEN // 2,
            "Dynamic anchor should let the replay recover most of the prefix from L3",
        )
        # And it should recover materially more than the anchor-off control,
        # which cannot SWA-match the P prefix after the flush.
        self.assertGreaterEqual(
            cached_on - cached_off,
            2 * PAGE_SIZE,
            "Dynamic anchors should recover meaningfully more cached tokens than "
            "the anchor-off control after an L1/L2 flush",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
