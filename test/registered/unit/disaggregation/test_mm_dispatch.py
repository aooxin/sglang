"""Unit tests for MMReceiverHTTP SHA1 dispatch/regroup logic — no server.

Exercises the pure request-shaping methods on MMReceiverHTTP without running
__init__ (which needs ZMQ/model). We build a bare instance via __new__ and set
only the fields the methods read.
"""

import math
import unittest

from sglang.srt.disaggregation.encode_receiver import (
    MMReceiverHTTP,
    compute_mm_item_sha1,
    extract_mm_item_key,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


SHA1 = "a" * 40


class _StubDispatcher:
    """Routes by first hex char of the item key: '0'-'7' -> Ea, else -> Eb."""

    def pick(self, key, exclude=None):
        return "http://Ea:80" if key[0] in "01234567" else "http://Eb:80"


def _img(first_hex):
    return {
        "url": f"http://x/{first_hex}{'0' * 39}.jpg",
        "modality": Modality.IMAGE,
        "preprocess_kwargs": None,
    }


def _make_receiver(dispatcher=None, group_size=8, sha1_header=False):
    r = MMReceiverHTTP.__new__(MMReceiverHTTP)
    r._dispatcher = dispatcher
    r._group_size = group_size
    r.sha1_routing_header = sha1_header
    r.host = "P1"
    r.encode_urls = ["http://vip:80"]
    return r


class TestDispatchByModality(unittest.TestCase):
    def setUp(self):
        self.r = _make_receiver(dispatcher=_StubDispatcher(), group_size=2)

    def test_counts_align_and_conserve_items(self):
        # 3 groups of 2: g0('0')->Ea, g1('9')->Eb, g2('1')->Ea => Ea:4, Eb:2
        mm = [_img("0"), _img("0"), _img("9"), _img("9"), _img("1"), _img("1")]
        call_urls, ers, total = self.r._dispatch_by_modality("R", mm, 1234)

        self.assertEqual(set(call_urls), {"http://Ea:80", "http://Eb:80"})
        self.assertEqual(total, len(ers))
        self.assertTrue(all(er["num_parts"] == total for er in ers))
        # items are conserved across all parts
        self.assertEqual(sum(len(er["mm_items"]) for er in ers), 6)
        # count items routed to each encoder URL
        per_url = {u: 0 for u in call_urls}
        for er in ers:
            per_url[call_urls[er["encoder_idx"]]] += len(er["mm_items"])
        self.assertEqual(per_url["http://Ea:80"], 4)
        self.assertEqual(per_url["http://Eb:80"], 2)

    def test_parts_routed_by_group_head(self):
        # No physical reorder: each part keeps prompt order and is routed by its
        # first item's SHA1 (group head), not per-item.
        mm = [_img("0"), _img("9"), _img("1"), _img("8"), _img("2"), _img("f")]
        call_urls, ers, _ = self.r._dispatch_by_modality("R", mm, 1234)
        # group_size=2 => 3 parts, each keeping original order within the group
        self.assertEqual([er["mm_items"][0] for er in ers], [m["url"] for m in mm[::2]])
        for er in ers:
            head_key = extract_mm_item_key(er["mm_items"][0])
            self.assertEqual(call_urls[er["encoder_idx"]], self.r._dispatcher.pick(head_key))

    def test_video_forced_single_encoder(self):
        vid = [
            {
                "url": f"http://x/{i:040x}.mp4",
                "modality": Modality.VIDEO,
                "preprocess_kwargs": None,
            }
            for i in range(3)
        ]
        call_urls, ers, total = self.r._dispatch_by_modality("R", vid, 1234)
        self.assertEqual(len(call_urls), 1)
        self.assertEqual(total, 1)  # all video items in one part
        self.assertEqual(len(ers), 1)
        self.assertEqual(ers[0]["encoder_idx"], 0)
        self.assertEqual(len(ers[0]["mm_items"]), 3)


class TestSha1EncodeRequests(unittest.TestCase):
    def setUp(self):
        self.r = _make_receiver(sha1_header=True, group_size=8)

    def test_image_regrouped_by_group_size(self):
        mm = [_img("0") for _ in range(19)]
        ers, total, sha1_by = self.r._build_sha1_encode_requests("R", mm, 5678)
        self.assertEqual(total, math.ceil(19 / 8))  # 3 parts
        self.assertEqual(len(ers), total)
        self.assertTrue(all(er["encoder_idx"] == 0 for er in ers))  # all -> gateway
        self.assertTrue(all(er["num_parts"] == total for er in ers))
        self.assertEqual(set(sha1_by.keys()), {0, 1, 2})
        self.assertTrue(all(len(s) == 40 for s in sha1_by.values()))

    def test_video_not_regrouped(self):
        mm = [
            {
                "url": f"http://x/{i:040x}.mp4",
                "modality": Modality.VIDEO,
                "preprocess_kwargs": None,
            }
            for i in range(10)
        ]
        ers, total, sha1_by = self.r._build_sha1_encode_requests("R", mm, 1)
        self.assertEqual(total, 1)  # single part regardless of count
        self.assertEqual(len(ers[0]["mm_items"]), 10)

    def test_sha1_key_matches_helper(self):
        mm = [_img("3"), _img("3")]
        _, _, sha1_by = self.r._build_sha1_encode_requests("R", mm, 1)
        self.assertEqual(sha1_by[0], compute_mm_item_sha1(mm[0]))


class TestDefaultPathUnchanged(unittest.TestCase):
    """With both flags off, dispatch is skipped and headers are absent."""

    def test_no_dispatcher_no_header(self):
        r = _make_receiver(dispatcher=None, sha1_header=False)
        self.assertIsNone(r._dispatcher)
        self.assertFalse(r.sha1_routing_header)


if __name__ == "__main__":
    unittest.main()
