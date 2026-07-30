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
        call_urls, nia, reordered = self.r._dispatch_by_modality(mm)

        self.assertEqual(set(call_urls), {"http://Ea:80", "http://Eb:80"})
        self.assertEqual(len(reordered), 6)  # length conserved
        counts = nia[Modality.IMAGE]
        self.assertEqual(len(counts), len(call_urls))  # padded to call_urls len
        self.assertEqual(sum(counts), 6)
        ea, eb = call_urls.index("http://Ea:80"), call_urls.index("http://Eb:80")
        self.assertEqual(counts[ea], 4)
        self.assertEqual(counts[eb], 2)

    def test_reordered_aligns_with_encoder_idx(self):
        # Every item in a built part must route (by SHA1) to that part's URL.
        mm = [_img("0"), _img("9"), _img("1"), _img("8"), _img("2"), _img("f")]
        call_urls, nia, reordered = self.r._dispatch_by_modality(mm)
        modalities = list(dict.fromkeys(m.get("modality") for m in reordered))
        ers, _ = self.r._build_encode_requests("R", reordered, 1234, nia, modalities)
        for er in ers:
            url = call_urls[er["encoder_idx"]]
            for item_url in er["mm_items"]:
                key = extract_mm_item_key(item_url)
                self.assertEqual(self.r._dispatcher.pick(key), url)

    def test_video_forced_single_encoder(self):
        vid = [
            {
                "url": f"http://x/{i:040x}.mp4",
                "modality": Modality.VIDEO,
                "preprocess_kwargs": None,
            }
            for i in range(3)
        ]
        call_urls, nia, reordered = self.r._dispatch_by_modality(vid)
        self.assertEqual(len(call_urls), 1)
        self.assertEqual(nia[Modality.VIDEO], [3])
        self.assertEqual(len(reordered), 3)


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
