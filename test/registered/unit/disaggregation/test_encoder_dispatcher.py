"""Unit tests for encoder_dispatcher.py — consistent-hash routing (no server)."""

import hashlib
import unittest

from sglang.srt.disaggregation.encoder_dispatcher import (
    HashRing,
    LocalHashRingDispatcher,
    _key_to_pos,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


SHA1 = "a" * 40


class TestKeyToPos(unittest.TestCase):
    def test_40_hex_uses_prefix(self):
        # A 40-char hex string (a real SHA1) maps via its first 16 hex chars.
        self.assertEqual(_key_to_pos(SHA1), int(SHA1[:16], 16))

    def test_non_hex_falls_back_to_sha1(self):
        key = "http://example/pic.jpg"
        expected = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")
        self.assertEqual(_key_to_pos(key), expected)

    def test_wrong_length_hex_falls_back(self):
        # 39 hex chars is not a SHA1 -> hashed, not prefix-parsed.
        key = "a" * 39
        expected = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")
        self.assertEqual(_key_to_pos(key), expected)


class TestHashRing(unittest.TestCase):
    def setUp(self):
        self.eps = ["http://a:80", "http://b:80", "http://c:80"]
        self.ring = HashRing(self.eps)

    def test_empty_ring_returns_none(self):
        self.assertIsNone(HashRing([]).get("anything"))

    def test_endpoints_deduped_and_sorted(self):
        ring = HashRing(["http://b:80", "http://a:80", "http://a:80"])
        self.assertEqual(ring.endpoints, ["http://a:80", "http://b:80"])

    def test_pick_is_deterministic(self):
        self.assertEqual(self.ring.get(SHA1), self.ring.get(SHA1))

    def test_pick_returns_known_endpoint(self):
        self.assertIn(self.ring.get(SHA1), self.eps)

    def test_exclude_skips_endpoint(self):
        first = self.ring.get(SHA1)
        second = self.ring.get(SHA1, exclude={first})
        self.assertNotEqual(second, first)
        self.assertIn(second, self.eps)

    def test_exclude_all_returns_none(self):
        self.assertIsNone(self.ring.get(SHA1, exclude=set(self.eps)))

    def test_distribution_spans_endpoints(self):
        # Many distinct keys should hit more than one endpoint. Use real SHA1
        # hexdigests: _key_to_pos maps a 40-hex key via its first 16 chars, so
        # keys must carry entropy in the leading digits (f"{i:040x}" would not).
        picks = {
            self.ring.get(hashlib.sha1(str(i).encode()).hexdigest())
            for i in range(200)
        }
        self.assertGreater(len(picks), 1)


class TestLocalHashRingDispatcher(unittest.TestCase):
    def test_pick_falls_back_to_vip_when_no_ring(self):
        # Before start()/endpoint sync, ring is None -> fallback to VIP url.
        vip = "http://encoder-vip:80"
        disp = LocalHashRingDispatcher(vip)
        self.assertEqual(disp.pick(SHA1), vip)


if __name__ == "__main__":
    unittest.main()
