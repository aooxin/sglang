import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestHiRadixDeferredSWARewrite(unittest.TestCase):
    def _make_cache(self, *, pending_full_write: bool):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.is_hybrid_swa = True
        cache.page_size = 1
        cache.enable_hicache_metrics = False
        cache.evict_host = MagicMock()
        cache.inc_lock_ref = MagicMock()

        node = SimpleNamespace(
            id=7,
            value=torch.tensor([1, 2]),
            host_value=torch.tensor([11, 12]),
        )
        snapshot = {
            "node": node,
            "key": None,
            "hash_value": None,
            "host_value": node.host_value,
            "pending_write_count": 1,
        }
        cache.ongoing_write_through = {node.id: snapshot}

        host_mapping = torch.zeros(32, dtype=torch.int64)
        host_pool = MagicMock()
        host_pool.gather_swa_mapping.side_effect = lambda indices: host_mapping[indices]
        host_pool.scatter_swa_mapping.side_effect = (
            lambda indices, values: host_mapping.__setitem__(indices, values)
        )
        host_pool.alloc_unmapped_host_by_len.return_value = (
            None,
            torch.tensor([101]),
            0,
            0,
            None,
        )
        cache.token_to_kv_pool_host = host_pool

        device_mapping = torch.zeros(32, dtype=torch.int64)
        device_mapping[1] = 501
        cache.token_to_kv_pool_allocator = SimpleNamespace(
            full_to_swa_index_mapping=device_mapping
        )

        controller = MagicMock()
        controller.has_pending_full_write.return_value = pending_full_write
        cache.cache_controller = controller
        return cache, node, snapshot, host_mapping, controller

    def test_pending_full_write_absorbs_swa_rewrite(self):
        cache, node, snapshot, host_mapping, controller = self._make_cache(
            pending_full_write=True
        )

        result = cache.rewrite_swa_backup(
            node, torch.tensor([True, False]), reason="test_pending_full_write"
        )

        self.assertEqual(result, (0, 0))
        self.assertEqual(host_mapping[node.host_value].tolist(), [101, 0])
        self.assertEqual(snapshot["pending_write_count"], 1)
        controller.write.assert_not_called()
        cache.inc_lock_ref.assert_not_called()

    def test_inflight_write_appends_rewrite_and_waits_for_both_acks(self):
        cache, node, snapshot, host_mapping, controller = self._make_cache(
            pending_full_write=False
        )

        cache.rewrite_swa_backup(
            node, torch.tensor([True, False]), reason="test_inflight_write"
        )

        self.assertEqual(host_mapping[node.host_value].tolist(), [101, 0])
        self.assertEqual(snapshot["pending_write_count"], 2)
        controller.write.assert_called_once()
        self.assertEqual(
            controller.write.call_args.kwargs["device_indices"].tolist(), [1]
        )
        self.assertEqual(
            controller.write.call_args.kwargs["host_indices"].tolist(), [11]
        )
        cache.inc_lock_ref.assert_not_called()

        self.assertIsNone(cache._consume_write_ack(node.id))
        self.assertEqual(snapshot["pending_write_count"], 1)
        self.assertIs(cache._consume_write_ack(node.id), snapshot)
        self.assertNotIn(node.id, cache.ongoing_write_through)

    def test_missing_pending_write_count_fails_fast(self):
        cache, node, _snapshot, _host_mapping, _controller = self._make_cache(
            pending_full_write=True
        )
        del cache.ongoing_write_through[node.id]["pending_write_count"]

        with self.assertRaises(KeyError):
            cache._consume_write_ack(node.id)


if __name__ == "__main__":
    unittest.main()
