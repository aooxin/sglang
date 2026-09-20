import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestCacheControllerDeferWrites(unittest.TestCase):
    def _make_controller(self, *, io_backend="kernel"):
        controller = HiCacheController.__new__(HiCacheController)
        controller.io_backend = io_backend
        controller.is_hybrid_swa = False
        controller.write_queue = []
        controller._defer_write_depth = 0
        controller.start_writing = MagicMock()
        return controller

    def test_defer_writes_batches_multiple_operations(self):
        controller = self._make_controller()

        with controller.defer_writes():
            controller.write(torch.tensor([1]), torch.tensor([11]), node_id=1)
            controller.write(torch.tensor([2]), torch.tensor([12]), node_id=2)
            controller.start_writing.assert_not_called()

        controller.start_writing.assert_called_once_with()
        self.assertEqual(len(controller.write_queue), 2)

        merged = CacheOperation.merge_ops(controller.write_queue)
        self.assertEqual(merged.device_indices.tolist(), [1, 2])
        self.assertEqual(merged.host_indices.tolist(), [11, 12])
        self.assertEqual(merged.node_ids, [1, 2])

    def test_merge_ops_preserves_swa_mappings(self):
        first = CacheOperation(
            torch.tensor([11, 12]),
            torch.tensor([1, 2]),
            node_id=1,
            swa_host_indices=torch.tensor([101, 102]),
            swa_device_indices=torch.tensor([201, 202]),
        )
        second = CacheOperation(
            torch.tensor([13]),
            torch.tensor([3]),
            node_id=2,
            swa_host_indices=torch.tensor([103]),
            swa_device_indices=torch.tensor([203]),
        )

        merged = CacheOperation.merge_ops([first, second])

        self.assertEqual(merged.swa_host_indices.tolist(), [101, 102, 103])
        self.assertEqual(merged.swa_device_indices.tolist(), [201, 202, 203])

    def test_deferred_hybrid_write_does_not_precompute_swa_indices(self):
        controller = self._make_controller()
        controller.is_hybrid_swa = True
        controller.mem_pool_host = MagicMock()
        controller.mem_pool_device = MagicMock()

        with controller.defer_writes():
            controller.write(torch.tensor([1]), torch.tensor([11]), node_id=1)
            controller.mem_pool_host.gather_swa_mapping.assert_not_called()

        operation = controller.write_queue[0]
        self.assertIsNone(operation.swa_host_indices)
        self.assertIsNone(operation.swa_device_indices)

    def test_has_pending_full_write_only_matches_full_node_operation(self):
        controller = self._make_controller()

        with controller.defer_writes():
            controller.write(
                torch.tensor([1, 2]),
                torch.tensor([11, 12]),
                node_id=1,
                is_full_node_write=True,
            )
            controller.write(torch.tensor([3]), torch.tensor([13]), node_id=2)

            self.assertTrue(controller.has_pending_full_write(1))
            self.assertFalse(controller.has_pending_full_write(2))
            self.assertFalse(controller.has_pending_full_write(3))

        controller.start_writing.assert_called_once_with()

    def _make_swa_controller(self, *, write_policy="write_through"):
        controller = self._make_controller()
        controller.is_hybrid_swa = True
        controller.write_policy = write_policy
        controller.device = "cpu"
        controller.mem_pool_host = MagicMock()
        controller.mem_pool_device = MagicMock()
        return controller

    @staticmethod
    def _install_swa_mappings(controller, device_mapping, host_mapping):
        controller.mem_pool_device.full_to_swa_index_mapping.__getitem__.side_effect = (
            lambda idx: device_mapping[idx]
        )
        controller.mem_pool_host.gather_swa_mapping.side_effect = (
            lambda idx: host_mapping[idx]
        )

    def test_write_back_skips_device_only_swa_positions(self):
        # Reproduces the flush-time state after a deferred write overlapped with a
        # later insert's splice_swa_mapping: position 1 gained device SWA that the
        # host has not been repaired for yet (legal under write_back, fixed at
        # evict). The backup must skip it instead of writing into host swa slot 0.
        controller = self._make_swa_controller(write_policy="write_back")
        device_mapping = torch.zeros(16, dtype=torch.int64)
        host_mapping = torch.zeros(16, dtype=torch.int64)
        self._install_swa_mappings(controller, device_mapping, host_mapping)

        op = CacheOperation(
            torch.tensor([11, 12, 13, 14]),
            torch.tensor([1, 2, 3, 4]),
            node_id=1,
        )
        device_mapping[1] = 501  # device-only: skip
        device_mapping[2] = 502  # both valid: copy
        host_mapping[12] = 102
        host_mapping[13] = 103  # host-only: never selected (device is zero)

        swa_args = controller._write_swa_args(op, op.host_indices, op.device_indices)

        self.assertEqual(swa_args["swa_device_indices"].tolist(), [502])
        self.assertEqual(swa_args["swa_host_indices"].tolist(), [102])
        # Raw indices are still cached on the op for later consumers.
        self.assertEqual(op.swa_device_indices.tolist(), [501, 502, 0, 0])
        self.assertEqual(op.swa_host_indices.tolist(), [0, 102, 103, 0])

    def test_write_through_warns_and_skips_device_only_positions(self):
        controller = self._make_swa_controller(write_policy="write_through")
        device_mapping = torch.zeros(16, dtype=torch.int64)
        host_mapping = torch.zeros(16, dtype=torch.int64)
        self._install_swa_mappings(controller, device_mapping, host_mapping)

        op = CacheOperation(
            torch.tensor([11, 12]),
            torch.tensor([1, 2]),
            node_id=1,
        )
        device_mapping[1] = 501  # device-only, host not repaired

        with self.assertLogs(
            "sglang.srt.managers.cache_controller", level="WARNING"
        ) as logs:
            swa_args = controller._write_swa_args(
                op, op.host_indices, op.device_indices
            )

        self.assertIn("SWA backup skipped 1 device-only positions", logs.output[0])
        self.assertEqual(swa_args["swa_device_indices"].numel(), 0)
        self.assertEqual(swa_args["swa_host_indices"].numel(), 0)

    def test_load_swa_mapping_mismatch_raises(self):
        controller = self._make_swa_controller()
        device_mapping = torch.zeros(16, dtype=torch.int64)
        host_mapping = torch.zeros(16, dtype=torch.int64)
        self._install_swa_mappings(controller, device_mapping, host_mapping)

        device_mapping[1] = 501

        with self.assertRaisesRegex(
            AssertionError, "SWA mapping mismatch: 1 device-only positions"
        ):
            controller._compute_swa_indices(
                torch.tensor([11]),
                torch.tensor([1]),
                raise_on_mismatch=True,
            )

    def test_write_through_selection_unchanged_when_host_in_sync(self):
        controller = self._make_swa_controller(write_policy="write_through")
        device_mapping = torch.zeros(16, dtype=torch.int64)
        host_mapping = torch.zeros(16, dtype=torch.int64)
        self._install_swa_mappings(controller, device_mapping, host_mapping)

        op = CacheOperation(
            torch.tensor([11, 12, 13, 14]),
            torch.tensor([1, 2, 3, 4]),
            node_id=1,
        )
        device_mapping[2] = 502
        host_mapping[12] = 102
        host_mapping[13] = 103  # host-only: excluded, as before the fix

        swa_args = controller._write_swa_args(op, op.host_indices, op.device_indices)

        self.assertEqual(swa_args["swa_device_indices"].tolist(), [502])
        self.assertEqual(swa_args["swa_host_indices"].tolist(), [102])

    def test_preset_snapshot_ops_skip_mismatched_positions(self):
        # Snapshot-path ops (swa indices precomputed at enqueue) always satisfy
        # device⊆host, so the intersection is an identity for them; but if one
        # ever carried a device-only position it must still be skipped rather
        # than written into host swa slot 0.
        controller = self._make_swa_controller(write_policy="write_through")

        op = CacheOperation(
            torch.tensor([11, 12]),
            torch.tensor([1, 2]),
            node_id=1,
            swa_host_indices=torch.tensor([0, 102]),
            swa_device_indices=torch.tensor([501, 502]),
        )

        swa_args = controller._write_swa_args(op, op.host_indices, op.device_indices)

        self.assertEqual(swa_args["swa_device_indices"].tolist(), [502])
        self.assertEqual(swa_args["swa_host_indices"].tolist(), [102])
        controller.mem_pool_host.gather_swa_mapping.assert_not_called()

    def test_nested_defer_writes_flushes_once(self):
        controller = self._make_controller()

        with controller.defer_writes():
            controller.write(torch.tensor([1]), torch.tensor([11]))
            with controller.defer_writes():
                controller.write(torch.tensor([2]), torch.tensor([12]))
            controller.start_writing.assert_not_called()

        controller.start_writing.assert_called_once_with()

    def test_defer_writes_flushes_on_exception(self):
        controller = self._make_controller()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with controller.defer_writes():
                controller.write(torch.tensor([1]), torch.tensor([11]))
                raise RuntimeError("boom")

        controller.start_writing.assert_called_once_with()
        self.assertEqual(controller._defer_write_depth, 0)

    def test_non_kernel_backend_keeps_immediate_flush(self):
        controller = self._make_controller(io_backend="direct")

        with controller.defer_writes():
            controller.write(torch.tensor([1]), torch.tensor([11]))

        controller.start_writing.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
