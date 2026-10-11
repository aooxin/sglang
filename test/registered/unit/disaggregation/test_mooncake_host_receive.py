"""Host destination wire compatibility, drain lifetime and TP admission consensus."""

import dataclasses
import os
import tempfile
import threading
import unittest
from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.base.conn import KVArgs, KVTransferDestination
from sglang.srt.disaggregation.common.conn import CommonKVManager, KVTransferError
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _receiver():
    receiver = object.__new__(MooncakeKVReceiver)
    mgr = object.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    mgr.enable_deferred_decode_kv_release = True
    mgr.attn_tp_size = 8
    mgr.dcp_size = 1
    mgr.enable_staging = False
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = 9000
    receiver.kv_mgr = mgr
    receiver.prefill_info = SimpleNamespace(
        supports_host_destination=True, attn_tp_size=8, attn_cp_size=1
    )
    receiver.bootstrap_infos = [dict(is_dummy=False), dict(is_dummy=True)]
    receiver.bootstrap_room = 17
    receiver.session_id = "session"
    receiver.require_staging = False
    receiver.required_dst_info_num = 2
    receiver._host_destination_published = False
    receiver._host_drain_armed = False
    receiver._connect_to_bootstrap_server = Mock(
        return_value=(Mock(), threading.Lock())
    )
    return receiver


def _registration():
    return KVArgsRegisterInfo(
        room="None",
        endpoint="127.0.0.1",
        dst_port=9000,
        mooncake_session_id="session",
        dst_kv_ptrs=[10, 20],
        dst_aux_ptrs=[30],
        dst_state_data_ptrs=[[40, 50]],
        dst_tp_rank=0,
        dst_attn_tp_size=8,
        dst_kv_item_len=1024,
        dst_state_item_lens=[[512, 256]],
        dst_state_dim_per_tensor=[[0, 0]],
        dst_kv_layer_ids=[0, 0],
        dst_state_layer_ids=[[1, 1]],
        dst_host_kv_ptrs=[110, 120],
        dst_host_state_data_ptrs=[[140, 150]],
        dst_host_state_item_lens=[[512, 256]],
    )


def _consensus_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        queue = object.__new__(DecodePreallocQueue)
        queue.gloo_group = dist.group.WORLD
        # One delayed/short rank vetoes admission; both resume in the same tick.
        for states, expected in [
            ((True, False), False),
            ((False, True), False),
            ((False, False), False),
            ((True, True), True),
        ]:
            assert queue._host_receive_all_ready(states[rank]) == expected
        # Reuse the actual host allocator method. A local SWA failure rolls back
        # the peer's successful allocation as well as both full allocations.
        queue.host_pool = Mock(page_size=4)
        queue.host_pool.available_size.return_value = 32
        queue.host_pool.alloc.return_value = torch.arange(4, 12)
        queue.host_swa_pool = Mock()
        queue.host_swa_pool.available_size.return_value = 32
        queue.host_swa_pool.alloc.return_value = None if rank else torch.arange(4, 12)
        queue.host_reserved_tokens = 4
        queue.host_swa_reserved_tokens = 4
        queue.scheduler = SimpleNamespace(sliding_window_size=8)
        queue.tree_cache = SimpleNamespace(disable=True)
        request = SimpleNamespace(req=SimpleNamespace(origin_input_ids=[1] * 7))
        assert not queue._pre_alloc_host(request)
        queue.host_pool.free.assert_called_once()
        assert queue.host_swa_pool.free.call_count == (0 if rank else 1)
        assert queue._host_receive_all_ready(True)
    finally:
        dist.destroy_process_group()


class TestMooncakeHostReceive(CustomTestCase):
    def test_destination_does_not_mutate_cached_device_registration(self):
        device = _registration()
        original = dataclasses.asdict(device)
        host = device.for_destination(KVTransferDestination.HOST)
        self.assertEqual(host.dst_kv_ptrs, [110, 120])
        self.assertEqual(host.dst_state_data_ptrs, [[140, 150]])
        self.assertEqual(host.dst_aux_ptrs, [30])
        self.assertIs(device.for_destination(KVTransferDestination.DEVICE), device)
        self.assertEqual(dataclasses.asdict(device), original)

    def test_rejects_missing_or_incompatible_host_descriptors(self):
        for values in [
            dict(dst_host_kv_ptrs=[]),
            dict(dst_host_state_data_ptrs=[]),
            dict(requires_dcp_relayout=True),
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                dataclasses.replace(_registration(), **values).for_destination(
                    KVTransferDestination.HOST
                )

    def test_metadata_roundtrip_and_legacy_device_default(self):
        receiver = _receiver()
        receiver.send_metadata(
            np.array([2, 4], dtype=np.int32),
            3,
            [[7, 8]],
            destination=KVTransferDestination.HOST,
        )
        calls = receiver._connect_to_bootstrap_server.return_value[
            0
        ].send_multipart.call_args_list
        first = calls[0].args[0]
        parsed = TransferInfo.from_zmq(first)
        self.assertEqual(parsed.destination, KVTransferDestination.HOST)
        self.assertEqual(parsed.dst_kv_indices.tolist(), [2, 4])
        self.assertEqual(parsed.dst_state_indices, [[7, 8]])
        self.assertTrue(TransferInfo.from_zmq(calls[1].args[0]).is_dummy)
        self.assertEqual(
            TransferInfo.from_zmq(first[:10]).destination, KVTransferDestination.DEVICE
        )
        self.assertEqual(
            TransferInfo.from_zmq(first[:9]).destination, KVTransferDestination.DEVICE
        )
        self.assertTrue(receiver._host_destination_published)

    def test_peer_and_layout_capability(self):
        receiver = _receiver()
        self.assertTrue(receiver.supports_host_destination)
        for attribute, value in [
            ("supports_host_destination", False),
            ("attn_tp_size", 4),
            ("attn_cp_size", 2),
        ]:
            before = getattr(receiver.prefill_info, attribute)
            setattr(receiver.prefill_info, attribute, value)
            self.assertFalse(receiver.supports_host_destination)
            with self.assertRaises(ValueError):
                receiver.send_metadata(
                    np.array([1], dtype=np.int32),
                    destination=KVTransferDestination.HOST,
                )
            setattr(receiver.prefill_info, attribute, before)
        self.assertFalse(receiver._host_destination_published)

    def test_registers_both_host_and_device_regions(self):
        mgr = object.__new__(MooncakeKVManager)
        args = KVArgs()
        args.kv_data_ptrs, args.kv_data_lens = [10], [64]
        args.aux_data_ptrs, args.aux_data_lens = [30], [16]
        args.state_data_ptrs, args.state_data_lens = [[20]], [[32]]
        args.host_kv_data_ptrs, args.host_kv_data_lens = [110], [128]
        args.host_state_data_ptrs, args.host_state_data_lens = [[120]], [[96]]
        mgr.kv_args = args
        self.assertEqual(
            set(mgr._registerable_regions()),
            {(10, 64), (20, 32), (30, 16), (110, 128), (120, 96)},
        )

    def test_abort_arms_before_immediate_ack_and_waits_for_every_sender(self):
        receiver = _receiver()
        self.assertTrue(receiver.is_abort_release_safe())
        receiver._host_destination_published = True
        self.assertFalse(receiver.is_abort_release_safe())
        socket = receiver._connect_to_bootstrap_server.return_value[0]
        rank = [0]

        def acknowledge(_):
            self.assertTrue(receiver._host_drain_armed)
            receiver.kv_mgr.note_abort_ack(17, rank[0])
            rank[0] += 1

        socket.send_multipart.side_effect = acknowledge
        receiver._send_abort_notification()
        self.assertTrue(receiver.is_abort_release_safe())
        # A redundant notification must not wipe previously received ACKs.
        socket.send_multipart.side_effect = None
        receiver._send_abort_notification()
        self.assertTrue(receiver.is_abort_release_safe())
        receiver.kv_mgr._deferred_abort_ack_tracker[17].remove(1)
        self.assertFalse(receiver.is_abort_release_safe())

    def test_real_gloo_capacity_and_swa_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _consensus_worker,
                args=("file://" + os.path.join(directory, "gloo"),),
                nprocs=2,
                join=True,
            )

    def test_failure_exception_keeps_early_and_late_host_drain_acks(self):
        receiver = _receiver()
        receiver._host_destination_published = True
        receiver._host_drain_armed = True
        receiver.conclude_state = None
        receiver.bootstrap_addr = "prefill"
        mgr = receiver.kv_mgr
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {17: "test transfer failure"}
        mgr.request_status = {17: 0}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}
        mgr.addr_to_rooms_tracker = defaultdict(set, {"prefill": {17}})
        mgr.register_deferred_abort_room(17)
        mgr.note_abort_ack(17, 0)
        with self.assertRaisesRegex(KVTransferError, "test transfer failure"):
            receiver.failure_exception()
        self.assertEqual(mgr._deferred_abort_ack_tracker[17], {0})
        self.assertFalse(receiver.is_abort_release_safe())
        mgr.note_abort_ack(17, 1)
        self.assertTrue(receiver.is_abort_release_safe())
        # This is the queue's final clear, after all remote writes drained.
        receiver.clear()
        self.assertNotIn(17, mgr._deferred_abort_ack_tracker)
        self.assertNotIn(17, mgr.request_status)


if __name__ == "__main__":
    unittest.main()
