import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.utils import (
    collect_kv_transfer_infos,
    get_kv_transfer_buf_infos,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestKVTransferMetadata(unittest.TestCase):
    @staticmethod
    def pool(layout, descriptors):
        pool = SimpleNamespace(kv_transfer_layout=layout)
        pool.get_contiguous_buf_infos = Mock(return_value=descriptors)
        pool.get_page_level_buf_infos = Mock(return_value=descriptors)
        return pool

    def test_layout_dispatch(self):
        for layout in ("layer_first", "page_first"):
            pool = self.pool(layout, ([1], [16], [4]))
            self.assertEqual(get_kv_transfer_buf_infos(pool), ([1], [16], [4], layout))
            selected = (
                pool.get_page_level_buf_infos
                if layout == "page_first"
                else pool.get_contiguous_buf_infos
            )
            other = (
                pool.get_contiguous_buf_infos
                if layout == "page_first"
                else pool.get_page_level_buf_infos
            )
            selected.assert_called_once_with()
            other.assert_not_called()

    def test_draft_order_precedence_and_descriptor_ownership(self):
        main = ([1], [16], [4], "page_first")
        drafts = [self.pool("page_first", ([i], [i * 16], [i * 4])) for i in (2, 3)]
        ignored = self.pool("layer_first", ([99], [99], [99]))
        actual = collect_kv_transfer_infos(main, drafts, ignored)
        self.assertEqual(actual, ([1, 2, 3], [16, 32, 48], [4, 8, 12], "page_first"))
        self.assertEqual(main[:3], ([1], [16], [4]))
        self.assertEqual(drafts[0].get_page_level_buf_infos(), ([2], [32], [8]))
        ignored.get_contiguous_buf_infos.assert_not_called()
        # Repeated collection must not accumulate earlier draft descriptors.
        self.assertEqual(collect_kv_transfer_infos(main, drafts, None), actual)

    def test_host_main_single_draft_and_no_draft(self):
        # Decode's HiSparse descriptors come from a host pool, not a KVCache.
        main = ([101], [64], [8], "layer_first")
        draft = self.pool("layer_first", ([202], [32], [4]))
        self.assertEqual(
            collect_kv_transfer_infos(main, [], draft),
            ([101, 202], [64, 32], [8, 4], "layer_first"),
        )
        copied = collect_kv_transfer_infos(main, [], None)
        copied[0].append(303)
        self.assertEqual(main[0], [101])

    def test_mismatch_leaves_input_descriptors_unchanged(self):
        main = ([1], [16], [4], "layer_first")
        draft = self.pool("page_first", ([2], [32], [8]))
        with self.assertRaisesRegex(AssertionError, "must match"):
            collect_kv_transfer_infos(main, [], draft)
        self.assertEqual(main, ([1], [16], [4], "layer_first"))

    def test_kv_args_default_layout(self):
        self.assertEqual(KVArgs().kv_transfer_layout, "layer_first")


class TestKVRegistrationCompatibility(unittest.TestCase):
    @staticmethod
    def legacy_message():
        # Literal wire positions cover compatibility with older senders.
        return [
            b"17",
            b"127.0.0.1",
            b"1234",
            b"session",
            struct.pack("2Q", 100, 200),
            struct.pack("Q", 300),
            b"",
            b"1",
            b"2",
            b"4096",
            struct.pack("I", 128),
            struct.pack("I", 2),
            b"1",
            b"1",
        ]

    def test_legacy_and_layout_messages_with_and_without_staging(self):
        for layout in (None, "layer_first", "page_first"):
            for staging in ([], [b"", b""], [struct.pack("Q", 123456), b"8192"]):
                with self.subTest(layout=layout, staging=bool(staging)):
                    msg = self.legacy_message()
                    if layout is not None:
                        msg.append(layout.encode("ascii"))
                    msg.extend(staging)
                    info = KVArgsRegisterInfo.from_zmq(msg)
                    self.assertEqual(info.dst_kv_ptrs, [100, 200])
                    self.assertEqual(info.dst_aux_ptrs, [300])
                    self.assertEqual(info.dst_state_data_ptrs, [])
                    self.assertEqual(info.dst_state_item_lens, [128])
                    self.assertEqual(info.dst_state_dim_per_tensor, [2])
                    self.assertTrue(info.enable_hisparse)
                    self.assertTrue(info.enable_pd_direct2host)
                    self.assertEqual(info.dst_kv_layout, layout or "layer_first")
                    if staging and staging[0]:
                        self.assertEqual(
                            (info.staging.base_ptr, info.staging.total_size),
                            (123456, 8192),
                        )
                    else:
                        self.assertIsNone(info.staging)

    def test_older_messages_omit_optional_fields(self):
        for count in range(10, 14):
            with self.subTest(count=count):
                info = KVArgsRegisterInfo.from_zmq(self.legacy_message()[:count])
                self.assertEqual(info.dst_kv_layout, "layer_first")
                self.assertIsNone(info.staging)
                self.assertFalse(info.enable_pd_direct2host)
                self.assertEqual(info.enable_hisparse, count == 13)

    def test_failure_state_is_set_before_notifying_decode(self):
        manager = Mock()
        transfer_info = SimpleNamespace(endpoint="127.0.0.1", dst_port=1234, room=19)
        MooncakeKVManager._fail_transfer_and_sync(
            manager, 17, transfer_info, "layout mismatch", 3
        )
        self.assertEqual(
            manager.mock_calls,
            [
                call.record_failure(17, "layout mismatch"),
                call.update_status(17, KVPoll.Failed),
                call.sync_status_to_decode_endpoint(
                    "127.0.0.1", 1234, 19, KVPoll.Failed, 3
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
