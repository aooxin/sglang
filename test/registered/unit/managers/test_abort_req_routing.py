"""Unit tests for scheduler -> tokenizer abort routing — no server.

Scheduler-side aborts (prefetch-drop 429, queue-full, waiting-timeout, priority
rejection) send a bare AbortReq through ``SenderWrapper.send_output(abort_req, req)``
where ``req`` is a scheduler-side ``Req``. ``Req`` carries ``http_worker_ipc`` but is
not a ``BaseReq`` subclass, so an ``isinstance(recv_obj, BaseReq)`` gate skipped the
copy and the AbortReq left with ``http_worker_ipc=None``. Under
``--tokenizer-worker-num > 1`` the router then drops it (``SocketMapping`` sees a
None ipc), so the client hangs with no 429 at all.

Covers the three halves of the fix: ipc propagation in SenderWrapper, the rid on the
drop warning, and _handle_abort_req tolerating a rid_to_state miss.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.managers.io_struct import AbortReq, BaseBatchReq, BaseReq
from sglang.srt.managers.multi_tokenizer_mixin import SocketMapping
from sglang.srt.managers.scheduler import SenderWrapper
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

SOCKET_LOGGER = "sglang.srt.managers.multi_tokenizer_mixin"
TOKENIZER_LOGGER = "sglang.srt.managers.tokenizer_manager"


class _FakeReq:
    """Scheduler-side Req stand-in: has http_worker_ipc, is not a BaseReq."""

    def __init__(self, rid, http_worker_ipc):
        self.rid = rid
        self.http_worker_ipc = http_worker_ipc


class _FakeBatchOutput(BaseBatchReq):
    """Minimal BaseBatchReq: routes by http_worker_ipcs, never the singular."""


class TestSenderWrapperIpcPropagation(CustomTestCase):
    def _send(self, output, recv_obj):
        socket = MagicMock()
        SenderWrapper(socket).send_output(output, recv_obj)
        socket.send_pyobj.assert_called_once_with(output)
        return output

    def test_recv_obj_is_not_a_base_req(self):
        # Documents the precondition the old isinstance() gate relied on: a
        # scheduler-side Req carries the ipc but fails the BaseReq check.
        req = _FakeReq("rid-1", "ipc://worker-3")
        self.assertNotIsInstance(req, BaseReq)
        self.assertEqual(req.http_worker_ipc, "ipc://worker-3")

    def test_ipc_copied_from_non_base_req_recv_obj(self):
        # The regression: a Req-typed recv_obj must still propagate its ipc,
        # otherwise the multi-http-worker router drops the abort and the client
        # never sees the 429.
        out = AbortReq(rid="rid-1")
        self.assertIsNone(out.http_worker_ipc)
        self._send(out, _FakeReq("rid-1", "ipc://worker-3"))
        self.assertEqual(out.http_worker_ipc, "ipc://worker-3")

    def test_ipc_copied_from_base_req_recv_obj(self):
        out = AbortReq(rid="rid-1")
        self._send(out, AbortReq(rid="rid-1", http_worker_ipc="ipc://worker-3"))
        self.assertEqual(out.http_worker_ipc, "ipc://worker-3")

    def test_existing_ipc_not_overwritten(self):
        out = AbortReq(rid="rid-1", http_worker_ipc="ipc://worker-9")
        self._send(out, _FakeReq("rid-1", "ipc://worker-3"))
        self.assertEqual(out.http_worker_ipc, "ipc://worker-9")

    def test_recv_obj_without_ipc_is_noop(self):
        out = AbortReq(rid="rid-1")
        self._send(out, AbortReq(rid="rid-1"))
        self.assertIsNone(out.http_worker_ipc)

    def test_none_recv_obj_is_noop(self):
        # Batch outputs are sent with no recv_obj; they already carry
        # http_worker_ipcs and must stay untouched.
        out = _FakeBatchOutput(rids=["rid-1"], http_worker_ipcs=["ipc://worker-3"])
        self._send(out, None)
        self.assertEqual(out.http_worker_ipcs, ["ipc://worker-3"])

    def test_batch_output_gets_no_stray_singular_ipc(self):
        # BaseBatchReq routes by http_worker_ipcs; setting a singular
        # http_worker_ipc on it would be dead weight at best and a routing
        # hazard at worst.
        out = _FakeBatchOutput(rids=["rid-1"], http_worker_ipcs=["ipc://worker-3"])
        self._send(out, _FakeReq("rid-1", "ipc://worker-1"))
        self.assertFalse(hasattr(out, "http_worker_ipc"))
        self.assertEqual(out.http_worker_ipcs, ["ipc://worker-3"])

    def test_socket_none_sends_nothing(self):
        out = AbortReq(rid="rid-1")
        SenderWrapper(None).send_output(out, _FakeReq("rid-1", "ipc://worker-3"))
        self.assertIsNone(out.http_worker_ipc)


class TestSocketMappingDropWarning(CustomTestCase):
    def _make_mapping(self):
        mapping = SocketMapping.__new__(SocketMapping)
        mapping._zmq_context = MagicMock()
        mapping._mapping = {}
        return mapping

    def test_none_ipc_warning_carries_rid(self):
        # If a drop ever happens again, the warning has to name the request or
        # there is nothing to correlate with the client's hang.
        mapping = self._make_mapping()
        with self.assertLogs(SOCKET_LOGGER, level="WARNING") as cm:
            mapping.send_output(None, AbortReq(rid="rid-42"))
        self.assertEqual(mapping._mapping, {})
        self.assertIn("IPC name is None", cm.output[0])
        self.assertIn("rid=rid-42", cm.output[0])
        self.assertIn("skipping", cm.output[0])

    def test_none_ipc_warning_carries_batch_rids(self):
        mapping = self._make_mapping()
        out = _FakeBatchOutput(rids=["rid-a", "rid-b"], http_worker_ipcs=[None, None])
        with self.assertLogs(SOCKET_LOGGER, level="WARNING") as cm:
            mapping.send_output(None, out)
        self.assertIn("rid-a", cm.output[0])
        self.assertIn("rid-b", cm.output[0])

    def test_valid_ipc_sends_without_warning(self):
        mapping = self._make_mapping()
        mapping._mapping["ipc://worker-3"] = MagicMock()
        with self.assertNoLogs(SOCKET_LOGGER, level="WARNING"):
            mapping.send_output("ipc://worker-3", AbortReq(rid="rid-1"))
        mapping._mapping["ipc://worker-3"].send_pyobj.assert_called_once()


class TestHandleAbortReq(CustomTestCase):
    def _make_tm(self):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.rid_to_state = {}
        tm.server_args = MagicMock()
        return tm

    def _make_state(self, *, stream=False, return_logprob=False):
        state = MagicMock()
        state.finished = False
        state.text = ""
        state.output_ids = [11, 22]
        state.out_list = []
        state.obj = MagicMock(stream=stream, return_logprob=return_logprob)
        state.time_stats.get_e2e_latency.return_value = 0.5
        return state

    def test_unknown_rid_logs_error_and_returns(self):
        # A bare rid_to_state[rid] raises KeyError here, and handle_loop is
        # wrapped in print_exception_wrapper, which kills the whole tokenizer
        # process tree. A stale abort must be a no-op instead.
        tm = self._make_tm()
        with self.assertLogs(TOKENIZER_LOGGER, level="ERROR") as cm:
            TokenizerManager._handle_abort_req(tm, AbortReq(rid="rid-gone"))
        self.assertIn("rid-gone", cm.output[0])
        self.assertEqual(tm.rid_to_state, {})

    def test_known_rid_marks_finished_and_emits(self):
        tm = self._make_tm()
        state = self._make_state()
        tm.rid_to_state["rid-1"] = state
        reason = {"type": "abort", "message": "prefetch retry budget exhausted"}
        TokenizerManager._handle_abort_req(
            tm, AbortReq(rid="rid-1", finished_reason=reason)
        )
        self.assertTrue(state.finished)
        state.event.set.assert_called_once()
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(state.out_list[0]["meta_info"]["finish_reason"], reason)
        self.assertEqual(state.out_list[0]["meta_info"]["completion_tokens"], 2)

    def test_health_check_rid_is_ignored(self):
        tm = self._make_tm()
        with self.assertNoLogs(TOKENIZER_LOGGER, level="ERROR"):
            TokenizerManager._handle_abort_req(
                tm, AbortReq(rid=f"{HEALTH_CHECK_RID_PREFIX}x")
            )
        self.assertEqual(tm.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
