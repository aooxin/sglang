import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestReqPrefetchRounds(unittest.TestCase):
    def _bare_req(self):
        req = Req.__new__(Req)
        # The only two attributes reset_for_retract reads (all others are pure
        # assignments); everything else it writes.
        req.retraction_count = 0
        req.input_embeds = None
        return req

    def test_reset_for_retract_zeroes_rounds(self):
        req = self._bare_req()
        req.prefetch_rate_limited_rounds = 5
        Req.reset_for_retract(req)
        self.assertEqual(req.prefetch_rate_limited_rounds, 0)
        # sanity: the method actually ran (existing behavior)
        self.assertEqual(req.retraction_count, 1)


if __name__ == "__main__":
    unittest.main()
