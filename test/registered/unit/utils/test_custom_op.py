"""Unit tests for register_custom_op's eager direct-call path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import itertools
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import sglang_lib
from sglang.srt.utils.custom_op import register_custom_op
from sglang.test.test_utils import CustomTestCase

_op_ids = itertools.count()


def _unique_name(prefix: str) -> str:
    # torch.ops registrations are process-global, so every case needs a fresh op.
    return f"{prefix}_{next(_op_ids)}"


class TestCustomOpEagerDirectCall(CustomTestCase):
    def _define_cuda_only_add_one(self):
        def add_one_(x: torch.Tensor) -> None:
            x.add_(1)

        return register_custom_op(
            add_one_, op_name=_unique_name("test_eager_add_one"), mutates_args=["x"]
        )

    def test_eager_call_skips_dispatcher_when_enabled(self):
        """The op is registered for the CUDA key only; a CPU tensor reaching the
        dispatcher fails, so succeeding proves the eager call bypassed it."""
        with envs.SGLANG_OPT_CUSTOM_OP_EAGER_DIRECT_CALL.override(True):
            op = self._define_cuda_only_add_one()
        x = torch.zeros(4)
        op(x)
        self.assertTrue(torch.equal(x, torch.ones(4)))

    def test_eager_call_goes_through_dispatcher_by_default(self):
        with envs.SGLANG_OPT_CUSTOM_OP_EAGER_DIRECT_CALL.override(False):
            op = self._define_cuda_only_add_one()
        with self.assertRaises(NotImplementedError):
            op(torch.zeros(4))

    def test_compile_tracing_keeps_op_opaque(self):
        """Dynamo must record the torch.ops node instead of inlining the Python
        impl: the impl graph-breaks, which fails a fullgraph compile if inlined."""
        name = _unique_name("test_compile_opaque_add_one")

        def add_one_(x: torch.Tensor) -> None:
            torch._dynamo.graph_break()
            x.add_(1)

        with envs.SGLANG_OPT_CUSTOM_OP_EAGER_DIRECT_CALL.override(True):
            op = register_custom_op(add_one_, op_name=name, mutates_args=["x"])
        # Let the compiled graph's torch.ops call run on CPU in this test.
        sglang_lib.impl(name, add_one_, "CPU")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            op(x)
            return x * 2

        torch._dynamo.reset()
        out = fn(torch.zeros(4))
        self.assertTrue(torch.equal(out, torch.full((4,), 2.0)))


if __name__ == "__main__":
    unittest.main()
