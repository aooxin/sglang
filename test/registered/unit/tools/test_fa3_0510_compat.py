"""Compile the real FA3 binding against a 0510-signature CPU test backend."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
AOT = REPO_ROOT / "python/sglang/kernels/aot"

spec = importlib.util.spec_from_file_location(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
ci_register.register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def check_native_adapter(build_dir):
    import torch
    from torch.utils.cpp_extension import load

    load(
        name="fa3_0510_native_test",
        sources=[
            str(AOT / "csrc/flash_extension.cc"),
            str(Path(__file__).with_name("fa3_0510_legacy_stub.cc")),
        ],
        extra_include_paths=[str(AOT / "include")],
        extra_cflags=["-O0", "-g0"],
        build_directory=build_dir,
        with_cuda=False,
        is_python_module=False,
    )
    op = torch.ops.sgl_kernel.fwd.default
    names = [arg.name for arg in op._schema.arguments]
    assert len(names) == 37, op._schema
    assert names[-3:] == ["sinks", "sparse_mask_fine", "only_qv"], op._schema
    out_arg = op._schema.arguments[names.index("out")]
    assert out_arg.alias_info.is_write
    assert op._schema.returns[0].alias_info.after_set == out_arg.alias_info.after_set

    q = torch.tensor([1.0, 2.0])
    k = torch.tensor([3.0, 4.0])
    v = torch.tensor([5.0, 6.0])
    out = torch.empty_like(q)
    metadata = torch.tensor([11], dtype=torch.int32)
    sinks = torch.tensor([13.0])
    args = dict.fromkeys(names)
    args.update(
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_scale=0.5,
        is_causal=True,
        window_size_left=7,
        window_size_right=0,
        attention_chunk=4,
        softcap=1.5,
        is_rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=3,
        pack_gqa=True,
        sm_margin=2,
        sinks=sinks,
        only_qv=False,
    )
    result = op(**args)
    assert len(result) == 4
    torch.testing.assert_close(result[0], torch.tensor([8.5, 11.0]))
    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == q.data_ptr()
    assert result[2].data_ptr() == metadata.data_ptr()
    assert result[3].data_ptr() == sinks.data_ptr()

    optional_result = op(
        **{
            **args,
            "out": None,
            "softmax_scale": None,
            "scheduler_metadata": None,
            "sinks": None,
        }
    )
    torch.testing.assert_close(optional_result[0], torch.tensor([9.0, 12.0]))
    assert optional_result[2].data_ptr() == k.data_ptr()
    assert optional_result[3].data_ptr() == v.data_ptr()

    for name, value in (
        ("only_qv", True),
        ("sparse_mask_fine", torch.zeros((1, 1, 32), dtype=torch.int32)),
    ):
        out.fill_(-99.0)
        try:
            op(**{**args, name: value})
        except RuntimeError as error:
            assert "0510" in str(error) and name in str(error), str(error)
        else:
            raise AssertionError(f"Unsupported {name} reached the legacy backend")
        torch.testing.assert_close(out, torch.tensor([-99.0, -99.0]))

    metadata_result = torch.ops.sgl_kernel.get_scheduler_metadata.default(
        1,
        1,
        11,
        4,
        2,
        192,
        128,
        torch.bfloat16,
        metadata,
        None,
        None,
        None,
        None,
        None,
        64,
    )
    torch.testing.assert_close(metadata_result, metadata)
    print(
        "Native adapter passed: 37-argument schema, output aliasing, forwarding, optional arguments, feature guards, metadata binding"
    )


class TestFA30510Compatibility(unittest.TestCase):
    def test_native_binding_accepts_0510_backend(self):
        # Isolate the sgl_kernel operator registrations from the test runner and
        # any installed extension. Only the old CUDA implementation is replaced.
        with tempfile.TemporaryDirectory(prefix="fa3_0510_native_") as build_dir:
            result = subprocess.run(
                [sys.executable, __file__, "--native-check", build_dir],
                cwd=REPO_ROOT,
                env={**os.environ, "MAX_JOBS": "1"},
                capture_output=True,
                text=True,
                timeout=180,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--native-check":
        check_native_adapter(sys.argv[2])
    else:
        unittest.main()
