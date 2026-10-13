"""Exercise the FA3 build entry point without a CUDA compiler or GPU."""

import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[4]
AOT = REPO_ROOT / "python/sglang/kernels/aot"
SCRIPT = AOT / "build_fa.sh"

spec = importlib.util.spec_from_file_location(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBuildFA(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fa3 build test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.build = self.root / "build"
        self.dist = self.root / "dist"
        self.trace = self.root / "calls.jsonl"
        self.python = self.bin / "selected python"
        self.python.symlink_to(sys.executable)
        self.nvcc = self.root / "cuda 12.9/bin/nvcc"
        self.nvcc.parent.mkdir(parents=True)
        self.write_executable(
            self.nvcc,
            "#!/bin/sh\necho 'Cuda compilation tools, release 12.9, V12.9.86'\n",
        )
        # Only the native compiler and Torch loading boundary are replaced.
        # The real shell entry point, log handling and wheel packager run below.
        self.write_executable(
            self.bin / "cmake",
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                r"""
                import json, os, subprocess, sys
                from pathlib import Path
                args = sys.argv[1:]
                with open(os.environ['FA3_TEST_TRACE'], 'a') as f:
                    f.write(json.dumps(['cmake', *args]) + '\n')
                stage = 'build' if '--build' in args else 'configure'
                print('[mock ' + stage + ']', flush=True)
                if os.environ.get('FA3_TEST_FAIL') == stage:
                    sys.exit(23)
                if stage == 'configure':
                    build = Path(args[args.index('-B') + 1])
                    build.mkdir(parents=True, exist_ok=True)
                    info = json.loads(subprocess.check_output([
                        os.environ['FA3_PYTHON'], os.environ['FA3_TEST_PACKAGE'], '--describe'
                    ], text=True))
                    info.update(nvcc='12.9.86', gpu_arch='sm_90a')
                    if os.environ.get('FA3_TEST_OLD_TORCH'):
                        info['torch_version'] = '2.12.0+cu130'
                    (build / 'fa3-build-info.json').write_text(json.dumps(info))
                else:
                    build = Path(args[args.index('--build') + 1])
                    (build / 'lib').mkdir(exist_ok=True)
                    (build / 'lib/flash_ops.abi3.so').write_bytes(b'fake FA3 native binary')
                    print('[1/1] Linking flash_ops.abi3.so', flush=True)
                """
            ).lstrip(),
        )
        modules = self.root / "modules"
        modules.mkdir()
        (modules / "torch.py").write_text(
            textwrap.dedent(
                r"""
                import json, os
                from types import SimpleNamespace as NS
                __version__ = '2.13.0+cu130'
                version = NS(cuda='13.0')
                def compiled_with_cxx11_abi():
                    return True
                def load_library(path):
                    with open(os.environ['FA3_TEST_TRACE'], 'a') as f:
                        f.write(json.dumps(['load_library', path]) + '\n')
                    if os.environ.get('FA3_TEST_FAIL') == 'load':
                        raise OSError('undefined symbol: torch::Library::_def')
                count = 35 if os.environ.get('FA3_TEST_FAIL') == 'schema' else 37
                names = [f'arg{i}' for i in range(count - 3)] + ['sinks', 'sparse_mask_fine', 'only_qv']
                schema = NS(arguments=[NS(name=name) for name in names])
                ops = NS(load_library=load_library, sgl_kernel=NS(
                    fwd=NS(default=NS(_schema=schema)),
                    get_scheduler_metadata=NS(default=NS(_schema='scheduler schema')),
                ))
                """
            ).lstrip()
        )
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(modules),
            "FA3_NVCC": str(self.nvcc),
            "FA3_PYTHON": str(self.python),
            "FA3_BUILD_DIR": str(self.build),
            "FA3_DIST_DIR": str(self.dist),
            "MAX_JOBS": "2",
            "FA3_TEST_TRACE": str(self.trace),
            "FA3_TEST_PACKAGE": str(AOT / "cmake/fa3/package_fa.py"),
        }

    @staticmethod
    def write_executable(path, source):
        path.write_text(source)
        path.chmod(0o755)

    def run_build(self, *args, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=self.root,
            env={**self.env, **env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )

    def calls(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_build_uses_selected_toolchain_and_packages_only_fa3(self):
        result = self.run_build("-DFA3_CUTLASS_URL=/cache/cutlass.tar.gz")
        self.assertEqual(result.returncode, 0, result.stdout)
        configure, build, load = self.calls()
        self.assertIn(f"-DPython_EXECUTABLE={self.python}", configure)
        self.assertIn(f"-DCMAKE_CUDA_COMPILER={self.nvcc}", configure)
        self.assertIn(f"-DCUDAToolkit_ROOT={self.nvcc.parent.parent}", configure)
        self.assertIn("-DFA3_CUTLASS_URL=/cache/cutlass.tar.gz", configure)
        self.assertEqual(build[build.index("--target") + 1], "flash_ops")
        self.assertEqual(build[build.index("--parallel") + 1], "2")
        self.assertEqual(
            load, ["load_library", str(self.build / "lib/flash_ops.abi3.so")]
        )
        self.assertIn("[1/1] Linking", (self.build / "build.log").read_text())
        library = self.dist / "flash_ops.abi3.so"
        manifest = json.loads((self.dist / "manifest.json").read_text())
        self.assertEqual(manifest["torch_version"], "2.13.0+cu130")
        self.assertEqual(
            manifest["sha256"], hashlib.sha256(library.read_bytes()).hexdigest()
        )
        self.assertEqual(manifest["gpu_validation"], "not_run")
        wheels = list(self.dist.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        with ZipFile(wheels[0]) as wheel:
            native_files = [p for p in wheel.namelist() if ".dist-info/" not in p]
            self.assertEqual(native_files, ["sgl_kernel/flash_ops.abi3.so"])
            self.assertEqual(wheel.read(native_files[0]), library.read_bytes())
            metadata_path = next(p for p in wheel.namelist() if p.endswith("/METADATA"))
            self.assertIn(
                "Requires-Dist: torch==2.13.0+cu130", wheel.read(metadata_path).decode()
            )

    def test_failed_build_does_not_load_or_publish(self):
        for stage in ("configure", "build"):
            with self.subTest(stage=stage):
                result = self.run_build(FA3_TEST_FAIL=stage)
                self.assertEqual(result.returncode, 23, result.stdout)
                self.assertFalse(list(self.dist.glob("*.whl")))
                self.assertFalse((self.dist / "flash_ops.abi3.so").exists())
                self.assertFalse(
                    any(call[0] == "load_library" for call in self.calls())
                )
                self.assertIn(
                    "[mock " + stage + "]", (self.build / "build.log").read_text()
                )

    def test_failed_native_checks_preserve_previous_artifacts(self):
        self.dist.mkdir()
        previous = self.dist / "flash_ops.abi3.so"
        previous.write_bytes(b"previous successful build")
        for stage in ("load", "schema"):
            with self.subTest(stage=stage):
                result = self.run_build(FA3_TEST_FAIL=stage)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                expected = (
                    "undefined symbol"
                    if stage == "load"
                    else "Unexpected 0520 FA3 fwd schema"
                )
                self.assertIn(expected, result.stdout)
                self.assertEqual(previous.read_bytes(), b"previous successful build")
                self.assertFalse(list(self.dist.glob("*.whl")))
                self.assertFalse((self.dist / "manifest.json").exists())

    def test_changed_torch_environment_is_rejected_before_loading(self):
        result = self.run_build(FA3_TEST_OLD_TORCH="1")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("torch_version", result.stdout)
        self.assertFalse(any(call[0] == "load_library" for call in self.calls()))

    def test_missing_compiler_and_bad_job_count_fail_before_configuring(self):
        for env in ({"FA3_NVCC": "/missing/nvcc"}, {"MAX_JOBS": "0"}):
            with self.subTest(env=env):
                result = self.run_build(**env)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(self.calls(), [])

    def test_cuda13_compiler_is_rejected_before_configuring(self):
        self.write_executable(
            self.nvcc,
            "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.0'\n",
        )
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("12.9", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_concurrent_build_is_rejected(self):
        self.build.mkdir()
        with (self.build / ".build_fa.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_build()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("already", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_help_does_not_require_cuda(self):
        result = self.run_build("--help", FA3_NVCC="/missing/nvcc")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("FA3_NVCC", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_relative_python_path_and_existing_build_files_are_preserved(self):
        self.build.mkdir()
        previous_object = self.build / "existing-object.o"
        previous_object.write_bytes(b"incremental build cache")
        result = self.run_build(FA3_PYTHON="./bin/selected python")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"-DPython_EXECUTABLE={self.python}", self.calls()[0])
        self.assertEqual(previous_object.read_bytes(), b"incremental build cache")


if __name__ == "__main__":
    unittest.main()
