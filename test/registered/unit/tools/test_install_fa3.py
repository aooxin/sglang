"""Exercise mounted FA3 installation without touching the real Torch environment."""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALLER = REPO_ROOT / "docker/install_fa3.py"
REPLACEMENT = b"FA3 built for Torch 2.13 with CUDA 12.9"

spec = importlib.util.spec_from_file_location(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestInstallFA3(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fa3 install test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "cloud disk"
        self.source_dir.mkdir()
        self.source = self.source_dir / "flash_ops.abi3.so"
        self.source.write_bytes(REPLACEMENT)
        self.modules = self.root / "site-packages"
        self.package = self.modules / "sgl_kernel"
        self.package.mkdir(parents=True)
        # Finding the extension must not initialize the installed package/GPU.
        (self.package / "__init__.py").write_text(
            "raise RuntimeError('sgl_kernel imported before replacement')\n"
        )
        self.target = self.package / (
            "flash_ops" + importlib.machinery.EXTENSION_SUFFIXES[0]
        )
        self.target.write_bytes(b"original active FA3")
        self.target.chmod(0o640)
        self.backup = self.target.with_name(self.target.name + ".before_fa3_cu129")
        self.fallback = self.package / "flash_ops.abi3.so"
        self.fallback.write_bytes(b"lower priority extension")
        self.wrapper = self.package / "flash_attn.py"
        self.wrapper.write_text("# existing 0520 wrapper\n")
        # Only the unavailable Torch 2.13/CUDA loading boundary is simulated.
        # Discovery, staging, hashing, backups and replacement use real files.
        (self.modules / "torch.py").write_text(
            textwrap.dedent(
                """\
                import os
                from pathlib import Path
                from types import SimpleNamespace as NS
                __version__ = os.environ.get('FA3_TEST_TORCH', '2.13.0+cu130')
                version = NS(cuda=os.environ.get('FA3_TEST_CUDA', '13.0'))
                def compiled_with_cxx11_abi():
                    return os.environ.get('FA3_TEST_ABI', '1') == '1'
                def load_library(path):
                    if os.environ.get('FA3_TEST_LOAD_FAIL'):
                        raise OSError('undefined symbol: torch::Library::_def')
                    assert Path(path).read_bytes() == b'FA3 built for Torch 2.13 with CUDA 12.9'
                count = int(os.environ.get('FA3_TEST_SCHEMA', '37'))
                names = [f'arg{i}' for i in range(count - 3)] + ['sinks', 'sparse_mask_fine', 'only_qv']
                schema = NS(arguments=[NS(name=name) for name in names])
                ops = NS(load_library=load_library, sgl_kernel=NS(
                    fwd=NS(default=NS(_schema=schema)),
                    get_scheduler_metadata=NS(default=NS(_schema='scheduler schema')),
                ))
                """
            )
        )
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.modules),
            "FA3_ARTIFACT_DIR": str(self.source_dir),
            "FA3_ARTIFACT_SHA256": hashlib.sha256(REPLACEMENT).hexdigest(),
        }

    def run_install(self, **env):
        return subprocess.run(
            [sys.executable, str(INSTALLER)],
            env={**self.env, **env},
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

    def assert_original_intact(self):
        self.assertEqual(self.target.read_bytes(), b"original active FA3")
        self.assertFalse(self.backup.exists())

    def test_installs_the_active_extension_and_preserves_original_files(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.target.read_bytes(), REPLACEMENT)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.backup.read_bytes(), b"original active FA3")
        self.assertEqual(self.fallback.read_bytes(), b"lower priority extension")
        self.assertEqual(self.wrapper.read_text(), "# existing 0520 wrapper\n")
        self.assertEqual(self.source.read_bytes(), REPLACEMENT)

    def test_missing_artifact_and_bad_checksum_preserve_the_installed_library(self):
        self.source.unlink()
        missing = self.run_install()
        self.assertNotEqual(missing.returncode, 0, missing.stdout)
        self.assertIn(str(self.source), missing.stdout)
        self.assert_original_intact()
        self.source.write_bytes(b"incomplete cloud copy")
        corrupt = self.run_install()
        self.assertNotEqual(corrupt.returncode, 0, corrupt.stdout)
        self.assertIn("SHA256", corrupt.stdout)
        self.assert_original_intact()

    def test_native_or_abi_failures_leave_the_installed_library_intact(self):
        cases = (
            ({"FA3_TEST_LOAD_FAIL": "1"}, "undefined symbol"),
            ({"FA3_TEST_SCHEMA": "35"}, "schema"),
            ({"FA3_TEST_TORCH": "2.8.0+cu128"}, "requires Torch"),
            ({"FA3_TEST_CUDA": "12.8"}, "requires Torch"),
            ({"FA3_TEST_ABI": "0"}, "requires Torch"),
        )
        for env, error in cases:
            with self.subTest(env=env):
                result = self.run_install(**env)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(error, result.stdout)
                self.assert_original_intact()

    def test_repeated_installation_reuses_the_file_and_original_backup(self):
        first = self.run_install()
        self.assertEqual(first.returncode, 0, first.stdout)
        inode = self.target.stat().st_ino
        second = self.run_install()
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertEqual(self.target.stat().st_ino, inode)
        self.assertEqual(self.backup.read_bytes(), b"original active FA3")
        # Even an already-installed artifact must load in the current runtime.
        failed = self.run_install(FA3_TEST_LOAD_FAIL="1")
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertEqual(self.target.read_bytes(), REPLACEMENT)

    def test_concurrent_installations_preserve_one_original_backup(self):
        with ExitStack() as stack:
            processes = [
                stack.enter_context(
                    subprocess.Popen(
                        [sys.executable, str(INSTALLER)],
                        cwd=self.root,
                        env=self.env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                )
                for _ in range(2)
            ]
            for process in processes:
                output, _ = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, output)
        self.assertEqual(self.target.read_bytes(), REPLACEMENT)
        self.assertEqual(self.backup.read_bytes(), b"original active FA3")

    def test_fa3_entrypoints_install_before_launch_and_stop_on_failure(self):
        binary_dir = self.root / "bin"
        binary_dir.mkdir()
        python = binary_dir / "python3"
        # Run the real installer; replace only the long-running GPU server.
        python.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                import json, os, sys
                from pathlib import Path
                if sys.argv[1:] == ['/opt/sglang/fa3/install.py']:
                    os.execv(sys.executable, [sys.executable, os.environ['FA3_TEST_INSTALLER']])
                if sys.argv[1:3] == ['-m', 'sglang.launch_server']:
                    Path(os.environ['FA3_TEST_SERVER']).write_text(json.dumps({
                        'args': sys.argv[3:],
                        'library': Path(os.environ['FA3_TEST_TARGET']).read_text(),
                    }))
                    sys.exit(0)
                sys.exit('unexpected Python invocation: ' + repr(sys.argv))
                """
            )
        )
        python.chmod(0o755)
        server_trace = self.root / "server.json"
        env = {
            **self.env,
            "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
            "FA3_TEST_INSTALLER": str(INSTALLER),
            "FA3_TEST_TARGET": str(self.target),
            "FA3_TEST_SERVER": str(server_trace),
            "LWS_GROUP_SIZE": "1",
            "LWS_WORKER_INDEX": "1",
            "LWS_LEADER_IP": "127.0.0.1",
            "HPN_IB_HCA": "mlx5_0",
        }
        for script in ("start_p.sh", "start_d.sh", "start_26p.sh", "start_26d.sh"):
            self.target.write_bytes(b"original active FA3")
            self.backup.unlink(missing_ok=True)
            for valid in (True, False):
                with self.subTest(script=script, valid=valid):
                    server_trace.unlink(missing_ok=True)
                    self.source.write_bytes(REPLACEMENT)
                    if not valid:
                        self.source.unlink()
                    result = subprocess.run(
                        ["bash", str(REPO_ROOT / script)],
                        cwd=self.root,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=30,
                    )
                    if valid:
                        self.assertEqual(result.returncode, 0, result.stdout)
                        trace = json.loads(server_trace.read_text())
                        self.assertEqual(trace["library"], REPLACEMENT.decode())
                        self.assertIn("fa3", trace["args"])
                    else:
                        self.assertNotEqual(result.returncode, 0, result.stdout)
                        self.assertFalse(server_trace.exists())


if __name__ == "__main__":
    unittest.main()
