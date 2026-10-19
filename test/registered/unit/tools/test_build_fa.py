"""Check pip-wheel orchestration without compiling CUDA or running GPU kernels."""

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[4]
AOT = REPO_ROOT / "python/sglang/kernels/aot"

spec = importlib.util.spec_from_file_location(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBuildFA(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fa3 build test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Run a copy so even a regression to the old cleanup/metadata-rewrite
        # script cannot delete the checkout's build cache or edit its pyproject.
        self.fixture_aot = self.root / "aot"
        (self.fixture_aot / "cmake/fa3").mkdir(parents=True)
        self.script = self.fixture_aot / "build_fa.sh"
        for relative in ("build_fa.sh", "pyproject.toml", "cmake/fa3/pyproject.toml"):
            shutil.copy2(AOT / relative, self.fixture_aot / relative)
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
        # A direct CMake invocation is a regression; only pip may drive it.
        self.write_executable(
            self.bin / "cmake",
            "#!/bin/sh\necho 'unexpected direct CMake invocation'\nexit 41\n",
        )
        self.modules = self.root / "modules"
        self.modules.mkdir()
        (self.modules / "pip.py").write_text(textwrap.dedent(r"""
                import json, os, sys
                from pathlib import Path
                from zipfile import ZipFile
                args = sys.argv[1:]
                with open(os.environ['FA3_TEST_TRACE'], 'a') as f:
                    f.write(json.dumps(['pip', dict(
                        args=args, python=sys.executable,
                        nvcc=os.environ.get('CUDACXX'),
                        jobs=os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL'),
                    )]) + '\n')
                print('[mock pip wheel]', flush=True)
                assert args[0] == 'wheel', args
                if os.environ.get('FA3_TEST_FAIL') == 'build':
                    sys.exit(23)
                output = Path(args[args.index('-w') + 1])
                output.mkdir(parents=True, exist_ok=True)
                wheel = output / 'sglang_kernel_fa3-0.4.7-cp310-abi3-linux_x86_64.whl'
                with ZipFile(wheel, 'w') as archive:
                    archive.writestr('sgl_kernel/flash_ops.abi3.so', b'fake FA3 native binary')
                """))
        (self.modules / "torch.py").write_text(textwrap.dedent(r"""
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
                """))
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(self.modules),
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "FA3_NVCC": str(self.nvcc),
            "FA3_PYTHON": str(self.python),
            "FA3_BUILD_DIR": str(self.build),
            "FA3_DIST_DIR": str(self.dist),
            "MAX_JOBS": "2",
            "FA3_TEST_TRACE": str(self.trace),
        }

    @staticmethod
    def write_executable(path, source):
        path.write_text(source)
        path.chmod(0o755)

    def run_build(self, *args, **env):
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.root,
            env={**self.env, **env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=45,
        )

    def calls(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_pip_uses_current_python_and_passes_toolchain_and_overrides(self):
        result = self.run_build("-DFA3_CUTLASS_URL=/cache with spaces/cutlass.tar.gz")
        self.assertEqual(result.returncode, 0, result.stdout)
        pip, load = self.calls()
        self.assertEqual(pip[0], "pip")
        invocation = pip[1]
        self.assertEqual(invocation["python"], str(self.python))
        self.assertEqual(invocation["nvcc"], str(self.nvcc))
        self.assertEqual(invocation["jobs"], "2")
        args = invocation["args"]
        self.assertIn("--no-build-isolation", args)
        self.assertIn("--no-deps", args)
        self.assertIn(f"-Ccmake.source-dir={self.script.parent}", args)
        self.assertIn("-Ccmake.define.BUILD_FA_ONLY=ON", args)
        self.assertIn(f"-Cbuild-dir={self.build}", args)
        self.assertIn(f"-Ccmake.define.Python_EXECUTABLE={self.python}", args)
        self.assertIn(f"-Ccmake.define.CMAKE_CUDA_COMPILER={self.nvcc}", args)
        self.assertIn(
            f"-Ccmake.define.CUDAToolkit_ROOT={self.nvcc.parent.parent}", args
        )
        self.assertIn(
            "-Ccmake.define.FA3_CUTLASS_URL=/cache with spaces/cutlass.tar.gz", args
        )
        self.assertEqual(load[0], "load_library")
        self.assertEqual(Path(load[1]).name, "flash_ops.abi3.so")
        self.assertEqual(
            (self.dist / "flash_ops.abi3.so").read_bytes(), b"fake FA3 native binary"
        )
        self.assertEqual(len(list(self.dist.glob("*.whl"))), 1)
        self.assertIn("[mock pip wheel]", (self.build / "build.log").read_text())

    def test_failed_pip_build_does_not_load_or_publish(self):
        result = self.run_build(FA3_TEST_FAIL="build")
        self.assertEqual(result.returncode, 23, result.stdout)
        self.assertFalse(list(self.dist.glob("*.whl")))
        self.assertFalse((self.dist / "flash_ops.abi3.so").exists())
        self.assertFalse(any(call[0] == "load_library" for call in self.calls()))

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

    def test_invalid_inputs_fail_before_pip(self):
        for env in ({"FA3_NVCC": "/missing/nvcc"}, {"MAX_JOBS": "0"}):
            with self.subTest(env=env):
                result = self.run_build(**env)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(self.calls(), [])

    def test_other_cuda_releases_are_rejected_before_pip(self):
        for version in ("12.8", "13.1", "13.3", "13.20"):
            with self.subTest(version=version):
                self.write_executable(
                    self.nvcc,
                    f"#!/bin/sh\necho 'Cuda compilation tools, release {version}, V{version}.0'\n",
                )
                result = self.run_build()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("12.9", result.stdout)
                self.assertEqual(self.calls(), [])

    def test_default_cuda_symlink_uses_130_toolkit_and_separate_artifacts(self):
        for name in ("FA3_NVCC", "FA3_BUILD_DIR", "FA3_DIST_DIR"):
            self.env.pop(name)
        nvcc130 = self.root / "cuda 13.0/bin/nvcc"
        nvcc130.parent.mkdir(parents=True)
        self.write_executable(
            nvcc130,
            "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n",
        )
        cuda = self.root / "cuda"
        cuda.symlink_to(nvcc130.parent.parent, target_is_directory=True)
        # Redirect only the default absolute-path lookup into the fixture;
        # exercise the real script's selection and symlink resolution without
        # changing the machine's /usr/local/cuda installation.
        self.env["FA3_TEST_DEFAULT_NVCC"] = str(cuda / "bin/nvcc")
        self.env["BASH_FUNC_command%%"] = """() {
            if [[ "$1" == -v && "$2" == /usr/local/cuda/bin/nvcc ]]; then
                builtin printf '%s\\n' "$FA3_TEST_DEFAULT_NVCC"
            else
                builtin command "$@"
            fi
        }"""
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout)
        invocation = self.calls()[0][1]
        self.assertEqual(invocation["nvcc"], str(nvcc130))
        self.assertIn(
            f"-Ccmake.define.CMAKE_CUDA_COMPILER={nvcc130}", invocation["args"]
        )
        self.assertIn(
            f"-Ccmake.define.CUDAToolkit_ROOT={nvcc130.parent.parent}",
            invocation["args"],
        )
        self.assertIn(
            f"-Cbuild-dir={self.fixture_aot / 'build_fa_0510_cu130'}",
            invocation["args"],
        )
        self.assertEqual(
            (self.fixture_aot / "dist_fa_0510_cu130/flash_ops.abi3.so").read_bytes(),
            b"fake FA3 native binary",
        )
        self.assertIn(
            "release 13.0",
            (self.fixture_aot / "build_fa_0510_cu130/build.log").read_text(),
        )
        self.assertFalse((self.fixture_aot / "dist_fa_0510").exists())
        self.assertFalse((self.fixture_aot / "dist_fa_0510_cu132").exists())

    def test_cuda132_build_uses_its_toolkit_and_preserves_cuda129_artifacts(self):
        self.env.pop("FA3_BUILD_DIR")
        self.env.pop("FA3_DIST_DIR")
        previous_object = self.fixture_aot / "build_fa_0510/existing.o"
        previous_library = self.fixture_aot / "dist_fa_0510/flash_ops.abi3.so"
        previous_object.parent.mkdir()
        previous_library.parent.mkdir()
        previous_object.write_bytes(b"CUDA 12.9 object")
        previous_library.write_bytes(b"CUDA 12.9 library")
        nvcc132 = self.root / "cuda 13.2/bin/nvcc"
        nvcc132.parent.mkdir(parents=True)
        self.write_executable(
            nvcc132,
            "#!/bin/sh\necho 'Cuda compilation tools, release 13.2, V13.2.0'\n",
        )
        result = self.run_build(FA3_NVCC=str(nvcc132))
        self.assertEqual(result.returncode, 0, result.stdout)
        invocation = self.calls()[0][1]
        self.assertEqual(invocation["nvcc"], str(nvcc132))
        self.assertIn(
            f"-Ccmake.define.CMAKE_CUDA_COMPILER={nvcc132}", invocation["args"]
        )
        self.assertIn(
            f"-Ccmake.define.CUDAToolkit_ROOT={nvcc132.parent.parent}",
            invocation["args"],
        )
        self.assertIn(
            f"-Cbuild-dir={self.fixture_aot / 'build_fa_0510_cu132'}",
            invocation["args"],
        )
        self.assertEqual(previous_object.read_bytes(), b"CUDA 12.9 object")
        self.assertEqual(previous_library.read_bytes(), b"CUDA 12.9 library")
        self.assertEqual(
            (self.fixture_aot / "dist_fa_0510_cu132/flash_ops.abi3.so").read_bytes(),
            b"fake FA3 native binary",
        )
        self.assertIn(
            "release 13.2",
            (self.fixture_aot / "build_fa_0510_cu132/build.log").read_text(),
        )

    def test_cmake_overrides_cannot_bypass_toolchain_selection(self):
        for option in (
            "-DCMAKE_CUDA_COMPILER=/different/nvcc",
            "-DCMAKE_CUDA_COMPILER:FILEPATH=/different/nvcc",
            "-DCUDAToolkit_ROOT=/different/toolkit",
        ):
            with self.subTest(option=option):
                result = self.run_build(option)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("FA3_NVCC", result.stdout)
                self.assertEqual(self.calls(), [])

    def test_build_directory_with_another_compiler_is_preserved_and_rejected(self):
        self.build.mkdir()
        cache = self.build / "CMakeCache.txt"
        previous_cache = "CMAKE_CUDA_COMPILER:FILEPATH=/different/toolkit/bin/nvcc\n"
        cache.write_text(previous_cache)
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("FA3_BUILD_DIR", result.stdout)
        self.assertEqual(cache.read_text(), previous_cache)
        self.assertEqual(self.calls(), [])

    def test_disabling_fa_only_is_rejected_before_pip(self):
        result = self.run_build("-DBUILD_FA_ONLY=OFF")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("BUILD_FA_ONLY", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_project_metadata_and_existing_backup_are_preserved(self):
        project = self.fixture_aot / "pyproject.toml"
        original = project.read_bytes()
        backup = self.fixture_aot / "pyproject.toml.bak"
        backup.write_bytes(b"existing user backup")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(project.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), b"existing user backup")

    def test_default_directories_preserve_previous_0520_build(self):
        self.env.pop("FA3_BUILD_DIR")
        self.env.pop("FA3_DIST_DIR")
        old_object = self.fixture_aot / "build_fa/existing.o"
        old_library = self.fixture_aot / "dist_fa/flash_ops.abi3.so"
        old_object.parent.mkdir()
        old_library.parent.mkdir()
        old_object.write_bytes(b"0520 object")
        old_library.write_bytes(b"0520 library")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(old_object.read_bytes(), b"0520 object")
        self.assertEqual(old_library.read_bytes(), b"0520 library")
        self.assertTrue((self.fixture_aot / "build_fa_0510/build.log").is_file())
        self.assertEqual(
            (self.fixture_aot / "dist_fa_0510/flash_ops.abi3.so").read_bytes(),
            b"fake FA3 native binary",
        )

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

    def test_relative_python_path_and_incremental_files_are_preserved(self):
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_CUDA_COMPILER:STRING={self.nvcc}\n"
        )
        previous_object = self.build / "existing-object.o"
        previous_object.write_bytes(b"incremental build cache")
        result = self.run_build(FA3_PYTHON="./bin/selected python")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.calls()[0][1]["python"], str(self.python))
        self.assertEqual(previous_object.read_bytes(), b"incremental build cache")

    def test_real_pip_and_cmake_package_only_the_selected_target(self):
        # Exercise the real pip -> scikit-build-core -> CMake -> wheel chain
        # with a tiny CPU library; only Torch's loading boundary remains fake.
        (self.modules / "pip.py").unlink()
        (self.bin / "cmake").unlink()
        fixture = self.root / "fixture"
        project = fixture / "cmake/fa3"
        project.mkdir(parents=True)
        shutil.copy2(AOT / "build_fa.sh", fixture / "build_fa.sh")
        shutil.copy2(AOT / "CMakeLists.txt", fixture / "CMakeLists.txt")
        shutil.copy2(AOT / "cmake/fa3/pyproject.toml", project / "pyproject.toml")
        self.script = fixture / "build_fa.sh"
        (project / "fake.c").write_text("int fake_fa3(void) { return 0; }\n")
        (project / "CMakeLists.txt").write_text(textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.26)
            project(fa3_wheel_fixture LANGUAGES C)
            add_library(flash_ops MODULE fake.c)
            set_target_properties(flash_ops PROPERTIES PREFIX "" SUFFIX ".abi3.so")
            install(TARGETS flash_ops LIBRARY DESTINATION sgl_kernel)
            add_custom_target(unrelated ALL COMMAND "${CMAKE_COMMAND}" -E false)
            """))
        unwanted_package = project / "sglang_kernel_fa3"
        unwanted_package.mkdir()
        (unwanted_package / "__init__.py").write_text(
            "# Must not be included in the wheel.\n"
        )
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout)
        wheels = list(self.dist.glob("*.whl"))
        self.assertEqual(len(wheels), 1)
        with ZipFile(wheels[0]) as wheel:
            native_files = [p for p in wheel.namelist() if ".dist-info/" not in p]
            self.assertEqual(native_files, ["sgl_kernel/flash_ops.abi3.so"])
            self.assertEqual(
                wheel.read(native_files[0]),
                (self.dist / "flash_ops.abi3.so").read_bytes(),
            )
            metadata_path = next(p for p in wheel.namelist() if p.endswith("/METADATA"))
            metadata = wheel.read(metadata_path).decode()
            self.assertIn("Name: sglang-kernel-fa3", metadata.splitlines())
            self.assertIn("Requires-Dist: torch==2.13.0", metadata)

    def test_root_fa_only_skips_full_kernel_configuration(self):
        # Use the real root CMakeLists but a tiny FA3 child: the parent must
        # delegate before probing CUDA/Torch or downloading unrelated sources.
        fixture = self.root / "root-cmake"
        child = fixture / "cmake/fa3"
        child.mkdir(parents=True)
        shutil.copy2(AOT / "CMakeLists.txt", fixture / "CMakeLists.txt")
        (child / "CMakeLists.txt").write_text(textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.26)
            project(fa3_configure_fixture LANGUAGES NONE)
            file(WRITE "${CMAKE_BINARY_DIR}/fa3-configured" "FA3 child configured")
            add_custom_target(flash_ops COMMAND "${CMAKE_COMMAND}" -E true)
            """))
        build = self.root / "root-cmake-build"
        result = subprocess.run(
            [
                shutil.which("cmake"),
                "-S",
                str(fixture),
                "-B",
                str(build),
                "-DBUILD_FA_ONLY=ON",
                "-DCMAKE_CUDA_COMPILER=/missing/full-build-nvcc",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((build / "fa3-configured").read_text(), "FA3 child configured")


if __name__ == "__main__":
    unittest.main()
