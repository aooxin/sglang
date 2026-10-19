# 0510 FA3 kernels with the 0520 interface

Both the normal AOT build and this standalone build use the complete, unmodified
0510 sgl-attn source tree:

- Revision: `bcf72ccc6816b36a5fae2c5a3c027604629785e0`
- Archive SHA256: `2110d8ca1ed9b330b9f99c1d4088be12a37e44152c39fdf90dfb6526f532ba96`

`include/sgl_flash_kernel_ops.h` declares the upstream 35-argument `mha_fwd`.
`include/fa3_0510_compat.h` adapts it at registration to the existing 0520
37-argument `sgl_kernel::fwd` schema. The Python wrappers, output alias annotation
and four returned tensors remain compatible with 0520. The adapter uses the
existing PyTorch shim for all original argument conversions.

The two extra arguments must be `sparse_mask_fine=None` and `only_qv=False`.
Other values raise an error before the 0510 implementation executes. In
particular, DSA paths requiring Only-QV are not supported by this build.
The full AOT build rejects `SGL_KERNEL_ENABLE_FA3_SPARSE_MASK=ON` when FA3 is
enabled. The attention calculation, scheduler metadata preparation and automatic
Split-KV heuristic all come from 0510.

## Build entry points

`build_fa.sh` invokes the AOT root `CMakeLists.txt` with `BUILD_FA_ONLY=ON`.
That branch delegates to `cmake/fa3/CMakeLists.txt` before the full project's
CUDA/Torch configuration or unrelated dependency downloads. The child defines
the `flash_ops` target, using the 0510 sources and the local 0520 registration
and compatibility adapter. `cmake/fa3/pyproject.toml` supplies the wheel metadata
and selects only that target; the root `pyproject.toml` and any existing backup
are preserved.

The compiled sources are the local `csrc/flash_extension.cc`, upstream
`hopper/flash_api.cpp`, `hopper/flash_prepare_scheduler.cu`,
`hopper/flash_fwd_combine.cu`, and the selected forward instantiations under
`hopper/instantiations/`. The source list retains BF16, FP16 and FP8 support,
including separate head-dimension instantiations. The device flags retain
SM90a, optional SM80/86, C++17, `-O3`, fast math, GDC and extended SM90 MMA shapes
from the supplied 0510/0520 recipes.

The selected Python supplies the Torch headers, libraries and CXX11 ABI value.
The FA3 configuration queries these directly instead of using Torch's CMake
package, whose CUDA-major check rejects CUDA 12.9 with Torch cu13. The compiler
and toolkit versions must match. The script defaults to `/usr/local/cuda/bin/nvcc`
and resolves that symlink to select the matching toolkit. CUDA 12.9, 13.0 and
experimental 13.2 are accepted. The build links the selected toolkit's static
CUDA runtime, so switching toolkits changes the compiler, CUDA headers and
runtime together.

## Rebuild

Run in a Linux CUDA environment with Python >= 3.10, CUDA-enabled Torch 2.13,
CUDA toolkit/nvcc 12.9, 13.0 or experimental 13.2, a host C++ compiler,
CMake >= 3.26, Ninja and scikit-build-core >= 0.10. The existing deployment
installer specifically checks Torch `2.13.0+cu130`, CUDA runtime `13.0`, and
CXX11 ABI=True.

For a CloudML container, mount or copy the updated 0520 checkout before building.
The build uses `cmake/fa3/CMakeLists.txt`, `cmake/fa3/pyproject.toml`,
`csrc/flash_extension.cc`, `include/sgl_flash_kernel_ops.h` and
`include/fa3_0510_compat.h` in addition to the two root build entry points. Keep
these files from the same migrated checkout. The selected CUDA toolkit must include
its headers, libraries and NVVM as well as `nvcc`.

Select the container's runtime Python and check its build environment:

```bash
export FA3_PYTHON="$(command -v python3)"
export FA3_NVCC=/usr/local/cuda/bin/nvcc
"$FA3_NVCC" --version
"$FA3_PYTHON" - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CXX11 ABI:", torch.compiled_with_cxx11_abi())
print("GPU available:", torch.cuda.is_available())
assert sys.version_info >= (3, 10), "Python >= 3.10 is required"
assert str(torch.__version__).startswith("2.13.") and torch.version.cuda, (
    "Use the target 0520 container's CUDA-enabled Torch 2.13"
)
PY
```

If build/test tools are missing, install them into that same Python environment:

```bash
"$FA3_PYTHON" -m pip install 'scikit-build-core>=0.10' 'cmake>=3.26' ninja wheel pytest
```

From the repository root:

```bash
cd python/sglang/kernels/aot
MAX_JOBS=2 bash build_fa.sh -DENABLE_BELOW_SM90=OFF
```

This command builds SM90a for H20/H100. Use `-DENABLE_BELOW_SM90=ON` to include
SM80/86; this is the default on x86_64, while ARM64 defaults to SM90a only.
Set `FA3_NVCC` to override `/usr/local/cuda/bin/nvcc`; set `FA3_PYTHON` to select
the target Python. An existing exported `FA3_NVCC` overrides the script's default.

Default directories are relative to the AOT directory:

| Selected nvcc | Incremental build directory | Validated artifacts |
| --- | --- | --- |
| 12.9 | `build_fa_0510` | `dist_fa_0510` |
| 13.0 | `build_fa_0510_cu130` | `dist_fa_0510_cu130` |
| 13.2 | `build_fa_0510_cu132` | `dist_fa_0510_cu132` |

`FA3_BUILD_DIR` and `FA3_DIST_DIR` override these defaults. The build log is
appended to `<build directory>/build.log` and includes the full nvcc version.
Use a separate build and output directory for each toolchain. The script rejects
a build directory whose CMake cache selects another compiler, and rejects
`-DCMAKE_CUDA_COMPILER` / `-DCUDAToolkit_ROOT` overrides: use `FA3_NVCC` to select
the compiler and matching toolkit together.

By default, CUTLASS and sgl-attn are downloaded from Xiaomi's GitHub mirror,
`https://pkgs.d.xiaomi.net/artifactory/github`, matching the 0510 Feishu build
instructions. `GITHUB_ARTIFACTORY` selects the mirror host/path for a fresh
configuration. `FA3_CUTLASS_URL` and `FA3_ATTENTION_URL` can override individual
archive URLs or select local archives; the pinned revisions and SHA256 checks
still apply.

CMake caches the complete archive URLs. If this build directory was previously
configured with direct GitHub URLs, pass both mirror URLs once to update that
cache while reusing the build directory. From the AOT directory:

```bash
MAX_JOBS=2 bash build_fa.sh \
    -DFA3_CUTLASS_URL=https://pkgs.d.xiaomi.net/artifactory/github/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz \
    -DFA3_ATTENTION_URL=https://pkgs.d.xiaomi.net/artifactory/github/sgl-project/sgl-attn/archive/bcf72ccc6816b36a5fae2c5a3c027604629785e0.tar.gz
```

Subsequent builds in that directory retain these URLs. A fresh build directory
uses the mirror by default without either override.

Use a fresh build directory when changing source revisions. The older build
directory may cache the 0520 `FA3_ATTENTION_URL` or a source-directory override.
`-DFA3_ATTENTION_URL=/absolute/path/to/0510.sgl-attn.tar.gz` reuses a local archive;
the pinned archive digest is still checked. A
`FETCHCONTENT_SOURCE_DIR_FA3_ATTENTION` override must point to the complete 0510
source tree. Do not mix 0510 and 0520 headers or object files.

The script checks native loading and the 37-argument schema, then writes the
wheel and `flash_ops.abi3.so` into the selected output directory. It does not
install them. The FA3-only wheel is named `sglang_kernel_fa3-*.whl` for all
supported toolchains; identify each build by its output directory, build log and
.so hash.

## Use the container's CUDA 13.0 toolkit

The following commands use `/usr/local/cuda/bin/nvcc`, with `/usr/local/cuda`
pointing to the container's complete CUDA 13.0 toolkit. Keep the same Python,
Torch, host C++ compiler, kernel sources and architecture flags as the 12.9
baseline. The existing installer's target remains Torch `2.13.0+cu130`.

The paths below match the example CloudML checkout and Python environment.
Set `FA3_PYTHON` to the benchmark's actual `sys.executable` if it differs:

```bash
export FA3_REPO_DIR=/sgl-workspace/sglang
export FA3_PYTHON=/opt/sglang/bin/python
unset FA3_NVCC
export FA3_BUILD_DIR="$FA3_REPO_DIR/python/sglang/kernels/aot/build_fa_0510_cu130"
export FA3_DIST_DIR="$FA3_REPO_DIR/python/sglang/kernels/aot/dist_fa_0510_cu130"

/usr/local/cuda/bin/nvcc --version
"$FA3_PYTHON" -c 'import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)'
cd "$FA3_REPO_DIR"
MAX_JOBS=16 bash python/sglang/kernels/aot/build_fa.sh -DENABLE_BELOW_SM90=OFF
```

`MAX_JOBS=16` is a starting point; choose concurrency according to the container's
available memory as well as CPU allocation. The same Xiaomi mirror, pinned 0510
FA3/CUTLASS revisions and 0520 adapter are used. A successful build checks native
loading and registration only; run the GPU validation below before benchmarking.
No CUDA compilation or GPU correctness/performance result is implied by the CPU
build-orchestration tests.

To select CUDA 12.9 or 13.2 explicitly, set `FA3_NVCC` to that toolkit's compiler
and choose its corresponding build/output directories from the table above.
Clear or update previously exported `FA3_BUILD_DIR` and `FA3_DIST_DIR` when
switching toolchains. The validation and installation examples below default to
the CUDA 13.0 output unless `FA3_DIST_DIR` is set explicitly.

## Validate

From the repository root, test the build orchestration and the real C++ adapter
and registration with a CPU test backend that has the 0510 signatures:

```bash
python -m pytest -q \
    test/registered/unit/tools/test_build_fa.py \
    test/registered/unit/tools/test_fa3_0510_compat.py
```

This checks the root FA3-only CMake entry, real pip/CMake wheel packaging with a
small CPU library, preservation of existing metadata/artifacts, and build
failure propagation. The C++ adapter checks cover schema, argument
conversion/forwarding, output aliasing, optional arguments, feature rejection
before output mutation, and metadata binding. These tests require Torch,
CMake, Ninja, scikit-build-core and a C++ compiler; they do not compile or execute
the CUDA attention kernels.

On the target GPU, test the newly built binary in a fresh process:

```bash
FA3_TEST_LIBRARY="${FA3_DIST_DIR:-$PWD/python/sglang/kernels/aot/dist_fa_0510_cu130}/flash_ops.abi3.so" \
"${FA3_PYTHON:-python3}" -m pytest -q -rs test/manual/attention/test_fa3_0510_kernel.py
```

Compilation can run without a GPU. This validation step requires a visible
H20/H100-class GPU for the SM90a-only build; all three tests should pass.
`-rs` reports why any tests were skipped, which does not validate GPU execution.

The GPU test compares BF16 attention against a float32 reference for Q/K dimension
192, V dimension 128, GQA, shuffled paged KV, variable-length prefill and decode,
automatic/explicit splits, and precomputed scheduler metadata. It also checks
CUDA graph replay and unsupported feature errors. The existing backend suite
`test/registered/attention/unittests/dense/test_fa3.py` covers SGLang integration
after the new artifact is installed.

## Start the service with the cloud artifact

The deployed build is `/mnt/xiaoxin7-dev/fa3/flash_ops.abi3.so`. The installer
defaults in `docker/install_fa3.py` and the image's `FA3_ARTIFACT_DIR` in
`docker/Dockerfile.xiaomi` both select that directory, and the installer pins
its SHA256, so no extra environment variables are needed.

`start_p.sh`, `start_d.sh`, `start_26p.sh` and `start_26d.sh` already run
`python3 /opt/sglang/fa3/install.py || exit 1` before launching SGLang. For
another entrypoint, run that installer explicitly before the server command.

Every node needs the `/mnt/xiaoxin7-dev` mount. A missing file or a checksum
mismatch makes the installer fail, which stops the startup script before any
model process starts. Keep the existing `--attention-backend fa3` and
model/parallelism arguments. The `.so` installation is sufficient; the wheel
does not need a separate install.

The installer discovers the extension Python actually imports, validates the
staged file's checksum and 0520 operator schema, and atomically replaces that
extension while keeping a backup. Its final `[fa3-install] ready:` line gives
the installed path and SHA256. The server will still load a path under
`site-packages/sgl_kernel/`; that is the expected installed copy of the cloud
artifact. Start fresh server processes after installation.

## Install the new artifact

The installer defaults identify the deployed cloud artifact above. To try a
newly rebuilt binary, explicitly select it and its actual digest; keep these
environment variables in service launch commands that invoke the installer
again. From the repository root:

```bash
export FA3_ARTIFACT_DIR="${FA3_DIST_DIR:-$PWD/python/sglang/kernels/aot/dist_fa_0510_cu130}"
export FA3_ARTIFACT_SHA256="$(sha256sum "$FA3_ARTIFACT_DIR/flash_ops.abi3.so" | cut -d ' ' -f 1)"
"${FA3_PYTHON:-python3}" docker/install_fa3.py
```

Run installation in the target 0520 environment before starting a new server
process. For another node, copy the validated artifact to a mounted directory and
use that directory with the same digest. A process that already loaded the old
extension must be restarted to use the new one.

To make a validated build the deployed default, copy it to the cloud disk and
update `DEFAULT_SHA256` in `docker/install_fa3.py`. If the directory changes,
update `DEFAULT_ARTIFACT_DIR` there and `ENV FA3_ARTIFACT_DIR` in
`docker/Dockerfile.xiaomi` together; the image's environment variable takes
precedence over the installer default.

Use the same Python to build, install and benchmark. When running the Feishu
`bench_fa3_attention.py`, this wrapper prints and checks the loaded library's
path/hash, then runs the benchmark in that same fresh process. A mismatch aborts
before timing any kernels:

```bash
"${FA3_PYTHON:-python3}" - bench_fa3_attention.py \
    --from-config /preset-models/config.json \
    --forward-mode prefill --layer-type full \
    --attention-tp 8 --page-size 64 \
    --simulate-prefill 983040 --prefill-seqlen-q 16384 <<'PY'
import hashlib
import os
import runpy
import sys
from pathlib import Path
from sgl_kernel import flash_ops

library = Path(flash_ops.__file__).resolve()
digest = hashlib.sha256()
with library.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
actual = digest.hexdigest()
print("Python:", sys.executable, flush=True)
print("Loaded FA3:", library, flush=True)
print("SHA256:", actual, flush=True)
if actual != os.environ["FA3_ARTIFACT_SHA256"]:
    raise RuntimeError("Benchmark FA3 differs from the selected build artifact")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
```

For the 12.9 comparison, select its artifact directory, recompute
`FA3_ARTIFACT_SHA256`, rerun the installer, and start a fresh benchmark process.
Use the same GPU, inputs and benchmark options for both runs. The benchmark's
`torch.version.cuda` line describes the Torch build; it can remain `13.0` for
both experiments. The nvcc version is recorded in each build's `build.log`.
