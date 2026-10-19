#!/usr/bin/env bash
# Compile 0510 FA3 with the 0520 operator interface using the selected CUDA nvcc.
# The root CMakeLists.txt selects BUILD_FA_ONLY; cmake/fa3/pyproject.toml
# supplies FA3-only wheel metadata without modifying the full-kernel project.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash build_fa.sh [CMake -D options...]

  FA3_NVCC       CUDA 12.9/13.0/13.2 compiler (default: /usr/local/cuda/bin/nvcc)
  FA3_PYTHON     Python with Torch 2.13 and scikit-build-core (default: python3)
  MAX_JOBS       Parallel compiler jobs (default: 2)
  FA3_BUILD_DIR  Incremental build directory (CUDA 13.0: <aot>/build_fa_0510_cu130)
  FA3_DIST_DIR   Validated .so and FA3-only wheel (CUDA 13.0: <aot>/dist_fa_0510_cu130)

Builds 0510 FA3 (bcf72ccc) with the 0520 37-argument operator interface.
Uses the root CMakeLists.txt with BUILD_FA_ONLY=ON.
CUDA 12.9 uses build_fa_0510/dist_fa_0510; CUDA 13.0 appends _cu130.
CUDA 13.2 is experimental; its default build/output directories end in _cu132.
Select the complete toolkit via FA3_NVCC; CMake uses its matching headers/libs.
Use a fresh FA3_BUILD_DIR when switching compiler or kernel source versions.
The build log is appended to FA3_BUILD_DIR/build.log. Nothing is installed.
For H20/H100-only builds: bash build_fa.sh -DENABLE_BELOW_SM90=OFF
HELP
    exit 0
fi

fail() { echo "[fa3-build] ERROR: $*" >&2; exit 1; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fa3_python=$(command -v "${FA3_PYTHON:-python3}") || fail "Cannot find FA3_PYTHON=${FA3_PYTHON:-python3}"
# Make relative paths absolute, but retain a venv's Python symlink: resolving
# that symlink would select the system interpreter and lose the venv's Torch.
fa3_python="$(cd -- "$(dirname -- "$fa3_python")" && pwd)/$(basename -- "$fa3_python")"
fa3_nvcc=$(command -v "${FA3_NVCC:-/usr/local/cuda/bin/nvcc}") || fail "Cannot find nvcc; set FA3_NVCC to a CUDA 12.9, 13.0 or 13.2 compiler"
[[ -x "$fa3_nvcc" ]] || fail "Compiler is not executable: $fa3_nvcc"
fa3_nvcc=$(readlink -f -- "$fa3_nvcc")
fa3_toolkit=$(dirname -- "$(dirname -- "$fa3_nvcc")")
nvcc_version=$("$fa3_nvcc" --version)
case "$nvcc_version" in
    *"release 12.9,"*) cuda_suffix="" ;;
    *"release 13.0,"*) cuda_suffix="_cu130" ;;
    *"release 13.2,"*) cuda_suffix="_cu132" ;;
    *) fail "This recipe accepts CUDA 12.9, 13.0 or 13.2 (experimental); got: $nvcc_version" ;;
esac

jobs=${MAX_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
for tool in cmake ninja flock tee; do
    command -v "$tool" >/dev/null || fail "Required tool not found: $tool"
done
cmake_settings=()
for arg in "$@"; do
    [[ "$arg" == -D*=* ]] || fail "Expected a CMake -DNAME=VALUE option: $arg"
    definition=${arg#-D}
    variable=${definition%%=*}
    variable=${variable%%:*}
    case "$variable" in
        BUILD_FA_ONLY)
            [[ "${definition#*=}" == ON ]] || fail "build_fa.sh requires BUILD_FA_ONLY=ON" ;;
        CMAKE_CUDA_COMPILER|CUDAToolkit_ROOT)
            fail "Use FA3_NVCC to select the compiler and matching toolkit; $variable is set by build_fa.sh" ;;
    esac
    cmake_settings+=("-Ccmake.define.${arg#-D}")
done

build_dir=${FA3_BUILD_DIR:-"$script_dir/build_fa_0510$cuda_suffix"}
dist_dir=${FA3_DIST_DIR:-"$script_dir/dist_fa_0510$cuda_suffix"}
mkdir -p -- "$build_dir" "$dist_dir"
build_dir=$(cd -- "$build_dir" && pwd)
dist_dir=$(cd -- "$dist_dir" && pwd)
exec 9>"$build_dir/.build_fa.lock"
flock -n 9 || fail "A build_fa.sh is already using $build_dir"
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    cached_nvcc=$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
    if [[ -n "$cached_nvcc" && "$cached_nvcc" != "$fa3_nvcc" ]]; then
        fail "FA3_BUILD_DIR contains another compiler ($cached_nvcc); select a fresh FA3_BUILD_DIR"
    fi
fi

log_file="$build_dir/build.log"
wheel_dir=$(mktemp -d "$build_dir/.fa3-wheel.XXXXXX")
cleanup() {
    local status=$?
    rm -rf -- "$wheel_dir"
    if (( status != 0 )); then echo "[fa3-build] FAILED (exit $status); log: $log_file" >&2; fi
}
trap cleanup EXIT
export PATH="$fa3_toolkit/bin:$PATH"
export CUDACXX="$fa3_nvcc"
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
export NINJA_STATUS='[%f/%t, %es] '

# pipefail preserves configure/compiler/validation failures through tee.
(
    echo "[fa3-build] started: $(date -Is)"
    echo "[fa3-build] python: $fa3_python; nvcc: $fa3_nvcc; jobs: $jobs"
    echo "$nvcc_version"
    echo "[fa3-build] build: $build_dir; output: $dist_dir"
    "$fa3_python" -m pip wheel "$script_dir/cmake/fa3" \
        --no-build-isolation --no-deps --disable-pip-version-check -v \
        -w "$wheel_dir" \
        -Cbuild-dir="$build_dir" \
        -Cbuild.verbose=true \
        -Ccmake.source-dir="$script_dir" \
        -Ccmake.define.BUILD_FA_ONLY=ON \
        -Ccmake.define.CMAKE_CUDA_COMPILER="$fa3_nvcc" \
        -Ccmake.define.CUDAToolkit_ROOT="$fa3_toolkit" \
        -Ccmake.define.Python_EXECUTABLE="$fa3_python" \
        "${cmake_settings[@]}"

    # Validate the extension actually packaged by pip, then expose it for
    # manual replacement. Importing sgl_kernel could load the old extension.
    "$fa3_python" - "$wheel_dir" <<'PY'
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile
import torch

directory = Path(sys.argv[1])
wheel, = directory.glob("sglang_kernel_fa3-*.whl")
library = directory / "flash_ops.abi3.so"
with ZipFile(wheel) as archive, archive.open("sgl_kernel/flash_ops.abi3.so") as source, library.open("wb") as output:
    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
torch.ops.load_library(str(library))
schema = torch.ops.sgl_kernel.fwd.default._schema
names = tuple(arg.name for arg in schema.arguments)
if len(names) != 37 or names[-3:] != ("sinks", "sparse_mask_fine", "only_qv"):
    raise RuntimeError(f"Unexpected 0520 FA3 fwd schema: {schema}")
torch.ops.sgl_kernel.get_scheduler_metadata.default._schema
print(f"[fa3-build] native load/schema OK: torch={torch.__version__}", flush=True)
PY
    mv -- "$wheel_dir"/*.whl "$wheel_dir/flash_ops.abi3.so" "$dist_dir/"
    echo "[fa3-build] COMPLETE: $dist_dir (native load checked; GPU execution still needs testing)"
) 2>&1 | tee -a "$log_file"
