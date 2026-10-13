#!/usr/bin/env bash
# FA3-only counterpart of the 0510 build_fa.sh. Build against the active Torch
# with CUDA 12.9 nvcc, without modifying pyproject.toml or installing the wheel.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash build_fa.sh [CMake -D options...]

  FA3_NVCC       CUDA 12.9 compiler (default: /usr/local/cuda-12.9/bin/nvcc)
  FA3_PYTHON     Python with the target Torch 2.13 and wheel (default: python3)
  MAX_JOBS       Parallel compiler jobs (default: 2)
  FA3_BUILD_DIR  Incremental build directory (default: <aot>/build_fa)
  FA3_DIST_DIR   Validated .so, FA3-only wheel and manifest (default: <aot>/dist_fa)

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
fa3_nvcc=$(command -v "${FA3_NVCC:-/usr/local/cuda-12.9/bin/nvcc}") || fail "Cannot find CUDA 12.9 nvcc; set FA3_NVCC to its path"
[[ -x "$fa3_nvcc" ]] || fail "Compiler is not executable: $fa3_nvcc"
fa3_nvcc=$(readlink -f -- "$fa3_nvcc")
fa3_toolkit=$(dirname -- "$(dirname -- "$fa3_nvcc")")
nvcc_version=$("$fa3_nvcc" --version)
[[ "$nvcc_version" == *"release 12.9,"* ]] || fail "This recipe requires CUDA 12.9 nvcc; got: $nvcc_version"

jobs=${MAX_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
for tool in cmake ninja flock tee; do
    command -v "$tool" >/dev/null || fail "Required tool not found: $tool"
done

build_dir=${FA3_BUILD_DIR:-"$script_dir/build_fa"}
dist_dir=${FA3_DIST_DIR:-"$script_dir/dist_fa"}
mkdir -p -- "$build_dir" "$dist_dir"
build_dir=$(cd -- "$build_dir" && pwd)
dist_dir=$(cd -- "$dist_dir" && pwd)
exec 9>"$build_dir/.build_fa.lock"
flock -n 9 || fail "A build_fa.sh is already using $build_dir"

log_file="$build_dir/build.log"
trap 'status=$?; if (( status != 0 )); then echo "[fa3-build] FAILED (exit $status); log: $log_file" >&2; fi' EXIT
export PATH="$fa3_toolkit/bin:$PATH"
export NINJA_STATUS='[%f/%t, %es] '

# pipefail preserves configure/compiler/validation failures through tee.
(
    echo "[fa3-build] started: $(date -Is)"
    echo "[fa3-build] python: $fa3_python; nvcc: $fa3_nvcc; jobs: $jobs"
    echo "[fa3-build] build: $build_dir; output: $dist_dir"
    cmake -S "$script_dir/cmake/fa3" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$fa3_nvcc" \
        -DCUDAToolkit_ROOT="$fa3_toolkit" \
        -DPython_EXECUTABLE="$fa3_python" \
        "$@"
    cmake --build "$build_dir" --target flash_ops --parallel "$jobs" --verbose
    "$fa3_python" "$script_dir/cmake/fa3/package_fa.py" \
        --library "$build_dir/lib/flash_ops.abi3.so" \
        --build-info "$build_dir/fa3-build-info.json" \
        --dist-dir "$dist_dir"
    echo "[fa3-build] COMPLETE: $dist_dir (native load checked; GPU execution still needs testing)"
) 2>&1 | tee -a "$log_file"
