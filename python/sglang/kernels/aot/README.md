# sglang-kernel (prior sgl-kernel)

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot) for LLM inference engines

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sglang-kernel)](https://pypi.org/project/sglang-kernel)

</div>

`sglang-kernel` provides optimized compute primitives for LLM inference engines, enabling efficient inference for large language models and vision-language models through custom kernel operations. The source tree lives under the `python/sglang/kernels/aot/` directory and the Python import path remains `sgl_kernel`.

## Installation
Requires torch == 2.13.0

```bash
# Latest version
pip3 install sglang-kernel --upgrade
```

## Building from Source
Requires
- CMake ≥3.31,
- Python ≥3.10
- scikit-build-core
- ninja(optional)

### Use Makefile to build from the sgl-kernel source tree

```bash
make build
```

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Build FA3 only with CUDA 12.9

`build_fa.sh` adapts the 0510 FA3 build script referenced in the
[CUDA 13.2 adaptation document](https://mi.feishu.cn/docx/B0tGdaEZCoPfsIxIo6Pcuhn0nDb)
to the 0520 source layout. It produces a `flash_ops.abi3.so` and a separate
`sglang-kernel-fa3-cu129` wheel under `dist_fa/`, using the selected Python's
Torch headers, libraries and C++ ABI. This supports the intended combination of
CUDA 12.9 nvcc and an existing `torch==2.13.0+cu130` environment.

Prerequisites: a CUDA 12.9 toolkit (compiler, headers and libraries), a compatible
C++ compiler, CMake >=3.26, Ninja, `flock`, and CPython >=3.10 with matching Python
development headers, CUDA-enabled Torch 2.13 and `wheel`. The selected Python must
already be able to import Torch. The final native loading check also needs the
CUDA driver library `libcuda.so.1`; it does not execute GPU kernels.

From the **0520 repository root**:

```bash
cd python/sglang/kernels/aot
bash build_fa.sh
```

The default compiler is `/usr/local/cuda-12.9/bin/nvcc`. Select a different
CUDA 12.9 location or Python environment explicitly when needed:

```bash
FA3_NVCC=/usr/local/cuda-12.9/bin/nvcc \
FA3_PYTHON=/opt/sglang/bin/python3 \
MAX_JOBS=2 \
bash build_fa.sh
```

Like the original script, the default build retains the normal FA3 architecture
list: SM80, SM86 and SM90a on x86_64. For an H20/H100-only build, reduce the work:

```bash
bash build_fa.sh -DENABLE_BELOW_SM90=OFF
```

The standalone project in `cmake/fa3/` implements the original `BUILD_FA_ONLY`
behavior. It builds only `flash_ops`, using the same pinned CUTLASS/sgl-attn
revisions, forward dtypes and default feature flags as this tree's FA3 target.
It links directly to the current Torch libraries, so Torch's CMake CUDA-major
check does not reject the intentional compiler/runtime combination. The
extension uses the CUDA toolkit's static runtime, matching CMake's default in
the original build; it does not add a dynamic `libcudart.so.12` dependency.

Builds retain intermediate files in `build_fa/` and append output to
`build_fa/build.log`. Ninja reports completed steps as `[completed/total, elapsed]`.
Watch the log from another shell:

```bash
tail -F build_fa/build.log
```

For a build that survives an SSH disconnect, launch it with `nohup`:

```bash
nohup bash build_fa.sh -DENABLE_BELOW_SM90=OFF \
  > /tmp/sglang-fa3-launch.log 2>&1 < /dev/null &
tail -F build_fa/build.log
```

The script prints `[fa3-build] COMPLETE` only after compilation, native loading,
the 0520 operator-schema check and packaging succeed. A failed step returns a
nonzero exit code. Re-running the script uses Ninja's incremental build; a second
script invocation on the same build directory is rejected while one is active.

`FA3_BUILD_DIR` and `FA3_DIST_DIR` override the build and output directories.
Use a fresh build directory when switching compiler versions. For an offline
build, supply the pinned dependency archives (their SHA256 is still checked):

```bash
bash build_fa.sh \
  -DFA3_CUTLASS_URL=/path/to/cutlass-57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz \
  -DFA3_ATTENTION_URL=/path/to/sgl-attn-f89bc2306632d1ec5f97b014dded4254f5b4a907.tar.gz
```

Successful outputs are:

- `dist_fa/flash_ops.abi3.so`: the extension for manual deployment/testing.
- `dist_fa/sglang_kernel_fa3_cu129-*.whl`: a FA3-only wheel containing
  `sgl_kernel/flash_ops.abi3.so` and build metadata.
- `dist_fa/manifest.json`: Torch/CUDA/ABI, nvcc, architecture, size and SHA256.

The script only builds artifacts. It leaves installation and replacement to a
separate step, and the manifest records that GPU execution has not been tested.
The `.abi3.so` suffix describes Python's stable ABI; the extension still needs a
compatible Torch C++ ABI when deployed. Validate it on the target GPU before
integrating it into an image.

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/python/sgl_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from sgl_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## FAQ
- Q: Segmentation fault with CUDA 12.6
- A: Update ptxas to 12.8, reference: [segment fault error](https://github.com/Dao-AILab/flash-attention/issues/1453)
