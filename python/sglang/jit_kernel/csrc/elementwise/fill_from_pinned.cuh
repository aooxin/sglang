/*
 * SM-driven H2D fill kernel: copy a contiguous buffer (int32 or int64) from
 * PINNED host memory into a CUDA tensor via the SM load/store pipeline over
 * PCIe rather than the DMA copy engine, avoiding DMA-CE FIFO contention.
 *
 * `src` MUST be page-locked (pinned) host memory: a device kernel can only
 * dereference a raw host pointer when it is pinned (mapped into UVA). The
 * pointer is passed as a raw int64 (data_ptr()) rather than a TensorView to
 * sidestep the FFI marshaling layer's CPU-tensor device assertion.
 */

#pragma once

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize
#include <sgl_kernel/utils.h>   // div_ceil, RuntimeCheck

#include <sgl_kernel/utils.cuh>  // host::LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__global__ void fill_from_ptr_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx];
  }
}

constexpr size_t kBlockSize = 128;

// dst: 1D contiguous CUDA tensor of element type T.
// src_ptr: raw data_ptr() of a 1D contiguous PINNED host buffer of element
//          type T and the SAME length as dst.
template <typename T>
void fill_from_pinned_impl(tvm::ffi::TensorView dst, int64_t src_ptr) {
  using namespace host;

  // Validate dst only; src is a raw pinned-host pointer (see header note).
  // `.template with_...<T>()`: T is a dependent template parameter, so the
  // `template` disambiguator is required to parse `<...>` as a template
  // argument list. Read the device from `dst.device()` afterward.
  SymbolicSize N = {"num_elements"};
  TensorMatcher({N}).template with_dtype<T>().template with_device<kDLCUDA>().verify(dst);

  const size_t num_elements = N.unwrap();
  RuntimeCheck(num_elements > 0, "fill_from_pinned: empty dst");
  RuntimeCheck(src_ptr != 0, "fill_from_pinned: null src_ptr");

  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = dst.device();

  // Launched on dst's current stream (tvm-ffi env stream follows the torch
  // stream), so `with torch.cuda.stream(s)` runs it on stream s.
  LaunchKernel(grid_size, kBlockSize, device)(
      fill_from_ptr_kernel<T>, static_cast<T*>(dst.data_ptr()), reinterpret_cast<const T*>(src_ptr), num_elements);
}

}  // namespace
