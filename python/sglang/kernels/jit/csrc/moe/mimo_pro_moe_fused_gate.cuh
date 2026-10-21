// MiMo-V2 Pro router: sigmoid scoring + correction bias + top-k over 384 experts
// in one group. Ported from the internal sgl-kernel source of MR !448 / !1110
// (sgl-kernel/csrc/moe/mimo_pro_moe_fused_gate.cu) with the selection algorithm
// unchanged. One deliberate difference: the routed scaling factor applies even
// without renormalize, matching the other routers (the original only scaled
// inside the renormalize branch; MiMo always renormalizes).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace sglang {

inline constexpr int32_t kMimoProNumExperts = 384;
inline constexpr int32_t kMimoProWarpSize = 32;
// Up to this many tokens, one CTA serves one token with one thread per expert;
// above it, one warp serves one token. Threshold taken from the original kernel.
inline constexpr int64_t kMimoProSmallTokenThreshold = 512;
inline constexpr int32_t kMimoProWarpsPerTokenSmall = kMimoProNumExperts / kMimoProWarpSize;
inline constexpr int32_t kMimoProWarpsPerCta = 6;

/// \brief Warp-wide argmax; on equal scores the lower expert id wins.
SGL_DEVICE void mimo_pro_warp_argmax(float& value, int32_t& expert) {
#pragma unroll
  for (int32_t offset = kMimoProWarpSize / 2; offset > 0; offset /= 2) {
    const float other_value = __shfl_down_sync(device::kFullMask, value, offset);
    const int32_t other_expert = __shfl_down_sync(device::kFullMask, expert, offset);
    if (other_value > value || (other_value == value && other_expert < expert)) {
      value = other_value;
      expert = other_expert;
    }
  }
}

/// \brief Renormalize one row of top-k weights, then apply the routed scaling factor.
SGL_DEVICE void mimo_pro_finalize_row(
    float* row_weights, int64_t topk, bool renormalize, float routed_scaling_factor, bool apply_scaling) {
  float sum = 0.0f;
  for (int64_t k = 0; k < topk; ++k) {
    sum += row_weights[k];
  }
  const bool divide = renormalize && sum > 0.0f;
  for (int64_t k = 0; k < topk; ++k) {
    float weight = row_weights[k];
    if (divide) {
      weight /= sum;
    }
    if (apply_scaling) {
      weight *= routed_scaling_factor;
    }
    row_weights[k] = weight;
  }
}

/**
 * \brief Small-batch kernel: one CTA per token, one thread per expert.
 *
 * \tparam T                     Router logit type: fp32_t | fp16_t | bf16_t
 * \param input                  Router logits, [num_tokens, 384]
 * \param bias                   Correction bias, [384] fp32
 * \param topk_weights           Output, [num_tokens, topk] fp32 (unbiased sigmoid scores)
 * \param topk_ids               Output, [num_tokens, topk] int32
 * \param topk                   Experts selected per token
 * \param renormalize            Divide the selected weights by their sum
 * \param routed_scaling_factor  Factor applied when `apply_scaling` is set
 * \param apply_scaling          Multiply the weights by `routed_scaling_factor`
 */
template <typename T>
__global__ __launch_bounds__(kMimoProNumExperts) void mimo_pro_moe_fused_gate_small_token_kernel(
    const T* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int64_t topk,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_scaling) {
  __shared__ float biased_scores[kMimoProNumExperts];
  __shared__ float scores[kMimoProNumExperts];
  __shared__ float warp_best_scores[kMimoProWarpsPerTokenSmall];
  __shared__ int32_t warp_best_experts[kMimoProWarpsPerTokenSmall];

  const int64_t row = blockIdx.x;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp_id = tid / kMimoProWarpSize;
  const int32_t lane_id = tid % kMimoProWarpSize;

  const float logit = device::cast<fp32_t>(input[row * kMimoProNumExperts + tid]);
  const float score = 1.0f / (1.0f + expf(-logit));
  scores[tid] = score;
  biased_scores[tid] = score + bias[tid];
  __syncthreads();

  float* row_weights = topk_weights + row * topk;
  int32_t* row_ids = topk_ids + row * topk;
  for (int64_t k = 0; k < topk; ++k) {
    float best_score = -FLT_MAX;
    int32_t best_expert = -1;
    if (biased_scores[tid] > best_score) {
      best_score = biased_scores[tid];
      best_expert = tid;
    }
    mimo_pro_warp_argmax(best_score, best_expert);
    if (lane_id == 0) {
      warp_best_scores[warp_id] = best_score;
      warp_best_experts[warp_id] = best_expert;
    }
    __syncthreads();

    if (warp_id == 0) {
      best_score = -FLT_MAX;
      best_expert = -1;
      if (lane_id < kMimoProWarpsPerTokenSmall) {
        best_score = warp_best_scores[lane_id];
        best_expert = warp_best_experts[lane_id];
      }
      mimo_pro_warp_argmax(best_score, best_expert);
      if (lane_id == 0 && best_expert != -1) {
        row_weights[k] = scores[best_expert];
        row_ids[k] = best_expert;
        biased_scores[best_expert] = -FLT_MAX;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    mimo_pro_finalize_row(row_weights, topk, renormalize, routed_scaling_factor, apply_scaling);
  }
}

/**
 * \brief Large-batch kernel: one warp per token, `kMimoProWarpsPerCta` tokens per CTA.
 *
 * Each warp only touches its own shared-memory rows, so warp-level syncs suffice
 * and warps past `num_rows` can exit early.
 *
 * \tparam T  Router logit type: fp32_t | fp16_t | bf16_t
 * \param num_rows  Number of tokens; other parameters as in the small-batch kernel
 */
template <typename T>
__global__ __launch_bounds__(kMimoProWarpSize* kMimoProWarpsPerCta) void mimo_pro_moe_fused_gate_kernel(
    const T* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_scaling) {
  __shared__ float biased_scores_smem[kMimoProWarpsPerCta][kMimoProNumExperts];
  __shared__ float scores_smem[kMimoProWarpsPerCta][kMimoProNumExperts];

  const int32_t lane_id = static_cast<int32_t>(threadIdx.x);
  const int32_t warp_id = static_cast<int32_t>(threadIdx.y);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kMimoProWarpsPerCta + warp_id;
  if (row >= num_rows) {
    return;
  }

  float* biased_scores = biased_scores_smem[warp_id];
  float* scores = scores_smem[warp_id];
  const T* row_input = input + row * kMimoProNumExperts;
  for (int32_t expert = lane_id; expert < kMimoProNumExperts; expert += kMimoProWarpSize) {
    const float score = 1.0f / (1.0f + expf(-device::cast<fp32_t>(row_input[expert])));
    scores[expert] = score;
    biased_scores[expert] = score + bias[expert];
  }
  __syncwarp();

  float* row_weights = topk_weights + row * topk;
  int32_t* row_ids = topk_ids + row * topk;
  for (int64_t k = 0; k < topk; ++k) {
    float best_score = -FLT_MAX;
    int32_t best_expert = -1;
    for (int32_t expert = lane_id; expert < kMimoProNumExperts; expert += kMimoProWarpSize) {
      if (biased_scores[expert] > best_score) {
        best_score = biased_scores[expert];
        best_expert = expert;
      }
    }
    mimo_pro_warp_argmax(best_score, best_expert);
    if (lane_id == 0 && best_expert != -1) {
      row_weights[k] = scores[best_expert];
      row_ids[k] = best_expert;
      biased_scores[best_expert] = -FLT_MAX;
    }
    __syncwarp();
  }

  if (lane_id == 0) {
    mimo_pro_finalize_row(row_weights, topk, renormalize, routed_scaling_factor, apply_scaling);
  }
}

/**
 * \brief Validate the tensors and launch the MiMo-V2 Pro fused gate.
 *
 * \tparam T                     Router logit type: fp32_t | fp16_t | bf16_t
 * \param input                  Router logits, [num_tokens, 384], contiguous
 * \param bias                   Correction bias, [384] fp32
 * \param topk_weights           Output, [num_tokens, topk] fp32
 * \param topk_ids               Output, [num_tokens, topk] int32
 * \param renormalize            Divide the selected weights by their sum
 * \param routed_scaling_factor  Factor applied when `apply_scaling` is set
 * \param apply_scaling          Multiply the weights by `routed_scaling_factor`
 */
template <typename T>
void mimo_pro_moe_fused_gate(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_ids,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_scaling) {
  using namespace host;

  auto num_tokens_ = SymbolicSize{"num_tokens"};
  auto topk_ = SymbolicSize{"topk"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({num_tokens_, kMimoProNumExperts}).with_dtype<T>().with_device(device_).verify(input);
  TensorMatcher({kMimoProNumExperts}).with_dtype<fp32_t>().with_device(device_).verify(bias);
  TensorMatcher({num_tokens_, topk_}).with_dtype<fp32_t>().with_device(device_).verify(topk_weights);
  TensorMatcher({num_tokens_, topk_}).with_dtype<int32_t>().with_device(device_).verify(topk_ids);

  const int64_t num_tokens = num_tokens_.unwrap();
  const int64_t topk = topk_.unwrap();
  CHECK_HOST(topk > 0 && topk <= kMimoProNumExperts)
      << "mimo_pro_moe_fused_gate: topk must be in [1, " << kMimoProNumExperts << "], got " << topk;
  if (num_tokens == 0) {
    return;
  }

  const DLDevice device = device_.unwrap();
  const auto* input_ptr = static_cast<const T*>(input.data_ptr());
  const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  auto* weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  auto* ids_ptr = static_cast<int32_t*>(topk_ids.data_ptr());
  const auto scale = static_cast<float>(routed_scaling_factor);

  if (num_tokens <= kMimoProSmallTokenThreshold) {
    LaunchKernel(dim3(static_cast<uint32_t>(num_tokens)), dim3(kMimoProNumExperts), device)(
        mimo_pro_moe_fused_gate_small_token_kernel<T>,
        input_ptr,
        bias_ptr,
        weights_ptr,
        ids_ptr,
        topk,
        renormalize,
        scale,
        apply_scaling);
  } else {
    const auto num_ctas = static_cast<uint32_t>(div_ceil(num_tokens, int64_t{kMimoProWarpsPerCta}));
    LaunchKernel(dim3(num_ctas), dim3(kMimoProWarpSize, kMimoProWarpsPerCta), device)(
        mimo_pro_moe_fused_gate_kernel<T>,
        input_ptr,
        bias_ptr,
        weights_ptr,
        ids_ptr,
        num_tokens,
        topk,
        renormalize,
        scale,
        apply_scaling);
  }
}

}  // namespace sglang
