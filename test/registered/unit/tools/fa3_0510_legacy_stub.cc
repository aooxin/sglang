// CPU test backend with the exact 0510 FA3 entry signatures. The production
// registration and compatibility shim are compiled unchanged alongside it.
#include "fa3_0510_compat.h"
#include "sgl_flash_kernel_ops.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> k_new_,
    std::optional<at::Tensor> v_new_, std::optional<at::Tensor> q_v_,
    std::optional<at::Tensor> out_, std::optional<at::Tensor> cu_seqlens_q_,
    std::optional<at::Tensor> cu_seqlens_k_,
    std::optional<at::Tensor> cu_seqlens_k_new_,
    std::optional<at::Tensor> seqused_q_, std::optional<at::Tensor> seqused_k_,
    std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,
    std::optional<at::Tensor> kv_batch_idx_,
    std::optional<at::Tensor> leftpad_k_, std::optional<at::Tensor> rotary_cos_,
    std::optional<at::Tensor> rotary_sin_,
    std::optional<at::Tensor> seqlens_rotary_,
    std::optional<at::Tensor> q_descale_, std::optional<at::Tensor> k_descale_,
    std::optional<at::Tensor> v_descale_, std::optional<double> softmax_scale_,
    bool is_causal, int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, double softcap, bool is_rotary_interleaved,
    std::optional<at::Tensor> scheduler_metadata_, int64_t num_splits,
    std::optional<bool> pack_gqa_, int64_t sm_margin,
    std::optional<const at::Tensor> &sinks_) {
  TORCH_CHECK(num_splits == 3 && pack_gqa_.value_or(false) && sm_margin == 2,
              "Lost scheduling arguments");
  TORCH_CHECK(is_causal && window_size_left == 7 && window_size_right == 0,
              "Lost masking arguments");
  TORCH_CHECK(attention_chunk == 4 && softcap == 1.5 && !is_rotary_interleaved,
              "Lost scalar arguments");
  auto out = out_.has_value() ? out_.value() : at::empty_like(q);
  out.copy_(q * softmax_scale_.value_or(1.0) + k + v);
  return {out, q, scheduler_metadata_.value_or(k),
          sinks_.has_value() ? sinks_.value() : v};
}

at::Tensor mha_fwd_get_scheduler_metadata(
    int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t num_heads, int64_t num_heads_k, int64_t headdim, int64_t headdim_v,
    at::ScalarType qkv_dtype, at::Tensor seqused_k,
    std::optional<at::Tensor> cu_seqlens_q_,
    std::optional<at::Tensor> cu_seqlens_k_,
    std::optional<at::Tensor> cu_seqlens_k_new_,
    std::optional<at::Tensor> seqused_q_, std::optional<at::Tensor> leftpad_k_,
    std::optional<int64_t> page_size, int64_t max_seqlen_k_new, bool is_causal,
    int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, bool has_softcap, int64_t num_splits,
    std::optional<bool> pack_gqa_, int64_t sm_margin) {
  return seqused_k.clone();
}

// The real extension registers CUDA kernels. Add a CPU dispatch solely for
// exercising its same C++ adapter without CUDA hardware or a CUDA build.
TORCH_LIBRARY_IMPL(sgl_kernel, CPU, m) {
  m.impl("fwd", make_fa3_0510_pytorch_shim(&mha_fwd));
  m.impl("get_scheduler_metadata",
         make_pytorch_shim(&mha_fwd_get_scheduler_metadata));
}
