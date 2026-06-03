// Adapted from
// https://github.com/vllm-project/vllm/blob/014ece97c7aa49084a1119dca792af081a18dbc1/csrc/pos_encoding_kernels.cu

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  cache_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = SGLANG_LDG(cos_ptr + x_index);
    sin = SGLANG_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = SGLANG_LDG(cos_ptr + x_index / 2);
    sin = SGLANG_LDG(sin_ptr + x_index / 2);
  }

  // Upcast to cache_t for computation, then downcast to scalar_t for output
  const cache_t x = static_cast<cache_t>(arr[x_index]);
  const cache_t y = static_cast<cache_t>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x * cos - y * sin);
  arr[y_index] = static_cast<scalar_t>(y * cos + x * sin);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const cache_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t query_head_stride,
    const int64_t key_head_stride) {
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * query_head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head = token_idx * key_stride + head_idx * key_head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,      // [batch_size, seq_len] or
                                                // [num_tokens]
    scalar_t* __restrict__ query,               // [batch_size, seq_len, num_heads,
                                                // head_size] or [num_tokens, num_heads,
                                                // head_size]
    scalar_t* __restrict__ key,                 // nullptr or
                                                // [batch_size, seq_len, num_kv_heads,
                                                // head_size] or [num_tokens, num_kv_heads,
                                                // head_size]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t query_head_stride,
    const int64_t key_head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int64_t max_position) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];

  // TODO(yingchun): 这里应该 TORCH_CHECK
  // Bounds check for position to prevent illegal memory access
  if (pos < 0 || pos >= max_position) {
    return;
  }

  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, cache_t, IS_NEOX>(
      query,
      key,
      cache_ptr,
      head_size,
      num_heads,
      num_kv_heads,
      rot_dim,
      token_idx,
      query_stride,
      key_stride,
      query_head_stride,
      key_head_stride);
}

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,      // [batch_size, seq_len, num_heads * head_size] or
                               // [num_tokens, num_heads * head_size] or
                               // [batch_size, seq_len, num_heads, head_size] or
                               // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2, "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) && (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0, "query hidden size must be divisible by head_size");
  TORCH_CHECK(key_hidden_size % head_size == 0, "key hidden size must be divisible by head_size");

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

  int rot_dim = cos_sin_cache.size(1);
  int64_t max_position = cos_sin_cache.size(0);

  // Check rot_dim does not exceed head_size to prevent out-of-bounds access
  TORCH_CHECK(rot_dim <= head_size, "rot_dim (", rot_dim, ") must be <= head_size (", head_size, ")");

  // TODO(yingchun): 对非连续输入执行 .contiguous() + kernel + .copy_() 相当于 3 倍内存带宽开销。
  //  建议在调用方保证连续性或在 kernel 中直接支持 strided 访问。
  // Handle non-contiguous tensors by making contiguous copies
  // We'll copy results back to original tensors after processing
  torch::Tensor original_query;
  torch::Tensor original_key;
  bool query_was_contiguous = query.is_contiguous();
  bool key_was_contiguous = true;

  if (!query_was_contiguous) {
    original_query = query;
    query = query.contiguous();
  }
  if (key.has_value() && !key->is_contiguous()) {
    key_was_contiguous = false;
    original_key = key.value();
    key = key->contiguous();
  }

  TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache tensor must be contiguous");

  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;

  // Determine head stride separately for query and key
  // For [*, heads, head_size] use stride of heads dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size head_size
  int query_ndim = query.dim();
  int key_ndim = key.has_value() ? key->dim() : query_ndim;
  int64_t query_head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;
  int64_t key_head_stride =
      key.has_value() ? ((key_ndim == positions_ndim + 2) ? key->stride(-2) : head_size) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Nested dispatch: outer for query/key type, inner for cos_sin_cache type
  DISPATCH_FLOAT_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    using qk_scalar_t = scalar_t;
    DISPATCH_FLOAT_TYPES(cos_sin_cache.scalar_type(), "rotary_embedding_cache", [&] {
      using cache_scalar_t = scalar_t;
      if (is_neox) {
        rotary_embedding_kernel<qk_scalar_t, cache_scalar_t, true><<<grid, block, 0, stream>>>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<qk_scalar_t>(),
            key.has_value() ? key->data_ptr<qk_scalar_t>() : nullptr,
            cos_sin_cache.data_ptr<cache_scalar_t>(),
            rot_dim,
            query_stride,
            key_stride,
            query_head_stride,
            key_head_stride,
            num_heads,
            num_kv_heads,
            head_size,
            max_position);
      } else {
        rotary_embedding_kernel<qk_scalar_t, cache_scalar_t, false><<<grid, block, 0, stream>>>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<qk_scalar_t>(),
            key.has_value() ? key->data_ptr<qk_scalar_t>() : nullptr,
            cos_sin_cache.data_ptr<cache_scalar_t>(),
            rot_dim,
            query_stride,
            key_stride,
            query_head_stride,
            key_head_stride,
            num_heads,
            num_kv_heads,
            head_size,
            max_position);
      }
    });
  });

  // Copy results back to original tensors if they were non-contiguous
  if (!query_was_contiguous) {
    original_query.copy_(query);
  }
  if (!key_was_contiguous) {
    original_key.copy_(key.value());
  }
}
