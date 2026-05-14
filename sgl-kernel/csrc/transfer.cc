/* PyTorch 扩展注册 - 只注册 transfer_kernel_impl (transfer.cu) */
#include <torch/extension.h>
#include <torch/library.h>
void transfer_kv_per_layer(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_pf_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_ph_lf(
    const at::Tensor src_k,
    at::Tensor dst_k,
    const at::Tensor src_v,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer(
    const at::Tensor src_k_layers,
    const at::Tensor dst_k_layers,
    const at::Tensor src_v_layers,
    const at::Tensor dst_v_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_lf_pf(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_lf_ph(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_fuse_lf_pf(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t k_item_size,
    int64_t k_dst_layout_dim,
    int64_t v_item_size,
    int64_t v_dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla_pf_lf(
    const at::Tensor src,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla(
    const at::Tensor src_layers,
    const at::Tensor dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla_lf_pf(
    const at::Tensor src_layers,
    at::Tensor dst,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_direct(
    const std::vector<at::Tensor>& src_layers,
    std::vector<at::Tensor> dst_layers,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t page_size);

void transfer_kv_per_layer_direct_pf_lf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t layer_id,
    int64_t page_size);

void transfer_kv_all_layer_direct_lf_pf(
    const std::vector<at::Tensor>& src_ptrs,
    std::vector<at::Tensor> dst_ptrs,
    const at::Tensor& src_indices,
    const at::Tensor& dst_indices,
    int64_t page_size);

// 注册到 PyTorch - 使用独立的命名空间避免与 sgl_kernel 冲突
TORCH_LIBRARY(transfer_kernel, m) {
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer", torch::kCUDA, &transfer_kv_per_layer);
  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_pf_lf", torch::kCUDA, &transfer_kv_per_layer_pf_lf);
  m.def(
      "transfer_kv_per_layer_ph_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int page_size, int head_num, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_ph_lf", torch::kCUDA, &transfer_kv_per_layer_ph_lf);
  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer", torch::kCUDA, &transfer_kv_all_layer);
  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_pf", torch::kCUDA, &transfer_kv_all_layer_lf_pf);
  m.def(
      "transfer_kv_all_layer_lf_ph(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int page_size, int "
      "head_num, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_lf_ph", torch::kCUDA, &transfer_kv_all_layer_lf_ph);
  m.def(
      "transfer_kv_all_layer_fuse_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int k_item_size, int k_dst_layout_dim, int v_item_size, int "
      "v_dst_layout_dim, "
      "int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_fuse_lf_pf", torch::kCUDA, &transfer_kv_all_layer_fuse_lf_pf);
  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int "
      "block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla", torch::kCUDA, &transfer_kv_per_layer_mla);
  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, "
      "int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_per_layer_mla_pf_lf", torch::kCUDA, &transfer_kv_per_layer_mla_pf_lf);
  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla", torch::kCUDA, &transfer_kv_all_layer_mla);
  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, "
      "int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.impl("transfer_kv_all_layer_mla_lf_pf", torch::kCUDA, &transfer_kv_all_layer_mla_lf_pf);
  m.def(
      "transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "page_size) -> ()");
  m.impl("transfer_kv_direct", torch::kCUDA, &transfer_kv_direct);
  m.def(
      "transfer_kv_per_layer_direct_pf_lf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int layer_id, int page_size)->() ");
  m.impl("transfer_kv_per_layer_direct_pf_lf", torch::kCUDA, &transfer_kv_per_layer_direct_pf_lf);
  m.def(
      "transfer_kv_all_layer_direct_lf_pf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int page_size) ->() ");
  m.impl("transfer_kv_all_layer_direct_lf_pf", torch::kCUDA, &transfer_kv_all_layer_direct_lf_pf);
}

// Python 模块初始化
PyMODINIT_FUNC PyInit_transfer_ops() {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "transfer_ops", nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
