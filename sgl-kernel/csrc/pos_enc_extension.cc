/* PyTorch 扩展注册 - 只注册 rotary_embedding (pos_enc.cu) */
#include <torch/extension.h>
#include <torch/library.h>

// 声明 rotary_embedding 函数 (实现在 csrc/elementwise/pos_enc.cu)
void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

// 注册到 PyTorch - 使用独立的命名空间避免与 sgl_kernel 冲突
TORCH_LIBRARY(pos_enc_kernel, m) {
  m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
}

// Python 模块初始化
PyMODINIT_FUNC PyInit_pos_enc_ops() {
  static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "pos_enc_ops", nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
