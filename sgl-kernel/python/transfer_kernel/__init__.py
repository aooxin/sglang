import torch
from sgl_kernel.load_utils import _preload_cuda_library
from transfer_kernel.load_utils import load_transfer_kernel_ops

# Initialize the ops library based on current GPU
load_transfer_kernel_ops()

# Preload the CUDA library to avoid the issue of libcudart.so.12 not found
if torch.version.cuda is not None:
    _preload_cuda_library()


from transfer_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_fuse_lf_pf,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
