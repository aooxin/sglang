"""pos_enc_kernel - Standalone Rotary Position Encoding CUDA Kernel"""

import torch

# 导入编译的扩展模块 (触发 TORCH_LIBRARY 注册)
from pos_enc_kernel import pos_enc_ops  # noqa: F401


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
):
    torch.ops.pos_enc_kernel.rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


__all__ = ["rotary_embedding"]
__version__ = "0.1.0"
