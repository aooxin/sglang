import glob
import logging
from pathlib import Path
from typing import List, Optional

import torch
from sgl_kernel.load_utils import _filter_compiled_extensions

logger = logging.getLogger(__name__)


def load_transfer_kernel_ops() -> None:
    """
    Load transfer_kernel TORCH_LIBRARY shared object.

    This function:
    - Locates transfer_kernel *.so
    - Loads it via torch.ops.load_library
    - Ensures transfer_kernel ops are registered
    """
    # Directory of current python file (e.g. transfer_kernel/)
    base_dir = Path(__file__).parent
    logger.debug(f"[transfer_kernel] base dir: {base_dir}")

    # Patterns to try (按优先级)
    patterns = [
        # 正常安装后的结构
        str(base_dir / "transfer_ops*.so"),
    ]

    so_candidates: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        so_candidates.extend(matches)
        logger.debug(f"[transfer_kernel] glob {pattern} -> {matches}")

    # 过滤 & 排序（复用你已有的逻辑思想）
    so_candidates = _filter_compiled_extensions(so_candidates)

    if not so_candidates:
        raise ImportError(
            f"[transfer_kernel] No transfer_ops shared library found. "
            f"Tried patterns: {patterns}"
        )

    load_errors: List[Exception] = []

    for so_path in so_candidates:
        try:
            so_path = str(so_path)
            logger.debug(f"[transfer_kernel] Loading shared library: {so_path}")

            torch.ops.load_library(so_path)

            # 关键校验：确保 op 已注册
            if not hasattr(torch.ops, "transfer_kernel"):
                raise RuntimeError("torch.ops.transfer_kernel namespace not found")

            # 至少校验一个你肯定存在的 op
            if not hasattr(torch.ops.transfer_kernel, "transfer_kv_per_layer_mla"):
                raise RuntimeError(
                    "transfer_kernel loaded but op "
                    "'transfer_kv_per_layer_mla' not found"
                )

            logger.debug("[transfer_kernel] ✓ transfer_kernel ops loaded successfully")
            return

        except Exception as e:
            load_errors.append(e)
            logger.debug(
                f"[transfer_kernel] ✗ Failed to load {so_path}: "
                f"{type(e).__name__}: {e}"
            )

    error_msg = "\n".join(f"- {type(err).__name__}: {err}" for err in load_errors)
    raise ImportError(
        "[transfer_kernel] CRITICAL: Failed to load transfer_kernel ops.\n"
        f"Errors:\n{error_msg}"
    )
