"""SM-driven H2D fill for small int32/int64 buffers (off the DMA copy engine).

Copies a contiguous PINNED host buffer into a CUDA tensor via a real CUDA
kernel launched on the caller's current stream, instead of a
``cudaMemcpyAsync`` on the DMA copy engine.
"""

from __future__ import annotations

from collections import deque

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

_WRAPPERS = [
    ("fill_i32_from_pinned", "fill_from_pinned_impl<int32_t>"),
    ("fill_i64_from_pinned", "fill_from_pinned_impl<int64_t>"),
]

_PENDING_PINNED_SOURCES: deque[tuple[torch.cuda.Event, torch.Tensor]] = deque()


def _retain_pinned_source(src: torch.Tensor, stream: torch.cuda.Stream) -> None:
    while _PENDING_PINNED_SOURCES and _PENDING_PINNED_SOURCES[0][0].query():
        _PENDING_PINNED_SOURCES.popleft()
    event = torch.cuda.Event()
    event.record(stream)
    _PENDING_PINNED_SOURCES.append((event, src))


@cache_once
def _jit_fill_from_pinned_module():
    return load_jit(
        "fill_from_pinned",
        cuda_files=["elementwise/fill_from_pinned.cuh"],
        cuda_wrappers=_WRAPPERS,
    )


def fill_from_pinned(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy ``src`` (pinned host) into ``dst`` (CUDA) via an SM kernel.

    Both tensors must be 1D, contiguous, of equal length, and the same dtype
    (int32 or int64). ``src`` MUST be page-locked (pinned) host memory. Launched
    on the current CUDA stream.
    """
    if not dst.is_cuda:
        raise ValueError("fill_from_pinned: dst must be a CUDA tensor")
    if src.is_cuda:
        raise ValueError("fill_from_pinned: src must be a host tensor")
    if not src.is_pinned():
        raise ValueError("fill_from_pinned: src must be pinned host memory")
    if dst.dtype != src.dtype:
        raise ValueError("fill_from_pinned: dst and src dtype mismatch")
    if not (dst.is_contiguous() and src.is_contiguous()):
        raise ValueError("fill_from_pinned: both dst and src must be contiguous")
    if dst.numel() != src.numel():
        raise ValueError("fill_from_pinned: dst and src length mismatch")

    module = _jit_fill_from_pinned_module()
    if dst.dtype == torch.int32:
        module.fill_i32_from_pinned(dst, src.data_ptr())
    elif dst.dtype == torch.int64:
        module.fill_i64_from_pinned(dst, src.data_ptr())
    else:
        raise TypeError(f"fill_from_pinned: unsupported dtype {dst.dtype}")
    _retain_pinned_source(src, torch.cuda.current_stream(dst.device))
