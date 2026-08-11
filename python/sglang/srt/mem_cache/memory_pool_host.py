from __future__ import annotations

import abc
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

import numpy as np
import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.nvtx_utils import nvtx_annotated_method
from sglang.srt.observability.startup_func_log_and_timer import startup_timer
from sglang.srt.utils import is_cuda, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if not (_is_npu or _is_xpu or _is_mps):
    try:
        from transfer_kernel.kvcacheio import (
            transfer_kv_all_layer,
            transfer_kv_all_layer_direct_lf_pf,
            transfer_kv_all_layer_fuse_lf_pf,
            transfer_kv_all_layer_lf_pf,
            transfer_kv_all_layer_lf_ph,
            transfer_kv_all_layer_mla,
            transfer_kv_all_layer_mla_lf_pf,
            transfer_kv_direct,
            transfer_kv_per_layer,
            transfer_kv_per_layer_direct_pf_lf,
            transfer_kv_per_layer_mla,
            transfer_kv_per_layer_mla_pf_lf,
            transfer_kv_per_layer_pf_lf,
            transfer_kv_per_layer_ph_lf,
        )
    except ImportError:
        # transfer_kernel is a GPU-only native extension. Allow CPU/import-only
        # contexts (unit tests, docs builds) to load this module; any code that
        # actually invokes these symbols will get a NameError at call time,
        # which is the right failure mode for a missing GPU runtime dep.
        transfer_kv_all_layer = None
        transfer_kv_all_layer_direct_lf_pf = None
        transfer_kv_all_layer_fuse_lf_pf = None
        transfer_kv_all_layer_lf_pf = None
        transfer_kv_all_layer_lf_ph = None
        transfer_kv_all_layer_mla = None
        transfer_kv_all_layer_mla_lf_pf = None
        transfer_kv_direct = None
        transfer_kv_per_layer = None
        transfer_kv_per_layer_direct_pf_lf = None
        transfer_kv_per_layer_mla = None
        transfer_kv_per_layer_mla_pf_lf = None
        transfer_kv_per_layer_pf_lf = None
        transfer_kv_per_layer_ph_lf = None
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange


logger = logging.getLogger(__name__)

# Cached once at import: gating envs.SGLANG_SWA_DIAG_ENABLED.get() per alloc/free
# would add an os.getenv lookup to the hot path, defeating the purpose.
_SWA_DIAG_ENABLED = envs.SGLANG_SWA_DIAG_ENABLED.get()
# Hot-path subset (in-lock ring checks, per-free mapping scans): these lengthen
# allocator critical sections and perturb thread interleaving, so they need an
# explicit second opt-in on top of the base flag.
_SWA_DIAG_HOT_ENABLED = _SWA_DIAG_ENABLED and envs.SGLANG_SWA_DIAG_HOT_ENABLED.get()


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class _FenceSection:
    """Reusable ``with`` target for a _CudaStreamFence.

    Implemented as a plain __enter__/__exit__ class (not @contextmanager) to
    avoid creating a generator object per entry on the alloc/free/gather hot
    paths. One instance is created per fence and reused; safe because it holds
    no per-entry state (the fence owns all state, and the caller must hold the
    owner's lock).
    """

    __slots__ = ("_fence",)

    def __init__(self, fence: "_CudaStreamFence"):
        self._fence = fence

    def __enter__(self):
        self._fence._acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fence._release()
        return False


class _CudaStreamFence:
    """Serialize async CUDA ops on a shared GPU tensor across threads/streams.

    Why: the host-pool bookkeeping tensors (``_free_slots_buf``,
    ``full_to_swa_index_mapping``) are CUDA-resident, but are touched by the
    scheduler thread (running inside ``schedule_stream``) and the HiCache
    prefetch / IO-aux threads (running on the default stream). A Python lock
    only serializes kernel SUBMISSION, not EXECUTION: a slice ``.clone()``
    enqueued on one stream may still be in flight when another thread reads
    or overwrites the same region from a different stream, producing torn
    reads/writes (observed in production as the
    "Each page must be all-zero or all-valid" release asserts).

    Protocol: every op touching the guarded tensor runs inside
    ``with fence.order():`` while holding the owner's Python lock. Entering
    makes the current thread's stream wait on the event recorded by the
    previous op (possibly recorded on a different stream); exiting re-records
    the event on the current stream. Combined with the lock, this yields a
    total cross-stream execution order matching lock-acquisition order.

    Deliberately total-order (reads included): a reader/writer split would
    allow concurrent gathers, but these bookkeeping ops are low-QPS
    page-granularity kernels where the win is negligible against the extra
    protocol complexity (reader-event tracking, write-after-read hazards).

    Re-recording the single pooled event is safe here: re-record happens only
    inside the lock, after every consumer of the previous recording has
    already submitted its wait (waits bind to the recording at submission
    time).

    Device pinning: torch.cuda's "current device" is THREAD-LOCAL. Background
    threads (prefetch/IO) default to cuda:0 even when the pool tensor lives on
    cuda:N, so ``torch.cuda.current_stream()`` without a device argument would
    return cuda:0's stream there — waiting on the wrong stream and raising
    "Event device does not match recording stream's device" on record. All
    stream lookups therefore explicitly use the guarded tensor's device: ops
    on a cuda:N tensor are enqueued on the calling thread's current stream
    FOR cuda:N regardless of the thread's current device.
    """

    __slots__ = ("_enabled", "_device", "_event", "_recorded", "_section")

    def __init__(self, device) -> None:
        device = torch.device(device)
        self._enabled = device.type == "cuda" and torch.cuda.is_available()
        # A bare "cuda" (no index) would be re-resolved against each calling
        # thread's thread-local current device — background threads would get
        # cuda:0 again. Pin the index NOW, on the constructing (scheduler)
        # thread, where the current device is the correct one.
        if self._enabled and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        self._event: Optional[torch.cuda.Event] = None
        self._recorded = False
        self._section = _FenceSection(self)

    def order(self) -> _FenceSection:
        return self._section

    def _acquire(self):
        if not self._enabled:
            return
        if self._recorded:
            torch.cuda.current_stream(self._device).wait_event(self._event)

    def _release(self):
        if not self._enabled:
            return
        if self._event is None:
            with torch.cuda.device(self._device):
                self._event = torch.cuda.Event()
        self._event.record(torch.cuda.current_stream(self._device))
        self._recorded = True


class _LockedFenceSection:
    """Single ``with`` target combining the owner lock and a fence section,
    so call sites cannot forget one half of the pairing."""

    __slots__ = ("_lock", "_section")

    def __init__(self, lock: threading.RLock, section: _FenceSection):
        self._lock = lock
        self._section = section

    def __enter__(self):
        self._lock.acquire()
        try:
            self._section.__enter__()
        except BaseException:
            self._lock.release()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._section.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._lock.release()


def log_subpage_swa_pattern(
    valid_mask: torch.Tensor,
    page_size: int,
    where: str,
    extra: Union[str, Callable[[], str]] = "",
) -> bool:
    """Diagnostic (log-only): token-granular SWA validity patterns must be
    page-uniform — every page all-True or all-False. A sub-page pattern here is
    the direct generator of "mixed" full_to_swa_index_mapping pages, which later
    blow up the all-zero-or-all-valid release asserts (cache_controller
    append_swa_only_host_mem_release / assert_host_indices).

    ``extra`` may be a callable for lazy construction: it is only evaluated on
    the error path, so call sites can reference expensive data (e.g. ``.tolist()``
    of a CUDA tensor, which forces a D2H sync) without paying on the happy path.

    Returns True when a violation was found (and logged with stack).
    """
    if not _SWA_DIAG_ENABLED:
        return False
    n = valid_mask.numel()
    if n == 0:
        return False
    if n % page_size != 0:
        logger.error(
            "[SWA-DIAG] %s: length %d not a multiple of page_size %d. %s",
            where,
            n,
            page_size,
            extra() if callable(extra) else extra,
            stack_info=True,
        )
        return True
    per_page = valid_mask.view(-1, page_size).sum(dim=1)
    bad = (per_page != 0) & (per_page != page_size)
    if not bool(bad.any()):
        return False
    bad_idx = bad.nonzero(as_tuple=False).view(-1)
    counts = per_page.tolist()
    logger.error(
        "[SWA-DIAG] %s: sub-page SWA pattern detected. num_pages=%d "
        "bad_pages(idx)=%s per_page_valid_counts=%s%s "
        "first_bad_page_raw=%s %s",
        where,
        len(counts),
        bad_idx.tolist()[:64],
        counts[:256],
        "...(truncated)" if len(counts) > 256 else "",
        valid_mask.view(-1, page_size)[bad_idx[0]].tolist(),
        extra() if callable(extra) else extra,
        stack_info=True,
    )
    return True


def assert_page_uniform_swa_pattern(
    valid_mask: torch.Tensor,
    page_size: int,
    where: str,
    extra: Union[str, Callable[[], str]] = "",
) -> None:
    """Fail-fast variant of log_subpage_swa_pattern for deterministic mapping
    writes / input masks: a sub-page pattern at these sites is an actual bug
    (not a concurrent torn read), so raise at the source instead of letting the
    mixed page propagate to a much later release assert.
    """
    if log_subpage_swa_pattern(valid_mask, page_size, where, extra):
        raise AssertionError(
            f"[SWA-DIAG] {where}: sub-page SWA pattern at source; "
            f"mapping write would produce mixed full_to_swa pages"
        )


def assert_indices_in_bounds(
    indices: torch.Tensor,
    page_size: int,
    upper_exclusive: int,
    where: str,
    allow_zero: bool = False,
    hot: bool = False,
) -> None:
    """[SWA-DIAG] Fail-fast slot-value domain check before indices are turned
    into raw pointers, kernel offsets, or handed across threads.

    The free-slot rings are initialized as ``arange(page_size, size +
    page_size)``, so every legal slot value lies in ``[page_size,
    upper_exclusive)`` -- plus 0 for mapping-gathered inputs where unmapped
    pages legitimately read as 0 (allow_zero=True). Anything else means the
    index tensor was corrupted upstream (2026-08-03 production incident: a
    full ``arange(0, page_size)`` page, i.e. the content of some *other*
    tensor written over ours by a foreign/wild write). The page_size lower
    bound matters: a plain ``[0, pool_size)`` bounds check would pass 63 of
    those 64 corrupted values.

    Letting a corrupt index through turns the downstream KV transfer into an
    out-of-bounds host write (memory stomp) that corrupts unrelated tensors.

    One aminmax + D2H sync per call. Cold call sites (per IO batch / per
    prefetch operation) use the base SGLANG_SWA_DIAG_ENABLED gate. Hot call
    sites (alloc exits inside the scheduler thread, latency-critical H2D load
    path) pass hot=True so the sync -- which stalls until pending async
    clones complete and perturbs thread interleaving -- is only incurred
    under the second opt-in SGLANG_SWA_DIAG_HOT_ENABLED.
    """
    if not (_SWA_DIAG_HOT_ENABLED if hot else _SWA_DIAG_ENABLED):
        return
    if indices.numel() == 0:
        return
    amin, amax = torch.aminmax(indices)
    min_val, max_val = torch.stack((amin, amax)).tolist()
    below_ok = min_val >= page_size or (allow_zero and min_val == 0)
    if below_ok and max_val < upper_exclusive:
        if not (allow_zero and min_val == 0):
            return
        # min == 0 proved only that SOME value is 0; values in (0, page_size)
        # may still hide between 0 and the max. Rescan precisely.
        interior_bad = (indices > 0) & (indices < page_size)
        if not bool(interior_bad.any()):
            return
    bad_mask = (indices < page_size) | (indices >= upper_exclusive)
    if allow_zero:
        bad_mask &= indices != 0
    if not bool(bad_mask.any()):
        return
    bad_pos = bad_mask.nonzero(as_tuple=False).view(-1)
    legal_domain = ("0 or " if allow_zero else "") + (
        f"[{page_size}, {upper_exclusive})"
    )
    logger.error(
        "[SWA-DIAG][%s] slot values outside legal domain %s: "
        "min=%d max=%d numel=%d num_bad=%d "
        "bad_positions(head)=%s bad_values(head)=%s indices_head=%s",
        where,
        legal_domain,
        min_val,
        max_val,
        indices.numel(),
        int(bad_mask.sum()),
        bad_pos[:64].tolist(),
        indices[bad_pos[:64]].tolist(),
        indices[:128].tolist(),
        stack_info=True,
    )
    raise AssertionError(
        f"[SWA-DIAG][{where}] slot values outside legal domain "
        f"{legal_domain} (min={min_val}, max={max_val}); foreign write into "
        f"allocator bookkeeping or index tensor suspected"
    )


class HostTensorAllocator(abc.ABC):
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor of given dims and dtype on the memory."""
        self.dtype = dtype
        self.dims = dims
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


# TODO(yingchun): unify to return Tuple[Tensor, Tensor]
def alloc_with_host_register(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    k_dims: tuple = None,
    v_dims: tuple = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    if dims is not None:
        assert k_dims is None
        assert v_dims is None
        buffer = allocator.allocate(dims, dtype=dtype, device="cpu")
        if pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
            )
    else:
        assert k_dims is not None
        assert v_dims is not None
        k_buffer = allocator.allocate(k_dims, dtype=dtype, device="cpu")
        v_buffer = allocator.allocate(v_dims, dtype=dtype, device="cpu")
        buffer = [k_buffer, v_buffer]
        if pin_memory:
            for cache in buffer:
                if cache.numel() > 0:
                    torch.cuda.cudart().cudaHostRegister(
                        cache.data_ptr(), cache.numel() * cache.element_size(), 0
                    )
    torch.cuda.synchronize()
    return buffer


def alloc_with_pin_memory(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    k_dims: tuple = None,
    v_dims: tuple = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    assert dims is not None, "dims should be provided"
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
    },
)


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.h2d_block_quota = envs.SGLANG_HICACHE_H2D_BLOCK_QUOTA.get()
        self.d2h_block_quota = envs.SGLANG_HICACHE_D2H_BLOCK_QUOTA.get()

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the usable host memory pool size to the page size. Host pools
        # reserve page 0 as padding and return usable slots from
        # [page_size, size + page_size), so the backing tensors need one extra
        # page even though self.size remains the usable capacity.
        self.size = self.size - (self.size % self.page_size)
        self.page_num = self.size // self.page_size
        self.alloc_size = self.size + self.page_size
        self.alloc_page_num = self.page_num + 1
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.alloc_size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self._ring_fence = _CudaStreamFence(self.device)
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.alloc_size,), dtype=torch.uint8, device=self.device
        )
        # Pre-allocate a fixed-size ring buffer for free slot management.
        # Allocations read from head, frees append at tail.  The explicit
        # count disambiguates full vs empty when head == tail and avoids
        # periodic O(pool size) compaction.
        with self._ring_fence.order():
            self._free_slots_buf = torch.arange(
                self.page_size,
                self.size + self.page_size,
                dtype=torch.int64,
                device=self.device,
            )
        self._free_slots_head = 0
        self._free_slots_tail = 0
        self._free_slots_count = len(self._free_slots_buf)

    def available_size(self):
        return self._free_slots_count

    # [SWA-DIAG] Human-readable label for this pool instance, set by the owning
    # SWATokenToKVPoolHost so alloc/free diagnostics can tell full vs swa apart.
    _debug_pool_label: str = "host_pool"

    # [SWA-DIAG] Compact snapshot of the free-slot ring's internal state, used
    # only inside diagnostic logging when an invariant is violated.
    def _debug_free_slots_state(self) -> str:
        cap = len(self._free_slots_buf)
        head = self._free_slots_head
        tail = self._free_slots_tail
        count = self._free_slots_count
        if count == 0:
            live = self._free_slots_buf[:0]
        elif head + count <= cap:
            live = self._free_slots_buf[head : head + count]
        else:
            first = cap - head
            live = torch.cat(
                (self._free_slots_buf[head:], self._free_slots_buf[: count - first])
            )
        num_zero = int((live == 0).sum()) if live.numel() > 0 else 0
        num_dup = (
            int(live.numel() - torch.unique(live).numel()) if live.numel() > 0 else 0
        )
        return (
            f"pool={self._debug_pool_label} size={self.size} page_size={self.page_size} "
            f"capacity={cap} head={head} tail={tail} available={count} "
            f"live_zeros={num_zero}(should be 0) live_duplicates={num_dup}(should be 0)"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if need_size > self._free_slots_count:
            return None

        capacity = len(self._free_slots_buf)
        end = self._free_slots_head + need_size
        with self._ring_fence.order():
            if end <= capacity:
                select_index = self._free_slots_buf[
                    self._free_slots_head : end
                ].clone()
            else:
                first = capacity - self._free_slots_head
                second = need_size - first
                select_index = torch.empty(
                    (need_size,), dtype=torch.int64, device=self.device
                )
                select_index[:first] = self._free_slots_buf[self._free_slots_head :]
                select_index[first:] = self._free_slots_buf[:second]

        self._free_slots_head = end % capacity
        self._free_slots_count -= need_size

        # [SWA-DIAG] Consumption-point ring integrity check: legal slot values
        # are >= page_size (ring is initialized with arange(page_size, ...)),
        # and every page-sized block handed out must be a consecutive +1 run
        # (all legitimate frees return whole contiguous pages). Since free()
        # rejects illegal values and broken runs at entry, a violation here
        # means the ring was corrupted through a channel other than the Python
        # writers (e.g. a wild device-side write into this CUDA-resident
        # buffer). Fail hard: handing out corrupted slots would silently write
        # KV data to wrong host offsets. Gated by the HOT flag: this runs
        # inside the allocator lock on every alloc.
        if _SWA_DIAG_HOT_ENABLED:
            pages = select_index.view(-1, self.page_size)
            runs_ok = bool((pages[:, 1:] - pages[:, :-1] == 1).all())
            min_val = int(pages[:, 0].min())
            if not runs_ok or min_val < self.page_size:
                bad_runs = (
                    ((pages[:, 1:] - pages[:, :-1]) != 1)
                    .any(dim=1)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
                logger.error(
                    "[SWA-DIAG] %s.alloc: corrupted free-slot ring. need_size=%d "
                    "min_page_start=%d bad_run_pages(idx)=%s first_bad_page=%s %s",
                    type(self).__name__,
                    need_size,
                    min_val,
                    bad_runs.tolist()[:64],
                    pages[bad_runs[0]].tolist() if len(bad_runs) > 0 else None,
                    self._debug_free_slots_state(),
                    stack_info=True,
                )
                raise AssertionError(
                    f"[SWA-DIAG] {type(self).__name__}.alloc: corrupted free-slot "
                    f"ring (need_size={need_size}, min_page_start={min_val}, "
                    f"runs_ok={runs_ok}); refusing to hand out corrupted slots"
                )

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        n = len(indices)
        if n == 0:
            return 0
        # [SWA-DIAG] Entry-point ring integrity check: freed values must be
        # legal slot ids (>= page_size), whole pages, and each page must be a
        # consecutive +1 run (all legitimate free paths preserve within-page
        # slot order). Letting violations through would permanently poison the
        # ring and defer the failure to a much later alloc() with the culprit's
        # stack long gone, so this one must fail hard (dump first, then raise).
        # Gated by the HOT flag: this runs inside the allocator lock on every
        # free.
        if _SWA_DIAG_HOT_ENABLED:
            illegal_shape = n % self.page_size != 0
            min_val = int(indices.min())
            runs_ok = True
            if not illegal_shape:
                in_pages = indices.view(-1, self.page_size)
                runs_ok = bool((in_pages[:, 1:] - in_pages[:, :-1] == 1).all())
            if illegal_shape or min_val < self.page_size or not runs_ok:
                logger.error(
                    "[SWA-DIAG] %s.free: illegal free. n=%d (page_size=%d) min=%d "
                    "runs_ok=%s values(head)=%s %s",
                    type(self).__name__,
                    n,
                    self.page_size,
                    min_val,
                    runs_ok,
                    indices[:128].tolist(),
                    self._debug_free_slots_state(),
                    stack_info=True,
                )
                raise AssertionError(
                    f"[SWA-DIAG] {type(self).__name__}.free: illegal free "
                    f"(n={n}, page_size={self.page_size}, min={min_val}, "
                    f"runs_ok={runs_ok}) would poison the free-slot ring"
                )
            # Capacity guard: freeing more slots than the ring can hold
            # indicates a double-free or an ownership/accounting bug.
            live_before = self._free_slots_count
            if live_before + n > len(self._free_slots_buf):
                logger.error(
                    "[SWA-DIAG] %s.free: capacity overflow (double-free suspected): "
                    "live(%d) + n(%d) > capacity(%d). %s",
                    type(self).__name__,
                    live_before,
                    n,
                    len(self._free_slots_buf),
                    self._debug_free_slots_state(),
                    stack_info=True,
                )
                raise AssertionError(
                    f"[SWA-DIAG] {type(self).__name__}.free: capacity overflow "
                    f"(live={live_before} + n={n} > capacity={len(self._free_slots_buf)}), "
                    f"double-free suspected"
                )
        capacity = len(self._free_slots_buf)
        if n > capacity - self._free_slots_count:
            raise RuntimeError(
                "HostKVCache free-list overflow: "
                f"freeing={n}, available={self._free_slots_count}, capacity={capacity}"
            )

        # Append freed indices to the tail (in-place, no alloc).
        end = self._free_slots_tail + n
        with self._ring_fence.order():
            if end <= capacity:
                self._free_slots_buf[self._free_slots_tail : end] = indices
            else:
                first = capacity - self._free_slots_tail
                second = n - first
                self._free_slots_buf[self._free_slots_tail :] = indices[:first]
                self._free_slots_buf[:second] = indices[first:]
        self._free_slots_tail = end % capacity
        self._free_slots_count += n
        return n


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = (
            _is_cuda
            and self.device_pool.head_dim == self.device_pool.v_head_dim
            and can_use_hicache_jit_kernel(
                element_size=self.element_dim * self.dtype.itemsize
            )
        )

        if self.layout == "page_first":
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            k_transposed = self.k_buffer.transpose(0, 1)
            v_transposed = self.v_buffer.transpose(0, 1)
            self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
            self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
            self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        self.v_head_dim = self.device_pool.v_head_dim
        if self.head_dim != self.v_head_dim:
            return (
                (self.head_dim + self.v_head_dim)
                * self.head_num
                * self.layer_num
                * self.dtype.itemsize
            )

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        if self.head_dim != self.v_head_dim:
            return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.head_dim == self.v_head_dim:
            k_dims = None
            v_dims = None
            if self.layout == "layer_first":
                dims = (2, self.layer_num, self.alloc_size, self.head_num, self.head_dim)
            elif self.layout == "page_first":
                dims = (2, self.alloc_size, self.layer_num, self.head_num, self.head_dim)
            elif self.layout == "page_first_direct":
                dims = (
                    2,
                    self.alloc_page_num,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            elif self.layout == "page_head":
                dims = (
                    2,
                    self.alloc_page_num,
                    self.head_num,
                    self.page_size,
                    self.layer_num,
                    self.head_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            dims = None
            if self.layout == "layer_first":
                k_dims = (
                    self.layer_num,
                    self.alloc_size,
                    self.head_num,
                    self.head_dim,
                )
                v_dims = (
                    self.layer_num,
                    self.alloc_size,
                    self.head_num,
                    self.v_head_dim,
                )
            elif self.layout == "page_first":
                k_dims = (
                    self.alloc_size,
                    self.layer_num,
                    self.head_num,
                    self.head_dim,
                )
                v_dims = (
                    self.alloc_size,
                    self.layer_num,
                    self.head_num,
                    self.v_head_dim,
                )
            elif self.layout == "page_first_direct":
                k_dims = (
                    self.alloc_page_num,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
                v_dims = (
                    self.alloc_page_num,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")

        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        self.v_token_stride_size = self.head_num * self.v_head_dim * self.dtype.itemsize
        self.v_layout_dim = self.v_token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            k_dims=k_dims,
            v_dims=v_dims,
        )
        return buffer

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                        block_quota=self.h2d_block_quota,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                        block_quota=self.h2d_block_quota,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Transpose [page, layer, ...] -> [layer, page, ...] then
                    # index by layer_id to get a per-layer view with strided layout.
                    # The kernel handles different src/dst strides automatically.
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_data_refs[layer_id],
                        v_cache_src=self.v_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                        block_quota=self.h2d_block_quota,
                    )
                else:
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                        block_quota=self.h2d_block_quota,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                    block_quota=self.h2d_block_quota,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_multi_pool_h2d_descriptors(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
    ):
        if io_backend != "kernel" or self.layout != "layer_first":
            return None
        return {
            "src_k_layer_ptrs": self.k_data_ptrs,
            "dst_k_layer_ptrs": device_pool.k_data_ptrs,
            "src_v_layer_ptrs": self.v_data_ptrs,
            "dst_v_layer_ptrs": device_pool.v_data_ptrs,
            "src_indices": [host_indices],
            "dst_indices": [device_indices],
            "pool_num_items": [host_indices.numel()],
            "pool_k_item_sizes": [self.token_stride_size],
            "pool_v_item_sizes": [self.v_token_stride_size],
            "max_num_items": host_indices.numel(),
            "block_quota": self.h2d_block_quota,
        }

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                        block_quota=self.d2h_block_quota,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                        block_quota=self.d2h_block_quota,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Use transposed data ptrs so the kernel writes to
                    # [layer, page, item] view with stride layout_dim per token.
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        kv_cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.element_dim * self.dtype.itemsize,
                        block_quota=self.d2h_block_quota,
                    )
                else:
                    transfer_kv_all_layer_lf_pf(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k=self.k_buffer,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v=self.v_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                        block_quota=self.d2h_block_quota,
                    )
            elif self.layout == "page_head":
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                    block_quota=self.d2h_block_quota,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    # TODO(yingchun): deal with different self.v_head_dim
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    # TODO(yingchun): deal with different self.v_head_dim
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        ).flatten()

    # TODO(yingchun): deal with different self.v_head_dim
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    # TODO(yingchun): deal with different self.v_head_dim
    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """
        get meta data for zero copy of heterogeneous ranks' KVCache
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.alloc_size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        # TODO(yingchun): 改为 indices[::self.page_size].tolist() 可减少 (page_size-1)/page_size 的对象分配
        indices = indices.tolist()
        if self.head_dim == self.v_head_dim:
            kv_buffer_data_ptr = self.kv_buffer.data_ptr()
            v_offset = (
                self.layer_num
                * self.alloc_size
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            if self.layout == "layer_first":
                for index in range(0, len(indices), self.page_size):
                    for layer_id in range(self.layer_num):
                        k_ptr = (
                            kv_buffer_data_ptr
                            + indices[index]
                            * self.head_num
                            * self.head_dim
                            * self.dtype.itemsize
                            + layer_id
                            * self.alloc_size
                            * self.head_num
                            * self.head_dim
                            * self.dtype.itemsize
                        )
                        v_ptr = k_ptr + v_offset
                        ptr_list.append(k_ptr)
                        ptr_list.append(v_ptr)
                element_size = (
                    self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
                )
                element_size_list = [element_size] * len(ptr_list)
            elif self.layout in ["page_first", "page_first_direct", "page_head"]:
                for index in range(0, len(indices), self.page_size):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.layer_num
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
                element_size = (
                    self.layer_num
                    * self.dtype.itemsize
                    * self.page_size
                    * self.head_num
                    * self.head_dim
                )
                element_size_list = [element_size] * len(ptr_list)
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            k_base_ptr = self.k_buffer.data_ptr()
            v_base_ptr = self.v_buffer.data_ptr()
            if self.layout in ["page_first", "page_first_direct"]:
                # TODO(yingchun): 可预计算变量降低重复计算
                k_element_size = (
                    self.layer_num
                    * self.dtype.itemsize
                    * self.page_size
                    * self.head_num
                    * self.head_dim
                )
                v_element_size = (
                    self.layer_num
                    * self.dtype.itemsize
                    * self.page_size
                    * self.head_num
                    * self.v_head_dim
                )
                element_size_list = []
                for index in range(0, len(indices), self.page_size):
                    k_ptr = (
                        k_base_ptr
                        + indices[index]
                        * self.layer_num
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = (
                        v_base_ptr
                        + indices[index]
                        * self.layer_num
                        * self.head_num
                        * self.v_head_dim
                        * self.dtype.itemsize
                    )
                    ptr_list.extend([k_ptr, v_ptr])
                    element_size_list.extend([k_element_size, v_element_size])
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.override_kv_cache_dim = override_kv_cache_dim
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize
        )

        if self.layout == "page_first" and self.can_use_jit:
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            transposed = self.kv_buffer.transpose(0, 1)
            self.data_refs = [transposed[i] for i in range(self.layer_num)]
        else:
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_contiguous_buf_infos(self):
        """Return (data_ptrs, data_lens, item_lens) in the same format as device pool,
        for registering host memory with the disaggregation transfer engine."""
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim or (
            self.kv_lora_rank + self.qk_rope_head_dim
        )
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.alloc_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.alloc_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.alloc_page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.alloc_page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            self.k_buffer = alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*base_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Return k_buffer to preserve original kv_buffer and data_refs init logic,
            # though Ascend doesn't use these parameters.
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.kv_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                        block_quota=self.h2d_block_quota,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[layer_id],
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                        block_quota=self.h2d_block_quota,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[layer_id],
                        cache_src=self.data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                        block_quota=self.h2d_block_quota,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                        block_quota=self.h2d_block_quota,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                        block_quota=self.d2h_block_quota,
                    )
                else:
                    transfer_kv_all_layer_mla(
                        src_layers=device_pool.data_ptrs,
                        dst_layers=self.data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                        block_quota=self.d2h_block_quota,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_pool.data_ptrs,
                        indices_src=device_indices,
                        cache_src_stride_bytes=self.token_stride_size,
                        cache_dst_stride_bytes=self.layout_dim,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                        block_quota=self.d2h_block_quota,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_pool.data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                        block_quota=self.d2h_block_quota,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    device_index_k=device_pool.index_k_buffer,
                    host_index_k=self.index_k_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id
                        * self.alloc_size
                        * self.kv_cache_dim
                        * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.kv_cache_dim
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.kv_cache_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MambaPoolHost(HostKVCache):

    def __init__(
        self,
        device_pool: MambaPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        layout: str = "layer_first",
    ):
        self.device_pool = device_pool
        self.page_size = 1
        assert layout in [
            "page_first",
            "page_first_direct",
            "layer_first",
        ], "Unsupported layout: {layout}"

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()

        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical Mamba cache (layout=%s).",
            requested_bytes / 1e9,
            self.layout,
        )

        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]

        if self.layout in ["page_first", "page_first_direct"]:
            # page-first: (page_num, num_layers, 1, *shape) — per-page data is contiguous
            temporal_dims = (
                self.size,
                self.num_mamba_layers,
                1,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.size, self.num_mamba_layers, 1) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )
        else:
            # layer-first: (num_layers, size, *shape)
            temporal_dims = (
                self.num_mamba_layers,
                self.size,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.num_mamba_layers, self.size) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )

    def _iter_page_tensors(self, index: int):
        if self.layout in ["page_first", "page_first_direct"]:
            yield self.temporal_buffer[index]
            for conv_buf in self.conv_buffer:
                yield conv_buf[index]
        else:
            yield self.temporal_buffer[:, index : index + self.page_size]
            for conv_buf in self.conv_buffer:
                yield conv_buf[:, index : index + self.page_size]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def get_size_per_token(self):
        conv_total_size = 0
        for conv_shape in self.conv_state_shapes:
            conv_total_size += int(np.prod(conv_shape)) * self.conv_dtype.itemsize
        temporal_size = (
            int(np.prod(self.temporal_state_shape)) * self.temporal_dtype.itemsize
        )
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: Rename the interface for clarity.
            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.
            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            transfer_kv_per_layer_mla_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        device: str,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            src_ptrs = torch.tensor(
                [src_layers[i].data_ptr() for i in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=src_ptrs,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=item_size,
                dst_layout_dim=item_size * num_layers,
                num_layers=num_layers,
            )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_pf_lf(
                src=self.temporal_buffer,
                dst=device_pool.mamba_cache.temporal[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                num_layers=self.num_mamba_layers,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            self._copy_tensor_all_layers_lf_pf(
                src_layers=device_pool.mamba_cache.temporal,
                dst=self.temporal_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=self.num_mamba_layers,
                device=self.device_pool.device,
                io_backend=io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.conv[conv_idx],
                    dst=self.conv_buffer[conv_idx],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    device=self.device_pool.device,
                    io_backend=io_backend,
                )
        else:
            for layer_id in range(self.num_mamba_layers):
                self._copy_tensor(
                    device_pool.mamba_cache.temporal[layer_id],
                    self.temporal_buffer[layer_id],
                    device_indices,
                    host_indices,
                    io_backend,
                )
                for conv_idx in range(len(self.conv_state_shapes)):
                    self._copy_tensor(
                        device_pool.mamba_cache.conv[conv_idx][layer_id],
                        self.conv_buffer[conv_idx][layer_id],
                        device_indices,
                        host_indices,
                        io_backend,
                    )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], Optional[int]]
    is_primary_index_anchor: bool = False
    # Optional eviction callbacks for auto-alloc in HybridCacheController.
    # host_evict_fn(n): evict n slots from the host pool (used by write()).
    # device_evict_fn(n): evict n slots from the device pool (used by load()).
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None


class HostPoolGroup:
    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        self.entries = entries
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = next(
            (entry for entry in entries if entry.is_primary_index_anchor),
            entries[0],
        )

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.anchor_entry.host_pool.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.anchor_entry.host_pool.free(indices)

    def get_data_page(self, index, flat: bool = True):
        return self.anchor_entry.host_pool.get_data_page(index, flat)

    def get_dummy_flat_data_page(self):
        return self.anchor_entry.host_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        return self.anchor_entry.host_pool.set_from_flat_data_page(index, data_page)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) transfer
        anchor = self.anchor_entry
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. Extra pool transfers
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) backup
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. Extra pool backup
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )


class NSATokenToKVPoolHost(MLATokenToKVPoolHost):
    device_pool: NSATokenToKVPool

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        # Initialize indexer metadata before HostKVCache.__init__ calls get_size_per_token.
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
        self.indexer_page_stride_size = (
            self.indexer_size_per_token * self.page_size * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self._init_indexer_buffers()
        logger.info(
            f"NSATokenToKVPoolHost initialized with indexer page stride size: {self.indexer_page_stride_size}, page num: {self.indexer_page_num}"
        )

    def get_size_per_token(self):
        base = super().get_size_per_token()
        return (
            base
            + self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def _init_indexer_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = [
                alloc_func(
                    (self.indexer_page_num, self.indexer_page_stride_size),
                    dtype=self.indexer_dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for NSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def _load_indexer_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_indexer_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.index_k_device_ptrs,
                    dst=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    dst_layout_dim=self.indexer_layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        self._load_indexer_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        self._backup_indexer_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )


# TODO(yingchun): 最好继承自 HostKVCache，但目前 HostKVCache 的接口设计不太适合 TokenToKVPoolHost 的双 pool 结构。
class SWATokenToKVPoolHost:
    device_pool: SWAKVPool
    # Only used on prefill; requires swa kv indices length == full kv indices length.

    def __init__(
        self,
        device_pool: SWAKVPool,
        host_to_device_ratio: float,
        host_size: int,
        host_size_swa: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        host_to_device_ratio_swa: float = None,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        if host_size or host_size_swa:
            assert (
                host_size == host_size_swa
            ), "Host size for full and swa should be the same"
        if host_to_device_ratio_swa is None:
            host_to_device_ratio_swa = host_to_device_ratio
        # full layers host pool
        with startup_timer("Initializing full pool host cache"):
            self.full_pool_host = MHATokenToKVPoolHost(
                device_pool=self.device_pool.full_kv_pool,
                host_to_device_ratio=host_to_device_ratio,
                host_size=host_size,
                page_size=page_size,
                layout=layout,
                pin_memory=pin_memory,
                device=self.device,
                allocator_type=allocator_type,
            )

        # swa layers host pool (different kv shape handled by swa_kv_pool)
        with startup_timer("Initializing SWA pool host cache"):
            self.swa_pool_host = MHATokenToKVPoolHost(
                device_pool=self.device_pool.swa_kv_pool,
                host_to_device_ratio=host_to_device_ratio_swa,
                host_size=host_size_swa,
                page_size=page_size,
                layout=layout,
                pin_memory=pin_memory,
                device=self.device,
                allocator_type=allocator_type,
            )
        self.h2d_block_quota = self.full_pool_host.h2d_block_quota
        self.d2h_block_quota = self.full_pool_host.d2h_block_quota
        self.full_pool_host._debug_pool_label = "full_pool_host"
        self.swa_pool_host._debug_pool_label = "swa_pool_host"

        # Validate host pool capacities: in disaggregated prefill mode the
        # host pools buffer KV data evicted from the device before transfer to
        # the decode node.  If the host pool is smaller than 1.5x the device
        # pool, eviction may stall because there is nowhere to offload.
        self._validate_host_pool_sizes(
            host_to_device_ratio, host_size, host_to_device_ratio_swa, host_size_swa
        )

        # TODO(yingchun): 从 SWATokenToKVPoolAllocator 拷过来的，还不确定最后的 -1 的意义
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        # Mapping must be sized by the ACTUAL host pool capacity (self.full_pool_host.size),
        # not the raw host_size argument. host_size may be interpreted as bytes/GB in
        # HostKVCache and converted to token slots, so using raw host_size can under-allocate.
        # TODO(yingchun): mapping 使用前，也做 assert，确保内容都为 0
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    self.full_pool_host.size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
        # Guards full_to_swa_index_mapping: it is read/written by the scheduler
        # thread (schedule_stream), the prefetch thread (default stream), and
        # write/load threads (their own streams); a Python lock alone does not
        # order the async CUDA ops across those streams.
        self._mapping_fence = _CudaStreamFence(device)
        self._mapping_lock = threading.RLock()
        self._mapping_section = _LockedFenceSection(
            self._mapping_lock, self._mapping_fence.order()
        )
        # {global_layer_id: (pool_layer_id, is_swa_layer)}
        self.layers_mapping = self.device_pool.layers_mapping
        self.layer_num = self.device_pool.layer_num
        self.reserve_swa_size = envs.SGLANG_RESERVE_SWA_SIZE_BEFORE_PREFETCHING.get()
        assert self.reserve_swa_size % self.page_size == 0
        # Cached iota, mirroring PagedTokenToKVPoolAllocator._page_offsets:
        # avoids re-materializing an int64 [0..page_size) temporary per
        # mapping call.
        self._page_offsets = torch.arange(self.page_size, device=device)
        self._token_stride_sizes, self._page_element_sizes = self._compute_page_meta()

    def _validate_host_pool_sizes(
        self,
        host_to_device_ratio: float,
        host_size: int,
        host_to_device_ratio_swa: float,
        host_size_swa: int,
    ) -> None:
        """Ensure host pools are large enough relative to device pools.

        In disaggregated prefill mode, KV data is offloaded from the device to
        the host before being transferred to the decode node. The host pool
        must be at least 1.5x the device pool to avoid back-pressure when the
        device pool is fully utilized.
        """
        _HOST_MIN_RATIO = 1.5

        device_full_size = self.device_pool.full_kv_pool.size
        host_full_size = self.full_pool_host.size
        min_host_full_size = int(device_full_size * _HOST_MIN_RATIO)
        if host_full_size < min_host_full_size:
            raise RuntimeError(
                f"Host Full pool size ({host_full_size}) must be at least "
                f"{_HOST_MIN_RATIO}x the device Full pool size ({device_full_size}), "
                f"i.e. >= {min_host_full_size}. "
                f"Please increase --hicache-ratio (current: {host_to_device_ratio}) "
                f"or set a larger --hicache-size (current: {host_size} GB) "
                f"to enlarge the host Full pool capacity."
            )

        device_swa_size = self.device_pool.swa_kv_pool.size
        host_swa_size = self.swa_pool_host.size
        min_host_swa_size = int(device_swa_size * _HOST_MIN_RATIO)
        if host_swa_size < min_host_swa_size:
            raise RuntimeError(
                f"Host SWA pool size ({host_swa_size}) must be at least "
                f"{_HOST_MIN_RATIO}x the device SWA pool size ({device_swa_size}), "
                f"i.e. >= {min_host_swa_size}. "
                f"Please increase --hicache-ratio (current: {host_to_device_ratio_swa}) "
                f"or set a larger --hicache-size (current: {host_size_swa} GB) "
                f"to enlarge the host SWA pool capacity."
            )

    # full 和 swa 两个子池容量恒等（构造时 host_size==host_size_swa），全程也按相同
    # need_size 一起 alloc/free（见 alloc/free），所以 size / available_size 直接代理
    # 到 full_pool_host 即可，与外层 HostKVCache 类签名保持一致。
    @property
    def size(self) -> int:
        return self.full_pool_host.size

    @property
    def dtype(self) -> torch.dtype:
        return self.full_pool_host.dtype

    def available_size(self) -> int:
        return self.full_pool_host.available_size()

    def swa_available_size(self) -> tuple[int, int]:
        return self.full_pool_host.available_size(), self.swa_pool_host.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        # Legacy alloc API is no longer valid for SWA host pool: full and SWA
        # sub-pools are now coupled through full_to_swa_index_mapping and must
        # be allocated together via alloc_host(). Direct alloc() callers
        # would skip mapping setup and break SWA load/backup paths.
        assert False, (
            "SWATokenToKVPoolHost.alloc() is not supported; "
            "use alloc_host(full_device_indices, swa_device_indices) instead."
        )
        # full_select_index = self.full_pool_host.alloc(need_size)
        # swa_select_index = self.swa_pool_host.alloc(need_size)
        # assert (full_select_index is None) == (swa_select_index is None)
        # if full_select_index is not None:
        #     assert len(full_select_index) == need_size
        #     assert len(swa_select_index) == need_size
        # return full_select_index

    # 注意这里的 full 和 swa 未做 mapping。释放时需要注意：
    # - 对于后续已做 mapping 的部分，使用 free 即可释放 full + swa
    # - 对于后续未做 mapping 的部分，使用 free 可释放 full，使用 free_swa_only 可释放 swa
    def alloc_unmapped_host_by_len(
        self, full_need_size: int, swa_need_size: int = None
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        int,
        int,
        Optional[torch.cuda.Event],
    ]:
        assert full_need_size % self.page_size == 0
        if swa_need_size is None:
            swa_need_size = min(full_need_size, self.reserve_swa_size)
        assert swa_need_size % self.page_size == 0

        lack_full_size = full_need_size - self.full_pool_host.available_size()
        lack_swa_size = swa_need_size - self.swa_pool_host.available_size()
        if lack_full_size > 0 or lack_swa_size > 0:
            return None, None, max(0, lack_full_size), max(0, lack_swa_size), None

        full_host_indices = self.full_pool_host.alloc(full_need_size)
        # [SWA-DIAG] alloc() returns None only when need_size > available_size().
        # We just checked availability above, so a None here means another
        # allocator thread consumed the full ring between the check and this
        # alloc (check-then-act race). Fail hard: proceeding would return a
        # torn (full-None) result and write a bad mapping below.
        assert full_host_indices is not None, (
            f"race: full ring drained between availability check and alloc "
            f"(need={full_need_size}, avail_now={self.full_pool_host.available_size()})"
        )
        # 注意: 此时并不做 mapping，而会在 cc._storage_hit_query 之后确定要真正要 batch_get 的 SWA 区间之后再做 mapping
        with self._mapping_section:
            self.full_to_swa_index_mapping[full_host_indices] = 0
        swa_host_indices = self.swa_pool_host.alloc(swa_need_size)
        # [SWA-DIAG] Same race on the swa ring: full already succeeded and the
        # mapping was written, so a None swa index is a torn allocation the
        # caller cannot handle (it unconditionally does len(swa)). Fail hard
        # at the source instead of surfacing as a downstream TypeError.
        assert swa_host_indices is not None, (
            f"race: swa ring drained between availability check and alloc "
            f"(need={swa_need_size}, avail_now={self.swa_pool_host.available_size()})"
        )
        # [SWA-DIAG] Value-domain check at the alloc exit (production
        # incident 2026-08-03: an arange(0,64) page surfaced seconds later at
        # the release assert). Checking here pins the corruption moment to
        # "at/before alloc" vs "while parked during the storage query".
        assert_indices_in_bounds(
            full_host_indices,
            self.page_size,
            self.full_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.alloc_unmapped_host_by_len(full)",
            hot=True,
        )
        assert_indices_in_bounds(
            swa_host_indices,
            self.page_size,
            self.swa_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.alloc_unmapped_host_by_len(swa)",
            hot=True,
        )

        # Both full_host_indices and swa_host_indices are produced by an async
        # .clone() on this (the scheduler) thread's current CUDA stream, from
        # sub-pools that share self.device. They then cross to the prefetch
        # thread via a plain Python queue, which carries no CUDA dependency, so
        # the consumer must wait on this event before reading either on its own
        # stream. The event is recorded after both clones, so it covers both;
        # both indices share a device, so gating on full is sufficient.
        host_indices_ready_event = None
        if full_host_indices.is_cuda:
            host_indices_ready_event = torch.cuda.Event()
            host_indices_ready_event.record()

        return full_host_indices, swa_host_indices, 0, 0, host_indices_ready_event

    def alloc_host(
        self,
        full_device_indices: torch.Tensor,
        swa_device_indices: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], int, int]:
        full_need_size = len(full_device_indices)
        assert full_need_size % self.page_size == 0
        assert len(full_device_indices) == len(swa_device_indices)

        swa_valid_mask = swa_device_indices != 0
        swa_need_size = int(swa_valid_mask.sum())
        assert swa_need_size % self.page_size == 0
        # Device-side input snapshot: a sub-page pattern here may be a concurrent
        # torn read of swa_device_indices, not a deterministic bug, so log only
        # (the test crash-1 reproduction relies on this mask penetrating to the
        # later release site where it is caught).
        log_subpage_swa_pattern(
            swa_valid_mask,
            self.page_size,
            "SWATokenToKVPoolHost.alloc_host(device swa_valid_mask)",
        )

        lack_full_size = full_need_size - self.full_pool_host.available_size()
        lack_swa_size = swa_need_size - self.swa_pool_host.available_size()
        if lack_full_size > 0 or lack_swa_size > 0:
            return None, max(0, lack_full_size), max(0, lack_swa_size)

        full_host_indices = self.full_pool_host.alloc(full_need_size)
        # [SWA-DIAG] None after a passing availability check == another
        # allocator drained the full ring in the check-then-act gap.
        assert full_host_indices is not None, (
            f"race: full ring drained between availability check and alloc "
            f"(need={full_need_size}, avail_now={self.full_pool_host.available_size()})"
        )
        swa_mapping = torch.zeros(full_need_size, dtype=torch.int64, device=self.device)
        if swa_need_size > 0:
            swa_host_indices = self.swa_pool_host.alloc(swa_need_size)
            # [SWA-DIAG] Same race on the swa ring; full is already allocated
            # so this is a torn allocation, not a clean capacity shortfall.
            assert swa_host_indices is not None, (
                f"race: swa ring drained between availability check and alloc "
                f"(need={swa_need_size}, avail_now={self.swa_pool_host.available_size()})"
            )
            assert_indices_in_bounds(
                swa_host_indices,
                self.page_size,
                self.swa_pool_host.size + self.page_size,
                "SWATokenToKVPoolHost.alloc_host(swa)",
                hot=True,
            )
            # swa_mapping is on the host pool device (CPU); swa_valid_mask comes
            # from swa_device_indices (CUDA). Align the mask before indexing.
            swa_mapping[swa_valid_mask.to(swa_mapping.device)] = swa_host_indices
        assert_indices_in_bounds(
            full_host_indices,
            self.page_size,
            self.full_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.alloc_host(full)",
            hot=True,
        )
        with self._mapping_section:
            self.full_to_swa_index_mapping[full_host_indices] = swa_mapping

        return full_host_indices, 0, 0

    def free(self, full_host_indices: torch.Tensor) -> tuple[int, int]:
        if full_host_indices.numel() == 0:
            return 0, 0

        # Detach the SWA mapping BEFORE returning the full indices to the free
        # ring: once freed, another allocator call may re-alloc the same full
        # indices and zero their mapping entries, so gathering afterwards
        # would read 0 and leak the SWA slots.
        with self._mapping_section:
            swa_host_indices = self.full_to_swa_index_mapping[full_host_indices]
            self.full_to_swa_index_mapping[full_host_indices] = 0
        self.full_pool_host.free(full_host_indices)
        swa_valid = swa_host_indices.bool()
        # Earliest possible detection of mixed mapping pages: without this,
        # the .bool() filter below silently frees the valid half of a mixed
        # page and the corruption only surfaces much later at the release
        # asserts in cache_controller. HOT-gated: runs on every free and its
        # timing shift can mask the very race being investigated.
        if _SWA_DIAG_HOT_ENABLED:
            log_subpage_swa_pattern(
                swa_valid,
                self.page_size,
                "SWATokenToKVPoolHost.free(mapping gather)",
                extra=lambda: f"full_head={full_host_indices[:8].tolist()}",
            )
        swa_freed = int(swa_valid.sum())
        if swa_freed > 0:
            self.swa_pool_host.free(swa_host_indices[swa_valid])

        return int(full_host_indices.numel()), swa_freed

    def mapping_full_and_swa_indices(
        self,
        full_host_indices: torch.Tensor,
        swa_host_indices: torch.Tensor,
        swa_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Vectorized version of mapping_full_and_swa_indices.

        Same semantics: bind pre-allocated SWA pages to True pages in
        swa_valid_mask from tail.  When SWA pages are insufficient the
        frontmost True entries are trimmed; when surplus, leftover SWA
        indices are returned from the head of swa_host_indices.
        """
        ps = self.page_size
        num_pages = len(swa_valid_mask)
        assert len(full_host_indices) == num_pages * ps

        swa_available_tokens = len(swa_host_indices)
        assert swa_available_tokens % ps == 0
        swa_available_pages = swa_available_tokens // ps

        nonzero_idx = swa_valid_mask.nonzero(as_tuple=False).view(-1)
        num_needed_pages = len(nonzero_idx)
        assert num_needed_pages > 0
        assert swa_available_pages > 0

        updated_mask = swa_valid_mask.clone()
        surplus_swa = None

        if swa_available_pages >= num_needed_pages:
            pages_to_bind = nonzero_idx
            surplus_swa_tokens = (swa_available_pages - num_needed_pages) * ps
            swa_to_bind = swa_host_indices[surplus_swa_tokens:]
            if surplus_swa_tokens > 0:
                surplus_swa = swa_host_indices[:surplus_swa_tokens]
        else:
            trim_count = num_needed_pages - swa_available_pages
            trimmed_indices = nonzero_idx[:trim_count]
            updated_mask[trimmed_indices] = False
            pages_to_bind = nonzero_idx[trim_count:]
            swa_to_bind = swa_host_indices

        # pages_to_bind: [P], page offsets: [ps] → broadcast [P, ps] → flatten
        # [P*ps]. The .to() is a no-op on the expected pool-device path and
        # only materializes a copy if a caller ever passes a CPU mask.
        token_indices = (
            pages_to_bind.unsqueeze(1) * ps
            + self._page_offsets.to(pages_to_bind.device)
        ).flatten()

        full_at_tokens = full_host_indices[token_indices]
        with self._mapping_section:
            self.full_to_swa_index_mapping[full_at_tokens] = swa_to_bind[
                : len(full_at_tokens)
            ]

        return updated_mask, surplus_swa

    def unmapping_full_and_swa_indices(
        self,
        full_host_indices: torch.Tensor,
    ) -> torch.Tensor:
        # 由于 full 和 SWA 在一开始是 unmapped 的，且 SWA 也只分配了 self.reserve_swa_size 长度，
        # 所以 full 对应的 SWA 不一定有效，不能做 assert
        with self._mapping_section:
            swa_host_indices = self.full_to_swa_index_mapping[full_host_indices]
            # 解除 mapping
            self.full_to_swa_index_mapping[full_host_indices] = 0

        # 返回悬垂的 SWA，供 caller 释放
        return swa_host_indices

    # TODO(yingchun): 与 SWATokenToKVPoolAllocator 中的 splice_swa_mapping 实现基本一样，可重构
    def splice_swa_mapping(
        self,
        tgt_full_indices: torch.Tensor,
        src_full_indices: torch.Tensor,
    ) -> int:
        """
        Transfer SWA host slots from src to tgt where tgt lacks SWA but src has it.

        Analogous to SWATokenToKVPoolAllocator.splice_swa_mapping on the device
        side, but operates on the host-side full_to_swa_index_mapping.

        For each position i where tgt has no SWA (mapping == 0) and src does
        (mapping != 0), the SWA slot ownership is moved: tgt gets the SWA index,
        src's mapping is zeroed out.  The SWA KV data already resides in the host
        pool buffer, so no data copy is needed — only the mapping pointer changes.

        tgt_full_indices may contain -1 placeholders for non-backuped tree nodes
        (write_back mode); those positions are skipped automatically.

        Returns the number of tokens whose SWA mapping was transferred.
        """
        assert len(tgt_full_indices) == len(src_full_indices)
        # full_to_swa_index_mapping is on the host pool device (CPU); tree-side
        # indices may arrive on CUDA. Align both once so all gather/scatter
        # below stays on the mapping's device (.to() is a no-op if already CPU).
        mapping_device = self.full_to_swa_index_mapping.device
        tgt_full_indices = tgt_full_indices.to(mapping_device)
        src_full_indices = src_full_indices.to(mapping_device)
        # tgt_full_indices may contain -1 placeholders for non-backuped tree
        # nodes (write_back mode). Skip those positions entirely.
        valid_mask = tgt_full_indices >= 0
        if not valid_mask.all():
            tgt_full_indices = tgt_full_indices[valid_mask]
            src_full_indices = src_full_indices[valid_mask]
        if len(tgt_full_indices) == 0:
            return 0
        # Keep the critical section free of D2H syncs: count_nonzero().item()
        # would block every other mapping user while holding the lock. The
        # masked scatter is a no-op when the mask is all-false, so it can run
        # unconditionally; the count and diagnostics read snapshots afterwards.
        post_tgt_swa = post_src_swa = None
        with self._mapping_section:
            tgt_swa = self.full_to_swa_index_mapping[tgt_full_indices]
            src_swa = self.full_to_swa_index_mapping[src_full_indices]
            transfer_mask: torch.Tensor = (tgt_swa == 0) & (src_swa != 0)
            self.full_to_swa_index_mapping[tgt_full_indices[transfer_mask]] = src_swa[
                transfer_mask
            ]
            self.full_to_swa_index_mapping[src_full_indices[transfer_mask]] = 0
            if _SWA_DIAG_ENABLED:
                post_tgt_swa = self.full_to_swa_index_mapping[tgt_full_indices]
                post_src_swa = self.full_to_swa_index_mapping[src_full_indices]

        num_transferred = int(transfer_mask.count_nonzero())
        if num_transferred > 0:
            assert num_transferred % self.page_size == 0
            # A sub-page transfer mask means tgt/src mapping pages are already
            # mixed, or their page phases disagree; the scatter above would
            # then propagate mixed pages further.
            log_subpage_swa_pattern(
                transfer_mask,
                self.page_size,
                "SWATokenToKVPoolHost.splice_swa_mapping(transfer_mask)",
            )
            if _SWA_DIAG_ENABLED:
                # [SWA-DIAG] 写后校验：即使 transfer_mask 本身页对齐，若 tgt/src 的
                # 页相位不一致，scatter 后 tgt 或 src 的 mapping 也可能变成混合页。
                # 在源头抓，而不是等到 free 时才炸。
                assert_page_uniform_swa_pattern(
                    post_tgt_swa.bool(),
                    self.page_size,
                    "SWATokenToKVPoolHost.splice_swa_mapping(post_tgt_mapping)",
                )
                assert_page_uniform_swa_pattern(
                    post_src_swa.bool(),
                    self.page_size,
                    "SWATokenToKVPoolHost.splice_swa_mapping(post_src_mapping)",
                )
        return num_transferred

    def free_swa_only(self, swa_indices: torch.Tensor) -> int:
        """Free raw SWA indices that were never mapped to any full index."""
        if swa_indices.numel() == 0:
            return 0

        self.swa_pool_host.free(swa_indices)
        return int(swa_indices.numel())

    def gather_swa_mapping(self, full_host_indices: torch.Tensor) -> torch.Tensor:
        # full_to_swa_index_mapping lives on the host pool device (CPU). Index
        # it on that device (the .to() is a no-op when already there), then
        # hand the result back on the caller's device so downstream ops with
        # device-side indices/masks stay consistent.
        mapping_device = self.full_to_swa_index_mapping.device
        with self._mapping_section:
            swa = self.full_to_swa_index_mapping[full_host_indices.to(mapping_device)]
        return swa.to(full_host_indices.device)

    def scatter_swa_mapping(
        self, full_host_indices: torch.Tensor, swa_values: torch.Tensor
    ) -> None:
        with self._mapping_section:
            self.full_to_swa_index_mapping[full_host_indices] = swa_values

    def clear(self):
        self.full_pool_host.clear()
        self.swa_pool_host.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        with self._mapping_section:
            self.full_to_swa_index_mapping[:-1].fill_(0)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        swa_host_indices: torch.Tensor,
        swa_device_indices: torch.Tensor,
    ):
        layer_id_pool, is_swa = self.device_pool.layers_mapping[layer_id]
        assert len(host_indices) == len(device_indices)
        if is_swa:
            assert swa_host_indices is not None
            assert swa_device_indices is not None
            if swa_host_indices.numel() == 0:
                return
            self.swa_pool_host.load_to_device_per_layer(
                self.device_pool.swa_kv_pool,
                swa_host_indices,
                swa_device_indices,
                layer_id_pool,
                io_backend,
            )
        else:
            # The full indices may be empty if load SWA only, skip launching kernel in this case.
            if host_indices.numel() == 0:
                return
            self.full_pool_host.load_to_device_per_layer(
                self.device_pool.full_kv_pool,
                host_indices,
                device_indices,
                layer_id_pool,
                io_backend,
            )

    def get_multi_pool_h2d_descriptors(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        swa_host_indices: torch.Tensor,
        swa_device_indices: torch.Tensor,
    ):
        if io_backend != "kernel":
            return None
        full_desc = self.full_pool_host.get_multi_pool_h2d_descriptors(
            self.device_pool.full_kv_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        if full_desc is None:
            return None
        if swa_host_indices is None or swa_device_indices is None:
            return full_desc
        swa_desc = self.swa_pool_host.get_multi_pool_h2d_descriptors(
            self.device_pool.swa_kv_pool,
            swa_host_indices,
            swa_device_indices,
            io_backend,
        )
        if swa_desc is None:
            return None

        src_k_layer_ptrs = []
        dst_k_layer_ptrs = []
        src_v_layer_ptrs = []
        dst_v_layer_ptrs = []
        layer_to_pool = []
        layer_to_pool_layer = []
        for layer_id in range(self.layer_num):
            pool_layer_id, is_swa = self.device_pool.layers_mapping[layer_id]
            desc = swa_desc if is_swa else full_desc
            src_k_layer_ptrs.append(desc["src_k_layer_ptrs"][pool_layer_id])
            dst_k_layer_ptrs.append(desc["dst_k_layer_ptrs"][pool_layer_id])
            src_v_layer_ptrs.append(desc["src_v_layer_ptrs"][pool_layer_id])
            dst_v_layer_ptrs.append(desc["dst_v_layer_ptrs"][pool_layer_id])
            layer_to_pool.append(1 if is_swa else 0)
            layer_to_pool_layer.append(pool_layer_id)

        return {
            "src_k_layer_ptrs": torch.stack(src_k_layer_ptrs),
            "dst_k_layer_ptrs": torch.stack(dst_k_layer_ptrs),
            "src_v_layer_ptrs": torch.stack(src_v_layer_ptrs),
            "dst_v_layer_ptrs": torch.stack(dst_v_layer_ptrs),
            "src_indices": full_desc["src_indices"] + swa_desc["src_indices"],
            "dst_indices": full_desc["dst_indices"] + swa_desc["dst_indices"],
            "layer_to_pool": layer_to_pool,
            "layer_to_pool_layer": layer_to_pool_layer,
            "pool_num_items": full_desc["pool_num_items"] + swa_desc["pool_num_items"],
            "pool_k_item_sizes": full_desc["pool_k_item_sizes"]
            + swa_desc["pool_k_item_sizes"],
            "pool_v_item_sizes": full_desc["pool_v_item_sizes"]
            + swa_desc["pool_v_item_sizes"],
            "max_num_items": max(full_desc["max_num_items"], swa_desc["max_num_items"]),
            "block_quota": self.h2d_block_quota,
        }

    @nvtx_annotated_method("backup_from_device_all_layer")
    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        swa_host_indices,
        swa_device_indices,
    ):
        assert io_backend == "kernel", "Only kernel IO backend is supported."
        # D2H kernel writes to host at dst_indices * stride with NO bounds
        # check inside the kernel; a corrupted index turns the KV backup into
        # a host memory stomp. Validate before launch.
        assert_indices_in_bounds(
            host_indices,
            self.page_size,
            self.full_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.backup_from_device_all_layer(host_indices)",
        )
        assert_indices_in_bounds(
            swa_host_indices,
            self.page_size,
            self.swa_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.backup_from_device_all_layer(swa_host_indices)",
        )
        # Backup full-attention KV cache.
        transfer_kv_all_layer_fuse_lf_pf(
            src_k_layers=device_pool.full_kv_pool.k_data_ptrs,
            dst_k=self.full_pool_host.k_buffer,
            src_v_layers=device_pool.full_kv_pool.v_data_ptrs,
            dst_v=self.full_pool_host.v_buffer,
            src_indices=device_indices,
            dst_indices=host_indices,
            item_size=self.full_pool_host.token_stride_size,
            dst_layout_dim=self.full_pool_host.layout_dim,
            v_item_size=self.full_pool_host.v_token_stride_size,
            v_dst_layout_dim=self.full_pool_host.v_layout_dim,
            num_layers=self.device_pool.full_layer_nums,
            block_quota=self.d2h_block_quota,
        )
        # Backup SWA KV cache.
        if swa_host_indices.numel() == 0:
            return
        transfer_kv_all_layer_fuse_lf_pf(
            src_k_layers=device_pool.swa_kv_pool.k_data_ptrs,
            dst_k=self.swa_pool_host.k_buffer,
            src_v_layers=device_pool.swa_kv_pool.v_data_ptrs,
            dst_v=self.swa_pool_host.v_buffer,
            src_indices=swa_device_indices,
            dst_indices=swa_host_indices,
            item_size=self.swa_pool_host.token_stride_size,
            dst_layout_dim=self.swa_pool_host.layout_dim,
            v_item_size=self.swa_pool_host.v_token_stride_size,
            v_dst_layout_dim=self.swa_pool_host.v_layout_dim,
            num_layers=self.device_pool.swa_layer_nums,
            block_quota=self.d2h_block_quota,
        )

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError(
            "SWATokenToKVPoolHost does not support get_data_page."
        )

    def get_dummy_flat_data_page(self):
        raise NotImplementedError(
            "SWATokenToKVPoolHost does not support get_dummy_flat_data_page."
        )

    def set_from_flat_data_page(self, index: int, data_page):
        raise NotImplementedError(
            "SWATokenToKVPoolHost does not support set_from_flat_data_page."
        )

    def get_size_per_token(self):
        return (
            self.full_pool_host.get_size_per_token()
            + self.swa_pool_host.get_size_per_token()
        )

    def get_ksize_per_token(self):
        return (
            self.full_pool_host.get_ksize_per_token()
            + self.swa_pool_host.get_ksize_per_token()
        )

    def _compute_page_meta(self):
        itemsize = self.dtype.itemsize
        # TODO(yingchun): 与已有的 full/swa_pool_host.layout_dim/v_layout_dim 完全相同
        stride_sizes = (
            self.device_pool.full_layer_nums
            * self.full_pool_host.head_num
            * self.full_pool_host.head_dim
            * itemsize,
            self.device_pool.full_layer_nums
            * self.full_pool_host.head_num
            * self.full_pool_host.v_head_dim
            * itemsize,
            self.device_pool.swa_layer_nums
            * self.swa_pool_host.head_num
            * self.swa_pool_host.head_dim
            * itemsize,
            self.device_pool.swa_layer_nums
            * self.swa_pool_host.head_num
            * self.swa_pool_host.v_head_dim
            * itemsize,
        )
        element_sizes = tuple(s * self.page_size for s in stride_sizes)
        return stride_sizes, element_sizes

    def get_page_element_sizes(self):
        return self._page_element_sizes

    def get_page_buffer_meta(self, full_host_indices, swa_needed_per_page=None):
        """
        meta data for zero copy
        """
        assert len(full_host_indices) % self.page_size == 0
        assert_indices_in_bounds(
            full_host_indices,
            self.page_size,
            self.full_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.get_page_buffer_meta(full_host_indices)",
        )
        full_k_base_ptr = self.full_pool_host.k_buffer.data_ptr()
        full_v_base_ptr = self.full_pool_host.v_buffer.data_ptr()
        swa_k_base_ptr = self.swa_pool_host.k_buffer.data_ptr()
        swa_v_base_ptr = self.swa_pool_host.v_buffer.data_ptr()

        full_k_stride, full_v_stride, swa_k_stride, swa_v_stride = (
            self._token_stride_sizes
        )
        (
            full_k_element_size,
            full_v_element_size,
            swa_k_element_size,
            swa_v_element_size,
        ) = self._page_element_sizes

        # Vectorized: take the first token index of every page in one shot.
        page_full_idx = full_host_indices[:: self.page_size]
        page_swa_idx = self.gather_swa_mapping(page_full_idx)
        assert_indices_in_bounds(
            page_swa_idx,
            self.page_size,
            self.swa_pool_host.size + self.page_size,
            "SWATokenToKVPoolHost.get_page_buffer_meta(page_swa_idx)",
            allow_zero=True,
        )

        # Vectorized per-page pointer computation (one tensor op instead of a
        # Python scalar loop over pages).
        full_k_ptrs = (full_k_base_ptr + page_full_idx * full_k_stride).tolist()
        full_v_ptrs = (full_v_base_ptr + page_full_idx * full_v_stride).tolist()
        swa_k_ptrs = (swa_k_base_ptr + page_swa_idx * swa_k_stride).tolist()
        swa_v_ptrs = (swa_v_base_ptr + page_swa_idx * swa_v_stride).tolist()

        swa_needed_mask = page_swa_idx != 0
        if swa_needed_per_page is not None:
            swa_needed_mask &= torch.as_tensor(
                swa_needed_per_page, dtype=torch.bool, device=swa_needed_mask.device
            )
        swa_valid_list = swa_needed_mask.tolist()

        ptr_list = []
        element_size_list = []
        for page_idx, swa_needed in enumerate(swa_valid_list):
            if swa_needed:
                ptr_list.extend(
                    [
                        full_k_ptrs[page_idx],
                        full_v_ptrs[page_idx],
                        swa_k_ptrs[page_idx],
                        swa_v_ptrs[page_idx],
                    ]
                )
                element_size_list.extend(
                    [
                        full_k_element_size,
                        full_v_element_size,
                        swa_k_element_size,
                        swa_v_element_size,
                    ]
                )
            else:
                ptr_list.extend([full_k_ptrs[page_idx], full_v_ptrs[page_idx]])
                element_size_list.extend(
                    [
                        full_k_element_size,
                        full_v_element_size,
                    ]
                )
        return ptr_list, element_size_list, swa_valid_list
