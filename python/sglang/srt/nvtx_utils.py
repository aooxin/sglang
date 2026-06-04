import logging
from contextlib import contextmanager, nullcontext
from functools import wraps

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NVTX_ENV_ENABLED = envs.SGLANG_SCHEDULER_ENABLE_NVTX.get()
_nvtx_module = None
if _NVTX_ENV_ENABLED:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "SGLANG_SCHEDULER_ENABLE_NVTX is set, but the `nvtx` package is missing. "
            "Scheduler NVTX annotations are disabled."
        )

NVTX_ENABLED = _nvtx_module is not None

_NVTX_COLOR_MAP = {
    # === Scheduler main loop (pipeline order) ===
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.update_running_batch": "green",
    "scheduler.prepare_mlp_sync_batch": "orange",
    "scheduler.run_batch": "red",
    "scheduler.launch_batch_sample_if_needed": "magenta",
    "scheduler.process_batch_result": "cyan",
    # === Disagg decode loop (mirrors main-loop colors) ===
    "scheduler.disagg_decode.event_loop_normal": "blue",
    "scheduler.disagg_decode.event_loop_overlap": "blue",
    "scheduler.disagg_decode.process_decode_queue": "purple",
    "scheduler.disagg_decode.get_next_batch": "green",
    "scheduler.disagg_decode.get_new_prebuilt_batch": "green",
    # === Disagg prefill ===
    "scheduler.process_batch_result_disagg_prefill": "cyan",
    "scheduler.process_disagg_prefill_inflight_queue": "cyan",
    # === ScheduleBatch helpers (state mutations) ===
    "scheduler.schedule_batch.prepare_for_decode": "pink",
    "scheduler.schedule_batch.maybe_wait_verify_done": "pink",
    "scheduler.schedule_batch.filter_batch": "pink",
    "schedulebatch.prepare_for_prebuilt": "pink",
    "decode_schedule_batch.process_prebuilt": "pink",
    # === GPU forward path ===
    "target_worker.forward_batch_generation": "brown",
    "tp_worker.batch_init_new": "pink",
    "model_runner.forward_decode": "red",
    "cuda_graph_runner.replay_prepare": "green",
    "cuda_graph_runner.replay": "red",
    # === Sampler ===
    "sampler.apply_custom_logit_processor": "purple",
    # === EAGLE V2 - draft (yellow) ===
    "eagle_worker_v2.forward_batch_generation": "brown",
    "eagle_draft_worker.draft": "yellow",
    "eagle_draft_worker.draft_forward": "yellow",
    "eagle_draft_worker._draft_extend_for_decode": "yellow",
    "eagle_draft_input_v2_mixin.prepare_for_decode": "pink",
    "eagle_draft_input_v2_mixin.prepare_for_v2_draft": "pink",
    "eagle_draft_input_v2_mixin.prepare_for_extend_to_fill_draft_kvcache": "pink",
    # === EAGLE V2 - verify (gray) ===
    "eagle_verify_input_v2_mixin.prepare_for_v2_verify": "pink",
    "eagle_verify_input_v2_mixin.sample": "magenta",
    "eagle_worker_v2.verify": "gray",
    # === Multi-layer EAGLE V2 (same color scheme) ===
    "multi_layer_eagle_draft_worker.draft": "yellow",
    "multi_layer_eagle_draft_worker.draft_forward": "yellow",
    "multi_layer_eagle_draft_worker._draft_extend_for_prefill": "yellow",
    "multi_layer_eagle_draft_worker._draft_extend_for_decode": "yellow",
    "multi_layer_eagle_worker_v2.verify": "gray",

    # Forward gap
    "scheduler.process_batch_result_disagg_prefill": "yellow",
    "scheduler.disagg_prefill.process_inflight_queue": "blue",
    "swa_radix.cache_unfinished_req": "blue",
    "swa_radix.cache_finished_req": "purple",
}


@contextmanager
def _nvtx_range_enabled(debug_name: str):
    color = _NVTX_COLOR_MAP.get(debug_name)
    # record_function carries a non-trivial (~microseconds) cost per call even
    # when no PyTorch profiler is collecting, so only pay it when one is active
    # (e.g. Chrome-trace export). For Nsight-only runs the nvtx.annotate marker
    # alone is enough.
    if torch.autograd._profiler_enabled():
        with torch.autograd.profiler.record_function(debug_name):
            with _nvtx_module.annotate(debug_name, color=color):
                yield
    else:
        with _nvtx_module.annotate(debug_name, color=color):
            yield


if NVTX_ENABLED:
    nvtx_range = _nvtx_range_enabled
else:
    # When NVTX is disabled, hand back a shared no-op context manager so hot
    # paths using `with nvtx_range(...)` pay no per-call generator overhead.
    _NULL_CONTEXT = nullcontext()

    def nvtx_range(debug_name: str):
        return _NULL_CONTEXT


def nvtx_annotated_method(debug_name: str):
    # Decide at decoration time. When NVTX is disabled this returns the
    # original function untouched, so decorated methods on hot paths
    # (e.g. forward_decode, cuda_graph_runner.replay) have zero runtime cost.
    if not NVTX_ENABLED:
        return lambda func: func

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(debug_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
