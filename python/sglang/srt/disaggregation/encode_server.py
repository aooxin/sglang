import asyncio
import concurrent.futures
import contextlib
import contextvars
import copy
import ctypes
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import re
import time
import traceback
from collections import defaultdict
from http import HTTPStatus
from multiprocessing import Queue
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, Response
from transformers import AutoProcessor

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_receiver import (
    EmbeddingData,
    video_meta_attrs_for,
)
from sglang.srt.disaggregation.image_loader import _MultiImageProcessingLoader
from sglang.srt.distributed.parallel_state import (
    get_default_distributed_backend,
    get_mooncake_transfer_engine,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqInput, ProfileReqType
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_loader import get_model
from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.observability.req_time_stats import convert_time_to_realtime_ns
from sglang.srt.observability.trace import (
    TraceReqContext,
    TraceSliceContext,
    get_global_tracing_enabled,
    process_tracing_init,
    trace_set_thread_info,
)
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import (
    MMDownloadError,
    add_prometheus_middleware,
    configure_logger,
    load_audio,
    load_image,
    load_video,
    random_uuid,
    set_prometheus_multiproc_dir,
)
from sglang.srt.utils.common import maybe_reindex_device_id
from sglang.srt.utils.network import (
    NetworkAddress,
    get_free_port,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

IMAGE_LOADER_NUM_WORKERS = int(os.getenv("SGLANG_IMAGE_LOADER_NUM_WORKERS", "10"))
image_processing_queue = Queue(maxsize=65535)
image_processed_queue = Queue(maxsize=65535)

HEALTH_CHECK_TIMEOUT = 30

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

rid_lock = asyncio.Lock()
rid_to_receive_endpoint: Dict[str, List[str]] = dict()
rid_to_receive_count: Dict[str, int] = dict()
rid_to_err_msg: Dict[str, str] = dict()
cond_dict_lock = asyncio.Lock()
rid_to_cond: Dict[str, asyncio.Condition] = {}

use_image_processor_gpu = envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get()

ENCODER_MAX_BATCH_SIZE = envs.SGLANG_ENCODER_MAX_BATCH_SIZE.get()
# Watchdog: max time to wait for a batched /encode result. Bounds HTTP latency
# if the batch worker stalls (NCCL hang, dead worker proc, etc.).
ENCODER_REQ_TIMEOUT = envs.SGLANG_ENCODER_REQ_TIMEOUT.get()

# Per-modality dispatch weight (image=1 baseline). Tunable.
MODALITY_WEIGHTS: Dict[str, int] = {"image": 1, "audio": 20, "video": 50}
DEFAULT_MODALITY_WEIGHT = 1

# per-task batch timing sink, for span emission
_batch_timing_var: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "encoder_batch_timing", default=None
)


def _merge_batch_timing(stage_timing: Optional[dict], num_items: int) -> None:
    # min/max merge so per-video loops accumulate into one timing
    box = _batch_timing_var.get()
    if box is None or stage_timing is None:
        return
    for key, pick in (
        ("load_start_ts", min),
        ("preprocess_start_ts", min),
        ("forward_start_ts", min),
        ("load_end_ts", max),
        ("preprocess_end_ts", max),
        ("forward_end_ts", max),
    ):
        vals = [v for v in (box.get(key), stage_timing.get(key)) if v is not None]
        if vals:
            box[key] = pick(vals)
    box["load_timings"] = (box.get("load_timings") or []) + (
        stage_timing.get("load_timings") or []
    )
    box["num_miss_items"] = box.get("num_miss_items", 0) + num_items
    box["num_total_items"] = box.get("num_total_items", 0) + num_items


class MMError(Exception):
    def __init__(self, message, code=HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BadRequestError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.BAD_REQUEST)


class InternalError(MMError):
    def __init__(self, message):
        super().__init__(message, code=HTTPStatus.INTERNAL_SERVER_ERROR)


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy."""

    def __init__(self, tensor):
        # Ensure tensor is on CPU and contiguous
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Keep tensor reference
        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


def _convert(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return torch.tensor(np.array(data))
    elif isinstance(data, list) and isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        return data


_mm_grid_attrs = {
    Modality.IMAGE: ["image_grid_thw", "image_grid_hws"],
    Modality.VIDEO: ["video_grid_thw"],
    Modality.AUDIO: ["audio_feature_lens_raw"],
}

_mm_feature_attrs = {
    Modality.IMAGE: ["pixel_values"],
    Modality.VIDEO: ["pixel_values_videos"],
    Modality.AUDIO: ["input_features"],
}


def _get_mm_grid_dim(mm_inputs, modality, model_type: Optional[str] = None):
    for attr in _mm_grid_attrs[modality]:
        if attr in mm_inputs and mm_inputs[attr] is not None:
            return mm_inputs[attr]
    raise ValueError(f"Grid dim ({_mm_grid_attrs[modality]}) not found in {mm_inputs}")


def _get_mm_feature(mm_inputs, modality):
    for attr in _mm_feature_attrs[modality]:
        if attr in mm_inputs:
            return mm_inputs[attr]
    raise ValueError(
        f"Feature attrs ({_mm_feature_attrs[modality]}) not found in {mm_inputs}"
    )


def _build_mm_aux_data(mm_inputs, model_type=None):
    # Video aux metadata, scoped to model_type's video-meta attrs.
    return {attr: mm_inputs.get(attr) for attr in video_meta_attrs_for(model_type)}


_image_grid_attrs = ["image_grid_thw", "image_grid_hws"]


def _get_image_grid_dim(images_input):
    for attr in _image_grid_attrs:
        if attr in images_input:
            return images_input[attr]
    raise ValueError(
        f"Image grid dim ({_image_grid_attrs}) not found in {images_input}"
    )


def _get_image_grid_attr_name(images_input):
    """Get the name of the grid attribute used in images_input."""
    for attr in _image_grid_attrs:
        if attr in images_input:
            return attr
    raise ValueError(
        f"Image grid dim ({_image_grid_attrs}) not found in {images_input}"
    )


def _compute_grid_products(grid_dim, grid_attr_name: str) -> List[int]:
    """
    Compute the raw grid products for each image (before any merge factor).

    Args:
        grid_dim: Grid dimensions tensor/list.
                  For image_grid_thw: tensor of shape (num_images, 3) with (t, h, w)
                  For image_grid_hws: list of (h, w) tuples
        grid_attr_name: The attribute name ("image_grid_thw" or "image_grid_hws")

    Returns:
        List of grid products, one per image
    """
    grid_products = []

    if grid_attr_name == "image_grid_thw":
        # grid_dim is tensor of shape (num_images, 3) with (temporal, height, width)
        if isinstance(grid_dim, torch.Tensor):
            for i in range(grid_dim.shape[0]):
                t, h, w = grid_dim[i].tolist()
                product = int(t * h * w)
                grid_products.append(product)
        elif isinstance(grid_dim, (list, tuple)):
            for grid in grid_dim:
                if isinstance(grid, torch.Tensor):
                    t, h, w = grid.tolist()
                else:
                    t, h, w = grid
                product = int(t * h * w)
                grid_products.append(product)
    elif grid_attr_name == "image_grid_hws":
        # grid_dim is list of (height, width) tuples
        if isinstance(grid_dim, torch.Tensor):
            for i in range(grid_dim.shape[0]):
                h, w = grid_dim[i].tolist()
                product = int(h * w)
                grid_products.append(product)
        elif isinstance(grid_dim, (list, tuple)):
            for grid in grid_dim:
                if isinstance(grid, torch.Tensor):
                    h, w = grid.tolist()
                else:
                    h, w = grid
                product = int(h * w)
                grid_products.append(product)
    else:
        # Fallback: try to compute product of dimensions
        if isinstance(grid_dim, torch.Tensor):
            for i in range(grid_dim.shape[0]):
                product = int(torch.prod(grid_dim[i]).item())
                grid_products.append(product)
        elif isinstance(grid_dim, (list, tuple)):
            for grid in grid_dim:
                if isinstance(grid, torch.Tensor):
                    product = int(torch.prod(grid).item())
                else:
                    product = int(np.prod(grid))
                grid_products.append(product)

    return grid_products


def _compute_tokens_per_image(
    grid_products: List[int],
    total_tokens: int,
) -> List[int]:
    """
    Compute the actual number of tokens for each image, accounting for model merge factors.

    The model may apply spatial merging (e.g., merge_size=2 means 2x2 patches merged into 1),
    so actual tokens = grid_product / merge_factor^2.

    We compute the merge factor from the total tokens and grid products.

    Args:
        grid_products: List of grid products (t*h*w or h*w) per image
        total_tokens: Actual total tokens from model output

    Returns:
        List of actual token counts per image
    """
    total_grid_product = sum(grid_products)

    if total_grid_product == 0:
        return [0] * len(grid_products)

    # Compute the ratio (merge factor squared)
    # This handles cases where model applies spatial merging
    ratio = total_grid_product / total_tokens

    # Calculate tokens per image based on the ratio
    tokens_per_image = []
    for product in grid_products:
        tokens = int(round(product / ratio))
        tokens_per_image.append(tokens)

    # Adjust for rounding errors to ensure sum matches exactly
    current_sum = sum(tokens_per_image)
    if current_sum != total_tokens and len(tokens_per_image) > 0:
        # Distribute the difference to the last image
        diff = total_tokens - current_sum
        tokens_per_image[-1] += diff

    return tokens_per_image


def _describe_mm_source(mm_items, max_len: int = 96) -> str:
    def _one(item) -> str:
        if isinstance(item, str):
            # data URI: keep the scheme/mime prefix, drop the base64 payload.
            if item.startswith("data:"):
                return f"{item.split(',', 1)[0]},<{len(item)}B>"
            if len(item) > max_len:
                return f"{item[:max_len]}...(<{len(item)}B>)"
            return item
        if isinstance(item, dict):
            for k in ("url", "image_url", "video_url", "audio_url", "path", "name"):
                v = item.get(k)
                if isinstance(v, dict):
                    v = v.get("url") or v.get("path")
                if isinstance(v, str) and v:
                    return _one(v)
            return f"<dict keys={list(item.keys())[:5]}>"
        return f"<{type(item).__name__}>"

    if isinstance(mm_items, (list, tuple)):
        flat = []
        for it in mm_items:
            flat.extend(it if isinstance(it, (list, tuple)) else [it])
        descs = [_one(x) for x in flat[:4]]
        if len(flat) > 4:
            descs.append(f"...(+{len(flat) - 4})")
        return "[" + ", ".join(descs) + "]"
    return _one(mm_items)


_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _item_to_url(item) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("url", "image_url", "video_url", "audio_url", "path", "name"):
            v = item.get(k)
            if isinstance(v, dict):
                v = v.get("url") or v.get("path")
            if isinstance(v, str) and v:
                return v
        return None
    return getattr(item, "url", None)


def _item_preprocess_kwargs(item) -> Optional[dict]:
    if isinstance(item, dict):
        return item.get("preprocess_kwargs")
    return getattr(item, "preprocess_kwargs", None)


def _extract_url_sha1_key(
    item,
    include_preprocess_kwargs: bool = False,
    preprocess_kwargs: Optional[dict] = None,
) -> Optional[int]:
    # None => no 40-hex SHA1 in the URL basename => skip the cache (no fallback).
    url_str = _item_to_url(item)
    if not url_str:
        return None
    # .lower() keeps the cache-key contract aligned with prefill-side routing
    # (encode_receiver.extract_mm_item_key), which also lowercases the basename.
    name = os.path.splitext(os.path.basename(url_str.split("?")[0]))[0].lower()
    if not _SHA1_RE.fullmatch(name):
        return None
    url_hash = int(name, 16)
    if not include_preprocess_kwargs:
        return url_hash
    # URL-string items carry no kwargs; caller passes them from the parallel
    # mm_items_preprocess_kwargs. Fall back to item-carried kwargs for dicts.
    pp = (
        preprocess_kwargs
        if preprocess_kwargs is not None
        else _item_preprocess_kwargs(item)
    )
    pp_str = str(sorted(pp.items())) if pp else ""
    return int(hashlib.sha1(f"{url_hash}:{pp_str}".encode()).hexdigest(), 16)


class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
        shm_resources=None,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.dp_rank = 0
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        self._build_vision_config(server_args.mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()
        logger.info(f"Resolved model audio sample rate: {self.model_audio_sr} Hz")

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )

        self.context = zmq.asyncio.Context(2)
        self.sync_context = zmq.Context()  # Reuse sync context for thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
        self.mm_cache = MultiModalStaticCache(
            embedding_cache_size * 1024 * 1024, shm_resources=shm_resources
        )
        self.mm_cache_lock = asyncio.Lock()

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_ENCODER_MM_LOAD_WORKERS", 4))
        )
        self.send_timeout = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()

        if schedule_path is not None:
            self.schedule_socket = get_zmq_socket(
                self.context, zmq.PULL, schedule_path, True
            )
        self.background_tasks: Set[asyncio.Task] = set()
        # Encoder-DP: req_id -> scheduled send-timeout embedding-cleanup task,
        # so the /send path can cancel it once the embedding is consumed.
        self._dp_inflight_cleanup_tasks: Dict[str, asyncio.Task] = {}

        if self.server_args.enable_mm_global_cache:
            from sglang.srt.mem_cache.storage.mooncake_store.embedding_cache_controller import (
                EmbeddingCacheController,
            )

            hidden_dims = self._infer_embedding_dims()
            self.mm_global_cache = EmbeddingCacheController(
                rank,
                server_args.tp_size,
                hidden_dims=hidden_dims,
                tp_group=get_tp_group().cpu_group,
                all_rank_get=False,
            )
        else:
            self.mm_global_cache = None

        if self.rank == 0:
            logger.info(
                f"Using transfer backend: {self.server_args.encoder_transfer_backend}"
            )

            if self.server_args.encoder_transfer_backend == "mooncake":
                self.local_ip = get_local_ip_auto()

                self.engine = get_mooncake_transfer_engine()
                if self.engine is None:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        init_mooncake_transfer_engine,
                    )

                    self.engine = init_mooncake_transfer_engine(
                        hostname=self.local_ip,
                        gpu_id=self.gpu_id,
                        ib_device=(
                            self.server_args.disaggregation_ib_device
                            or self.server_args.mooncake_ib_device
                        ),
                    )

            self.embedding_to_send = dict()

        self.image_loader = _MultiImageProcessingLoader(
            IMAGE_LOADER_NUM_WORKERS,
            self.load_image_sync,
            image_processing_queue,
            image_processed_queue,
        )
        logger.info(f"rank {rank} init finish ")

    def load_image_sync(self, input_queue, output_queue):
        while True:
            batch_mm_items, modality, input_texts, batch_id = input_queue.get()
            all_image, images_per_request, total_images = self.load_image_process(
                batch_mm_items, modality, input_texts
            )
            output_queue.put((all_image, images_per_request, total_images, batch_id))

    def load_image_process(
        self,
        batch_mm_items: List[List],
        modality: Modality = Modality.IMAGE,
        input_texts: List[str] = None,
    ):
        if input_texts is None:
            input_texts = [""] * len(batch_mm_items)

        # Currently only support IMAGE modality for batching
        if modality != Modality.IMAGE:
            # Fall back to sequential processing for non-image modalities
            results = []
            for i, mm_items in enumerate(batch_mm_items):
                input_text = input_texts[i] if i < len(input_texts) else ""
                loop = asyncio.get_running_loop()
                result = loop.run_until_complete(
                    self._encode(mm_items, modality, input_text)
                )
                results.append(result)
            return results

        # Step 1: Load and collect all images in parallel, tracking which belong to
        # which request
        all_images = []
        images_per_request = []  # Number of images in each request

        # Flatten all mm_items with their indices for parallel loading
        load_tasks = []  # List of (req_idx, img_idx, mm_item)
        for req_idx, mm_items in enumerate(batch_mm_items):
            if not isinstance(mm_items, (list, tuple)):
                mm_items = [mm_items]
            for img_idx, mm_item in enumerate(mm_items):
                load_tasks.append((req_idx, img_idx, mm_item))

        # Parallel image loading function
        async def load_single_image(
            req_idx: int, img_idx: int, mm_item
        ) -> Tuple[int, int, any]:
            """Load a single image, returns (req_idx, img_idx, loaded_image)."""
            loop = asyncio.get_event_loop()
            try:
                # Run blocking load_image in thread pool
                loaded = await loop.run_in_executor(None, lambda: load_image(mm_item))
                return (req_idx, img_idx, loaded[0] if loaded else None)
            except Exception as e:
                item_preview = (
                    mm_item[:200] if isinstance(mm_item, str) else str(type(mm_item))
                )
                logger.error(
                    f"Failed to load image {img_idx} in request {req_idx}: "
                    f"item={item_preview}, error={type(e).__name__}: {e}"
                )
                raise RuntimeError(
                    f"Failed to load image {img_idx} in request {req_idx} "
                    f"(item={item_preview[:50]}...): {type(e).__name__}: {e}"
                ) from e

        # Load all images in parallel
        logger.info(f"Loading {len(load_tasks)} images in parallel...")
        load_start = time.perf_counter()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        parallel_results = loop.run_until_complete(
            asyncio.gather(
                *[
                    load_single_image(req_idx, img_idx, mm_item)
                    for req_idx, img_idx, mm_item in load_tasks
                ],
                return_exceptions=True,
            )
        )

        load_end = time.perf_counter()
        logger.info(
            f"Parallel image loading took {(load_end - load_start) * 1000:.2f}ms"
        )

        # Check for errors
        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                raise result

        # Organize results by request
        # Build a dict: req_idx -> [(img_idx, image), ...]
        images_by_request = {}
        for req_idx, img_idx, image in parallel_results:
            if req_idx not in images_by_request:
                images_by_request[req_idx] = []
            images_by_request[req_idx].append((img_idx, image))

        # Sort images within each request by img_idx and collect
        for req_idx in range(len(batch_mm_items)):
            if req_idx in images_by_request:
                # Sort by img_idx to maintain order
                sorted_images = sorted(images_by_request[req_idx], key=lambda x: x[0])
                req_images = [img for _, img in sorted_images]
                images_per_request.append(len(req_images))
                all_images.extend(req_images)
            else:
                images_per_request.append(0)

        total_images = len(all_images)
        logger.info(
            f"Batch processing {total_images} images from "
            f"{len(batch_mm_items)} requests"
        )
        return all_images, images_per_request, total_images

    def _infer_embedding_dims(self) -> dict:
        """Infer per-modality embedding dimensions from hf_config at init time."""
        default = self.model_config.hidden_size
        hf_cfg = self.model_config.hf_config
        thinker_cfg = getattr(hf_cfg, "thinker_config", None)
        dims = {
            Modality.IMAGE: default,
            Modality.VIDEO: default,
            Modality.AUDIO: default,
        }

        vision_cfg = getattr(thinker_cfg, "vision_config", None) or getattr(
            hf_cfg, "vision_config", None
        )
        if vision_cfg is not None:
            out_hs = getattr(vision_cfg, "out_hidden_size", None)
            if out_hs is not None:
                ds = getattr(vision_cfg, "deepstack_visual_indexes", None)
                vis_dim = (
                    out_hs * (1 + len(ds))
                    if isinstance(ds, (list, tuple)) and ds
                    else out_hs
                )
                dims[Modality.IMAGE] = vis_dim
                dims[Modality.VIDEO] = vis_dim

        audio_cfg = getattr(thinker_cfg, "audio_config", None) or getattr(
            hf_cfg, "audio_config", None
        )
        if audio_cfg is not None:
            for attr in ("output_dim", "d_model"):
                val = getattr(audio_cfg, attr, None)
                if val and int(val) > 0:
                    dims[Modality.AUDIO] = int(val)
                    break

        logger.info(f"Global cache embedding dims: {dims}")
        return dims

    def _resolve_audio_sr(self) -> int:
        # Must match MiMoProcessor.from_hf_config — on drift, mimo tags the
        # ndarray with its own audio_sampling_rate and skips resample, so the
        # waveform is interpreted at the wrong rate and warped.
        def _read(obj, attr):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(attr)
            return getattr(obj, attr, None)

        audio_cfg = self.vision_config.get("audio", {})
        sr = audio_cfg.get("audio_sampling_rate")
        if sr:
            return int(sr)

        hf_cfg = self.model_config.hf_config
        thinker_cfg = _read(hf_cfg, "thinker_config")
        pc = _read(thinker_cfg, "processor_config") or _read(hf_cfg, "processor_config")
        sr = _read(pc, "audio_sampling_rate")
        if sr:
            return int(sr)
        ac = _read(thinker_cfg, "audio_config") or _read(hf_cfg, "audio_config")
        for attr in ("sampling_rate", "sample_rate"):
            sr = _read(ac, attr)
            if sr:
                return int(sr)

        sr = audio_cfg.get("sampling_rate")
        if sr:
            return int(sr)
        logger.warning(
            "No audio sampling rate found in mm_config or hf_config; "
            "falling back to 16000 Hz. If the model expects a different SR "
            "(e.g. MiMo-V2 defaults to 24000), audio will be warped."
        )
        return 16000

    def _build_vision_config(self, mm_process_config):
        """
        Validate vision config, used for image/video/audio.
        If not provided, keep default values.
        """
        self.vision_config = (
            mm_process_config.get("vision_config", {})
            if mm_process_config is not None
            else {}
        )
        for modality_str in ["image", "video", "audio"]:
            if not self.vision_config.get(modality_str, None):
                self.vision_config[modality_str] = {}
            if self.use_image_processor_gpu:
                self.vision_config[modality_str]["device"] = self.device

            if modality_str == "video":
                video_defaults = {"fps": 2.0, "max_frames": 768, "min_frames": 4}
                for k, v in video_defaults.items():
                    self.vision_config["video"].setdefault(k, v)

            if modality_str == "audio":
                if "return_attention_mask" not in self.vision_config["audio"]:
                    self.vision_config["audio"]["return_attention_mask"] = True
                if "padding" not in self.vision_config["audio"]:
                    if self.model_type == "qwen2_audio":
                        # For Qwen2Audio, use padding="max_length"
                        # (same as https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L93)
                        self.vision_config["audio"]["padding"] = "max_length"
                    else:
                        self.vision_config["audio"]["padding"] = True
                if "truncation" not in self.vision_config["audio"]:
                    # keep same logic as base_processor.py
                    if (
                        hasattr(self, "audio_processor")
                        and self.audio_processor is not None
                    ):
                        if self.audio_processor.__class__.__name__ in {
                            "Gemma3nProcessor",
                            "GlmAsrProcessor",
                            "Qwen2AudioProcessor",
                            "Qwen3OmniMoeProcessor",
                        }:
                            self.vision_config["audio"]["truncation"] = False

    def _load_mm_processor(self, server_args: ServerArgs):
        """
        Load image/video/audio processor separately,
        avoid issues with AutoProcessor not recognizing certain models
        """
        from transformers import AutoImageProcessor, AutoVideoProcessor

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}")
            self.image_processor = None

        try:
            self.video_processor = AutoVideoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
        except Exception as e:
            logger.warning(f"Failed to load video processor: {e}")
            self.video_processor = None

        try:
            # Note: AutoProcessor is used for audio processor
            _audio_proc = AutoProcessor.from_pretrained(
                server_args.tokenizer_path or server_args.model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                use_fast=not server_args.disable_fast_image_processor,
            )
            if not hasattr(_audio_proc, "feature_extractor"):
                logger.warning(
                    "Loaded AutoProcessor has no feature_extractor attribute, "
                    "audio processing will be unavailable."
                )
                self.audio_processor = None
            else:
                self.audio_processor = _audio_proc
        except Exception as e:
            logger.warning(f"Failed to load audio processor: {e}")
            self.audio_processor = None

    def _load_single_item(
        self,
        data,
        modality: Modality,
        frame_count_limit=None,
        discard_alpha_channel=True,
        timing: Optional[dict] = None,
    ):
        """
        Load a single multimodal data.
        If data is precomputed, returns directly.
        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, False, timing=timing)
                if (
                    discard_alpha_channel
                    and not isinstance(img, torch.Tensor)
                    and img.mode != "RGB"
                ):
                    # Needed only when `img` is a PIL image
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit, timing=timing)
            elif modality == Modality.AUDIO:
                return load_audio(data, self.model_audio_sr, timing=timing)

        except MMDownloadError:
            # Network failure with an already-concise message; let it surface
            # as a 500 instead of being wrapped into a 400.
            raise
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(self, items, modalities, timings=None):
        futures = []
        task_info = []

        for data, modality in zip(items, modalities):
            if modality is not None:
                item_timing = {} if timings is not None else None
                if timings is not None:
                    timings.append(item_timing)
                futures.append(
                    self.io_executor.submit(
                        self._load_single_item,
                        data,
                        modality,
                        timing=item_timing,
                    )
                )
                task_info.append((modality, data))
        return futures, task_info

    def _get_feat_extract_output_lengths(self, feature_lens):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # qwen2_audio/qwen2.5_omni
        if self.model_type in ["qwen2_audio", "qwen2_5_omni"]:
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1
        # qwen3_omni_moe
        elif self.model_type == "qwen3_omni_moe":
            input_lengths_leave = feature_lens % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            output_lengths = (
                ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13
            )
            return output_lengths
        elif self.model_type == "mimo_v2":
            # MiMo-V2's preprocess_audio returns audio_token_len (already
            # post-encoder/avg-pooler/group-size). Stored in audio_feature_lens_raw,
            # so no further reduction here.
            return feature_lens
        else:
            # fallback to original HF audio sample logic for other models
            logger.warning(
                f"Fallback to original HF audio sample logic for {self.model_type}"
            )
            input_length = (feature_lens - 1) // 2 + 1
            return (input_length - 2) // 2 + 1

    async def _flatten_and_load_videos(self, mm_items, timings=None):
        if not isinstance(mm_items, (list, tuple)):
            mm_items = [mm_items]

        futures, _ = self.submit_data_loading_tasks(
            mm_items, [Modality.VIDEO] * len(mm_items), timings=timings
        )
        async_futures = [asyncio.wrap_future(f) for f in futures]
        video_items = await asyncio.gather(*async_futures)

        video_processor_kwargs = {}
        if "qwen" in self.model_type:
            # for qwen-series model, do sample frames before preprocess
            video_processed = [
                await preprocess_video(
                    video,
                    video_config=self.vision_config.get("video", {}),
                    timing=(timings[i] if timings is not None else None),
                )
                for i, video in enumerate(video_items)
            ]
            videos, video_metadata = map(list, zip(*video_processed))
            video_processor_kwargs["do_sample_frames"] = False
            if video_metadata:
                video_processor_kwargs["video_metadata"] = video_metadata
            return videos, video_processor_kwargs
        else:
            raise NotImplementedError(
                f"Video processing is not supported for {self.model_type} model."
            )

    async def _flatten_and_load_data_by_modality(
        self, mm_items, modality, timings=None
    ):
        """
        Flatten mm_items structure, load multimodal data concurrently, and restore original structure.

        Returns:
            Same structure as load_mm_items would return, support for image/audio
        """
        # Handle single mm_item (not a list)
        if not isinstance(mm_items, (list, tuple)):
            futures, _ = self.submit_data_loading_tasks(
                [mm_items], [modality], timings=timings
            )
            return await asyncio.wrap_future(futures[0])

        # Handle nested list (list of lists)
        if len(mm_items) > 0 and isinstance(mm_items[0], (list, tuple)):
            # Flatten nested structure
            flat_data = []
            flat_indices = []  # Track which group each item belongs to
            for group_idx, item_group in enumerate(mm_items):
                for item in item_group:
                    flat_data.append(item)
                    flat_indices.append(group_idx)

            # Submit all tasks concurrently
            futures, _ = self.submit_data_loading_tasks(
                flat_data, [modality] * len(flat_data), timings=timings
            )

            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            results = await asyncio.gather(*async_futures)

            # Restore nested structure
            nested_results = [[] for _ in range(len(mm_items))]
            for idx, result in zip(flat_indices, results):
                nested_results[idx].append(result)

            return nested_results

        # Handle simple list
        else:
            futures, _ = self.submit_data_loading_tasks(
                mm_items, [modality] * len(mm_items), timings=timings
            )
            # Wait for all tasks to complete asynchronously
            async_futures = [asyncio.wrap_future(f) for f in futures]
            return await asyncio.gather(*async_futures)

    def get_num_patches(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of raw patches (before merge/sampling). Used for pixel_values slicing."""
        if modality == Modality.AUDIO:
            return int(grid.item())
        else:
            return int(grid[0] * grid[1] * grid[2])

    def get_num_tokens(
        self, grid: Union[torch.Tensor, List[int]], modality: Modality
    ) -> int:
        """Calculate number of tokens (after 2x2 merge). Used for mm_embedding slicing."""
        if modality == Modality.AUDIO:
            input_length = self.get_num_patches(grid, modality)
            return self._get_feat_extract_output_lengths(input_length)
        else:
            merge_size = getattr(self.image_processor, "merge_size", 2)
            return self.get_num_patches(grid, modality) // (merge_size**2)

    def slice_embedding(
        self, mm_embedding: torch.Tensor, grid_thw: List, modality: Modality
    ) -> List[torch.Tensor]:
        """Slice a concatenated embedding tensor into individual image embeddings."""
        slices, offset = [], 0
        for grid in grid_thw:
            count = self.get_num_tokens(grid, modality)
            slices.append(mm_embedding[offset : offset + count])
            offset += count
        return slices

    def _calculate_hashes_from_features(
        self, mm_feature, grid_thw: List, modality: Modality
    ) -> List[str]:
        """CPU Task: Compute hashes based on processed feature patches."""
        hashes = []
        if modality == Modality.AUDIO and isinstance(mm_feature, list):
            for feature in mm_feature:
                tmp_item = MultimodalDataItem(modality=modality, feature=feature)
                tmp_item.set_pad_value()
                hashes.append(tmp_item.hash)
            return hashes

        offset = 0
        for grid in grid_thw:
            num_patches = self.get_num_patches(grid, modality)
            feature_slice = mm_feature[offset : offset + num_patches]
            tmp_item = MultimodalDataItem(modality=modality, feature=feature_slice)
            tmp_item.set_pad_value()
            hashes.append(tmp_item.hash)
            offset += num_patches
        return hashes

    async def _encode_missing(
        self,
        mm_feature,
        mm_inputs: dict,
        indices: List[int],
        modality: Modality = Modality.IMAGE,
        get_feature_fn=None,
        forward_timing: Optional[dict] = None,
        log_metrics: bool = True,
    ) -> List[torch.Tensor]:
        """
        GPU Task: Run ViT inference ONLY on the subset of mm items missing from the cache.
        """
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

        # Audio features are per-item (list of mels for mimo_v2, or batched
        # N x n_mels x T_max for qwen2_audio); slice by item index and keep
        # per-item shape. Image/video features are concatenated along the
        # patch dim; slice by cumulative patch offsets and cat.
        if modality == Modality.AUDIO:
            if isinstance(mm_feature, list):
                sub_feature = [mm_feature[i] for i in indices]
            else:
                sub_feature = mm_feature[list(indices)]
        else:
            sub_feature_list = []
            offsets = [0]
            curr = 0
            for g in grid_thw:
                curr += self.get_num_patches(g, modality)
                offsets.append(curr)
            for idx in indices:
                sub_feature_list.append(mm_feature[offsets[idx] : offsets[idx + 1]])
            sub_feature = torch.cat(sub_feature_list, dim=0)

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": (
                    sub_feature
                    if isinstance(sub_feature, list)
                    else _convert(sub_feature)
                ),
            }
        )

        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs.get(modality, []):
                continue
            val = _convert(v)
            if k in _mm_grid_attrs.get(modality, []):
                mm_item.set(k, val[indices])
            else:
                mm_item.set(k, val)

        forward_start = time.perf_counter()
        with torch.inference_mode():
            new_embeddings = get_feature_fn([mm_item]).cpu()
            if new_embeddings.ndim != 2:
                new_embeddings = new_embeddings.reshape(-1, new_embeddings.shape[-1])
        forward_end = time.perf_counter()
        forward_elapsed = forward_end - forward_start
        if forward_timing is not None:
            forward_timing["model_forward"] = (
                forward_timing.get("model_forward", 0.0) + forward_elapsed
            )
            forward_timing.setdefault("forward_start_ts", forward_start)
            forward_timing["forward_end_ts"] = forward_end
        elif log_metrics and encoder_metrics_collector is not None:
            encoder_metrics_collector.observe_model_forward(
                forward_elapsed, modality=modality.name.lower()
            )

        sub_grids = [grid_thw[i] for i in indices]
        return self.slice_embedding(new_embeddings, sub_grids, modality)

    async def _encode_missing_with_cache(
        self,
        mm_feature,
        mm_inputs: dict,
        grid_dim: List,
        modality: Modality,
        get_feat,
        forward_timing: Optional[dict] = None,
        log_metrics: bool = True,
    ) -> List[torch.Tensor]:
        # Content-feature-hash cache. TP-safe: missing_indices is deterministic across ranks.
        total = len(grid_dim)
        if not self.server_args.enable_prefix_mm_cache:
            return await self._encode_missing(
                mm_feature,
                mm_inputs,
                list(range(total)),
                modality,
                get_feat,
                forward_timing=forward_timing,
                log_metrics=log_metrics,
            )

        mc = encoder_metrics_collector if log_metrics else None
        modality_str = modality.name.lower()
        keys = self._calculate_hashes_from_features(mm_feature, grid_dim, modality)
        final_slices: List[Optional[torch.Tensor]] = [None] * total
        missing_indices: List[int] = []

        _t0 = time.perf_counter()
        async with self.mm_cache_lock:
            for i, k in enumerate(keys):
                hit = self.mm_cache.get_single(k)
                if hit is not None:
                    final_slices[i] = hit.embedding
                else:
                    missing_indices.append(i)
        if mc is not None:
            mc.observe_cache_read(time.perf_counter() - _t0, modality=modality_str)
            hits = total - len(missing_indices)
            mc.record_cache_files(hits, total, modality=modality_str)

        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature,
                mm_inputs,
                missing_indices,
                modality,
                get_feat,
                forward_timing=forward_timing,
                log_metrics=log_metrics,
            )
            _t0 = time.perf_counter()
            async with self.mm_cache_lock:
                for idx, emb in zip(missing_indices, new_slices):
                    final_slices[idx] = emb
                    self.mm_cache.set(keys[idx], EmbeddingResult(embedding=emb))
            if mc is not None:
                mc.observe_cache_write(time.perf_counter() - _t0, modality=modality_str)
                mc.set_cache_state(self.mm_cache.current_size, len(self.mm_cache))

        if mc is not None:
            total_tokens = sum(int(s.shape[0]) for s in final_slices)
            miss_tokens = sum(int(final_slices[i].shape[0]) for i in missing_indices)
            mc.record_cache_tokens(
                total_tokens - miss_tokens, total_tokens, modality=modality_str
            )

        return final_slices

    async def encode_with_global_cache(
        self,
        mm_items,
        modality: Modality,
        req_id: str,
        num_parts: int,
        part_idx: int,
        hashes: Optional[List[str]] = None,
        mm_items_preprocess_kwargs=None,
        log_metrics: bool = True,
    ) -> torch.Tensor:
        modality_str = modality.name.lower()
        mc = encoder_metrics_collector if log_metrics else None
        # mm_inputs: dict
        stage_timing = {} if mc is not None else None
        mm_inputs, get_feature_fn = await self._process_mm_items(
            mm_items, modality, mm_items_preprocess_kwargs, stage_timing=stage_timing
        )
        _observe_single_request_load(mc, stage_timing, modality_str)
        grid_thw = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
        num_items = len(grid_thw)

        # Hashes must be grid-space; a leaf-space list would size-mismatch
        # rank>0's mask (zeros(num_items)) and deadlock TP.
        if hashes is not None and len(hashes) != num_items:
            raise BadRequestError(
                f"User-supplied hashes length {len(hashes)} != grid count "
                f"{num_items} for {self.model_type}/{modality.name}; hashes "
                f"must be in grid space (1 per encoder grid entry)."
            )

        # Step 1: Rank 0 checks global cache and broadcasts hit/miss mask to all ranks.
        if self.rank == 0:
            if hashes is None:
                mm_hashes = self._calculate_hashes_from_features(
                    mm_feature, grid_thw, modality
                )
            else:
                mm_hashes = hashes
            exist_mask = await self.mm_global_cache.batch_is_exist(mm_hashes)
            mask_tensor = torch.tensor(
                [1 if e else 0 for e in exist_mask], dtype=torch.int32
            )
        else:
            mm_hashes = None
            mask_tensor = torch.zeros(num_items, dtype=torch.int32)

        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                mask_tensor,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        exist_mask = [m.item() == 1 for m in mask_tensor]
        missing_indices = [i for i, e in enumerate(exist_mask) if not e]
        hit_indices = [i for i, e in enumerate(exist_mask) if e]

        # Step 2: All ranks run ViT together on cache-miss images.
        new_slices = []
        if missing_indices:
            new_slices = await self._encode_missing(
                mm_feature,
                mm_inputs,
                missing_indices,
                modality,
                get_feature_fn,
                log_metrics=log_metrics,
            )

        # Step 3: Rank 0 prefetches cache-hit embeddings from global cache.
        prefetch_status = torch.tensor([1], dtype=torch.int32)

        if self.rank == 0:
            if hit_indices:
                hit_hashes = [mm_hashes[i] for i in hit_indices]
                hit_tokens = [
                    self.get_num_tokens(grid_thw[i], modality) for i in hit_indices
                ]
                self.mm_global_cache.prefetch(req_id, hit_hashes, hit_tokens, modality)

                try:

                    async def _wait_prefetch():
                        while not self.mm_global_cache.check_prefetch_progress(req_id):
                            await asyncio.sleep(0.005)

                    await asyncio.wait_for(_wait_prefetch(), timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        f"Prefetch failed for req {req_id}: {e}. "
                        f"Falling back to ViT for {len(hit_indices)} hit items."
                    )
                    prefetch_status[0] = 0

        # Step 4: Broadcast prefetch result to all ranks so they stay in sync.
        if self.server_args.tp_size > 1:
            torch.distributed.broadcast(
                prefetch_status,
                src=0,
                group=self.mm_global_cache.prefetch_tp_group,
            )

        # Step 5: If prefetch failed, all ranks fallback to ViT for the hit mm items.
        if prefetch_status.item() == 0 and hit_indices:
            logger.info(
                f"Req {req_id}: Prefetch failed, all ranks running ViT fallback "
                f"for {len(hit_indices)} mm items."
            )
            fallback_slices = await self._encode_missing(
                mm_feature,
                mm_inputs,
                hit_indices,
                modality,
                get_feature_fn,
                log_metrics=log_metrics,
            )
        else:
            fallback_slices = None

        # Step 6: Rank 0 assembles final embedding and prepares for sending.
        if self.rank == 0:
            final_slices = [None] * num_items

            for i, idx in enumerate(missing_indices):
                final_slices[idx] = new_slices[i]

            # Fill in cache-hit embeddings (from prefetch or fallback)
            if prefetch_status.item() == 1 and hit_indices:
                cached_slices = self.mm_global_cache.get_embeddings(
                    [mm_hashes[i] for i in hit_indices]
                )
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = cached_slices[i]
            elif fallback_slices is not None:
                for i, idx in enumerate(hit_indices):
                    final_slices[idx] = fallback_slices[i]

            mm_embedding = torch.cat(final_slices, dim=0)

            # Background insert: store newly computed embeddings into global cache.
            # Includes both original misses and fallback-recomputed hits.
            all_new_hashes = [mm_hashes[i] for i in missing_indices]
            all_new_slices = list(new_slices)
            if fallback_slices is not None:
                all_new_hashes += [mm_hashes[i] for i in hit_indices]
                all_new_slices += list(fallback_slices)

            if all_new_hashes:

                async def _background_insert():
                    await asyncio.to_thread(
                        self.mm_global_cache.insert_batch,
                        all_new_hashes,
                        all_new_slices,
                    )

                task = asyncio.create_task(_background_insert())
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            aux_data = _build_mm_aux_data(mm_inputs, self.model_type)
            self.embedding_to_send[req_id] = EmbeddingData(
                req_id,
                num_parts,
                part_idx,
                grid_thw,
                modality,
                mm_embedding,
                **aux_data,
            )
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        else:
            return (0, 0, 0, None, None)

    @staticmethod
    def _flatten_nested_items(items):
        if not isinstance(items, (list, tuple)):
            return [items]

        flat = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flat.extend(MMEncoder._flatten_nested_items(item))
            else:
                flat.append(item)
        return flat

    async def _flatten_and_load_audios(self, mm_items, timings=None):
        """
        Flatten mm_items, load audios concurrently as np.ndarray at
        self.model_audio_sr, restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(
            mm_items, Modality.AUDIO, timings=timings
        )

    async def _flatten_and_load_images(self, mm_items, timings=None):
        """
        Flatten mm_items structure, load images concurrently, and restore original structure.
        """
        return await self._flatten_and_load_data_by_modality(
            mm_items, Modality.IMAGE, timings=timings
        )

    def _calculate_timestamps(self, indices, video_fps: float, merge_size: int = 2):
        """Calculate timestamps for video frames, used for qwen3_vl models."""
        # refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L255
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    async def _process_mm_items(
        self,
        mm_items,
        modality,
        mm_items_preprocess_kwargs=None,
        stage_timing: Optional[dict] = None,
    ):
        model_preprocessor = getattr(self.model, "preprocess_mm_for_encoder", None)

        load_timings = [] if stage_timing is not None else None
        _pp = {"preprocess": 0.0}
        if stage_timing is not None:
            stage_timing["load_start_ts"] = time.perf_counter()

        if modality == Modality.IMAGE:
            processor_input = await self._process_image_items(
                mm_items, model_preprocessor, load_timings, _pp
            )
        elif modality == Modality.VIDEO:
            processor_input = await self._process_video_items(
                mm_items,
                model_preprocessor,
                mm_items_preprocess_kwargs,
                load_timings,
                _pp,
            )
        elif modality == Modality.AUDIO:
            processor_input = await self._process_audio_items(
                mm_items, model_preprocessor, load_timings, _pp
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        if stage_timing is not None:
            stage_timing["load_timings"] = load_timings
            stage_timing["preprocess"] = _pp["preprocess"] + sum(
                t.get("preprocess", 0.0) for t in load_timings
            )
            # load ends where preprocess starts
            now = time.perf_counter()
            stage_timing["load_end_ts"] = _pp.get("preprocess_start_ts", now)
            stage_timing["preprocess_start_ts"] = _pp.get("preprocess_start_ts", now)
            stage_timing["preprocess_end_ts"] = now

        target = self.model.thinker if hasattr(self.model, "thinker") else self.model
        get_feature_method = getattr(target, f"get_{modality.name.lower()}_feature")
        return processor_input, get_feature_method

    async def _process_image_items(
        self, mm_items, model_preprocessor, load_timings=None, pp_timing=None
    ):
        if not (self.image_processor or model_preprocessor):
            raise ValueError("No image processor available")
        images = await self._flatten_and_load_images(mm_items, timings=load_timings)
        _pp_start = time.perf_counter()
        if pp_timing is not None:
            pp_timing.setdefault("preprocess_start_ts", _pp_start)
        if model_preprocessor:
            result = model_preprocessor(images, Modality.IMAGE, self.vision_config)
        else:
            image_config = self.vision_config.get("image", {})
            result = self.image_processor(images=images, **image_config)
        if pp_timing is not None:
            pp_timing["preprocess"] += time.perf_counter() - _pp_start
        return result

    async def _process_video_items(
        self,
        mm_items,
        model_preprocessor,
        mm_items_preprocess_kwargs=None,
        load_timings=None,
        pp_timing=None,
    ):
        if model_preprocessor:
            _pp_start = time.perf_counter()
            if pp_timing is not None:
                pp_timing.setdefault("preprocess_start_ts", _pp_start)
            _n_before = len(load_timings) if load_timings is not None else 0
            result = model_preprocessor(
                mm_items,
                Modality.VIDEO,
                self.vision_config,
                mm_items_preprocess_kwargs,
                load_timings=load_timings,
            )
            if pp_timing is not None:
                elapsed = time.perf_counter() - _pp_start
                # exclude in-preprocessor download/decode
                if load_timings is not None:
                    for t in load_timings[_n_before:]:
                        elapsed -= t.get("download", 0.0) + t.get("decode", 0.0)
                pp_timing["preprocess"] += max(0.0, elapsed)
            return result
        if not self.video_processor:
            raise ValueError("No video processor available")

        videos, video_processor_kwargs = await self._flatten_and_load_videos(
            mm_items, timings=load_timings
        )
        _pp_start = time.perf_counter()
        if pp_timing is not None:
            pp_timing.setdefault("preprocess_start_ts", _pp_start)
        processor_input = self.video_processor(videos=videos, **video_processor_kwargs)
        if pp_timing is not None:
            pp_timing["preprocess"] += time.perf_counter() - _pp_start

        # Get additional video metadata
        if (
            self.model_type in ["qwen3_vl", "qwen3_vl_moe"]
            and video_processor_kwargs.get("video_metadata", None) is not None
        ):
            # For qwen3-vl models, we need to store the video timestamps
            video_metadata = video_processor_kwargs["video_metadata"]
            try:
                merge_size = (
                    self.model_config.hf_config.vision_config.spatial_merge_size
                )
            except (AttributeError, KeyError):
                merge_size = 2  # Default merge_size

            video_timestamps = []
            for metadata in video_metadata:
                video_fps = metadata.get("fps", None) or 24  # original video fps
                frames_indices = metadata.get("frames_indices", None)
                timestamps = self._calculate_timestamps(
                    frames_indices, video_fps, merge_size
                )
                video_timestamps.append(timestamps)
            processor_input["video_timestamps"] = video_timestamps
        elif (
            self.model_type in ["qwen2_5_vl", "qwen2_5_omni", "qwen3_omni_moe"]
            and processor_input.get("video_grid_thw", None) is not None
        ):
            # For omni/qwen2_5_vl models, calculate second_per_grid_ts for rotary embedding
            video_grid_thw = processor_input["video_grid_thw"]
            try:
                temporal_patch_size = self.video_processor.temporal_patch_size
            except AttributeError:
                temporal_patch_size = 2  # Default temporal_patch_size
            # get sampled fps, default: 2
            fps_list = [
                self.vision_config.get("video", {}).get("fps", None) or 2
            ] * len(video_grid_thw)
            second_per_grid_ts = [(temporal_patch_size / fps) for fps in fps_list]
            second_per_grid_ts_tensor = torch.tensor(
                second_per_grid_ts, dtype=torch.float32
            )
            processor_input["second_per_grid_ts"] = second_per_grid_ts_tensor

        return processor_input

    async def _process_audio_items(
        self, mm_items, model_preprocessor, load_timings=None, pp_timing=None
    ):
        # Await off the event loop so EncoderScheduler can accumulate
        # cross-request batches during download.
        audios = await self._flatten_and_load_audios(mm_items, timings=load_timings)

        _pp_start = time.perf_counter()
        if pp_timing is not None:
            pp_timing.setdefault("preprocess_start_ts", _pp_start)
        if model_preprocessor:
            result = model_preprocessor(audios, Modality.AUDIO, self.vision_config)
            if pp_timing is not None:
                pp_timing["preprocess"] += time.perf_counter() - _pp_start
            return result

        if not self.audio_processor:
            raise ValueError("No audio processor available")

        audio_config = self.vision_config.get("audio", {})
        processor_input = self.audio_processor.feature_extractor(audios, **audio_config)
        processor_input["feature_attention_mask"] = processor_input.pop(
            "attention_mask"
        )
        input_lengths = torch.tensor(
            processor_input["feature_attention_mask"].sum(-1), dtype=torch.long
        )
        processor_input["audio_feature_lens_raw"] = input_lengths
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        processor_input["audio_feature_lens"] = output_lengths
        if pp_timing is not None:
            pp_timing["preprocess"] += time.perf_counter() - _pp_start
        return processor_input

    async def _encode_group(
        self,
        mm_items,
        modality: Modality,
        content_hash_lookup: bool = True,
        mm_items_preprocess_kwargs=None,
        log_metrics: bool = True,
    ) -> Tuple[object, torch.Tensor, dict, Optional[int], bool]:
        # Shared encode core (download+preprocess+ViT). Does NOT store.
        # Returns (grid, embedding, aux, content_mm_hash, cache_hit); in content
        # mode it also does the content-hash lookup (cache_hit True on a hit).
        modality_str = modality.name.lower()
        mc = encoder_metrics_collector if log_metrics else None
        stage_timing = {} if (mc is not None or get_global_tracing_enabled()) else None
        try:
            mm_inputs, get_feature_fn = await self._process_mm_items(
                mm_items,
                modality,
                mm_items_preprocess_kwargs,
                stage_timing=stage_timing,
            )
        except NotImplementedError as e:
            raise InternalError(f"Not implemented error: {str(e)}")
        except MMDownloadError as e:
            raise InternalError(str(e))
        except Exception as e:
            raise BadRequestError(f"Failed to process mm items: {str(e)}")
        _observe_single_request_load(mc, stage_timing, modality_str)

        mm_embedding = None
        cache_hit = False
        content_mm_hash = None
        cache_enabled = self.server_args.enable_prefix_mm_cache
        url_sha1_only = envs.SGLANG_MM_CACHE_URL_SHA1_ONLY.get()

        mm_item = MultimodalDataItem.from_dict(
            {
                "modality": modality,
                "feature": _convert(_get_mm_feature(mm_inputs, modality)),
            }
        )
        for k, v in mm_inputs.items():
            if k in _mm_feature_attrs[modality]:
                continue
            mm_item.set(k, _convert(v))

        # Content-hash lookup needs the preprocessed feature, so it happens here (SHA1 mode looks up earlier).
        if content_hash_lookup and cache_enabled and not url_sha1_only:
            mm_item.set_pad_value()
            content_mm_hash = MultiModalStaticCache.combine_hashes([mm_item.hash])
            _t0 = time.perf_counter()
            async with self.mm_cache_lock:
                mm_cache = self.mm_cache.get_single(content_mm_hash)
                if mm_cache is not None:
                    mm_embedding = mm_cache.embedding
                    cache_hit = True
            if mc is not None:
                mc.observe_cache_read(time.perf_counter() - _t0, modality=modality_str)

        if mm_embedding is None:
            forward_start = time.perf_counter()
            with torch.inference_mode():
                mm_embedding = get_feature_fn([mm_item])
                mm_embedding = mm_embedding.cpu()
            if len(mm_embedding.shape) != 2:
                mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])
            forward_end = time.perf_counter()
            if stage_timing is not None:
                stage_timing["forward_start_ts"] = forward_start
                stage_timing["forward_end_ts"] = forward_end
            if mc is not None:
                mc.observe_model_forward(
                    forward_end - forward_start, modality=modality_str
                )

        aux_data = _build_mm_aux_data(mm_inputs, self.model_type)

        if modality == Modality.VIDEO and mm_inputs.get("video_audio_features"):
            target = (
                self.model.thinker if hasattr(self.model, "thinker") else self.model
            )
            encode_video_audio_fn = getattr(target, "encode_video_audio", None)
            if encode_video_audio_fn is not None:
                audio_forward_start = time.perf_counter()
                audio_embedding = encode_video_audio_fn(mm_inputs)
                if stage_timing is not None:
                    stage_timing.setdefault("forward_start_ts", audio_forward_start)
                    stage_timing["forward_end_ts"] = time.perf_counter()
                if mc is not None:
                    mc.observe_model_forward(
                        time.perf_counter() - audio_forward_start, modality="audio"
                    )
                if audio_embedding is not None:
                    aux_data["video_audio_embedding"] = audio_embedding
            else:
                logger.warning(
                    "Videos carry audio tracks but model has no "
                    "encode_video_audio; dropping audio for EPD encoding."
                )

        grid_dim = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
        _merge_batch_timing(stage_timing, num_items=len(grid_dim))
        return grid_dim, mm_embedding, aux_data, content_mm_hash, cache_hit

    @staticmethod
    def _recombine_video(per_video: List[Tuple]) -> Tuple[object, torch.Tensor, dict]:
        # Recombine per-video in original order, byte-identical to the whole-request path.
        grids = [g for g, _, _ in per_video]
        embs = [e for _, e, _ in per_video]
        grid_dim = grids[0] if len(grids) == 1 else torch.cat(grids, dim=0)
        mm_embedding = embs[0] if len(embs) == 1 else torch.cat(embs, dim=0)

        # Preserve aux key order as first seen across videos.
        aux_keys: List[str] = []
        for _, _, a in per_video:
            for k in a.keys():
                if k not in aux_keys:
                    aux_keys.append(k)

        aux_data: dict = {}
        for k in aux_keys:
            parts = [a[k] for _, _, a in per_video if a.get(k) is not None]
            if not parts:
                continue
            if isinstance(parts[0], torch.Tensor):
                aux_data[k] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            elif isinstance(parts[0], (list, tuple)):
                chained: list = []
                for p in parts:
                    chained.extend(list(p))
                aux_data[k] = chained
            else:
                # Scalars / other: collect into a list (one per contributing video).
                aux_data[k] = parts if len(parts) > 1 else parts[0]
        return grid_dim, mm_embedding, aux_data

    async def _encode_video_per_item(
        self,
        leaves: List,
        sha1_keys: List[int],
        mm_items_preprocess_kwargs=None,
    ) -> Tuple[object, torch.Tensor, dict]:
        # Per-video URL-SHA1 cache: hits skip download+preprocess+ViT, only misses are encoded.
        mc = encoder_metrics_collector
        n = len(leaves)
        per_video: List[Optional[Tuple]] = [None] * n  # (grid, embedding, aux)
        miss_pos: List[int] = []

        _t0 = time.perf_counter()
        async with self.mm_cache_lock:
            for i, k in enumerate(sha1_keys):
                hit = self.mm_cache.get_single(k)
                # Require grid; legacy entries without it are re-encoded.
                if hit is not None and hit.grid is not None:
                    per_video[i] = (hit.grid, hit.embedding, hit.aux or {})
                else:
                    miss_pos.append(i)
        if mc is not None:
            mc.observe_cache_read(time.perf_counter() - _t0, modality="video")
            hits = n - len(miss_pos)
            mc.record_cache_files(hits, n, modality="video")

        # Encode each miss video on its own so its cache entry is a genuine single-video result.
        for pos in miss_pos:
            # preprocess_kwargs is index-aligned with leaves; pass this video's
            # entry as a single-element list so its sampling params still apply.
            pk = (
                [mm_items_preprocess_kwargs[pos]]
                if mm_items_preprocess_kwargs is not None
                and pos < len(mm_items_preprocess_kwargs)
                else None
            )
            grid_dim, mm_embedding, aux_data, _, _ = await self._encode_group(
                [leaves[pos]],
                Modality.VIDEO,
                content_hash_lookup=False,
                mm_items_preprocess_kwargs=pk,
            )
            per_video[pos] = (grid_dim, mm_embedding, aux_data)
            async with self.mm_cache_lock:
                self.mm_cache.set(
                    sha1_keys[pos],
                    EmbeddingResult(
                        embedding=mm_embedding, grid=grid_dim, aux=aux_data
                    ),
                )

        if self.profiler is not None:
            self.profiler.step()
        return self._recombine_video(per_video)

    async def _encode(
        self,
        mm_items,
        modality: Modality,
        mm_items_preprocess_kwargs=None,
        log_metrics: bool = True,
    ) -> torch.Tensor:
        modality_str = modality.name.lower()
        mc = encoder_metrics_collector if log_metrics else None

        cache_enabled = self.server_args.enable_prefix_mm_cache
        url_sha1_only = envs.SGLANG_MM_CACHE_URL_SHA1_ONLY.get()
        mm_source = _describe_mm_source(mm_items) if cache_enabled else None

        # SHA1-only mode keys the whole request on URL SHA1s (known pre-download); any non-SHA1 item skips the cache.
        sha1_skip = False
        sha1_keys = None
        mm_hash = None
        if cache_enabled and url_sha1_only:
            leaves = MMEncoder._flatten_nested_items(mm_items)
            # VIDEO folds per-video sampling params into the key so they don't
            # collide; params come from the leaf-aligned parallel list.
            include_pp = modality == Modality.VIDEO
            pp_list = mm_items_preprocess_kwargs or []
            sha1_keys = [
                _extract_url_sha1_key(
                    it,
                    include_preprocess_kwargs=include_pp,
                    preprocess_kwargs=pp_list[i] if i < len(pp_list) else None,
                )
                for i, it in enumerate(leaves)
            ]
            if not leaves or any(k is None for k in sha1_keys):
                sha1_skip = True
                logger.info(
                    f"[mm_cache] SKIP (no url-sha1) modality={modality.name} "
                    f"url={mm_source} mode=sha1"
                )
            # VIDEO multi-leaf: cache per-video so a request can reuse one cached video. TP-safe.
            elif modality == Modality.VIDEO and len(leaves) > 1:
                return await self._encode_video_per_item(
                    leaves, sha1_keys, mm_items_preprocess_kwargs
                )
            else:
                _t0 = time.perf_counter()
                async with self.mm_cache_lock:
                    hit = self.mm_cache.get_single(
                        mm_hash := MultiModalStaticCache.combine_hashes(sha1_keys)
                    )
                if mc is not None:
                    mc.observe_cache_read(
                        time.perf_counter() - _t0, modality=modality_str
                    )
                # Require grid; legacy entries without it fall through to recompute.
                if hit is not None and hit.grid is not None:
                    if self.profiler is not None:
                        self.profiler.step()
                    return (hit.grid, hit.embedding, hit.aux or {})

        try:
            grid_dim, mm_embedding, aux_data, content_mm_hash, cache_hit = (
                await self._encode_group(
                    mm_items,
                    modality,
                    mm_items_preprocess_kwargs=mm_items_preprocess_kwargs,
                )
            )
            if mm_hash is None:
                mm_hash = content_mm_hash

            if cache_enabled and mc is not None:
                total_tokens = int(mm_embedding.shape[0])
                hit_tokens = total_tokens if cache_hit else 0
                mc.record_cache_tokens(hit_tokens, total_tokens, modality=modality_str)
                mc.record_cache_files(1 if cache_hit else 0, 1, modality=modality_str)

            # Store grid + aux so a later pre-download hit can rebuild EmbeddingData without preprocess.
            if cache_enabled and not sha1_skip:
                _t0 = time.perf_counter()
                async with self.mm_cache_lock:
                    entries_before = len(self.mm_cache)
                    already_present = self.mm_cache.has(mm_hash)
                    stored = self.mm_cache.set(
                        mm_hash,
                        EmbeddingResult(
                            embedding=mm_embedding, grid=grid_dim, aux=aux_data
                        ),
                    )
                    entries_after = len(self.mm_cache)
                    if mc is not None:
                        added = 0 if already_present else (1 if stored else 0)
                        evictions = max(0, added - (entries_after - entries_before))
                        if evictions > 0:
                            mc.inc_cache_evictions(
                                modality=modality_str, count=evictions
                            )
                        mc.set_cache_state(self.mm_cache.current_size, entries_after)
                if mc is not None:
                    mc.observe_cache_write(
                        time.perf_counter() - _t0, modality=modality_str
                    )
            if self.profiler is not None:
                self.profiler.step()

            return (
                grid_dim,
                mm_embedding,
                aux_data,
            )
        except BadRequestError as e:
            raise BadRequestError(f"Bad request error: {str(e)}")
        except Exception as e:
            raise InternalError(f"Internal encoding error: {str(e)}")

    async def _send(
        self,
        embedding: torch.Tensor,
        mm_data: EmbeddingData,
        session_id=None,
        buffer_address=None,
        prefill_host=None,
        embedding_port=None,
        url=None,
    ):
        if self.server_args.encoder_transfer_backend == "mooncake":
            self.engine.register(embedding.data_ptr(), embedding.nbytes)
            self.engine.transfer_sync(
                session_id, embedding.data_ptr(), buffer_address, embedding.nbytes
            )
            self.engine.deregister(embedding.data_ptr())

            mm_data.embedding = None

        # Send ack/data
        if url is not None:
            endpoint = NetworkAddress.parse(url).to_tcp()
        else:
            endpoint = NetworkAddress(prefill_host, embedding_port).to_tcp()
        logger.info(f"{endpoint = }")

        # Serialize data
        if self.server_args.encoder_transfer_backend == "mooncake":
            mm_data.send_time = time.time()
            serialized_data = pickle.dumps(mm_data)
            buffer = None
        else:
            new_mm_data = mm_data.copy_without_embedding()
            new_mm_data.send_time = time.time()
            if new_mm_data.error_msg is not None:
                buffer = None
                serialized_data = pickle.dumps(new_mm_data)
            else:
                embedding_tensor = TensorWrapper(mm_data.embedding)
                serialized_data = pickle.dumps(new_mm_data)
                buffer = embedding_tensor.__buffer__()

        # Use thread pool executor for parallel ZMQ send operations
        def send_with_socket():
            # No config_socket here: a fixed SO_SNDBUF disables kernel TCP
            # autotuning and is truncated by net.core.wmem_max, capping the
            # E->P embedding transfer at ~40MB/s.
            sock = self.sync_context.socket(zmq.PUSH)
            try:
                sock.connect(endpoint)
                if buffer is not None:
                    sock.send_multipart([serialized_data, buffer], copy=False)
                else:
                    sock.send_multipart([serialized_data], copy=False)
            finally:
                sock.close()

        await asyncio.get_event_loop().run_in_executor(self.executor, send_with_socket)

        if (
            encoder_metrics_collector is not None
            and embedding is not None
            and mm_data.error_msg is None
        ):
            modality_str = (
                mm_data.modality.name.lower()
                if hasattr(mm_data.modality, "name")
                else str(mm_data.modality).lower()
            )
            encoder_metrics_collector.observe_embedding_tokens_per_send(
                embedding.shape[0], modality=modality_str
            )

    async def _cleanup_inflight_encode_state(self, req_id: str):
        # Encoder-DP cleanup adapter. This MMEncoder has no inflight-encode state
        # machine (the upstream mooncake async / forward-results path), so the
        # cleanup is: cancel any pending send-timeout task and free the embedding.
        task = self._dp_inflight_cleanup_tasks.pop(req_id, None)
        if task is not None and not task.done():
            task.cancel()
        self.embedding_to_send.pop(req_id, None)

    def _schedule_inflight_encode_cleanup(self, req_id: str):
        # Free the held embedding if the follow-up /send never arrives, bounded to
        # send_timeout (pairs with the dispatcher-side stale-mapping GC). Cancelled
        # by _cleanup_inflight_encode_state when /send does arrive.
        async def _cleanup_later():
            try:
                await asyncio.sleep(self.send_timeout)
                self.embedding_to_send.pop(req_id, None)
            finally:
                self._dp_inflight_cleanup_tasks.pop(req_id, None)

        prev = self._dp_inflight_cleanup_tasks.pop(req_id, None)
        if prev is not None and not prev.done():
            prev.cancel()
        task = asyncio.create_task(_cleanup_later())
        self._dp_inflight_cleanup_tasks[req_id] = task
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def encode(
        self,
        mm_items,
        modality: Modality,
        req_id,
        num_parts,
        part_idx,
        mm_items_preprocess_kwargs=None,
        log_metrics: bool = True,
    ):
        try:
            grid_dim, mm_embedding, aux_data = await self._encode(
                mm_items,
                modality,
                mm_items_preprocess_kwargs,
                log_metrics=log_metrics,
            )

            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    grid_dim,
                    modality,
                    mm_embedding,
                    **aux_data,
                )
                self.embedding_to_send[req_id] = mm_data
            return (
                mm_embedding.nbytes,
                mm_embedding.shape[0],
                mm_embedding.shape[1],
                None,
                None,
            )
        except Exception as e:
            error_code = getattr(e, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
            error_msg = str(e)
            logger.error(
                f"Rank {self.rank} encode failed: {error_msg} {error_code = }",
                exc_info=True,
            )
            if self.rank == 0:
                mm_data = EmbeddingData(
                    req_id,
                    num_parts,
                    part_idx,
                    None,
                    modality,
                    error_msg=error_msg,
                    error_code=error_code,
                )
                self.embedding_to_send[req_id] = mm_data
                logger.debug(f"Created error EmbeddingData: {mm_data}")
            return 0, 0, 0, error_msg, error_code

    async def encode_request(self, req: dict, modality: Modality):
        """Single-request encode dispatcher: picks cache vs no-cache path."""
        mm_items_preprocess_kwargs = req.get("mm_items_preprocess_kwargs")
        log_metrics = req.get("log_metrics", True)
        if self.mm_global_cache is not None:
            return await self.encode_with_global_cache(
                mm_items=req["mm_items"],
                modality=modality,
                req_id=req["req_id"],
                num_parts=req["num_parts"],
                part_idx=req["part_idx"],
                hashes=req.get("hashes"),
                mm_items_preprocess_kwargs=mm_items_preprocess_kwargs,
                log_metrics=log_metrics,
            )
        return await self.encode(
            mm_items=req["mm_items"],
            modality=modality,
            req_id=req["req_id"],
            num_parts=req["num_parts"],
            part_idx=req["part_idx"],
            mm_items_preprocess_kwargs=mm_items_preprocess_kwargs,
            log_metrics=log_metrics,
        )

    def _finish_batch_timing(
        self,
        stage_timing: Optional[dict],
        forward_timing: Optional[dict],
        mc,
        items_per_req: List[int],
        num_requests: int,
        num_miss_items: int,
        num_total_items: int,
        modality_str: str,
        miss_indices: Optional[List[int]] = None,
    ) -> None:
        if stage_timing is None:
            return
        if mc is not None:
            try:
                _observe_download_decode_per_request(
                    mc,
                    stage_timing.get("load_timings"),
                    items_per_req,
                    modality_str,
                    miss_indices=miss_indices,
                )
                _observe_per_request(
                    mc.observe_preprocess,
                    stage_timing.get("preprocess", 0.0),
                    num_requests=num_requests,
                    modality=modality_str,
                )
                _observe_per_request(
                    mc.observe_model_forward,
                    forward_timing.get("model_forward", 0.0),
                    num_requests=num_requests,
                    modality=modality_str,
                )
            except Exception:
                logger.warning(
                    "Failed to record batch encode metrics", exc_info=True
                )
        stage_timing.update(forward_timing)
        stage_timing["num_miss_items"] = num_miss_items
        stage_timing["num_total_items"] = num_total_items
        box = _batch_timing_var.get()
        if box is not None:
            box.update(stage_timing)

    async def batch_encode(
        self, requests: List[dict], modality: Modality, log_metrics: bool = True
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        # IMAGE/AUDIO cross-request fusion. URL-SHA1 mode → _batch_encode_sha1, else _legacy_batch_encode.
        fast = (
            self.server_args.enable_prefix_mm_cache
            and envs.SGLANG_MM_CACHE_URL_SHA1_ONLY.get()
        )
        if fast:
            return await self._batch_encode_sha1(
                requests, modality, log_metrics=log_metrics
            )
        return await self._legacy_batch_encode(
            requests, modality, log_metrics=log_metrics
        )

    async def _batch_encode_sha1(
        self, requests: List[dict], modality: Modality, log_metrics: bool = True
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        # URL-SHA1 fast path: look up before download; only misses are encoded. TP-safe.
        mc = encoder_metrics_collector if log_metrics else None
        modality_str = modality.name.lower()
        items_per_req, flat_items = [], []
        for req in requests:
            leaves = MMEncoder._flatten_nested_items(req["mm_items"])
            items_per_req.append(len(leaves))
            flat_items.extend(leaves)

        total = len(flat_items)
        if mc is not None:
            for n in items_per_req:
                mc.observe_mm_items_per_request(n, modality=modality_str)
            mc.observe_mm_items_per_batch(total, modality=modality_str)
        keys = [_extract_url_sha1_key(it) for it in flat_items]
        cached_emb: List[Optional[torch.Tensor]] = [None] * total
        cached_grid: List[Optional[torch.Tensor]] = [None] * total
        miss_indices: List[int] = []
        skip_indices: List[int] = []  # no url-sha1: encode but never store

        _t0 = time.perf_counter()
        async with self.mm_cache_lock:
            for i, k in enumerate(keys):
                if k is None:
                    skip_indices.append(i)
                    miss_indices.append(i)
                    continue
                hit = self.mm_cache.get_single(k)
                # Require grid; legacy entries without it are recomputed and re-stored.
                if hit is not None and hit.grid is not None:
                    cached_emb[i], cached_grid[i] = hit.embedding, hit.grid
                else:
                    miss_indices.append(i)
        if mc is not None:
            mc.observe_cache_read(time.perf_counter() - _t0, modality=modality_str)
            hits = total - len(miss_indices)
            mc.record_cache_tokens(hits, total, modality=modality_str)
            mc.record_cache_files(hits, total, modality=modality_str)

        # Download + preprocess + ViT only for the misses.
        stage_timing = {} if (mc is not None or get_global_tracing_enabled()) else None
        forward_timing = {} if stage_timing is not None else None
        if miss_indices:
            miss_items = [flat_items[i] for i in miss_indices]
            try:
                mm_inputs, get_feat = await self._process_mm_items(
                    miss_items, modality, stage_timing=stage_timing
                )
            except NotImplementedError as e:
                return self._batch_set_error(
                    requests, modality, InternalError(f"Not implemented error: {e}")
                )
            except MMDownloadError as e:
                return self._batch_set_error(requests, modality, InternalError(str(e)))
            except Exception as e:
                return self._batch_set_error(
                    requests,
                    modality,
                    BadRequestError(f"Failed to process mm items: {e}"),
                )

            try:
                mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
                miss_grid = _get_mm_grid_dim(mm_inputs, modality, self.model_type)
                # IMAGE/AUDIO expect one item per grid entry; otherwise the index<->grid map is unsafe.
                if len(miss_grid) != len(miss_items):
                    logger.info(
                        f"[mm_cache] batch SKIP-FAST (sha1; miss items/grid "
                        f"mismatch items={len(miss_items)} grid={len(miss_grid)}) "
                        f"modality={modality.name}"
                    )
                    return await self._legacy_batch_encode(requests, modality)

                miss_slices = await self._encode_missing(
                    mm_feature,
                    mm_inputs,
                    list(range(len(miss_items))),
                    modality,
                    get_feat,
                    forward_timing=forward_timing,
                    log_metrics=log_metrics,
                )

                skip_set = set(skip_indices)
                _t0 = time.perf_counter()
                async with self.mm_cache_lock:
                    for j, gidx in enumerate(miss_indices):
                        emb = miss_slices[j]
                        g = miss_grid[j : j + 1]
                        cached_emb[gidx], cached_grid[gidx] = emb, g
                        if gidx in skip_set:
                            continue
                        self.mm_cache.set(
                            keys[gidx], EmbeddingResult(embedding=emb, grid=g)
                        )
                if mc is not None:
                    mc.observe_cache_write(
                        time.perf_counter() - _t0, modality=modality_str
                    )
                    mc.set_cache_state(self.mm_cache.current_size, len(self.mm_cache))
            except Exception as e:
                return self._batch_set_error(
                    requests, modality, InternalError(f"Internal encoding error: {e}")
                )

        if self.profiler is not None:
            for _ in requests:
                self.profiler.step()

        if miss_indices:
            self._finish_batch_timing(
                stage_timing,
                forward_timing,
                mc,
                items_per_req,
                len(requests),
                len(miss_indices),
                total,
                modality_str,
                miss_indices=miss_indices,
            )

        # No aux_data: batch_encode only handles IMAGE/AUDIO.
        results = []
        offset = 0
        for req, n in zip(requests, items_per_req):
            es = cached_emb[offset : offset + n]
            gs = cached_grid[offset : offset + n]
            emb = es[0] if n == 1 else torch.cat(es, dim=0)
            grid = gs[0] if n == 1 else torch.cat(gs, dim=0)
            if self.rank == 0:
                self.embedding_to_send[req["req_id"]] = EmbeddingData(
                    req["req_id"],
                    req["num_parts"],
                    req["part_idx"],
                    grid,
                    modality,
                    emb,
                )
            results.append((emb.nbytes, emb.shape[0], emb.shape[1], None, None))
            offset += n
        return results

    async def _legacy_batch_encode(
        self, requests: List[dict], modality: Modality, log_metrics: bool = True
    ) -> List[Tuple[int, int, int, Optional[str], Optional[int]]]:
        # Whole-batch path: download+preprocess all, then content-hash cache.
        items_per_req, flat_items = [], []
        for req in requests:
            leaves = MMEncoder._flatten_nested_items(req["mm_items"])
            items_per_req.append(len(leaves))
            flat_items.extend(leaves)

        total = sum(items_per_req)
        mc = encoder_metrics_collector if log_metrics else None
        modality_str = modality.name.lower()
        if mc is not None:
            for n in items_per_req:
                mc.observe_mm_items_per_request(n, modality=modality_str)
            mc.observe_mm_items_per_batch(total, modality=modality_str)

        stage_timing = {} if (mc is not None or get_global_tracing_enabled()) else None
        try:
            mm_inputs, get_feat = await self._process_mm_items(
                flat_items, modality, stage_timing=stage_timing
            )
        except NotImplementedError as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Not implemented error: {e}")
            )
        except MMDownloadError as e:
            return self._batch_set_error(requests, modality, InternalError(str(e)))
        except Exception as e:
            return self._batch_set_error(
                requests, modality, BadRequestError(f"Failed to process mm items: {e}")
            )

        try:
            mm_feature = _convert(_get_mm_feature(mm_inputs, modality))
            grid_dim = _get_mm_grid_dim(mm_inputs, modality, self.model_type)

            forward_timing = {} if stage_timing is not None else None
            final_slices = await self._encode_missing_with_cache(
                mm_feature,
                mm_inputs,
                grid_dim,
                modality,
                get_feat,
                forward_timing=forward_timing,
                log_metrics=log_metrics,
            )

            if self.profiler is not None:
                for _ in requests:
                    self.profiler.step()

            self._finish_batch_timing(
                stage_timing,
                forward_timing,
                mc,
                items_per_req,
                len(requests),
                len(flat_items),
                total,
                modality_str,
            )

            # No aux_data: batch_encode only handles IMAGE/AUDIO.
            results = []
            offset = 0
            for req, n in zip(requests, items_per_req):
                slices = final_slices[offset : offset + n]
                emb = slices[0] if n == 1 else torch.cat(slices, dim=0)
                if self.rank == 0:
                    self.embedding_to_send[req["req_id"]] = EmbeddingData(
                        req["req_id"],
                        req["num_parts"],
                        req["part_idx"],
                        grid_dim[offset : offset + n],
                        modality,
                        emb,
                    )
                results.append((emb.nbytes, emb.shape[0], emb.shape[1], None, None))
                offset += n
            return results
        except Exception as e:
            return self._batch_set_error(
                requests, modality, InternalError(f"Internal encoding error: {e}")
            )

    def _batch_set_error(
        self, requests: List[dict], modality: Modality, exc: Exception
    ) -> List[Tuple[int, int, int, str, int]]:
        code = getattr(exc, "code", HTTPStatus.INTERNAL_SERVER_ERROR)
        msg = str(exc)
        logger.error(
            f"Rank {self.rank} batch_encode failed: {msg} {code = }", exc_info=True
        )
        if self.rank == 0:
            for req in requests:
                self.embedding_to_send[req["req_id"]] = EmbeddingData(
                    req["req_id"],
                    req["num_parts"],
                    req["part_idx"],
                    None,
                    modality,
                    error_msg=msg,
                    error_code=code,
                )
        return [(0, 0, 0, msg, code)] * len(requests)

    # For zmq_to_tokenizer zmq_to_scheduler and mooncake
    async def send(
        self, req_id, prefill_host, embedding_port, session_id=None, buffer_address=None
    ):
        mm_data: EmbeddingData = self.embedding_to_send[req_id]
        await self._send(
            mm_data.embedding,
            mm_data,
            session_id=session_id,
            buffer_address=buffer_address,
            prefill_host=prefill_host,
            embedding_port=embedding_port,
        )

    # For zmq_to_scheduler
    async def send_with_url(
        self,
        req_id,
    ):
        mm_data = self.embedding_to_send.get(req_id)
        if not mm_data:
            return
        sent_urls: Set[str] = set()
        all_tasks: List[Tuple[asyncio.Task, str]] = []
        start_time = asyncio.get_running_loop().time()
        timeout = self.send_timeout
        cond = await get_condition(req_id)

        try:
            while True:
                async with rid_lock:
                    current_targets = rid_to_receive_endpoint.get(req_id, set()).copy()
                    expected_count = rid_to_receive_count.get(req_id)

                new_targets = current_targets - sent_urls

                if new_targets:
                    logger.info(
                        f"Found {len(new_targets)} new endpoints for {req_id}. Starting tasks..."
                    )
                    for url in new_targets:
                        task = asyncio.create_task(
                            self._send(
                                mm_data.embedding,
                                mm_data,
                                url=url,
                            )
                        )
                        all_tasks.append((task, url))
                        sent_urls.add(url)  # Mark as handled immediately
                if expected_count is not None and len(sent_urls) >= expected_count:
                    logger.info(
                        f"All {expected_count} endpoints initiated for {req_id}. Breaking loop."
                    )
                    break
                remaining = timeout - (asyncio.get_running_loop().time() - start_time)
                if remaining <= 0:
                    logger.error(
                        f"[{req_id}] Timeout! Sent {len(sent_urls)}/{expected_count}"
                    )
                    break

                async with cond:
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue

            if all_tasks:
                logger.info(
                    f"Loop finished. Awaiting completion of {len(all_tasks)} sending tasks..."
                )
                tasks_only = [t[0] for t in all_tasks]
                results = await asyncio.gather(*tasks_only, return_exceptions=True)

                # Process results and log errors
                for i, result in enumerate(results):
                    url = all_tasks[i][1]  # Retrieve URL associated with the task
                    if isinstance(result, Exception):
                        logger.error(f"Failed to send to {url}: {result}")
                    else:
                        logger.debug(f"Successfully sent to {url}")

            logger.info(f"All tasks completed for req_id: {req_id}")

        finally:
            logger.info(f"Cleaning up resources for req_id {req_id}")
            async with rid_lock:
                rid_to_receive_endpoint.pop(req_id, None)
                rid_to_receive_count.pop(req_id, None)
            async with cond_dict_lock:
                rid_to_cond.pop(req_id, None)
            self.embedding_to_send.pop(req_id, None)

    async def get_embedding_port(self, prefill_url):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800)
        ) as session:
            response = await session.post(
                f"{prefill_url}/embedding_bootstrap",
                json={"embedding_port": None},
            )
            response_json = await response.json()
            return response_json["embedding_port"]


class EncoderProfiler:
    def __init__(self, rank: int):
        self.rank = rank
        self.profiler = None
        self.steps_left = None
        self.output_dir = None
        self.prefix = None
        self.profile_id = None

    def start(self, obj: ProfileReq):
        if self.profiler is not None:
            return False, "profiling already running"

        output_dir = obj.output_dir or os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.prefix = obj.profile_prefix or "encoder"
        self.profile_id = str(time.time())

        activities = obj.activities or ["CPU", "GPU"]
        torch_activities = []
        if "CPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in activities:
            torch_activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_memory = "MEM" in activities
        if not torch_activities and not profile_memory:
            return False, "no supported activities"

        self.profiler = torch.profiler.profile(
            activities=torch_activities,
            with_stack=True if obj.with_stack is None else obj.with_stack,
            record_shapes=False if obj.record_shapes is None else obj.record_shapes,
            profile_memory=profile_memory,
        )
        self.profiler.start()
        self.steps_left = obj.num_steps
        logger.info(
            f"Encoder profiling started. output_dir={self.output_dir} profile_id={self.profile_id}"
        )
        return True, None

    def step(self):
        if self.profiler is None:
            return
        self.profiler.step()
        if self.steps_left is not None:
            self.steps_left -= 1
            if self.steps_left <= 0:
                self.stop()

    def stop(self):
        if self.profiler is None:
            return False, "profiling not running"
        self.profiler.stop()
        filename = f"{self.prefix}-rank{self.rank}-{self.profile_id}.trace.json"
        trace_path = os.path.join(self.output_dir, filename)
        self.profiler.export_chrome_trace(trace_path)
        logger.info("Encoder profiling saved to: %s", trace_path)
        self.profiler = None
        self.steps_left = None
        return True, None


class PendingRequest:
    __slots__ = ("request", "future", "submit_time")

    def __init__(self, request: dict, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.time()


# VIDEO excluded: per-video preprocess kwargs (do_sample_frames, video_metadata)
# vary per request and can't merge into one HF processor call.
_BATCHABLE_MODALITIES = {Modality.IMAGE, Modality.AUDIO}


class EncoderScheduler:
    """Aggregate concurrent /encode requests into bounded image/audio batches."""

    def __init__(
        self,
        encoder: "MMEncoder",
        send_sockets: List[zmq.Socket],
        max_batch_size: int,
        request_timeout: float = ENCODER_REQ_TIMEOUT,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.max_batch_size = max(1, int(max_batch_size))
        self.request_timeout = max(1.0, float(request_timeout))
        self.pending_queue: "asyncio.Queue[PendingRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info(
                f"EncoderScheduler started with max_batch_size={self.max_batch_size}"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        # Reject any requests still queued so their HTTP handlers don't hang.
        while True:
            try:
                pending = self.pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))

    async def submit(self, request: dict) -> Tuple:
        pending = PendingRequest(request, asyncio.get_running_loop())
        await self.pending_queue.put(pending)
        try:
            return await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not pending.future.done():
                pending.future.cancel()
            req_id = request.get("req_id")
            logger.error(
                f"EncoderScheduler.submit timed out after {self.request_timeout}s "
                f"for req_id={req_id}"
            )
            raise

    async def _collect_batch(self) -> List[PendingRequest]:
        batch = [await self.pending_queue.get()]
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect_batch()
                groups: Dict[Modality, List[PendingRequest]] = defaultdict(list)
                for p in batch:
                    groups[
                        Modality.from_str(p.request.get("modality", "image"))
                    ].append(p)
                for modality, group in groups.items():
                    await self._dispatch_group(group, modality)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as e:
                logger.error(
                    f"Error in EncoderScheduler batch worker: {e}", exc_info=True
                )
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

    @staticmethod
    def _validate_request_shape(req: dict) -> Optional[str]:
        # Cheap pre-broadcast checks: shape errors that don't require running
        # the HF processor. Once a request reaches TP workers they enter
        # batch_encode and expect to join its collectives — a malformed batch
        # that makes rank-0 bail mid-flight would deadlock the workers.
        if not isinstance(req, dict):
            return f"request is not a dict: {type(req).__name__}"
        if not req.get("req_id"):
            return "missing req_id"
        if not req.get("mm_items"):
            return "missing or empty mm_items"
        if "num_parts" not in req or "part_idx" not in req:
            return "missing num_parts / part_idx"
        h = req.get("hashes")
        if h is not None and not isinstance(h, (list, tuple, str, int, bytes)):
            return f"hashes must be list/scalar, got {type(h).__name__}"
        return None

    async def _dispatch_group(
        self, group: List[PendingRequest], modality: Modality
    ) -> None:
        # Video can't fuse (per-video preprocess kwargs vary).
        if modality not in _BATCHABLE_MODALITIES:
            await self._dispatch_per_request(group, modality)
            return

        # Drop structurally-bad requests before broadcasting; otherwise TP
        # workers would join batch_encode collectives that rank-0 has already
        # abandoned.
        valid: List[PendingRequest] = []
        for p in group:
            err = self._validate_request_shape(p.request)
            if err is None:
                valid.append(p)
                continue
            logger.error(f"Dropping req_id={p.request.get('req_id')} from batch: {err}")
            if not p.future.done():
                p.future.set_exception(BadRequestError(err))
        if not valid:
            return
        group = valid

        requests = [p.request for p in group]
        start = time.time()
        modality_str = modality.name.lower()
        # Non-business requests (log_metrics=False) are withheld from metrics.
        batch_log_metrics = all(p.request.get("log_metrics", True) for p in group)
        if encoder_metrics_collector is not None:
            for p in group:
                if p.request.get("log_metrics", True):
                    encoder_metrics_collector.observe_queue_wait(
                        max(0.0, start - p.submit_time), modality=modality_str
                    )
        for sock in self.send_sockets:
            sock.send_pyobj(
                {
                    "type": "batch_encode",
                    "modality": modality.name,
                    "requests": requests,
                    "enter_time": start,
                }
            )

        logger.info(f"Dispatching batch of {len(group)} {modality.name} requests")

        try:
            batch_timing: dict = {}
            token = _batch_timing_var.set(batch_timing)
            try:
                results = await self.encoder.batch_encode(
                    requests, modality, log_metrics=batch_log_metrics
                )
            finally:
                _batch_timing_var.reset(token)
            now = time.time()
            if encoder_metrics_collector is not None:
                if batch_log_metrics:
                    encoder_metrics_collector.observe_batch_e2e_latency(
                        max(0.0, now - start), modality=modality_str
                    )
                for p in group:
                    if p.request.get("log_metrics", True):
                        encoder_metrics_collector.observe_request_e2e(
                            max(0.0, now - p.submit_time), modality=modality_str
                        )
            for p in group:
                _emit_encode_trace(
                    p.request,
                    submit_time=p.submit_time,
                    dispatch_start=start,
                    finish_time=now,
                    batch_timing=batch_timing,
                    modality_str=modality_str,
                    batch_size=len(group),
                )
            if len(group) > 1:
                logger.info(
                    f"Batch of {len(group)} {modality.name} requests completed in "
                    f"{(time.time() - start) * 1000:.1f}ms"
                )
        except Exception as e:
            # batch_encode normally catches and returns errors via _batch_set_error.
            # If it raised, rank-0 may have skipped a collective broadcast, leaving
            # TP workers stuck. Don't try to recover — fail every pending future
            # and let the client retry. Re-broadcasting would risk a deadlock.
            logger.error(f"batch_encode raised: {e}", exc_info=True)
            for p in group:
                if not p.future.done():
                    p.future.set_exception(e)
            return

        if len(results) != len(group):
            err = RuntimeError(
                f"batch_encode returned {len(results)} results for {len(group)} requests"
            )
            logger.error(str(err))
            for p in group:
                if not p.future.done():
                    p.future.set_exception(err)
            return

        for p, result in zip(group, results):
            if not p.future.done():
                p.future.set_result(result)

    async def _dispatch_per_request(
        self,
        group: List[PendingRequest],
        modality: Modality,
    ) -> None:
        for p in group:
            try:
                p.request["enter_time"] = p.submit_time
                for sock in self.send_sockets:
                    sock.send_pyobj(p.request)
                result = await _encode_request_direct(self.encoder, p.request, modality)
                if not p.future.done():
                    p.future.set_result(result)
            except Exception as e:
                logger.error(
                    f"Per-request encode failed for req_id={p.request.get('req_id')}: {e}"
                )
                if not p.future.done():
                    p.future.set_exception(e)


encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []
encoder_scheduler: Optional[EncoderScheduler] = None

# Per-process encoder metrics collector. Set in launch_server (non-DP) and in
# run_dp_worker (DP mode, with the worker's dp_rank). None when metrics disabled.
encoder_metrics_collector: Optional[EncoderMetricsCollector] = None

# DP mode (--dp-size > 1): each rank runs as a subprocess with its own
# MMEncoder on its own GPU; the main process only routes via ZMQ so the
# asyncio event loop is never blocked by GPU work.
dp_dispatcher: Optional["DPDispatcher"] = None


def _observe_per_request(
    observe_fn, batch_seconds: float, num_requests: int, modality: str
) -> None:
    """Emit a batch stage's wall-clock once per request (not summed)."""
    if observe_fn is None or num_requests <= 0:
        return
    for _ in range(num_requests):
        observe_fn(batch_seconds, modality=modality)


def _observe_single_request_load(mc, stage_timing, modality_str: str) -> None:
    """Emit download/decode/preprocess for a single request (sum all items)."""
    if mc is None or not stage_timing:
        return
    load_timings = stage_timing.get("load_timings") or []
    dl = sum((t.get("download", 0.0) for t in load_timings), 0.0)
    dec = sum((t.get("decode", 0.0) for t in load_timings), 0.0)
    if dl > 0.0:
        mc.observe_download(dl, modality=modality_str)
    if dec > 0.0:
        mc.observe_decode(dec, modality=modality_str)
    mc.observe_preprocess(stage_timing.get("preprocess", 0.0), modality=modality_str)


def _observe_download_decode_per_request(
    mc, load_timings, items_per_req, modality_str: str, miss_indices=None
) -> None:
    """Sum each request's leaf-item download/decode."""
    if mc is None or not load_timings:
        return
    if miss_indices is not None:
        miss_pos = {gidx: j for j, gidx in enumerate(miss_indices)}
    else:
        miss_pos = {i: i for i in range(len(load_timings))}
    offset = 0
    for n in items_per_req:
        idxs = [
            miss_pos[i] for i in range(offset, offset + n) if i in miss_pos
        ]
        dl = sum((load_timings[j].get("download", 0.0) for j in idxs), 0.0)
        dec = sum((load_timings[j].get("decode", 0.0) for j in idxs), 0.0)
        offset += n
        if dl <= 0.0 and dec <= 0.0:
            continue
        if dl > 0.0:
            mc.observe_download(dl, modality=modality_str)
        if dec > 0.0:
            mc.observe_decode(dec, modality=modality_str)


_BATCH_SPAN_KEYS = (
    ("mm_load", "load_start_ts", "load_end_ts"),
    ("preprocess", "preprocess_start_ts", "preprocess_end_ts"),
    ("model_forward", "forward_start_ts", "forward_end_ts"),
)


def _emit_encode_trace(
    request: dict,
    submit_time: float,
    dispatch_start: float,
    finish_time: float,
    batch_timing: Optional[dict],
    modality_str: str,
    batch_size: int,
) -> None:
    if not get_global_tracing_enabled() or not request.get("trace_context"):
        return
    try:
        req_id = str(request.get("req_id", ""))
        ctx = TraceReqContext(
            rid=req_id.split("_local_part_")[0],
            role="encode",
            module_name="encoder",
            external_trace_header=request["trace_context"],
        )
        if not ctx.is_tracing_enabled():
            return

        def emit(name, start_ns, end_ns, attrs=None):
            if end_ns > start_ns:
                ctx.trace_slice(
                    TraceSliceContext(name, start_ns, end_ns, level=1, attrs=attrs)
                )

        ctx.trace_req_start(int(submit_time * 1e9))
        ctx.trace_set_root_attrs({"rid": req_id, "modality": modality_str})
        emit("queue_wait", int(submit_time * 1e9), int(dispatch_start * 1e9))

        if batch_timing:
            attrs = {
                "batch_size": batch_size,
                "num_miss_items": batch_timing.get("num_miss_items", 0),
                "num_total_items": batch_timing.get("num_total_items", 0),
            }
            loads = batch_timing.get("load_timings") or []
            load_attrs = {
                **attrs,
                "download_seconds_sum": sum(t.get("download", 0.0) for t in loads),
                "decode_seconds_sum": sum(t.get("decode", 0.0) for t in loads),
            }
            for name, start_key, end_key in _BATCH_SPAN_KEYS:
                start_ts = batch_timing.get(start_key)
                end_ts = batch_timing.get(end_key)
                if start_ts is not None and end_ts is not None:
                    emit(
                        name,
                        convert_time_to_realtime_ns(start_ts),
                        convert_time_to_realtime_ns(end_ts),
                        load_attrs if name == "mm_load" else attrs,
                    )

        ctx.trace_req_finish(int(finish_time * 1e9))
    except Exception:
        logger.warning("encoder trace emit failed", exc_info=True)


def _stamp_encoder_e2e(enc: MMEncoder, request: dict) -> None:
    mm_data = enc.embedding_to_send.get(request["req_id"])
    if mm_data is not None:
        mm_data.encoder_e2e_latency = time.time() - request["enter_time"]


async def _encode_request_direct(enc: MMEncoder, request: dict, modality: Modality):
    """Non-batchable (video) path: same per-request metrics as the dispatcher."""
    modality_str = modality.name.lower()
    mc = encoder_metrics_collector if request.get("log_metrics", True) else None
    enter = request.get("enter_time") or time.time()
    start = time.time()
    if mc is not None:
        mc.observe_queue_wait(max(0.0, start - enter), modality=modality_str)
        mm_items = request.get("mm_items", [])
        mm_count = len(mm_items) if isinstance(mm_items, (list, tuple)) else 1
        mc.observe_mm_items_per_request(mm_count, modality=modality_str)
        mc.observe_mm_items_per_batch(mm_count, modality=modality_str)
    batch_timing: dict = {}
    token = _batch_timing_var.set(batch_timing)
    try:
        result = await enc.encode_request(request, modality)
    finally:
        _batch_timing_var.reset(token)
    now = time.time()
    if mc is not None:
        mc.observe_request_e2e(max(0.0, now - enter), modality=modality_str)
        mc.observe_batch_e2e_latency(max(0.0, now - start), modality=modality_str)
    _emit_encode_trace(
        request,
        submit_time=enter,
        dispatch_start=start,
        finish_time=now,
        batch_timing=batch_timing,
        modality_str=modality_str,
        batch_size=1,
    )
    return result


async def _push_embedding_to_prefill(enc: MMEncoder, request: dict) -> None:
    # No-op for mooncake (its /send is separate). embedding_port=None is
    # rejected upfront, so ports is always a concrete list here.
    req_id = request["req_id"]
    backend = enc.server_args.encoder_transfer_backend

    if backend == "zmq_to_tokenizer":
        await enc.send(
            req_id=req_id,
            prefill_host=request["prefill_host"],
            embedding_port=request["embedding_port"],
        )
        enc.embedding_to_send.pop(req_id, None)
        return

    if backend == "zmq_to_scheduler":
        ports = request["embedding_port"]
        assert isinstance(ports, list)
        await asyncio.gather(
            *(
                enc.send(
                    req_id=req_id,
                    prefill_host=request["prefill_host"],
                    embedding_port=p,
                )
                for p in ports
            )
        )
        enc.embedding_to_send.pop(req_id, None)


async def _dp_worker_encode_and_send(
    enc: MMEncoder,
    sched: Optional[EncoderScheduler],
    request: dict,
) -> Optional[dict]:
    # Mooncake returns metadata for main to forward; zmq inlines the send.
    # Soft errors raise MMError so the dispatcher route maps them to HTTP.
    req_id = request["req_id"]
    request["enter_time"] = time.time()
    modality = Modality.from_str(request["modality"])
    backend = enc.server_args.encoder_transfer_backend

    # URL state lives in main process module globals; workers don't see it.
    if backend == "zmq_to_scheduler" and request.get("embedding_port") is None:
        raise MMError(
            "Encoder DP mode does not support zmq_to_scheduler with "
            "embedding_port=None (URL state isn't synchronised to workers). "
            "Provide an explicit embedding_port list, switch to mooncake / "
            "zmq_to_tokenizer, or run without --dp-size.",
            code=HTTPStatus.BAD_REQUEST,
        )

    encode_coro = (
        sched.submit(request)
        if sched is not None and modality in _BATCHABLE_MODALITIES
        else _encode_request_direct(enc, request, modality)
    )
    nbytes, embedding_len, embedding_dim, error_msg, error_code = await encode_coro

    if not error_msg:
        _stamp_encoder_e2e(enc, request)

    if error_msg:
        # zmq backends still forward an error EmbeddingData to P so it
        # doesn't block; send failures here are swallowed.
        try:
            await _push_embedding_to_prefill(enc, request)
        except Exception as e:
            logger.error(
                f"DP error-send failed for req_id={req_id}: {e}", exc_info=True
            )
        # Free the error EmbeddingData stored during encode, or it leaks in
        # embedding_to_send and pins /health into "busy" (a non-empty
        # embedding_to_send reads as busy, skipping the probe). Neither path
        # guarantees cleanup on its own: mooncake's _push_embedding_to_prefill
        # is a no-op, and a swallowed zmq send failure above skips its own pop.
        # zmq lacks the inflight attrs so _cleanup_inflight_encode_state would
        # early-return on it — pop directly. Mirrors the non-DP error path.
        if backend == "mooncake":
            await enc._cleanup_inflight_encode_state(req_id)
        else:
            enc.embedding_to_send.pop(req_id, None)
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)

    if backend == "mooncake":
        request.pop("mm_items", None)
        request.update(
            embedding_size=nbytes,
            embedding_len=embedding_len,
            embedding_dim=embedding_dim,
        )
        # Free the held embedding if the follow-up /send never arrives (same
        # send_timeout cleanup the non-DP path uses).
        enc._schedule_inflight_encode_cleanup(req_id)
        return request

    await _push_embedding_to_prefill(enc, request)
    return None


async def _dp_worker_health_encode(enc: MMEncoder) -> None:
    """Functional health probe run on a DP worker.

    Process-liveness (proc.sentinel) can't see a worker that's alive but
    wedged — hung GPU, NCCL deadlock, stalled ZMQ, or a blocked event loop.
    When idle, run a tiny dummy encode to exercise the VIT forward and surface
    those stalls. No prefill destination: the embedding is discarded, mirroring
    the non-DP /health path. Raises on encode failure so the worker envelope
    carries ``_error`` back to the dispatcher.
    """
    # Busy worker: in-flight traffic already proves liveness, so skip the probe
    # and report healthy — same `embedding_to_send` signal the non-DP /health
    # path uses. A wedged-but-busy worker never reaches here (it can't service
    # the recv), so the dispatcher's broadcast still times out → 503.
    if enc.embedding_to_send:
        return None

    if enc.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif enc.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor → can't functionally probe; liveness alone is healthy.
        return None

    req_id = f"{HEALTH_CHECK_RID_PREFIX}_{time.time()}"
    try:
        _, _, _, error_msg, error_code = await enc.encode(
            mm_items=mm_items,
            modality=modality,
            req_id=req_id,
            num_parts=1,
            part_idx=0,
            log_metrics=False,
        )
    finally:
        # Never leave the dummy embedding sitting in the send map.
        enc.embedding_to_send.pop(req_id, None)

    if error_msg:
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)


class DPDispatcher:
    """Routes encode requests across DP ranks by least modality-weighted load."""

    def __init__(
        self,
        dp_size: int,
        dispatch_sockets: List,
        result_socket,
        worker_processes: List[mp.Process],
        enable_metrics: bool = False,
    ):
        self.dp_size = dp_size
        self.dispatch_sockets = dispatch_sockets
        self.result_socket = result_socket
        self.worker_processes = worker_processes
        # Key = req_id for encode/broadcast, req_id + "_send" for mooncake /send.
        self.pending_futures: List[Dict[str, asyncio.Future]] = [
            {} for _ in range(dp_size)
        ]
        self.req_id_to_rank: Dict[str, int] = {}
        # Only encode requests are tracked here; send/broadcast skip it.
        self.pending_items: List[Dict[str, dict]] = [{} for _ in range(dp_size)]
        self._rr_counter = 0
        self._broadcast_counter = 0
        self._dead_ranks: Set[int] = set()
        # req_id -> monotonic ts a mooncake mapping has waited for its /send.
        self._pending_send_at: Dict[str, float] = {}
        # Set when _result_listener gives up; makes alive_ranks report empty.
        self._listener_failed = False

        # Prometheus gauge: pending requests per DP rank.
        self.pending_gauge = None
        self.pending_weight_gauge = None
        if enable_metrics:
            from prometheus_client import Gauge

            self.pending_gauge = Gauge(
                name="sglang:encoder_dp_pending_requests",
                documentation="Number of pending requests per encoder DP rank.",
                labelnames=["dp_rank"],
                multiprocess_mode="mostrecent",
            )
            self.pending_weight_gauge = Gauge(
                name="sglang:encoder_dp_pending_weight",
                documentation=(
                    "Accumulated modality-weighted in-flight encode load per "
                    "encoder DP rank (image=1, audio=20, video=50)."
                ),
                labelnames=["dp_rank"],
                multiprocess_mode="mostrecent",
            )

    @property
    def pending_counts(self) -> List[int]:
        return [len(d) for d in self.pending_futures]

    @staticmethod
    def _weight_for(modality: str) -> int:
        return MODALITY_WEIGHTS.get(modality, DEFAULT_MODALITY_WEIGHT)

    @staticmethod
    def _mm_count(request: dict) -> int:
        mm_items = request.get("mm_items", [])
        if isinstance(mm_items, (list, tuple)):
            return max(1, len(mm_items))
        return 1

    @property
    def pending_weights(self) -> List[int]:
        return [
            sum(
                self._weight_for(item["modality"]) * item["count"]
                for item in rank_items.values()
            )
            for rank_items in self.pending_items
        ]

    def _update_pending_gauge(self) -> None:
        """Push current pending counts to the Prometheus gauge (absolute set)."""
        if self.pending_gauge is not None:
            for i, c in enumerate(self.pending_counts):
                self.pending_gauge.labels(dp_rank=str(i)).set(c)
        if self.pending_weight_gauge is not None:
            for i, w in enumerate(self.pending_weights):
                self.pending_weight_gauge.labels(dp_rank=str(i)).set(w)

    @property
    def alive_ranks(self) -> List[int]:
        # Empty if the result listener died; else ranks not marked dead.
        if self._listener_failed:
            return []
        return [r for r in range(self.dp_size) if r not in self._dead_ranks]

    @property
    def all_ranks_alive(self) -> bool:
        # Strict health (only /health uses this); routing still degrades.
        return len(self.alive_ranks) == self.dp_size

    def start(self) -> None:
        logger.info(f"DP dispatcher started: {self.dp_size} ranks (all remote)")
        asyncio.create_task(self._result_listener())
        asyncio.create_task(self._worker_watchdog())
        asyncio.create_task(self._cleanup_stale_mappings())

    def _drop_pending_and_mapping(self, rank: int, req_id: str) -> None:
        # dispatch / broadcast failure: no follow-up /send expected.
        self.pending_futures[rank].pop(req_id, None)
        self.pending_items[rank].pop(req_id, None)
        self.req_id_to_rank.pop(req_id, None)
        self._update_pending_gauge()

    def _fail_pending_for_rank(self, rank: int, reason: str, error_type: str) -> None:
        # Resolve a rank's outstanding futures with 503 so awaiters don't hang.
        pending = self.pending_futures[rank]
        for key, future in list(pending.items()):
            if not future.done():
                future.set_result(
                    {
                        "req_id": key.removesuffix("_send"),
                        "_dp_type": "send" if key.endswith("_send") else "encode",
                        "content": None,
                        "_error": reason,
                        "_error_type": error_type,
                        "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
                    }
                )
            pending.pop(key, None)
        self.pending_items[rank].clear()
        self._update_pending_gauge()

    def _fail_all_pending(self, reason: str, error_type: str) -> None:
        for rank in range(self.dp_size):
            self._fail_pending_for_rank(rank, reason, error_type)
        self.req_id_to_rank.clear()
        self._pending_send_at.clear()

    @staticmethod
    def _timeout_envelope(req_id: str, dp_type: str, reason: str) -> dict:
        return {
            "req_id": req_id,
            "_dp_type": dp_type,
            "content": None,
            "_error": reason,
            "_error_type": "TimeoutError",
            "_error_code": int(HTTPStatus.GATEWAY_TIMEOUT),
        }

    async def dispatch(self, request: dict) -> dict:
        # Skip ranks whose worker process has died.
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        # Pick the rank with the least modality-weighted pending load
        # (video/audio items cost far more encode time than images).
        modality = str(request.get("modality", "image")).lower()
        count = self._mm_count(request)
        weights = self.pending_weights
        min_w = min(weights[r] for r in alive_ranks)
        candidates = [r for r in alive_ranks if weights[r] == min_w]
        rank = candidates[self._rr_counter % len(candidates)]
        self._rr_counter += 1
        req_id = request["req_id"]
        self.req_id_to_rank[req_id] = rank
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][req_id] = future
        self.pending_items[rank][req_id] = {"modality": modality, "count": count}
        self._update_pending_gauge()
        logger.info(
            f"MM-Encoder DP dispatch: req_id={req_id}, "
            f"modality={modality}, count={count}, dp_rank={rank}, "
            f"pending_weights={self.pending_weights}, pending={self.pending_counts}"
        )

        try:
            await self.dispatch_sockets[rank].send_pyobj(request)
            # An alive-but-stuck worker (NCCL deadlock etc.) wouldn't trip
            # the watchdog, so bound the wait explicitly.
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self._drop_pending_and_mapping(rank, req_id)
            return self._timeout_envelope(
                req_id,
                "encode",
                f"Encoder DP rank={rank} timed out after {ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self._drop_pending_and_mapping(rank, req_id)
            raise

    async def dispatch_send(self, request: dict) -> dict:
        req_id = request["req_id"]
        # /send arrived → stop tracking it for stale-mapping GC.
        self._pending_send_at.pop(req_id, None)
        if self._listener_failed:
            return {
                "req_id": req_id,
                "_error": "encoder DP result listener stopped; cannot route /send",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        rank = self.req_id_to_rank.get(req_id)
        if rank is None:
            logger.warning(
                f"MM-Encoder dispatch_send: unknown req_id={req_id}, "
                f"cannot route to worker"
            )
            return {"req_id": req_id, "_error": f"Unknown req_id: {req_id}"}
        if rank in self._dead_ranks:
            # Worker died between encode and /send; embedding is gone.
            self.req_id_to_rank.pop(req_id, None)
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before /send for req_id={req_id}",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        key = req_id + "_send"
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        request["_dp_type"] = "send"
        logger.info(
            f"MM-Encoder DP dispatch_send: req_id={req_id}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )
        try:
            await self.dispatch_sockets[rank].send_pyobj(request)
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            return self._timeout_envelope(
                req_id,
                "send",
                f"Encoder DP rank={rank} /send timed out after {ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            raise

    async def broadcast(
        self, request: dict, timeout: Optional[float] = None
    ) -> List[dict]:
        # Skip dead ranks: a PUSH to a gone worker would just buffer and then
        # surface as a spurious per-rank timeout. All dead → 503 (same as
        # dispatch), which the profile endpoints turn into an HTTP error.
        eff_timeout = timeout if timeout is not None else ENCODER_REQ_TIMEOUT
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        batch_id = self._broadcast_counter
        self._broadcast_counter += 1
        rank_keys: List[Tuple[int, str]] = []
        futures: List[asyncio.Future] = []
        dp_type = request.get("_dp_type", "unknown")
        try:
            for rank in alive_ranks:
                req_id = f"_broadcast_{batch_id}_{rank}"
                future = asyncio.get_running_loop().create_future()
                self.pending_futures[rank][req_id] = future
                self.req_id_to_rank[req_id] = rank
                rank_keys.append((rank, req_id))
                request_copy = {**request, "req_id": req_id}
                await self.dispatch_sockets[rank].send_pyobj(request_copy)
                futures.append(future)
            # Concurrent wait → total bounded by eff_timeout, not
            # dp_size × eff_timeout.
            outcomes = await asyncio.gather(
                *(asyncio.wait_for(fut, timeout=eff_timeout) for fut in futures),
                return_exceptions=True,
            )
            results: List[dict] = []
            for (rank, req_id), outcome in zip(rank_keys, outcomes):
                if isinstance(outcome, asyncio.TimeoutError):
                    self._drop_pending_and_mapping(rank, req_id)
                    results.append(
                        self._timeout_envelope(
                            req_id,
                            dp_type,
                            f"Encoder DP rank={rank} broadcast timed out "
                            f"after {eff_timeout}s",
                        )
                    )
                elif isinstance(outcome, BaseException):
                    self._drop_pending_and_mapping(rank, req_id)
                    raise outcome
                else:
                    results.append(outcome)
            return results
        except BaseException:
            for rank, req_id in rank_keys:
                self._drop_pending_and_mapping(rank, req_id)
            raise

    async def _worker_watchdog(self) -> None:
        # proc.sentinel becomes readable on process exit; fail this rank's
        # pending futures so awaiters don't hang on a dead worker.
        loop = asyncio.get_running_loop()
        watch: Dict[int, asyncio.Future] = {}
        for rank, proc in enumerate(self.worker_processes):
            fut: asyncio.Future = loop.create_future()

            # add_reader is level-triggered, so remove_reader inside the
            # callback to avoid spinning every loop iteration.
            def _on_exit(r=rank, f=fut, p=proc, lp=loop):
                try:
                    lp.remove_reader(p.sentinel)
                except (ValueError, OSError):
                    pass
                if not f.done():
                    f.set_result(r)

            try:
                loop.add_reader(proc.sentinel, _on_exit)
            except (ValueError, OSError):
                continue
            watch[rank] = fut

        while watch:
            done, _ = await asyncio.wait(
                watch.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                rank = fut.result()
                proc = self.worker_processes[rank]
                logger.error(
                    f"DP worker rank={rank} (pid={proc.pid}) exited "
                    f"with code={proc.exitcode}; failing pending requests"
                )
                self._dead_ranks.add(rank)
                reason = f"DP worker rank={rank} died (exitcode={proc.exitcode})"
                self._fail_pending_for_rank(rank, reason, "WorkerDied")
                dead_rids = [r for r, rk in self.req_id_to_rank.items() if rk == rank]
                for rid in dead_rids:
                    self._pending_send_at.pop(rid, None)
                self.req_id_to_rank = {
                    r: rk for r, rk in self.req_id_to_rank.items() if rk != rank
                }
                watch.pop(rank, None)

    async def _result_listener(self) -> None:
        # Bounded back-off + give-up so a torn-down context exits in ~3s
        # rather than spinning forever on recv errors.
        consecutive_errors = 0
        while True:
            try:
                msg = await self.result_socket.recv_pyobj()
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                logger.error("_result_listener recv error", exc_info=True)
                if consecutive_errors >= 30:
                    logger.error(
                        "_result_listener giving up after 30 consecutive errors"
                    )
                    self._listener_failed = True
                    self._fail_all_pending(
                        "encoder DP result listener stopped after repeated "
                        "recv errors",
                        "ResultListenerStopped",
                    )
                    return
                await asyncio.sleep(min(0.1 * consecutive_errors, 1.0))
                continue
            req_id = msg.get("req_id", "")
            dp_type = msg.get("_dp_type", "encode")
            key = (req_id + "_send") if dp_type == "send" else req_id
            rank = self.req_id_to_rank.get(req_id)
            if rank is None or key not in self.pending_futures[rank]:
                logger.warning(
                    f"_result_listener: no pending future for "
                    f"req_id={req_id}, dp_type={dp_type}, dropping"
                )
                continue
            future = self.pending_futures[rank].pop(key)
            self.pending_items[rank].pop(key, None)
            self._update_pending_gauge()
            # Only mooncake encode (content=request dict) needs the mapping
            # kept for the follow-up /send.
            keep_mapping = dp_type == "encode" and msg.get("content") is not None
            if keep_mapping:
                self._pending_send_at[req_id] = time.monotonic()
            else:
                self.req_id_to_rank.pop(req_id, None)
            try:
                future.set_result(msg)

            except asyncio.InvalidStateError:
                logger.warning(
                    f"_result_listener: future already done for "
                    f"req_id={req_id}, dp_type={dp_type}"
                )

    async def _cleanup_stale_mappings(self) -> None:
        # Evict req_id->rank mappings whose /send never came. The worker frees
        # its own embedding via the send_timeout cleanup scheduled at encode,
        # so both sides key off the same timeout.
        ttl = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        interval = max(ttl / 4, 30)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale = [rid for rid, ts in self._pending_send_at.items() if now - ts > ttl]
            for rid in stale:
                self._pending_send_at.pop(rid, None)
                self.req_id_to_rank.pop(rid, None)
            if stale:
                logger.warning(
                    f"Evicted {len(stale)} stale encoder DP /send mapping(s) "
                    f"with no /send within {ttl}s"
                )


async def _dp_worker_handle_profile(
    enc: MMEncoder, dp_rank: int, dp_type: str, request: dict
) -> dict:
    prefix = f"dp_rank={dp_rank}: "
    if dp_type == "start_profile":
        obj = request.get("profile_req")
        # `is None` (not `if not obj`) so empty dict still raises.
        req = (
            ProfileReq(**obj)
            if obj is not None
            else ProfileReq(ProfileReqType.START_PROFILE)
        )
        if enc.profiler is None:
            enc.profiler = EncoderProfiler(dp_rank)
        ok, msg = enc.profiler.start(req)
        detail = (
            f"started profiling, output_dir={enc.profiler.output_dir}" if ok else msg
        )
    else:  # stop_profile
        if enc.profiler is None:
            return {"ok": False, "msg": prefix + "profiling not initialized"}
        ok, msg = enc.profiler.stop()
        detail = "stopped profiling" if ok else msg
    return {"ok": ok, "msg": prefix + detail}


async def _dp_worker_handle_request(
    enc: MMEncoder,
    sched: EncoderScheduler,
    send_sock,
    send_lock: asyncio.Lock,
    dp_rank: int,
    request: dict,
    dp_type: str,
) -> None:
    t0 = time.time()
    modality_str = str(request.get("modality", "image")).lower()
    is_encode = dp_type not in (
        "start_profile",
        "stop_profile",
        "health_encode",
        "send",
    )
    log_metrics = request.get("log_metrics", True)
    count_metrics = is_encode and log_metrics and encoder_metrics_collector is not None
    if count_metrics:
        encoder_metrics_collector.inc_requests_received(
            dp_rank=dp_rank, modality=modality_str
        )
    try:
        if dp_type in ("start_profile", "stop_profile"):
            content = await _dp_worker_handle_profile(enc, dp_rank, dp_type, request)
        elif dp_type == "health_encode":
            content = await _dp_worker_health_encode(enc)
        elif dp_type == "send":
            req_id = request["req_id"]
            await enc.send(
                req_id=req_id,
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
                session_id=request["session_id"],
                buffer_address=request["buffer_address"],
            )
            # cancels the scheduled cleanup + frees embedding/forward state
            await enc._cleanup_inflight_encode_state(req_id)
            content = None
        else:
            content = await _dp_worker_encode_and_send(enc, sched, request)

        logger.info(
            f"MM-Encoder [dp_rank={dp_rank}] {dp_type} done: "
            f"req_id={request.get('req_id', '?')}, "
            f"modality={request.get('modality', 'image')}, "
            f"cost={(time.time() - t0) * 1000:.1f}ms"
        )
        if count_metrics:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="success"
            )
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": content,
        }
    except Exception as e:
        logger.error(
            f"DP worker {dp_rank} error on {dp_type} "
            f"req_id={request.get('req_id', '?')}: {e}",
            exc_info=True,
        )
        if count_metrics:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
        err_code = int(getattr(e, "code", None) or HTTPStatus.INTERNAL_SERVER_ERROR)
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": None,
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_error_code": err_code,
        }

    # pyzmq async send_pyobj isn't safe for concurrent senders.
    try:
        async with send_lock:
            await send_sock.send_pyobj(envelope)
    except Exception:
        logger.error(
            f"DP worker {dp_rank} failed to send envelope for "
            f"req_id={request.get('req_id', '?')}",
            exc_info=True,
        )


async def run_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
    shm_resources=None,
):
    logger.info(
        f"DP worker {dp_rank} starting on gpu_id={gpu_id} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    )

    # gpu_id is the device chosen by maybe_reindex_device_id in the parent:
    # 0 when CVD is pinned to one GPU, else the absolute id. rank=0, so
    # MMEncoder runs set_device(base_gpu_id).
    args = copy.deepcopy(server_args)
    args.base_gpu_id = gpu_id
    args.tp_size = 1
    enc = MMEncoder(
        args,
        dist_init_method=f"tcp://127.0.0.1:{get_free_port()}",
        rank=0,
        shm_resources=shm_resources,
    )

    global encoder_metrics_collector
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {"model_name": server_args.served_model_name}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encoder_metrics_collector = EncoderMetricsCollector(labels)
        enc.dp_rank = dp_rank

    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        trace_set_thread_info(f"Encoder-DP{dp_rank}", dp_rank=dp_rank)

    sched = EncoderScheduler(
        encoder=enc, send_sockets=[], max_batch_size=ENCODER_MAX_BATCH_SIZE
    )

    ctx = zmq.asyncio.Context(2)
    recv_sock = get_zmq_socket(ctx, zmq.PULL, dispatch_path, False)
    send_sock = get_zmq_socket(ctx, zmq.PUSH, result_path, False)
    send_lock = asyncio.Lock()
    inflight: Set[asyncio.Task] = set()
    # Acquire-before-recv → back-pressure propagates to the dispatcher
    # PUSH buffer. Must be ≥ ENCODER_MAX_BATCH_SIZE or batching degrades.
    max_inflight = envs.SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT.get()
    if max_inflight < ENCODER_MAX_BATCH_SIZE:
        logger.warning(
            f"SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT={max_inflight} is below "
            f"ENCODER_MAX_BATCH_SIZE={ENCODER_MAX_BATCH_SIZE}; the encoder "
            f"will never assemble a full batch."
        )
    inflight_sem = asyncio.Semaphore(max_inflight)
    sched.start()
    logger.info(f"DP worker {dp_rank} ready")

    # Task-per-request so EncoderScheduler.pending_queue accumulates and
    # actual cross-request batching can happen.
    try:
        while True:
            await inflight_sem.acquire()
            # Released by _run on success or the outer finally if not spawned.
            spawned = False
            try:
                try:
                    request = await recv_sock.recv_pyobj()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error(f"DP worker {dp_rank} recv error", exc_info=True)
                    continue
                if not isinstance(request, dict):
                    logger.error(
                        f"DP worker {dp_rank} received non-dict request "
                        f"({type(request).__name__}); dropping"
                    )
                    continue
                dp_type = request.pop("_dp_type", "encode")

                async def _run(req=request, t=dp_type):
                    try:
                        await _dp_worker_handle_request(
                            enc, sched, send_sock, send_lock, dp_rank, req, t
                        )
                    finally:
                        inflight_sem.release()

                task = asyncio.create_task(_run())
                # Ownership transferred to _run; mark before any op that could
                # raise (theoretical: set.add / add_done_callback) and cause a
                # double-release.
                spawned = True
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            finally:
                if not spawned:
                    inflight_sem.release()
    finally:
        # Close zmq on exception/cancellation (normal stop is parent SIGKILL).
        for task in inflight:
            task.cancel()
        ctx.destroy(linger=0)


def launch_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
    shm_resources=None,
):
    try:
        configure_logger(server_args, prefix=f" encode_dp_worker[{dp_rank}]")
        asyncio.run(
            run_dp_worker(
                server_args, dp_rank, gpu_id, dispatch_path, result_path, shm_resources
            )
        )
    except KeyboardInterrupt:
        logger.info(f"DP worker {dp_rank} exiting")
    except Exception:
        traceback.print_exc()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    global encoder_scheduler
    if dp_dispatcher is not None:
        dp_dispatcher.start()
        yield
        return
    if encoder is not None:
        encoder_scheduler = EncoderScheduler(
            encoder, send_sockets, max_batch_size=ENCODER_MAX_BATCH_SIZE
        )
        encoder_scheduler.start()
    try:
        yield
    finally:
        if encoder_scheduler is not None:
            await encoder_scheduler.stop()


app = FastAPI(lifespan=_lifespan)


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        if isinstance(request, ProfileReq):
            if request.type == ProfileReqType.START_PROFILE:
                if encoder.profiler is None:
                    encoder.profiler = EncoderProfiler(encoder.rank)
                encoder.profiler.start(request)
            else:
                encoder.profiler.stop()
        elif isinstance(request, dict) and request.get("type") == "batch_encode":
            await encoder.batch_encode(
                request["requests"],
                Modality.from_str(request["modality"]),
            )
        else:
            await encoder.encode_request(
                request, Modality.from_str(request["modality"])
            )


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def launch_server(server_args: ServerArgs):
    configure_logger(server_args, prefix=" encode_server")
    if server_args.dp_size > 1:
        _launch_server_dp(server_args)
        return

    global encoder, encoder_metrics_collector

    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {"model_name": server_args.served_model_name}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encoder_metrics_collector = EncoderMetricsCollector(labels)
        add_prometheus_middleware(app)

    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        trace_set_thread_info("Encoder")

    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _launch_server_dp(server_args: ServerArgs):
    global dp_dispatcher

    if server_args.dp_size <= 1 or server_args.tp_size != 1:
        raise ValueError(
            "Encoder DP mode requires --dp-size > 1 and --tp-size 1; got "
            f"dp_size={server_args.dp_size}, tp_size={server_args.tp_size}."
        )
    dp_size = server_args.dp_size
    logger.info(f"Launching encoder in DP mode: dp_size={dp_size}")

    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        add_prometheus_middleware(app)

    ctx = mp.get_context("spawn")
    ipc_prefix = random_uuid()
    async_zmq_ctx = zmq.asyncio.Context(dp_size + 1)

    result_path = f"ipc:///tmp/{ipc_prefix}_dp_result"
    result_socket = get_zmq_socket(async_zmq_ctx, zmq.PULL, result_path, True)

    dispatch_sockets: List[zmq.asyncio.Socket] = [
        get_zmq_socket(
            async_zmq_ctx, zmq.PUSH, f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{r}", True
        )
        for r in range(dp_size)
    ]

    shm_resources = None
    if server_args.enable_prefix_mm_cache and envs.SGLANG_MM_CACHE_SHM.get():
        mgr = ctx.Manager()
        shm_resources = {
            "index": mgr.dict(),
            "lru": mgr.list(),
            "size": mgr.Value("l", 0),
            "lock": ctx.Lock(),
        }
        logger.info(f"Shared mm_cache enabled across {dp_size} DP workers")

    # Register atexit BEFORE spawn loop so partial spawns get reaped on
    # exception (atexit holds the list ref and reads it at exit time).
    import atexit

    worker_processes: List[mp.Process] = []

    def _kill_workers():
        for p in worker_processes:
            if p.is_alive():
                p.kill()
        for p in worker_processes:
            p.join(timeout=5)
        if shm_resources is not None:
            from multiprocessing.shared_memory import SharedMemory

            for meta in list(shm_resources["index"].values()):
                names = [meta.get("shm_name")]
                names += [seg.get("shm_name") for seg in meta.get("aux_segments", [])]
                for name in names:
                    if not name:
                        continue
                    try:
                        shm = SharedMemory(name=name, create=False)
                        shm.close()
                        shm.unlink()
                    except FileNotFoundError:
                        pass

    atexit.register(_kill_workers)

    for dp_rank in range(dp_size):
        gpu_id = server_args.base_gpu_id + dp_rank
        # Pin the device parent-side around spawn (same convention as the
        # scheduler launcher and DP controller) so the child inherits
        # CUDA_VISIBLE_DEVICES from its first instruction, before any import
        # can enumerate CUDA. No-op unless SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
        # is set, in which case gpu_id is reindexed to 0 and CVD is pinned.
        with maybe_reindex_device_id(gpu_id) as gpu_id:
            proc = ctx.Process(
                target=launch_dp_worker,
                args=(
                    server_args,
                    dp_rank,
                    gpu_id,
                    f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{dp_rank}",
                    result_path,
                    shm_resources,
                ),
                daemon=False,
            )
            proc.start()
        worker_processes.append(proc)

    dp_dispatcher = DPDispatcher(
        dp_size,
        dispatch_sockets,
        result_socket,
        worker_processes,
        enable_metrics=server_args.enable_metrics,
    )

    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _summarise_dp_broadcast(results: List[dict]) -> Response:
    # Treat missing/None content as failure so a stuck rank doesn't hide
    # behind the others' "ok". Status = the most severe per-rank error code
    # (5xx beats 4xx) rather than a blanket 400, so a worker's 500/503/504
    # isn't misreported as a client error.
    msgs: List[str] = []
    error_codes: List[int] = []
    for r in results:
        content = r.get("content")
        if isinstance(content, dict):
            msgs.append(content.get("msg", ""))
            if not content.get("ok"):
                # Worker ran but reported a logical failure; no transport code,
                # so treat as a bad request (matches the non-DP profile path).
                error_codes.append(int(r.get("_error_code") or HTTPStatus.BAD_REQUEST))
        else:
            msgs.append(r.get("_error", "unknown error"))
            error_codes.append(
                int(r.get("_error_code") or HTTPStatus.INTERNAL_SERVER_ERROR)
            )
    status_code = 200 if not error_codes else max(error_codes)
    return Response(
        content="\n".join(msgs) + "\n",
        status_code=status_code,
    )


async def get_condition(rid):
    async with cond_dict_lock:
        if rid not in rid_to_cond:
            rid_to_cond[rid] = asyncio.Condition()
        return rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_id = request["req_id"]
    start_time = time.monotonic()
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch(request)
        except MMError as e:
            # Surface MMError.code (503 when all workers dead) instead of
            # FastAPI's default 500.
            logger.error(f"DP dispatch refused req_id={req_id}: {e}")
            return ORJSONResponse(
                status_code=int(e.code),
                content={"status": "error", "message": str(e), "req_id": req_id},
            )
        if result.get("_error"):
            error_type = result.get("_error_type", "")
            # `or` (not `dict.get(key, default)`) so explicit None falls back too.
            status_code = result.get("_error_code") or (
                HTTPStatus.BAD_REQUEST
                if error_type == "ValueError"
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(f"DP worker error for req_id={req_id}: {result['_error']}")
            return ORJSONResponse(
                status_code=status_code,
                content={
                    "status": "error",
                    "message": result["_error"],
                    "req_id": req_id,
                },
            )
        elapsed = time.monotonic() - start_time
        logger.info(
            f"[{req_id}] /encode completed in {elapsed:.3f}s, "
            f"modality={request.get('modality', 'image')}"
        )
        return ORJSONResponse(content=result.get("content"))

    modality_str = str(request.get("modality", "image")).lower()
    log_metrics = request.get("log_metrics", True)
    try:

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        request.update({"enter_time": time.time()})
        modality = Modality.from_str(request["modality"])
        modality_str = modality.name.lower()
        if encoder_metrics_collector is not None and log_metrics:
            encoder_metrics_collector.inc_requests_received(
                dp_rank=0, modality=modality_str
            )
        if encoder_scheduler is not None and modality in _BATCHABLE_MODALITIES:
            try:
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder_scheduler.submit(request)
                )
            except asyncio.TimeoutError:
                return ORJSONResponse(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    content={
                        "status": "error",
                        "message": "encoder batch timed out",
                        "req_id": req_id,
                    },
                )
        else:
            for socket in send_sockets:
                socket.send_pyobj(request)
            nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                await _encode_request_direct(encoder, request, modality)
            )

        if error_msg:
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            if encoder_metrics_collector is not None and log_metrics:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="error"
                )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        _stamp_encoder_e2e(encoder, request)
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            if encoder_metrics_collector is not None and log_metrics:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            if encoder_metrics_collector is not None and log_metrics:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            elapsed = time.monotonic() - start_time
            logger.info(
                f"[{req_id}] /encode completed in {elapsed:.3f}s, "
                f"modality={request['modality']}, tokens={embedding_len}"
            )
            if encoder_metrics_collector is not None and log_metrics:
                encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        rid_to_err_msg[req_id] = error_msg
        if encoder_metrics_collector is not None and log_metrics:
            encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_send(request)
        except MMError as e:
            req_id = request.get("req_id", "?")
            logger.error(f"DP dispatch_send refused req_id={req_id}: {e}")
            return Response(
                content=f"Encoder DP worker send error: {e}",
                status_code=int(e.code),
            )
        if result.get("_error"):
            req_id = request.get("req_id", "?")
            status_code = result.get("_error_code") or int(
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(
                f"DP worker send error for req_id={req_id}: {result['_error']}"
            )
            return Response(
                content=f"Encoder DP worker send error: {result['_error']}",
                status_code=status_code,
            )
        return ORJSONResponse(content=result.get("content"))
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    encoder.embedding_to_send.pop(request["req_id"], None)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with rid_lock:
        global rid_to_receive_endpoint
        if rid not in rid_to_receive_endpoint:
            rid_to_receive_endpoint[rid] = set()
            rid_to_receive_count[rid] = request["receive_count"]
        assert rid_to_receive_count[rid] == request["receive_count"]
        rid_to_receive_endpoint[rid].add(request["receive_url"])
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Returns 200 if the encoder is initialized and ready.
    """
    if dp_dispatcher is not None:
        # Strict: any dead (exited) rank fails health → orchestrator restarts.
        if not dp_dispatcher.all_ranks_alive:
            return Response(status_code=503)
        # Process-liveness (proc.sentinel) can't see a worker that's alive but
        # wedged (hung GPU / NCCL deadlock / stalled ZMQ). Probe every rank with
        # a tiny dummy encode; each worker runs it only when idle and otherwise
        # reports healthy at once, keeping the probe off the GPU under load.
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "health_encode"},
                timeout=HEALTH_CHECK_TIMEOUT,
            )
        except MMError:
            return Response(status_code=503)
        if any(r.get("_error") for r in results):
            return Response(status_code=503)
        return Response(status_code=200)
    if encoder is None:
        return Response(status_code=503)
    return Response(status_code=200)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    if dp_dispatcher is not None:
        profile_req = None
        if obj is not None:
            profile_req = {
                "type": ProfileReqType.START_PROFILE,
                "output_dir": obj.output_dir,
                "start_step": obj.start_step,
                "num_steps": obj.num_steps,
                "activities": obj.activities,
                "with_stack": obj.with_stack,
                "record_shapes": obj.record_shapes,
                "profile_by_stage": obj.profile_by_stage,
                "profile_id": str(time.time()),
                "merge_profiles": obj.merge_profiles,
                "profile_prefix": obj.profile_prefix,
                "profile_stages": obj.profile_stages,
            }
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "start_profile", "profile_req": profile_req}
            )
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = None
    if obj is None:
        req = ProfileReq(ProfileReqType.START_PROFILE)
    else:
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=obj.output_dir,
            start_step=obj.start_step,
            num_steps=obj.num_steps,
            activities=obj.activities,
            with_stack=obj.with_stack,
            record_shapes=obj.record_shapes,
            profile_by_stage=obj.profile_by_stage,
            profile_id=str(time.time()),
            merge_profiles=obj.merge_profiles,
            profile_prefix=obj.profile_prefix,
            profile_stages=obj.profile_stages,
        )
    for socket in send_sockets:
        socket.send_pyobj(req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    if dp_dispatcher is not None:
        try:
            results = await dp_dispatcher.broadcast({"_dp_type": "stop_profile"})
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        socket.send_pyobj(req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
