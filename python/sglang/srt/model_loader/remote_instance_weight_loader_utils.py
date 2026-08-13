# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import time
from typing import List

import requests

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


MAIN_MODEL_IDX = -1


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


def get_remote_instance_model_idx(draft_model_idx: int = None) -> int:
    """Return the model index used to key bootstrap server entries.

    Convention:
        draft_model_idx is None  → MAIN_MODEL_IDX  (main model, not a draft worker)
        draft_model_idx is int   → draft_model_idx (draft model layer index, 0-indexed)

    Under speculative decoding the same tp_rank may host multiple ModelRunners
    (one main model + one or more draft models).  The bootstrap server uses
    (tp_rank, model_idx) as a composite key so each runner's registration is
    isolated.
    """
    return draft_model_idx if draft_model_idx is not None else MAIN_MODEL_IDX


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
            timeout=envs.SGLANG_REMOTE_INSTANCE_WEIGHT_LOADER_REQUEST_TIMEOUT.get(),
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
            timeout=envs.SGLANG_REMOTE_INSTANCE_WEIGHT_LOADER_REQUEST_TIMEOUT.get(),
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(
    seed_url: str, rank: int, model_idx: int = MAIN_MODEL_IDX
):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
                "model_idx": model_idx,
            },
            timeout=envs.SGLANG_REMOTE_INSTANCE_WEIGHT_LOADER_REQUEST_TIMEOUT.get(),
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def register_memory_region(model, transfer_engine):
    weight_mr_dict, _ = register_memory_region_with_handles(model, transfer_engine)
    return weight_mr_dict


def register_memory_region_with_handles(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def unregister_memory_region(transfer_engine, registered_memory_addrs):
    if not registered_memory_addrs:
        return

    for address in registered_memory_addrs:
        try:
            ret = transfer_engine.unregister_memory(address)
        except Exception as e:
            logger.warning(
                f"Failed to unregister memory region at address {address}: {e}"
            )
            continue
        if ret != 0:
            logger.warning(
                f"unregister memory failed for address {address}, error: {ret}"
            )


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    registered_memory_addrs = []
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            unregister_memory_region(transfer_engine, registered_memory_addrs)
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        registered_memory_addrs.append(weight.data_ptr())
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict, registered_memory_addrs


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Register merged memory blocks that hold weights.
    registered_memory_addrs = []
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            unregister_memory_region(transfer_engine, registered_memory_addrs)
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )
        registered_memory_addrs.append(address)

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict, registered_memory_addrs
