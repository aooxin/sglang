from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4W4A8MoEMethod:
    """Humming MXFP4 W4A8 expert method for DeepEP dispatch."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        import sglang.srt.layers.moe.moe_runner.mxfp4_w4a8  # noqa: F401
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MXFP4_W4A8, moe_runner_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Check the hardware before the fp8 pass rather than after: that pass walks
        # and rewrites every expert tensor, so failing afterwards means paying the
        # full per-layer conversion on a device that cannot run the kernels, and
        # surfacing whatever the fp8 path happens to raise first instead of this
        # message.
        if not is_cuda() or not is_sm90_supported():
            raise RuntimeError("mxfp4_w4a8 MoE runner currently requires Hopper/SM90.")

        self._fp8.process_weights_after_loading(layer)

        from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8_humming import (
            replace_mxfp4_w4a8_weights_with_humming,
        )

        # Expose the EP size so the Humming decode path can pick the fine-grained
        # per-(N, K, ep_size) masked tuning table. Read it from the parallel state
        # rather than dividing num_experts by num_local_experts: FusedMoE defines
        # num_local_experts as (num_experts - num_fused_shared_experts) // ep_size
        # + num_fused_shared_experts, so that quotient is only the EP size while
        # no shared expert is fused into the routed set.
        layer.mxfp4_w4a8_ep_size = get_moe_expert_parallel_world_size()

        replace_mxfp4_w4a8_weights_with_humming(layer)
        log_info_on_rank0(
            logger,
            f"Using Humming MXFP4 W4A8 repacked weights for MoE layer {self.prefix}.",
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.moe_runner.mxfp4_w4a8 import (
            Mxfp4W4A8QuantInfo,
        )

        quant_info = Mxfp4W4A8QuantInfo(
            w13_weight=layer.w13_humming_weight,
            w2_weight=layer.w2_humming_weight,
        )
        return self.runner.run(dispatch_output, quant_info=quant_info)
