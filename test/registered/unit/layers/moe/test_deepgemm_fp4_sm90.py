"""Host-side contracts for the internal v0.5.10-xm SM90 W4A8 kernels."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.moe.moe_runner import deep_gemm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    _DeepEPDispatcherImplLowLatency,
)
from sglang.srt.runtime_context import get_context


@pytest.mark.parametrize(
    "tokens,experts,want",
    [(0, 256, 1), (256, 256, 32), (768, 256, 96), (769, 256, 97), (256, 384, 22)],
)
def test_deepep_expected_m_does_not_cross_fp4_threshold_on_exact_division(
    tokens, experts, want
):
    dispatcher = object.__new__(_DeepEPDispatcherImplLowLatency)
    dispatcher.num_experts = experts
    dispatcher._get_buffer = lambda: SimpleNamespace(group_size=4)
    # Only the external collective is replaced; dispatch_a still computes the
    # hint passed to the real runner from this rank's token and top-k counts.
    dispatcher._dispatch_core = lambda hidden, ids, weights: (hidden, None, None, None)
    hidden = torch.empty(tokens, 128, dtype=torch.bfloat16)
    topk = SimpleNamespace(
        topk_ids=torch.zeros(tokens, 8, dtype=torch.int64),
        topk_weights=torch.ones(tokens, 8),
    )
    result = dispatcher.dispatch_a(hidden, topk)
    assert result[4] == want


def _masked_case(expected_m):
    runner = deep_gemm.DeepGemmRunnerCore(
        MoeRunnerConfig(
            num_experts=1, num_local_experts=1,
            intermediate_size_per_partition=128, top_k=1,
            activation="silu", is_gated=True,
        )
    )
    inputs = deep_gemm.DeepGemmRunnerInput(
        hidden_states=torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(1, 128, 1),
        use_masked_gemm=True,
        masked_m=torch.tensor([expected_m], dtype=torch.int32),
        expected_m=expected_m,
    )
    quant = deep_gemm.DeepGemmMoeQuantInfo(
        w13_weight=torch.zeros(1, 256, 64, dtype=torch.int8),
        w2_weight=torch.zeros(1, 128, 64, dtype=torch.int8),
        use_fp8=True,
        w13_scale=torch.full((1, 256, 4), 127, dtype=torch.uint8),
        w2_scale=torch.full((1, 128, 4), 127, dtype=torch.uint8),
        block_shape=[128, 128],
        is_fp4_experts=True,
    )
    return runner, inputs, quant


@pytest.mark.parametrize("overlap_key", ["down_gemm_overlap_args", "meta_overlap_args"])
def test_fp4_masked_overlap_fails_before_launch_or_wait(overlap_key):
    with (
        get_context().override_server_args(moe_a2a_backend="none"),
        patch.object(
            deep_gemm, "get_platform", return_value=SimpleNamespace(is_sm90=True),
            create=True,
        ),
    ):
        runner, inputs, quant = _masked_case(16)
        with pytest.raises(NotImplementedError, match="overlap"):
            runner._run_masked_gemm(
                inputs, quant, {"hidden_states_device": "cpu", overlap_key: object()}
            )


def test_fp4_large_masked_batch_requires_decoded_scales_before_launch():
    with (
        get_context().override_server_args(moe_a2a_backend="none"),
        patch.object(
            deep_gemm, "get_platform", return_value=SimpleNamespace(is_sm90=True),
            create=True,
        ),
    ):
        runner, inputs, quant = _masked_case(97)
        with pytest.raises(ValueError, match="fp32.*scale"):
            runner._run_masked_gemm(inputs, quant, {"hidden_states_device": "cpu"})


def test_decode_warmup_is_bounded_by_batch_and_speculative_tokens():
    with (
        get_context().override_server_args(
            disaggregation_mode="decode", chunked_prefill_size=49152,
            max_running_requests=128, speculative_num_draft_tokens=8,
        ),
        patch.object(compile_utils, "_FAST_WARMUP", False),
        patch.object(compile_utils, "_BUILTIN_M_LIST", []),
    ):
        compile_utils.update_deep_gemm_config(gpu_id=0)
        assert max(compile_utils._BUILTIN_M_LIST) == 1032
        assert 96 in compile_utils._BUILTIN_M_LIST
        assert 97 in compile_utils._BUILTIN_M_LIST
