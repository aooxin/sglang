"""CUDA DP acceptance and draft pool sizing use the draft's actual TP width."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _resolve_dflash_draft_cell_size,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def dflash_args(**overrides):
    fields = dict(
        model_path="unused-target",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="unused-draft",
        speculative_num_draft_tokens=8,
        speculative_draft_attention_backend="fa3",
        device="cuda",
        tp_size=8,
        dp_size=2,
        pp_size=1,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
    )
    fields.update(overrides)
    return ServerArgs(**fields)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "npu"])
def test_dflash_dp_configuration(device):
    args = dflash_args(device=device)
    _handle_dflash(args)
    resolved = resolving_view(args)
    assert resolved.speculative_num_steps == 1
    assert resolved.speculative_eagle_topk == 1
    assert resolved.speculative_num_draft_tokens == 8


def test_cuda_dp_requires_local_vocabulary_head():
    with pytest.raises(ValueError, match="enable-dp-lm-head"):
        _handle_dflash(dflash_args(enable_dp_lm_head=False))


def test_dflash_dp_rejects_unsupported_device():
    with pytest.raises(ValueError, match="dp attention"):
        _handle_dflash(dflash_args(device="xpu"))


def test_dflash_keeps_pipeline_parallel_restriction():
    with pytest.raises(ValueError, match="pp_size == 1"):
        _handle_dflash(dflash_args(pp_size=2))


@pytest.mark.parametrize("dp_attention, expected_heads", [(True, 2), (False, 1)])
def test_draft_byte_budget_matches_attention_tp_width(dp_attention, expected_heads):
    # Eight KV heads are sharded over four draft ranks under TP8/DP2.
    # Counting with the target's global TP8 would under-budget this pool by 2x.
    draft = SimpleNamespace(
        dtype=torch.bfloat16,
        head_dim=128,
        v_head_dim=128,
        get_num_kv_heads=lambda tp_size: max(1, 8 // tp_size),
    )
    with get_context().override_server_args(
        tp_size=8,
        dp_size=2,
        enable_dp_attention=dp_attention,
        kv_cache_dtype="auto",
        speculative_draft_kv_cache_dtype=None,
        speculative_draft_attention_backend="fa3",
    ):
        assert (
            _resolve_dflash_draft_cell_size(
                draft_model_config=draft, draft_num_layers=5
            )
            == expected_heads * (128 + 128) * 5 * 2
        )
