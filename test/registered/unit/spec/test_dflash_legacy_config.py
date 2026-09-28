"""Keep MiMo's published DFlash checkpoint semantics across HF config loading."""

import pytest
import torch
from transformers import Qwen3Config

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.dflash import DFlashAttention, _get_dflash_layer_attention_params
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def legacy_config(**overrides):
    fields = dict(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=5,
        sliding_window=1024,
        dflash_config=dict(
            target_layer_ids=[0, 11, 23, 35, 47],
            mask_token_id=151675,
            num_anchors=4096,
            block_size=8,
            loss_decay_gamma=7.0,
            use_swa=True,
            swa_window_size=1024,
            attention_value_scale=0.612,
            attention_sink_bias=True,
        ),
    )
    fields.update(overrides)
    return fields


@pytest.mark.parametrize("hf_loaded", [False, True])
def test_legacy_window_survives_hf_default_layer_types(hf_loaded):
    config = legacy_config()
    if hf_loaded:
        config = Qwen3Config(**config)
        assert config.layer_types == ["full_attention"] * 5
    assert get_dflash_layer_types(config) == ["sliding_attention"] * 5
    assert get_dflash_attention_sliding_window_size(config) == 1024
    parsed = parse_dflash_draft_config(draft_hf_config=config)
    assert parsed.require_num_layers() == 5
    assert parsed.resolve_block_size() == 8
    assert parsed.resolve_target_layer_ids(target_num_layers=48) == [0, 11, 23, 35, 47]
    assert parsed.mask_token_id == 151675
    assert parsed.attention_value_scale == 0.612
    assert parsed.attention_sink_bias is True
    assert parsed.use_swa is True
    assert parsed.swa_window_size == 1024


def test_legacy_layers_are_bidirectional_with_raw_window_units():
    config = Qwen3Config(**legacy_config(sliding_window=2048))
    for layer in range(5):
        assert _get_dflash_layer_attention_params(config, layer) == (
            1024,
            AttentionType.ENCODER_ONLY,
        )
    # Resolving the attention layout must not overwrite checkpoint provenance.
    assert config.layer_types == ["full_attention"] * 5


def test_legacy_window_falls_back_to_top_level():
    fields = legacy_config()
    del fields["dflash_config"]["swa_window_size"]
    assert get_dflash_attention_sliding_window_size(fields) == 1024
    # Qwen3 discards sliding_window when its own use_sliding_window is false.
    fields["use_sliding_window"] = True
    assert get_dflash_attention_sliding_window_size(Qwen3Config(**fields)) == 1024


def test_explicit_legacy_disable_overrides_layer_types():
    fields = legacy_config(layer_types=["sliding_attention"] * 5)
    fields["dflash_config"]["use_swa"] = False
    config = Qwen3Config(**fields)
    assert get_dflash_attention_sliding_window_size(config) is None
    for layer in range(5):
        assert _get_dflash_layer_attention_params(config, layer) == (
            -1,
            AttentionType.ENCODER_ONLY,
        )


def test_modern_mixed_layers_keep_hf_window_and_causality():
    config = Qwen3Config(
        num_hidden_layers=3,
        use_sliding_window=True,
        sliding_window=1024,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
    )
    assert _get_dflash_layer_attention_params(config, 0) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )
    assert _get_dflash_layer_attention_params(config, 1) == (
        1023,
        AttentionType.DECODER,
    )
    assert _get_dflash_layer_attention_params(config, 2) == (
        -1,
        AttentionType.ENCODER_ONLY,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_causality_override_is_honored(legacy):
    fields = legacy_config(is_causal=True)
    if not legacy:
        fields.pop("dflash_config")
        fields["use_sliding_window"] = True
        fields["layer_types"] = ["sliding_attention"] * 5
    assert _get_dflash_layer_attention_params(Qwen3Config(**fields), 0) == (
        1024 if legacy else 1023,
        AttentionType.DECODER,
    )


@pytest.mark.parametrize("flag", [1, 0, "true", None])
def test_invalid_legacy_flag_fails_at_config_parse(flag):
    config = legacy_config()
    config["dflash_config"]["use_swa"] = flag
    with pytest.raises(ValueError, match="use_swa.*bool"):
        parse_dflash_draft_config(draft_hf_config=config)


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "1024"])
def test_invalid_legacy_window_fails_at_config_parse(window):
    config = legacy_config()
    config["dflash_config"]["swa_window_size"] = window
    with pytest.raises(ValueError, match="swa_window_size.*positive integer"):
        parse_dflash_draft_config(draft_hf_config=config)


def test_legacy_swa_requires_a_window():
    config = legacy_config()
    del config["sliding_window"]
    del config["dflash_config"]["swa_window_size"]
    with pytest.raises(ValueError, match="swa_window_size.*positive integer"):
        get_dflash_attention_sliding_window_size(config)


@pytest.mark.parametrize("backend", ["fa3", "fa4"])
@pytest.mark.parametrize("legacy_use_swa", [None, False, True])
def test_sink_dtype_preserves_legacy_fa3_contract(backend, legacy_use_swa):
    fields = legacy_config(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
    )
    if legacy_use_swa is None:
        fields["dflash_config"].pop("use_swa")
    else:
        fields["dflash_config"]["use_swa"] = legacy_use_swa
    with (
        get_context().override_server_args(
            tp_size=1, speculative_draft_attention_backend=backend
        ),
        get_parallel().override(tp_rank=0, attn_tp_rank=0),
        set_default_torch_dtype(torch.bfloat16),
    ):
        attn = DFlashAttention(Qwen3Config(**fields), layer_id=0)
    # The legacy FA3 kernel consumes BF16 sinks. Preserve the existing FP32
    # contract for modern checkpoints and other attention backends.
    expected = (
        torch.bfloat16
        if legacy_use_swa is not None and backend == "fa3"
        else torch.float32
    )
    assert attn.attention_sink_bias.dtype == expected


@pytest.mark.parametrize("legacy", [True, False])
def test_trtllm_fallback_respects_legacy_noncausal_attention(tmp_path, legacy):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.speculative_hook import (
        _resolve_dflash_draft_attention_backend,
    )
    from sglang.srt.server_args import ServerArgs

    fields = legacy_config(use_sliding_window=True)
    if not legacy:
        fields.pop("dflash_config")
        fields["layer_types"] = ["sliding_attention"] * 5
    Qwen3Config(**fields).save_pretrained(tmp_path)
    args = ServerArgs(
        model_path="unused-target",
        speculative_draft_model_path=str(tmp_path),
        speculative_draft_attention_backend="trtllm_mha",
    )
    _resolve_dflash_draft_attention_backend(args)
    assert resolving_view(args).speculative_draft_attention_backend == (
        "flashinfer" if legacy else "trtllm_mha"
    )
