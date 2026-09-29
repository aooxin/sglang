"""DFlash's noncausal window must include both sides of each query."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import flashattention_backend as fa
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize(
    "attn_type,window,expected_window,causal",
    [
        (AttentionType.ENCODER_ONLY, 2, (2, 2), False),
        (AttentionType.DECODER, 2, (2, 0), True),
        (AttentionType.ENCODER_ONLY, -1, (-1, -1), False),
        (AttentionType.DECODER, -1, (-1, -1), True),
    ],
)
def test_decode_window_contract(
    monkeypatch, version, attn_type, window, expected_window, causal
):
    backend = object.__new__(fa.FlashAttentionBackend)
    backend.fa_impl_ver = version
    backend.use_mla = False
    backend.has_local_attention = False
    backend.is_prefill_aware_swa = False
    backend.use_sliding_window_kv_pool = False
    backend.kv_cache_dtype_str = "auto"
    backend.kv_cache_is_mxfp8 = False
    backend.page_size = 1
    backend.topk = 1
    backend.num_splits = backend.decode_num_splits = 1
    backend._decode_uses_static_max_seqlen_k = version == 4
    backend.forward_metadata = SimpleNamespace(
        page_table=torch.arange(4, dtype=torch.int32)[None, :],
        cache_seqlens_int32=torch.tensor([4], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        max_seq_len_q=4,
        max_seq_len_k=4,
        scheduler_metadata=None,
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda layer_id: (torch.empty(4, 1, 16), torch.empty(4, 1, 16)),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=window,
        attn_type=attn_type,
        is_cross_attention=False,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=16,
        v_head_dim=16,
        scaling=0.25,
        logit_cap=0,
    )
    batch = SimpleNamespace(batch_size=1, spec_info=None)
    kernel = Mock(return_value=torch.empty(4, 2, 16))
    monkeypatch.setattr(fa, "flash_attn_with_kvcache", kernel)
    backend.forward_decode(torch.empty(4, 2, 16), None, None, layer, batch)
    assert kernel.call_args.kwargs["window_size"] == expected_window
    assert kernel.call_args.kwargs["causal"] is causal
    assert kernel.call_args.kwargs["ver"] == version
