"""Validate a rebuilt FA3 .so on a GPU, without importing an installed sgl_kernel.

FA3_TEST_LIBRARY=/absolute/path/flash_ops.abi3.so python -m pytest -q <this file>
"""

import math
import os
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class TestFA30510Kernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("FA3_TEST_LIBRARY")
        if not library:
            raise unittest.SkipTest("Set FA3_TEST_LIBRARY to the newly built .so")
        torch.ops.load_library(str(Path(library).resolve(strict=True)))

    def make_case(self, q_lens, num_splits, use_metadata):
        torch.manual_seed(51)
        device = "cuda"
        dtype = torch.bfloat16
        heads, kv_heads, dim, v_dim, page_size = 4, 2, 192, 128, 64
        k_lens = (769, 513)
        batch = len(q_lens)
        pages = math.ceil(max(k_lens) / page_size)
        q = torch.randn(sum(q_lens), heads, dim, device=device, dtype=dtype)
        k = torch.randn(
            batch * pages, page_size, kv_heads, dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch * pages, page_size, kv_heads, v_dim, device=device, dtype=dtype
        )
        page_table = torch.randperm(
            batch * pages, device=device, dtype=torch.int32
        ).view(batch, pages)
        cu_q = torch.tensor(
            [0, q_lens[0], sum(q_lens)], device=device, dtype=torch.int32
        )
        used_k = torch.tensor(k_lens, device=device, dtype=torch.int32)
        out = torch.empty(sum(q_lens), heads, v_dim, device=device, dtype=dtype)
        scale = dim**-0.5

        reference = []
        q_start = 0
        for b, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
            kb = k[page_table[b].long()].reshape(-1, kv_heads, dim)[:k_len]
            vb = v[page_table[b].long()].reshape(-1, kv_heads, v_dim)[:k_len]
            kb = kb.float().repeat_interleave(heads // kv_heads, dim=1)
            vb = vb.float().repeat_interleave(heads // kv_heads, dim=1)
            qb = q[q_start : q_start + q_len].float()
            scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
            # FA3 aligns the causal mask to the end of the KV sequence.
            allowed = torch.arange(k_len, device=device)[None, :] <= (
                k_len - q_len + torch.arange(q_len, device=device)[:, None]
            )
            scores.masked_fill_(~allowed, float("-inf"))
            reference.append(torch.einsum("hqk,khv->qhv", scores.softmax(-1), vb))
            q_start += q_len

        metadata = None
        if use_metadata:
            metadata = torch.ops.sgl_kernel.get_scheduler_metadata.default(
                batch,
                max(q_lens),
                pages * page_size,
                heads,
                kv_heads,
                dim,
                v_dim,
                dtype,
                used_k,
                cu_q,
                None,
                None,
                None,
                None,
                page_size,
                is_causal=True,
                num_splits=num_splits,
            )

        def forward(**overrides):
            args = dict(
                q=q,
                k=k,
                v=v,
                k_new=None,
                v_new=None,
                q_v=None,
                out=out,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=None,
                cu_seqlens_k_new=None,
                seqused_q=None,
                seqused_k=used_k,
                max_seqlen_q=max(q_lens),
                max_seqlen_k=None,
                page_table=page_table,
                kv_batch_idx=None,
                leftpad_k=None,
                rotary_cos=None,
                rotary_sin=None,
                seqlens_rotary=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                softmax_scale=scale,
                is_causal=True,
                window_size_left=-1,
                window_size_right=-1,
                attention_chunk=0,
                softcap=0.0,
                is_rotary_interleaved=True,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=None,
                sm_margin=0,
                sinks=None,
                sparse_mask_fine=None,
                only_qv=False,
            )
            args.update(overrides)
            return torch.ops.sgl_kernel.fwd.default(**args)

        return forward, out, torch.cat(reference)

    def test_paged_prefill_and_decode(self):
        for q_lens in ((3, 1), (1, 1)):
            for num_splits in (0, 1, 2):
                for use_metadata in (False, True):
                    with self.subTest(
                        q_lens=q_lens, num_splits=num_splits, metadata=use_metadata
                    ):
                        forward, out, reference = self.make_case(
                            q_lens, num_splits, use_metadata
                        )
                        result = forward()
                        self.assertEqual(len(result), 4)
                        self.assertEqual(result[0].data_ptr(), out.data_ptr())
                        torch.testing.assert_close(
                            out.float(), reference, atol=2e-2, rtol=2e-2
                        )

    def test_decode_cuda_graph(self):
        forward, out, reference = self.make_case((1, 1), 0, True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        out.zero_()
        graph.replay()
        torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)

    def test_unsupported_features_are_rejected(self):
        forward, _, _ = self.make_case((1, 1), 1, False)
        with self.assertRaisesRegex(RuntimeError, "0510.*only_qv"):
            forward(only_qv=True)
        mask = torch.zeros((2, 1, 32), device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "0510.*sparse_mask_fine"):
            forward(sparse_mask_fine=mask)


if __name__ == "__main__":
    unittest.main()
