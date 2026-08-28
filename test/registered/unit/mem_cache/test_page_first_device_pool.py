import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHAPageFirstTokenToKVPool, MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


PAGE_SIZE = 4
SIZE = 16
HEAD_NUM = 2
HEAD_DIM = 4
V_HEAD_DIM = 3
LAYER_NUM = 2


def make_pool(pool_class, *, v_head_dim=HEAD_DIM):
    return pool_class(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float32,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=v_head_dim,
        layer_num=LAYER_NUM,
        device="cpu",
        enable_memory_saver=False,
    )


class FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class TestPageFirstDevicePool(unittest.TestCase):
    def test_buffer_shape_stride_and_flat_access(self):
        pool = make_pool(MHAPageFirstTokenToKVPool)
        self.assertEqual(tuple(pool.k_buffer[0].shape), (5, 4, 2, 4))
        self.assertEqual(pool.k_buffer[0].stride(1), LAYER_NUM * HEAD_NUM * HEAD_DIM)

        locations = torch.tensor([4, 5, 8, 11], dtype=torch.int64)
        values = torch.randn(4, HEAD_NUM, HEAD_DIM)
        pool.set_kv_buffer(FakeLayer(0), locations, values, values)
        self.assertTrue(
            torch.equal(
                pool.get_key_buffer(0).view(-1, HEAD_NUM, HEAD_DIM)[locations],
                values,
            )
        )

    def test_matches_token_linear_pool(self):
        reference = make_pool(MHATokenToKVPool)
        page_first = make_pool(MHAPageFirstTokenToKVPool)
        locations = torch.tensor([4, 6, 7, 12, 15], dtype=torch.int64)
        key = torch.randn(5, HEAD_NUM, HEAD_DIM)
        value = torch.randn(5, HEAD_NUM, HEAD_DIM)
        for pool in (reference, page_first):
            pool.set_kv_buffer(FakeLayer(1), locations, key, value)
        self.assertTrue(
            torch.equal(
                reference.get_key_buffer(1)[locations],
                page_first.get_key_buffer(1).view(-1, HEAD_NUM, HEAD_DIM)[locations],
            )
        )

    def test_raw_buffers_match_host_page_first_layout(self):
        pool = make_pool(MHAPageFirstTokenToKVPool, v_head_dim=V_HEAD_DIM)
        k_raw, v_raw = pool.get_page_first_raw_buffers()

        self.assertEqual(
            tuple(k_raw.shape),
            (SIZE + PAGE_SIZE, LAYER_NUM, HEAD_NUM, HEAD_DIM),
        )
        self.assertEqual(
            tuple(v_raw.shape),
            (SIZE + PAGE_SIZE, LAYER_NUM, HEAD_NUM, V_HEAD_DIM),
        )
        self.assertEqual(k_raw.data_ptr(), pool._k_raw.data_ptr())
        self.assertEqual(v_raw.data_ptr(), pool._v_raw.data_ptr())

    def test_rejects_legacy_disaggregation_descriptors(self):
        pool = make_pool(MHAPageFirstTokenToKVPool)
        with self.assertRaisesRegex(NotImplementedError, "descriptor protocol"):
            pool.get_contiguous_buf_infos()

    def test_page_level_disaggregation_descriptors(self):
        pool = make_pool(MHAPageFirstTokenToKVPool, v_head_dim=V_HEAD_DIM)
        ptrs, lens, item_lens = pool.get_page_level_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(len(lens), 2)
        self.assertEqual(len(item_lens), 2)
        self.assertEqual(item_lens[0], PAGE_SIZE * LAYER_NUM * HEAD_NUM * HEAD_DIM * 4)
        self.assertEqual(item_lens[1], PAGE_SIZE * LAYER_NUM * HEAD_NUM * V_HEAD_DIM * 4)


if __name__ == "__main__":
    unittest.main()
