import gc
import unittest
import weakref

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


def make_pool(pool_class, *, v_head_dim=HEAD_DIM, layer_num=LAYER_NUM, start_layer=0):
    return pool_class(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float32,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=v_head_dim,
        layer_num=layer_num,
        start_layer=start_layer,
        end_layer=start_layer + layer_num - 1,
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
        raw_k, raw_v = pool.get_page_first_raw_buffers()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(len(lens), 2)
        self.assertEqual(len(item_lens), 2)
        self.assertEqual(ptrs, [raw_k.data_ptr(), raw_v.data_ptr()])
        self.assertEqual(lens, [raw_k.nbytes, raw_v.nbytes])
        self.assertEqual(item_lens[0], PAGE_SIZE * LAYER_NUM * HEAD_NUM * HEAD_DIM * 4)
        self.assertEqual(item_lens[1], PAGE_SIZE * LAYER_NUM * HEAD_NUM * V_HEAD_DIM * 4)

    def test_cached_views_preserve_writes_and_moves(self):
        pool = make_pool(
            MHAPageFirstTokenToKVPool, v_head_dim=V_HEAD_DIM, start_layer=5
        )
        raw_buffers = pool.get_page_first_raw_buffers()
        self.assertIs(raw_buffers, pool.get_page_first_raw_buffers())
        raw_k, raw_v = raw_buffers
        src_locations = torch.tensor([4, 7, 12])
        dst_locations = torch.tensor([8, 11, 16])
        keys = (
            torch.arange(3 * HEAD_NUM * HEAD_DIM).reshape(3, HEAD_NUM, HEAD_DIM).float()
        )
        values = (
            torch.arange(3 * HEAD_NUM * V_HEAD_DIM)
            .reshape(3, HEAD_NUM, V_HEAD_DIM)
            .float()
        )
        for layer in range(LAYER_NUM):
            pool.set_kv_buffer(
                FakeLayer(5 + layer), src_locations, keys + layer, values + layer
            )
        pool.move_kv_cache(dst_locations, src_locations)
        for layer in range(LAYER_NUM):
            torch.testing.assert_close(raw_k[dst_locations, layer], keys + layer)
            torch.testing.assert_close(raw_v[dst_locations, layer], values + layer)
            torch.testing.assert_close(raw_k[src_locations, layer], keys + layer)
        # Offload/restore also needs to use the cached views of this allocation.
        saved = pool.get_cpu_copy(dst_locations)
        raw_k.zero_()
        raw_v.zero_()
        pool.load_cpu_copy(saved, src_locations)
        for layer in range(LAYER_NUM):
            torch.testing.assert_close(raw_k[src_locations, layer], keys + layer)
            torch.testing.assert_close(raw_v[src_locations, layer], values + layer)

    def test_clear_releases_cached_views_and_recreation_uses_new_storage(self):
        pool = make_pool(MHAPageFirstTokenToKVPool, v_head_dim=V_HEAD_DIM)
        k_ref, v_ref = weakref.ref(pool._k_raw), weakref.ref(pool._v_raw)
        pool._clear_buffers()
        gc.collect()
        self.assertIsNone(k_ref())
        self.assertIsNone(v_ref())
        pool._create_buffers()
        key = torch.ones(1, HEAD_NUM, HEAD_DIM)
        value = torch.full((1, HEAD_NUM, V_HEAD_DIM), 2.0)
        pool.set_kv_buffer(FakeLayer(1), torch.tensor([5]), key, value)
        raw_k, raw_v = pool.get_page_first_raw_buffers()
        torch.testing.assert_close(raw_k[5:6, 1], key)
        torch.testing.assert_close(raw_v[5:6, 1], value)

    def test_descriptors_do_not_share_mutable_lists(self):
        pool = make_pool(MHAPageFirstTokenToKVPool)
        infos = pool.get_page_level_buf_infos()
        expected = tuple(list(values) for values in infos)
        for values in infos:
            values.append(123)
        self.assertEqual(pool.get_page_level_buf_infos(), expected)

    def test_zero_layers_have_no_transfer_descriptors(self):
        pool = make_pool(MHAPageFirstTokenToKVPool, layer_num=0)
        self.assertEqual(pool.get_page_level_buf_infos(), ([], [], []))
        self.assertEqual(pool.get_page_first_raw_buffers()[0].numel(), 0)
        pool.move_kv_cache(torch.tensor([8]), torch.tensor([4]))


if __name__ == "__main__":
    unittest.main()
