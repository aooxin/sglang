# SimpleSWARadixCache 当前实现说明

本文档描述当前 `HEAD` 上 prefill SWA 的实现语义，不按 commit 时间线展开。
重点回答四个问题：

1. `SimpleSWARadixCache` 在什么场景下会被启用；
2. 它和旧 `SWARadixCache` 的核心语义差异是什么；
3. allocator / scheduler / model runner 需要如何配合；
4. 现在应该看哪些测试来验证行为。

## 1. 适用范围

- `SimpleSWARadixCache` 只用于 **hybrid SWA + prefill disaggregation node**。
- scheduler 仅在 `self.is_hybrid_swa` 且 `disaggregation_mode == "prefill"` 时实例化它。
- decode 节点不会实例化它（scheduler 仅在 `disaggregation_mode == "prefill"` 才创建）；`ScheduleBatch.maybe_evict_swa()` 对 `SimpleSWARadixCache` 是 no-op（它只在 decode 路径或 `is_chunk_cache()` 路径下做事），因为它的真实缩窗发生在 chunk/finish 边界、而不是 extend 之前。
- 它不是“所有 hybrid SWA 模式下的通用默认 cache”，而是 prefill 侧的特化实现。

## 2. 核心不变量

### 2.1 full index 是统一口径，SWA index 可以被提前回收

- `req_to_token_pool`、radix tree、prefix match 看到的都是 full KV index。
- SWA 层真正访问的物理槽位通过 `full_to_swa_index_mapping` 映射得到。
- 一旦窗口外 SWA 被主动回收，对应 mapping 会被清成 `0`。
- `0` 表示“这个 full index 仍然存在，但已经没有 live SWA state”，后续会被当作 padding slot 处理。

### 2.2 unfinished request 不会把中间状态插入树

- `cache_unfinished_req()` 只做两件事：
  - 释放窗口外 SWA；
  - 保存 `prefix_indices`，供下一轮调度继续沿用。
- 真正的 tree insertion 只发生在 `cache_finished_req()`。
- 这样可以避免 chunked prefill 中途把“请求私有 KV”过早变成“树共享 KV”，减少 double free 和 ownership 混乱。

### 2.3 SimpleSWA 和旧 SWARadixCache 的差别

- 旧 `SWARadixCache` 依赖 tombstone / LRU 记账去维护 full 与 SWA 的生命周期。
- `SimpleSWARadixCache` 的核心思路是：**full KV 继续按普通 radix 复用，SWA KV 则在 prefill 过程中主动缩窗**。
- 因此它的 tree memory 语义、prefix 可复用语义、以及调度预算口径都和旧实现不同。

## 3. SWA 缩窗与前缀复用语义

### 3.1 `evict_swa()` 的 free 区间

- 可回收右边界是 `swa_free_end = max(0, prelen - sliding_window_size)`。
- 实际 free 起点是 `effective_start = max(req.swa_evicted_seqlen, tree_prefix_len)`。
- `req.swa_evicted_seqlen` 负责避免重复 free 同一段请求本地 SWA。
- `tree_prefix_len` 由 caller 传入 `req.cache_protected_len`，负责保护已经被 radix tree 共享的前缀，避免请求本地的缩窗逻辑误清共享前缀。

### 3.2 page size 大于 1 时必须 page-align

- 当 `page_size > 1` 时，`evict_swa()` 会把 free 的起止边界对齐到 page。
- 原因是 paged allocator 释放的是整页；如果 free 区间切到半页，会把还在使用的 token 一起放回 free list，后续容易触发 double free。
- 因此：
  - `swa_free_end` 向下对齐；
  - `effective_start` 向上对齐。

### 3.3 pre-extend 阶段不会对 SimpleSWA 做真实缩窗

- `alloc_for_extend()` 入口虽然会调用 `batch.maybe_evict_swa()`，但该方法对 `SimpleSWARadixCache` 是 no-op，等价于不在 pre-extend 阶段做真实缩窗。
- 原因是这个阶段请求还没有新的 `req_to_token` row：
  - carry-over chunked request 的旧 row 已在上一轮 chunk 边界释放；
  - prefix-hit 新请求的 `prefix_indices` 也可能只指向树共享 full index。
- 因此 pre-extend 阶段没有“可安全回收的请求私有 SWA row”。
- SimpleSWA 的真实缩窗统一发生在 `cache_unfinished_req()` / `cache_finished_req()`，也就是 forward 完成之后、row 仍归当前请求所有的时候。

### 3.4 `match_prefix()` 要同时满足 token path 命中和 SWA tail-window 有效

- `RadixCache.match_prefix()` 只知道 token path 和 full index。
- `SimpleSWARadixCache.match_prefix()` 会进一步检查命中前缀尾部窗口的 SWA mapping 是否全部仍然有效。
- `_compute_safe_prefix_len()` 返回“尾部窗口全为 live SWA”时允许复用的最大前缀长度。
- 这避免了“树上还能命中 full index，但对应 SWA 已被提前缩窗回收”的伪命中。
- EAGLE 模式下还会额外把 `speculative_num_steps` 加回 token 长度口径。

### 3.5 `cache_finished_req()` 的顺序

- finished 路径先按窗口语义回收窗口外 SWA；
- 再调用 `RadixCache.cache_finished_req()` 完成 tree insertion 和请求清理；
- allocator 的 `free_swa()` 会过滤已经为 `0` 的 mapping，所以 finished 路径允许重复经过同一段 full index，而不会因为 SWA 已回收再次报错。

## 4. Tree Eviction 与 allocator 安全机制

### 4.1 eviction 按 full / SWA 两本账推进

- `SimpleSWARadixCache.evict(params: EvictParams) -> EvictResult` 通过 `params.num_tokens` 和 `params.swa_num_tokens` 同时驱动 full / SWA 两路驱逐，分别统计：
  - full 可回收量：leaf 的 `value` 长度;
  - SWA 可回收量：`value` 中仍然映射到 live SWA slot 的数量。
- 这样可以处理“full 还能回收很多，但 live SWA 其实已经不多”的情况。

### 4.2 evictable / protected 也分 full / SWA 统计

- full 的 evictable / protected 沿用 radix tree 的节点大小。
- SWA 的 evictable / protected 则通过遍历树节点并检查 live mapping 数量得到。
- scheduler 的 usage 统计和预算判断依赖这个区分，不能再把 `tree_cache.evictable_size()` 直接同时当成 full 和 SWA 的可回收量。

### 4.3 allocator 当前的安全语义

- `free_swa()` 是幂等的：只释放 `mapping > 0` 的 live SWA slot，然后把 mapping 清成 `0`。
- `free()` 不是幂等的：full allocator 和 paged allocator 都带 bitmap 检查，重复 free / 重复 alloc 会直接报错。
- 这两个语义需要区分：
  - “窗口外 SWA 重复缩窗”是允许的；
  - “同一个 full slot 被重复归还”是不允许的。

### 4.4 paged extend 的预算修正

- paged allocator 现在用 `get_num_new_pages()` 计算 extend 真实需要的新页数；
- 不再使用旧的 `extend_num_tokens + bs * page_size` 过度估算；
- 这解决了 page-aligned extend 下的 false OOM 问题。

## 5. 调度与容量模型

### 5.1 `PrefillAdder` 里有两种不同的剩余预算口径

- `rem_total_tokens` 用来判断“整体还能不能再塞新请求”，对 hybrid SWA 必须看 full / SWA 双池共同约束，因此取两者的 `min(...)`。
- `cur_rem_tokens` 更偏向“当前这一次 prefill chunk 还能推进多少”。
- 对 `SimpleSWARadixCache`，因为 chunk 边界会主动缩窗，当前 chunk 的 admission 不必再按“整个剩余请求都占着 SWA”去保守估算，所以这里更靠近 full 侧当前可分配量。

### 5.2 chunk truncation 现在依赖真实双池预算

- `_get_effective_chunk_trunc_len()` 会取：
  - `rem_chunk_tokens`；
  - 当前真实可推进预算 `rem_total_tokens`；
  - 两者的较小值，再按 page size 对齐。
- `_should_chunk_simple_swa_req()` 的语义是：
  - 整条请求本轮塞不下；
  - 但裁成一个安全 chunk 后仍然能推进。

### 5.3 prefill 模式下的 SWA pool 估算已经更新

- `ModelRunner` 在 prefill + chunked prefill 场景下，不再把 SWA 峰值按“整段请求长度”估算。
- 当前估算更接近：
  - 树中缓存前缀保留的窗口尾部；
  - 当前 batch 正在 forward 的 chunk；
  - 再乘上 `SGLANG_SWA_PREFILL_HEADROOM_FACTOR` 预留 headroom。
- 没有 chunked prefill 时会回退到更保守的 full / SWA 均分策略。

### 5.4 当前相关环境变量

- `SGLANG_SWA_EVICT_LENGTH_PER_STEP`
  - decode 侧 SWA 峰值估算使用。
- `SGLANG_SWA_PREFILL_HEADROOM_FACTOR`
  - prefill 侧 SWA pool 预留倍数。

## 6. 当前建议看的测试

- `test/registered/unit/mem_cache/test_simple_swa_radix_cache.py`
  - 核心 cache 语义：`evict_swa()`、safe prefix match、unfinished / finished 流程、tree eviction、EAGLE 行为。
  - 通过模块顶层 `register_cuda_ci(suite="stage-b-test-1-gpu-small")` 自动进 CI。
- `test/registered/unit/mem_cache/test_simple_swa_allocator.py`
  - allocator / common 层回归：`free_swa()` 与 `free()` 的安全语义、paged false OOM 回归、`alloc_for_extend()` 中 prefix-hit 且 `req_pool_idx is None` 的共享前缀保护。
  - 同样以 `register_cuda_ci(suite="stage-b-test-1-gpu-small")` 注册。
- `test/manual/test_swa_cache_serving.py`
  - 手工压测脚本，不属于 CI 单测；放在 `test/manual/` 下，需先起 server 再手动跑。

## 7. 如何验证

### 7.1 前置条件

- 使用能正常导入 `torch` 的 Python 环境。
- 机器需要能看到 CUDA 设备，因为这两组 SWA 单测都直接创建 GPU KV pool。
- 建议在仓库根目录下运行。

可以先做两个快速检查：

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -m py_compile \
    test/registered/unit/mem_cache/test_simple_swa_radix_cache.py \
    test/registered/unit/mem_cache/test_simple_swa_allocator.py
```

### 7.2 想测什么，就跑什么

| 目标 | 推荐命令 | 说明 |
|---|---|---|
| 核心 cache 语义全量回归 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_radix_cache` | 覆盖 `evict_swa()`、safe prefix match、unfinished/finished、tree eviction、EAGLE |
| 只看 `evict_swa()` 边界与 page 对齐 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_evict_swa_boundary_cases test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_evict_swa_page_aligned_when_paged test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_evict_swa_respects_tree_prefix_len` | 适合改了 `evict_swa()` 或 page 对齐逻辑后快速回归 |
| 只看 prefix safety 语义 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_compute_safe_prefix_len_cases test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_match_prefix_requires_swa_tail_window` | 适合改了 `match_prefix()` 或 `_compute_safe_prefix_len()` 后验证 |
| 只看 unfinished / finished 生命周期 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_cache_unfinished_req_chunked_updates_prefix_and_frees test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_cache_unfinished_req_non_chunked_no_tree_insert test.registered.unit.mem_cache.test_simple_swa_radix_cache.TestSimpleSWARadixCache.test_cache_finished_req_frees_swa_and_inserts_tree` | 适合改了 request 生命周期、prefix 保存或 finished insertion 后验证 |
| allocator / common 回归全量 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_allocator` | 覆盖 `free_swa()` 语义、paged false OOM、`alloc_for_extend()` 共享前缀保护 |
| 只看 paged false OOM 回归 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_allocator.TestSimpleSWAAllocator.test_alloc_extend_page_size64_no_false_oom` | 适合改了 paged allocator 或 `get_num_new_pages()` 相关逻辑后验证 |
| 只看 prefix-hit 且 `req_pool_idx is None` 分支 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_allocator.TestSimpleSWAAllocator.test_alloc_for_extend_prefix_hit_without_req_pool_idx_keeps_shared_prefix` | 适合改了 `alloc_for_extend()` 共享前缀保护逻辑后验证 |
| CI 当前覆盖的 SimpleSWA 单测集合 | `python3 -m unittest test.registered.unit.mem_cache.test_simple_swa_radix_cache test.registered.unit.mem_cache.test_simple_swa_allocator` | 两份测试通过模块顶层 `register_cuda_ci(suite="stage-b-test-1-gpu-small")` 自动被顶层 `test/run_suite.py` 收录 |

### 7.3 手工 serving / stress 验证

- 如果你要验证真实服务行为，而不是内存语义单测，使用：
  - `test/manual/test_swa_cache_serving.py`
- 这部分不在 CI 中，定位是开发期手工压测。
- 常见用途：
  - 验证 prefix cache hit 行为；
  - 验证长上下文 chunked prefill 下的 SWA 回收；
  - 验证高压下 eviction / speculative decode 是否稳定。

## 8. 阅读顺序建议

如果你要从代码入口往下读，推荐顺序如下：

1. `python/sglang/srt/managers/scheduler.py`
   - 看什么时候会实例化 `SimpleSWARadixCache`。
2. `python/sglang/srt/mem_cache/simple_swa_radix_cache.py`
   - 看缩窗、prefix safety、finished insertion 的核心语义。
3. `python/sglang/srt/mem_cache/common.py`
   - 看 extend 分配前如何处理共享前缀、如何做双池 shortage 判断。
4. `python/sglang/srt/mem_cache/allocator.py`
   - 看 mapping 清零、double-free 防护、paged extend 预算修正。
5. `python/sglang/srt/managers/schedule_policy.py`
   - 看 `rem_total_tokens` / `cur_rem_tokens` / chunk truncation 的预算逻辑。
6. `python/sglang/srt/model_executor/model_runner.py`
   - 看 prefill / decode 两侧的 pool sizing 模型。
