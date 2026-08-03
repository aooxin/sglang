# SWA HiCache 跨流竞态崩溃：根因分析与修复记录

- **日期**: 2026-07-30
- **分支**: `v0.5.10-xm`（基于社区 sglang v0.5.10，SWA HiCache 自定义改动从 `v0.5.5-pro-online` 迁移而来）
- **崩溃基线 commit**: `3a4259bb60d31ea64498901a71b00c696325821a`
- **场景**: PD 分离部署的 prefill 节点，`--enable-hierarchical-cache` + gcache L3 + hybrid SWA（`mimo-v2.5-pro`，page_size=64）

---

## 1. 故障现象

prefetch 线程（`Thread-6, prefetch_thread_func`）反复触发 assert，进程崩溃，每隔数小时出现一次：

```
File ".../sglang/srt/managers/cache_controller.py", in prefetch_thread_func
    self.append_swa_only_host_mem_release(operation.swa_host_indices)
File ".../sglang/srt/managers/cache_controller.py", in append_swa_only_host_mem_release
AssertionError: Each page must be all-zero or all-valid, got [64, 48, 64, 64, 48]
```

变体包括：
- 混合页：`[64, 64, ..., 32, ..., 64]`（某页部分为 0）
- 全零页：`All-zero pages in pristine-alloc input, got [0, 0, 0, 0, 0]`
- 差 1 页：`[63, 64, 64, 64, 64]`（单个元素为 0）

**老分支 `v0.5.5-pro-online`（f2ba72a7）线上数月从未出现此崩溃。**

## 2. 关键日志证据（加诊断日志后）

### 证据 A：同一块显存两次读取结果不同（撕裂读实锤）

日志 (b)（TP5）：assert 计算出末页 `valid_count=56`，但**同一条日志**后半段 dump
`bad_pages_raw` 时读到的却是 64 个连续非零值 `885760..885823`。同一 GPU 内存，
毫秒间隔的两次读返回不同数据——第一次读时 `alloc()` 的 async `.clone()`
（排队在 scheduler 的 `schedule_stream`）尚未完成，第二次读时已完成。

日志 (c)（DP1 TP8）：zero 检测判定 5 页全零，但 `raw_head` dump 显示合理的
连续 slot 值 `254400..254463, 491456..`——同一现象。

### 证据 B：free ring 被持久污染

日志 (a)：混合页原始值为 `[0×32, 99, 360225..360255]`——32 个 0、一个游离值
99、再接连续段，是垃圾被写入 free ring 后再被 alloc 切出来的形态。

日志 (b)：`raw_head` 中出现 `885312+` 这类超出 SWA host 池实际容量的值——
full 池 slot 值或垃圾串进了 SWA ring。

## 3. 根因

三个因素叠加，缺一不可：

### 3.1 host 池索引簿记张量在 GPU 上（自定义分支独有）

`memory_pool_host.py`：
- `HostKVCache._free_slots_buf = torch.arange(..., device=self.device)`，
  `device` 由 `kv_pool.device` 传入 = `cuda`
- `SWATokenToKVPoolHost.full_to_swa_index_mapping = torch.zeros(..., device=device)` 同样在 CUDA

而**社区 v0.5.10** 的 `free_slots = torch.arange(size, dtype=torch.int64)` 在
**CPU** 上，`free()` 还显式 `indices.cpu()`——上游根本不存在此问题。GPU ring
实现来自 `4af13100d2`（Simple Radix Cache L2，从 v0.5.5 分支迁移）。

### 3.2 v0.5.10 引入 schedule_stream 全程包裹（v0.5.5 没有）

当前分支 `scheduler.py run_event_loop()`：

```python
self.schedule_stream = self.device_module.Stream(priority=0)
with self.device_module.StreamContext(self.schedule_stream):
    dispatch_event_loop(self)
```

`v0.5.5-pro-online` 中 disaggregation **PREFILL** 模式的事件循环没有任何
stream context——scheduler 线程与 HiCache 后台线程都跑在同一条 legacy
default stream 上。PyTorch 的 legacy default stream 是每设备一条、所有线程
共享的，所有 kernel 提交自然串行 → **老分支的安全不是运气，是隐式流同步**。
迁移到 0.5.10 后，代码没变，流语义变了。

### 3.3 跨线程交接只用 Python queue，不带 CUDA 依赖

Python 锁 / queue 只序列化 **kernel 提交**，不序列化 **kernel 执行**。
scheduler 线程（schedule_stream）与 prefetch/IO 线程（default stream）之间
传递 GPU 张量时没有任何 event/wait，产生撕裂读写。

### 3.4 为什么 `4d32a084` 修不好：反向路径污染 free ring

`4d32a084` 只 fence 了 **alloc → prefetch** 方向（且默认关闭，
`SGLANG_SWA_ALLOC_SYNC` 在 import 时缓存）。但**释放方向完全裸奔**：

```
prefetch 线程 (default stream):
  paged[valid_page_mask]              ← async gather
  → put 进 swa_only_host_mem_release_queue（纯 Python queue）

scheduler 线程 (schedule_stream):
  _drain_swa_only_release: torch.cat → free_swa_only → free()
  → 就地写 _free_slots_buf[tail:tail+n]  ← 可能读到 gather 未完成的垃圾/零
```

垃圾一旦写进 `_free_slots_buf`，**ring 被持久污染**：之后每次 `alloc()`
从污染区切出的"pristine"结果天然带零页/混合页，assert 在离原始竞态很远的
地方反复触发。这解释了 alloc 侧 fix 无效、以及证据 B 的全部形态。

`full_to_swa_index_mapping` 同理：prefetch 线程写
（`mapping_full_and_swa_indices`，default stream），scheduler 线程读写
（`free()` gather、`alloc_host`、`splice_swa_mapping`，schedule_stream），
无任何同步——第二条污染通道。

## 4. 修复方案对比

| 方案 | 说明 | 评估 |
|---|---|---|
| A. 簿记搬回 CPU | 对齐社区 v0.5.10：`_free_slots_buf`、`full_to_swa_index_mapping` 及其操作全部 CPU 化 | **结构性消除**问题；页粒度元数据操作在 CPU 上是微秒级，还省掉小 kernel launch。中期推荐 |
| B. 全交接点 event fence | 每个跨线程 GPU 张量交接补 CUDA event 协议 | 保持 GPU 簿记；固有脆弱性：未来新增交接点必须记得走协议。**本次采用** |
| C. 设备级 sync 止血 | 生产者发布前 `torch.cuda.synchronize()` | 正确但性能伤害大，仅适合定位验证 |

注意：消费侧调 `current_stream().synchronize()` 是**无效**的——它只等
调用线程自己的流，等不到生产者流上未完成的操作。正确姿势是
**生产者发布前 sync/record 自己的流**，或消费者 wait 生产者的 event。

## 5. 本次修复实施（方案 B）

### 5.1 `_CudaStreamFence`（memory_pool_host.py，新增）

"last-write event" 协议原语：

```python
with fence.order():
    ...  # 触碰受保护 GPU 张量的操作
```

进入时让当前线程的 stream `wait_event` 上一个操作记录的 event（可能来自
另一条 stream）；退出时在当前 stream 记录新 event。配合 Python 锁，形成与
锁获取顺序一致的全序跨流执行顺序。CPU 张量 / 无 CUDA 时自动 no-op。

### 5.2 修复的交接点

| # | 交接点 | 原状态 | 修复 |
|---|---|---|---|
| 1 | free ring `_free_slots_buf`（clear/alloc 的 async clone、free 的就地写+压缩） | 仅 RLock | `HostKVCache._ring_fence` 包裹全部 ring 读写（堵死 ring 污染根源） |
| 2 | alloc → prefetch（经 `prefetch_queue`） | event 默认关闭 | 无条件记录 `host_indices_ready_event`（CUDA 张量时）；**删除 `SGLANG_SWA_ALLOC_SYNC`** |
| 3 | release → drain 反向路径（**本次崩溃主因**） | 完全无 fence | `host_mem_release_queue` / `swa_only_host_mem_release_queue` 条目改为 `(page, ready_event)` 元组；producer 侧 `_record_release_ready_event()` 在 gather **之后**记录；drain 侧 `wait_event`（按 event id 去重）后再 `torch.cat`/`free` |
| 4 | `full_to_swa_index_mapping` 多线程读写 | 无锁无 fence | 新增 `_mapping_lock` + `_mapping_fence`；内部所有 read-modify-write（`free`/`unmapping_full_and_swa_indices`/`splice_swa_mapping`/`mapping_full_and_swa_indices`/`alloc_host`/`alloc_unmapped_host_by_len`/`clear`）进入临界区 |
| 5 | 外部 mapping 访问 | 裸索引 | 新增 `gather_swa_mapping()` / `scatter_swa_mapping()` fenced 助手；cache_controller 7 处、hiradix_cache 6 处、`get_page_buffer_meta` 全部改走助手 |
| 6 | hi_mamba_radix_cache `_drain_release` | — | 仅适配元组解包 + TODO（mamba 不使用，暂不 fence） |

### 5.3 顺带修复的逻辑点

- `SWATokenToKVPoolHost.free()`：mapping 的 gather 与清零原本分离在函数
  两端（中间夹着 `swa_pool_host.free`），已合并进同一 fence 临界区。
- swa release 的 event 原本可能在 `paged[valid_page_mask]` gather 之前
  记录——已修正为 gather 之后。

### 5.4 改动文件

```
python/sglang/srt/environ.py                     |   8 -   (删 SGLANG_SWA_ALLOC_SYNC)
python/sglang/srt/managers/cache_controller.py   |  43 +-  (release event 协议 + fenced 助手)
python/sglang/srt/mem_cache/hi_mamba_radix_cache.py |  9 +- (元组适配 + TODO)
python/sglang/srt/mem_cache/hiradix_cache.py     |  63 +-  (drain wait_event + fenced 助手)
python/sglang/srt/mem_cache/memory_pool_host.py  | 203 +-  (_CudaStreamFence + ring/mapping fence)
```

## 6. 验证与遗留

- 已验证：全部改动文件 `py_compile` / AST 通过；Pyright 报错均为文件既有
  类型标注问题（基类上访问 SWA 专属属性等），与本次改动同模式。
- **待验证（本地无 torch 环境）**：在 GPU 机器跑
  `pytest test/registered/unit/mem_cache/test_swa_token_to_kv_pool_host.py
  test/registered/unit/mem_cache/test_swa_unittest.py`；
  开 `SGLANG_SWA_DIAG_ENABLED=1` 压测，ring 完整性 / free 非法值 /
  splice post-check 应全部静默，原 assert 不再触发。
- 性能：每次受保护操作多一次 event record + wait_event（同流时 wait 近零
  开销），页粒度元数据操作可忽略。
- **固有脆弱性提醒**：未来任何新增的跨线程 GPU 张量交接（新队列、新后台
  线程）都必须走 fence/event 协议，否则复发。hi_mamba 的 TODO 即第一个
  待办。中期建议仍按方案 A 把簿记搬回 CPU，与社区对齐后可整体移除
  `_CudaStreamFence`。

## 7. 时间线备查

| 时间 | 事件 |
|---|---|
| 2026-05-29 | `4af13100d2` L2：GPU 驻留 free ring（`_free_slots_buf`）从 v0.5.5 分支迁入 |
| — | `13b64bb566` L3：引入 `append_swa_only_host_mem_release` / release 队列 |
| 2026-07-10 起 | 线上 prefill 节点开始出现 "all-zero or all-valid" assert |
| 2026-07-27 | `d848ce9d2f` 加 SWA-DIAG 诊断日志，捕获撕裂读/ring 污染证据 |
| — | `4d32a084` alloc 侧 event fix（默认关）——问题仍复现 |
| 2026-07-30 | 本次修复：全交接点 event fence（方案 B） |
