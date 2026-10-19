# DFlash Hybrid-SWA 双窗口 KV Cache 实现梳理

> 参考模型：[XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash)
> 基线 tag：`v0.5.10-xm`（`019a66028a5bec4a4e2f0e025b438765c9de146f`）
> 规模：9 commits（按模块渐进式整理）、42 files、约 +4535 / −595 行

### commit 分层（渐进式，依赖自上而下）

| # | commit | 层 |
|---|---|---|
| 1 | `feat(dflash): plumb independent draft SWA window configuration` | 配置基础：draft 窗口配置、cache-init 字段、eviction 结果字段 |
| 2 | `feat(dflash): add independent draft SWA device KV pool (L1)` | L1 设备池：第三套子分配器 + 独立映射 |
| 3 | `feat(dflash): draft-window-aware prefill radix caching` | prefill：SimpleSWARadixCache 双窗口淘汰/保护/安全长度 |
| 4 | `feat(dflash): independent draft SWA HiCache transfer lifecycle (L2/L3)` | L2/L3：host cache + gcache prefetch/writeback |
| 5 | `feat(dflash): draft SWA PD disaggregation transfers` | PD 分离：decode D2H + pd_direct2host + conn 后端 |
| 6 | `feat(dflash): draft safe-length handling in speculative decoding` | spec worker：draft 接受长度计算 |
| 7 | `feat(dflash): draft SWA scheduling and observability` | 调度 + metrics |
| 8 | `docs(dflash): document and benchmark two-window SWA` | 文档 + benchmark 脚本 |
| 9 | `chore(dflash): xiaomi docker env tweaks` | 构建环境 |

---

## 1. 背景与动机

DFlash 是一种投机解码（speculative decoding）方案：一个轻量 **draft** 模型先预测若干 token，**target** 模型再一次性验证。在 **hybrid-SWA**（sliding window attention，滑动窗口注意力）模型上，target 和 draft **各自的滑动窗口大小（`swa_window_size`）不同**：

- Target 模型：全量 full 层 + 自己的 SWA 层，窗口 = `sliding_window_size`
- Draft 模型（`DFlashDraftModel`）：**每一层都用窗口注意力**，窗口 = `dflash_config.swa_window_size`

由于窗口不同，两者对 KV cache 的保留需求也不同。若共用一套 SWA pool，就必须按两者的**并集/最大值**保留，造成显存与 L2/L3 传输的浪费。

### ⚠️ 关键语义：draft 窗口不是严格意义的 SWA

这一点决定了后面 §3.2 前缀复用安全长度的判定逻辑，务必先理解：

**严格 SWA（target 侧）**：KV 是一份**从序列位置 0 连续累积**的不可变缓存。要复用长度 `L` 的前缀，窗口 `[L-window, L)` 内的 KV slot 必须**全部存活**——任一 slot 被 evict/置零，该 `L` 就不安全（这正是 §3.2 `compute_swa_safe_len` 的判定）。

**draft 侧不是这样**。draft 的 KV **不是从头累积的不可变前缀**，而是**由 target hidden states 推导/重算**出来的：

- draft 前向的输入是 `input_embeds`，不是 target 的 KV，也不是某份历史 draft KV（`dflash.py:427-441`）。
- draft KV slot 的内容由 worker 在 target verify 时捕获的 target aux hidden states，经 `project_target_hidden` 投影后**物化**写入（`_append_target_hidden_to_draft_kv_by_loc` → `FusedKVMaterializeHelper.materialize`，`dflash_worker_v2.py:1061-1075/1184`）。
- worker 会**按需重建 draft-local 的窗口视图**：`suffix_start = prefix_lens - draft_prefix_lens`，只把当前可见的、长度约 `window_size` 的 suffix 从 target 的 `req_to_token` gather 进 draft 映射（`dflash_worker_v2.py:1718-1752`）。

**结论**：draft 只要能拿到「**window_size 个 full 的 hidden states**」就能算它的那一段——它**不要求 KV 从位置 0 连续存在**，任意 window_size 长度的可见上下文即可。所以 draft 的"窗口"是**注意力读取范围的约束**，而**不是**"必须从头连续保留的 KV 前缀"。

这带来两个直接后果，贯穿全篇：

1. **前缀复用的 draft 豁免**（§3.2 的 `draft_free`）：当复用前缀离请求尾部足够远（`total_len - L > draft_window`）时，draft 窗口根本够不到这段前缀，于是**无论 draft slot 是否存活都安全**——因为需要时可从 target HS 重算。这在严格 SWA 上是不成立的。
2. **draft 无 refill/rewrite 路径**（§9.1）：draft slot 缺失不需要"回填历史 KV"，缺的部分在 draft 前向时由 target HS 重新物化即可；代价只是那一段要重算，不影响正确性。

### 老实现（单窗口 / 保留窗口妥协）

基线代码走的是 `swa_retention_window = max(target_window, draft_window)`：

- Draft 与 target **共用同一 SWA slot 空间**和同一张 `full_to_swa_index_mapping`
- 为同时满足两窗口，保留窗口被撑大到两者较大值
- **代价**：draft 窗口偏大时，整个 SWA pool 全按大窗口保留 → 显存浪费 + L2/L3 冗余传输

### 新实现（双窗口 / 独立池）

本分支将 draft SWA 拆成**第三个独立子分配器 + 独立映射表**，target 与 draft 的窗口各自精细管理，`alloc / evict / free / match` 全链路分离，并贯通 **L1(device) → L2(host) → L3(gcache) → PD 传输** 四层。

---

## 2. 数据结构核心：三套子分配器

`SWATokenToKVPoolAllocator`（`mem_cache/swa_memory_pool.py`）从管 2 个子池变为管 **3 个**：

| 子分配器 | 映射表 | 覆盖对象 | 窗口 |
|---|---|---|---|
| `full_attn_allocator` | — | target full 层 | 全量 |
| `swa_attn_allocator` | `full_to_swa_index_mapping` | target SWA 层 | `sliding_window_size` |
| **`draft_swa_attn_allocator`**（新增） | **`full_to_draft_swa_index_mapping`**（新增） | draft 全 SWA 层 | `draft_swa_window_size` |

关键设计（`swa_memory_pool.py:320-446`）：

- 两套映射表都以**同一 full 索引空间**为 key（即 `req_to_token` 里的统一 loc），但映射到**不同的物理 slot 环**。
- `register_draft_swa(draft_pool, target_window, draft_window)` 在 draft worker 初始化时建立第二套映射，返回 `full_to_draft_swa_index_mapping`。
- **仅当 `draft_window != target_window` 时启用**（`dflash_worker_v2.py:346` 的 `use_independent`）；相等时退回共享映射（单窗口）。
- 新增字段：`draft_swa_attn_allocator` / `full_to_draft_swa_index_mapping` / `draft_swa_window_size` / `_size_draft_swa`（`:320-323`）。

---

## 3. 分层改动详解

### 3.1 分配器层 —— 双窗口的物理基础
`mem_cache/swa_memory_pool.py`

每个操作对三个池分别处理，任一不足即失败：

| 方法 | 行号 | 双窗口行为 |
|---|---|---|
| `alloc(need_size)` | `:448` | 三池都够才分配，任一不足返回 `None`；draft 用独立 alloc + 更新 draft 映射 |
| `alloc_device(...)` | `:489` | HiCache load-back，返回 `(full_idx, lack_full, lack_swa, lack_draft_swa)` 三路缺口 |
| `alloc_decode(...)` | `:802` | decode 时 draft 用 `translate_loc_from_full_to_draft_swa(last_loc)` 独立算 tail |
| `free(free_index)` | `:845` | 依次 free full / swa / draft_swa |
| `free_draft_swa(free_index)` | `:896` | 幂等：只释放 `>0` 的 live slot，之后 mapping 清零 |
| `translate_loc_from_full_to_draft_swa` | `:444` | draft 专属 full→draft-swa 地址翻译 |
| `draft_swa_available_size()` | `:344` | draft 池可用容量查询 |
| `splice_swa_mapping(...)` | `:911` | 前缀所有权转移时同步 draft 映射 |

### 3.2 Radix Cache / 淘汰策略层 —— 双窗口的正确性核心
`mem_cache/simple_swa_radix_cache.py`

**核心是前缀复用安全长度的联合判定** `compute_target_and_joint_safe_len`（`:113`）。要精确理解它,先看两个底层原语和单窗口版本:

#### (a) 逐位置窗口有效性 `_window_ok`（`:57`）—— 不是只看边界

判定「对候选前缀长度 `L`,窗口 `[L-window, L)` 内是否**全部存活**」。用 zero-slot 的前缀和做**逐位置**判定,一次性对所有 `L` 向量化:

```python
def _window_ok(indices, window, lens):          # indices: 0=freed/invalid, >0=live
    bad        = (indices == 0).to(int32)         # 每个被 evict/置零的 slot 记 1
    prefix_bad = pad(cumsum(bad), (1, 0))         # bad 的前缀和(前面补 0)
    starts     = clamp(lens - window, min=0)      # 每个 L 的窗口左端 = max(0, L-window)
    return (prefix_bad[lens] - prefix_bad[starts]) == 0   # 窗口内 bad 计数为 0 才 True
```

关键:它检查**窗口区间内每个位置**有没有洞(靠区间前缀和之差),**不是只看边界那一个 slot**。`_last_valid_len(valid)`（`:71`）再取 `valid` 为 True 的**最大** `L`。

#### (b) 严格 SWA 单窗口 `compute_swa_safe_len`（`:78`）—— target 侧

```python
matched_len = len(swa_indices)
tail_start  = max(0, matched_len - sliding_window_size)
if swa_indices[tail_start:].all():        # fast path: 尾窗全存活 → 整段前缀可复用
    return matched_len
valid = _window_ok(swa_indices, sliding_window_size, lens=arange(1, matched_len+1))
return _last_valid_len(valid)             # slow path: 取最大安全 L
```

这是**严格 SWA** 语义:窗口内任一 slot 被淘汰,该 `L` 即不安全(KV 是从 0 累积的不可变前缀,不能重算)。

#### (c) 联合双窗口 `compute_target_and_joint_safe_len`（`:113`）—— target ∩ draft

> **命名更新**:文档早期版本记录的 `compute_joint_safe_len` 已重构为 `compute_target_and_joint_safe_len`，返回 `(target_safe_len, joint_safe_len)`（`target_safe_len` 供 HiCache refill gate）。
>
> **⚠️ 重要纠正(fixed-edge)**:早期实现对 draft **误用了和 target 相同的滑动窗口 `[L-draft_window, L)`**,这是 **bug**。draft 的正确语义是**固定左端** `[s, L)`,其中 `s = max(0, total_len - draft_window)` 是 draft KV 的**物化边界**。详见下方 (c.1)。随机对拍显示两者在约 **15%** 的用例上结果不同。

**关键区别:target 用滑动窗口,draft 用固定左端。**

- **target(严格 SWA,滑动)**:候选 `L` 的窗口是 `[L-target_window, L)`,**左端随 L 滑动**。窗口内任一 slot 被 evict → 该 `L` 不安全。非单调(洞可滑出窗口,更大的 L 又变安全)。
- **draft(固定左端)**:候选 `L` 的可读 draft slot 是 `[s, L)`,**左端固定在 `s = max(0, total_len - draft_window)`**。因为 prefill 只物化了 `[s, total_len)` 这最后 `draft_window` 个 token 的 draft KV,`s` 之前根本没有 draft KV(需要时由 target HS 重算,故那里的洞无所谓)。

```python
def _draft_safe_fixed_edge(draft_indices, draft_window_size, total_len, n):
    s = max(0, total_len - draft_window_size)     # 物化边界(固定左端)
    if s >= n:
        return n                                  # 整段匹配前缀都在边界之前 → 无约束
    # [s, n) 内第一个 dead slot 截断; 因左端固定 → 谓词单调(连续前缀)
    dead = nonzero(draft_indices[s:n] == 0)
    return n if dead.numel() == 0 else s + int(dead[0])

def compute_target_and_joint_safe_len(target_indices, target_window_size,
                                       draft_indices, draft_window_size, total_len):
    n = len(target_indices)
    target_valid    = _window_ok(target_indices, target_window_size, arange(1, n+1))  # 滑动
    target_safe_len = _last_valid_len(target_valid)
    draft_safe      = _draft_safe_fixed_edge(draft_indices, draft_window_size, total_len, n)
    # draft 谓词单调 ⟹ 接受集恰为 [0, draft_safe] ⟹ joint = 框内最长 target-safe 前缀
    joint_safe_len  = _last_valid_len(target_valid[:draft_safe])
    return target_safe_len, joint_safe_len
```

**为什么固定左端是对的?** —— §1 的语义差异 + 物化边界:
- target KV 从 0 连续累积、不可重算 → 严格滑动窗口,窗口内必须真存活。
- draft KV 只物化了尾部 `[s, total_len)` 一段;对复用点 `L`,draft 实际能读的是 `[s, L)`。`s` 之前没物化,靠 target HS 重算 → 那里的洞与 draft 判定无关。用滑动 `[L-dw, L)` 会错误地去检查 `s` 之前没物化的区域,导致前缀被误缩(这正是原 bug)。

#### (c.1) fixed-edge 的单调性、区分性反例,与「先算 dflash 框再算 swa」

**单调性**:固定左端 ⟹ 一旦 `[s, N)` 内遇到第一个洞(位置 `h ≥ s`),所有 `L > h` 的框 `[s, L)` 都含该洞 → 全部不安全。故 draft 接受集恰为**连续前缀** `[0, draft_safe]`,`draft_safe = [s, n)` 内第一个洞位置(否则 `n`)。**这正是「dflash 判定结果恒为 `input[0:xx]` 无空洞」的由来。**

**区分性反例**(滑动 vs 固定左端结果不同):`draft=[1,1,1,1,1,0,1,0,1,1]`, `dw=3`, `total_len=10`,`s=7`:
- **固定左端(正确)**:`[7,8)=[0]` → 第一个洞在 index 7 → `draft_safe=7`。
- **旧滑动(bug)**:`L=7` 窗口 `[4,7)=[1,0,1]` 含 index 5 的洞 → 不安全;`L=6` 走旧 draft_free → `draft_safe=6`。**错误地被 `s` 之前的洞(index 5)缩短。**

**「先算 dflash 框、再算 swa」现在正确且等价**:因为固定左端 draft 谓词**单调**(接受集 = `[0, draft_safe]`),所以 `joint = max{L : target_ok(L) ∧ draft_ok(L)}` 精确等于 `max{L ≤ draft_safe : target_ok(L)}` = `_last_valid_len(target_valid[:draft_safe])`。这正是你要的两步式,且经 Oracle 证明 + 300000 随机用例 oracle 校验(含"先框后 target == 逐 L 求交"等价性)。

> 注:早期基于**滑动窗口**的 `draft_free` 短路(`extend > dw`)在 fixed-edge 下被 `s >= n` 分支自然涵盖(`s = total_len - dw >= n` 即 `extend >= dw`,`_draft_safe_fixed_edge` 直接返回 `n`,不读 draft) —— 性能收益保留,语义更正确。

#### (c.2) prefill 侧 dflash KV 的补齐范围:`[-draft_window:]`

prefill 侧**补齐/物化** dflash KV 只需 **`[total_len-draft_window, total_len)` = `[-draft_window:]`**(= `min(input_len, 1024)` 的由来):draft 每层窗口 = `draft_window`,处理完整序列时 draft 前向真正要读的只有最后 `draft_window` 个 token。更早的 token draft 窗口够不到,需要时由 target HS 重算。这正是 PD 传输切片 `floor_align(seq_len - draft_swa_window_size)`(`prefill.py:955`）发送的范围,也是 fixed-edge 判定中 `s = total_len - draft_window` 这个**固定左端**的物理来源 —— **补齐边界与判定左端是同一个 `s`**。

#### 命中率影响指标

因 draft 约束比 target 更严导致的前缀收缩量 = `target_safe_len - joint_safe_len`,累加到**统一**计数器 `draft_swa_prefix_shrink_tokens` → `sglang:draft_swa_prefix_shrink_tokens_total`。两条路径都累加到它:PD-prefill 走 `SimpleSWARadixCache._compute_safe_prefix_len`（`:867-904`）,HiCache(L2/L3) 走 `HiSimpleSWARadixCache._compute_safe_prefix_len`（`hiradix_cache.py:2662-2820`);均经 scheduler-delta 导出。
- HiCache 里只有 `target_safe_device_len` 能触发 target-SWA refill;**纯 draft 约束造成的收缩无法靠 refill target SWA 修复**,且 `draft_safe_device_len = compute_draft_safe_len(device_draft, draft_window_size, total_len)` 用来 **cap** refill 上限(因为 draft 无 refill 路径,见 §9.1)。

新增 draft 专属逻辑：

| 方法 / 配置 | 行号 | 作用 |
|---|---|---|
| `_load_draft_swa_reserved_config` | `:217` | 解析 `SGLANG_DFLASH_DRAFT_SWA_RESERVED_*` env |
| `_evict_draft_swa` | `:529` | draft 窗口外主动缩窗释放 |
| `_filter_draft_swa_free_slots_before_free` | `:556` | draft reserved-point 过滤 |
| `_count_live_draft_swa_tokens` | `:597` | draft 映射的 live token 计数 |
| 计数器 | `:315` | `draft_swa_active_freed_tokens` / `draft_swa_evicted_from_tree` |

### 3.3 调度层 —— 双窗口的准入预算
`managers/schedule_policy.py` / `managers/schedule_batch.py`

- `PrefillAdder` 新增整套 `draft_swa_*` 预算维度：`rem_draft_swa_total_tokens`、`cur_draft_swa_rem_tokens`（`schedule_policy.py:621-680`）。
- 作为**独立准入约束**：即使 target 池够，draft 池不够也返回 `NO_TOKEN`（`:662`, `:1050`）。
- decode 淘汰触发从 `live_len >= 2*window` 改为更精细的 `live_len >= window + evict_len_per_step`（`schedule_batch.py:2593`）。
- draft 走独立淘汰路径 `_maybe_evict_draft_swa`（`speculative/dflash_info_v2.py:132`），在 `prepare_for_decode` 中于 target 淘汰后调用。

### 3.4 HiCache L2 (host) 层
`managers/cache_controller.py` / `mem_cache/memory_pool_host.py`

- Draft host pool 有自己的 SWA 环与映射：`register_draft_swa_host()`（`memory_pool_host.py:2720`）。
- Target 的 `free()` **级联释放** draft host slot（`memory_pool_host.py:2949`，`free_swa_by_full_indices`）。
- 独立异步释放队列 `dflash_only_host_mem_release_queue` + `append_dflash_only_host_mem_release`（`cache_controller.py:2016`），带严格的 domain / page-uniform 校验。
- `draft_independent_swa` 属性（`cache_controller.py:1134`）作为 HiCache 层的 draft 开关判据：判定 device 侧存在独立 draft SWA 分配器；在 HiCache 语境内额外 `assert` `is_hybrid_swa` 且 `mem_pool_host_draft is not None`（device 独立 ⟹ host 独立,不成立则 fast-fail 而非静默降级）。
- L2 backup/load 对 draft 用独立映射构建拷贝参数（`_build_draft_swa_args`, `cache_controller.py:1214`）。

### 3.5 HiCache L3 (storage) 层
`managers/cache_controller.py` / `mem_cache/storage/gcache/gcache_storage.py`

- **只有 gcache 后端支持 draft L3**（`draft_dflash_l3_enabled`, `cache_controller.py`）；其它后端**直接 `raise NotImplementedError`（fail-fast，不再静默降级 L2-only）** —— 部署时须在非 gcache 后端上显式关闭 draft L3。generic 后备路径及其 `_draft_page_get/set_generic` 已删除。
- Draft 用独立 key 命名空间：`_convert_hybrid_keys(..., draft=True)` 加 `d:` 前缀（`gcache_storage.py:622`）。
- Draft 零拷贝方法：`batch_set_v1_draft` / `batch_get_v1_draft` / `batch_exists_draft`（`gcache_storage.py:373/464/474`）。

> **✅ 已补齐（原为 "base" 阶段缺口）**：L3 draft 的**写入/注册 + 读回路径现已完整接线**。prefetch 路径新增 `_resolve_draft_l3_prefetch`（`cache_controller.py`）：探测 `batch_exists_draft` → TP `all_reduce(MIN)` 同步 → `_compute_safe_storage_prefix` 算 draft 安全前缀（cap 到 target 命中集）→ `mem_pool_host_draft.mapping_full_and_swa_indices` 映射 draft host 槽位；`_page_transfer` 对 two-window 只对已映射前缀发 `batch_get_v1_draft`（传 full host indices，gcache 内部经 draft 映射 gather）。详见 §9.1(3)。

### 3.6 PD 分离传输层
`disaggregation/{base,common,mooncake}/conn.py` / `disaggregation/{prefill,decode}.py`

Draft SWA state 作为**独立传输组** `draft_state_*`（不折叠进 target 的 `state_*` 组）：

- `KVArgs` 新增 `draft_state_data_ptrs / _lens / _item_lens`（`base/conn.py`）。
- Prefill 端只发送 **draft 窗口切片**：`window_start = floor_align(seq_len - draft_swa_window_size, page_size)`（`prefill.py:955`），并用 `translate_loc_from_full_to_draft_swa` 翻译。
- Decode 端对称接收：`_get_draft_state_indices`（`decode.py:701`）。
- Mooncake 新增 `maybe_send_draft_state`（独立 buffer 指针 msg[16]/[17]，`mooncake/conn.py`）。
- 两端配置不一致（一端双窗口、一端单窗口）**显式报错**。

**PD direct-to-host（P2D 直传 host）现已支持 draft**（`disaggregation/pd_direct2host_manager.py`）：

- `PDDirect2HostManager` 为 draft 构建独立 host transit 池 `transit_draft_state_pools` / `transit_draft_state_pool`（仅当 allocator 暴露 `draft_swa_attn_allocator` 且 target 也有 SWA state 时启用，否则显式 `NotImplementedError`）。
- `pre_alloc` 按 draft 窗口切片分配 `host_draft_state_indices`：`draft_window_start = floor_align(seq_len - draft_swa_window_size, page_size)`。注意 `req_to_token` 行的分配**仍在 `start_h2d` 完成**（沿用 v0.5.10-xm 的延迟分配模型，`pre_alloc` 不再分配 req 行），避免 double-alloc。
- H2D 准入快照 `h2d_admission_snapshot` 现为 **8 元组**：`(full_available, swa_available, draft_swa_available, reserved_decode_tokens, retract_in_progress, h2d_gate_busy, req_rows_available, has_retracted_reqs)`；`can_admit_h2d_from_snapshot` 三路 need（`need_full/need_swa/need_draft_swa`）分别校验，任一不足即拒绝准入。
- decode 侧 `_get_draft_state_indices` 在 `pd_d2h_mgr` 路径下从 `req.host_draft_state_indices` 产出 draft page indices（不再返回 `None`）。

### 3.7 显存 sizing（节省的量化来源）
`model_executor/model_runner_kv_cache_mixin.py`

- `_resolve_draft_swa_tokens`（`:1001`）：draft 池物理大小 = target SWA 池 × `draft_window / target_window`，上限 `full_tokens`。
  - **这就是"节省"的量化**：双窗口下 draft 池仅按窗口比例分配，而非老实现的全量最大窗口。
- `get_draft_cell_size_per_token`（`:146`）：`window_ratio = draft_window / target_window`（`:197`）用于 profile 阶段的容量规划，与上式一致。

### 3.8 模型 & 配置
`configs/model_config.py`

- `DFlashDraftModel` 注册为 hybrid-SWA 架构（`:1527`）。
- `use_swa` 时所有 draft 层归为 SWA 层、无 full 层（`get_hybrid_layer_ids`, `:1560`）。

---

## 4. 观测性

### 新增 Prometheus 指标（`observability/metrics_collector.py`）

**Pool / tree 用量（Gauge）**
- `sglang:draft_swa_token_usage`
- `sglang:draft_swa_available_size`
- `sglang:draft_swa_tree_evictable_tokens` / `sglang:draft_swa_tree_protected_tokens`
- `sglang:host_draft_swa_available_size`（L2）
- `sglang:dev_draft_swa_tokens[_locked]` / `sglang:host_draft_swa_tokens[_locked]`（tree token 分解，§9.0.1c）

**Counter**
- `sglang:draft_swa_active_freed_tokens_total` / `sglang:draft_swa_evicted_from_tree_total`
- `sglang:draft_swa_prefix_shrink_tokens_total`（**统一**指标：draft 窗口约束导致的前缀收缩，PD-prefill 与 HiCache 两条路径累加到同一计数器；旧名 `hicache_dflash_prefix_shrink_tokens_total` 已废弃）
- `sglang:storage_draft_swa_hit_tokens_total`（draft L3 storage 命中，§9.0.1b）

**Histogram**
- `sglang:draft_l3_resolve_latency_seconds`（draft L3 prefetch 解析耗时）
- `sglang:draft_prefetch_io_latency_seconds`（draft L3 get IO 耗时）

### 调试开关
- `SGLANG_DFLASH_KV_DEBUG=1`：draft forward 前 dump draft prefix KV，验证 L2 回填是否真实恢复 draft KV（`dflash_worker_v2.py`）。
- `SGLANG_DEBUG_MEMORY_POOL=1`：pool 诊断。

### 新增环境变量（`environ.py`）
- `SGLANG_DFLASH_DRAFT_SWA_RESERVED_POINTS`（默认空）
- `SGLANG_DFLASH_DRAFT_SWA_RESERVED_CHUNK`（默认 0）
- `SGLANG_DFLASH_DRAFT_SWA_RESERVED_MAX_POINT`（默认 0）

---

## 5. 提交演进脉络

> rebase 到 `v0.5.10-xm` 后，历史整理为下列 10 个逻辑提交（按依赖顺序，自底向上）：

| 提交 | 说明 |
|---|---|
| `c1921130` | `feat(dflash): plumb independent draft SWA window config` —— 奠基：贯通独立 draft 窗口配置 |
| `6c07e4f7` | `feat(dflash): device SWA pool double-window (L1)` —— L1 device 三池 alloc/decode/evict/free |
| `5d2a2867` | `feat(dflash): SimpleSWARadixCache draft evict/protected + drift fix` —— radix 双窗口 joint safe len、draft 淘汰/protected |
| `41ff6d6d` | `feat(dflash): HiCache L2/L3 double-window prefetch & writeback` —— L2/L3 双窗口预取与回写 |
| `dec81588` | `feat(dflash): PD disaggregation draft-swa D2H (decode)` —— PD 分离 draft-swa D2H（含 direct-to-host） |
| `3d390e80` | `feat(dflash): draft compute - safe accept lens & spec worker` —— draft 计算：安全接受长度、投机 worker |
| `64c6896e` | `feat(dflash): scheduling & metrics for draft SWA` —— 调度准入三路预算与指标 |
| `4759b290` | `docs(dflash): double-window SWA design + example scripts` —— 本文档与示例脚本 |
| `4140cb5f` | `chore: fix ubuntu mirror sed path in Dockerfile.xiaomi` |
| `bb924476` | `fix: fix unit tests` —— 单测修复 |

---

## 6. 待关注 / 风险点

1. **L3 draft 读路径缺口**：写入/注册已连通，但 existence query + 回填映射疑似未接线（见 §3.5）。需确认是否为有意的 "base" 阶段。
2. **仅 gcache 支持 draft L3**：其它 storage 后端 draft L3 **fail-fast `raise`**(不再降级 L2-only);须显式关闭 draft L3。
3. ~~**`compute_joint_safe_len` 边界正确性**~~ → ✅ **已纠正并覆盖**：draft 从错误的滑动窗口 `[L-dw,L)` 纠正为 **fixed-edge** `[s,L)`（`s=max(0,total_len-dw)`，见 §3.2(c)）；`compute_target_and_joint_safe_len` / `compute_draft_safe_len` 及区分性反例（洞在 `s` 前/后）已由 `test_compute_swa_safe_len.py` 的 fixed-edge brute-force oracle + 300000 随机用例覆盖。
4. **两端配置一致性**：PD 分离下 prefill/decode 必须双窗口配置一致，否则运行期报错——部署时需保证配置同步。
5. **调度预算耦合**：`PrefillAdder` 新增第三维预算，与既有 full/swa 预算的交互在高并发下需压测验证。

---

## 7. 关键文件索引

| 层 | 文件 | 关键符号 |
|---|---|---|
| 分配器 | `mem_cache/swa_memory_pool.py` | `register_draft_swa`, `alloc`, `free_draft_swa`, `translate_loc_from_full_to_draft_swa` |
| Radix/淘汰 | `mem_cache/simple_swa_radix_cache.py` | `compute_target_and_joint_safe_len`, `compute_draft_safe_len`, `_evict_draft_swa` |
| 分层缓存 | `mem_cache/hiradix_cache.py` | `_compute_safe_prefix_len`, `estimate_load_back` |
| L2 host | `mem_cache/memory_pool_host.py` | `register_draft_swa_host`, `free_swa_by_full_indices`, `alloc_swa_only_unmapped` |
| 控制器 | `managers/cache_controller.py` | `draft_independent_swa`, `append_dflash_only_host_mem_release` |
| L3 storage | `mem_cache/storage/gcache/gcache_storage.py` | `batch_*_draft`, `_convert_hybrid_keys(draft=True)` |
| 调度 | `managers/schedule_policy.py` | `rem_draft_swa_total_tokens` |
| 投机 | `speculative/dflash_worker_v2.py`, `dflash_info_v2.py`, `dflash_utils.py` | `_draft_independent_swa`, `_maybe_evict_draft_swa` |
| PD | `disaggregation/{prefill,decode,mooncake/conn}.py` | `draft_state_*`, `maybe_send_draft_state` |
| sizing | `model_executor/model_runner_kv_cache_mixin.py` | `_resolve_draft_swa_tokens`, `get_draft_cell_size_per_token` |
| 配置 | `configs/model_config.py`, `environ.py` | `DFlashDraftModel`, `SGLANG_DFLASH_DRAFT_SWA_RESERVED_*` |

---

## 8. 原始设计方案 vs 实际实现（逐条对照）

> 本节保留**初版设计方案原文**，并逐条标注其在当前代码中的落地状态：
> **✅ 已实现** / **🟡 部分实现或有调整** / **❌ 未按此方案实现**。
>
> **命名对照（设计 → 代码）**：
> | 设计方案命名 | 实际代码命名 |
> |---|---|
> | `dflash_attn_allocator` | `draft_swa_attn_allocator` |
> | `full_to_dflash_index_mapping` | `full_to_draft_swa_index_mapping` |
> | `dflash_num_tokens`（EvictParams） | `draft_swa_num_tokens` |
> | `dflash_evictable_size` / `dflash_protected_size` | `draft_swa_evictable_size` / `draft_swa_protected_size` |
> | `dflash_pool_host` / `full_to_dflash_index_mapping`（L2） | 复用整个 `SWATokenToKVPoolHost` 实例 `mem_pool_host_draft`，其 `.swa_pool_host` 环即 draft 池 |
> | L3 key `_{tp}_dflash_k` / `_{tp}_dflash_v` | gcache 用 `d:` 命名空间前缀；generic 用 `{hash}.draft` 后缀 |

### L1

#### `SWATokenToKVPoolAllocator`
**设计**：新增 DFlash 的 `TokenToKVPoolAllocator`（`dflash_attn_allocator`），大小 = target swa pool 的 `(draft_window / target_window)` 倍、上限 target full 大小；指标对标 swa pool；新增 `full_to_dflash_index_mapping`。

- **✅ 已实现**（命名为 `draft_swa_attn_allocator` / `full_to_draft_swa_index_mapping`，`swa_memory_pool.py:320-446`）。
- **✅ 大小公式**：`_resolve_draft_swa_tokens` = `swa_tokens * draft_window / target_window`，`min(_, full_tokens)`，`floor_align(page_size)`（`model_runner_kv_cache_mixin.py:1001`）。
- **✅ 指标对标 swa**：见 §4。

#### `SimpleSWARadixCache`

**预置打桩点（reserved points）**
**设计**：Dflash window=1024，默认打桩点拼接已接近 full，可考虑直接去除（配置空串），自然留存已接近 full。

- **✅ 已实现且默认即空**：`SGLANG_DFLASH_DRAFT_SWA_RESERVED_POINTS` 默认 `""`、`_CHUNK`/`_MAX_POINT` 默认 `0`（`environ.py:542`），解析于 `_load_draft_swa_reserved_config`（`simple_swa_radix_cache.py:217`）。即采纳了"默认去除"的建议。

**`evict_swa`（调用点 `cache_unfinished_req` / `cache_finished_req`）**
**设计**：对 req 私有 cache 驱逐时新增 dflash 驱逐；可能需在 req 上新增对标 `swa_evicted_seqlen` 的 dflash 驱逐点记录。

- **✅ 已实现**：`Req.draft_swa_evicted_seqlen`（`schedule_batch.py:644`）；`evict_swa` 内 draft 驱逐点推进（`simple_swa_radix_cache.py:538/552`）。
- **注**：解码步的 draft 主动缩窗另有独立入口 `_maybe_evict_draft_swa`（`dflash_info_v2.py:132`），在 `prepare_for_decode` 于 target 缩窗后调用。

**`super().cache_finished_req` 的隐式写入**
**设计**：dflash cache 随 request 私有 full attn KV 隐式写入 radix tree，释放随 full 进行，full 未驱逐则 dflash 存在。

- **✅ 语义成立**：draft 映射以 full loc 为 key；`free()` 按 full 级联释放 swa + draft_swa（`swa_memory_pool.py:845-854`）。

**`evict`（EvictParams 新增 `dflash_num_tokens`）**
**设计**：仍按 full 驱逐，顺带 swa/dflash；dflash 驱逐累加计数与 `dflash_num_tokens` 目标值对比。

- **✅ 已实现**（命名 `draft_swa_num_tokens`）：`EvictParams`/`EvictResult` 新增字段（`base_prefix_cache.py:80/90`）；`common.py:242` 计算三路缺口传入 `evict`。

**Dflash 池大小记录（`dflash_evictable_size` / `dflash_protected_size`）**
**设计**：用于准入判断、监控指标。

- **✅ 已实现**（命名 `draft_swa_evictable_size` / `draft_swa_protected_size`，`simple_swa_radix_cache.py:709/713`）；准入侧 `schedule_policy.py:627/636` 读取。

**`match_prefix` → `_compute_safe_prefix_len`**
**设计**：full 完全匹配；swa 需 window(128) 内有效否则回退到 0；dflash 需 window(1024) 内有效，但**若 swa 匹配点前的 dflash window 内 dflash 无效而 extend > 1024，则可豁免 dflash 有效性**（dflash 部分重算即可，不影响结果）。

- **✅ 完全实现（fixed-edge，见 §3.2(c)）**：`compute_target_and_joint_safe_len`（`simple_swa_radix_cache.py:113`）：
  ```
  s = max(0, total_len - draft_window)          # 物化边界 = 固定左端
  draft_safe = [s, n) 内第一个 dead slot(否则 n)  # 单调连续前缀; s>=n 即 extend>=window 时全豁免
  joint = 框内最长 target-safe 前缀 = target_valid[:draft_safe] 的 last_valid
  ```
  注:draft 用固定左端 `[s,L)` 而非滑动 `[L-dw,L)`（后者是已纠正的 bug）。

### L2

#### `SWATokenToKVPoolHost`
**设计**：新增 `self.dflash_pool_host` 与 `self.full_to_dflash_index_mapping`。

- **🟡 实现方式不同（更简洁）**：并非在单个 host pool 内新增 `dflash_pool_host` 字段，而是**整体复用一个独立的 `SWATokenToKVPoolHost` 实例 `mem_pool_host_draft`**，用其 `.swa_pool_host` 环承载 draft SWA，用其自带的 `full_to_swa_index_mapping` 承载 draft 映射。通过 `register_draft_swa_host()` 挂接（`memory_pool_host.py:2720`）。语义等价，命名/组织不同。

#### `alloc_device`
**设计**：解除 assert，按 `draft_valid_mask` 分配 device draft slot，回写 device 的 `full_to_draft_swa_index_mapping`。

- **✅ 已实现**：`alloc_device(host_indices, swa_valid_mask, draft_swa_valid_mask)` 返回 `(full_idx, lack_full, lack_swa, lack_draft_swa)`，按 `draft_swa_valid_mask` 分配并回写 draft 映射（`swa_memory_pool.py:489-554`）。

#### `BaseHiRadixCache`
- **`write_backup` / `alloc_host`**：**✅** 额外 alloc draft host 空间（`cache_controller.py:1154` `alloc_swa_for_full_indices`）。
- **`load_back` / `alloc_device`**：**✅** 额外 alloc draft device 空间（三路缺口，见上）。
- **`evict` / `evict_host` 新增 `dflash_num_tokens`**：**✅**（命名 `draft_swa_num_tokens`；`evict_host` 返回 `(full, swa, draft_swa)`，`cache_controller.py:1772`）。
- **`match_prefix`（device+host 并集，规则同 L1）**：**✅** `_compute_safe_prefix_len` 拼接 device+host 后走 `compute_target_and_joint_safe_len`（`hiradix_cache.py:2598`）。
- **`estimate_load_back`（预估 dflash 用量）**：**✅** 返回三元组，新增 `draft_swa_load_back`（`hiradix_cache.py:1431-1461`），调度侧消费（`schedule_policy.py:1031`）。
- **`drain_storage_control_queues`（TP 对齐检查加 dflash）**：**✅** draft 侧对齐检查（`hiradix_cache.py:1081/1107/1665/1707`）。
- **`collect_tree_info`（指标搜集）**：**✅** draft 指标见 §4。

#### `Schedule_policy`
**设计**：`draft_swa_total_tokens` 要考虑从 L2 load back 的 draft KV。

- **✅ 已实现**：`draft_swa_total_tokens = swa_extend_tokens + draft_swa_load_back_tokens`（`schedule_policy.py:1043`）。

#### `HiSimpleSWARadixCache`
- **`match_prefix`：`swa_refill_len` cap 到 draft safe device lens**：**✅** `swa_refill_len = min(full_matched_device_len, draft_safe_device_len)`（`hiradix_cache.py:2951`）。
- **refill / rewrite：dflash 不考虑这两个**：**✅** 无 draft refill/rewrite 路径（注释明确说明 draft 不可 refill，见 `hiradix_cache.py:2947-2949`）。

### L3

#### `HiCacheController`

**队列（新增 `dflash_only_host_mem_release_queue`）**
- **✅ 已实现**：队列 + `append_dflash_only_host_mem_release`（`cache_controller.py:766/2016`），经 `mem_pool_host_draft.free_swa_only` 排空。

**`start_writing`（新增 dflash D2H，算子 `transfer_kv_all_layer_fuse_lf_pf` 兼容）**
- **✅ 已实现**：draft D2H `backup_from_device_all_layer`（`cache_controller.py:1301/1320`），two-window 用 `_build_draft_swa_args`（`:1214`）。

**`start_loading`（新增 dflash H2D，算子 `transfer_kv_per_layer_pf_lf` 兼容）**
- **✅ 已实现**：draft H2D `load_to_device_per_layer`（`cache_controller.py:1630-1634`）。

**Key 形式（设计：`_{tp}_dflash_k` / `_{tp}_dflash_v`）**
- **🟡 实现但命名不同**：gcache 用 **`d:` 命名空间前缀**（`_convert_hybrid_keys(draft=True)`，`gcache_storage.py:633`）；generic 后备用 **`{hash}.draft` 后缀**（`cache_controller.py:2642`）。语义一致（避免 target/draft key 碰撞），但非设计中的 `_dflash_k/_v` 形式。

**`safe_prefix` 计算（规则同 L1）**
- **单 rank：device+host+storage 并集**：**✅** 关键修正——设计特别指出"之前 swa 只考虑 storage，128 尚可接受，1024 损失大"，实现中 `_compute_safe_prefix_len` 已拼接 device+host（`hiradix_cache.py:2598`），并有 `batch_exists_draft` 查询 storage 存在性（`gcache_storage.py:474`）。
- **TP 组对齐：dflash 求交**：**✅** draft 侧对齐求交（见上 drain 检查）。

**裁剪计算（只看最后 dflash window；MTP 现无裁剪）**
- **✅** draft window 裁剪，`floor_align(seq_len - draft_swa_window_size)`（多处，如 `decode.py:719/1763`）。

**L3 读路径完整性（修正早先判断）**
- **✅ gcache 读路径已连通**：`_page_transfer` 在 `draft_l3_enabled` 时调用 `draft_page_get_func`（→ `batch_get_v1_draft`，`cache_controller.py:2175/2633`）。
- **限制**：仅 **gcache** 后端支持 draft L3（`draft_dflash_l3_enabled`）；其它后端 draft L3 **`raise NotImplementedError`**（fail-fast）。`operation.draft_swa_host_indices` 在 prefetch 预留后释放,属存在性查询优化的独立分支,不影响主读路径。

#### `Decode prealloc`（P2D 传输 + decode 侧 dflash pool 维护）
**设计**：需 P2D 传输支持与 decode 上 dflash pool 维护；为方便实现，**按 `max(target_window, dflash_window)` pre_alloc 显存**，target 有瞬时浪费，多分配 slot 随窗口滑动被驱逐。

- **🟡 P2D + decode 维护：✅ 已实现**（`decode.py` 的 `_get_draft_state_indices` `:701`、独立 draft 预算 `:1201-1219`、draft 缩窗点 `:1759`）。
- **❌ `max(target, dflash)` pre_alloc 简化被放弃 / 升级**：实际实现对 target 与 draft **各用独立窗口独立预算**（`sliding_window_size` 与 `draft_swa_window_size` 分开传入 `_swa_required_tokens`/`_max_swa_tokens_for_req`，`decode.py:1199-1204` 等），**没有**用 max 合并。这是相对初版设计的**主动改进**——避免了 target 的瞬时显存浪费。

#### PD KV cache 传输（mooncake 新增 `draft_state_indices`）
**设计**：mooncake 新增 `draft_state_indices` 传输 dflash swa indices，增加部分 transfer 开销。

- **✅ 已实现**：独立传输组 `draft_state_*`（`base/conn.py`），prefill 端只发 draft 窗口切片（`prefill.py:955`），mooncake `maybe_send_draft_state` 独立 buffer（`mooncake/conn.py`），两端配置不一致显式报错。

### 设计 → 实现 的主要差异小结

| 项 | 差异 | 结论 |
|---|---|---|
| 命名 | `dflash_*` → `draft_swa_*`；L3 key `_dflash_k/v` → `d:`/`.draft` | 纯命名，语义一致 |
| L2 host 组织 | 设计在单 pool 内加 `dflash_pool_host` → 实际复用整个 `mem_pool_host_draft` 实例 | 组织更简洁，语义等价 |
| Decode pre_alloc | 设计用 `max(target, draft)` 简化 → 实际独立双预算 | **实现优于设计**，消除 target 瞬时浪费 |
| L3 safe_prefix | 设计要求 device+host+storage 并集 | 已按此实现（修正 128→1024 的损失问题）|
| reserved points | 设计建议默认去除 | 已采纳，默认空串 |

---

## 9. Draft SWA vs Target SWA：现存缺口分析

> 对比成熟的 **target SWA**，draft SWA 的**核心内存管理链路（alloc / free / evict / match / splice / L3 prefetch 读回 / PD 传输 / 可观测性）已基本对齐**，剩余缺口集中在**"修复类"路径（refill / rewrite）、诊断、边角配置**。下表按影响从高到低排列。
>
> **进度**：
> - §9.0（L3 prefetch 读回）已完成。
> - §9.0.1（size_list bug 修复、draft L3 指标、tree token 分解、gcache 日志开关）已完成。
> - §9.1 的 (1)(2)（refill / rewrite）按指示暂不处理。
> - §9.3 (7)（collect_tree_info draft 分解）已完成，见 §9.0.1。
> - **修正**：早先记录的"HiCache 下 draft evictable/protected 指标恒 0"经核查是**误判**，实际经 MRO 继承正常工作，见 §9.7。
> - §9.2 / 其余 §9.3 视需要补齐。

### 9.1 🔴 高影响：功能性缺失（剩余）

> 注：原 (3) L3 prefetch 读回已实现，见 §9.0。以下 (1)(2) 为你指定暂不处理的已知问题。

#### (1) 无 draft refill（回填）路径
- **Target 有**：`alloc_missing_swa_for_full_indices`（`swa_memory_pool.py:556`）+ `refill_device_swa`（`cache_controller.py:1450`）+ `init_swa_refill`（`hiradix_cache.py:2834`）。当只有 full-KV 命中 device、SWA tail 缺失时，能从 host 回填 target SWA，把可复用前缀拉长。
- **Draft 无**。当前做法是**反向妥协**：`swa_refill_len = min(full_matched_device_len, draft_safe_device_len)`（`hiradix_cache.py:2951`），把 target 的 refill 长度 **cap 到 draft 安全长度** —— draft 反而拖累 target 的复用。
- **后果**：draft window(1024) ≫ target(128)，draft 不可回填意味着一旦 draft tail 有洞，整段前缀复用被砍短，命中率损失可能显著（即设计文档中提到的 1024 窗口损失问题）。

#### (2) 无 draft rewrite（重写/修复）路径
- **Target 有** `rewrite_swa_backup`（`hiradix_cache.py:861`），修复 host/device SWA 映射不一致。
- **Draft 无**。device-only 的 draft slot 不会回写到 host（`_build_draft_swa_args` 注释明说 "no draft rewrite path"，`cache_controller.py:1214`）。

#### ~~(3) L3 draft 存储命中不完整（prefetch 只写不读）~~ → ✅ 已实现（见 §9.0）

### 9.0 ✅ 近期已补齐：L3 draft prefetch 读回

> 原为 §9.1(3) 的 P0 缺口，现已完整实现，把 draft L3 从 "base"（只写）升级为完整读回链路。

**问题回顾**：此前 `PrefetchOperation.draft_swa_host_indices` 被预留后在 I/O 前直接释放（旧 `cache_controller.py:2578`），`batch_exists_draft` 从未被调用，draft prefetch 槽位从不 map/populate —— 从 storage 预取回的 draft KV 无法落地。

**实现**（`managers/cache_controller.py`）：

1. **`PrefetchOperation.draft_safe_pages`（新增字段）**：记录已从 storage 解析并映射进 draft host pool 的 leading 页数，作为 `prefetch_thread_func`（写）↔ `_page_transfer`（读）的跨方法契约。

2. **`_resolve_draft_l3_prefetch(operation, hit_pages)`（新增方法）**，替换原"只释放不映射"块，在 target 命中前缀 + swa 映射确定后执行：
   - `batch_exists_draft(hash_value[:hit_pages])` 探测 draft 存在性（`d:` 命名空间）
   - **独立 `all_reduce(MIN)`** TP 同步 draft mask（保证各 rank 一致）
   - `_compute_safe_storage_prefix(draft_mask, draft_sliding_window_pages)` 算 draft 安全前缀，**cap 到 `hit_pages`**（draft 不能超出 target 命中集）
   - `mem_pool_host_draft.mapping_full_and_swa_indices(...)` 映射 draft host 槽位，释放 surplus
   - **TP 分歧防护**：预留不足以覆盖商定前缀时，整体禁用本 op 的 draft L3（释放全部预留）而非部分映射，避免各 rank draft-valid mask 分歧

3. **`_page_transfer` 修正 draft get**：two-window（`draft_dflash_l3_enabled`）下只对 `[0, draft_safe_pages)` 已映射前缀发 `batch_get_v1_draft`，传 **full host indices**（gcache 内部经 `full_to_swa_index_mapping` gather draft 槽位——不能传 raw draft SWA indices）；legacy 单池 draft 保留原行为。

**正确性保证**：draft **存在性** miss（resolve 阶段 `batch_exists_draft` 未命中）只收缩 draft-safe，不影响 target 长度——因为 `[0, draft_start)` 是 draft 豁免区（可由 target hidden states 重物化）。**但 draft 传输失败（§9.0.2）会 cap target**：进入 draft 窗口 `[draft_start, ...)` 后 draft KV 无法重物化，故 target 复用前缀不得超过 `draft_start + draft_completed_pages`（豁免区 `[0, draft_start)` 不受约束）。draft 映射 page-uniform、TP-identical、只指向已 populate 页；未用预留恰好释放一次；下游 L2→L1 经 `alloc_device`→`gather_swa_mapping` / `start_loading`→`_build_draft_swa_args` 自动接管，无需额外改动。

**限制不变**：仍仅 gcache 后端（`draft_dflash_l3_enabled`）；且 draft 只能回填 target 已命中的页集（draft-only 前缀扩展属独立特性，未做）。

### 9.0.2 ✅ 近期已补齐：draft L3 取回失败处理 + TP 同步

**问题**：`_page_transfer` 中 `draft_page_get_func`（→ `batch_get_v1_draft`，返回 per-page `List[bool]` 成功掩码）的**返回值曾被忽略**。`batch_exists_draft` 探测"存在"到实际 `batch_get` 之间,页可能被 evict / IO 失败 / 部分读；一旦某 draft 页取回失败:
1. 无处截断 draft-safe 前缀 —— 已映射的 draft host slot 被下游当作有效,但内容为空/垃圾 → draft verify 读到 stale KV。
2. **无 TP 后置对齐** —— target 在 `check_prefetch_progress` 有 `all_reduce(MIN)`,但 draft 的唯一 TP 对齐是取回**之前**的 `batch_exists_draft`(存在性)。若 rank A 在页 5 取回失败、rank B 成功,则各 rank 的 draft host pool **发散**,破坏 TP 内存池同步。

**修复**(`cache_controller.py` + `hiradix_cache.py`):

1. **`_draft_page_get_zero_copy` 封装计数/失败检测/warning**(签名对齐 target:`(operation, hash_values, host_indices, page_offset, extra_info) -> bool`)。内部调 `batch_get_v1_draft`,扫首个失败页,记录 `operation.draft_completed_pages = page_offset + completed_pages`(绝对页索引,**不碰** `completed_tokens`),返回是否截断供 loop short-circuit。draft L3 已 gcache-only,generic 路径及 `_draft_page_get_generic`/`_first_draft_fail` 已删除。

2. **`_page_transfer` 用 bool 返回值 short-circuit + cap target**:draft 是 fixed-edge 连续前缀,首个失败页 `M` 后停止后续 draft 取回;**同时把 target 复用前缀 cap 到 `M = draft_start + draft_completed_pages`**——当前 batch target 只取 `[i, M)`,随后 `mark_terminate` 停止循环。豁免区 `[0, draft_start)` 不受影响。（draft 进入窗口后 KV 无法重物化,target 不得超过实际取到的 draft 页。）

3. **`check_prefetch_progress` 复用同一 `all_reduce(MIN)` 集合通信**:把 `draft_completed_pages` 加入现有 `states` 张量(与 target 的 `min_completed_tokens` 一起 MIN),**一次集合通信原子对齐** target + draft。集合通信在 scheduler 线程(非 IO aux 线程)。

4. **`_release_draft_prefetch_surplus`（新增）**:TP 对齐后释放 `[aligned, mapped)` 的尾部 draft host slot,draft 长度 **cap 到 `min(draft_completed_pages, mapped_pages, target_pages)`**(`max(0,...)` 死代码已移除、改 assert)。

5. **⚠️ double-free 修复(TP 分歧真凶)**:surplus 释放**必须用 `unmapping_full_and_swa_indices`(读+清零),不能用 `gather_swa_mapping`(只读)**。因为 target 侧 `mem_pool_host.free()` 会级联 `mem_pool_host_draft.free_swa_by_full_indices()`,若 draft 映射未清零,同一 draft slot 会被**释放两次**(surplus 队列一次 + target 级联一次)→ 破坏 draft free-ring → **draft host swa availability 跨 TP rank 分歧**(触发 `drain_storage_control_queues` 的 `values[7]` assert)。这是 pre-existing bug,draft 错误注入(扩大 surplus 范围)将其暴露。

6. **draft 错误注入(测试)**:`FaultInjector.maybe_inject_draft_fault(completed_pages)`,复用 `SGLANG_TEST_GCACHE_FI_TRANSFER_PROB` 开关但用独立 RNG;per-rank drop 部分 draft 页,由 §3 的 MIN 对齐 reconcile(TP-safe)。

**正确性保证**:draft 传输失败时,`_page_transfer` 把 target 复用前缀 **cap 到 `draft_start + draft_completed_pages`**(绝对页)——draft 窗口内 `[draft_start, M)` 之后 draft KV 无法重物化,target 若延伸过去会读到无对应 draft 的页;豁免区 `[0, draft_start)` 不受影响,照常复用。截断在首个失败 draft 页 `M` 处发生:该 batch 的 target 只取 `[i, M)`,之后 `mark_terminate` 停止循环,`completed_tokens` 随之收敛,下游 tree 插入(`check_prefetch_progress` 用 `min_completed_tokens`)自然只复用 `[0, M)`。每个 draft slot 所有权唯一转移(mapped → unmapped → queued free → allocator free),无 double-free;各 rank 经单次 MIN 对齐后 map/release **完全相同**的 draft slot,TP 池同步不变。double-free 修复经 slot-lifecycle 模型对拍验证(buggy=DOUBLE FREE,fixed=每 slot 释放一次)。

### 9.0.1 ✅ 近期已补齐：正确性修复 + 可观测性

在 §9.0 之后又完成了以下项：

**(a) gcache draft `size_list` 用错池（真 bug，已修）**
- `_convert_hybrid_keys(with_size=True)` 对 draft key 也用 **target 池**的 `get_page_element_sizes()`，导致 `batch_exists_draft` 传给 gcache 的 size 是 target 尺寸 → draft key 尺寸校验失败 → 系统性 miss。
- 修复：按 `draft` 标志选池 `size_pool = self.draft_mem_pool_host if draft else self.mem_pool_host`（`gcache_storage.py`）。

**(b) draft L3 storage-hit 全链路指标(与 swa 对称)**
- `draft_swa_hit_count`（`PrefetchOperation`）→ `storage_draft_swa_hit_by_reqid` → `req.storage_draft_swa_hit_tokens` → output processor → io_struct → tokenizer/detokenizer → Prometheus `sglang:storage_draft_swa_hit_tokens_total`。
- 另有 `sglang:draft_l3_resolve_latency_seconds`、`sglang:draft_prefetch_io_latency_seconds` 两个 histogram。

**(c) draft SWA tree token 分解**（原 §9.3 (7)，已完成）
- `SWATreeInfo` 新增 `draft_swa_tokens` / `draft_swa_tokens_locked`；`collect_tree_info` 用 `full_to_draft_swa_index_mapping`(dev) 与 `mem_pool_host_draft.gather_swa_mapping`(host) 统计（draft 池不存在时守卫跳过）。
- Prometheus 新增 `sglang:dev_draft_swa_tokens[_locked]`、`sglang:host_draft_swa_tokens[_locked]`。

**(d) gcache 日志开关**（调试 draft L3 用）
- `SGLANG_GCACHE_LOG`（总开关，INFO）/ `SGLANG_GCACHE_LOG_DETAILS`（额外打印具体 key/size/result）。
- `_page_backup` 加 guard：two-window 下 draft 映射全 0 时跳过空写（消除 "0 keys" 无谓调用）。

**(e) draft L3 probe/set 不对称：经 Oracle 判定为设计使然,非 bug**
- 现象：`draft L3 probe: exist_pages=0`。
- 结论：全 SWA draft 无 full-key 锚点,probe(`swa_k && swa_v` 都在)是**唯一正确**语义；set 侧 best-effort 只 backup live 窗口内的 draft 页（`_build_draft_swa_args` 只拷贝 device∩host 交集,device-only 有意不落盘,无 rewrite 路径）。那些页从未写入 storage → probe 正确 miss。
- **不修**：放松 probe 会产生 false hit；扩大 set 需要 §9.1 的 refill/rewrite（已暂缓）。真正提命中率 = 实现 refill/rewrite，属架构性限制。

**(f) prefix-shrink 指标统一**
- draft 窗口约束导致的匹配前缀收缩(`target_safe_len - joint_safe_len`)在两条路径统一到**同一个** `draft_swa_prefix_shrink_tokens` 计数器 → `sglang:draft_swa_prefix_shrink_tokens_total`。
- PD-prefill(`SimpleSWARadixCache`)与 HiCache(`HiSimpleSWARadixCache`)各自的 `_compute_safe_prefix_len` 都累加到它;经 scheduler-delta 导出。
- 旧的 HiCache 专用 Counter `hicache_dflash_prefix_shrink_tokens_total` 已**废弃删除**。

**(g) prefetch `reserve_swa_size` 按窗口修正(命中率优化)**
- 问题：`reserve_swa_size` 默认 512(为 target 128 窗口调),但 draft 窗口 1024 → prefetch 预留 < draft-safe 前缀 → `_resolve_draft_l3_prefetch` 里 `reserved_pages < draft_safe_pages` 整单放弃 draft L3,浪费命中。
- 修复：新增 `maybe_bump_reserve_swa_size(window)`(只增不减、page 对齐)。target 池在 `BaseHiRadixCache.__init__` 创建时按 `sliding_window_size` bump(**target-only 也生效**);draft 池在 `register_draft_swa_host` 按 `draft_swa_window_size` bump。

### 9.2 🟡 中影响：平台 / 配置限制

#### (4) draft L3 仅支持 gcache 后端
- `_maybe_register_draft_with_storage` 只在 `gcache` 启用 draft L3；其它后端(two-window 与 single-pool 均)在 `register_draft_mem_pool_host` **之前**即 `raise NotImplementedError`(fail-fast,避免部分状态)。原 generic 后备路径(`{hash}.draft`)及 `_draft_page_get/set_generic` 已删除。

#### (5) draft 不支持多 pool（draft pool list）
- `set_draft_kv_pool` 明确**拒绝** two-window + draft pool list 组合（`cache_controller.py:1698`）。

#### ~~(6) PD direct-to-host（P2D 直传 host）不支持 draft~~ → ✅ 已实现
- **已补齐**：`PDDirect2HostManager` 现构建独立 draft host transit 池并在 `pre_alloc` 分配 `host_draft_state_indices`；decode 侧 `_get_draft_state_indices` 在 `pd_d2h_mgr` 路径下产出真实 draft page indices（不再返回 `None`）。准入快照升级为 8 元组、三路 need 校验。详见 §3.6。
- **前置条件**：仅当 allocator 暴露 `draft_swa_attn_allocator` 且 target 池也有 SWA state 时启用；否则构造期显式 `NotImplementedError`。

### 9.3 🟢 低影响：可观测性 & 诊断 & 边角

#### ~~(7) `collect_tree_info` 无 draft 分解~~ → ✅ 已实现（见 §9.0.1(c)）
- `SWATreeInfo` 现有 `draft_swa_tokens` / `draft_swa_tokens_locked`；`collect_tree_info` 统计 device/host 的 draft SWA token；Prometheus 有 `sglang:dev/host_draft_swa_tokens[_locked]`。

#### (8) 诊断（DIAG）覆盖弱于 target
- Target 在 host 的 alloc / free / **splice** 都有 `[SWA-DIAG]` page-uniform 检查。
- Draft：底层 ring 的 `[SWA-DIAG]` 因复用同类自动生效，`append_dflash_only_host_mem_release` 也有 `[DFLASH-DIAG]`；**但 splice 的 draft cascade 缺少 `assert_page_uniform_swa_pattern` 后置检查**（`swa_memory_pool.py:954-970` vs target 的 `:940-951`），host splice 的 draft cascade 也不返回 transfer count（`memory_pool_host.py:3138`）。

#### (9) `backup_state` / `restore_state` 不含 draft_swa（潜在埋雷）
- `SWATokenToKVPoolAllocator.backup_state` 只存 full+swa 且 `assert len(state)==2`（`swa_memory_pool.py:973-982`），**不回滚 draft_swa free-list**。
- 当前 v2 worker 路径下影响有限（`dflash_worker.py:713` 备份的是 draft runner 自身的 allocator，非持有 draft 子池的 target allocator）；但若未来 target 侧 allocator 走投机备份/回滚，draft_swa 状态会**不一致**。属埋雷项。

### 9.4 ✅ 已对齐（无缺口）

`alloc` / `alloc_extend` / `alloc_decode` / `alloc_device` / `free` / `free_draft_swa` / `translate_loc_from_full_to_draft_swa` / **device+host splice cascade** / `clear` / `evict` / `evict_host` / `_compute_safe_prefix_len`(joint safe len) / `estimate_load_back` / reserved points / `evictable`&`protected` size（含 HiCache，§9.7）/ 调度准入三路预算 / L2 D2H+H2D / **L3 prefetch 读回（`_resolve_draft_l3_prefetch`，§9.0）** / `dflash_only_host_mem_release_queue` / TP 对齐 / PD mooncake `draft_state_*` 传输 / **PD direct-to-host draft state 传输（§3.6，8 元组准入快照）** / decode 独立预算 / **可观测性（pool 指标 + tree token 分解 + L3 hit/latency 指标，§9.0.1）**。

### 9.5 建议补齐优先级

| 优先级 | 缺口 | 理由 |
|---|---|---|
| ~~P0~~ ✅ | ~~(3) draft L3 prefetch 读回映射~~ | **已完成**，见 §9.0 |
| ~~P2~~ ✅ | ~~(7) collect_tree_info draft 分解~~ | **已完成**，见 §9.0.1(c) |
| **P1** | (1) draft refill、(2) draft rewrite | 大窗口下命中率损失主因（含 §9.0.1(e) 的 probe miss 根源）；已按指示暂缓 |
| **P2** | (8) splice DIAG 后置检查 | 低成本，提升诊断信心 |
| ~~P3~~ ✅ | ~~(6) PD 直传 host draft~~ | **已完成**，见 §3.6 / §9.2(6) |
| **P3** | (4)(5)(9) | 视部署形态（非 gcache / draft pool list / 投机回滚）按需补齐 |

### 9.6 缺口速查表

| # | 缺口 | Target 位置 | Draft 状态 | 影响 |
|---|---|---|---|---|
| 1 | refill 回填 | `swa_memory_pool.py:556`, `hiradix_cache.py:2834` | ❌ 无（反向 cap target） | 🔴 命中率 |
| 2 | rewrite 修复 | `hiradix_cache.py:861` | ❌ 无 | 🔴 一致性 |
| 3 | L3 prefetch 读回 | `cache_controller.py` `_resolve_draft_l3_prefetch` | ✅ 已实现（§9.0） | 🔴 L3 收益 |
| 4 | 非 gcache L3 | 全后端 | 🟡 仅 gcache | 🟡 平台 |
| 5 | draft pool list | 支持 | ❌ 拒绝 | 🟡 配置 |
| 6 | PD 直传 host | 支持 | ✅ 已实现（§3.6） | 🟡 部署 |
| 7 | collect_tree_info | `hiradix_cache.py` | ✅ 已实现（§9.0.1c） | 🟢 可观测 |
| 8 | splice DIAG | `swa_memory_pool.py:940-951` | 🟡 缺后置检查 | 🟢 诊断 |
| 9 | backup/restore_state | `swa_memory_pool.py:973-982` | ❌ 不含 draft | 🟢 潜在埋雷 |

### 9.7 修正记录：HiCache draft evictable/protected 指标（曾误判为缺口）

早先版本记录过一个"缺口":*"HiCache 下 draft_swa_evictable_size/protected_size 恒 0，因为只有 `simple_swa_radix_cache.py` 定义了它们，`hiradix_cache.py` 没有"*。**经核查这是误判,实际正常工作**：

- DFlash two-window 的 HiCache 树是 `HiSimpleSWARadixCache(BaseHiRadixCache, SimpleSWARadixCache)`（`scheduler.py:849` 实例化）。
- MRO：`HiSimpleSWARadixCache → BaseHiRadixCache → SimpleSWARadixCache → RadixCache`。
- `HiSimpleSWARadixCache.__init__` 虽注释说"不直接调 `SimpleSWARadixCache.__init__`",但它调的 `BaseHiRadixCache.__init__` **末尾有 `super().__init__(params=params)`**,经协作式 `super()` 沿 MRO **仍到达 `SimpleSWARadixCache.__init__`** → `_draft_independent` / `_draft_swa_evictable_size` / `_draft_swa_protected_size` 等**全部被初始化**。
- `draft_swa_evictable_size()` / `draft_swa_protected_size()` 方法与 `_on_node_becomes_protected/evictable` / `_on_tree_value_added/removed` 钩子经 MRO 继承自 `SimpleSWARadixCache`,且 BaseHiRadixCache 的 `inc_lock_ref`/`evict`/`insert` 会调用它们 → **计数器正常维护**。
- 误判根源：只在 `hiradix_cache.py` grep 方法**定义**(经继承而非本文件定义)+ 误读 `__init__` 注释。
- **结论(Oracle 确认)**：无 bug,不需改代码。`_record_draft_swa_stats` 的 `hasattr` 在 HiCache 下为 True,指标正常上报。

### 9.8 draft host pool available_size 跨 TP rank 不一致（根因与修复结论）

**症状**：`draft_independent_swa` + TP≥2 + HiCache，在 `writing_check`（`hiradix_cache.py:1149`）或 `drain_storage_control_queues`（`hiradix_cache.py:1774`）触发
`AssertionError: draft host swa availability diverged across TP ranks`。
特征：**只有 draft 池 available_size 发散，full / target-SWA 池永远一致**。在 fault injection（`[exists]` / `[draft]` / `[timeout]` / `[transfer]`）下才复现。

**不变量**：draft host SWA 驻留在设计上要求 TP-identical（各 rank 广播相同请求流、走相同 alloc/free 序列）。因此 draft pool `available_size()` 在每个 scheduler 边界必须逐 rank 相等，上述两处 assert 是正确的守卫，**不应移除**。

**根因（三条独立缺陷，共同表现为 draft 池发散；target 池因不具备这些路径而免疫）**：

1. **host `free()` 的 detach 顺序竞态** —— `SWATokenToKVPoolHost.free()` 原先在 `full_pool_host.free(full_host_indices)` **之后**才 detach draft 映射。full 索引一旦归还，另一分配可复用它并写入**新的** draft 映射，随后旧 `free` 读到被并发改写的映射 → 各 rank 释放数不同。target-SWA 映射在归还前就 detach，故不受影响。
   修复：draft 映射的 detach/free 移到 `full_pool_host.free()` **之前**，与 target-SWA 对称（`memory_pool_host.py:3415` 起）。并删除 `free()` 内重复的第二次 draft cascade 调用。

2. **prefetch 后台线程即时改环** —— `_resolve_draft_l3_prefetch` / `_release_unresolved_draft_reservation`（prefetch 线程）原先用 `free_swa_only(...)` **同步**释放 draft slot，直接改 draft 环 `available_size`。`[timeout]` fault 延迟该后台线程后，各 rank 释放完成进度在主线程对齐点（`drain` / `writing_check`）不一致 → 发散。target 池所有释放都经 release queue 由主线程排空，无此路径。
   修复：`draft_independent_swa` 下所有后台线程 draft 释放改为 `append_dflash_only_host_mem_release(...)` 入队（surplus / miss / exemption / map≤0 / unresolved 全部；`cache_controller.py:2575/2633/2643/2658/...`），draft 环只在主线程排空点变化。

3. **draft release queue 按本地 qsize 排空** —— `drain_storage_control_queues` 原先用 `dflash_only_host_mem_release_queue.qsize()`（**本地队列长度**）作为排空计数，各 rank 队列长度不同 → 排空数不同 → 发散。
   修复：改用 `values[6]`（与其它队列大小一起经 `all_reduce(MIN)` 对齐的值）作为排空计数（`hiradix_cache.py:1765`）。draft 池 available_size 也纳入同一 `qsizes` 张量做 MIN 对齐，供 assert 校验（`values[7]`）。

**正确性前提**：draft 预留（`alloc_swa_only_unmapped`）在主线程执行、且为「全有或全无」（仅依赖 `available_size()`）。因此环一致时 `reserved_pages` 逐 rank 相等、surplus 恒为 0，入队释放计数 TP-identical，`ReduceOp.MIN` 排空精确、无数量错配。

**关键原则（防回归）**：draft host SWA 环的 `available_size()` **只允许在主线程的受控点（alloc_host / drain）变化，且释放计数必须经 TP-MIN 对齐**；任何 prefetch 后台线程的直接 `free_swa_only`、或按本地 `qsize()` 排空，都会破坏 TP 一致性。`not draft_independent_swa` 分支例外（draft 与 target 共享 SWA 环，随 target 对齐，保留同步释放）。
</content>
</invoke>
