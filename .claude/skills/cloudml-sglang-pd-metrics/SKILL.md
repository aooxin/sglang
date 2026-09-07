---
name: cloudml-sglang-pd-metrics
description: CloudML 上 SGLang PD 分离部署的 Prometheus 指标分析方法论：按指标类型（Histogram/Gauge/Counter）正确计算延迟、吞吐、缓存命中、KV 使用率、投机接受率的分位与均值，并区分 prefill/decode 角色。分析某 PD 服务/部署的指标、判断分位分布、定位 TTFT/TPOT 瓶颈在 P 还是 D、对照 MiMo SLO 目标、判断 Prefill/Decode 扩缩容触发时使用；指标名与公式对齐 sglang 源码（metrics_collector.py / scheduler_metrics_mixin.py / req_time_stats.py）。
---

# CloudML SGLang PD 分离部署指标分析

面向 CloudML 上以 **Prefill / Decode 分离（PD disaggregation）** 部署的 SGLang 服务，回答「某服务/部署的某项指标是多少、分位分布如何、瓶颈在 P 还是 D」。本 skill 只沉淀**指标定义与计算方法**；实际发起查询、集群路由发现、看板一致性约束请配合 `cloudml-mimo-analyze` skill 使用（它负责 `deploy_id`/`service_id` 识别、集群探测、看板 PromQL 白名单）。

> **⚠️ 依赖：`cloudml-mimo-analyze`**。本 skill 只提供指标计算方法论，**发起真实查询依赖 `cloudml-mimo-analyze`**（集群路由探测、看板 PromQL 白名单、`deploy_id`/`service_id` 识别都在它那里）。若环境中尚无该 skill，先安装：
> ```
> /plugin marketplace add git@git.n.xiaomi.com:ai-studio/cloudml-claude-plugins.git
> /plugin install cloudml-mimo-analyze@cloudml-claude-plugins
> ```

> **📋 部署 ID 从哪来**：各集群的部署列表 / `deploy_id`（`s-<14位>-<5位>-<5位>`）可从 CloudML 部署看板查：**<http://dashboard-10023.bcecn-bj-cloudml.xiaomi.srv/>**。在看板选中目标服务拿到 `deploy_id` 后，再用 `cloudml-mimo-analyze --discover-only` 探测它所在集群的 Prometheus 路由，即可开查。

> **一句话原则**：先分清指标**类型**再谈分位——
> - **Histogram**（有 `_bucket`/`_sum`/`_count`）→ 分位用 `histogram_quantile(q, sum by(le,…) (rate(x_bucket[w])))`，均值用 `rate(x_sum[w])/rate(x_count[w])`。这是**按事件（请求/token）加权**的真实分位。
> - **Gauge**（瞬时值）→ **没有分桶，不能** `histogram_quantile`。分位只能是「时间样本分位」`quantile_over_time(q, x[w])` 或「同一时刻跨 pod 分位」`quantile(q, x)`，务必在结论里注明是哪一种，不要冒充请求级分位。
> - **Counter**（`_total` 等）→ 速率用 `rate(x[w])`，占比用两个 rate 相除。Counter 无分位。

---

## 0. 标签约定（PD 角色如何区分）

SGLang 侧 `metrics_collector` 的基础 labels（见 `scheduler_metrics_mixin.py:170`，engine_type 判定块 :161-175）：
`model_name, engine_type, tp_rank, pp_rank, moe_ep_rank[, dp_rank][, priority]` + `extra_metric_labels`（来自 server_args）。

- **`engine_type`** 是 SGLang 原生的 PD 角色区分位：`"prefill"` / `"decode"` / `"unified"`（非 PD 时为 unified）。由 `disaggregation_mode` 决定（mixin 159-166）。**注意 `engine_type` 只有这三个值——没有 `"encoder"`**（见下）。
- **CloudML 平台**通过 k8s Prometheus relabeling / `extra_metric_labels` 注入并映射出：`role_name`（= engine_type 的平台化名）、`deploy_id`、`model_service_id`、`pod`、`namespace`、`cluster`。**生产查询主用 `deploy_id` / `model_service_id` + `role_name`**。
- **⚠️ 第三种角色 `role_name="encoder"`（EPD 分离，多模态编码器）**：多模态输入的视觉/音频编码由**独立的 encode server**（`disaggregation/encode_server.py`）承担，形成 **E/P/D 三段分离**。它**不是 scheduler 的 `engine_type`**（scheduler 只有 prefill/decode/unified），而是一个**独立进程 + 独立 `EncoderMetricsCollector`**，base labels **只有 `model_name`（+`extra_metric_labels`），没有 `engine_type`/`tp_rank`/`pp_rank`**（`encode_server.py:3813` / `metrics_collector.py:EncoderMetricsCollector`）。因此：
>   - encoder pod **不上报** scheduler 系列指标（`num_running_reqs` / `token_usage` / `max_total_num_tokens` / TTFT / TPOT / `prompt_tokens_total` / `generation_tokens_total` …）——**§2–§12 的方法对 encoder 一律不适用**。
>   - encoder 有**自己的 `sglang:encoder_*` 指标族**（`encoder_e2e_request_latency_seconds`、`encoder_queue_wait_seconds`、`encoder_model_forward_seconds`、`encoder_transfer_seconds`、`encoder_cache_*`、`encoder_requests_received_total`、`encoder_dp_pending_requests` 等；见 catalog）。
>   - encoder 指标**不按 tp_rank 复制**（无该 label），`count by(pod)` / `sum by(pod)` 天然正确，**无 TP 去重问题**（§0 的 TP 陷阱只针对 scheduler gauge）。平台侧 `role_name="encoder"` 圈定。
- 少数自建部署（如 deepseek-r1-deploy）用 **`model_name` 后缀**区分：`model_name=~".*-prefill"` / `".*-decoder"`，配 `model_version` / `namespace` / `pod`。遇到这类先探测标签集再选筛选式。

**本 skill PromQL 模板统一用占位**：`$sel` = 目标选择器（`deploy_id="…"` 或 `model_service_id="…"`），`$w` = 速率窗口（默认 `5m`；瞬时 gauge 趋势用 `avg_over_time`/`quantile_over_time` 时窗口取分析区间）。角色筛选写 `role_name=~"(?i)prefill"` / `role_name=~"(?i)decode"`；若该环境无 `role_name`，回退 `engine_type="prefill"` / `"decode"` 或 `model_name` 后缀。

> **⚠️ TP 复制陷阱（Gauge 求和必读，已实测验证）**：scheduler 的 **Gauge** 指标（`num_running_reqs` / `gen_throughput` / `decode_sum_seq_lens` / `num_queue_reqs` / `token_usage` / `spec_*` / `cache_hit_rate` 等）**每个 pod 会按 `tp_rank`（及 pp_rank/moe_ep_rank）复制多份完全相同的序列**（TP=8 时每 pod 8 条）。因此：
> - **对 Gauge 做 `sum()` 会按 TP 度数放大**（实测 `sum(num_running_reqs)=136`，真值 dedup 后 `17`；`sum(gen_throughput)=37526`，真值 `≈5386`）。**求和前必须按「pod × dp_rank」去重**：`sum( avg by(pod, dp_rank) (gauge{…}) )`（同一 `(pod, dp_rank)` 内各 tp/pp/ep rank 值相同，`avg`=原值）。
>   - **去重单元是 `(pod, dp_rank)` 而不是 `pod`**：开 DP 时同一 pod 上每个 `dp_rank` 是**独立 batch**，可加 gauge（`num_running_reqs` / `decode_sum_seq_lens` / `gen_throughput`）必须**跨 dp_rank 求和**。若写 `max by(pod)`（或 `sum(max by(pod))`）会把同 pod 的多个 dp_rank 压成一个 → **按 DP 度数少算**。
>   - **未开 DP（`dp_rank` 为空/不存在）时** `avg by(pod, dp_rank)` **退化为** `max by(pod)`，两者完全等价（线上 MiMo 部署实测 `dp_rank` 为空，故历史用 `max by(pod)` 的数字仍准确）。**遇到新部署先探测 `dp_rank` 是否非空再决定**。
>   - 线上生产看板即按此**分两路**：`dp_rank=~".+"` 走 `sum by(deploy_id)(avg by(pod, dp_rank, deploy_id)(…))`、`dp_rank=""` 走 `sum by(deploy_id)(avg by(pod, deploy_id)(…))`。
> - **`avg()` / `avg_over_time` / `quantile_over_time` 读 Gauge 不受影响**（相同副本求平均=原值），可直接用。
> - **Counter**（`*_total`、`*_count`）只由每 pod 一条序列上报（**不按 tp_rank 复制**），`sum(rate(…))` 天然正确，无需去重。
> - `count(count by(pod)(…))` 数节点也天然去重（§1），不受影响。

**分位表头固定升序 `P50 | P70 | P90 | P99`**（即使看板图例按 P99→P50 排列，答复也升序）。均值是独立表达式，不能填进分位列。**无序列/查询失败写「无序列」/「查询失败」，绝不写 0。**

**比率型指标统一 ×100 按 % 报**（下面各节不再重复）：所有原始 0–1 的比值/使用率（`token_usage` / `cache_hit_rate` / `spec_accept_rate` …）PromQL 前缀 `100 *`；唯一例外 `spec_accept_length` 是 token 数（每 forward 产出），保持原值不乘。

> **⚠️ 看板 0 ≠ 真 0**：**看板上的 0 可能是「该窗口无请求」而非真的 0 延迟/0 命中——用看板 0 值下结论前，先用 `_count`/`rate` 确认样本量。** 因为线上看板的直方图面板用自定义 `histogram_quantile_or_zero`（无数据返回 **0**、非 `NaN`），本 skill 用原生 `histogram_quantile`（无数据报「无序列」）；有数据时两者分位值完全相同，只在空数据时相反。

---

## 1. Prefill / Decode / Encoder 节点数分析

**计算**：某角色下**上报指标的 pod 去重计数**（`count by(pod)` 再 `count`）。**底 metric 按角色不同**——用「进程存活即上报」的 gauge（scheduler tick 刷、空闲也不为 0）做底，比 `num_running_reqs`（空闲 pod 可能为 0）更稳；encoder 无 scheduler，另用其自有指标。

```promql
# Prefill 节点数
count( count by (pod) (sglang:max_total_num_tokens{$sel, role_name=~"(?i)prefill"}) )
# Decode 节点数
count( count by (pod) (sglang:max_total_num_tokens{$sel, role_name=~"(?i)decode"}) )
# Encoder 节点数（encoder 不上报 scheduler gauge，用 encoder 自有指标做底）
count( count by (pod) (sglang:encoder_cache_size_mb{$sel, role_name=~"(?i)encoder"}) )
```

- **encoder 底 metric**：`sglang:encoder_cache_size_mb`（encoder 侧 gauge，进程存活即注册；这正是生产看板「部署级别总览/EPD数目」面板数 encoder 节点用的底）。若该 metric 无序列，退回 `sglang:encoder_requests_received_total`（Counter，收到过请求即有序列）或 `sglang:encoder_dp_pending_requests`（router gauge）——但这两个需 pod 处理过请求才出现，**空载 encoder 用 `encoder_cache_size_mb` 更稳**。encoder 指标**只有 encoder pod 上报**，`role_name` 过滤可省，但带上更明确。
- **生产看板口径**（「EPD数目」面板，可直接对齐）：prefill 数 = `count by(pod)(sglang:prompt_tokens_total{…prefill})`、decode 数 = `count by(pod)(sglang:generation_tokens_total{…decode})`、encoder 数 = `count by(pod)(sglang:encoder_cache_size_mb{…})`；三者外层 `sum by(deploy_id)`。看板用吞吐 Counter 数 P/D 节点，本 skill 用 `max_total_num_tokens` gauge——**有稳定流量时两者一致**；但吞吐 Counter 在**空载 pod（窗口内零请求）上无序列会漏数**，gauge 底更稳，按需二选一。
- 需要时间轴上的节点数变化：`count by(role_name) (max_over_time(sglang:max_total_num_tokens{$sel}[$w]))`（P/D 按 role 分组；encoder 换 `encoder_cache_size_mb`）。
- **不是分位指标**——只报当前值/时间序列，不要编 P50/P90。TP/PP 多卡时同一 pod 只算一个节点（`count by(pod)` 已去重 rank；encoder 无 tp_rank 更无此问题）。

---

## 2. 平均输入 / 输出长度分析

**指标**（Histogram，`TokenizerMetricsCollector`）：
- 输入长度：`sglang:prompt_tokens_histogram`（在入口/prefill 侧观测）
- 输出长度：`sglang:generation_tokens_histogram`（decode 侧观测）
- 还有 `sglang:uncached_tokens_per_request`（未命中缓存的输入 token 数直方图；注册名带 `_per_request` 后缀，漏写会查空。另有同源 Counter `sglang:uncached_tokens_total`）

**分位（P50/P70/P90/P99）**：
```promql
histogram_quantile(0.50, sum by (le) (rate(sglang:prompt_tokens_histogram_bucket{$sel}[$w])))
histogram_quantile(0.90, sum by (le) (rate(sglang:generation_tokens_histogram_bucket{$sel}[$w])))
```
**均值**（更稳，抗长尾）：
```promql
rate(sglang:prompt_tokens_histogram_sum{$sel}[$w])   / rate(sglang:prompt_tokens_histogram_count{$sel}[$w])
rate(sglang:generation_tokens_histogram_sum{$sel}[$w]) / rate(sglang:generation_tokens_histogram_count{$sel}[$w])
```
> 也可从 Counter 反推平均：`rate(sglang:prompt_tokens_total[$w]) / rate(sglang:num_requests_total[$w])`。注意 `num_requests_total` 在 decode 侧计数（完成请求），`prompt_tokens_total` 输入侧——PD 下要在同一角色作用域内比，否则分子分母角色不一致。**优先用 histogram_sum/count**，角色自洽。

---

## 3. TTFT：首 token 延迟分析

**指标**：`sglang:time_to_first_token_seconds`（Histogram）。
分桶：`[0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10,20,40,60,80,100,200,400]`（最大 400s，尾部分辨率粗）。
带额外 label `is_streaming`（`true`/`false`，源码 `metrics_collector.py` `observe_time_to_first_token`）——本节 `sum by (le)` 已跨该 label 聚合、无需处理；要分流式/非流式再显式 `by (is_streaming)`。

> **PD 关键**：TTFT 在 **decode 角色**上观测（首 token 由 decode 产出），PD 分离时 prefill pod **不发** `time_to_first_token_seconds`。因此 TTFT 按 decode pod 展开，"prefill 对 TTFT 的贡献"看第 4 节的阶段拆解。

```promql
# P50/P70/P90/P99（把 0.50 换成 0.70/0.90/0.99）
histogram_quantile(0.99, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{$sel, role_name=~"(?i)decode"}[$w])))
# 按 pod 展开：sum by (le, pod)
# 均值
rate(sglang:time_to_first_token_seconds_sum{$sel, role_name=~"(?i)decode"}[$w])
  / rate(sglang:time_to_first_token_seconds_count{$sel, role_name=~"(?i)decode"}[$w])
```
- 低 QPS 时段单桶样本极少 → 分位（尤其 P99）可能跳到桶边界（如 300/400s）是**直方图伪影**，不是真延迟，需结合 `_count` 速率判断样本量。

---

## 4. TTFT 中 prefill 各阶段延时分析

**主指标**：`sglang:per_stage_req_latency_seconds`（Histogram，**额外 label `stage`**）。
分桶：`exponential_buckets(start=0.001, width=1.62, length=30)`（覆盖 1ms→数百秒，指数分辨率）。

**`stage` 取值**（源码 `req_time_stats.py`，仅 `metrics_is_observed=True` 的会进该指标）——**prefill 链路相关**：
`pre_process`(HTTP到达→建 Req) → `request_process` → `prefill_bootstrap` → `prefill_transfer_kv_cache`(KV 传给 decode) → `prefill_forward` / `chunked_prefill`。
decode 侧：`decode_prepare` / `decode_bootstrap` / `decode_waiting` / `decode_transferred` / `decode_h2d_wait` / `fake_output` / `quick_finish`。HiCache 预取：`hicache_prefetch_{wait_total,queue,query,resolve,io_queue,io,commit}`。
> **⚠️ 不在本指标里的常见 stage**：`prefill_waiting`（源码显式 `metrics_is_observed=False`；其排队耗时另由专用 Histogram `queue_time_seconds` 上报，见下）和 `prefill_prepare`（默认 `metrics_is_observed=False`，仅 trace）——查 `stage="prefill_waiting"` / `"prefill_prepare"` 会返回空，排队时长请改查 `queue_time_seconds`。

**分位（按阶段分组）**：
```promql
histogram_quantile(0.90,
  sum by (le, stage) (rate(sglang:per_stage_req_latency_seconds_bucket{$sel, role_name=~"(?i)prefill"}[$w])))
# 单阶段：加 stage="prefill_bootstrap"
```

**专用 Histogram（与 stage 互为佐证，PD 队列耗时）**：
- `sglang:queue_time_seconds` — 排队时长，分桶 `0.0…3000`
- `sglang:prefill_bootstrap_time_seconds` — prefill bootstrap 握手
- `sglang:decode_bootstrap_time_seconds` — decode bootstrap
- `sglang:decode_alloc_wait_time_seconds` — decode 预分配等待
  （后三者默认桶 `[0.01,0.05,0.1,0.5,1,5,10,30,60,120,300,600]`）
```promql
histogram_quantile(0.99, sum by (le) (rate(sglang:queue_time_seconds_bucket{$sel, role_name=~"(?i)prefill"}[$w])))
histogram_quantile(0.99, sum by (le) (rate(sglang:prefill_bootstrap_time_seconds_bucket{$sel}[$w])))
```
> **拆解逻辑**：TTFT ≈ 排队(`queue_time_seconds`，对应 `prefill_waiting`) + prefill 计算(`prefill_forward`/`chunked_prefill`) + KV 传输(`prefill_transfer_kv_cache`) + decode 侧握手/分配(`decode_bootstrap`/`decode_alloc_wait`) + 首步 forward。逐 stage 报 P90/P99，定位是「排队」还是「传输」还是「计算」主导。

---

## 5. Prefill 吞吐分析

**计算**：prefill 侧处理的**输入 token 速率**（tokens/s）。Counter `sglang:prompt_tokens_total` 每完成一个请求 `inc(prompt_tokens)`（`metrics_collector.py` `observe_one_finished_request`）。

```promql
# 总吞吐 tokens/s
sum(rate(sglang:prompt_tokens_total{$sel, role_name=~"(?i)prefill"}[$w]))
# 按 pod：sum by (pod) (...)
# 请求速率 req/s（配合看输入长度）
sum(rate(sglang:num_requests_total{$sel}[$w]))
```
- 无「prefill 吞吐」gauge；`sglang:gen_throughput` 是 **decode** 的生成吞吐，别用来当 prefill 吞吐。
- 吞吐是速率量，**报均值/峰值**（`max_over_time(...[$w])`）而非请求级分位。
- **⚠️ 角色标签**：`prompt_tokens_total` 由 tokenizer manager 的 `TokenizerMetricsCollector` 上报（`metrics_collector.py:1516`，`observe_one_finished_request` 在请求完成侧调用），label 只有 `model_name` + `extra_metric_labels`，**不含 `engine_type`/`tp_rank`**（`num_requests_total` / `generation_tokens_total` / `uncached_tokens_total` 同源同理）。故按 `engine_type="prefill"` 回退**匹配不到**；能否按 `role_name` 筛取决于平台是否经 `extra_metric_labels` / k8s relabeling 注入。PD 下 prefill/decode 通常是**独立 deploy**，**优先用 `$sel`（deploy_id）圈角色**，别依赖 `engine_type`。

---

## 6. Prefill 节点 cache 命中率分析

**Counter 法（token 加权，推荐，跨窗口稳定）**：
```promql
# 有效前缀缓存命中率(%) = 命中 token / 输入 token × 100
100 *
  sum(rate(sglang:cached_tokens_total{$sel, role_name=~"(?i)prefill"}[$w]))
  / sum(rate(sglang:prompt_tokens_total{$sel, role_name=~"(?i)prefill"}[$w]))
```
- `sglang:cached_tokens_total` 带额外 label `cache_source`（device/host/storage）；不分来源时 `sum` 掉。
- `sglang:raw_cached_tokens_total` = **SWA 裁剪前**的前缀命中 token（滑窗注意力场景下 > 有效命中），比值会偏高，用于看「理论前缀重合度」；`cached_tokens_total` 才是实际省下的算力。
- `sglang:cached_tokens_excluding_abort_total` / `prompt_tokens_excluding_abort_total` = 排除 abort 请求的口径，算「正常请求命中率」时更干净。

**Gauge 法（瞬时，scheduler 上报）**：`sglang:cache_hit_rate`（原始 0–1，× 100 报 %）——
```promql
100 * avg_over_time(sglang:cache_hit_rate{$sel, role_name=~"(?i)prefill"}[$w])   # 或 quantile_over_time / max_over_time
```
> 两法并报：Counter 比值是窗口累计真实命中；gauge 是采样瞬时值。差距大说明命中率在窗口内波动。命中率天然偏低常有**结构性原因**（prompt 太短低于缓存阈值 / 长尾大 prompt 稀释），不等于「缓存坏了」。

---

## 7. TPOT / ITL：token 间延迟分析

**指标**：`sglang:inter_token_latency_seconds`（Histogram，= ITL/TPOT，decode 侧）。
分桶：`[0.002,…,0.1,0.2,0.4,0.6,0.8,1,2,4,6,8]`（最大 8s）。
另有 `sglang:inter_token_latency_seconds_rm_first`（剔除 TTFT 后首个 ITL，更纯的稳态 TPOT）。

```promql
histogram_quantile(0.99, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket{$sel, role_name=~"(?i)decode"}[$w])))
# 稳态 TPOT（推荐）
histogram_quantile(0.90, sum by (le) (rate(sglang:inter_token_latency_seconds_rm_first_bucket{$sel, role_name=~"(?i)decode"}[$w])))
# 均值
rate(sglang:inter_token_latency_seconds_sum{$sel}[$w]) / rate(sglang:inter_token_latency_seconds_count{$sel}[$w])
```
- ITL P99 > 150ms 通常预警（受 batch、投机解码、KV 压力影响，非绝对阈值）。

---

## 8. Decode 吞吐分析

**Counter 法（推荐）**：`sglang:generation_tokens_total` 每完成一个请求 `inc(generation_tokens)`（`metrics_collector.py` `observe_one_finished_request`）。
```promql
sum(rate(sglang:generation_tokens_total{$sel, role_name=~"(?i)decode"}[$w]))   # tokens/s，按 pod 加 by(pod)
```
**Gauge 法**：`sglang:gen_throughput`（scheduler 瞬时生成吞吐 tok/s，PD decode 上报）——
```promql
sum( avg by(pod, dp_rank) (sglang:gen_throughput{$sel, role_name=~"(?i)decode"}) )    # 当前全体（按 pod×dp_rank 去 TP 重复！无 DP 时等价 max by(pod)，见 §0）
avg_over_time( sum(avg by(pod, dp_rank)(sglang:gen_throughput{$sel, role_name=~"(?i)decode"}))[$w:] )   # 窗口均值（subquery）
```
> Counter rate 是窗口平均真实产出；gauge 是采样瞬时，报均值+峰值。gauge 求和须先按 pod×dp_rank 去重（见 §0），否则被 tp_rank 放大 TP 倍。

---

## 9. Decode 并行度（batch size）分析

**指标（Gauge）**：`sglang:num_running_reqs`（当前运行请求数 = 运行 batch size，decode 侧即并发解码数）。辅助：`sglang:decode_sum_seq_lens`（batch 内序列长度总和，反映 KV 压力）、`sglang:num_queue_reqs`（排队数）。

```promql
# 当前并行度（全体/按 pod）—— gauge 按 tp_rank 复制，求和须先按 pod×dp_rank 去重（见 §0；无 DP 时 avg by(pod,dp_rank)≡max by(pod)）
sum( avg by(pod, dp_rank) (sglang:num_running_reqs{$sel, role_name=~"(?i)decode"}) )
# 时间样本分位（Gauge 无请求级分位！）——单 pod 序列的时间分布（分位读 gauge 不受 TP 复制影响）
quantile_over_time(0.90, sglang:num_running_reqs{$sel, role_name=~"(?i)decode"}[$w])
# 窗口均值 / 峰值（按 pod×dp_rank 去重后再聚合）
avg( avg_over_time(avg by(pod, dp_rank)(sglang:num_running_reqs{$sel, role_name=~"(?i)decode"})[$w:]) )
max( max_over_time(avg by(pod, dp_rank)(sglang:num_running_reqs{$sel, role_name=~"(?i)decode"})[$w:]) )
```
> **P50/P70/P90/P99 对 batch size 指的是「时间维度上的分位」**（用 `quantile_over_time`，逐 pod 序列），必须注明；不是请求级分位。`decode_sum_seq_lens` / `num_queue_reqs` 同为 gauge，求和同样先去重（见 §0）。

---

## 10. Decode 推测（投机）解码接受率分析

**指标（Gauge，scheduler 每 decode batch 刷新，仅开启投机解码时非 0）**：
- `sglang:spec_accept_length` = `spec_num_accepted_tokens / spec_num_forward_ct`（`scheduler_metrics_mixin.py:762`）= **每次 forward 平均产出 token 数**（含恒接受的 bonus token，≥1；越大越省）。
- `sglang:spec_accept_rate` = `spec_num_accepted_tokens / (spec_num_forward_ct × num_draft_tokens)`（`:772`）= **草稿 token 接受比例**（原始 0–1，**× 100 报 %**）。`num_draft_tokens` = `speculative_num_draft_tokens`（缺省回退 `speculative_num_steps+1`）。

> 注意 `update_spec_metrics(bs, num_accepted_tokens)` 里 `spec_num_accepted_tokens += num_accepted_tokens + bs`（bonus token 计入，`scheduler_metrics_mixin.py:217-219`），故 accept_length 天然 ≥1，accept_rate 分母含全部草稿位。

```promql
avg_over_time(sglang:spec_accept_length{$sel, role_name=~"(?i)decode"}[$w])              # 每 forward 产出 token 数，非比率，不换算
100 * quantile_over_time(0.50, sglang:spec_accept_rate{$sel, role_name=~"(?i)decode"}[$w])   # 接受率(%)，时间分位
```
- Gauge 分位为时间样本分位（见 §0），注明。指标缺失/恒 0 = **未启用投机解码**，不是故障。

---

## 11. Decode KV Cache 使用率分析

**指标（Gauge，0–1 或绝对 token 数）**：
- `sglang:token_usage` = KV 使用率（已用/最大，原始 0–1，**× 100 报 %**）——**主指标**
- `sglang:num_used_tokens`（已用 KV token 绝对数）、`sglang:max_total_num_tokens`（KV 池容量）→ 可自算 `100 * num_used/max_total`
- `sglang:swa_token_usage`（滑窗注意力池使用率）、`full_token_usage`/`pending_prealloc_token_usage`/`mamba_usage`（对应模型结构才有）

```promql
100 * avg_over_time(sglang:token_usage{$sel, role_name=~"(?i)decode"}[$w])          # 均值(%)
100 * max_over_time(sglang:token_usage{$sel, role_name=~"(?i)decode"}[$w])          # 峰值(%)
100 * quantile_over_time(0.90, sglang:token_usage{$sel, role_name=~"(?i)decode"}[$w])  # 时间分位(%)
100 * max by (pod) (max_over_time(sglang:token_usage{$sel, role_name=~"(?i)decode"}[$w]))  # 按 pod：看单 pod 是否打满
```
- 持续 > **85%** 预警、> **95%** 严重（易触发 retract/OOM），配合 `sglang:num_retracted_reqs` 看是否已发生驱逐回退。Gauge 分位为时间分位（见 §0）。

---

## 12. Decode 复读率指标分析

**指标**：
- Counter `sglang:no_repeat_processor_detected_count_total` — no-repeat 处理器**检测到**复读次数（`custom_logit_processor.py:332`）。**真实序列名带 `_total`**（实测 306 序列）；漏写 `_total` 查 0 序列会误判成「未启用」。
- Counter `sglang:no_repeat_processor_truncated_count_total` — 因复读**被截断**次数（`:417`），同样带 `_total`。
- 请求完成态 `finish_reason="repetition_truncation"`（`schedule_batch.py:207`）

> **⚠️ 这两个 Counter 无任何 sglang app label**：`LogitProcessorCollector` 注册时**未传 `labelnames`**、打点用裸 `.inc()`（不带 `.labels()`，`metrics_collector.py:2375-2383`/`custom_logit_processor.py:332,417`）——**没有 `engine_type`/`role_name`/`tp_rank`**，加 `role_name=~"(?i)decode"` 过滤会**查空**。只有 k8s target label（`deploy_id`/`pod`/`namespace`/`cluster`）会附着；no-repeat 逻辑只在 decode（生成）侧跑，故 `$sel`（deploy_id）圈定即可，**无需也不能**按 role_name 筛。

```promql
# 复读截断率 ≈ 被截断请求 / 完成请求（注意 _count_total 后缀；复读 counter 不带 role_name，只按 $sel 圈）
sum(rate(sglang:no_repeat_processor_truncated_count_total{$sel}[$w]))
  / sum(rate(sglang:num_requests_total{$sel, role_name=~"(?i)decode"}[$w]))
# 检出率（含未截断的检出）
sum(rate(sglang:no_repeat_processor_detected_count_total{$sel}[$w]))
  / sum(rate(sglang:num_requests_total{$sel, role_name=~"(?i)decode"}[$w]))
```
- Counter → 用 rate 比值，无分位。复读常伴随**异常长输出 + 高 ITL/TTFT 尾部**，与 `generation_tokens_histogram` P99、`inter_token_latency` P99 交叉看同现。指标缺失 = 未启用 no-repeat 处理器。

---

## 附：MiMo SGLang 推理服务 SLO 目标（结论对比基准）

线上 MiMo 推理服务的验收目标，按**模型**与**上下文长度档位**分档。TTFT 看 **P70**、TPOT 看 **P50**（`TPOT < 25ms` 等价 `TPS ≥ 40 tok/s`，即 `1 / 0.025`）。

| 模型 | 档位 | 上下文长度 | TTFT P70 | TPOT P50 |
|---|---|---|---|---|
| **MiMo-V2.5** | 短 | 0–64K | ≤ 3s | < 25ms（TPS ≥ 40） |
| | 长 | 64K–256K | ≤ 5s | < 25ms |
| | 超长 | > 256K | ≤ 10s | < 25ms |
| **MiMo-V2.5-Pro** | 短 | 0–64K | ≤ 3s | < 25ms |
| | 长 | 64K–256K | ≤ 6s | < 25ms |
| | 超长 | > 256K | ≤ 10s | < 25ms |

> 两模型仅**长档 TTFT** 不同（V2.5 ≤ 5s，Pro ≤ 6s），其余一致。TPOT P50 全档统一 < 25ms。

**如何对照**：
- **TTFT P70** → §3，分位取 **0.70**：`histogram_quantile(0.70, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{$sel, role_name=~"(?i)decode"}[$w])))`。
- **TPOT P50** → §7，用**稳态** `inter_token_latency_seconds_rm_first` 的 **0.50** 分位（剔首个 ITL 更贴近 TPOT），阈值 `0.025s`。
- **⚠️ 档位口径**：SLO 按**输入长度**分档，但导出的 TTFT/TPOT 直方图是**全量聚合、不按 prompt 长度切片**（§2/§3）。对全量 histogram 直接报的 P70/P50 是**混档结果**，仅当该 `deploy_id`/`model_service_id` 只承接单一长度档时才能直接对应某行 SLO。

---

## 附：Prefill / Decode 扩缩容经验（持续迭代中）

> **⚠️ 经验性阈值，随线上观察持续迭代优化**。「预期」= 上节 SLO 的 TTFT/TPOT 目标（按**模型 × 档位**取对应行）；每条为 **AND** 条件，须在给定**持续时长**内一直满足才触发；`step` = 平台侧扩缩容步长。

### 扩容

| 角色 | 触发条件（AND） | 持续 | 动作 |
|---|---|---|---|
| **Prefill** | TTFT > 预期 ×1.3 **且** Prefill 排队时间 > 3s | 5 min | 扩容 `step` |
| **Prefill** | TTFT > 预期 ×1.6 **且** Prefill 排队时间 > 10s | 5 min | 扩容 `step × 2` |
| **Decode** | TTFT > 预期 ×1.3 **且** Prefill prealloc 队列 > 10 | 1 min | 扩容 `step` |
| **Decode** | TTFT > 预期 ×1.3 **且** Decode waiting 队列 > 1 | 1 min | 扩容 `step` |

### 缩容

| 角色 | 触发条件（AND） | 持续 | 动作 |
|---|---|---|---|
| **Prefill** | TTFT P70 < 预期 ×0.9 **且** Prefill 排队时间 < 1s | 10 min | 缩容 `step` |
| **Decode** | TPOT P50 ≤ SLO ×1.2 **且** Decode KV 使用率 < 70% | 10 min | 缩容 `step` |

### 指标映射（对照本 skill）
- **TTFT**（「预期」= SLO TTFT P70）→ §3 `sglang:time_to_first_token_seconds`（decode 侧观测）。
- **Prefill 排队时间**（时长）→ §4 `sglang:queue_time_seconds`（= `prefill_waiting` stage），取分位判断（建议 P90）；`> 3s / 10s / < 1s` 都是**时长**阈值，别与「队列条数」混。
- **Prefill prealloc 队列**（条数）→ catalog PD 队列 gauge `sglang:num_prefill_prealloc_queue_reqs`（瞬时数量；求和须按 pod×dp_rank 去重，见 §0）。
- **Decode waiting 队列**（条数）→ `sglang:num_decode_prealloc_queue_reqs`（decode 预分配等待队列）；若口径不符可换 `sglang:num_queue_reqs`（decode 侧），**精确 metric 待线上确认**。
- **TPOT P50**（SLO = 25ms → ×1.2 = 30ms）→ §7 稳态 `sglang:inter_token_latency_seconds_rm_first` 的 P50。
- **Decode KV 使用率** → §11 `sglang:token_usage`（×100 %），阈值 70%。

---

## 附：分析该覆盖的顺序（PD 视角）

1. **覆盖面**：service/deploy、cluster、namespace、各 role 的 pod 数（§1）、时间窗与时区。
2. **P/D 分工体检**：prefill 吞吐/命中（§5/§6）、queue/bootstrap/transfer 各 stage（§4）→ 定位 TTFT 瓶颈在 P 还是传输。
3. **Decode 稳态**：TPOT/ITL（§7）、batch size（§9）、KV 使用率（§11）、投机接受率（§10）→ 定位吞吐与延迟权衡。
4. **异常/长尾**：复读（§12）、`num_retracted_reqs`/`num_aborted_requests_total`、输出长度 P99（§2）。
5. 每条结论标注：指标名、**指标类型**、实际 PromQL、时间窗、role/pod、原始值；分位区分「请求级(Histogram)」还是「时间/跨pod(Gauge)」。**无数据写「无序列」，绝不写 0。**

完整指标名/类型/分桶/标签清单见 [`references/metrics-catalog.md`](references/metrics-catalog.md)。
按 pod 维度分析 TTFT/TPOT/P·D 吞吐的端到端示例见 [`references/example-per-pod-analysis.md`](references/example-per-pod-analysis.md)。
