# 使用示例：按 pod 维度分析 TTFT / TPOT / P·D 吞吐

> 一个端到端的真实 walk-through：**「分析 `s-20260608115222-cb8b6-j2qqv` 短集群最近 5 分钟的 TTFT / TPOT / prefill / decode 吞吐，按 pod 维度展开」**。演示 skill 的标准流程——① 定位集群路由 → ② 按指标类型选对 PromQL → ③ 按 pod 分组 → ④ 结论区分请求级 / 时间级分位、对齐 SLO。下面数字是某次 5 分钟窗口快照（`$w=5m`），仅示范格式，实际值随时间变化。

## ① 集群路由发现（依赖 `cloudml-mimo-analyze`）

```bash
python3 <cloudml-mimo-analyze>/scripts/analyze_mimo.py \
  --target s-20260608115222-cb8b6-j2qqv --discover-only
```
拿到：`cluster=ak-cloudml-prod-2`、`cluster_api_base=https://prometheus.ak-cloudml-prod-2.cc.d.xiaomi.net/api/v1`、`role_name` 存在（prefill/decode）、`dp_rank` 为空（→ gauge 求和用 `max by(pod)` 即可，见 §0；本例四项均为 Counter/Histogram，无 TP 去重问题）。
- `$sel` = `deploy_id="s-20260608115222-cb8b6-j2qqv"`
- 覆盖面：**9 个 prefill pod + 6 个 decode pod**（`count(count by(pod)(...))`，§1）。

## ② + ③ 四项指标的 per-pod PromQL

| 指标 | 类型 | PromQL（`by(pod)` 展开） |
|---|---|---|
| Prefill 吞吐 | Counter → rate | `sum by(pod)(rate(sglang:prompt_tokens_total{$sel, role_name=~"(?i)prefill"}[$w]))` |
| Decode 吞吐 | Counter → rate | `sum by(pod)(rate(sglang:generation_tokens_total{$sel, role_name=~"(?i)decode"}[$w]))` |
| TTFT | Histogram → quantile | `histogram_quantile(q, sum by(le,pod)(rate(sglang:time_to_first_token_seconds_bucket{$sel, role_name=~"(?i)decode"}[$w])))` |
| TPOT（稳态） | Histogram → quantile | `histogram_quantile(q, sum by(le,pod)(rate(sglang:inter_token_latency_seconds_rm_first_bucket{$sel, role_name=~"(?i)decode"}[$w])))` |

> 关键点：**吞吐是 Counter → 只能 `rate`，报速率/峰值，不编分位**；**TTFT/TPOT 是 Histogram → `histogram_quantile`，是按请求/token 加权的真·请求级分位**。TTFT/TPOT 都在 **decode 角色**观测（prefill 不发这两个），故按 decode pod 展开。per-pod 就是把聚合里的 `by(...)` 加上 `pod`（分位则是 `by(le,pod)`）。

## ④ 结果（快照）+ 结论

**Prefill 吞吐（tok/s，9 pod，Counter rate）**：228k / 227k / 278k / 247k / 281k / 260k / 271k / 267k / 256k → 合计 ≈ **2.31 M tok/s**，pod 间 227k–281k，**均衡（极差 ~24%，无热点）**。

**Decode 吞吐（tok/s，6 pod，Counter rate）**：9351 / 9340 / 9131 / 8577 / 9007 / 8339 → 合计 ≈ **53.7 k tok/s**，pod 间 8.3k–9.4k，**均衡**。

**TTFT（s，decode pod，请求级分位）**：

| pod | P50 | P90 | P99 |
|---|---|---|---|
| decode-0 | 1.04 | 3.08 | 6.26 |
| decode-1 | 1.06 | 3.29 | 7.22 |
| decode-2 | 1.03 | 3.11 | 6.00 |
| decode-3 | 1.04 | 3.20 | 5.93 |
| decode-4 | 1.10 | 3.25 | 6.83 |
| decode-5 | 1.07 | 3.14 | 5.82 |

**TPOT / ITL 稳态（s，`_rm_first`，decode pod，请求级分位）**：

| pod | P50 | P90 | P99 |
|---|---|---|---|
| decode-0 | 0.0080 | 0.0246 | 0.0595 |
| decode-1 | 0.0081 | 0.0247 | 0.0597 |
| decode-2 | 0.0079 | 0.0239 | 0.0585 |
| decode-3 | 0.0084 | 0.0243 | 0.0588 |
| decode-4 | 0.0078 | 0.0238 | 0.0589 |
| decode-5 | 0.0078 | 0.0239 | 0.0586 |

**判读（对齐短档 SLO：TTFT P70 ≤ 3s、TPOT P50 < 25ms）**：
- **TPOT P50 ≈ 8 ms**，远优于 25 ms SLO；P99 ≈ 59 ms 也健康。6 个 decode pod 几乎完全一致 → decode 负载均衡、无慢节点。
- **TTFT P50 ≈ 1.05 s** 达标；**P90 ≈ 3.1–3.3 s** 已贴近/略超 3s（但 SLO 卡在 P70，P70 需单独查——升序表头 `P50|P70|P90|P99` 里 P70 才是判定位）。TTFT 长尾（P99 ~6–7s）建议按 §4 做 stage 拆解（`queue_time_seconds` / `prefill_forward` / `prefill_transfer_kv_cache` / `decode_bootstrap`）定位是排队、计算还是 KV 传输主导。
- pod 间 TTFT/TPOT 分位高度一致 → **无单 pod 异常**；吞吐极差 <25% → **P、D 两侧路由均衡**。

## 复用要点
- **先类型后分位**：Counter 报 rate（吞吐），Histogram 才有请求级分位（TTFT/TPOT）。
- **per-pod = 加 `pod` 分组**；分位查询是 `sum by(le, pod)`（保留 `le` 才能 `histogram_quantile`）。
- **TTFT/TPOT 只在 decode 角色**；prefill 侧对 TTFT 的贡献走 §4 stage 拆解。
- 表头**升序 `P50|P70|P90|P99`**；空窗口写「无序列」不写 0（§0 看板 0≠真 0）。
