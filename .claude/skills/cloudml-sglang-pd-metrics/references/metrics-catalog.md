# SGLang PD 指标目录（对齐 sglang 源码）

> **⚠️ 依赖 `cloudml-mimo-analyze`**（集群路由探测、看板 PromQL 白名单、`deploy_id`/`service_id` 识别；安装方法见 `SKILL.md` 顶部）。本目录只列指标名/类型/分桶/标签。

来源：`python/sglang/srt/observability/metrics_collector.py`、`scheduler_metrics_mixin.py`、`req_time_stats.py`、`sampling/custom_logit_processor.py`。类型决定能否分位：**Histogram** 可 `histogram_quantile`（请求/token 级分位）；**Gauge** 只能 `quantile_over_time`/`quantile`（时间或跨pod分位）；**Counter** 只能 `rate`。

## 基础标签（scheduler 侧 sglang: 指标）
`model_name, engine_type(prefill|decode|unified), tp_rank, pp_rank, moe_ep_rank[, dp_rank][, priority]` + server_args 注入的 `extra_metric_labels`。CloudML 映射注入：`role_name, deploy_id, model_service_id, pod, namespace, cluster`。
> **encoder 角色例外（EPD 分离）**：`role_name="encoder"` 的多模态编码器由独立 `encode_server.py` + `EncoderMetricsCollector` 上报，base labels **只有 `model_name`（+`extra_metric_labels`）——无 `engine_type`/`tp_rank`/`pp_rank`**（`encode_server.py:3813`）。encoder **不上报** scheduler 系列指标，只上报下方 `sglang:encoder_*` 族；因无 tp_rank，`count by(pod)`/`sum by(pod)` 无 TP 去重问题。

## Histogram（可请求级分位）
| 指标 | 语义 | 分桶 | 观测侧 | 额外 label |
|---|---|---|---|---|
| `sglang:time_to_first_token_seconds` | TTFT 首 token | 0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10,20,40,60,80,100,200,400 | decode | `is_streaming` |
| `sglang:inter_token_latency_seconds` | ITL/TPOT | 0.002…0.1,0.2,0.4,0.6,0.8,1,2,4,6,8 | decode | — |
| `sglang:inter_token_latency_seconds_rm_first` | 稳态 TPOT（剔首个） | 同上 | decode | — |
| `sglang:e2e_request_latency_seconds` | 端到端请求延迟 | — | 入口 | — |
| `sglang:per_stage_req_latency_seconds` | 各阶段延迟 | exponential(0.001, ×1.62, 30) | 各角色 | **`stage`** |
| `sglang:queue_time_seconds` | 排队时长 | 0.0…3000 | prefill | — |
| `sglang:prefill_bootstrap_time_seconds` | prefill bootstrap | 0.01,0.05,0.1,0.5,1,5,10,30,60,120,300,600 | prefill | — |
| `sglang:decode_bootstrap_time_seconds` | decode bootstrap | 同上 | decode | — |
| `sglang:decode_alloc_wait_time_seconds` | decode 预分配等待 | 同上 | decode | — |
| `sglang:prompt_tokens_histogram` | 输入长度 | — | 入口/prefill | — |
| `sglang:generation_tokens_histogram` | 输出长度 | — | decode | — |
| `sglang:uncached_tokens_per_request` | 未命中输入 token（**名带 `_per_request`**；另有 Counter `uncached_tokens_total`） | — | prefill | — |
| `sglang:num_retractions` | 每请求 retract 次数 | — | decode | — |
| `sglang:pd_commit_wait_seconds` | prefill 侧等 decode COMMIT 时长 | 0.01,0.05,0.1,0.5,1,5,10,30,60,120,300,600 | prefill | — |
| `sglang:pd_decode_prefill_sched_wait_seconds` | decode 侧等 PREFILL_SCHED 时长 | 同上 | decode | — |
| `sglang:kv_transfer_speed_gb_s` | KV 传输速率 GB/s | 0.1,0.5,1,5,10,25,50,100,200,400 | prefill | — |
| `sglang:kv_transfer_latency_ms` | KV 传输延迟 ms | 1,2,5,10,25,50,100,250,500,1000,2500,5000 | prefill | — |

`stage` 全量取值（`metrics_is_observed=True` 才进 `per_stage_req_latency_seconds`）：
`pre_process, request_process, prefill_bootstrap, prefill_transfer_kv_cache, prefill_forward, chunked_prefill, decode_prepare, decode_bootstrap, decode_waiting, decode_transferred, decode_h2d_wait, fake_output, quick_finish, hicache_prefetch_{wait_total,queue,query,resolve,io_queue,io,commit}`。
（trace-only / 未进本指标：`prefill_waiting`（=排队，`metrics_is_observed=False`，改查 `queue_time_seconds`）、`prefill_prepare`（默认 False）、`prefill_schedule/gpu_forward/cache_unfinished/send_kv/chunk_*`、`decode_forward/loop`、`tokenize/entrypoint_preprocess/tokenize_preprocess/dispatch/dc_dispatch/mm_encode_wait`。）

## Gauge（瞬时；分位只能时间/跨pod）
| 指标 | 语义 |
|---|---|
| `sglang:num_running_reqs` | 运行请求数 = 运行 batch size（decode 并行度） |
| `sglang:num_queue_reqs` | 排队请求数 |
| `sglang:token_usage` | KV 使用率 原始0–1（**Decode KV 主指标**，报表 ×100 用 %） |
| `sglang:num_used_tokens` / `sglang:max_total_num_tokens` | 已用 / KV 池容量（可自算使用率） |
| `sglang:swa_token_usage` | 滑窗注意力池使用率 |
| `sglang:full_token_usage` / `sglang:pending_prealloc_token_usage` / `sglang:mamba_usage` | full-attn 池 / 预分配挂起 / mamba 状态池 使用率（对应模型结构才有） |
| `sglang:decode_sum_seq_lens` | batch 内序列长度总和 |
| `sglang:gen_throughput` | decode 瞬时生成吞吐 tok/s |
| `sglang:cache_hit_rate` | scheduler 瞬时前缀命中率 原始0–1（报表 ×100 用 %） |
| `sglang:spec_accept_length` | 投机：accepted/forward_ct（每 forward 产出 token，≥1；**非比率，不换算 %**） |
| `sglang:spec_accept_rate` | 投机：accepted/(forward_ct×num_draft) 草稿接受比 原始0–1（报表 ×100 用 %） |
| `sglang:utilization` | 综合利用率 |
| `sglang:new_token_ratio` | 新 token 比（chunked prefill 调度） |
| `sglang:num_retracted_reqs` / `sglang:num_paused_reqs` | 回退 / 暂停请求数 |
| `sglang:req_pool_usage` | 请求池占用 |
| PD 队列：`sglang:num_prefill_prealloc_queue_reqs`、`num_prefill_inflight_queue_reqs`、`num_prefill_total_reqs`、`num_decode_prealloc_queue_reqs`、`num_decode_prefill_sched_wait_queue_reqs`、`num_decode_transfer_queue_reqs`、`num_decode_h2d_queue_reqs` | 各阶段队列深度（decode 侧顺序：prealloc → 等 prefill 调度 `prefill_sched_wait` → transfer → `h2d`） |
| PD 传输（RDMA，Gauge）：`sglang:kv_transfer_rdma_speed_gb_s`、`sglang:kv_transfer_rdma_latency_ms` | RDMA KV 传输速率/延迟（非 RDMA 版是 Histogram，见上表） |

## Counter（只能 rate）
| 指标 | 语义 | 额外 label |
|---|---|---|
| `sglang:prompt_tokens_total` | 输入 token 累计（每批 inc(prompt_tokens)）→ **prefill 吞吐** | — |
| `sglang:generation_tokens_total` | 输出 token 累计 → **decode 吞吐** | — |
| `sglang:cached_tokens_total` | 命中缓存 token（有效）→ **cache 命中率分子** | `cache_source`(device/host/storage) |
| `sglang:raw_cached_tokens_total` | SWA 裁剪前前缀命中 token（偏高） | — |
| `sglang:prompt_tokens_excluding_abort_total` / `sglang:cached_tokens_excluding_abort_total` | 排除 abort 口径 | — |
| `sglang:num_requests_total` | 完成请求数（分母） | — |
| `sglang:num_aborted_requests_total` | abort 请求数 | — |
| `sglang:no_repeat_processor_detected_count_total` | 检测到复读次数（**真实序列带 `_total` 后缀**，实测 306 序列；写 `_count`（无 `_total`）查 0 序列） | — |
| `sglang:no_repeat_processor_truncated_count_total` | 因复读截断次数（同上，`_total` 后缀） | — |
| PD 失败：`sglang:num_bootstrap_failed_reqs_total`、`sglang:num_transfer_failed_reqs_total` | bootstrap/传输失败 | — |

## Encoder 指标族（EPD 分离，`role_name="encoder"` 专属）

> 由独立 `encode_server.py` + `EncoderMetricsCollector` 上报（`metrics_collector.py`）。**base labels 只有 `model_name`（+`extra_metric_labels`），无 `engine_type`/`tp_rank`/`pp_rank`**——**不按 TP 复制，`count by(pod)`/`sum by(pod)`/`sum(rate)` 天然正确，无 §0 去重问题**。encoder **不上报** scheduler 系列（`num_running_reqs`/`token_usage`/`max_total_num_tokens`/`time_to_first_token_seconds`/`inter_token_latency_seconds`/`prompt_tokens_total`/`generation_tokens_total` 等一律没有），§2–§12 方法对 encoder **不适用**。多数指标带额外 label **`modality`**（image/audio/video…），个别再带 `status`/`dp_rank`。**节点数**：`count( count by(pod) (sglang:encoder_cache_size_mb{$sel, role_name=~"(?i)encoder"}) )`（生产看板 EPD 面板同款底指标）。

### Histogram（可请求级分位，`histogram_quantile`）
| 指标 | 语义 | 分桶 | 额外 label |
|---|---|---|---|
| `sglang:encoder_e2e_request_latency_seconds` | encoder 端到端请求延迟 | 覆盖到 2400s（`bucket_encoder_e2e_latency`） | —（仅 base） |
| `sglang:encoder_request_e2e_seconds` | 每请求 e2e（as-experienced，勿跨请求求和） | 0.01,0.05,0.1,0.2,0.5,1,2,5,10,30 | `modality` |
| `sglang:encoder_batch_e2e_latency_seconds` | 每 batch e2e | 0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,30,60 | `modality` |
| `sglang:encoder_queue_wait_seconds` | 调度队列等待 | 0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10 | `modality` |
| `sglang:encoder_download_seconds` | 拉取媒体字节（HTTP GET/base64/文件读） | 0.01,0.05,0.1,0.2,0.5,1,2,5,10,30 | `modality` |
| `sglang:encoder_decode_seconds` | 字节→原始媒体（PIL/nvJPEG/音视频解码） | 同上 | `modality` |
| `sglang:encoder_preprocess_seconds` | HF processor 特征提取（不含 download/decode） | 同上 | `modality` |
| `sglang:encoder_model_forward_seconds` | GPU 前向 | 0.01,0.02,0.05,0.1,0.2,0.5,1,2,5 | `modality` |
| `sglang:encoder_transfer_seconds` | 每 part embedding 传给 prefill 节点 | 覆盖到 2400s | `modality` |
| `sglang:encoder_p2p_part_e2e_seconds` | 每 part P→E→P e2e（prefill 发→prefill 收） | 覆盖到 2400s | `modality` |
| `sglang:encoder_cache_read_seconds` / `sglang:encoder_cache_write_seconds` | 缓存读/写延迟 | — | `modality` |
| `sglang:encoder_mm_items_per_batch` | 每 batch 多模态 item 数 | 1..16,32,64,128 | `modality` |
| `sglang:encoder_mm_items_per_request` | 每请求多模态 item 数 | 1..16,24,32,64 | `modality` |
| `sglang:encoder_embedding_tokens_per_send` | 每次发给 prefill 的 embedding token（首维） | 16,64,128…65536 | `modality` |

> 另有 `sglang:p2p_encoder_e2e_request_latency_seconds`（注意前缀是 `p2p_encoder_` 非 `encoder_`；含网络的 P2P e2e，带 `modality`）。
> **阶段分解链**：`download → decode → preprocess → queue_wait → model_forward → transfer(→prefill)`；`request_e2e` 是整请求口径，`batch_e2e` 是批口径。

### Gauge（瞬时；分位只能时间/跨pod）
| 指标 | 语义 | 额外 label |
|---|---|---|
| `sglang:encoder_cache_size_mb` | 编码缓存占用 MB（`multiprocess_mode="mostrecent"`；**节点数底指标**） | —（仅 base） |
| `sglang:encoder_cache_entries` | 缓存条目数 | —（仅 base） |
| `sglang:encoder_cache_hit_rate` | 缓存命中率（原始 0–1，×100 报 %；`hit_tokens/total_tokens` 实时刷） | `modality` |
| `sglang:encoder_dp_pending_requests` | DP rank 待处理请求数（**事件驱动刷新，非周期**；定义于 `encode_server.py`） | `dp_rank` |
| `sglang:encoder_dp_pending_weight` | DP rank 待处理权重 | `dp_rank` |

### Counter（只能 rate）
| 指标 | 语义 | 额外 label |
|---|---|---|
| `sglang:encoder_requests_received_total` | 收到请求数（到达即 +1，早于处理）→ **per-encoder QPS = `rate(...[$w])`** | `dp_rank`, `modality` |
| `sglang:encoder_requests_total` | 完成请求数（按结果） | `modality`, `status`(success/error) |
| `sglang:encoder_cache_hit_tokens_total` / `sglang:encoder_cache_total_tokens_total` | 缓存命中 / 总 token（命中率分子/分母，token 加权） | `modality` |
| `sglang:encoder_cache_hit_files_total` / `sglang:encoder_cache_total_files_total` | 缓存命中 / 总文件数（文件粒度命中率） | `modality` |
| `sglang:encoder_cache_evictions_total` | 缓存驱逐次数 | `modality` |

## 投机解码计算（scheduler_metrics_mixin.py：`update_spec_metrics` 累加 :217-219，结算 :759-779）
```
update_spec_metrics(bs, num_accepted_tokens):
    spec_num_accepted_tokens += num_accepted_tokens + bs   # 含 bonus token
    spec_num_forward_ct       += bs
# 每 batch 结算：
spec_accept_length = spec_num_accepted_tokens / spec_num_forward_ct
num_draft_tokens   = server_args.speculative_num_draft_tokens or (speculative_num_steps + 1)
spec_accept_rate   = spec_num_accepted_tokens / (spec_num_forward_ct * num_draft_tokens)
```

## finish_reason 复读
`finish_reason="repetition_truncation"`（schedule_batch.py:207）；其余 `stop` / `length` 等。

## 分位速查
- Histogram：`histogram_quantile(q, sum by(le[,pod|stage]) (rate(X_bucket[$w])))`；均值 `rate(X_sum)/rate(X_count)`。
- Gauge 时间分位：`quantile_over_time(q, X[$w])`（逐序列）；跨 pod 瞬时分位：`quantile(q, X)`；汇总后时间分位用 subquery `quantile_over_time(q, sum(X)[$w:])`。
- Counter：`sum(rate(X[$w]))`，占比两 rate 相除。
- q ∈ {0.50, 0.70, 0.90, 0.99}，表头升序 `P50|P70|P90|P99`。
