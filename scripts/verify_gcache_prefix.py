#!/usr/bin/env python3
"""Verify prefix reuse through a live PD service, using only Python's stdlib.

Sequence: 32K chat -> flush P's L1/L2 -> 42K chat with a 30K common prefix.
Use --shared-tokens 32768 to retain the complete first prompt and append a
10K conversation suffix, including chat-template tokens.
K means 1024 tokens, including the chat template. Token IDs are measured with
P's /v1/tokenize endpoint. /flush_cache does NOT clear the storage backend.
Use --check-l1 for an additional identical 42K request without a flush.

Run against an otherwise idle service. Supply the P endpoint actually used by
the chat gateway. Artifacts include requests, token IDs, responses, and metrics.
Exit codes: 0 = GCache verified; 1 = execution error; 2 = GCache not enabled;
3 = verification failed or evidence is inconclusive.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

CONFIG_KEYS = (
    "served_model_name",
    "disaggregation_mode",
    "dp_size",
    "tp_size",
    "page_size",
    "enable_cache_report",
    "enable_metrics",
    "disable_radix_cache",
    "enable_hierarchical_cache",
    "hicache_storage_backend",
    "hicache_host_memory_mode",
    "hicache_write_policy",
    "hicache_storage_prefetch_policy",
    "hicache_storage_backend_extra_config",
)
METRIC_PREFIXES = (
    "sglang:gcache_",
    "sglang:storage_",
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:num_requests_total",
    "sglang:prefill_effective_tokens_total",
    "sglang:hicache_",
    "sglang:backuped_tokens_total",
    "sglang:prefetched_tokens_total",
)
SAMPLE_RE = re.compile(r"^([\w:]+)(?:\{(.*)\})?\s+(\S+)")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def common_prefix(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def parse_metrics(text):
    samples = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if not name.startswith(METRIC_PREFIXES) or not name.endswith(
            ("_total", "_count", "_sum")
        ):
            continue
        # Preserve TP/DP labels: token counters from TP ranks must not be summed.
        label_items = tuple(sorted(LABEL_RE.findall(labels or "")))
        samples[name, label_items] = float(value)
    return samples


def metric_delta(before, after):
    return [
        {
            "name": name,
            "labels": dict(labels),
            "delta": value - before.get((name, labels), 0),
        }
        for (name, labels), value in sorted(after.items())
        if value != before.get((name, labels), 0)
    ]


def metric_total(rows, name, **labels):
    return sum(
        row["delta"]
        for row in rows
        if row["name"] == name
        and all(row["labels"].get(k) == v for k, v in labels.items())
    )


class Client:
    def __init__(self, timeout, token=None):
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def open(self, url, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=self.headers)
        try:
            return self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {message[:2000]}") from exc

    def text(self, url, payload=None):
        with self.open(url, payload) as response:
            return response.read().decode()

    def json(self, url, payload=None):
        return json.loads(self.text(url, payload))


class Verification:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.p = args.prefill_url.rstrip("/")
        self.control = Client(
            args.control_timeout, os.environ.get("SGLANG_ADMIN_API_KEY")
        )
        self.chat = Client(args.request_timeout, os.environ.get("OPENAI_API_KEY"))
        self.run_id = uuid.uuid4().hex
        self.report = {
            "run_id": self.run_id,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "chat_url": args.chat_url,
            "prefill_url": self.p,
        }
        self.model = args.model

    def snapshot(self, name):
        raw = self.control.text(self.p + "/metrics")
        (self.out / f"{name}.prom").write_text(raw)
        return parse_metrics(raw)

    def tokenize(self, messages):
        result = self.control.json(
            self.p + "/v1/tokenize",
            {
                "model": self.model,
                "messages": messages,
            },
        )
        ids = result["tokens"]
        if not isinstance(ids, list) or result["count"] != len(ids):
            raise RuntimeError("Unexpected /v1/tokenize response")
        return ids

    def make_prompts(self):
        # Each repeated fragment must contribute one token in this tokenizer.
        # Calibrate with small chats, then verify both complete token sequences.
        header = (
            f"Cache verification {self.run_id}. Read the data and reply only OK. Data:"
        )
        suffix = "\nEnd of data. Reply only OK."

        def content(common, tail, branch):
            return header + " x" * common + branch * tail + suffix

        def user_message(text):
            return [{"role": "user", "content": text}]

        if self.args.shared_tokens == self.args.first_tokens:
            small = self.tokenize(user_message(content(16, 0, "")))
            first_count = self.args.first_tokens - len(small) + 16
            if first_count <= 0:
                raise RuntimeError("Token budget is too small for the chat template")
            first = user_message(content(first_count, 0, ""))

            def extend(tail):
                # Preserve the first prompt's user-end / assistant-start tokens.
                # Appending to the user text alone would replace those tokens
                # and can miss the final SWA checkpoint despite a shared body.
                return first + [
                    {"role": "assistant", "content": "OK"},
                    {
                        "role": "user",
                        "content": "Additional data:" + " z" * tail + suffix,
                    },
                ]

            tail_count = self.args.second_tokens - len(self.tokenize(extend(16))) + 16
            if tail_count <= 0:
                raise RuntimeError("Suffix budget is too small for the chat template")
            prompts = [first, extend(tail_count)]
            self.report["prompt_construction"] = "append_chat_turn"
        else:
            small = [
                self.tokenize(user_message(content(16, 16, branch)))
                for branch in (" y", " z")
            ]
            prefix = common_prefix(*small)
            common_count = self.args.shared_tokens - (prefix - 16)
            tails = [
                target - self.args.shared_tokens - (len(ids) - prefix - 16)
                for target, ids in zip(
                    (self.args.first_tokens, self.args.second_tokens), small
                )
            ]
            if min(common_count, *tails) <= 0:
                raise RuntimeError("Token budgets are too small for the chat template")
            prompts = [
                user_message(content(common_count, tail, branch))
                for tail, branch in zip(tails, (" y", " z"))
            ]
            self.report["prompt_construction"] = "divergent_suffixes"
        tokens = [self.tokenize(prompt) for prompt in prompts]
        actual = (len(tokens[0]), len(tokens[1]), common_prefix(*tokens))
        expected = (
            self.args.first_tokens,
            self.args.second_tokens,
            self.args.shared_tokens,
        )
        if actual != expected:
            raise RuntimeError(
                f"Tokenizer calibration failed: expected {expected}, got {actual}"
            )
        for i, (messages, ids) in enumerate(zip(prompts, tokens), 1):
            save_json(self.out / f"messages_{i}.json", messages)
            text = (
                messages[0]["content"]
                if len(messages) == 1
                else "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)
            )
            (self.out / f"prompt_{i}.txt").write_text(text)
            save_json(self.out / f"tokens_{i}.json", ids)
        self.report["tokens"] = dict(zip(("first", "second", "common_prefix"), actual))
        self.report["expected_storage_hit_tokens"] = (
            actual[2] // self.page_size * self.page_size
        )
        self.report["expected_storage_hit_rate"] = (
            self.report["expected_storage_hit_tokens"] / actual[1]
        )
        log(
            f"Verified tokens: first={actual[0]}, second={actual[1]}, common prefix={actual[2]}"
        )
        return prompts

    def request(self, name, messages, expected_tokens):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_cached_tokens_details": True,
            "routed_dp_rank": self.args.dp_rank,
        }
        save_json(self.out / f"{name}.request.json", payload)
        before = self.snapshot(f"{name}.before")
        started = time.monotonic()
        log(
            f"Sending {name}: {expected_tokens} prompt tokens, requested DP rank {self.args.dp_rank}"
        )
        usage, cache_details, first_event = None, None, None
        events = []
        done = False
        with self.chat.open(self.args.chat_url, payload) as response:
            with (self.out / f"{name}.response.sse").open("wb") as raw:
                for line in response:
                    raw.write(line)
                    raw.flush()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    if not data:
                        continue
                    event = json.loads(data)
                    events.append(event)
                    if "error" in event:
                        raise RuntimeError(f"Generation error: {event['error']}")
                    if first_event is None:
                        first_event = time.monotonic() - started
                    if event.get("usage"):
                        usage = event["usage"]
                    detail = (event.get("sglext") or {}).get("cached_tokens_details")
                    if detail is not None:
                        cache_details = detail
        elapsed = time.monotonic() - started
        save_json(self.out / f"{name}.response.json", events)
        if not done or usage is None:
            raise RuntimeError(
                f"{name}: missing final SSE usage or [DONE]; inspect saved response"
            )
        if usage.get("prompt_tokens") != expected_tokens:
            raise RuntimeError(f"{name}: gateway tokenization differs from P: {usage}")
        # Tokenizer metrics update immediately after emitting the response.
        time.sleep(1)
        after = self.snapshot(f"{name}.after")
        delta = metric_delta(before, after)
        save_json(self.out / f"{name}.metrics_delta.json", delta)
        response_cached = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        p_requests = metric_total(delta, "sglang:num_requests_total")
        p_prompt = metric_total(delta, "sglang:prompt_tokens_total")
        source_hits = {
            source: metric_total(
                delta, "sglang:cached_tokens_total", cache_source=source
            )
            for source in ("device", "host", "storage", "total")
        }
        isolated = p_requests == 1 and p_prompt == expected_tokens
        reset = any(row["delta"] < 0 for row in delta)
        cached = response_cached
        cached_source = "response_usage" if cached is not None else "unavailable"
        # SGLang omits prompt_tokens_details for a zero-hit text request.
        # Only use P's counters when exactly this request was observed there.
        if cached is None and isolated and not reset:
            cached = int(sum(source_hits.values()))
            cached_source = "isolated_prefill_metrics"
        summary = {
            "usage": usage,
            "cached_tokens": cached,
            "response_cached_tokens": response_cached,
            "cached_tokens_source": cached_source,
            "cache_hit_rate": cached / expected_tokens if cached is not None else None,
            "response_cache_details": cache_details,
            "elapsed_seconds": elapsed,
            "first_event_seconds": first_event,
            "prefill_request_delta": p_requests,
            "prefill_prompt_tokens_delta": p_prompt,
            "isolated_prefill_metrics": isolated,
            "prefill_cached_tokens_by_source": source_hits,
            "gcache_backup_bytes": metric_total(
                delta, "sglang:gcache_backup_bytes_total"
            ),
            "gcache_prefetch_bytes": metric_total(
                delta, "sglang:gcache_prefetch_bytes_total"
            ),
            "scheduler_tokens_by_rank": [
                r for r in delta if r["name"] == "sglang:prefill_effective_tokens_total"
            ],
            "counter_reset_detected": reset,
        }
        self.report[name] = summary
        save_json(self.out / "report.json", self.report)
        rate = "unavailable" if cached is None else f"{cached / expected_tokens:.4%}"
        log(
            f"{name}: cached={cached}/{expected_tokens}, hit rate={rate}, P sources={source_hits}, latency={elapsed:.2f}s"
        )
        if not isolated:
            log(
                f"WARNING: P saw {p_requests:g} requests / {p_prompt:g} tokens; metrics cannot be attributed to this request alone"
            )
        return before, after

    def wait_backup(self, baseline):
        if not self.gcache_enabled:
            self.report["writeback"] = {"status": "not_enabled"}
            return
        log("Waiting for GCache writeback metrics to advance and remain stable")
        started = changed = time.monotonic()
        previous = None
        observed_bytes = 0
        while time.monotonic() - started < self.args.writeback_timeout:
            current = self.snapshot("writeback.latest")
            delta = metric_delta(baseline, current)
            observed_bytes = metric_total(delta, "sglang:gcache_backup_bytes_total")
            signature = sorted(
                (key, value) for key, value in current.items() if "backup" in key[0]
            )
            if signature != previous:
                previous, changed = signature, time.monotonic()
            if (
                observed_bytes > 0
                and time.monotonic() - changed >= self.args.settle_seconds
            ):
                self.report["writeback"] = {
                    "status": "metrics_settled",
                    "backup_bytes": observed_bytes,
                    "wait_seconds": time.monotonic() - started,
                    "note": "A quiet metric interval is observed; it is not a storage durability guarantee.",
                }
                return
            time.sleep(2)
        self.report["writeback"] = {
            "status": "unconfirmed",
            "backup_bytes": observed_bytes,
        }
        log(
            "WARNING: GCache writeback did not settle; a storage miss will need further investigation"
        )

    def run(self):
        info = self.control.json(self.p + "/server_info")
        model_info = self.control.json(self.p + "/model_info")
        self.model = self.model or model_info["served_model_name"]
        states = info.get("internal_states") or [info]
        self.report["prefill_config"] = {k: info.get(k) for k in CONFIG_KEYS}
        self.report["scheduler_configs"] = [
            {k: s.get(k) for k in CONFIG_KEYS} for s in states
        ]
        # Read live per-scheduler state as storage can be attached after startup.
        self.gcache_enabled = all(
            s.get("enable_hierarchical_cache")
            and s.get("hicache_storage_backend") == "gcache"
            for s in states
        )
        self.page_size = int(info["page_size"])
        if info.get("disaggregation_mode") != "prefill":
            raise RuntimeError("--prefill-url must point to a Prefill server")
        if not 0 <= self.args.dp_rank < int(info.get("dp_size", 1)):
            raise RuntimeError("Requested DP rank is out of range")
        if not info.get("enable_cache_report") or not info.get("enable_metrics"):
            raise RuntimeError("P requires --enable-cache-report and --enable-metrics")
        self.report["gcache_enabled"] = self.gcache_enabled
        log(
            f"P: model={self.model}, page_size={self.page_size}, GCache enabled={self.gcache_enabled}"
        )
        if not self.gcache_enabled:
            log(
                "GCache is NOT enabled. Running the sequence to measure current behavior; expected exit code is 2."
            )
        prompts = self.make_prompts()
        first_before, _ = self.request("first_32k", prompts[0], self.args.first_tokens)
        self.wait_backup(first_before)
        self.snapshot("before_flush")
        log("POST /flush_cache: clearing P GPU/host caches; retaining L3 storage")
        flush_text = self.control.text(self.p + "/flush_cache?timeout=30", {})
        (self.out / "flush_response.txt").write_text(flush_text)
        if "Cache flushed" not in flush_text:
            raise RuntimeError(f"Flush did not confirm success: {flush_text}")
        self.report["flush"] = {"success": True, "response": flush_text}
        self.snapshot("after_flush")
        self.request("second_42k_after_flush", prompts[1], self.args.second_tokens)
        if self.args.check_l1:
            self.request(
                "repeat_42k_without_flush", prompts[1], self.args.second_tokens
            )
        second = self.report["second_42k_after_flush"]
        expected = self.report["expected_storage_hit_tokens"]
        if not self.gcache_enabled:
            verdict, code = "NOT_ENABLED", 2
        elif (
            self.report["first_32k"]["isolated_prefill_metrics"]
            and not self.report["first_32k"]["counter_reset_detected"]
            and second["cached_tokens"] == expected
            and second["isolated_prefill_metrics"]
            and second["prefill_cached_tokens_by_source"]["storage"] == expected
            and second["prefill_cached_tokens_by_source"]["device"] == 0
            and second["prefill_cached_tokens_by_source"]["host"] == 0
            and second["gcache_prefetch_bytes"] > 0
            and not second["counter_reset_detected"]
            and self.report["writeback"]["status"] == "metrics_settled"
        ):
            verdict, code = "PASS", 0
        else:
            verdict, code = "FAIL_OR_INCONCLUSIVE", 3
        self.report["verdict"] = verdict
        self.report["exit_code"] = code
        save_json(self.out / "report.json", self.report)
        log(
            f"GCache verdict: {verdict}. Expected enabled-backend hit: {expected}/{self.args.second_tokens} ({expected / self.args.second_tokens:.4%})"
        )
        log(f"Artifacts: {self.out}")
        return code


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chat-url", required=True, help="Full /v1/chat/completions URL"
    )
    parser.add_argument(
        "--prefill-url", required=True, help="P server origin, e.g. http://host:9001"
    )
    parser.add_argument("--model", help="Defaults to P's served_model_name")
    parser.add_argument("--first-tokens", type=int, default=32 * 1024)
    parser.add_argument("--second-tokens", type=int, default=42 * 1024)
    parser.add_argument(
        "--shared-tokens",
        type=int,
        default=30 * 1024,
        help="Set equal to first-tokens to append a chat turn after the complete first prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="Requested DP rank; gateway must forward routed_dp_rank",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900,
        help="HTTP socket timeout in seconds",
    )
    parser.add_argument("--control-timeout", type=float, default=60)
    parser.add_argument("--writeback-timeout", type=float, default=120)
    parser.add_argument("--settle-seconds", type=float, default=10)
    parser.add_argument(
        "--check-l1",
        action="store_true",
        help="Also repeat the second request without flushing",
    )
    parser.add_argument(
        "--output-dir",
        default=f"/tmp/gcache-prefix-{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
    )
    args = parser.parse_args()
    if not (
        0 < args.shared_tokens <= args.first_tokens
        and args.shared_tokens < args.second_tokens
    ):
        parser.error(
            "shared-tokens must be positive, no greater than first-tokens, and smaller than second-tokens"
        )
    verification = Verification(args)
    try:
        return verification.run()
    except Exception as exc:
        verification.report.update(verdict="ERROR", error=str(exc), exit_code=1)
        save_json(verification.out / "report.json", verification.report)
        log(f"ERROR: {exc}")
        log(f"Partial artifacts: {verification.out}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
