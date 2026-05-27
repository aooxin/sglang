"""
Manual SWA RadixCache stress script for sglang serving.
It is intended for ad-hoc serving validation rather than CI.
"""

import argparse
import asyncio
import json
import random
import string
import time
from pathlib import Path

import aiohttp


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_test_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_long_text(target_tokens: int) -> str:
    words_per_token = 0.75
    target_words = int(target_tokens * words_per_token)
    base = (
        "This is a detailed technical document about distributed systems, "
        "covering topics such as consensus algorithms, fault tolerance, "
        "replication strategies, and consistency models. "
    )
    repeated = (base * (target_words // len(base.split()) + 1)).split()
    return " ".join(repeated[:target_words])


async def send_chat_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    scenario_name: str,
    request_id: int,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            elapsed = time.monotonic() - t0
            status = resp.status

            raw_text = await resp.text()
            try:
                body = json.loads(raw_text) if raw_text else {}
            except json.JSONDecodeError:
                body = {}

            if not isinstance(body, dict):
                body = {}

            if status != 200:
                err_msg = (
                    body.get("error", {}).get("message", raw_text[:200])
                    if isinstance(body.get("error"), dict)
                    else body.get("error", raw_text[:200])
                )
                log(
                    f"  [{scenario_name}#{request_id}] HTTP {status} after {elapsed:.2f}s: {err_msg}"
                )
                return {
                    "scenario": scenario_name,
                    "request_id": request_id,
                    "elapsed": elapsed,
                    "error": str(err_msg),
                    "status": status,
                }

            usage = body.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cache_info = ""
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = details.get("cached_tokens", 0)
                cache_info = f", cached={cached}"
            log(
                f"  [{scenario_name}#{request_id}] {elapsed:.2f}s "
                f"prompt={prompt_tokens} completion={completion_tokens}{cache_info}"
            )
            return {
                "scenario": scenario_name,
                "request_id": request_id,
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usage": usage,
                "status": status,
            }
    except Exception as e:
        elapsed = time.monotonic() - t0
        log(f"  [{scenario_name}#{request_id}] FAILED after {elapsed:.2f}s: {e}")
        return {
            "scenario": scenario_name,
            "request_id": request_id,
            "elapsed": elapsed,
            "error": str(e),
            "status": -1,
        }


async def run_exact_cache_hit(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    for i in range(scenario["repeat"]):
        r = await send_chat_request(
            session,
            url,
            model,
            scenario["messages"],
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_messages_list(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    for i, msgs in enumerate(scenario["messages_list"]):
        r = await send_chat_request(
            session,
            url,
            model,
            msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_multi_turn(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    for i, msgs in enumerate(scenario["turns"]):
        r = await send_chat_request(
            session,
            url,
            model,
            msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_long_context(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    long_text = generate_long_text(scenario["long_context_token_target"])
    messages = []
    for m in scenario["messages"]:
        content = m["content"].replace("PLACEHOLDER_LONG_CONTEXT", long_text)
        messages.append({"role": m["role"], "content": content})
    r = await send_chat_request(
        session,
        url,
        model,
        messages,
        scenario["max_tokens"],
        scenario["temperature"],
        scenario["name"],
        0,
    )
    return [r]


async def run_partial_prefix_hit(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    for i, suffix in enumerate(scenario["suffixes"]):
        msgs = [{"role": "user", "content": scenario["shared_prefix"] + suffix}]
        r = await send_chat_request(
            session,
            url,
            model,
            msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_batch_concurrent(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    tasks = []
    for i in range(scenario["concurrency"]):
        tasks.append(
            send_chat_request(
                session,
                url,
                model,
                scenario["messages"],
                scenario["max_tokens"],
                scenario["temperature"],
                scenario["name"],
                i,
            )
        )
    return await asyncio.gather(*tasks)


async def run_speculative_stress(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    for i, msgs in enumerate(scenario["messages_list"]):
        r = await send_chat_request(
            session,
            url,
            model,
            msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_eviction_pressure(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    template = scenario["prompt_template"]
    for i, topic in enumerate(scenario["topics"][: scenario["num_prompts"]]):
        prompt = template.format(num=random.randint(5, 10), topic=topic)
        msgs = [{"role": "user", "content": prompt}]
        r = await send_chat_request(
            session,
            url,
            model,
            msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            i,
        )
        results.append(r)
    return results


async def run_interleaved(session, url, model, scenario):
    log(f"=== {scenario['name']}: {scenario['description']}")
    results = []
    req_id = 0
    for round_i in range(scenario["rounds"]):
        r = await send_chat_request(
            session,
            url,
            model,
            scenario["hittable_messages"],
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            req_id,
        )
        results.append(r)
        req_id += 1

        unique_suffix = "".join(random.choices(string.ascii_letters, k=20))
        unique_msgs = [
            {
                "role": "user",
                "content": scenario["unique_prefix"]
                + unique_suffix
                + ". Respond briefly.",
            }
        ]
        r = await send_chat_request(
            session,
            url,
            model,
            unique_msgs,
            scenario["max_tokens"],
            scenario["temperature"],
            scenario["name"],
            req_id,
        )
        results.append(r)
        req_id += 1
    return results


SCENARIO_RUNNERS = {
    "exact_cache_hit": run_exact_cache_hit,
    "prefix_sharing_short": run_messages_list,
    "multi_turn_conversation": run_multi_turn,
    "cache_miss_unique_prompts": run_messages_list,
    "long_context_chunked_prefill": run_long_context,
    "partial_prefix_hit": run_partial_prefix_hit,
    "batch_concurrent_same_prefix": run_batch_concurrent,
    "speculative_decode_stress": run_speculative_stress,
    "eviction_pressure": run_eviction_pressure,
    "interleaved_hit_miss": run_interleaved,
}


def print_summary(all_results: list[dict]):
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    by_scenario = {}
    for r in all_results:
        name = r["scenario"]
        by_scenario.setdefault(name, []).append(r)

    total_ok = 0
    total_fail = 0
    for name, results in by_scenario.items():
        ok = [r for r in results if r.get("status") == 200]
        fail = [r for r in results if r.get("status") != 200]
        total_ok += len(ok)
        total_fail += len(fail)
        if ok:
            avg_latency = sum(r["elapsed"] for r in ok) / len(ok)
            total_prompt = sum(r.get("prompt_tokens", 0) for r in ok)
            total_completion = sum(r.get("completion_tokens", 0) for r in ok)
            total_cached = 0
            for r in ok:
                usage = r.get("usage")
                if not isinstance(usage, dict):
                    continue
                details = usage.get("prompt_tokens_details")
                if not isinstance(details, dict):
                    continue
                total_cached += details.get("cached_tokens", 0)
            log(
                f"  {name}: {len(ok)} ok, {len(fail)} fail | "
                f"avg_latency={avg_latency:.2f}s | "
                f"prompt_tokens={total_prompt} completion_tokens={total_completion} "
                f"cached_tokens={total_cached}"
            )
        else:
            log(f"  {name}: 0 ok, {len(fail)} fail")

    log(f"\nTotal: {total_ok} succeeded, {total_fail} failed")


async def main():
    parser = argparse.ArgumentParser(description="SWA RadixCache load test")
    parser.add_argument(
        "--data",
        type=str,
        default="swa_cache_test_data.json",
        help="Path to test data JSON",
    )
    parser.add_argument(
        "--url", type=str, default=None, help="Server URL (overrides JSON)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (overrides JSON)"
    )
    parser.add_argument(
        "--scenario", type=str, default=None, help="Run only this scenario (by name)"
    )
    parser.add_argument(
        "--loops", type=int, default=1, help="Number of times to repeat all scenarios"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )
    args = parser.parse_args()

    data = load_test_data(args.data)
    url = args.url or data["server_url"]
    model = args.model or data["model_name"]

    log(f"Server: {url}")
    log(f"Model: {model}")
    log(f"Scenarios: {len(data['scenarios'])}")
    log(f"Loops: {args.loops}")

    all_results = []
    async with aiohttp.ClientSession() as session:
        for loop_i in range(args.loops):
            if args.loops > 1:
                log(f"\n{'=' * 60}\nLOOP {loop_i + 1}/{args.loops}\n{'=' * 60}")

            for scenario in data["scenarios"]:
                name = scenario["name"]
                if args.scenario and name != args.scenario:
                    continue
                runner = SCENARIO_RUNNERS.get(name)
                if runner is None:
                    log(f"WARNING: No runner for scenario '{name}', skipping")
                    continue
                results = await runner(session, url, model, scenario)
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    all_results.append(results)

    print_summary(all_results)

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        log(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
