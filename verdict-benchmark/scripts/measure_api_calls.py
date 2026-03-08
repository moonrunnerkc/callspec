#!/usr/bin/env python3
"""Measure actual API call counts by intercepting HTTP requests.

Patches urllib3 to count outbound HTTPS requests during suite execution.
This produces measured call counts, not architectural estimates.

Usage:
    python scripts/measure_api_calls.py --tool verdict
    python scripts/measure_api_calls.py --tool deepeval
    python scripts/measure_api_calls.py --all
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = BENCHMARK_ROOT / "baselines"
PROMPTS_DIR = BENCHMARK_ROOT / "prompts"
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class APICallCounter:
    """Thread-safe counter that wraps urllib3's urlopen to count HTTPS calls."""

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
        self._calls = []

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def calls(self) -> list:
        with self._lock:
            return list(self._calls)

    def increment(self, method: str, url: str) -> None:
        with self._lock:
            self._count += 1
            self._calls.append({
                "seq": self._count,
                "method": method,
                "url": url,
                "timestamp_ms": int(time.monotonic() * 1000),
            })

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._calls.clear()


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


def load_baseline_content(model_dir: str, prompt_name: str) -> str | None:
    filepath = BASELINES_DIR / model_dir / f"{prompt_name}.json"
    if not filepath.exists():
        return None
    baseline = json.loads(filepath.read_text())
    return baseline["response_content"]


def count_with_patched_http(run_fn) -> dict:
    """Run a function while counting all outbound HTTPS requests.

    Patches both urllib3 (used by OpenAI SDK) and httpx (used by Anthropic SDK)
    to capture every API call regardless of which HTTP transport the provider uses.
    """
    import urllib3

    counter = APICallCounter()

    # Patch urllib3 (OpenAI SDK transport)
    original_urlopen = urllib3.HTTPSConnectionPool.urlopen

    def counting_urlopen(pool_self, method, url, *args, **kwargs):
        host = getattr(pool_self, 'host', 'unknown')
        full_url = f"https://{host}{url}"
        counter.increment(method, full_url)
        return original_urlopen(pool_self, method, url, *args, **kwargs)

    # Patch httpx sync + async clients (Anthropic SDK uses sync, DeepEval's
    # internal LLM-as-judge calls use async httpx via OpenAI's async client)
    httpx_patches = []
    try:
        import httpx
        original_httpx_send = httpx.Client.send

        def counting_httpx_send(client_self, request, *args, **kwargs):
            url_str = str(request.url)
            counter.increment(request.method, url_str)
            return original_httpx_send(client_self, request, *args, **kwargs)

        httpx_patches.append(patch.object(httpx.Client, 'send', counting_httpx_send))

        original_async_send = httpx.AsyncClient.send

        async def counting_async_send(client_self, request, *args, **kwargs):
            url_str = str(request.url)
            counter.increment(request.method, url_str)
            return await original_async_send(client_self, request, *args, **kwargs)

        httpx_patches.append(patch.object(httpx.AsyncClient, 'send', counting_async_send))
    except ImportError:
        pass

    start = time.monotonic()
    urllib3_patch = patch.object(urllib3.HTTPSConnectionPool, 'urlopen', counting_urlopen)
    urllib3_patch.start()
    for p in httpx_patches:
        p.start()

    try:
        run_fn()
    finally:
        urllib3_patch.stop()
        for p in httpx_patches:
            p.stop()

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "api_call_count": counter.count,
        "elapsed_ms": elapsed_ms,
        "calls": counter.calls,
    }


def run_verdict_suite():
    """Run Verdict's 3-case YAML suite against the Anthropic provider."""
    from verdict.cli.main import cli as verdict_cli
    suite_path = str(BENCHMARK_ROOT / "suites" / "verdict" / "regression_suite.yml")
    try:
        verdict_cli(["run", suite_path, "-p", "anthropic"])
    except SystemExit:
        pass


def run_deepeval_tests():
    """Run DeepEval's 3-test regression suite with live OpenAI calls."""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from openai import OpenAI

    client = OpenAI()
    prompts = ["structured_output", "semantic_intent", "format_compliance"]

    for prompt_name in prompts:
        prompt_text = load_prompt(prompt_name)
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )
        actual_output = response.choices[0].message.content

        test_case = LLMTestCase(
            input=prompt_text,
            actual_output=actual_output,
        )
        metric = AnswerRelevancyMetric(threshold=0.7)
        metric.measure(test_case)


def measure_tool(tool: str) -> dict:
    print(f"=== Measuring API calls: {tool} ===")

    if tool == "verdict":
        measurement = count_with_patched_http(run_verdict_suite)
    elif tool == "deepeval":
        measurement = count_with_patched_http(run_deepeval_tests)
    else:
        return {"error": f"Tool {tool} not instrumented for live measurement."}

    # Separate LLM provider calls from model-download/telemetry traffic
    LLM_API_HOSTS = {"api.openai.com", "api.anthropic.com"}
    IGNORED_HOSTS = {"huggingface.co", "cdn-lfs.huggingface.co"}

    host_counts = {}
    provider_call_count = 0
    for call in measurement["calls"]:
        url = call["url"]
        host = url.split("/")[2] if len(url.split("/")) > 2 else "unknown"
        host_counts[host] = host_counts.get(host, 0) + 1
        if host in LLM_API_HOSTS:
            provider_call_count += 1

    non_provider_count = measurement["api_call_count"] - provider_call_count
    ignored_count = sum(
        count for host, count in host_counts.items() if host in IGNORED_HOSTS
    )

    result = {
        "tool": tool,
        "provider_api_calls": provider_call_count,
        "total_https_requests": measurement["api_call_count"],
        "model_download_requests": ignored_count,
        "elapsed_ms": measurement["elapsed_ms"],
        "calls_by_host": host_counts,
        "call_detail": measurement["calls"],
        "measurement_method": "urllib3_and_httpx_intercept",
        "note": (
            "Measured by intercepting actual HTTPS requests during suite execution. "
            "provider_api_calls counts only LLM inference calls. "
            "model_download_requests are one-time embedding model fetches (cached after first run)."
        ),
    }

    print(f"  Provider API calls: {provider_call_count}")
    print(f"  Total HTTPS requests: {measurement['api_call_count']}")
    for host, count in host_counts.items():
        print(f"    {host}: {count}")
    print(f"  Elapsed: {measurement['elapsed_ms']}ms")

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure actual API call counts.")
    parser.add_argument("--tool", choices=["verdict", "deepeval"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.tool and not args.all:
        parser.print_help()
        sys.exit(1)

    tools = ["verdict", "deepeval"] if args.all else [args.tool]
    all_results = {}

    for tool in tools:
        all_results[tool] = measure_tool(tool)
        print()

    results_path = RESULTS_DIR / "api_call_counts.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
