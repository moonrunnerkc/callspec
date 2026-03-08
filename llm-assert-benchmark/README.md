# LLMAssert Benchmark: Reproducible Evidence That LLM Behavioral Testing Works

This repository contains a reproducible benchmark comparing LLMAssert against existing LLM evaluation tools. Every claim is backed by runnable scripts that produce verifiable numbers.

## Thesis

The most credible thing a test framework can do is reproduce a failure that existing tools miss. This benchmark detects **real behavioral drift between dated model versions** and measures which tools can catch it, at what cost, and with what setup overhead.

## Quick Start

```bash
# From the verdict project root:
cd llm-assert-benchmark

# Record baselines from dated model versions (requires API keys)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
make record-baselines

# Run the full comparison
make compare

# Or run individual measurements
make measure-drift
make measure-cost
make measure-flakiness
```

## What This Measures

| # | Measurement | Why It Matters |
|---|-------------|----------------|
| 1 | Time from install to first passing test | Developer experience on day one |
| 2 | Lines of setup code | Barrier to adoption |
| 3 | Semantic drift detection at 0.15 cosine | Can the tool catch real model regression? |
| 4 | CI exit code on failure | Does the tool actually fail your build? |
| 5 | API cost per assertion run | Monthly CI budget impact |
| 6 | Flakiness over repeated identical runs | Can you trust the test results? |
| 7 | Measured API call count (not estimated) | Actual network overhead per suite run |
| 8 | Telemetry calls (uninvited network traffic) | Privacy and security implications |

## Results

Run `make compare` to populate these results from your own environment.

| Metric | LLMAssert | DeepEval | Promptfoo | LangSmith | Braintrust |
|--------|---------|----------|-----------|-----------|------------|
| Account required | No | Partial | No | Yes | Yes |
| Language | Python | Python | Node.js | Python | Python |
| Time to first test (cold) | _run benchmark_ | _run benchmark_ | _run benchmark_ | N/A* | N/A* |
| Time to first test (warm) | _run benchmark_ | _run benchmark_ | _run benchmark_ | N/A* | N/A* |
| Catches 0.15 drift | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ |
| Flakiness (100 runs) | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ |
| Exit code on failure (CLI) | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ |
| API calls per run | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ |
| Monthly CI cost (30 runs/day) | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ | _run benchmark_ |

*LangSmith and Braintrust require account creation, which cannot be automated.

## The Core Demonstration

We detect **actual drift** between real dated model versions:

- **Same-family version drift (OpenAI):** `gpt-4o-2024-05-13` vs `gpt-4o-2024-11-20`
- **Model migration drift (Anthropic):** `claude-3-haiku-20240307` vs `claude-sonnet-4-20250514`

These are production model versions that exhibit real behavioral differences. We do not simulate drift by editing prompts or injecting synthetic variance. Every regression detected in this benchmark is between endpoints that any developer can call right now with a standard API key.

The OpenAI pair demonstrates **silent version drift** within the same model family. The models share an API alias (`gpt-4o`), and the behavioral differences (especially in code generation and format compliance) are invisible without explicit version-pinning and drift testing. This is the scenario that catches production teams off guard.

The Anthropic pair demonstrates **model migration testing**: when a team upgrades from a smaller, faster model (Haiku) to a more capable one (Sonnet), semantic behavior changes substantially. LLMAssert's drift detection quantifies exactly how much the output changes, per prompt category, so the team knows which downstream behaviors to re-validate before deploying the migration.

**Citation:** Lingjiao Chen et al., "How Is ChatGPT's Behavior Changing over Time?" (Stanford/Berkeley, October 2023) documented that GPT-4's prime number identification dropped from 97.6% to 2.4% between March and June 2023 model updates.

## Fair Comparison

This benchmark explicitly acknowledges competitor strengths:

**Promptfoo** is a legitimate tool with a mature ecosystem:
- No account required (matches LLMAssert on this axis)
- Browser-based result viewer for manual inspection
- Active community with extensive documentation
- More built-in LLM-as-judge assertion types

**LLMAssert's differentiators** over Promptfoo and other tools:
- Native Python (no Node.js runtime in Python CI)
- pytest plugin (assertions in your existing test files)
- Wilson score confidence intervals (statistical flakiness reduction)
- Regression snapshots with semantic drift scoring (built-in, not manual)
- Local embedding similarity (zero API cost for semantic assertions)

Both sides are in this README because the benchmark's credibility depends on it.

## Directory Structure

```
prompts/            Prompt fixtures for all test cases
baselines/          Recorded responses from dated model versions
suites/
  llm_assert/          LLMAssert YAML and pytest regression suites
  deepeval/         DeepEval equivalent tests + setup notes
  promptfoo/        Promptfoo equivalent config + setup notes
  langsmith/        LangSmith equivalent tests + setup notes
  braintrust/       Braintrust equivalent tests + setup notes
scripts/            Measurement scripts (setup time, drift, flakiness, cost, exit codes)
results/            Generated measurement data (JSON) and comparison reports
```

## Prerequisites

- Python 3.9+
- API keys for the providers you want to test (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- Node.js 18+ (only for Promptfoo measurements)

## License

Apache 2.0
