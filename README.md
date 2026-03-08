<div align="center">

<br>

# LLMAssert

**Behavioral assertion testing for LLM applications.**

Created by [Bradley R. Kinnard](https://github.com/moonrunnerkc)

<br>

[![PyPI](https://img.shields.io/pypi/v/llm-assert)](https://pypi.org/project/llm-assert/)
&nbsp;&nbsp;
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
&nbsp;&nbsp;
[![Python](https://img.shields.io/pypi/pyversions/llm-assert)](https://pypi.org/project/llm-assert/)
&nbsp;&nbsp;
![Tests](https://img.shields.io/badge/tests-450%20passing-brightgreen.svg)

<br>

[Quick Start](#quick-start) · [What Is This](#what-is-this) · [Features](#features) · [Installation](#installation) · [Usage](#usage) · [Benchmarks](#benchmarks) · [Docs](#documentation)

<br>

</div>

---

<br>

## Quick Start

```bash
pip install "llm-assert[anthropic]"
```

```python
from llm_assert import LLMAssert
from llm_assert.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-20250514")
v = LLMAssert(provider)

result = (
    v.assert_that("Return a JSON object with keys: title, summary, tags")
    .is_valid_json()
    .contains_keys(["title", "summary", "tags"])
    .length_between(50, 2000)
    .semantic_intent_matches("a structured summary with metadata")
    .does_not_contain("I'm sorry")
    .run()
)

assert result.passed
```

Works with any provider. Swap `AnthropicProvider` for `OpenAIProvider`, `OllamaProvider`, or any other adapter and the assertions stay the same.

<br>

---

<br>

## What Is This

LLMAssert is a composable assertion library for verifying LLM output. It drops into your existing pytest suite and gives you a clean pass/fail on whether your AI system behaves correctly.

It is not a tracing platform, an observability tool, or a dashboard. Those tools monitor what happened. LLMAssert defines what *should* happen and fails your build if it does not.

Structural assertions (JSON validity, schema compliance, key presence, length bounds, regex) are deterministic, zero cost, and require no LLM calls. Semantic assertions (intent matching, topic avoidance, factual consistency) run locally via sentence-transformers with no API key and no external calls. Behavioral assertions run the model N times and assess the distribution. Regression assertions detect semantic drift and format shifts across model versions.

<br>

## Features

- **Structural assertions** verifying JSON, schema, keys, length, regex, and string patterns. Deterministic, no LLM calls.
- **Semantic assertions** using local embeddings (22MB model, CPU, no API key) for intent matching, topic avoidance, factual consistency, and reading level
- **Behavioral assertions** running N samples with Wilson confidence intervals for pass rate, refusal rate, and consistency checks
- **Regression and drift detection** comparing against versioned JSON baselines to catch silent model updates and format shifts
- **Composite logic** chaining assertions with AND, OR, NOT, and `satisfies()` for arbitrary assertion instances
- **Provider-agnostic** with adapters for OpenAI, Anthropic, Google, Mistral, Ollama, LiteLLM, and a MockProvider for zero-cost testing
- **pytest plugin** registering automatically with fixtures, marks, CLI flags, and JSON report hooks
- **YAML assertion suites** defined as configuration and runnable from CLI with non-zero exit on failure
- **GitHub Action** (`moonrunnerkc/llm-assert@v1`) annotating PRs inline with assertion details on failure
- **Deterministic semantic scoring** via local embeddings, producing identical scores across runs (zero flakiness)
- **No telemetry, no analytics, no background network traffic.** LLMAssert makes exactly the LLM calls you ask for.

<br>

---

<br>

## Installation

```bash
pip install llm-assert                            # structural assertions only (3s install)
pip install "llm-assert[anthropic]"               # add Anthropic provider (~5s)
pip install "llm-assert[anthropic,semantic]"       # add local semantic scoring (~85s, includes PyTorch)
```

The base install covers structural assertions with any provider. Adding `[semantic]` pulls in sentence-transformers and PyTorch, which is a heavy install but eliminates all runtime API costs for semantic scoring.

Available extras: `openai`, `anthropic`, `google`, `mistral`, `ollama`, `litellm`, `semantic`.

<br>

---

<br>

## Usage

### Assertion Types

<br>

<details>
<summary><strong>Structural:</strong> verify form (deterministic, zero cost)</summary>

<br>

```python
result = (
    v.assert_that(prompt)
    .is_valid_json()
    .matches_schema(my_schema)
    .contains_keys(["title", "summary"])
    .length_between(50, 2000)
    .starts_with("{")
    .ends_with("}")
    .matches_pattern(r'"title"\s*:')
    .does_not_contain("```")
    .run()
)
```

All 8 structural assertions verified against Claude Sonnet 4 with a single API call.

</details>

<details>
<summary><strong>Semantic:</strong> verify meaning (local embeddings, no API key)</summary>

<br>

Uses embedding similarity via sentence-transformers (22MB model, runs locally on CPU).

```python
result = (
    v.assert_that(prompt)
    .semantic_intent_matches("a helpful product recommendation", threshold=0.75)
    .does_not_discuss("competitor products", threshold=0.6)
    .is_factually_consistent_with(reference_doc, threshold=0.80)
    .uses_language_at_grade_level(8, tolerance=2)
    .run()
)
```

Tested against live Claude output: `semantic_intent_matches` scored 0.77, `does_not_discuss` correctly scored 0.10 (well below 0.6 rejection threshold), `is_factually_consistent_with` scored 0.81, and `uses_language_at_grade_level` correctly measured Flesch-Kincaid grade 5.0 for a simple-language prompt.

</details>

<details>
<summary><strong>Behavioral:</strong> verify patterns across multiple outputs</summary>

<br>

Runs the model N times and assesses the distribution with Wilson confidence intervals.

```python
from llm_assert.assertions.structural import IsValidJson
from llm_assert.sampling.strategies import FixedSetSampler

sampler = FixedSetSampler(["prompt one", "prompt two", "prompt three"])

result = (
    v.assert_that(prompt)
    .passes_rate(IsValidJson(), min_rate=0.95, n_samples=20, sampler=sampler)
    .run()
)
```

Tested with `FixedSetSampler` and `TemplateSampler` against Claude: 5/5 pass rate on structural checks, 4/5 refusal rate on adversarial inputs (meeting 0.80 threshold), and 1.000 consistency score across repeated calls.

</details>

<details>
<summary><strong>Regression:</strong> detect drift and format shifts</summary>

<br>

Compare against recorded baselines. Detect semantic drift, format shifts, and silent model updates. Baselines live in your repo as versioned JSON.

```python
from llm_assert.snapshots.manager import SnapshotManager

snap_mgr = SnapshotManager(snapshot_dir="llm_assert_snapshots/")

result = (
    v.assert_that(prompt)
    .matches_baseline("my_endpoint", snap_mgr, semantic_threshold=0.85)
    .run()
)
```

Also available: `.semantic_drift_is_below()` for drift-only checks, and `.format_matches_baseline()` for structural-only comparison. Drift detection verified: a marine biology response against a Python baseline correctly triggered failure with 0.89 semantic drift.

</details>

<details>
<summary><strong>Composite:</strong> chain assertions with boolean logic</summary>

<br>

```python
from llm_assert.assertions.structural import IsValidJson, StartsWith

# Implicit AND: every chained assertion must pass
result = v.assert_that(prompt).is_valid_json().contains_keys(["name"]).run()

# OR: accept either format
result = v.assert_that(prompt).or_(StartsWith("{"), StartsWith("[")).run()

# NOT: invert any assertion
result = v.assert_that(prompt).not_(IsValidJson()).run()

# satisfies: pass any BaseAssertion instance
result = v.assert_that(prompt).satisfies(IsValidJson()).run()
```

</details>

<br>

### Providers

LLMAssert works with any LLM provider. The provider layer is a thin adapter; assertions are provider-agnostic.

```bash
pip install "llm-assert[openai]"       # OpenAI
pip install "llm-assert[anthropic]"    # Anthropic
pip install "llm-assert[ollama]"       # Ollama (local)
pip install "llm-assert[google]"       # Google Generative AI
pip install "llm-assert[mistral]"      # Mistral
pip install "llm-assert[litellm]"      # Any provider via LiteLLM
```

```python
from llm_assert.providers.mock import MockProvider

# Test assertions without API calls or cost
mock = MockProvider(response_fn=lambda prompt, msgs=None: '{"title": "Test", "summary": "ok"}')
v = LLMAssert(mock)
```

Every provider returns a `NormalizedResponse` with consistent fields: `content`, `model` (exact identifier, not the alias), `provider`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `request_id`, and `raw` (original provider response). Verified against live Anthropic output.

<br>

### pytest Integration

LLMAssert registers as a pytest plugin. No new CLI or workflow to learn.

```python
def test_summarizer(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("Summarize the Q3 earnings report")
        .is_valid_json()
        .contains_keys(["summary", "highlights"])
        .semantic_intent_matches("financial summary with key highlights")
        .run()
    )
    assert result.passed
```

```bash
LLM_ASSERT_PROVIDER=anthropic pytest tests/ -v
```

| Flag | Effect |
|------|--------|
| `--llm-assert-report json --llm-assert-report-path report.json` | Save JSON report |
| `--llm-assert-skip-behavioral` | Skip expensive multi-sample tests |
| `--llm-assert-strict` | Borderline passes become failures |

The `--llm-assert-skip-behavioral` flag skips any test marked with `@pytest.mark.llm_assert_behavioral`, keeping commit-level runs fast while full behavioral suites run on a longer schedule.

<br>

### YAML Suites

Define assertion suites as configuration, committed alongside your model config:

```yaml
version: "1.0"
name: "summarizer_suite"
cases:
  - name: "valid_json_output"
    prompt: "Return a JSON summary of the document"
    assertions:
      - type: is_valid_json
      - type: contains_keys
        params:
          keys: ["title", "summary"]
      - type: length_between
        params:
          min_chars: 50
          max_chars: 2000
  - name: "semantic_check"
    prompt: "Explain why Python is popular for data science"
    assertions:
      - type: semantic_intent_matches
        params:
          reference_intent: "Python is widely used in data science"
          threshold: 0.55
      - type: does_not_discuss
        params:
          topic: "JavaScript web frameworks"
          threshold: 0.6
```

```bash
LLM_ASSERT_PROVIDER=anthropic llm-assert run suite.yml
```

Exits with non-zero on any failure. Supports `--format json` for CI report ingestion. A 7-case, 30-assertion YAML suite ran against Claude Sonnet 4 in 47 seconds with all cases passing.

<br>

### CLI

```bash
llm-assert check                                  # verify provider connectivity
llm-assert run suite.yml --provider anthropic      # execute a YAML assertion suite
llm-assert snapshot create my_key "prompt text"    # record a baseline snapshot
llm-assert snapshot diff my_key --prompt "prompt"  # compare current vs baseline
llm-assert snapshot update my_key "prompt text"    # update an existing baseline
llm-assert snapshot delete my_key                  # remove a baseline
llm-assert providers                              # list installed providers
llm-assert report result.json                     # pretty-print a saved report
```

<br>

### GitHub Actions

```yaml
- uses: moonrunnerkc/llm-assert@v1
  with:
    suite: tests/llm_assert_suite.yml
    llm-assert-extras: anthropic,semantic
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Failures annotate the PR inline with assertion details: which assertion failed, the actual score, the threshold, and the model version. Step summaries appear in the workflow run UI.

<br>

### Failure Output

When an assertion fails, the message tells you what went wrong, not just that something failed:

```
SemanticAssertion failed: score 0.68 below threshold 0.75
  using all-MiniLM-L6-v2, input 340 chars,
  provider claude-sonnet-4-20250514.
  Check embedding model version or lower threshold
  if intent ambiguity is acceptable.
```

```
Response JSON has 2 schema violations: 'email' is a required property
```

```
SemanticDrift failed for 'python_summary': drift 0.8863 exceeds max 0.1000
  (similarity 0.1137). Baseline model: claude-sonnet-4-20250514.
  Review the prompt or lower max_drift if the change is intentional.
```

<br>

---

<br>

## Benchmarks

Every number below is produced by a runnable script in [llm-assert-benchmark/](llm-assert-benchmark/) and backed by a JSON result file. Full methodology and reproduction instructions: [llm-assert-benchmark/README.md](llm-assert-benchmark/README.md).

<br>

<details>
<summary><strong>Comparison with existing tools</strong></summary>

<br>

| Metric | LLMAssert | DeepEval | Promptfoo | LangSmith | Braintrust |
|--------|---------|----------|-----------|-----------|------------|
| Account required | No | Partial | No | Yes | Yes |
| Native drift detection | Yes | No | No | No | No |
| Semantic scoring method | Local embeddings | LLM-as-judge | LLM-as-judge | LLM-as-judge | LLM-as-judge |
| Provider API calls per run | 7 | 28 | 17 | 21 | 21 |
| Monthly CI cost (30 runs/day) | $26.78 | $31.31 | $28.94 | $29.80 | $29.80 |
| Lines of code for drift test | 9 | 25 | 6 (YAML) | 27 | -- |

Source: [llm-assert-benchmark/results/cost.json](llm-assert-benchmark/results/cost.json), [llm-assert-benchmark/results/loc.json](llm-assert-benchmark/results/loc.json)

</details>

<details>
<summary><strong>API calls and telemetry</strong> (measured via request interception)</summary>

<br>

Actual HTTPS requests intercepted during a 3-case suite run via `urllib3` and `httpx` patching:

| Metric | LLMAssert | DeepEval |
|--------|---------|----------|
| Provider API calls | 3 | 12 (3 model + 9 judge) |
| Telemetry calls | 0 | 4 (3 PostHog + 1 ipify) |

LLMAssert makes exactly the LLM calls you ask for. No analytics, no IP lookups, no background network traffic.

Source: [llm-assert-benchmark/results/api_call_counts.json](llm-assert-benchmark/results/api_call_counts.json)

</details>

<details>
<summary><strong>Flakiness</strong> (identical input, repeated runs)</summary>

<br>

LLMAssert's embedding-based scoring is deterministic. LLM-as-judge scoring is not.

| Metric | LLMAssert (100 runs) | DeepEval (20 runs) |
|--------|--------------------|--------------------|
| Score stdev | 0.0 | 0.0160 |
| Score range | 0.0 | 0.0714 (0.929 to 1.0) |
| Scoring method | Local embeddings | LLM-as-judge |

Source: [llm-assert-benchmark/results/flakiness.json](llm-assert-benchmark/results/flakiness.json)

</details>

<details>
<summary><strong>Drift detection</strong> across real model versions</summary>

<br>

Drift measured as `1 - cosine_similarity` between recorded baselines and live model responses using sentence-transformers/all-MiniLM-L6-v2. Threshold: 0.15.

**GPT-4o version drift** (`gpt-4o-2024-05-13` vs `gpt-4o-2024-11-20`):

| Prompt | Cosine Drift | Detected |
|--------|-------------|----------|
| structured_output | 0.012 | No |
| semantic_intent | 0.066 | No |
| format_compliance | 0.269 | **Yes** |
| code_generation | 0.162 | **Yes** |
| numeric_reasoning | 0.028 | No |
| instruction_following | 0.037 | No |
| chain_of_thought | 0.090 | No |

**Anthropic model migration** (`claude-3-haiku-20240307` vs `claude-sonnet-4-20250514`):

| Prompt | Cosine Drift | Detected |
|--------|-------------|----------|
| structured_output | 0.100 | No |
| semantic_intent | 0.260 | **Yes** |
| format_compliance | 0.464 | **Yes** |
| code_generation | 0.124 | No |
| numeric_reasoning | 0.057 | No |
| instruction_following | 0.121 | No |
| chain_of_thought | 0.261 | **Yes** |

The OpenAI pair catches silent version drift within the same model family. The Anthropic pair quantifies behavioral change during a model tier migration. Both are real API endpoints any developer can call today.

Source: [llm-assert-benchmark/results/drift_detection.json](llm-assert-benchmark/results/drift_detection.json)

</details>

<details>
<summary><strong>Setup time</strong></summary>

<br>

LLMAssert's install footprint depends on which extras you need. Measured in fresh venvs with warm pip cache:

| Configuration | Install | First Test | Total |
|---|---|---|---|
| `pip install llm-assert` | 3.1s | 102ms | 3.2s |
| `pip install "llm-assert[anthropic]"` | 5.4s | 105ms | 5.5s |
| `pip install "llm-assert[anthropic,semantic]"` | 85.3s | 113ms | 85.4s |
| `pip install deepeval openai` | 12.4s | 1,491ms | 13.8s |

The base install (structural assertions, any provider) is 3.1 seconds. Adding `[semantic]` pulls in sentence-transformers and PyTorch, which is heavy (85s) but eliminates all runtime API costs for semantic scoring. DeepEval's lighter install shifts that cost to runtime: every semantic assertion makes an additional LLM API call, which is why its first test takes 14x longer (1,491ms vs 105ms).

Source: [llm-assert-benchmark/results/setup_time.json](llm-assert-benchmark/results/setup_time.json)

</details>

<details>
<summary><strong>CI exit codes</strong></summary>

<br>

| Interface | Exit Code on Failure |
|-----------|---------------------|
| `llm-assert run` (CLI) | 1 |
| `pytest` (plugin) | 2 |

Source: [llm-assert-benchmark/results/exit_codes.json](llm-assert-benchmark/results/exit_codes.json)

</details>

<details>
<summary><strong>Test results</strong> (458 unit tests + live integration)</summary>

<br>

Every assertion type has been verified against live Claude Sonnet 4 (`claude-sonnet-4-20250514`) via the Anthropic API:

| Category | Assertions Tested | Result |
|---|---|---|
| Structural | 8 (is_valid_json, matches_schema, contains_keys, length_between, matches_pattern, does_not_contain, starts_with, ends_with) | All passed |
| Semantic | 4 (semantic_intent_matches, does_not_discuss, is_factually_consistent_with, uses_language_at_grade_level) | All passed |
| Behavioral | 4 (passes_rate with FixedSetSampler, passes_rate with TemplateSampler, refusal_rate_is_above, is_consistent_across_samples) | All passed |
| Regression | 4 (matches_baseline, semantic_drift_is_below, format_matches_baseline, intentional drift detection) | All passed |
| Composite | 5 (chained AND, or_, not_, satisfies, mixed structural+semantic) | All passed |
| YAML Suite via CLI | 7 cases, 30 assertions | All passed |
| pytest Plugin | 3 live tests + 1 correctly skipped behavioral | All passed |
| Error handling | 6 (MockProvider, NormalizedResponse fields, failure messages, schema violations, connectivity, result metadata) | All passed |
| Unit test suite | 458 tests | 450 passed, 8 skipped |

</details>

<br>

---

<br>

## Status

Actively maintained. 458 unit tests plus live integration tests against Claude Sonnet 4. Development is ongoing.

<br>

---

<br>

## Documentation

- [Getting Started](docs/getting_started.md) -- install, first test, first failure in under 5 minutes
- [Assertion Types](docs/assertion_types.md) -- full reference for all assertion types
- [Provider Guide](docs/provider_guide.md) -- configure each provider, required env vars
- [pytest Guide](docs/pytest_guide.md) -- fixtures, marks, CLI flags, report hooks
- [YAML Suite Format](docs/yaml_suite_format.md) -- complete YAML suite specification
- [Scoring Guide](docs/scoring_guide.md) -- how each scorer works, tuning thresholds
- [Sampling Guide](docs/sampling_guide.md) -- input sampling for behavioral assertions
- [Regression Guide](docs/regression_guide.md) -- snapshot workflow, baseline management
- [CI Guide](docs/ci_guide.md) -- GitHub Actions, GitLab CI, CircleCI recipes
- [FAQ](docs/faq.md) -- cost, flakiness, LLM-as-judge tradeoffs
- [Contributing](docs/contributing.md) -- contributor guide
- [Architecture](docs/architecture.md) -- design decisions and rationale

<br>

---

<br>

## Contributing

Contributions are welcome. See the [contributor guide](docs/contributing.md) for setup, standards, and process.

```bash
pip install -e ".[dev,semantic]"
pytest
```

<br>

---

<br>

## License

Copyright 2025-2026 Bradley R. Kinnard. Licensed under [Apache 2.0](LICENSE).