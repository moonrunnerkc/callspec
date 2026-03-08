# Verdict

Behavioral assertion testing for LLM applications.

```python
from verdict import Verdict
from verdict.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-20250514")
v = Verdict(provider)

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

[![PyPI](https://img.shields.io/pypi/v/verdict)](https://pypi.org/project/verdict/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/verdict)](https://pypi.org/project/verdict/)

## Install

```bash
pip install verdict
```

With provider and semantic extras:

```bash
pip install "verdict[anthropic,semantic]"
```

Available extras: `openai`, `anthropic`, `google`, `mistral`, `ollama`, `litellm`, `semantic`.

## What Verdict Does

Verdict is a composable assertion library for verifying LLM output. It drops into your existing pytest suite and gives you a clean pass/fail on whether your AI system behaves correctly.

It is not a tracing platform, an observability tool, or a dashboard. Those tools monitor what happened. Verdict defines what *should* happen and fails your build if it does not.

## Assertion Types

### Structural

Verify the form of the output. JSON validity, schema compliance, key presence, length bounds, regex patterns. Deterministic, zero cost, no LLM calls.

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

### Semantic

Verify meaning, not form. Does the response address the user's intent? Does it avoid prohibited topics? Is it consistent with reference material? Uses embedding similarity via sentence-transformers (22MB model, runs locally on CPU, no API key required).

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

### Behavioral

Verify patterns across multiple outputs. Does the model always refuse dangerous inputs? Is it consistent across runs? Runs the model N times and assesses the distribution with Wilson confidence intervals.

```python
from verdict.assertions.structural import IsValidJson
from verdict.sampling.strategies import FixedSetSampler

sampler = FixedSetSampler(["prompt one", "prompt two", "prompt three"])

result = (
    v.assert_that(prompt)
    .passes_rate(IsValidJson(), min_rate=0.95, n_samples=20, sampler=sampler)
    .run()
)
```

Tested with `FixedSetSampler` and `TemplateSampler` against Claude: 5/5 pass rate on structural checks, 4/5 refusal rate on adversarial inputs (meeting 0.80 threshold), and 1.000 consistency score across repeated calls.

### Regression

Compare against recorded baselines. Detect semantic drift, format shifts, and silent model updates. Baselines live in your repo as versioned JSON.

```python
from verdict.snapshots.manager import SnapshotManager

snap_mgr = SnapshotManager(snapshot_dir="verdict_snapshots/")

result = (
    v.assert_that(prompt)
    .matches_baseline("my_endpoint", snap_mgr, semantic_threshold=0.85)
    .run()
)
```

Also available: `.semantic_drift_is_below()` for drift-only checks, and `.format_matches_baseline()` for structural-only comparison. Drift detection verified: a marine biology response against a Python baseline correctly triggered failure with 0.89 semantic drift.

### Composite

Chain assertions with boolean logic.

```python
from verdict.assertions.structural import IsValidJson, StartsWith

# Implicit AND: every chained assertion must pass
result = v.assert_that(prompt).is_valid_json().contains_keys(["name"]).run()

# OR: accept either format
result = v.assert_that(prompt).or_(StartsWith("{"), StartsWith("[")).run()

# NOT: invert any assertion
result = v.assert_that(prompt).not_(IsValidJson()).run()

# satisfies: pass any BaseAssertion instance
result = v.assert_that(prompt).satisfies(IsValidJson()).run()
```

## Providers

Verdict works with any LLM provider. The provider layer is a thin adapter; assertions are provider-agnostic.

```bash
pip install "verdict[openai]"       # OpenAI
pip install "verdict[anthropic]"    # Anthropic
pip install "verdict[ollama]"       # Ollama (local)
pip install "verdict[google]"       # Google Generative AI
pip install "verdict[mistral]"      # Mistral
pip install "verdict[litellm]"      # Any provider via LiteLLM
```

```python
from verdict.providers.mock import MockProvider

# Test assertions without API calls or cost
mock = MockProvider(response_fn=lambda prompt, msgs=None: '{"title": "Test", "summary": "ok"}')
v = Verdict(mock)
```

Every provider returns a `NormalizedResponse` with consistent fields: `content`, `model` (exact identifier, not the alias), `provider`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `request_id`, and `raw` (original provider response). Verified against live Anthropic output.

## pytest Integration

Verdict registers as a pytest plugin. No new CLI or workflow to learn.

```python
def test_summarizer(verdict_runner):
    result = (
        verdict_runner
        .assert_that("Summarize the Q3 earnings report")
        .is_valid_json()
        .contains_keys(["summary", "highlights"])
        .semantic_intent_matches("financial summary with key highlights")
        .run()
    )
    assert result.passed
```

Set the provider via environment variable:

```bash
VERDICT_PROVIDER=anthropic pytest tests/ -v
```

Flags:

```bash
pytest --verdict-report json --verdict-report-path report.json
pytest --verdict-skip-behavioral    # skip expensive multi-sample tests
pytest --verdict-strict             # borderline passes become failures
```

The `--verdict-skip-behavioral` flag skips any test marked with `@pytest.mark.verdict_behavioral`, keeping commit-level runs fast while full behavioral suites run on a longer schedule.

## YAML Suites

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
VERDICT_PROVIDER=anthropic verdict run suite.yml
```

Exits with non-zero on any failure. Supports `--format json` for CI report ingestion. A 7-case, 30-assertion YAML suite ran against Claude Sonnet 4 in 47 seconds with all cases passing.

## CLI

```bash
verdict check                                  # verify provider connectivity
verdict run suite.yml --provider anthropic      # execute a YAML assertion suite
verdict snapshot create my_key "prompt text"    # record a baseline snapshot
verdict snapshot diff my_key --prompt "prompt"  # compare current vs baseline
verdict snapshot update my_key "prompt text"    # update an existing baseline
verdict snapshot delete my_key                  # remove a baseline
verdict providers                              # list installed providers
verdict report result.json                     # pretty-print a saved report
```

## GitHub Actions

```yaml
- uses: moonrunnerkc/verdict@v1
  with:
    suite: tests/verdict_suite.yml
    verdict-extras: anthropic,semantic
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Failures annotate the PR inline with assertion details: which assertion failed, the actual score, the threshold, and the model version. Step summaries appear in the workflow run UI.

## Failure Output

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

## Test Results

Every assertion type has been verified against live Claude Sonnet 4 (`claude-sonnet-4-20250514`) via the Anthropic API, plus 458 unit tests passing:

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
| Unit test suite | 458 tests | 458 passed, 6 skipped |

## Benchmark Results

Measured against real LLM providers, not synthetic data. Every number is produced by a runnable script in [verdict-benchmark/](verdict-benchmark/) and backed by a JSON result file.

### Comparison with Existing Tools

| Metric | Verdict | DeepEval | Promptfoo | LangSmith | Braintrust |
|--------|---------|----------|-----------|-----------|------------|
| Account required | No | Partial | No | Yes | Yes |
| Native drift detection | Yes | No | No | No | No |
| Semantic scoring method | Local embeddings | LLM-as-judge | LLM-as-judge | LLM-as-judge | LLM-as-judge |
| Provider API calls per run | 7 | 28 | 17 | 21 | 21 |
| Monthly CI cost (30 runs/day) | $26.78 | $31.31 | $28.94 | $29.80 | $29.80 |
| Lines of code for drift test | 9 | 25 | 6 (YAML) | 27 | -- |

Source: [verdict-benchmark/results/cost.json](verdict-benchmark/results/cost.json), [verdict-benchmark/results/loc.json](verdict-benchmark/results/loc.json)

### Measured API Calls and Telemetry

Actual HTTPS requests intercepted during a 3-case suite run via `urllib3` and `httpx` patching:

| Metric | Verdict | DeepEval |
|--------|---------|----------|
| Provider API calls | 3 | 12 (3 model + 9 judge) |
| Telemetry calls | 0 | 4 (3 PostHog + 1 ipify) |

Verdict makes exactly the LLM calls you ask for. No analytics, no IP lookups, no background network traffic.

Source: [verdict-benchmark/results/api_call_counts.json](verdict-benchmark/results/api_call_counts.json)

### Flakiness

Identical input, same model, repeated runs. Verdict's embedding-based scoring is deterministic. DeepEval's LLM-as-judge scoring is not.

| Metric | Verdict (100 runs) | DeepEval (20 runs) |
|--------|--------------------|--------------------|
| Score stdev | 0.0 | 0.0160 |
| Score range | 0.0 | 0.0714 (0.929 to 1.0) |
| Scoring method | Local embeddings | LLM-as-judge |

Source: [verdict-benchmark/results/flakiness.json](verdict-benchmark/results/flakiness.json)

### Drift Detection: Real Model Versions

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

Source: [verdict-benchmark/results/drift_detection.json](verdict-benchmark/results/drift_detection.json)

### Setup Time

Verdict's install footprint depends on which extras you need. Measured in fresh venvs with warm pip cache:

| Configuration | Install | First Test | Total |
|---|---|---|---|
| `pip install verdict` | 3.1s | 102ms | 3.2s |
| `pip install "verdict[anthropic]"` | 5.4s | 105ms | 5.5s |
| `pip install "verdict[anthropic,semantic]"` | 85.3s | 113ms | 85.4s |
| `pip install deepeval openai` | 12.4s | 1,491ms | 13.8s |

The base install (structural assertions, any provider) is 3.1 seconds. Adding `[semantic]` pulls in sentence-transformers and PyTorch, which is heavy (85s) but eliminates all runtime API costs for semantic scoring. DeepEval's lighter install shifts that cost to runtime: every semantic assertion makes an additional LLM API call, which is why its first test takes 14x longer (1,491ms vs 105ms).

Source: [verdict-benchmark/results/setup_time.json](verdict-benchmark/results/setup_time.json)

### CI Exit Codes

Verdict exits with non-zero on assertion failure, making it safe for CI gates:

| Interface | Exit Code on Failure |
|-----------|---------------------|
| `verdict run` (CLI) | 1 |
| `pytest` (plugin) | 2 |

Source: [verdict-benchmark/results/exit_codes.json](verdict-benchmark/results/exit_codes.json)

Full benchmark methodology and reproduction instructions: [verdict-benchmark/README.md](verdict-benchmark/README.md)

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

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
