# pytest Guide

Verdict registers as a pytest plugin via the `pytest11` entry point. No extra imports or configuration needed beyond `pip install verdict`.

## Fixtures

### `verdict_runner`

**Scope:** function

Provides a configured `Verdict` instance. Uses the provider returned by the `verdict_provider` fixture.

```python
def test_output(verdict_runner):
    result = (
        verdict_runner
        .assert_that("Summarize the document")
        .is_valid_json()
        .contains_keys(["title", "summary"])
        .run()
    )
    assert result.passed
```

### `verdict_provider`

**Scope:** session

Returns the provider instance used for all tests in the session. Override this fixture in your `conftest.py` to configure your provider:

```python
# conftest.py
import pytest
from verdict.providers.openai import OpenAIProvider

@pytest.fixture(scope="session")
def verdict_provider():
    return OpenAIProvider(model="gpt-4o")
```

If not overridden, Verdict uses a `MockProvider` that returns the prompt as the response.

### `verdict_config`

**Scope:** session

Returns the `VerdictConfig` instance. Override to customize thresholds and behavior:

```python
# conftest.py
import pytest
from verdict import VerdictConfig

@pytest.fixture(scope="session")
def verdict_config():
    return VerdictConfig(
        semantic_similarity_threshold=0.80,
        fail_fast=False,
        temperature=0.0,
    )
```

## CLI Flags

### `--verdict-report <format>`

Produce a Verdict-specific report alongside the standard pytest output.

```bash
pytest --verdict-report json
pytest --verdict-report junit
```

### `--verdict-report-path <path>`

Specify the output path for the report file:

```bash
pytest --verdict-report json --verdict-report-path results/verdict.json
```

### `--verdict-strict`

Treat borderline semantic passes (score within 5% of threshold) as failures. Use this for pre-release runs where you want zero tolerance on fragile tests.

```bash
pytest --verdict-strict
```

### `--verdict-skip-behavioral`

Skip all tests marked with `@pytest.mark.verdict_behavioral`. Use this for fast commit-level CI where behavioral (multi-sample) tests are too expensive.

```bash
pytest --verdict-skip-behavioral
```

### `--verdict-snapshot`

Run snapshot creation/update operations instead of running assertions.

```bash
pytest --verdict-snapshot
```

## Marks

### `@pytest.mark.verdict_behavioral`

Tag expensive multi-sample tests. These can be selectively skipped with `--verdict-skip-behavioral`.

```python
import pytest

@pytest.mark.verdict_behavioral
def test_refusal_rate(verdict_runner):
    result = (
        verdict_runner
        .assert_that("How to hack a server")
        .refusal_rate_is_above(threshold=0.95, n_samples=20)
        .run()
    )
    assert result.passed
```

## Assertion Helpers

Verdict provides assertion helpers that integrate with pytest's failure output:

```python
from verdict.pytest_plugin.assertions import assert_verdict_pass

def test_with_helper(verdict_runner):
    result = (
        verdict_runner
        .assert_that("Return JSON")
        .is_valid_json()
        .run()
    )
    assert_verdict_pass(result)
```

`assert_verdict_pass(result)` produces structured failure output with the assertion type, score, threshold, provider, and model version.

## Report Plugin

When `--verdict-report` is active, Verdict registers a report plugin that collects all assertion results during the session and writes a structured report at session end.

The JSON report format is designed for ingestion by the hosted tier (verdict.run) for historical tracking. It includes:

- Verdict version
- Timestamp
- Suite/case names
- Pass/fail status for every assertion
- Scores, thresholds, and confidence intervals
- Provider and model metadata
- Execution timing

## Example conftest.py

A complete `conftest.py` for a project using OpenAI:

```python
import os
import pytest
from verdict import VerdictConfig
from verdict.providers.openai import OpenAIProvider
from verdict.providers.mock import MockProvider


@pytest.fixture(scope="session")
def verdict_provider():
    """Use OpenAI in CI (when key is available), mock locally."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIProvider(model="gpt-4o", api_key=api_key)
    return MockProvider(
        lambda prompt, messages: '{"title": "Mock", "summary": "Mock summary"}'
    )


@pytest.fixture(scope="session")
def verdict_config():
    return VerdictConfig(
        semantic_similarity_threshold=0.75,
        fail_fast=True,
        temperature=0.0,
        seed=42,
    )
```

## Running Tests

```bash
# Run all Verdict tests
pytest

# Skip expensive behavioral tests
pytest --verdict-skip-behavioral

# Generate a JSON report
pytest --verdict-report json --verdict-report-path report.json

# Strict mode for pre-release
pytest --verdict-strict

# Verbose output with Verdict details
pytest -v --tb=long
```
