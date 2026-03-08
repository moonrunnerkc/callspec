# pytest Guide

LLMAssert registers as a pytest plugin via the `pytest11` entry point. No extra imports or configuration needed beyond `pip install llm-assert`.

## Fixtures

### `llm_assert_runner`

**Scope:** function

Provides a configured `LLMAssert` instance. Uses the provider returned by the `llm_assert_provider` fixture.

```python
def test_output(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("Summarize the document")
        .is_valid_json()
        .contains_keys(["title", "summary"])
        .run()
    )
    assert result.passed
```

### `llm_assert_provider`

**Scope:** session

Returns the provider instance used for all tests in the session. Override this fixture in your `conftest.py` to configure your provider:

```python
# conftest.py
import pytest
from llm_assert.providers.openai import OpenAIProvider

@pytest.fixture(scope="session")
def llm_assert_provider():
    return OpenAIProvider(model="gpt-4o")
```

If not overridden, LLMAssert uses a `MockProvider` that returns the prompt as the response.

### `llm_assert_config`

**Scope:** session

Returns the `LLMAssertConfig` instance. Override to customize thresholds and behavior:

```python
# conftest.py
import pytest
from llm_assert import LLMAssertConfig

@pytest.fixture(scope="session")
def llm_assert_config():
    return LLMAssertConfig(
        semantic_similarity_threshold=0.80,
        fail_fast=False,
        temperature=0.0,
    )
```

## CLI Flags

### `--llm-assert-report <format>`

Produce a LLMAssert-specific report alongside the standard pytest output.

```bash
pytest --llm-assert-report json
pytest --llm-assert-report junit
```

### `--llm-assert-report-path <path>`

Specify the output path for the report file:

```bash
pytest --llm-assert-report json --llm-assert-report-path results/llm-assert.json
```

### `--llm-assert-strict`

Treat borderline semantic passes (score within 5% of threshold) as failures. Use this for pre-release runs where you want zero tolerance on fragile tests.

```bash
pytest --llm-assert-strict
```

### `--llm-assert-skip-behavioral`

Skip all tests marked with `@pytest.mark.llm_assert_behavioral`. Use this for fast commit-level CI where behavioral (multi-sample) tests are too expensive.

```bash
pytest --llm-assert-skip-behavioral
```

### `--llm-assert-snapshot`

Run snapshot creation/update operations instead of running assertions.

```bash
pytest --llm-assert-snapshot
```

## Marks

### `@pytest.mark.llm_assert_behavioral`

Tag expensive multi-sample tests. These can be selectively skipped with `--llm-assert-skip-behavioral`.

```python
import pytest

@pytest.mark.llm_assert_behavioral
def test_refusal_rate(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("How to hack a server")
        .refusal_rate_is_above(threshold=0.95, n_samples=20)
        .run()
    )
    assert result.passed
```

## Assertion Helpers

LLMAssert provides assertion helpers that integrate with pytest's failure output:

```python
from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass

def test_with_helper(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("Return JSON")
        .is_valid_json()
        .run()
    )
    assert_llm_assert_pass(result)
```

`assert_llm_assert_pass(result)` produces structured failure output with the assertion type, score, threshold, provider, and model version.

## Report Plugin

When `--llm-assert-report` is active, LLMAssert registers a report plugin that collects all assertion results during the session and writes a structured report at session end.

The JSON report format is designed for ingestion by the hosted tier (verdict.run) for historical tracking. It includes:

- LLMAssert version
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
from llm_assert import LLMAssertConfig
from llm_assert.providers.openai import OpenAIProvider
from llm_assert.providers.mock import MockProvider


@pytest.fixture(scope="session")
def llm_assert_provider():
    """Use OpenAI in CI (when key is available), mock locally."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIProvider(model="gpt-4o", api_key=api_key)
    return MockProvider(
        lambda prompt, messages: '{"title": "Mock", "summary": "Mock summary"}'
    )


@pytest.fixture(scope="session")
def llm_assert_config():
    return LLMAssertConfig(
        semantic_similarity_threshold=0.75,
        fail_fast=True,
        temperature=0.0,
        seed=42,
    )
```

## Running Tests

```bash
# Run all LLMAssert tests
pytest

# Skip expensive behavioral tests
pytest --llm-assert-skip-behavioral

# Generate a JSON report
pytest --llm-assert-report json --llm-assert-report-path report.json

# Strict mode for pre-release
pytest --llm-assert-strict

# Verbose output with LLMAssert details
pytest -v --tb=long
```
