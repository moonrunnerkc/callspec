# Getting Started

Install Verdict, write your first test, and see your first failure in under 5 minutes.

## Install

```bash
pip install verdict
```

For semantic assertions (embedding-based similarity), add the semantic extra:

```bash
pip install "verdict[semantic]"
```

For a specific provider:

```bash
pip install "verdict[openai]"
pip install "verdict[anthropic]"
pip install "verdict[ollama]"
```

Or install everything:

```bash
pip install "verdict[all]"
```

## Verify your setup

```bash
verdict check
```

This confirms Verdict is installed and any configured providers are reachable. If you have an `OPENAI_API_KEY` environment variable set, it will verify OpenAI connectivity.

## Your first test

Create a file called `test_my_llm.py`:

```python
from verdict import Verdict
from verdict.providers.mock import MockProvider

# MockProvider returns a deterministic response without API calls
provider = MockProvider(
    lambda prompt, messages: '{"title": "Quarterly Review", "summary": "Revenue increased 12%"}'
)
v = Verdict(provider)


def test_json_output():
    result = (
        v.assert_that("Summarize the Q3 earnings report as JSON")
        .is_valid_json()
        .contains_keys(["title", "summary"])
        .length_between(20, 500)
        .run()
    )
    assert result.passed
```

Run it:

```bash
pytest test_my_llm.py -v
```

Output:

```
test_my_llm.py::test_json_output PASSED
```

## Your first failure

Modify the test to expect a key the response does not contain:

```python
def test_missing_key():
    result = (
        v.assert_that("Summarize the Q3 earnings report as JSON")
        .is_valid_json()
        .contains_keys(["title", "summary", "recommendations"])
        .run()
    )
    assert result.passed, result.assertions[-1].message
```

Run it:

```bash
pytest test_my_llm.py::test_missing_key -v
```

The failure message tells you exactly what happened:

```
AssertionError: ContainsKeys failed: missing keys {'recommendations'}
from response with keys {'title', 'summary'}. Add the missing keys to
the prompt instructions or remove them from the assertion.
```

Every Verdict failure includes: which assertion failed, the actual value, the expected value, which provider and model were used, and what to try next.

## Using a real provider

Replace `MockProvider` with a real provider:

```python
from verdict import Verdict
from verdict.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")
v = Verdict(provider)


def test_real_output():
    result = (
        v.assert_that("Return a JSON object with keys: title, summary")
        .is_valid_json()
        .contains_keys(["title", "summary"])
        .run()
    )
    assert result.passed
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
pytest test_my_llm.py -v
```

## Adding semantic assertions

Semantic assertions check meaning, not just structure. They require the `semantic` extra:

```bash
pip install "verdict[semantic]"
```

```python
def test_intent():
    result = (
        v.assert_that("Explain quantum computing to a 10 year old")
        .semantic_intent_matches("a simple explanation of quantum computing")
        .uses_language_at_grade_level(5, tolerance=3)
        .run()
    )
    assert result.passed
```

The first semantic assertion call downloads the embedding model (22MB, one-time). Subsequent calls use the cached model.

## Using pytest fixtures

Verdict provides built-in pytest fixtures. Configure your provider once in `conftest.py`:

```python
# conftest.py
import pytest
from verdict.providers.mock import MockProvider

@pytest.fixture(scope="session")
def verdict_provider():
    return MockProvider(
        lambda prompt, messages: '{"title": "Test", "summary": "A summary"}'
    )
```

Then use `verdict_runner` in any test:

```python
def test_output(verdict_runner):
    result = (
        verdict_runner
        .assert_that("Summarize the document")
        .is_valid_json()
        .run()
    )
    assert result.passed
```

## Next steps

- [Assertion Types](assertion_types.md) -- full reference for every assertion
- [Provider Guide](provider_guide.md) -- configure OpenAI, Anthropic, Ollama, and others
- [pytest Guide](pytest_guide.md) -- fixtures, CLI flags, report output
- [YAML Suites](yaml_suite_format.md) -- define assertions as configuration
- [CI Guide](ci_guide.md) -- run Verdict in GitHub Actions
