# pytest and CI

callspec registers as a pytest plugin automatically on install. No configuration needed.

## Fixtures

### `callspec_runner`

Function-scoped. Provides a configured `Callspec` instance. The provider is resolved from the `CALLSPEC_PROVIDER` environment variable (defaults to `MockProvider`).

```python
def test_json_output(callspec_runner):
    result = (
        callspec_runner
        .assert_that("Return valid JSON")
        .is_valid_json()
        .run()
    )
    assert result.passed
```

### `trajectory_runner`

Function-scoped factory. Calls the provider, extracts tool calls, and returns a `TrajectoryBuilder` ready for assertion chaining.

```python
def test_booking_flow(trajectory_runner):
    builder = trajectory_runner("Book a flight from SFO to JFK")
    result = (
        builder
        .calls_tools_in_order(["search_flights", "book_flight"])
        .argument_not_empty("search_flights", "origin")
        .run()
    )
    assert result.passed
```

### `callspec_config`

Session-scoped. Provides `CallspecConfig`. Override in `conftest.py` for project-specific settings.

### `callspec_provider`

Session-scoped. Provides the provider instance. Override in `conftest.py` to use a custom provider.

## Markers

### `@pytest.mark.tool_contract`

Tags tests as contract tests. Skip them with `--callspec-skip-contracts`:

```python
@pytest.mark.tool_contract
def test_search_arguments(trajectory_runner):
    builder = trajectory_runner("Search for hotels")
    result = (
        builder
        .argument_contains_key("search", "query")
        .argument_not_empty("search", "query")
        .run()
    )
    assert result.passed
```

```bash
pytest --callspec-skip-contracts     # skip all contract tests
```

## CLI Flags

| Flag | Effect |
|---|---|
| `--callspec-skip-contracts` | Skip tests marked with `@pytest.mark.tool_contract` |
| `--callspec-skip-behavioral` | Skip tests marked with `@pytest.mark.callspec_behavioral` |
| `--callspec-snapshot` | Run snapshot creation/update instead of assertions |
| `--callspec-strict` | Treat borderline passes as failures |
| `--callspec-report <format>` | Produce a report (`json`, `html`, `junit`) |

## Environment Variables

| Variable | Purpose |
|---|---|
| `CALLSPEC_PROVIDER` | Which provider the fixtures use (`mock`, `openai`, `anthropic`, `google`, `mistral`, `ollama`, `litellm`) |
| `OPENAI_API_KEY` | API key for OpenAI provider |
| `ANTHROPIC_API_KEY` | API key for Anthropic provider |

## YAML Suites

Define assertion suites as YAML and run them from the CLI:

```yaml
version: "1.0"
name: "booking_contracts"
cases:
  - name: "flight booking"
    prompt: "Book a flight from SFO to JFK"
    trajectory:
      - calls_tools_in_order: ["search_flights", "book_flight"]
      - does_not_call: "cancel_flight"
    contracts:
      search_flights:
        - key: "query"
          not_empty: true
        - key: "origin"
          contains_key: true
      book_flight:
        - key: "flight_id"
          contains_key: true
```

```bash
callspec run suites/booking.yaml --provider openai --format json --output results.json
```

## GitHub Actions

Use the composite action in your workflow:

```yaml
- uses: moonrunnerkc/callspec@v1
  with:
    suite: suites/booking.yaml
    provider: openai
    strict: true
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Contract failures annotate the PR diff with actionable detail: which tool, which argument, what the constraint was, and what the actual value was.

## CI Recipes

### GitHub Actions (full workflow)

```yaml
name: Tool-call contracts
on: [pull_request]

jobs:
  contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "callspec[openai]"
      - run: callspec run suites/ --provider openai --format junit --output results.xml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - uses: dorny/test-reporter@v1
        if: always()
        with:
          name: Contract Results
          path: results.xml
          reporter: java-junit
```

### pytest in CI

```yaml
      - run: pip install "callspec[openai,dev]"
      - run: pytest tests/ -m "not callspec_behavioral"
        env:
          CALLSPEC_PROVIDER: openai
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```
