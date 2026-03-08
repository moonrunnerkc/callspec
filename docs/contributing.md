# Contributing

Guidelines for contributing to Verdict.

## Getting Started

```bash
git clone https://github.com/moonrunnerkc/verdict.git
cd verdict
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=verdict --cov-report=term-missing

# Skip behavioral tests (faster)
pytest --verdict-skip-behavioral

# Only unit tests
pytest tests/unit/

# Integration tests (require API keys)
pytest tests/integration/
```

## Code Style

Verdict uses ruff for linting and formatting:

```bash
ruff check verdict/ tests/
ruff format verdict/ tests/
```

Type checking with mypy:

```bash
mypy verdict/
```

## Engineering Standards

These are non-negotiable. Read them before submitting code.

**Comments explain why, never what.** The code says what. A comment above a sort call that reads "sort the list" is noise. A comment explaining the business reason for the sort order is signal.

**Names are precise.** `data`, `result`, `temp`, `val`, `item` are banned. A list of assertion failures is `failures`. A provider's raw response before normalization is `raw_response`.

**Error messages are for a peer engineer debugging at 2am.** Not "assertion failed" but: "SemanticAssertion failed: score 0.68 below threshold 0.75 using all-MiniLM-L6-v2, input 340 chars, provider gpt-4o-2024-11-20."

**No commented-out code committed.** Git history has it if you need it later.

**No magic numbers.** Every numeric constant with meaning is named and documented.

**Imports are ordered.** Standard library, third-party, internal. No unused imports. No star imports.

**Tests validate real behavior.** A test that passes on a broken implementation is worse than no test.

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Run `ruff check` and `pytest`
5. Submit a pull request

### Commit Messages

Write clear, imperative commit messages:

```
Add refusal pattern for Anthropic Claude 3.5 Sonnet

The new Claude 3.5 model uses a different refusal phrasing that
the existing library does not match. Added pattern covering the
"I cannot assist" and "I'm not able to help" variants.
```

### Pull Request Guidelines

- One concern per PR. Do not batch multiple unrelated changes.
- Include tests for new functionality.
- Update documentation if the public API changes.
- Reference the issue number if applicable.

## Adding a New Assertion Type

1. Create the assertion class in the appropriate module (`structural.py`, `semantic.py`, etc.)
2. Implement `evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult`
3. Add a builder method in `verdict/core/builder.py`
4. Add a YAML builder in `verdict/core/yaml_suite.py`
5. Write unit tests in `tests/unit/`
6. Add documentation in `docs/assertion_types.md`

Use the [assertion type request template](https://github.com/moonrunnerkc/verdict/issues/new?template=assertion_type_request.md) for proposing new types.

## Adding a New Provider

1. Create the adapter in `verdict/providers/`
2. Implement `BaseProvider.call()` returning a `ProviderResponse`
3. Add an optional dependency in `pyproject.toml`
4. Write integration tests (skip when API key is absent)
5. Add documentation in `docs/provider_guide.md`

## Project Structure

```
verdict/
  verdict/
    core/          # Runner, suite, config, types, report, builder
    assertions/    # Structural, semantic, behavioral, regression, composite
    providers/     # Provider adapters (OpenAI, Anthropic, etc.)
    sampling/      # Input samplers for behavioral assertions
    scoring/       # Embedding scorer, confidence estimator
    snapshots/     # Baseline snapshot management
    pytest_plugin/ # pytest integration
    cli/           # CLI commands
    integrations/  # GitHub Actions, external tool bridges
  tests/
    unit/          # Fast, deterministic, no API calls
    integration/   # Provider tests, CLI, pytest plugin
    fixtures/      # Test data: prompts, responses, suites
  docs/            # Documentation
  examples/        # Usage examples
```

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
