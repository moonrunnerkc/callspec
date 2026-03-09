# Changelog

All notable changes to callspec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-19

### Added

#### Core Assertions and Mock Provider
- Core assertion framework with `BaseAssertion` abstract class
- `MockProvider` for testing without API calls
- `AssertionRunner` and fluent `AssertionBuilder` API
- `AssertionResult` and `IndividualAssertionResult` structured result types
- `CallspecConfig` dataclass for global settings, thresholds, retry policy
- `NormalizedResponse` common structure all providers return
- `ReportFormatter` with JSON, plaintext, and JUnit XML output
- Structural assertions: `is_valid_json`, `matches_schema`, `contains_keys`, `length_between`, `matches_pattern`, `does_not_contain`, `starts_with`, `ends_with`
- Top-level `Callspec` class with `assert_that()` and `assert_trajectory()` entry points
- `callspec.errors` module with all exception types
- Unit test suite for all structural assertion types

#### Snapshot System and Regression Assertions
- `SnapshotManager` for creating, loading, updating, and deleting baselines
- `SnapshotSerializer` with JSON serialization and schema versioning
- `SnapshotDiff` for human-readable diffs between snapshots
- Trajectory regression assertions: `matches_baseline`, `sequence_matches_baseline`
- Trajectory diff across three dimensions: sequence, argument keys, SHA256 hash
- Snapshot files stored as versioned JSON in project repository
- Unit tests for full snapshot lifecycle and regression assertions

#### Trajectory Assertions
- `ToolCallTrajectory` and `TrajectoryBuilder` for tool-call contract testing
- Trajectory assertions: `calls_tool`, `calls_tools_in_order`, `does_not_call`, `call_count`, `first_tool_is`, `last_tool_is`, `no_duplicate_calls`
- Argument assertions: `argument_equals`, `argument_contains_key`, `argument_not_empty`, `argument_matches_schema`, `argument_satisfies`
- Composite assertions: `NegationWrapper`, `AndAssertion`, `OrAssertion`

#### Provider Adapters
- `OpenAIProvider` with tool-call extraction (`callspec[openai]`)
- `AnthropicProvider` with tool-use block normalization (`callspec[anthropic]`)
- `GoogleProvider` for Gemini function-calling models (`callspec[google]`)
- `MistralProvider` (`callspec[mistral]`)
- `OllamaProvider` for local models (`callspec[ollama]`)
- `LiteLLMProvider` as catch-all router for any provider (`callspec[litellm]`)
- Lazy imports in `callspec.providers` to avoid loading unused SDK dependencies
- Actual model identifier logging from provider response (not requested alias)
- Integration tests for each provider (skipped when API keys absent)

#### Pytest Plugin and CLI
- Pytest plugin registered via `pytest11` entry point
- Fixtures: `callspec_runner`, `callspec_provider`, `callspec_config`, `trajectory_runner`
- Custom `@pytest.mark.tool_contract` mark for contract tests
- CLI flags: `--callspec-report`, `--callspec-report-path`, `--callspec-strict`, `--callspec-skip-contracts`
- Pytest assertion helpers producing structured failure output
- Report hook adding callspec metadata to test reports
- Click-based CLI: `callspec run`, `callspec check`, `callspec snapshot`, `callspec report`, `callspec providers`
- YAML suite parser with JSON Schema validation of suite files
- Integration tests for pytest plugin and CLI commands

#### GitHub Actions Integration
- GitHub Actions annotation formatter (`callspec.integrations.github_actions`)
- PR annotations using `::error`, `::warning`, `::notice` workflow commands
- Borderline pass detection (score within 5% of threshold) emits warnings
- Step summary output via `$GITHUB_STEP_SUMMARY`
- Composite GitHub Action (`action/action.yml`) for one-step CI integration
- CI workflows: test matrix (Python 3.9-3.13), automated lint and type checks

#### Documentation
- Getting started guide
- Trajectory assertions reference
- Contract assertions reference
- Snapshots and drift detection guide
- pytest and CI integration guide
