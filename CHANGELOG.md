# Changelog

All notable changes to callspec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

#### Phase 1: Core Assertions and Mock Provider
- Core assertion framework with `BaseAssertion` abstract class
- `MockProvider` for testing without API calls
- `AssertionRunner` and fluent `AssertionBuilder` API
- `AssertionResult` and `IndividualAssertionResult` structured result types
- `callspecConfig` dataclass for global settings, thresholds, retry policy
- `NormalizedResponse` common structure all providers return
- `ReportFormatter` with JSON, plaintext, and JUnit XML output
- Structural assertions: `is_valid_json`, `matches_schema`, `contains_keys`, `length_between`, `matches_pattern`, `does_not_contain`, `starts_with`, `ends_with`
- Top-level `callspec` class with `assert_that()` entry point
- `callspec.errors` module with all exception types
- Unit test suite for all structural assertion types

#### Phase 2: Embedding Scorer and Semantic Assertions
- `EmbeddingScorer` using sentence-transformers/all-MiniLM-L6-v2 (22MB, CPU-native)
- Cosine similarity computation for semantic scoring
- `ConfidenceEstimator` with Wilson score confidence intervals
- Flesch-Kincaid grade level computation in `scoring.structural`
- Semantic assertions: `SemanticIntentMatches` (threshold 0.75), `DoesNotDiscuss` (threshold 0.6), `IsFactuallyConsistentWith` (threshold 0.80), `UsesLanguageAtGradeLevel`
- Optional `callspec[semantic]` extra for sentence-transformers and scipy dependencies
- Unit tests for all semantic assertions and confidence interval computation

#### Phase 3: Snapshot System and Regression Assertions
- `SnapshotManager` for creating, loading, updating, and deleting baselines
- `SnapshotSerializer` with JSON serialization and schema versioning
- `SnapshotDiff` for human-readable diffs between snapshots
- Regression assertions: `MatchesBaseline` (semantic threshold 0.85), `SemanticDriftIsBelow` (max drift 0.15), `FormatMatchesBaseline`
- Snapshot files stored as versioned JSON in project repository
- Unit tests for full snapshot lifecycle and regression assertions

#### Phase 4: Provider Adapters
- `OpenAIProvider` with deterministic seed support (`callspec[openai]`)
- `AnthropicProvider` with temperature=0 near-deterministic behavior (`callspec[anthropic]`)
- `GoogleProvider` for Gemini models (`callspec[google]`)
- `MistralProvider` (`callspec[mistral]`)
- `OllamaProvider` for local models with seed support (`callspec[ollama]`)
- `LiteLLMProvider` as catch-all router for any provider (`callspec[litellm]`)
- Lazy imports in `callspec.providers` to avoid loading unused SDK dependencies
- Actual model identifier logging from provider response (not requested alias)
- Integration tests for each provider (skipped when API keys absent)

#### Phase 5: Behavioral Assertions and Input Sampling
- Behavioral assertions: `PassesRate` (min_rate 0.95, n_samples 20), `RefusalRateIsAbove`, `IsConsistentAcrossSamples` (threshold 0.85, n_samples 10)
- `BaseSampler` abstract class and `InputItem` type
- `FixedSetSampler` for explicit input lists
- `TemplateSampler` for slot-based input generation
- `SemanticVariantSampler` for LLM-generated phrasing variants with disk cache
- `SeedManager` for deterministic, reproducible sampling
- Refusal pattern library with real patterns from OpenAI, Anthropic, and Google
- Composite assertions: `NegationWrapper`, `AndAssertion`, `OrAssertion`
- Unit tests for behavioral assertions and sampling strategies

#### Phase 6: Pytest Plugin and CLI
- Pytest plugin registered via `pytest11` entry point
- Fixtures: `callspec_runner`, `callspec_provider`, `callspec_config`
- Custom `@pytest.mark.callspec_behavioral` mark for expensive multi-sample tests
- CLI flags: `--callspec-report`, `--callspec-report-path`, `--callspec-strict`, `--callspec-skip-behavioral`, `--callspec-snapshot`
- Pytest assertion helpers producing structured failure output
- Report hook adding callspec metadata to test reports
- Click-based CLI: `callspec run`, `callspec check`, `callspec snapshot`, `callspec report`, `callspec providers`
- YAML suite parser with JSON Schema validation of suite files
- Integration tests for pytest plugin and CLI commands

#### Phase 7: GitHub Actions Integration and Documentation
- GitHub Actions annotation formatter (`callspec.integrations.github_actions`)
- PR annotations using `::error`, `::warning`, `::notice` workflow commands
- Borderline pass detection (score within 5% of threshold) emits warnings
- Step summary output via `$GITHUB_STEP_SUMMARY`
- Structured outputs (passed, passed-cases, failed-cases, total-cases) via `$GITHUB_OUTPUT`
- Composite GitHub Action (`action/action.yml`) for one-step CI integration
- CI workflows: test matrix (Python 3.9-3.13), automated PyPI release, nightly provider health
- Issue templates: bug report, feature request, assertion type request
- Complete documentation site: getting started, assertion types reference, provider guide, pytest guide, YAML suite format, scoring guide, sampling guide, CI guide, regression guide, FAQ, contributing guide, architecture overview
