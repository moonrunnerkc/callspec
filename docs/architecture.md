# Architecture

Design decisions, rationale, and technical structure of Verdict.

## Core Design Decision

Verdict is an assertion library, not an observability platform. Tracing tools (LangSmith, Braintrust, Arize Phoenix) monitor what happened after deployment. Verdict defines what should happen and fails the build if it does not. This positioning is deliberate: competing with platforms is a losing position; positioning as their complement opens the door for integration rather than conflict.

## Assertion Taxonomy

The four-layer taxonomy is grounded in what LLM behavioral failures actually look like in production:

**Structural assertions** verify form. Does the response parse as JSON? Does it match a schema? These are deterministic, cheap, and catch the most common class of LLM failure: malformed output. Every LLM application that expects structured output should have structural assertions.

**Semantic assertions** verify meaning. Does the response address the user's intent? Does it avoid prohibited topics? Semantic assertions are inherently probabilistic: the same output might score slightly differently due to floating point variance. Verdict handles this with thresholds calibrated against STS-B benchmarks, not binary pass/fail.

**Behavioral assertions** verify patterns across multiple outputs. They run the model N times and assess the distribution. A single-call assertion on a behavioral property is not statistically meaningful for stochastic systems. Behavioral assertions make this explicit with configurable sample sizes and Wilson confidence intervals.

**Regression assertions** compare against recorded baselines. They detect relative change, not absolute quality. This is the mechanism for catching silent model drift when providers update models without announcement.

## Provider Architecture

Every provider is a thin adapter implementing `BaseProvider.call()`. The response is normalized into `ProviderResponse` regardless of which provider produced it. The `model` field contains the actual model identifier from the API response, not the alias requested.

Providers ship as optional extras (`verdict[openai]`, `verdict[anthropic]`). The core library has zero provider dependencies. This keeps the base install small and avoids forcing developers to install SDKs they do not use.

The mock provider takes a deterministic function and enables testing assertion logic without API calls. This matters for Verdict's own test suite and for teams validating assertion configurations without spending credentials.

## Scoring Architecture

**Embedding similarity** uses sentence-transformers/all-MiniLM-L6-v2. Chosen for size (22MB), speed (5-20ms on CPU), determinism, and semantic quality (MTEB STS benchmark). Cosine similarity normalizes for vector magnitude, making it stable across different response lengths.

**Why not LLM-as-judge by default:** The 2024 TACL paper by Liu et al. (Stanford) showed LLM attention degrades for content in the middle of context. An LLM judge reading a full response is subject to the same positional bias. Embedding similarity is position-invariant. LLM-as-judge is available as an option, not the default.

**Flesch-Kincaid** for grade level: validated since 1948, purely arithmetic, deterministic, transparent.

**jsonschema** for schema validation: IETF-standardized (RFC 8927), language-neutral, JSON-serializable errors. Pydantic is Python-specific with Python-specific error objects.

## Statistical Reliability

LLMs are stochastic. A test suite that fails 5% of the time randomly is noise. Verdict's reliability design:

1. **Deterministic seeds.** Every provider call sets `temperature=0` and `seed=42` when supported. The docs are explicit about which providers offer true deterministic output.

2. **Wilson score confidence intervals** for proportional data (pass rates). More accurate than normal approximation at the extremes (near 0.0 and 1.0) where behavioral assertions operate.

3. **Retry on provider error.** Network failures and rate limits are retried with exponential backoff. Assertion failures are never retried.

## Execution Model

The `AssertionRunner` calls the provider once per assertion chain, then evaluates all assertions against the same response content. With `fail_fast=True` (default), evaluation stops at the first failure. Behavioral assertions break this model: they make N provider calls internally because they need the distribution, not a single sample.

## Snapshot Design

Baselines are versioned JSON files in the repository. The design mirrors pytest-snapshot and syrupy: developers already understand this workflow. Snapshots record content, model identifier, provider, timestamp, and JSON keys for structural comparison.

## Plugin Architecture

The pytest plugin registers via `pytest11` entry points: fixtures (`verdict_runner`, `verdict_provider`, `verdict_config`), marks (`verdict_behavioral`), CLI flags (`--verdict-report`, `--verdict-strict`, `--verdict-skip-behavioral`, `--verdict-snapshot`), and a report hook for structured output.

The CLI uses click with subcommands: `verdict run`, `verdict check`, `verdict snapshot`, `verdict report`, `verdict providers`. Both the YAML format and the Python API compile to the same `AssertionSuite` internal representation.

## File Organization

```
verdict/
  core/           # Runner, builder, suite, config, types, report, yaml parser
  assertions/     # One module per taxonomy layer + composite + refusal patterns
  providers/      # One module per provider + base + mock + response alias
  sampling/       # Sampler interface, strategies, seed manager
  scoring/        # Embedding scorer, structural scorer, confidence estimator
  snapshots/      # Manager, serializer, differ
  pytest_plugin/  # Plugin entry, fixtures, assertion helpers, reporter
  cli/            # Click entry point, command modules
  integrations/   # GitHub Actions, future external tool bridges
```

Each module has a single responsibility. Cross-boundary state is explicit. No shared mutable globals.
