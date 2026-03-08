# Promptfoo Setup Notes

## Account Requirement

**None for basic usage.** This is Promptfoo's genuine advantage.
`npx promptfoo@latest eval` works locally without account creation.

Promptfoo Cloud (optional) provides result sharing, team dashboards,
and historical tracking, but core evaluation is fully local.

## Installation

```bash
# Via npx (no install needed):
npx promptfoo@latest eval --config promptfoo_config.yaml

# Or install globally:
npm install -g promptfoo
```

## Prerequisite

- **Node.js 18+**: Promptfoo is a Node.js tool. Python teams must have
  Node.js installed in their CI environment to use it.
- `OPENAI_API_KEY`: Required for the model under test AND for any
  `llm-rubric` assertions (LLM-as-judge).

## What Promptfoo Does Well

- **No account required**: Genuinely local-first. This matches LLMAssert.
- **Browser result viewer**: `npx promptfoo view` opens an interactive
  browser UI for reviewing results. Useful for manual inspection.
- **Mature ecosystem**: Extensive documentation, active community,
  many built-in assertion types.
- **Multi-provider support**: Works with OpenAI, Anthropic, and others
  through a provider configuration layer.

## What Promptfoo Cannot Do Natively

- **Regression snapshots**: No built-in mechanism to record a baseline
  and compare future outputs against it with semantic similarity scoring.
- **Confidence intervals**: No statistical flakiness protection.
  Thresholds are raw comparisons without Wilson score intervals.
- **pytest integration**: Promptfoo is a standalone CLI tool. It does not
  integrate as a pytest plugin. Python teams cannot add Promptfoo
  assertions to existing pytest test files.
- **Embedding-based similarity**: Semantic checks use LLM-as-judge by
  default. No built-in cosine similarity via sentence-transformers.

## Language Mismatch for Python Teams

Promptfoo requires Node.js. For Python ML/AI teams, this means:
- Additional runtime dependency in CI
- Configuration in YAML (not Python test code)
- No access to Python testing ecosystem (pytest fixtures, marks, etc.)
- Custom assertions must be written in JavaScript/TypeScript
