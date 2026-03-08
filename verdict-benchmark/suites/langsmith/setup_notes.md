# LangSmith Setup Notes

## Account Requirement

**Mandatory.** LangSmith requires account creation at smith.langchain.com
before any functionality is available.

The setup path:
1. Navigate to smith.langchain.com
2. Create account (email or GitHub OAuth)
3. Navigate to Settings > API Keys
4. Generate a new API key
5. Set `LANGCHAIN_API_KEY` in your environment

This is a blocking prerequisite. No LangSmith functionality works without it.

## Installation

```bash
pip install langsmith langchain-openai
```

## API Keys Required

- `LANGCHAIN_API_KEY`: From the LangSmith dashboard (requires account)
- `OPENAI_API_KEY`: For the model under test

## Architecture Mismatch

LangSmith is fundamentally an **observability platform**, not a test framework.
Its primary workflow is:
1. Instrument your LLM calls with LangChain tracing
2. View traces in the LangSmith dashboard
3. Optionally run evaluations on traced data

The evaluate() API exists but is secondary to the tracing workflow.
There is no CLI command that runs assertions and exits with a non-zero
code on failure. CI integration requires wrapping evaluate() in pytest
and manually checking results.

## What LangSmith Cannot Do Without Custom Code

- **Baseline snapshots**: No built-in mechanism. Must create LangSmith
  datasets manually and write custom evaluators.
- **Semantic drift scoring**: Must implement embedding comparison in
  a custom evaluator function.
- **CI exit codes**: evaluate() does not fail on low scores. Must wrap
  in pytest and add manual assertions.
- **Offline operation**: All results are pushed to the LangSmith cloud.
  Cannot run fully offline.
