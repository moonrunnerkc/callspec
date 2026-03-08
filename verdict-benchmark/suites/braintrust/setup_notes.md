# Braintrust Setup Notes

## Account Requirement

**Mandatory.** Braintrust requires account creation at braintrust.dev
before any functionality is available.

Their documentation opens with "Sign up at braintrust.dev."

The setup path:
1. Navigate to braintrust.dev
2. Create account
3. Run `braintrust login` or set `BRAINTRUST_API_KEY`
4. Install the braintrust package

This is a blocking prerequisite. No Braintrust functionality works without it.

## Installation

```bash
pip install braintrust openai
```

## API Keys Required

- `BRAINTRUST_API_KEY`: From Braintrust dashboard (requires account)
- `OPENAI_API_KEY`: For the model under test

## What Braintrust Cannot Do Without Custom Code

- **Offline operation**: Eval() pushes results to the Braintrust platform.
  Cannot run fully offline without an account.
- **Baseline snapshots**: No built-in snapshot mechanism. Must manage
  baseline data externally and write custom scorers.
- **Embedding-based drift**: Built-in scorers use LLM-as-judge. Custom
  scorer required for cosine similarity.
- **pytest integration**: Braintrust has its own Eval() function that
  does not integrate natively with pytest's assertion model.
- **CI exit codes**: Eval() does not exit non-zero on low scores.
  Must wrap in pytest with manual assertions.
