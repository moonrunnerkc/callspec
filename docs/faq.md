# FAQ

Common questions about cost, flakiness, scoring, and tradeoffs.

## General

### What is Verdict?

A Python assertion library for verifying LLM output. It drops into your existing pytest suite and gives you pass/fail on whether your AI system behaves correctly. It is not a tracing platform, dashboard, or observability tool.

### How is Verdict different from LangSmith / Braintrust / Arize?

Those tools monitor what happened in production. Verdict defines what *should* happen and fails your build if it does not. They are observability platforms. Verdict is a test library. They complement each other: you trace in LangSmith, you define behavioral contracts in Verdict.

### Does Verdict work with any LLM?

Yes. The provider layer is a thin adapter. Verdict ships adapters for OpenAI, Anthropic, Google, Mistral, Ollama, and LiteLLM (which routes to any provider). Writing a custom adapter requires implementing one method.

### Does Verdict require an account?

No. The open-source library has no account requirement, no telemetry, no license checks, and no network calls that benefit the hosted tier.

## Cost

### How much does Verdict cost to run?

**Structural assertions:** Zero. They parse the response string locally.

**Semantic assertions:** Near-zero. They use a local embedding model (22MB, CPU, no API key). Cost is ~10ms of compute per assertion.

**Behavioral assertions:** N provider calls per assertion (default N=20). Cost is N times your provider's per-call cost. A behavioral assertion using GPT-4o at $2.50/1M input tokens and 20 samples costs roughly $0.01-$0.05 per test run, depending on prompt and response length.

**Regression assertions:** Near-zero (same as semantic, uses local embeddings for comparison).

### How do I control cost?

1. Use `--verdict-skip-behavioral` on every-commit CI runs. Reserve behavioral tests for nightly or pre-release.
2. Use `MockProvider` for tests that validate assertion configuration.
3. Start with `n_samples=20` for behavioral assertions and increase only when you need tighter confidence intervals.
4. Use `FixedSetSampler` instead of `SemanticVariantSampler` when hand-written inputs are sufficient.

## Flakiness

### My semantic test fails sometimes. Is this expected?

Semantic assertions use embedding similarity, which is deterministic (same input always produces the same embedding). If your test fails intermittently, the cause is the provider returning different content across runs, not the embedding scorer.

To diagnose:

1. Run with `MockProvider` to confirm the assertion logic is correct for a fixed response.
2. Check if your provider supports deterministic output. OpenAI supports `temperature=0` + `seed`. Anthropic is near-deterministic at `temperature=0` but not guaranteed.
3. If the provider's output varies legitimately, use a behavioral assertion (`passes_rate`) instead of a single-call assertion.

### My behavioral test is flaky at 20 samples.

At 20 samples, the 95% confidence interval for a 0.95 pass rate is approximately +/-0.095. This is statistical reality, not a Verdict bug. Options:

1. Increase `n_samples` to 50 or 100 for tighter confidence intervals.
2. Lower `min_rate` to 0.90 if 95% is stricter than your application requires.
3. Check the trial details in the result to identify which specific inputs are causing failures.

### How do I prevent flaky tests in CI?

- Set `temperature=0` and `seed=42` (the defaults in VerdictConfig).
- Use structural assertions wherever possible (they are deterministic).
- For semantic assertions, test against `MockProvider` first to confirm threshold calibration.
- For behavioral assertions, use adequate sample sizes and consider the confidence interval width.

## Scoring

### Why does Verdict use embedding similarity instead of LLM-as-judge?

Three reasons:

1. **Determinism.** Embeddings produce the same score for the same input. LLM-as-judge produces different scores across runs.
2. **Cost.** One embedding computation is ~10ms on CPU with no API key. One LLM-as-judge call is an additional API call at provider pricing.
3. **Position bias.** The 2024 TACL paper by Liu et al. (Stanford) showed that LLM attention degrades for content in the middle of context. An LLM judge reading a full response is subject to the same degradation it is evaluating. Embedding similarity is computed over the full output regardless of position.

LLM-as-judge is available via `LLMJudgeScorer` for cases where embeddings genuinely cannot capture the quality dimension you need.

### What embedding model does Verdict use?

`sentence-transformers/all-MiniLM-L6-v2` (22MB, CPU-native, no API key). It is the best tradeoff between size and semantic quality on the MTEB benchmark. Configurable via `VerdictConfig.embedding_model`.

### Can I use a different embedding model?

Yes:

```python
from verdict import VerdictConfig
config = VerdictConfig(embedding_model="sentence-transformers/all-mpnet-base-v2")
```

Note that different models produce different similarity scores. Thresholds calibrated for all-MiniLM-L6-v2 may need adjustment for other models.

### What does semantic similarity of 0.75 mean?

In the all-MiniLM-L6-v2 embedding space, 0.75 cosine similarity corresponds approximately to "clearly semantically related" as rated by human annotators on the STS-B benchmark. The response addresses the same topic and intent, but may use different words and structure.

| Score | Interpretation |
|-------|---------------|
| 0.90+ | Near-identical meaning |
| 0.80-0.90 | Strongly related, same core content |
| 0.70-0.80 | Clearly related, same topic and intent |
| 0.60-0.70 | Loosely related |
| Below 0.60 | Different topics or unrelated |

## Technical

### What Python versions does Verdict support?

Python 3.9 through 3.13.

### Does the embedding model require a GPU?

No. all-MiniLM-L6-v2 runs on CPU in 5-20ms per encoding. No GPU, no CUDA, no special hardware.

### The embedding model download is blocked in my CI.

Pre-download the model in a setup step:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Or cache `~/.cache/huggingface` as a CI artifact.

### Can I run Verdict offline?

Structural assertions work offline. Semantic assertions work offline after the embedding model is cached. Behavioral and regression assertions (that need the provider) require network access to the LLM provider. Use `MockProvider` for fully offline testing.

### How do I test my assertion configuration without API calls?

Use `MockProvider`:

```python
from verdict import Verdict
from verdict.providers.mock import MockProvider

provider = MockProvider(lambda prompt, messages: '{"title": "Test"}')
v = Verdict(provider)
result = v.assert_that("test").is_valid_json().run()
```
