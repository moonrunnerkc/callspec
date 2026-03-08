# Scoring Guide

How LLMAssert evaluates assertions, the reasoning behind each scoring method, and how to tune thresholds for your use case.

## Scoring Methods

LLMAssert uses three scoring systems. Each is chosen for specific properties that matter in a test library: determinism, cost, and reliability.

### Embedding Similarity

**Used by:** `semantic_intent_matches`, `does_not_discuss`, `is_factually_consistent_with`, `is_consistent_across_samples`, `matches_baseline`, `semantic_drift_is_below`

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (default, configurable via `LLMAssertConfig.embedding_model`)

**Properties:**

- 22MB download (one-time, cached locally)
- 5-20ms per encoding on CPU
- No GPU required, no API key required, no network connection after first download
- Deterministic: same input always produces same embedding vector
- MTEB STS benchmark score of 0.585 normalized, competitive with models 10x its size

**Why this model:** The MTEB benchmark (Muennighoff et al., 2022) consistently shows all-MiniLM-L6-v2 as the best tradeoff between size and semantic quality for English-language similarity. Larger models like all-mpnet-base-v2 (420MB) produce marginally better scores but are too heavy for a default that developers install without thinking.

**Similarity metric:** Cosine similarity. Normalizes for vector magnitude, making it stable across inputs of different lengths. Dot product similarity is length-sensitive and produces unreliable comparisons between short and long responses.

**Changing the model:**

```python
from llm_assert import LLMAssertConfig

config = LLMAssertConfig(embedding_model="sentence-transformers/all-mpnet-base-v2")
```

### Structural Scoring

**Used by:** `is_valid_json`, `matches_schema`, `contains_keys`, `length_between`, `matches_pattern`, `does_not_contain`, `starts_with`, `ends_with`, `format_matches_baseline`

Structural assertions are binary: they pass or fail deterministically based on the response content. No scoring model is involved.

- JSON validation uses Python's `json.loads`
- Schema validation uses `jsonschema` (IETF-standardized, RFC 8927)
- Pattern matching uses Python's `re` module

These assertions are free, instant, and perfectly reproducible.

### Flesch-Kincaid Grade Level

**Used by:** `uses_language_at_grade_level`

The Flesch-Kincaid formula has been continuously validated since 1948 and remains the standard for reading level estimation in English text. It is purely arithmetic (no model inference), deterministic, and transparent.

Formula:

$$FK = 0.39 \times \frac{\text{words}}{\text{sentences}} + 11.8 \times \frac{\text{syllables}}{\text{words}} - 15.59$$

Every developer can verify the calculation independently. No alternative readability score (SMOG, Coleman-Liau) improves meaningfully on Flesch-Kincaid for LLM output assessment.

### LLM-as-Judge (Optional)

**Used by:** Custom assertions via `LLMJudgeScorer`

Available for developers who need nuanced judgment on response quality dimensions that embeddings cannot capture well: tone compliance, instruction-following on complex multi-part instructions, and domain-specific quality criteria.

**Cost:** One additional LLM call per assertion per test run. Use only for assertions that genuinely cannot be expressed with embedding similarity.

**Why this is not the default:** The 2024 TACL paper by Liu et al. (Stanford) on the "lost-in-the-middle" effect showed that LLM attention degrades for relevant content positioned in the middle of context. An LLM judge reading a full response is subject to the same degradation it is supposed to evaluate. Embedding similarity is computed over the full output regardless of position, making it structurally more reliable.

## Threshold Tuning

### Semantic Similarity Thresholds

| Threshold | Meaning | Use case |
|-----------|---------|----------|
| 0.60 | Loosely related | Topic avoidance (default for `does_not_discuss`) |
| 0.75 | Clearly related | Intent matching (default for `semantic_intent_matches`) |
| 0.80 | Strongly aligned | Factual consistency (default for `is_factually_consistent_with`) |
| 0.85 | Very similar | Regression baseline (default for `matches_baseline`) |
| 0.90+ | Near-identical | Use when paraphrase-level similarity is required |

The defaults are calibrated against the SBERT STS-B benchmarks. If your assertions are failing unexpectedly:

1. **Check the embedding model version.** Different versions of sentence-transformers produce different scores for the same input.
2. **Check content length.** Very short responses (under 20 characters) produce less stable embeddings.
3. **Lower the threshold** if the intent is inherently ambiguous or the model is expected to produce varied phrasings.
4. **Raise the threshold** for production-critical assertions where false passes are more dangerous than false failures.

### Behavioral Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `behavioral_pass_rate` | `0.95` | Minimum proportion of samples that must pass |
| `behavioral_sample_count` | `20` | Number of provider calls per behavioral assertion |
| `consistency_threshold` | `0.85` | Minimum average pairwise similarity |
| `confidence_level` | `0.95` | Wilson confidence interval level |

At 20 samples, the 95% confidence interval for a 0.95 pass rate is approximately +/-0.095. This is wide. Teams with tighter requirements should increase to 50 or 100 samples.

### Regression Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `regression_semantic_threshold` | `0.85` | Cosine similarity floor for baseline comparison |
| `regression_drift_ceiling` | `0.15` | Maximum semantic drift (1 - similarity) |

Setting `regression_drift_ceiling` to 0.10 catches smaller changes. Setting it to 0.20 allows more variation. The right value depends on how much your model's output is expected to change between snapshot updates.

## Confidence Intervals

Behavioral assertions produce scores, not binaries. LLMAssert uses Wilson score confidence intervals for proportional data (pass rates, refusal rates).

Wilson intervals are more accurate than normal approximation intervals at both extremes of the probability range (near 0.0 and near 1.0), which is exactly where behavioral assertions operate. This is the method used by `scipy.stats.proportion_confint` with `method='wilson'`.

The confidence interval appears in assertion result details:

```python
result.assertions[0].details["ci_lower"]  # e.g., 0.751
result.assertions[0].details["ci_upper"]  # e.g., 0.998
```

## Pre-Downloading the Embedding Model

The sentence-transformers model downloads on first use (22MB). For CI environments that block outbound downloads:

```bash
# Pre-download in a setup step
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Or cache the model directory:

```yaml
# GitHub Actions
- uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: sentence-transformers-minilm
```

LLMAssert fails loudly if the model download fails, with a clear error message and instructions. It does not fail silently.
