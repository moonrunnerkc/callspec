# Regression Guide

Snapshot management, baseline comparison, and detecting silent model drift.

## The Problem

LLM providers update models without announcement. "gpt-4o" today may not produce the same output as "gpt-4o" last week. Regression assertions detect this drift before it reaches your users.

## Workflow

1. Record a baseline snapshot of your model's output
2. Commit the snapshot to version control
3. On every test run, compare current output against the baseline
4. When drift is detected, review the change and update the baseline if intentional

## Creating Snapshots

### Via CLI

```bash
# Create a snapshot for a specific prompt
llm-assert snapshot create --key "product_summary" --prompt "Summarize the product" --provider openai

# Create snapshots for all cases in a YAML suite
llm-assert snapshot create --suite tests/llm_assert_suite.yml
```

### Via pytest

```bash
pytest --llm-assert-snapshot
```

This runs all snapshot creation/update operations instead of running assertions.

### Programmatically

```python
from llm_assert.snapshots.manager import SnapshotManager

snapshot_mgr = SnapshotManager("llm_assert_snapshots/")

# Save a response as a baseline
snapshot_mgr.save_entry(
    key="product_summary",
    content='{"title": "Widget", "summary": "A great widget"}',
    model="gpt-4o-2024-11-20",
    provider="openai",
)
```

## Snapshot Storage

Snapshots are stored as versioned JSON files in a directory (default: `llm_assert_snapshots/`). The directory and files should be committed to version control.

```
llm_assert_snapshots/
  snapshots.json    # all snapshot entries
```

Each snapshot entry records:
- `key`: unique identifier
- `content`: the response text
- `model`: exact model identifier
- `provider`: provider name
- `timestamp`: when the snapshot was created
- `json_keys`: top-level JSON keys (if the response is JSON)
- `content_hash`: SHA-256 hash for change detection

## Regression Assertions

### matches_baseline

Checks both structure and meaning:

```python
from llm_assert.snapshots.manager import SnapshotManager

snapshot_mgr = SnapshotManager("llm_assert_snapshots/")

def test_baseline(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("Summarize the product")
        .matches_baseline("product_summary", snapshot_mgr)
        .run()
    )
    assert result.passed
```

Fails if:
- JSON keys changed (added or removed)
- Semantic similarity dropped below threshold (default: 0.85)

Both checks are independent. The result details show exactly which dimension changed:

```python
result.assertions[0].details["structural_match"]     # True/False
result.assertions[0].details["semantic_similarity"]   # 0.0 to 1.0
result.assertions[0].details["keys_added"]            # ["new_key"]
result.assertions[0].details["keys_removed"]          # ["old_key"]
```

### semantic_drift_is_below

Tolerates structural changes, catches meaning drift:

```python
result = (
    v.assert_that(prompt)
    .semantic_drift_is_below("product_summary", snapshot_mgr, max_drift=0.15)
    .run()
)
```

Drift is `1 - cosine_similarity`. A drift of 0.0 means identical output. A drift of 0.15 means the output has diverged ~15% from the baseline.

### format_matches_baseline

Catches format changes, ignores content:

```python
result = (
    v.assert_that(prompt)
    .format_matches_baseline("product_summary", snapshot_mgr)
    .run()
)
```

Useful when the model produces the same JSON schema but different values over time (dynamic data in a fixed structure).

## Updating Baselines

When model output changes intentionally (after a prompt update or model upgrade):

```bash
# Update a specific snapshot
llm-assert snapshot update --key "product_summary" --prompt "Summarize the product" --provider openai

# Update all snapshots in a suite
llm-assert snapshot update --suite tests/llm_assert_suite.yml

# Review changes before committing
llm-assert snapshot diff --key "product_summary"
```

The diff output shows:
- Structural changes (added/removed JSON keys)
- Semantic similarity between old and new baseline
- Content length changes
- Model version changes

## Version Control Practices

1. **Commit snapshots alongside code changes.** When you change a prompt, update and commit the snapshot in the same PR. Reviewers can see the output change alongside the code change.

2. **Review snapshot diffs in PRs.** The snapshot diff tells you exactly how model output changed. A semantic similarity drop from 0.95 to 0.72 is a red flag.

3. **Track model versions.** Each snapshot records the exact model identifier. When "gpt-4o" silently points to a newer version, the snapshot diff reveals it.

4. **Separate baselines per environment.** If staging and production use different models, maintain separate snapshot directories:

```python
staging_snapshots = SnapshotManager("llm_assert_snapshots/staging/")
production_snapshots = SnapshotManager("llm_assert_snapshots/production/")
```

## Drift Detection Patterns

### Detecting Silent Model Updates

```python
def test_model_drift(llm_assert_runner):
    """Fails when the provider silently updates the model."""
    result = (
        llm_assert_runner
        .assert_that("Summarize the quarterly report")
        .semantic_drift_is_below("quarterly_summary", snapshot_mgr, max_drift=0.10)
        .run()
    )
    assert result.passed, (
        f"Model output has drifted: {result.assertions[0].message}. "
        f"If intentional, run: llm-assert snapshot update --key quarterly_summary"
    )
```

### Monitoring Format Stability

```python
def test_schema_stability(llm_assert_runner):
    """Fails when the model changes its JSON output structure."""
    result = (
        llm_assert_runner
        .assert_that("Return a product listing as JSON")
        .format_matches_baseline("product_json", snapshot_mgr)
        .run()
    )
    assert result.passed
```

### Combining Regression with Structural Checks

```python
def test_full_regression(llm_assert_runner):
    result = (
        llm_assert_runner
        .assert_that("Summarize the document")
        .is_valid_json()
        .contains_keys(["title", "summary"])
        .matches_baseline("document_summary", snapshot_mgr)
        .run()
    )
    assert result.passed
```
