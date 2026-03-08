"""Unit tests for regression assertions against synthetic baselines.

Each test creates a baseline via SnapshotManager, then evaluates the
regression assertion against current content. The tests cover all three
regression assertion types across their pass/fail boundaries.

Semantic regression tests require sentence-transformers and trigger
embedding computation. Structural-only tests (FormatMatchesBaseline)
do not require embeddings and run without the semantic extra.
"""

from __future__ import annotations

import json
from pathlib import Path

from verdict.assertions.regression import (
    FormatMatchesBaseline,
    MatchesBaseline,
    SemanticDriftIsBelow,
)
from verdict.core.config import VerdictConfig
from verdict.snapshots.manager import SnapshotManager

CONFIG = VerdictConfig()


def _create_baseline(
    tmp_path: Path,
    snapshot_key: str,
    content: str,
    prompt: str = "test prompt",
    model: str = "mock-v1",
) -> SnapshotManager:
    """Helper: create a SnapshotManager with a single baseline entry."""
    manager = SnapshotManager(snapshot_dir=tmp_path)
    manager.create_entry(
        snapshot_key=snapshot_key,
        content=content,
        prompt=prompt,
        model=model,
    )
    return manager


# ---------------------------------------------------------------------------
# MatchesBaseline
# ---------------------------------------------------------------------------

class TestMatchesBaseline:

    def test_identical_content_passes(self, tmp_path: Path) -> None:
        content = json.dumps({"title": "Test Report", "score": 0.95})
        manager = _create_baseline(tmp_path, "identical", content)

        assertion = MatchesBaseline("identical", manager)
        assertion_result = assertion.evaluate(content, CONFIG)

        assert assertion_result.passed is True
        assert assertion_result.assertion_type == "regression"
        assert assertion_result.assertion_name == "matches_baseline"
        assert assertion_result.details["structural_match"] is True
        assert assertion_result.details["semantic_passed"] is True

    def test_semantic_drift_fails(self, tmp_path: Path) -> None:
        """Same structure, completely different meaning: semantic check fails."""
        baseline = json.dumps({"summary": "The weather today is sunny and warm."})
        current = json.dumps({"summary": "Quantum computing uses qubits for parallel processing."})
        manager = _create_baseline(tmp_path, "semantic_drift", baseline)

        assertion = MatchesBaseline("semantic_drift", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is False
        assert assertion_result.details["structural_match"] is True
        assert assertion_result.details["semantic_passed"] is False

    def test_structural_change_fails(self, tmp_path: Path) -> None:
        """Different JSON keys: structural check fails even if content is similar."""
        baseline_content = "Climate change affects coastal areas with rising sea levels."
        current_content = json.dumps(
            {"text": "Climate change affects coastal areas with rising sea levels."}
        )
        manager = _create_baseline(tmp_path, "struct_change", baseline_content)

        assertion = MatchesBaseline("struct_change", manager)
        assertion_result = assertion.evaluate(current_content, CONFIG)

        assert assertion_result.passed is False
        assert assertion_result.details["structural_match"] is False

    def test_custom_semantic_threshold(self, tmp_path: Path) -> None:
        """A very low threshold makes borderline semantic similarity pass."""
        baseline = "The effects of climate change on coastal areas are significant."
        current = "Weather patterns are shifting in coastal regions worldwide."
        manager = _create_baseline(tmp_path, "low_thresh", baseline)

        # Low threshold: should pass
        assertion = MatchesBaseline("low_thresh", manager, semantic_threshold=0.2)
        assertion_result = assertion.evaluate(current, CONFIG)
        assert assertion_result.passed is True

    def test_failure_message_includes_diagnostics(self, tmp_path: Path) -> None:
        baseline = json.dumps({"report": "Annual sales exceeded targets."})
        current = json.dumps({"report": "The mitochondria is the powerhouse of the cell."})
        manager = _create_baseline(tmp_path, "diag", baseline)

        assertion = MatchesBaseline("diag", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert "diag" in assertion_result.message
        lower_msg = assertion_result.message.lower()
        assert "semantic similarity" in lower_msg or "below" in lower_msg

    def test_both_plain_text_structural_match(self, tmp_path: Path) -> None:
        """Two plain-text (non-JSON) responses: structural dimension treated as match."""
        baseline = "Climate change causes rising sea levels and coastal erosion in many regions."
        # Close paraphrase to ensure semantic similarity is high
        current = (
            "Rising sea levels and coastal erosion are caused "
            "by climate change in many areas."
        )
        manager = _create_baseline(tmp_path, "plaintext", baseline)

        assertion = MatchesBaseline("plaintext", manager, semantic_threshold=0.5)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.details["structural_match"] is True


# ---------------------------------------------------------------------------
# SemanticDriftIsBelow
# ---------------------------------------------------------------------------

class TestSemanticDriftIsBelow:

    def test_identical_content_zero_drift(self, tmp_path: Path) -> None:
        content = "The project deadline is next Friday."
        manager = _create_baseline(tmp_path, "zero_drift", content)

        assertion = SemanticDriftIsBelow("zero_drift", manager)
        assertion_result = assertion.evaluate(content, CONFIG)

        assert assertion_result.passed is True
        # Drift should be at or very near 0.0 for identical text
        assert assertion_result.score is not None
        assert assertion_result.score < 0.01

    def test_high_drift_fails(self, tmp_path: Path) -> None:
        baseline = "The cat sleeps on the warm windowsill."
        current = "Advanced cryptographic protocols ensure data integrity in distributed systems."
        manager = _create_baseline(tmp_path, "high_drift", baseline)

        assertion = SemanticDriftIsBelow("high_drift", manager, max_drift=0.15)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is False
        assert assertion_result.score is not None
        assert assertion_result.score > 0.15

    def test_moderate_drift_within_tolerance(self, tmp_path: Path) -> None:
        """Close paraphrase: drift should be small enough to pass default ceiling."""
        baseline = "Regular exercise reduces the risk of heart disease significantly."
        current = "Physical activity lowers cardiovascular disease risk substantially."
        manager = _create_baseline(tmp_path, "moderate", baseline)

        assertion = SemanticDriftIsBelow("moderate", manager, max_drift=0.30)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is True

    def test_custom_max_drift(self, tmp_path: Path) -> None:
        content = "Test content for drift measurement."
        manager = _create_baseline(tmp_path, "custom_drift", content)

        # Identical content with very tight ceiling: should pass since drift ~0
        assertion = SemanticDriftIsBelow("custom_drift", manager, max_drift=0.001)
        assertion_result = assertion.evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_result_metadata(self, tmp_path: Path) -> None:
        content = "Metadata test content."
        manager = _create_baseline(tmp_path, "meta", content, model="gpt-4o")

        assertion = SemanticDriftIsBelow("meta", manager)
        assertion_result = assertion.evaluate(content, CONFIG)

        assert assertion_result.assertion_type == "regression"
        assert assertion_result.assertion_name == "semantic_drift_is_below"
        assert assertion_result.details["snapshot_key"] == "meta"
        assert assertion_result.details["baseline_model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# FormatMatchesBaseline
# ---------------------------------------------------------------------------

class TestFormatMatchesBaseline:

    def test_same_keys_passes(self, tmp_path: Path) -> None:
        baseline = json.dumps({"title": "Old", "score": 0.8})
        current = json.dumps({"title": "New", "score": 0.95})
        manager = _create_baseline(tmp_path, "same_keys", baseline)

        assertion = FormatMatchesBaseline("same_keys", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is True
        assert assertion_result.details["structural_match"] is True

    def test_added_key_fails(self, tmp_path: Path) -> None:
        baseline = json.dumps({"title": "Report"})
        current = json.dumps({"title": "Report", "extra_field": True})
        manager = _create_baseline(tmp_path, "added_key", baseline)

        assertion = FormatMatchesBaseline("added_key", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is False
        assert "extra_field" in assertion_result.details["keys_added"]

    def test_removed_key_fails(self, tmp_path: Path) -> None:
        baseline = json.dumps({"title": "Report", "score": 0.9, "summary": "Good"})
        current = json.dumps({"title": "Report"})
        manager = _create_baseline(tmp_path, "removed_key", baseline)

        assertion = FormatMatchesBaseline("removed_key", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.passed is False
        assert "score" in assertion_result.details["keys_removed"]
        assert "summary" in assertion_result.details["keys_removed"]

    def test_both_plain_text_passes(self, tmp_path: Path) -> None:
        """Non-JSON on both sides: structural comparison not applicable, treated as match."""
        manager = _create_baseline(tmp_path, "plain", "just text")

        assertion = FormatMatchesBaseline("plain", manager)
        assertion_result = assertion.evaluate("different text", CONFIG)

        assert assertion_result.passed is True

    def test_json_to_text_fails(self, tmp_path: Path) -> None:
        """Baseline was JSON, current is plain text: format mismatch."""
        baseline = json.dumps({"key": "value"})
        manager = _create_baseline(tmp_path, "json_to_text", baseline)

        assertion = FormatMatchesBaseline("json_to_text", manager)
        assertion_result = assertion.evaluate("not json anymore", CONFIG)

        assert assertion_result.passed is False

    def test_text_to_json_fails(self, tmp_path: Path) -> None:
        """Baseline was plain text, current is JSON: format mismatch."""
        manager = _create_baseline(tmp_path, "text_to_json", "plain text")

        assertion = FormatMatchesBaseline("text_to_json", manager)
        assertion_result = assertion.evaluate('{"key": "value"}', CONFIG)

        assert assertion_result.passed is False

    def test_no_semantic_computation(self, tmp_path: Path) -> None:
        """FormatMatchesBaseline should not compute semantics (no score field)."""
        baseline = json.dumps({"a": 1})
        current = json.dumps({"a": 2})
        manager = _create_baseline(tmp_path, "no_sem", baseline)

        assertion = FormatMatchesBaseline("no_sem", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.score is None

    def test_failure_message_lists_key_changes(self, tmp_path: Path) -> None:
        baseline = json.dumps({"title": "Report", "version": 1})
        current = json.dumps({"title": "Report", "status": "done"})
        manager = _create_baseline(tmp_path, "msg", baseline)

        assertion = FormatMatchesBaseline("msg", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert "version" in assertion_result.message
        assert "status" in assertion_result.message

    def test_unchanged_keys_in_details(self, tmp_path: Path) -> None:
        baseline = json.dumps({"shared": 1, "removed": 2})
        current = json.dumps({"shared": 1, "added": 3})
        manager = _create_baseline(tmp_path, "unchanged", baseline)

        assertion = FormatMatchesBaseline("unchanged", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert "shared" in assertion_result.details["keys_unchanged"]
