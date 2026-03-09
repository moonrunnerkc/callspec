"""Unit tests for regression assertions against synthetic baselines.

Each test creates a baseline via SnapshotManager, then evaluates the
regression assertion against current content. Tests cover MatchesBaseline
(structural comparison) and FormatMatchesBaseline (format-only comparison)
across their pass/fail boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

from callspec.assertions.regression import (
    FormatMatchesBaseline,
    MatchesBaseline,
)
from callspec.core.config import CallspecConfig
from callspec.snapshots.manager import SnapshotManager

CONFIG = CallspecConfig()


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

    def test_failure_message_includes_diagnostics(self, tmp_path: Path) -> None:
        baseline = json.dumps({"report": "Annual sales exceeded targets."})
        current = json.dumps({"report": "The mitochondria is the powerhouse of the cell."})
        manager = _create_baseline(tmp_path, "diag", baseline)

        assertion = MatchesBaseline("diag", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert "diag" in assertion_result.message
        # Same JSON keys, so structural match passes
        assert assertion_result.passed is True

    def test_both_plain_text_structural_match(self, tmp_path: Path) -> None:
        """Two plain-text (non-JSON) responses: structural dimension treated as match."""
        baseline = "Climate change causes rising sea levels and coastal erosion in many regions."
        current = (
            "Rising sea levels and coastal erosion are caused "
            "by climate change in many areas."
        )
        manager = _create_baseline(tmp_path, "plaintext", baseline)

        assertion = MatchesBaseline("plaintext", manager)
        assertion_result = assertion.evaluate(current, CONFIG)

        assert assertion_result.details["structural_match"] is True


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
