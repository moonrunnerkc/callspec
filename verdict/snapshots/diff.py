"""SnapshotDiff: produces human-readable diffs between snapshots.

Used by regression assertions to report exactly what changed between
the baseline and the current response, and by the CLI `verdict snapshot diff`
command to show developers what shifted before they commit updated baselines.

The diff covers three dimensions independently:
1. Structural: JSON key changes (added, removed, reordered)
2. Content: character-level text differences
3. Semantic: cosine similarity drift (requires verdict[semantic])

Each dimension produces its own section of the diff output, so a developer
can see whether the change is structural, semantic, or both.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiffResult:
    """Structured output of a snapshot comparison.

    Each field is independently populated so consumers can check
    only the dimensions they care about. Regression assertions
    use `semantic_similarity` and `structural_match`; the CLI
    uses all fields including the human-readable text diff.
    """

    snapshot_key: str
    structural_match: bool = True
    semantic_similarity: Optional[float] = None
    semantic_drift: Optional[float] = None

    # JSON key-level structural changes
    keys_added: List[str] = field(default_factory=list)
    keys_removed: List[str] = field(default_factory=list)
    keys_unchanged: List[str] = field(default_factory=list)

    # Content diff as a list of unified diff lines
    content_diff_lines: List[str] = field(default_factory=list)
    content_changed: bool = False

    # Length change
    baseline_length: int = 0
    current_length: int = 0
    length_delta: int = 0

    # Model drift
    baseline_model: str = ""
    current_model: str = ""
    model_changed: bool = False

    def summary(self) -> str:
        """One-line human-readable summary of the diff."""
        parts = []

        if not self.structural_match:
            added = len(self.keys_added)
            removed = len(self.keys_removed)
            parts.append(f"structure changed (+{added}/-{removed} keys)")

        if self.semantic_drift is not None and self.semantic_drift > 0.0:
            parts.append(f"semantic drift {self.semantic_drift:.4f}")

        if self.content_changed:
            parts.append(f"content length delta {self.length_delta:+d} chars")

        if self.model_changed:
            parts.append(f"model changed: {self.baseline_model} -> {self.current_model}")

        if not parts:
            return f"[{self.snapshot_key}] no changes detected"

        return f"[{self.snapshot_key}] " + "; ".join(parts)


class SnapshotDiff:
    """Computes structured diffs between a baseline entry and current content."""

    @staticmethod
    def compare(
        snapshot_key: str,
        baseline_content: str,
        current_content: str,
        baseline_model: str = "",
        current_model: str = "",
        baseline_json_keys: Optional[List[str]] = None,
        compute_semantic: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> DiffResult:
        """Compare baseline and current content across all dimensions.

        Args:
            snapshot_key: Identifier for the snapshot entry.
            baseline_content: The recorded baseline text.
            current_content: The current response text to compare.
            baseline_model: Model that produced the baseline.
            current_model: Model that produced the current response.
            baseline_json_keys: Pre-extracted top-level JSON keys from baseline.
            compute_semantic: Whether to compute semantic similarity (requires embeddings).
            embedding_model: Which sentence-transformers model to use.

        Returns:
            DiffResult with all comparison dimensions populated.
        """
        diff_result = DiffResult(snapshot_key=snapshot_key)

        # Content-level text diff
        _compute_content_diff(diff_result, baseline_content, current_content)

        # Structural JSON key comparison
        _compute_structural_diff(diff_result, baseline_content, current_content, baseline_json_keys)

        # Model drift
        diff_result.baseline_model = baseline_model
        diff_result.current_model = current_model
        diff_result.model_changed = (
            baseline_model != current_model
            and baseline_model != ""
            and current_model != ""
        )

        # Semantic similarity (optional, requires sentence-transformers)
        if compute_semantic:
            _compute_semantic_diff(diff_result, baseline_content, current_content, embedding_model)

        return diff_result


def _compute_content_diff(
    diff_result: DiffResult,
    baseline_content: str,
    current_content: str,
) -> None:
    """Populate the content diff fields on the DiffResult."""
    diff_result.baseline_length = len(baseline_content)
    diff_result.current_length = len(current_content)
    diff_result.length_delta = len(current_content) - len(baseline_content)
    diff_result.content_changed = baseline_content != current_content

    if diff_result.content_changed:
        baseline_lines = baseline_content.splitlines(keepends=True)
        current_lines = current_content.splitlines(keepends=True)
        diff_result.content_diff_lines = list(unified_diff(
            baseline_lines,
            current_lines,
            fromfile="baseline",
            tofile="current",
            lineterm="",
        ))


def _compute_structural_diff(
    diff_result: DiffResult,
    baseline_content: str,
    current_content: str,
    baseline_json_keys: Optional[List[str]],
) -> None:
    """Compare top-level JSON keys between baseline and current content.

    If either side is not valid JSON, structural comparison is skipped
    and structural_match is set based on whether both are non-JSON
    (match) or one is JSON and the other is not (mismatch).
    """
    baseline_keys = baseline_json_keys
    if baseline_keys is None:
        baseline_keys = _extract_json_keys(baseline_content)

    current_keys = _extract_json_keys(current_content)

    # Both non-JSON: structural comparison not applicable, treat as match
    if baseline_keys is None and current_keys is None:
        diff_result.structural_match = True
        return

    # One is JSON, the other is not: structural mismatch
    if baseline_keys is None or current_keys is None:
        diff_result.structural_match = False
        return

    baseline_set = set(baseline_keys)
    current_set = set(current_keys)

    diff_result.keys_added = sorted(current_set - baseline_set)
    diff_result.keys_removed = sorted(baseline_set - current_set)
    diff_result.keys_unchanged = sorted(baseline_set & current_set)
    diff_result.structural_match = (
        len(diff_result.keys_added) == 0 and len(diff_result.keys_removed) == 0
    )


def _compute_semantic_diff(
    diff_result: DiffResult,
    baseline_content: str,
    current_content: str,
    embedding_model: str,
) -> None:
    """Compute cosine similarity between baseline and current content.

    Gracefully handles missing sentence-transformers by logging a
    warning and leaving semantic fields as None.
    """
    try:
        from verdict.scoring.embeddings import score_similarity
        similarity = score_similarity(baseline_content, current_content, embedding_model)
        diff_result.semantic_similarity = similarity
        diff_result.semantic_drift = 1.0 - similarity
    except ImportError:
        logger.warning(
            "Cannot compute semantic diff: sentence-transformers not installed. "
            "Install with: pip install verdict[semantic]"
        )


def _extract_json_keys(text: str) -> Optional[List[str]]:
    """Extract sorted top-level keys from a JSON string, or None if not JSON."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return sorted(parsed.keys())
    except (json.JSONDecodeError, TypeError):
        pass
    return None
