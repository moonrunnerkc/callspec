"""Regression assertions: compare current output against a recorded baseline.

Regression assertions are not about absolute quality. They detect relative
change: has the model's output format shifted, has semantic meaning drifted,
has the response structure become incompatible with what was previously
recorded? This is the mechanism for catching silent model drift when a
provider updates a model without announcement.

Three regression assertions map to three distinct failure modes:

- matches_baseline: catches both structural and semantic regression
- semantic_drift_is_below: catches meaning drift while tolerating format changes
- format_matches_baseline: catches format changes while tolerating content changes

Each assertion loads its baseline from a SnapshotManager instance. The
manager is injected rather than constructed internally, so the snapshot
directory and filename are configurable per test or suite.
"""

from __future__ import annotations

from typing import Optional

from verdict.assertions.base import BaseAssertion
from verdict.core.config import VerdictConfig
from verdict.core.types import IndividualAssertionResult
from verdict.snapshots.diff import SnapshotDiff
from verdict.snapshots.manager import SnapshotManager
from verdict.snapshots.serializer import SnapshotEntry


class MatchesBaseline(BaseAssertion):
    """Assert that the response structurally and semantically matches a recorded baseline.

    Both structure (JSON keys) and semantic similarity must pass independently.
    A response that preserves structure but drifts semantically fails.
    A response that is semantically similar but changes format also fails.
    The two checks produce independent results in the details dict so
    a developer can see exactly which dimension changed.
    """

    assertion_type = "regression"
    assertion_name = "matches_baseline"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
        semantic_threshold: Optional[float] = None,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager
        self._semantic_threshold = semantic_threshold

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._semantic_threshold or config.regression_semantic_threshold
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        diff_result = SnapshotDiff.compare(
            snapshot_key=self._snapshot_key,
            baseline_content=baseline_entry.content,
            current_content=content,
            baseline_model=baseline_entry.model,
            baseline_json_keys=baseline_entry.json_keys,
            compute_semantic=True,
            embedding_model=config.embedding_model,
        )

        structural_passed = diff_result.structural_match
        semantic_similarity = diff_result.semantic_similarity or 0.0
        semantic_passed = semantic_similarity >= threshold
        both_passed = structural_passed and semantic_passed

        details = {
            "snapshot_key": self._snapshot_key,
            "structural_match": structural_passed,
            "semantic_similarity": semantic_similarity,
            "semantic_threshold": threshold,
            "semantic_passed": semantic_passed,
            "baseline_model": baseline_entry.model,
            "baseline_length": diff_result.baseline_length,
            "current_length": diff_result.current_length,
            "length_delta": diff_result.length_delta,
        }

        if not structural_passed:
            details["keys_added"] = diff_result.keys_added
            details["keys_removed"] = diff_result.keys_removed

        if both_passed:
            message = (
                f"MatchesBaseline passed for '{self._snapshot_key}': "
                f"structure matches, semantic similarity {semantic_similarity:.4f} "
                f">= threshold {threshold:.4f}."
            )
        else:
            failure_parts = []
            if not structural_passed:
                failure_parts.append(
                    f"structure mismatch (added keys: {diff_result.keys_added}, "
                    f"removed keys: {diff_result.keys_removed})"
                )
            if not semantic_passed:
                failure_parts.append(
                    f"semantic similarity {semantic_similarity:.4f} below "
                    f"threshold {threshold:.4f}"
                )
            message = (
                f"MatchesBaseline failed for '{self._snapshot_key}': "
                + "; ".join(failure_parts)
                + f". Baseline model: {baseline_entry.model}, "
                f"baseline length: {diff_result.baseline_length}, "
                f"current length: {diff_result.current_length}."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=both_passed,
            message=message,
            score=semantic_similarity,
            threshold=threshold,
            details=details,
        )


class SemanticDriftIsBelow(BaseAssertion):
    """Assert that semantic distance from baseline is below a maximum drift.

    More granular than matches_baseline for cases where structural changes
    are acceptable but semantic drift is not. Drift is defined as
    (1 - cosine_similarity): a drift of 0.0 means identical, 1.0 means
    completely unrelated.
    """

    assertion_type = "regression"
    assertion_name = "semantic_drift_is_below"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
        max_drift: Optional[float] = None,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager
        self._max_drift = max_drift

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        max_drift = self._max_drift or config.regression_drift_ceiling
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        diff_result = SnapshotDiff.compare(
            snapshot_key=self._snapshot_key,
            baseline_content=baseline_entry.content,
            current_content=content,
            compute_semantic=True,
            embedding_model=config.embedding_model,
        )

        semantic_drift = diff_result.semantic_drift or 0.0
        semantic_similarity = diff_result.semantic_similarity or 0.0
        passed = semantic_drift <= max_drift

        details = {
            "snapshot_key": self._snapshot_key,
            "semantic_drift": semantic_drift,
            "semantic_similarity": semantic_similarity,
            "max_drift": max_drift,
            "baseline_model": baseline_entry.model,
            "baseline_length": diff_result.baseline_length,
            "current_length": diff_result.current_length,
        }

        if passed:
            message = (
                f"SemanticDrift passed for '{self._snapshot_key}': "
                f"drift {semantic_drift:.4f} <= max {max_drift:.4f} "
                f"(similarity {semantic_similarity:.4f})."
            )
        else:
            message = (
                f"SemanticDrift failed for '{self._snapshot_key}': "
                f"drift {semantic_drift:.4f} exceeds max {max_drift:.4f} "
                f"(similarity {semantic_similarity:.4f}). "
                f"Baseline model: {baseline_entry.model}. "
                f"Review the prompt or lower max_drift if the change is intentional."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=semantic_drift,
            threshold=max_drift,
            details=details,
        )


class FormatMatchesBaseline(BaseAssertion):
    """Assert that the response format matches the baseline structure.

    Structural-only comparison. Does not evaluate semantic similarity.
    Useful when the model is expected to produce the same JSON shape
    but different content over time (e.g., dynamic data in a fixed schema).
    """

    assertion_type = "regression"
    assertion_name = "format_matches_baseline"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        diff_result = SnapshotDiff.compare(
            snapshot_key=self._snapshot_key,
            baseline_content=baseline_entry.content,
            current_content=content,
            baseline_json_keys=baseline_entry.json_keys,
            compute_semantic=False,
        )

        passed = diff_result.structural_match

        details = {
            "snapshot_key": self._snapshot_key,
            "structural_match": passed,
            "keys_added": diff_result.keys_added,
            "keys_removed": diff_result.keys_removed,
            "keys_unchanged": diff_result.keys_unchanged,
            "baseline_length": diff_result.baseline_length,
            "current_length": diff_result.current_length,
        }

        if passed:
            message = (
                f"FormatMatchesBaseline passed for '{self._snapshot_key}': "
                f"structural format preserved ({len(diff_result.keys_unchanged)} keys unchanged)."
            )
        else:
            message = (
                f"FormatMatchesBaseline failed for '{self._snapshot_key}': "
                f"added keys: {diff_result.keys_added}, "
                f"removed keys: {diff_result.keys_removed}. "
                f"Baseline model: {baseline_entry.model}."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            details=details,
        )
