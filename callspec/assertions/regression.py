"""Regression assertions: compare current output against a recorded baseline.

Regression assertions are not about absolute quality. They detect relative
change: has the model's output format shifted, has the response structure
become incompatible with what was previously recorded? This is the mechanism
for catching silent model drift when a provider updates a model without
announcement.

Two regression assertions map to two distinct failure modes:

- matches_baseline: catches structural regression (key changes, content changes)
- format_matches_baseline: catches format changes while tolerating content changes

Each assertion loads its baseline from a SnapshotManager instance. The
manager is injected rather than constructed internally, so the snapshot
directory and filename are configurable per test or suite.
"""

from __future__ import annotations

from callspec.assertions.base import BaseAssertion
from callspec.core.config import CallspecConfig
from callspec.core.types import IndividualAssertionResult
from callspec.snapshots.diff import SnapshotDiff
from callspec.snapshots.manager import SnapshotManager


class MatchesBaseline(BaseAssertion):
    """Assert that the response structurally matches a recorded baseline.

    Checks structure (JSON keys) and content length delta. A response
    that changes JSON key structure fails. The details dict includes
    all comparison dimensions so a developer can see exactly what shifted.
    """

    assertion_type = "regression"
    assertion_name = "matches_baseline"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager

    def evaluate(self, content: str, config: CallspecConfig) -> IndividualAssertionResult:
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        diff_result = SnapshotDiff.compare(
            snapshot_key=self._snapshot_key,
            baseline_content=baseline_entry.content,
            current_content=content,
            baseline_model=baseline_entry.model,
            baseline_json_keys=baseline_entry.json_keys,
        )

        passed = diff_result.structural_match

        details = {
            "snapshot_key": self._snapshot_key,
            "structural_match": passed,
            "baseline_model": baseline_entry.model,
            "baseline_length": diff_result.baseline_length,
            "current_length": diff_result.current_length,
            "length_delta": diff_result.length_delta,
        }

        if not passed:
            details["keys_added"] = diff_result.keys_added
            details["keys_removed"] = diff_result.keys_removed

        if passed:
            message = (
                f"MatchesBaseline passed for '{self._snapshot_key}': "
                f"structure matches, {len(diff_result.keys_unchanged)} keys unchanged."
            )
        else:
            message = (
                f"MatchesBaseline failed for '{self._snapshot_key}': "
                f"structure mismatch (added keys: {diff_result.keys_added}, "
                f"removed keys: {diff_result.keys_removed}). "
                f"Baseline model: {baseline_entry.model}, "
                f"baseline length: {diff_result.baseline_length}, "
                f"current length: {diff_result.current_length}."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
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

    def evaluate(self, content: str, config: CallspecConfig) -> IndividualAssertionResult:
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        diff_result = SnapshotDiff.compare(
            snapshot_key=self._snapshot_key,
            baseline_content=baseline_entry.content,
            current_content=content,
            baseline_json_keys=baseline_entry.json_keys,
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
