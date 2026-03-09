"""Trajectory regression assertions: compare tool-call baselines.

These assertions detect when a model swap or version bump changes
tool-calling behavior. They operate on ToolCallTrajectory and compare
against recorded baselines stored through the SnapshotManager.

MatchesTrajectoryBaseline checks that the tool sequence and argument
structure are preserved. TrajectorySequenceMatches checks only the
tool name sequence (ignoring arguments).
"""

from __future__ import annotations

from typing import Any

from llm_assert.assertions.trajectory_base import TrajectoryAssertion
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.trajectory import ToolCallTrajectory
from llm_assert.core.types import IndividualAssertionResult
from llm_assert.snapshots.diff import SnapshotDiff
from llm_assert.snapshots.manager import SnapshotManager
from llm_assert.snapshots.serializer import compute_trajectory_hash


class MatchesTrajectoryBaseline(TrajectoryAssertion):
    """Assert the trajectory matches a recorded baseline.

    Checks both tool name sequence and trajectory hash (tool names +
    argument keys). Passes only when both match. The detailed diff
    is included in the result details for debugging.
    """

    assertion_type = "regression"
    assertion_name = "matches_trajectory_baseline"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        if not baseline_entry.has_trajectory:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Baseline '{self._snapshot_key}' has no trajectory data. "
                    f"Update the baseline with tool-call data using "
                    f"'llm-assert snapshot update' with a trajectory-enabled run."
                ),
                details={
                    "snapshot_key": self._snapshot_key,
                    "error": "no_trajectory_in_baseline",
                },
            )

        current_calls = [c.to_dict() for c in trajectory.calls]
        current_hash = compute_trajectory_hash(current_calls)

        diff_result = SnapshotDiff.compare_trajectories(
            snapshot_key=self._snapshot_key,
            baseline_calls=baseline_entry.tool_calls,
            current_calls=current_calls,
            baseline_model=baseline_entry.model,
            current_model=trajectory.model,
            baseline_hash=baseline_entry.trajectory_hash,
            current_hash=current_hash,
        )

        passed = diff_result.sequence_match and diff_result.hash_match

        details: dict[str, Any] = {
            "snapshot_key": self._snapshot_key,
            "sequence_match": diff_result.sequence_match,
            "hash_match": diff_result.hash_match,
            "baseline_tools": diff_result.baseline_tool_names,
            "current_tools": diff_result.current_tool_names,
            "baseline_model": diff_result.baseline_model,
            "current_model": diff_result.current_model,
            "model_changed": diff_result.model_changed,
        }

        if not passed:
            details["tools_added"] = diff_result.tools_added
            details["tools_removed"] = diff_result.tools_removed
            details["call_diffs"] = [
                {
                    "position": cd.position,
                    "status": cd.status,
                    "baseline_tool": cd.baseline_tool,
                    "current_tool": cd.current_tool,
                    "args_added": cd.args_added,
                    "args_removed": cd.args_removed,
                    "args_changed": cd.args_changed,
                }
                for cd in diff_result.call_diffs
                if cd.status != "unchanged"
            ]

        if passed:
            message = (
                f"Trajectory matches baseline '{self._snapshot_key}': "
                f"{len(trajectory)} calls, sequence and argument structure preserved."
            )
        else:
            message = diff_result.summary()

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            details=details,
        )


class TrajectorySequenceMatches(TrajectoryAssertion):
    """Assert that the tool name sequence matches the baseline.

    Ignores argument changes. Useful when arguments are expected to vary
    (e.g., search queries change over time) but the tool sequence should
    remain stable.
    """

    assertion_type = "regression"
    assertion_name = "trajectory_sequence_matches"

    def __init__(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> None:
        self._snapshot_key = snapshot_key
        self._snapshot_manager = snapshot_manager

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        baseline_entry = self._snapshot_manager.get_entry(self._snapshot_key)

        if not baseline_entry.has_trajectory:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Baseline '{self._snapshot_key}' has no trajectory data."
                ),
                details={
                    "snapshot_key": self._snapshot_key,
                    "error": "no_trajectory_in_baseline",
                },
            )

        baseline_names = baseline_entry.tool_names
        current_names = trajectory.tool_names

        passed = baseline_names == current_names

        if passed:
            message = (
                f"Tool sequence matches baseline '{self._snapshot_key}': "
                f"{current_names}."
            )
        else:
            message = (
                f"Tool sequence changed for '{self._snapshot_key}'. "
                f"Baseline: {baseline_names}. Current: {current_names}."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            details={
                "snapshot_key": self._snapshot_key,
                "baseline_sequence": baseline_names,
                "current_sequence": current_names,
            },
        )
