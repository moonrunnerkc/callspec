"""SnapshotDiff: produces human-readable diffs between snapshots.

Used by regression assertions to report exactly what changed between
the baseline and the current response, and by the CLI `callspec snapshot diff`
command to show developers what shifted before they commit updated baselines.

The diff covers three dimensions independently:
1. Structural: JSON key changes (added, removed, reordered)
2. Content: character-level text differences
3. Trajectory: tool-call sequence and argument changes

Each dimension produces its own section of the diff output, so a developer
can see whether the change is structural, content-level, or trajectory-level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Any

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
    semantic_similarity: float | None = None
    semantic_drift: float | None = None

    # JSON key-level structural changes
    keys_added: list[str] = field(default_factory=list)
    keys_removed: list[str] = field(default_factory=list)
    keys_unchanged: list[str] = field(default_factory=list)

    # Content diff as a list of unified diff lines
    content_diff_lines: list[str] = field(default_factory=list)
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


@dataclass
class ToolCallDiff:
    """Diff for a single tool call position in the trajectory."""

    position: int
    baseline_tool: str | None = None
    current_tool: str | None = None
    status: str = "unchanged"  # unchanged, added, removed, changed

    # Argument-level changes (only populated when both sides have the same tool)
    args_added: list[str] = field(default_factory=list)
    args_removed: list[str] = field(default_factory=list)
    args_changed: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary_line(self) -> str:
        """One-line description of this call diff."""
        if self.status == "unchanged":
            return f"  [{self.position}] {self.baseline_tool}: unchanged"
        if self.status == "added":
            return f"  [{self.position}] + {self.current_tool}: new call"
        if self.status == "removed":
            return f"  [{self.position}] - {self.baseline_tool}: removed"
        # changed: tool name or arguments differ
        if self.baseline_tool != self.current_tool:
            return (
                f"  [{self.position}] ~ {self.baseline_tool} -> {self.current_tool}: "
                f"tool changed"
            )
        parts = []
        if self.args_added:
            parts.append(f"+args: {self.args_added}")
        if self.args_removed:
            parts.append(f"-args: {self.args_removed}")
        if self.args_changed:
            parts.append(f"~args: {list(self.args_changed.keys())}")
        return f"  [{self.position}] ~ {self.baseline_tool}: {'; '.join(parts)}"


@dataclass
class TrajectoryDiffResult:
    """Structured diff between two tool-call trajectories."""

    snapshot_key: str
    sequence_match: bool = True
    hash_match: bool = True
    call_diffs: list[ToolCallDiff] = field(default_factory=list)

    baseline_tool_names: list[str] = field(default_factory=list)
    current_tool_names: list[str] = field(default_factory=list)
    tools_added: list[str] = field(default_factory=list)
    tools_removed: list[str] = field(default_factory=list)

    baseline_hash: str = ""
    current_hash: str = ""

    # Model drift
    baseline_model: str = ""
    current_model: str = ""
    model_changed: bool = False

    @property
    def has_changes(self) -> bool:
        return not self.sequence_match or not self.hash_match or self.model_changed

    def summary(self) -> str:
        """Human-readable summary of trajectory changes."""
        parts = []

        if not self.sequence_match:
            parts.append(
                f"tool sequence changed: "
                f"{self.baseline_tool_names} -> {self.current_tool_names}"
            )

        if self.tools_added:
            parts.append(f"new tools: {self.tools_added}")

        if self.tools_removed:
            parts.append(f"removed tools: {self.tools_removed}")

        if self.sequence_match and not self.hash_match:
            parts.append("argument keys changed (same tool sequence)")

        if self.model_changed:
            parts.append(f"model: {self.baseline_model} -> {self.current_model}")

        if not parts:
            return f"[{self.snapshot_key}] trajectory unchanged"

        return f"[{self.snapshot_key}] " + "; ".join(parts)

    def detailed_report(self) -> str:
        """Multi-line report showing per-call diffs."""
        lines = [self.summary()]
        for cd in self.call_diffs:
            if cd.status != "unchanged":
                lines.append(cd.summary_line())
        return "\n".join(lines)


class SnapshotDiff:
    """Computes structured diffs between a baseline entry and current content."""

    @staticmethod
    def compare(
        snapshot_key: str,
        baseline_content: str,
        current_content: str,
        baseline_model: str = "",
        current_model: str = "",
        baseline_json_keys: list[str] | None = None,
    ) -> DiffResult:
        """Compare baseline and current content across structural dimensions.

        Args:
            snapshot_key: Identifier for the snapshot entry.
            baseline_content: The recorded baseline text.
            current_content: The current response text to compare.
            baseline_model: Model that produced the baseline.
            current_model: Model that produced the current response.
            baseline_json_keys: Pre-extracted top-level JSON keys from baseline.

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

        return diff_result

    @staticmethod
    def compare_trajectories(
        snapshot_key: str,
        baseline_calls: list[dict[str, Any]],
        current_calls: list[dict[str, Any]],
        baseline_model: str = "",
        current_model: str = "",
        baseline_hash: str = "",
        current_hash: str = "",
    ) -> TrajectoryDiffResult:
        """Compare two tool-call trajectories.

        Produces a detailed diff covering:
        - Tool name sequence changes (additions, removals, reorders)
        - Per-call argument changes (added/removed/changed keys and values)
        - Trajectory hash comparison for fast equality check

        Args:
            baseline_calls: List of tool call dicts from the baseline.
            current_calls: List of tool call dicts from the current run.
            baseline_hash: Pre-computed trajectory hash from baseline.
            current_hash: Pre-computed trajectory hash from current.

        Returns:
            TrajectoryDiffResult with full comparison detail.
        """
        from callspec.snapshots.serializer import compute_trajectory_hash

        result = TrajectoryDiffResult(snapshot_key=snapshot_key)

        result.baseline_tool_names = [
            tc.get("tool_name", "") for tc in baseline_calls
        ]
        result.current_tool_names = [
            tc.get("tool_name", "") for tc in current_calls
        ]

        # Hash comparison
        result.baseline_hash = baseline_hash or compute_trajectory_hash(baseline_calls)
        result.current_hash = current_hash or compute_trajectory_hash(current_calls)
        result.hash_match = result.baseline_hash == result.current_hash

        # Sequence comparison
        result.sequence_match = (
            result.baseline_tool_names == result.current_tool_names
        )

        # Set-level tool changes
        baseline_set = set(result.baseline_tool_names)
        current_set = set(result.current_tool_names)
        result.tools_added = sorted(current_set - baseline_set)
        result.tools_removed = sorted(baseline_set - current_set)

        # Model drift
        result.baseline_model = baseline_model
        result.current_model = current_model
        result.model_changed = (
            baseline_model != current_model
            and baseline_model != ""
            and current_model != ""
        )

        # Per-call diff using longest common subsequence alignment
        result.call_diffs = _compute_call_diffs(baseline_calls, current_calls)

        return result


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
    baseline_json_keys: list[str] | None,
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


def _extract_json_keys(text: str) -> list[str] | None:
    """Extract sorted top-level keys from a JSON string, or None if not JSON."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return sorted(parsed.keys())
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# -- Trajectory diff helpers --

def _compute_call_diffs(
    baseline_calls: list[dict[str, Any]],
    current_calls: list[dict[str, Any]],
) -> list[ToolCallDiff]:
    """Produce per-position diffs between two tool-call sequences.

    Uses a simple parallel walk. When sequences differ in length,
    extra calls on either side are flagged as additions or removals.
    When both sides have a call at the same position, compares tool
    names and arguments.
    """
    diffs: list[ToolCallDiff] = []
    max_len = max(len(baseline_calls), len(current_calls))

    for i in range(max_len):
        has_baseline = i < len(baseline_calls)
        has_current = i < len(current_calls)

        if has_baseline and has_current:
            b_call = baseline_calls[i]
            c_call = current_calls[i]
            b_name = b_call.get("tool_name", "")
            c_name = c_call.get("tool_name", "")

            if b_name == c_name:
                # Same tool at same position: compare arguments
                arg_diff = _diff_arguments(
                    b_call.get("arguments", {}),
                    c_call.get("arguments", {}),
                )
                if arg_diff["has_changes"]:
                    diffs.append(ToolCallDiff(
                        position=i,
                        baseline_tool=b_name,
                        current_tool=c_name,
                        status="changed",
                        args_added=arg_diff["added"],
                        args_removed=arg_diff["removed"],
                        args_changed=arg_diff["changed"],
                    ))
                else:
                    diffs.append(ToolCallDiff(
                        position=i,
                        baseline_tool=b_name,
                        current_tool=c_name,
                        status="unchanged",
                    ))
            else:
                # Different tool at same position
                diffs.append(ToolCallDiff(
                    position=i,
                    baseline_tool=b_name,
                    current_tool=c_name,
                    status="changed",
                ))

        elif has_current and not has_baseline:
            c_name = current_calls[i].get("tool_name", "")
            diffs.append(ToolCallDiff(
                position=i,
                current_tool=c_name,
                status="added",
            ))

        else:
            b_name = baseline_calls[i].get("tool_name", "")
            diffs.append(ToolCallDiff(
                position=i,
                baseline_tool=b_name,
                status="removed",
            ))

    return diffs


def _diff_arguments(
    baseline_args: dict[str, Any],
    current_args: dict[str, Any],
) -> dict[str, Any]:
    """Compare argument dicts and return structured changes.

    Returns a dict with keys: added, removed, changed, has_changes.
    The 'changed' dict maps key -> {baseline: ..., current: ...}.
    """
    b_keys = set(baseline_args.keys()) if isinstance(baseline_args, dict) else set()
    c_keys = set(current_args.keys()) if isinstance(current_args, dict) else set()

    added = sorted(c_keys - b_keys)
    removed = sorted(b_keys - c_keys)

    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(b_keys & c_keys):
        b_val = baseline_args[key]
        c_val = current_args[key]
        if b_val != c_val:
            changed[key] = {"baseline": b_val, "current": c_val}

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "has_changes": bool(added or removed or changed),
    }
