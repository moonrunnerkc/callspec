"""Tests for Phase 4: trajectory snapshots, trajectory diff, trajectory regression assertions."""

import json
import pytest
from pathlib import Path

from callspec.assertions.trajectory_regression import (
    MatchesTrajectoryBaseline,
    TrajectorySequenceMatches,
)
from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.snapshots.diff import SnapshotDiff, TrajectoryDiffResult, ToolCallDiff
from callspec.snapshots.manager import SnapshotManager
from callspec.snapshots.serializer import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotEntry,
    SnapshotSerializer,
    compute_trajectory_hash,
)

CONFIG = CallspecConfig()


# -- Helpers --

def _traj(*calls_spec: tuple[str, dict]) -> ToolCallTrajectory:
    """Build trajectory from (tool_name, arguments) pairs."""
    calls = [
        ToolCall(tool_name=name, arguments=args, call_index=i)
        for i, (name, args) in enumerate(calls_spec)
    ]
    return ToolCallTrajectory(calls=calls, model="gpt-4o", provider="openai")


def _calls_dicts(*calls_spec: tuple[str, dict]) -> list[dict]:
    """Build raw tool call dicts."""
    return [
        {"tool_name": name, "arguments": args, "call_index": i}
        for i, (name, args) in enumerate(calls_spec)
    ]


# ── SnapshotSerializer: trajectory fields ──

class TestSnapshotEntryTrajectory:
    def test_entry_with_tool_calls(self):
        tool_calls = _calls_dicts(("search", {"query": "hello"}), ("book", {"id": 1}))
        entry = SnapshotEntry(
            snapshot_key="test",
            content="",
            prompt="test prompt",
            tool_calls=tool_calls,
        )
        assert entry.has_trajectory
        assert len(entry.tool_calls) == 2
        assert entry.tool_names == ["search", "book"]
        assert entry.trajectory_hash != ""

    def test_entry_without_tool_calls(self):
        entry = SnapshotEntry(
            snapshot_key="test",
            content="hello",
            prompt="test",
        )
        assert not entry.has_trajectory
        assert entry.tool_names == []
        assert entry.trajectory_hash == ""

    def test_trajectory_hash_deterministic(self):
        calls = _calls_dicts(("search", {"query": "a"}))
        h1 = compute_trajectory_hash(calls)
        h2 = compute_trajectory_hash(calls)
        assert h1 == h2

    def test_trajectory_hash_changes_on_tool_name(self):
        h1 = compute_trajectory_hash(_calls_dicts(("search", {"q": "a"})))
        h2 = compute_trajectory_hash(_calls_dicts(("lookup", {"q": "a"})))
        assert h1 != h2

    def test_trajectory_hash_changes_on_arg_keys(self):
        h1 = compute_trajectory_hash(_calls_dicts(("search", {"query": "a"})))
        h2 = compute_trajectory_hash(_calls_dicts(("search", {"q": "a"})))
        assert h1 != h2

    def test_trajectory_hash_ignores_arg_values(self):
        h1 = compute_trajectory_hash(_calls_dicts(("search", {"query": "hello"})))
        h2 = compute_trajectory_hash(_calls_dicts(("search", {"query": "world"})))
        assert h1 == h2

    def test_serialize_deserialize_with_trajectory(self):
        calls = _calls_dicts(("search", {"query": "test"}))
        entry = SnapshotEntry(
            snapshot_key="key1",
            content="response",
            prompt="prompt",
            tool_calls=calls,
        )
        data = SnapshotSerializer.serialize_entry(entry)
        restored = SnapshotSerializer.deserialize_entry(data)

        assert restored.tool_calls == calls
        assert restored.trajectory_hash == entry.trajectory_hash
        assert restored.has_trajectory

    def test_deserialize_v1_entry_no_trajectory(self):
        """V1 entries without trajectory fields get empty defaults."""
        data = {
            "snapshot_key": "old_entry",
            "content": "hello",
            "prompt": "say hi",
            "model": "gpt-4",
        }
        entry = SnapshotSerializer.deserialize_entry(data)
        assert entry.tool_calls == []
        assert entry.trajectory_hash == ""
        assert not entry.has_trajectory


class TestSchemaVersionMigration:
    def test_v1_file_migrates_to_v2(self, tmp_path):
        """A V1 snapshot file should be loadable and migrated to V2."""
        v1_data = {
            "schema_version": 1,
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "entries": {
                "old_entry": {
                    "snapshot_key": "old_entry",
                    "content": "hello world",
                    "prompt": "say hi",
                    "model": "gpt-4",
                    "provider": "openai",
                    "timestamp": "2025-01-01T00:00:00",
                    "content_length": 11,
                    "json_keys": None,
                    "metadata": {},
                }
            },
        }
        filepath = tmp_path / "baselines.json"
        filepath.write_text(json.dumps(v1_data), encoding="utf-8")

        loaded = SnapshotSerializer.load(filepath)
        assert loaded.schema_version == SNAPSHOT_SCHEMA_VERSION

        entry = SnapshotSerializer.deserialize_entry(loaded.entries["old_entry"])
        assert entry.tool_calls == []
        assert not entry.has_trajectory

    def test_future_version_raises(self, tmp_path):
        future_data = {
            "schema_version": 999,
            "created_at": "",
            "updated_at": "",
            "entries": {},
        }
        filepath = tmp_path / "baselines.json"
        filepath.write_text(json.dumps(future_data), encoding="utf-8")

        with pytest.raises(Exception, match="newer than supported"):
            SnapshotSerializer.load(filepath)


# ── SnapshotManager with trajectory ──

class TestSnapshotManagerTrajectory:
    def test_create_entry_with_tool_calls(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        calls = _calls_dicts(("search", {"q": "hello"}))
        entry = manager.create_entry(
            snapshot_key="agent_flow",
            content="ok",
            prompt="do something",
            tool_calls=calls,
        )
        assert entry.has_trajectory
        assert len(entry.tool_calls) == 1

        # Reload and verify persistence
        loaded = manager.get_entry("agent_flow")
        assert loaded.has_trajectory
        assert loaded.tool_calls == calls

    def test_update_entry_with_tool_calls(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="v1",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"q": "a"})),
        )
        updated = manager.update_entry(
            snapshot_key="flow",
            content="v2",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"q": "a"}), ("book", {"id": 1})),
        )
        assert len(updated.tool_calls) == 2

        loaded = manager.get_entry("flow")
        assert len(loaded.tool_calls) == 2


# ── Trajectory Diff ──

class TestTrajectoryDiff:
    def test_identical_trajectories(self):
        calls = _calls_dicts(("search", {"q": "a"}), ("book", {"id": 1}))
        result = SnapshotDiff.compare_trajectories("key", calls, calls)
        assert result.sequence_match
        assert result.hash_match
        assert not result.has_changes

    def test_tool_added(self):
        baseline = _calls_dicts(("search", {"q": "a"}))
        current = _calls_dicts(("search", {"q": "a"}), ("book", {"id": 1}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        assert not result.sequence_match
        assert "book" in result.tools_added

    def test_tool_removed(self):
        baseline = _calls_dicts(("search", {"q": "a"}), ("book", {"id": 1}))
        current = _calls_dicts(("search", {"q": "a"}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        assert not result.sequence_match
        assert "book" in result.tools_removed

    def test_tool_reordered(self):
        baseline = _calls_dicts(("search", {}), ("book", {}))
        current = _calls_dicts(("book", {}), ("search", {}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        assert not result.sequence_match
        # No tools added or removed, just reordered
        assert result.tools_added == []
        assert result.tools_removed == []

    def test_argument_key_change_detected_by_hash(self):
        baseline = _calls_dicts(("search", {"query": "a"}))
        current = _calls_dicts(("search", {"q": "a"}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        assert result.sequence_match  # same tool sequence
        assert not result.hash_match  # argument keys changed

    def test_argument_value_change_not_detected_by_hash(self):
        baseline = _calls_dicts(("search", {"query": "hello"}))
        current = _calls_dicts(("search", {"query": "world"}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        assert result.sequence_match
        assert result.hash_match  # values dont affect hash

    def test_model_change_detected(self):
        calls = _calls_dicts(("search", {}))
        result = SnapshotDiff.compare_trajectories(
            "key", calls, calls,
            baseline_model="gpt-4o-2024-05", current_model="gpt-4o-2024-11",
        )
        assert result.model_changed

    def test_model_unchanged(self):
        calls = _calls_dicts(("search", {}))
        result = SnapshotDiff.compare_trajectories(
            "key", calls, calls,
            baseline_model="gpt-4o", current_model="gpt-4o",
        )
        assert not result.model_changed

    def test_call_diffs_added_position(self):
        baseline = _calls_dicts(("search", {}))
        current = _calls_dicts(("search", {}), ("book", {}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        added = [d for d in result.call_diffs if d.status == "added"]
        assert len(added) == 1
        assert added[0].current_tool == "book"

    def test_call_diffs_removed_position(self):
        baseline = _calls_dicts(("search", {}), ("book", {}))
        current = _calls_dicts(("search", {}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        removed = [d for d in result.call_diffs if d.status == "removed"]
        assert len(removed) == 1
        assert removed[0].baseline_tool == "book"

    def test_call_diffs_arg_change(self):
        baseline = _calls_dicts(("search", {"query": "a", "limit": 10}))
        current = _calls_dicts(("search", {"query": "b", "limit": 10}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        changed = [d for d in result.call_diffs if d.status == "changed"]
        assert len(changed) == 1
        assert "query" in changed[0].args_changed

    def test_call_diffs_arg_added_removed(self):
        baseline = _calls_dicts(("search", {"query": "a"}))
        current = _calls_dicts(("search", {"q": "a", "extra": "b"}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        changed = [d for d in result.call_diffs if d.status == "changed"]
        assert len(changed) == 1
        assert "query" in changed[0].args_removed
        assert "q" in changed[0].args_added
        assert "extra" in changed[0].args_added

    def test_summary_no_changes(self):
        calls = _calls_dicts(("search", {}))
        result = SnapshotDiff.compare_trajectories("key", calls, calls)
        assert "unchanged" in result.summary()

    def test_summary_with_changes(self):
        baseline = _calls_dicts(("search", {}))
        current = _calls_dicts(("search", {}), ("book", {}))
        result = SnapshotDiff.compare_trajectories("key", baseline, current)
        summary = result.summary()
        assert "book" in summary

    def test_empty_trajectories_match(self):
        result = SnapshotDiff.compare_trajectories("key", [], [])
        assert result.sequence_match
        assert result.hash_match


# ── Trajectory Regression Assertions ──

class TestMatchesTrajectoryBaseline:
    def test_passes_identical_trajectory(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        calls = _calls_dicts(("search", {"query": "test"}), ("book", {"id": 1}))
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=calls,
        )

        trajectory = _traj(("search", {"query": "test"}), ("book", {"id": 1}))
        assertion = MatchesTrajectoryBaseline("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert result.passed
        assert "matches baseline" in result.message.lower()

    def test_fails_different_tool_sequence(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"q": "a"}), ("book", {"id": 1})),
        )

        trajectory = _traj(("search", {"q": "a"}), ("summarize", {"text": "x"}))
        assertion = MatchesTrajectoryBaseline("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert not result.passed
        assert result.assertion_type == "regression"

    def test_fails_argument_key_change(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"query": "a"})),
        )

        trajectory = _traj(("search", {"q": "a"}))  # key changed from "query" to "q"
        assertion = MatchesTrajectoryBaseline("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert not result.passed

    def test_passes_argument_value_change(self, tmp_path):
        """Same tool, same arg keys, different values should still pass."""
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"query": "hello"})),
        )

        trajectory = _traj(("search", {"query": "world"}))
        assertion = MatchesTrajectoryBaseline("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert result.passed

    def test_fails_when_baseline_has_no_trajectory(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="text_only",
            content="hello",
            prompt="say hi",
        )

        trajectory = _traj(("search", {"q": "a"}))
        assertion = MatchesTrajectoryBaseline("text_only", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert not result.passed
        assert "no trajectory data" in result.message


class TestTrajectorySequenceMatches:
    def test_passes_same_sequence(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"q": "a"}), ("book", {"id": 1})),
        )

        trajectory = _traj(("search", {"q": "different"}), ("book", {"id": 99}))
        assertion = TrajectorySequenceMatches("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert result.passed

    def test_fails_different_sequence(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {}), ("book", {})),
        )

        trajectory = _traj(("book", {}), ("search", {}))
        assertion = TrajectorySequenceMatches("flow", manager)
        result = assertion.evaluate_trajectory(trajectory, CONFIG)
        assert not result.passed

    def test_fails_no_trajectory_in_baseline(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="text",
            content="hello",
            prompt="test",
        )

        trajectory = _traj(("search", {}))
        result = TrajectorySequenceMatches("text", manager).evaluate_trajectory(trajectory, CONFIG)
        assert not result.passed


# ── Fluent API integration ──

class TestTrajectoryBuilderRegression:
    def test_matches_baseline_via_builder(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        calls = _calls_dicts(("search", {"query": "test"}))
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=calls,
        )

        trajectory = _traj(("search", {"query": "test"}))
        result = (
            TrajectoryBuilder(trajectory)
            .matches_baseline("flow", manager)
            .run()
        )
        assert result.passed

    def test_sequence_matches_via_builder(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(("search", {"q": "a"})),
        )

        trajectory = _traj(("search", {"q": "different_value"}))
        result = (
            TrajectoryBuilder(trajectory)
            .sequence_matches_baseline("flow", manager)
            .run()
        )
        assert result.passed

    def test_mixed_contract_and_regression(self, tmp_path):
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="test.json")
        manager.create_entry(
            snapshot_key="flow",
            content="",
            prompt="test",
            tool_calls=_calls_dicts(
                ("search", {"query": "hello"}),
                ("book", {"id": 1}),
            ),
        )

        trajectory = _traj(
            ("search", {"query": "world"}),
            ("book", {"id": 2}),
        )
        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(["search", "book"])
            .argument_not_empty("search", "query")
            .matches_baseline("flow", manager)
            .run()
        )
        assert result.passed
        assert len(result.assertions) == 3
