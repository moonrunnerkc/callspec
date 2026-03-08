"""Unit tests for the snapshot system: serializer, manager, and diff.

Tests the full snapshot lifecycle end-to-end: create, load, update, compare,
diff, and delete. All tests use tmp_path to avoid polluting the working
directory with snapshot files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict.errors import SnapshotError
from verdict.snapshots.diff import SnapshotDiff
from verdict.snapshots.manager import SnapshotManager
from verdict.snapshots.serializer import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotEntry,
    SnapshotFile,
    SnapshotSerializer,
)

# ---------------------------------------------------------------------------
# SnapshotEntry
# ---------------------------------------------------------------------------

class TestSnapshotEntry:

    def test_content_length_auto_computed(self) -> None:
        entry = SnapshotEntry(snapshot_key="test", content="hello world", prompt="say hi")
        assert entry.content_length == 11

    def test_json_keys_extracted_from_json_content(self) -> None:
        content = json.dumps({"title": "Test", "summary": "A summary", "score": 0.9})
        entry = SnapshotEntry(snapshot_key="json_test", content=content, prompt="generate json")
        assert entry.json_keys == ["score", "summary", "title"]

    def test_json_keys_none_for_plain_text(self) -> None:
        entry = SnapshotEntry(snapshot_key="text_test", content="plain text", prompt="say hi")
        assert entry.json_keys is None

    def test_json_keys_none_for_json_array(self) -> None:
        entry = SnapshotEntry(snapshot_key="arr_test", content="[1, 2, 3]", prompt="list")
        assert entry.json_keys is None

    def test_explicit_json_keys_not_overwritten(self) -> None:
        """Pre-set json_keys are preserved, not re-extracted from content."""
        entry = SnapshotEntry(
            snapshot_key="explicit",
            content=json.dumps({"a": 1}),
            prompt="test",
            json_keys=["x", "y"],
        )
        assert entry.json_keys == ["x", "y"]


# ---------------------------------------------------------------------------
# SnapshotSerializer
# ---------------------------------------------------------------------------

class TestSnapshotSerializer:

    def test_round_trip_entry(self) -> None:
        original = SnapshotEntry(
            snapshot_key="roundtrip",
            content='{"key": "value"}',
            prompt="generate json",
            model="gpt-4o-2024-11-20",
            provider="openai",
            metadata={"run_id": "abc123"},
        )
        serialized = SnapshotSerializer.serialize_entry(original)
        restored = SnapshotSerializer.deserialize_entry(serialized)

        assert restored.snapshot_key == original.snapshot_key
        assert restored.content == original.content
        assert restored.prompt == original.prompt
        assert restored.model == original.model
        assert restored.provider == original.provider
        assert restored.content_length == original.content_length
        assert restored.json_keys == original.json_keys
        assert restored.metadata == original.metadata

    def test_deserialize_tolerates_missing_optional_fields(self) -> None:
        minimal = {"snapshot_key": "minimal", "content": "hello", "prompt": "say hi"}
        entry = SnapshotSerializer.deserialize_entry(minimal)
        assert entry.model == "unknown"
        assert entry.provider == "unknown"
        assert entry.content_length == 5
        assert entry.metadata == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        filepath = tmp_path / "test_snapshot.json"
        snapshot_file = SnapshotFile()
        entry = SnapshotEntry(snapshot_key="save_test", content="test content", prompt="test")
        snapshot_file.entries["save_test"] = SnapshotSerializer.serialize_entry(entry)

        SnapshotSerializer.save(filepath, snapshot_file)
        assert filepath.exists()

        loaded = SnapshotSerializer.load(filepath)
        assert loaded.schema_version == SNAPSHOT_SCHEMA_VERSION
        assert "save_test" in loaded.entries
        assert loaded.entries["save_test"]["content"] == "test content"

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        filepath = tmp_path / "nested" / "deep" / "snapshot.json"
        SnapshotSerializer.save(filepath, SnapshotFile())
        assert filepath.exists()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "nonexistent.json"
        with pytest.raises(SnapshotError, match="not found"):
            SnapshotSerializer.load(filepath)

    def test_load_corrupt_json_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "corrupt.json"
        filepath.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SnapshotError, match="not valid JSON"):
            SnapshotSerializer.load(filepath)

    def test_load_missing_schema_version_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "no_version.json"
        filepath.write_text('{"entries": {}}', encoding="utf-8")
        with pytest.raises(SnapshotError, match="missing 'schema_version'"):
            SnapshotSerializer.load(filepath)

    def test_load_future_schema_version_raises(self, tmp_path: Path) -> None:
        filepath = tmp_path / "future.json"
        payload = {"schema_version": SNAPSHOT_SCHEMA_VERSION + 99, "entries": {}}
        filepath.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SnapshotError, match="newer than supported"):
            SnapshotSerializer.load(filepath)

    def test_saved_json_is_formatted(self, tmp_path: Path) -> None:
        """Snapshot files use 2-space indent for readable git diffs."""
        filepath = tmp_path / "formatted.json"
        SnapshotSerializer.save(filepath, SnapshotFile())
        raw = filepath.read_text(encoding="utf-8")
        assert "\n  " in raw


# ---------------------------------------------------------------------------
# SnapshotManager
# ---------------------------------------------------------------------------

class TestSnapshotManager:

    def test_create_and_get_entry(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("key1", content="hello world", prompt="say hi", model="mock-v1")

        entry = manager.get_entry("key1")
        assert entry.content == "hello world"
        assert entry.model == "mock-v1"
        assert entry.snapshot_key == "key1"

    def test_create_duplicate_raises_without_overwrite(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("dup", content="first", prompt="test")
        with pytest.raises(SnapshotError, match="already exists"):
            manager.create_entry("dup", content="second", prompt="test")

    def test_create_duplicate_succeeds_with_overwrite(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("dup", content="first", prompt="test")
        manager.create_entry("dup", content="second", prompt="test", overwrite=True)

        entry = manager.get_entry("dup")
        assert entry.content == "second"

    def test_update_creates_if_missing(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.update_entry("new_key", content="created via update", prompt="test")

        entry = manager.get_entry("new_key")
        assert entry.content == "created via update"

    def test_update_replaces_existing(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("upd", content="original", prompt="test")
        manager.update_entry("upd", content="updated", prompt="test")

        entry = manager.get_entry("upd")
        assert entry.content == "updated"

    def test_get_missing_key_raises(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("exists", content="here", prompt="test")
        with pytest.raises(SnapshotError, match="No baseline entry found"):
            manager.get_entry("does_not_exist")

    def test_get_missing_key_lists_available(self, tmp_path: Path) -> None:
        """Error message includes available keys for discoverability."""
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("alpha", content="a", prompt="test")
        manager.create_entry("beta", content="b", prompt="test")
        with pytest.raises(SnapshotError, match="alpha") as exc_info:
            manager.get_entry("gamma")
        assert "beta" in str(exc_info.value)

    def test_delete_entry(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("to_delete", content="temp", prompt="test")
        assert "to_delete" in manager.list_keys()

        manager.delete_entry("to_delete")
        assert "to_delete" not in manager.list_keys()

    def test_delete_missing_entry_raises(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("keep", content="keep", prompt="test")
        with pytest.raises(SnapshotError, match="Cannot delete"):
            manager.delete_entry("ghost")

    def test_delete_all(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("a", content="1", prompt="test")
        manager.create_entry("b", content="2", prompt="test")
        manager.create_entry("c", content="3", prompt="test")

        count = manager.delete_all()
        assert count == 3
        assert manager.list_keys() == []

    def test_list_keys_sorted(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("charlie", content="c", prompt="test")
        manager.create_entry("alpha", content="a", prompt="test")
        manager.create_entry("bravo", content="b", prompt="test")

        assert manager.list_keys() == ["alpha", "bravo", "charlie"]

    def test_list_keys_empty_when_no_file(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        assert manager.list_keys() == []

    def test_exists_property(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        assert manager.exists is False
        manager.create_entry("x", content="y", prompt="z")
        assert manager.exists is True

    def test_delete_file(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        manager.create_entry("x", content="y", prompt="z")
        assert manager.filepath.exists()

        manager.delete_file()
        assert not manager.filepath.exists()

    def test_custom_filename(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path, filename="custom.json")
        manager.create_entry("k", content="v", prompt="p")
        assert (tmp_path / "custom.json").exists()

    def test_load_or_create_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        manager = SnapshotManager(snapshot_dir=tmp_path)
        snapshot_file = manager.load_or_create()
        assert len(snapshot_file.entries) == 0
        assert snapshot_file.schema_version == SNAPSHOT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# SnapshotDiff
# ---------------------------------------------------------------------------

class TestSnapshotDiff:

    def test_identical_content_no_changes(self) -> None:
        content = '{"title": "Test", "score": 0.9}'
        diff = SnapshotDiff.compare("key1", content, content)

        assert diff.structural_match is True
        assert diff.content_changed is False
        assert diff.length_delta == 0
        assert diff.keys_added == []
        assert diff.keys_removed == []

    def test_content_change_detected(self) -> None:
        baseline = "The cat sat on the mat."
        current = "The dog sat on the rug."
        diff = SnapshotDiff.compare("key2", baseline, current)

        assert diff.content_changed is True
        assert len(diff.content_diff_lines) > 0

    def test_structural_key_addition(self) -> None:
        baseline = '{"title": "Test"}'
        current = '{"title": "Test", "extra": true}'
        diff = SnapshotDiff.compare("key3", baseline, current)

        assert diff.structural_match is False
        assert "extra" in diff.keys_added
        assert diff.keys_removed == []

    def test_structural_key_removal(self) -> None:
        baseline = '{"title": "Test", "score": 0.9}'
        current = '{"title": "Test"}'
        diff = SnapshotDiff.compare("key4", baseline, current)

        assert diff.structural_match is False
        assert "score" in diff.keys_removed
        assert diff.keys_added == []

    def test_structural_match_with_value_changes(self) -> None:
        """Same keys, different values: structural match holds."""
        baseline = '{"title": "Old Title", "count": 5}'
        current = '{"title": "New Title", "count": 10}'
        diff = SnapshotDiff.compare("key5", baseline, current)

        assert diff.structural_match is True
        assert diff.content_changed is True

    def test_both_non_json_structural_match(self) -> None:
        """Two plain-text responses: structural comparison not applicable, treated as match."""
        diff = SnapshotDiff.compare("key6", "plain text baseline", "plain text current")
        assert diff.structural_match is True

    def test_json_vs_non_json_structural_mismatch(self) -> None:
        """One JSON, one plain text: structural mismatch."""
        diff = SnapshotDiff.compare("key7", '{"key": "value"}', "not json")
        assert diff.structural_match is False

    def test_model_change_detected(self) -> None:
        diff = SnapshotDiff.compare(
            "key8", "content", "content",
            baseline_model="gpt-4o-2024-08-06",
            current_model="gpt-4o-2024-11-20",
        )
        assert diff.model_changed is True
        assert diff.baseline_model == "gpt-4o-2024-08-06"
        assert diff.current_model == "gpt-4o-2024-11-20"

    def test_model_unchanged(self) -> None:
        diff = SnapshotDiff.compare(
            "key9", "content", "content",
            baseline_model="gpt-4o",
            current_model="gpt-4o",
        )
        assert diff.model_changed is False

    def test_length_delta(self) -> None:
        diff = SnapshotDiff.compare("key10", "short", "much longer content here")
        assert diff.length_delta > 0
        assert diff.baseline_length == 5
        assert diff.current_length == 24

    def test_pre_extracted_baseline_keys(self) -> None:
        """Passing baseline_json_keys avoids re-parsing the baseline content."""
        diff = SnapshotDiff.compare(
            "key11",
            baseline_content="not even json",
            current_content='{"alpha": 1, "beta": 2}',
            baseline_json_keys=["alpha", "beta"],
        )
        assert diff.structural_match is True

    def test_semantic_diff_computed_when_requested(self) -> None:
        """Semantic comparison requires compute_semantic=True."""
        baseline = "The effects of climate change on coastal areas are significant."
        current = "Climate change significantly impacts coastal regions."
        diff = SnapshotDiff.compare(
            "key12", baseline, current, compute_semantic=True,
        )
        assert diff.semantic_similarity is not None
        assert diff.semantic_similarity > 0.5
        assert diff.semantic_drift is not None
        assert diff.semantic_drift < 0.5

    def test_semantic_diff_not_computed_by_default(self) -> None:
        diff = SnapshotDiff.compare("key13", "baseline", "current")
        assert diff.semantic_similarity is None
        assert diff.semantic_drift is None

    def test_summary_no_changes(self) -> None:
        diff = SnapshotDiff.compare("key14", "same", "same")
        summary = diff.summary()
        assert "no changes" in summary

    def test_summary_with_changes(self) -> None:
        diff = SnapshotDiff.compare("key15", '{"a": 1}', '{"b": 2}')
        summary = diff.summary()
        assert "structure changed" in summary

    def test_summary_with_model_change(self) -> None:
        diff = SnapshotDiff.compare(
            "key16", "same", "same",
            baseline_model="v1",
            current_model="v2",
        )
        assert "model changed" in diff.summary()
