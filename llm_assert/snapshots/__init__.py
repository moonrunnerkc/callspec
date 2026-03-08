"""Snapshot system: baseline recording, comparison, and diff for regression assertions."""

from llm_assert.snapshots.diff import DiffResult, SnapshotDiff
from llm_assert.snapshots.manager import SnapshotManager
from llm_assert.snapshots.serializer import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotEntry,
    SnapshotFile,
    SnapshotSerializer,
)

__all__ = [
    "DiffResult",
    "SnapshotDiff",
    "SnapshotEntry",
    "SnapshotFile",
    "SnapshotManager",
    "SnapshotSerializer",
    "SNAPSHOT_SCHEMA_VERSION",
]
