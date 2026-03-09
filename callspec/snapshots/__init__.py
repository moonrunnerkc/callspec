"""Snapshot system: baseline recording, comparison, and diff for regression assertions."""

from callspec.snapshots.diff import DiffResult, SnapshotDiff
from callspec.snapshots.manager import SnapshotManager
from callspec.snapshots.serializer import (
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
