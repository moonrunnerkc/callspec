"""Snapshot system: baseline recording, comparison, and diff for regression assertions."""

from verdict.snapshots.diff import DiffResult, SnapshotDiff
from verdict.snapshots.manager import SnapshotManager
from verdict.snapshots.serializer import (
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