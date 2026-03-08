"""SnapshotManager: create, load, update, delete baselines.

Manages the lifecycle of snapshot files stored in a project's
verdict_snapshots/ directory. Each snapshot file contains one or more
entries keyed by snapshot_key. The manager operates on SnapshotFile
and SnapshotEntry objects through the SnapshotSerializer; it never
touches raw JSON directly.

The default snapshot directory (verdict_snapshots/) is placed at the
project root, alongside pyproject.toml. This mirrors the convention
used by pytest-snapshot and syrupy, so teams already familiar with
snapshot testing recognize the pattern immediately.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from verdict.errors import SnapshotError
from verdict.snapshots.serializer import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotEntry,
    SnapshotFile,
    SnapshotSerializer,
)

logger = logging.getLogger(__name__)

# Default directory name, relative to project root.
DEFAULT_SNAPSHOT_DIR = "verdict_snapshots"
DEFAULT_SNAPSHOT_FILENAME = "baselines.json"


class SnapshotManager:
    """Manages the full snapshot lifecycle: create, load, update, delete.

    Initialized with a directory path. All operations target a single
    snapshot file within that directory. Multiple snapshot files can
    coexist (one per suite or test module) by using different filenames.
    """

    def __init__(
        self,
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        filename: str = DEFAULT_SNAPSHOT_FILENAME,
    ) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._filename = filename
        self._filepath = self._snapshot_dir / self._filename

    @property
    def filepath(self) -> Path:
        return self._filepath

    @property
    def exists(self) -> bool:
        return self._filepath.exists()

    def load(self) -> SnapshotFile:
        """Load the snapshot file from disk.

        Raises SnapshotError if the file does not exist, is corrupt,
        or uses an incompatible schema version.
        """
        return SnapshotSerializer.load(self._filepath)

    def load_or_create(self) -> SnapshotFile:
        """Load the snapshot file if it exists, otherwise create an empty one."""
        if self.exists:
            return self.load()
        return SnapshotFile()

    def save(self, snapshot_file: SnapshotFile) -> None:
        """Persist the snapshot file to disk, updating the timestamp."""
        snapshot_file.updated_at = datetime.now(timezone.utc).isoformat()
        SnapshotSerializer.save(self._filepath, snapshot_file)

    def get_entry(self, snapshot_key: str) -> SnapshotEntry:
        """Retrieve a single entry by key.

        Raises SnapshotError if the key does not exist, with guidance
        on how to create it.
        """
        snapshot_file = self.load()
        entry_data = snapshot_file.entries.get(snapshot_key)

        if entry_data is None:
            available_keys = sorted(snapshot_file.entries.keys())
            available_str = ", ".join(available_keys) if available_keys else "(none)"
            raise SnapshotError(
                snapshot_key,
                f"No baseline entry found for key '{snapshot_key}'. "
                f"Available keys: {available_str}. "
                f"Run 'verdict snapshot create' to record a baseline for this key.",
            )

        return SnapshotSerializer.deserialize_entry(entry_data)

    def create_entry(
        self,
        snapshot_key: str,
        content: str,
        prompt: str,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> SnapshotEntry:
        """Create a new snapshot entry and persist it.

        If overwrite=False and the key already exists, raises SnapshotError.
        This prevents accidental baseline overwrites in CI.
        """
        snapshot_file = self.load_or_create()

        if snapshot_key in snapshot_file.entries and not overwrite:
            raise SnapshotError(
                snapshot_key,
                f"Baseline already exists for key '{snapshot_key}'. "
                f"Use overwrite=True or 'verdict snapshot update' to replace it.",
            )

        entry = SnapshotEntry(
            snapshot_key=snapshot_key,
            content=content,
            prompt=prompt,
            model=model,
            provider=provider,
            metadata=metadata or {},
        )

        snapshot_file.entries[snapshot_key] = SnapshotSerializer.serialize_entry(entry)
        self.save(snapshot_file)

        logger.info("Created snapshot entry: %s (model=%s, %d chars)", snapshot_key, model, len(content))
        return entry

    def update_entry(
        self,
        snapshot_key: str,
        content: str,
        prompt: str,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> SnapshotEntry:
        """Update an existing entry or create it if it does not exist."""
        return self.create_entry(
            snapshot_key=snapshot_key,
            content=content,
            prompt=prompt,
            model=model,
            provider=provider,
            metadata=metadata,
            overwrite=True,
        )

    def delete_entry(self, snapshot_key: str) -> None:
        """Remove a single entry by key.

        Raises SnapshotError if the key does not exist.
        """
        snapshot_file = self.load()

        if snapshot_key not in snapshot_file.entries:
            raise SnapshotError(
                snapshot_key,
                f"Cannot delete: no baseline entry found for key '{snapshot_key}'.",
            )

        del snapshot_file.entries[snapshot_key]
        self.save(snapshot_file)
        logger.info("Deleted snapshot entry: %s", snapshot_key)

    def delete_all(self) -> int:
        """Remove all entries from the snapshot file. Returns the count deleted."""
        snapshot_file = self.load()
        count = len(snapshot_file.entries)
        snapshot_file.entries.clear()
        self.save(snapshot_file)
        logger.info("Deleted all %d snapshot entries", count)
        return count

    def list_keys(self) -> List[str]:
        """Return sorted list of all snapshot keys in the file."""
        if not self.exists:
            return []
        snapshot_file = self.load()
        return sorted(snapshot_file.entries.keys())

    def delete_file(self) -> None:
        """Remove the snapshot file itself from disk."""
        if self._filepath.exists():
            self._filepath.unlink()
            logger.info("Deleted snapshot file: %s", self._filepath)
