"""SnapshotSerializer: JSON serialization with schema versioning.

Snapshots are the baseline format for regression assertions. The schema
version is embedded in every snapshot file so Verdict can detect and
refuse to compare incompatible formats rather than producing silent
wrong results. Changing this schema post-release requires migration
tooling, so the format is intentionally minimal and stable.

The serialization format is plain JSON (not YAML, not binary) because
JSON diffs well in git, parses in every language, and does not require
additional dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from verdict.errors import SnapshotError

logger = logging.getLogger(__name__)

# Increment this when the snapshot schema changes in a breaking way.
# Non-breaking additions (new optional fields) do not require a bump.
SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class SnapshotEntry:
    """A single recorded response within a snapshot.

    Each entry captures everything needed to compare a future response
    against this baseline: the content, the prompt that produced it,
    the model that generated it, and structural metadata for format
    comparison.
    """

    snapshot_key: str
    content: str
    prompt: str
    model: str = "unknown"
    provider: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_length: int = 0
    json_keys: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content_length = len(self.content)

        # Extract top-level JSON keys when the content is valid JSON.
        # Used by format_matches_baseline for structural comparison.
        if self.json_keys is None:
            try:
                parsed = json.loads(self.content)
                if isinstance(parsed, dict):
                    self.json_keys = sorted(parsed.keys())
            except (json.JSONDecodeError, TypeError):
                self.json_keys = None


@dataclass
class SnapshotFile:
    """Top-level structure of a serialized snapshot file.

    Contains the schema version, creation metadata, and a dict of
    snapshot entries keyed by snapshot_key. The schema version is
    checked on load to prevent silent incompatibility.
    """

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SnapshotSerializer:
    """Handles reading and writing snapshot files to disk.

    All I/O goes through this class. The rest of the snapshot system
    operates on SnapshotEntry and SnapshotFile objects, never raw
    file paths or JSON strings.
    """

    @staticmethod
    def serialize_entry(entry: SnapshotEntry) -> Dict[str, Any]:
        """Convert a SnapshotEntry to a JSON-serializable dict."""
        return asdict(entry)

    @staticmethod
    def deserialize_entry(data: Dict[str, Any]) -> SnapshotEntry:
        """Reconstruct a SnapshotEntry from a dict.

        Tolerates missing optional fields for forward compatibility
        with older snapshot files.
        """
        return SnapshotEntry(
            snapshot_key=data["snapshot_key"],
            content=data["content"],
            prompt=data["prompt"],
            model=data.get("model", "unknown"),
            provider=data.get("provider", "unknown"),
            timestamp=data.get("timestamp", ""),
            content_length=data.get("content_length", len(data["content"])),
            json_keys=data.get("json_keys"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def save(filepath: Path, snapshot_file: SnapshotFile) -> None:
        """Write a SnapshotFile to disk as formatted JSON.

        Creates parent directories if they do not exist. Uses 2-space
        indent for readable git diffs.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": snapshot_file.schema_version,
            "created_at": snapshot_file.created_at,
            "updated_at": snapshot_file.updated_at,
            "entries": snapshot_file.entries,
        }

        serialized = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False)
        filepath.write_text(serialized, encoding="utf-8")
        logger.info("Snapshot saved: %s (%d entries)", filepath, len(snapshot_file.entries))

    @staticmethod
    def load(filepath: Path) -> SnapshotFile:
        """Load a SnapshotFile from disk.

        Validates schema version compatibility. Raises SnapshotError
        with clear context if the file is missing, corrupt, or uses
        an incompatible schema version.
        """
        if not filepath.exists():
            raise SnapshotError(
                str(filepath),
                f"Snapshot file not found at {filepath}. "
                f"Run 'verdict snapshot create' or 'pytest --verdict-snapshot' to create a baseline.",
            )

        try:
            raw_text = filepath.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except json.JSONDecodeError as json_error:
            raise SnapshotError(
                str(filepath),
                f"Snapshot file is not valid JSON: {json_error}. "
                f"The file may be corrupt or was edited manually with syntax errors.",
            ) from json_error

        file_version = payload.get("schema_version")
        if file_version is None:
            raise SnapshotError(
                str(filepath),
                "Snapshot file is missing 'schema_version'. "
                "This file may predate the current Verdict version and needs to be regenerated.",
            )

        if file_version > SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError(
                str(filepath),
                f"Snapshot schema version {file_version} is newer than supported version "
                f"{SNAPSHOT_SCHEMA_VERSION}. Upgrade Verdict or regenerate the snapshot.",
            )

        return SnapshotFile(
            schema_version=file_version,
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            entries=payload.get("entries", {}),
        )
