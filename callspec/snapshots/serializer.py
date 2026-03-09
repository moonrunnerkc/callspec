"""SnapshotSerializer: JSON serialization with schema versioning.

Snapshots are the baseline format for regression assertions. The schema
version is embedded in every snapshot file so the tool can detect and
refuse to compare incompatible formats rather than producing silent
wrong results. Changing this schema post-release requires migration
tooling, so the format is intentionally minimal and stable.

The serialization format is plain JSON (not YAML, not binary) because
JSON diffs well in git, parses in every language, and does not require
additional dependencies.

Schema version history:
  1 - Original: text content baselines.
  2 - Added trajectory fields (tool_calls, trajectory_hash) for
      tool-call contract testing. Version 1 files are auto-migrated
      on load by adding empty trajectory fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from callspec.errors import SnapshotError

logger = logging.getLogger(__name__)

# Increment this when the snapshot schema changes in a breaking way.
# Non-breaking additions (new optional fields) do not require a bump.
SNAPSHOT_SCHEMA_VERSION = 2


@dataclass
class SnapshotEntry:
    """A single recorded response within a snapshot.

    Each entry captures everything needed to compare a future response
    against this baseline: the content, the prompt that produced it,
    the model that generated it, structural metadata for format
    comparison, and trajectory data for tool-call baselines.
    """

    snapshot_key: str
    content: str
    prompt: str
    model: str = "unknown"
    provider: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_length: int = 0
    json_keys: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Trajectory fields for tool-call baselines
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trajectory_hash: str = ""

    @property
    def has_trajectory(self) -> bool:
        """True if this entry has recorded tool-call data."""
        return len(self.tool_calls) > 0

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names from the trajectory."""
        return [tc.get("tool_name", "") for tc in self.tool_calls]

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

        # Compute trajectory hash if tool_calls are present but hash is empty.
        # The hash is a deterministic fingerprint of tool names + argument keys
        # for fast equality checks without deep comparison.
        if self.tool_calls and not self.trajectory_hash:
            self.trajectory_hash = compute_trajectory_hash(self.tool_calls)


def compute_trajectory_hash(tool_calls: list[dict[str, Any]]) -> str:
    """Deterministic hash of tool names and argument keys for fast comparison.

    Uses tool name sequence + sorted argument keys per call. The hash changes
    when tools are added, removed, reordered, or when argument keys change.
    Argument values are intentionally excluded so minor value changes do not
    trigger a hash mismatch.
    """
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        args = tc.get("arguments", {})
        arg_keys = sorted(args.keys()) if isinstance(args, dict) else []
        parts.append(f"{name}({','.join(arg_keys)})")

    fingerprint = "|".join(parts)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


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
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


class SnapshotSerializer:
    """Handles reading and writing snapshot files to disk.

    All I/O goes through this class. The rest of the snapshot system
    operates on SnapshotEntry and SnapshotFile objects, never raw
    file paths or JSON strings.
    """

    @staticmethod
    def serialize_entry(entry: SnapshotEntry) -> dict[str, Any]:
        """Convert a SnapshotEntry to a JSON-serializable dict."""
        return asdict(entry)

    @staticmethod
    def deserialize_entry(data: dict[str, Any]) -> SnapshotEntry:
        """Reconstruct a SnapshotEntry from a dict.

        Tolerates missing optional fields for forward compatibility
        with older snapshot files. V1 entries (no trajectory fields)
        are silently migrated by using empty defaults.
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
            tool_calls=data.get("tool_calls", []),
            trajectory_hash=data.get("trajectory_hash", ""),
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
                f"Run 'callspec snapshot create' or "
                f"'pytest --callspec-snapshot' to create a baseline.",
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
                "This file may predate the current Callspec version and needs to be regenerated.",
            )

        if file_version > SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError(
                str(filepath),
                f"Snapshot schema version {file_version} is newer than supported version "
                f"{SNAPSHOT_SCHEMA_VERSION}. Upgrade Callspec or regenerate the snapshot.",
            )

        # Auto-migrate older versions to current. V1 -> V2 adds trajectory
        # fields with empty defaults, handled transparently by deserialize_entry.
        if file_version < SNAPSHOT_SCHEMA_VERSION:
            logger.info(
                "Migrating snapshot %s from schema v%d to v%d",
                filepath, file_version, SNAPSHOT_SCHEMA_VERSION,
            )

        return SnapshotFile(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            entries=payload.get("entries", {}),
        )
