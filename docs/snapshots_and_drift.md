# Snapshots and Drift Detection

Snapshots record a baseline of your agent's tool-call behavior. Regression assertions compare future runs against the baseline and fail when the behavior drifts. This catches silent changes from model updates, prompt edits, or provider swaps.

## Workflow

1. Record a baseline: run your agent, capture the trajectory, save it as a snapshot.
2. Commit the snapshot file to version control.
3. On future runs, assert that the trajectory still matches the baseline.
4. When the behavior changes intentionally, update the snapshot.

## Creating a Snapshot

### From the CLI

```bash
callspec snapshot create --key "booking_flow" --file snapshots/booking.json
```

### From Python

```python
from callspec.snapshots.manager import SnapshotManager

manager = SnapshotManager(snapshot_dir="snapshots")
manager.create_entry(
    key="booking_flow",
    content="Booked flight",
    tool_calls=[
        {"name": "search", "arguments": {"query": "SFO to JFK"}},
        {"name": "book", "arguments": {"flight_id": "UA123"}},
    ],
    model="gpt-4o-2024-11-20",
    provider="openai",
)
```

The snapshot is a versioned JSON file. The schema version is tracked so older snapshots auto-migrate on load.

## Regression Assertions

### `matches_baseline(snapshot_key, snapshot_manager)`

Passes if the trajectory matches the recorded baseline: same tool sequence and same argument keys (argument values may differ).

```python
from callspec.snapshots.manager import SnapshotManager

manager = SnapshotManager(snapshot_dir="snapshots")

result = (
    v.assert_trajectory(trajectory)
    .matches_baseline("booking_flow", manager)
    .run()
)
assert result.passed
```

When this fails, the message shows the trajectory diff: which tools were added, removed, or reordered, and which argument keys changed.

### `sequence_matches_baseline(snapshot_key, snapshot_manager)`

Passes if the tool name sequence matches the baseline, ignoring argument changes entirely. Use this when you expect argument values to vary across runs but the tool sequence should be stable.

```python
result = (
    v.assert_trajectory(trajectory)
    .sequence_matches_baseline("booking_flow", manager)
    .run()
)
```

## Trajectory Diff

The diff system compares two trajectories across three dimensions:

1. **Sequence diff**: tools added, removed, or reordered.
2. **Argument diff**: keys added or removed per tool call.
3. **Hash diff**: a SHA256 fingerprint of tool names and sorted argument keys. Different hash means the contract shape changed.

```python
from callspec.snapshots.diff import SnapshotDiff

diff_result = SnapshotDiff.compare_trajectories(baseline_calls, current_calls)

print(f"Sequence changed: {diff_result.sequence_changed}")
print(f"Arguments changed: {diff_result.arguments_changed}")
print(f"Hash match: {diff_result.hash_match}")

for tool_diff in diff_result.call_diffs:
    print(f"  {tool_diff.tool_name}: keys_added={tool_diff.keys_added}, keys_removed={tool_diff.keys_removed}")
```

## Updating Snapshots

### From the CLI

```bash
callspec snapshot update --key "booking_flow"
```

### From Python

```python
manager.update_entry(
    key="booking_flow",
    content="Updated response",
    tool_calls=[...],
    model="gpt-4o-2025-01-15",
    provider="openai",
)
```

After updating, commit the snapshot file. The git diff shows exactly what changed in the baseline.

## Snapshot Storage

Snapshots are JSON files with a versioned schema. Each entry contains:

- The response content
- The tool calls (names and arguments)
- A trajectory hash (SHA256 of tool names + sorted argument keys)
- Model and provider metadata
- Timestamp

The schema version ensures forward compatibility. Snapshot files created with an older version are automatically migrated on load.
