# Trajectory Assertions

Trajectory assertions verify which tools an agent called and in what order. They operate on a `ToolCallTrajectory`, which is an ordered list of `ToolCall` records extracted from a single LLM response.

Every assertion is deterministic, requires no LLM calls, and runs in microseconds.

## Building a Trajectory

From a provider response:

```python
from callspec import ToolCallTrajectory

trajectory = ToolCallTrajectory.from_provider_response(response)
```

From raw data (testing without a provider):

```python
from callspec import ToolCall, ToolCallTrajectory

trajectory = ToolCallTrajectory(calls=[
    ToolCall(tool_name="search", arguments={"query": "hotels in Paris"}, call_index=0),
    ToolCall(tool_name="book", arguments={"hotel_id": "H456"}, call_index=1),
])
```

## Assertion Reference

All assertions are chained on a `TrajectoryBuilder` and evaluated on `.run()`.

### `calls_tool(tool_name)`

Passes if the trajectory contains at least one call to the named tool.

```python
result = v.assert_trajectory(trajectory).calls_tool("search").run()
```

Failure message: `CallsTool failed: tool 'search' not found in trajectory. Tools called: ['book', 'cancel'].`

### `calls_tools_in_order(expected_order)`

Passes if the named tools appear in the trajectory in the specified relative order. Other tools may appear between them.

```python
result = (
    v.assert_trajectory(trajectory)
    .calls_tools_in_order(["search", "book"])
    .run()
)
```

Failure message: `CallsToolsInOrder failed: expected order ['search', 'book'], actual order ['book', 'search']. Tool 'search' expected at position 0, first seen at position 1.`

### `calls_exactly(expected_tools)`

Passes if the trajectory calls exactly these tools in exactly this order. No extra tools allowed, no missing tools allowed.

```python
result = (
    v.assert_trajectory(trajectory)
    .calls_exactly(["search", "book", "confirm"])
    .run()
)
```

Failure message includes both the expected and actual sequences.

### `calls_subset(required_tools)`

Passes if every listed tool appears at least once. Order does not matter.

```python
result = (
    v.assert_trajectory(trajectory)
    .calls_subset(["search", "book"])
    .run()
)
```

### `does_not_call(tool_name)`

Passes if the trajectory never calls the named tool.

```python
result = v.assert_trajectory(trajectory).does_not_call("delete").run()
```

Failure message: `DoesNotCall failed: tool 'delete' was called 1 time(s) but should not have been called.`

### `call_count(tool_name, min_count, max_count)`

Passes if the tool is called between `min_count` and `max_count` times (inclusive). Omit `max_count` for no upper bound.

```python
# Called exactly once
result = (
    v.assert_trajectory(trajectory)
    .call_count("search", min_count=1, max_count=1)
    .run()
)

# Called at least twice
result = (
    v.assert_trajectory(trajectory)
    .call_count("retry", min_count=2)
    .run()
)
```

### `no_repeated_calls(tool_name)`

Passes if the tool is called at most once. Shortcut for `call_count(tool, 0, 1)`.

```python
result = (
    v.assert_trajectory(trajectory)
    .no_repeated_calls("book")
    .run()
)
```

## Chaining

Chain any combination of assertions. All are evaluated (unless `fail_fast=True` in config) and results are collected:

```python
result = (
    v.assert_trajectory(trajectory)
    .calls_tools_in_order(["search", "book"])
    .does_not_call("cancel")
    .does_not_call("delete")
    .no_repeated_calls("book")
    .run()
)

for assertion_result in result.assertions:
    print(f"{assertion_result.assertion_name}: {'PASS' if assertion_result.passed else 'FAIL'}")
```

## Result Object

`result.passed` is `True` only when every assertion passed. `result.assertions` is a list of `IndividualAssertionResult` objects, each with:

- `assertion_name`: e.g. `"CallsToolsInOrder"`
- `passed`: `True` or `False`
- `message`: human-readable outcome
- `details`: dict with assertion-specific data (expected, actual, tool names)
