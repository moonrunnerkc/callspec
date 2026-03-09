# Contract Assertions

Contract assertions verify the arguments passed to tool calls. They complement trajectory assertions: trajectory assertions check which tools were called, contract assertions check that the arguments are correct.

All contract assertions apply to every call to the named tool within the trajectory. If the tool was called three times, the assertion must pass for all three calls.

## Assertion Reference

### `argument_contains_key(tool_name, key)`

Passes if every call to the tool includes the named key in its arguments.

```python
result = (
    v.assert_trajectory(trajectory)
    .argument_contains_key("search", "query")
    .run()
)
```

Failure message: `ArgumentContainsKey failed: call to 'search' at index 0 missing required key 'query'. Keys present: ['limit', 'offset'].`

### `argument_not_empty(tool_name, key)`

Passes if the value for the key is not null, not an empty string, not an empty list, and not an empty dict. The key must also be present.

```python
result = (
    v.assert_trajectory(trajectory)
    .argument_not_empty("search", "query")
    .run()
)
```

### `argument_value_in(tool_name, key, allowed_values)`

Passes if the value for the key is one of the allowed values.

```python
result = (
    v.assert_trajectory(trajectory)
    .argument_value_in("book", "class", ["economy", "business", "first"])
    .run()
)
```

Failure message: `ArgumentValueIn failed: call to 'book' at index 1, key 'class' has value 'premium' which is not in allowed values ['economy', 'business', 'first'].`

### `argument_matches_pattern(tool_name, key, pattern)`

Passes if the string value for the key matches the given regex pattern. The key must be present in the arguments.

```python
result = (
    v.assert_trajectory(trajectory)
    .argument_matches_pattern("search", "date", r"^\d{4}-\d{2}-\d{2}$")
    .run()
)
```

### `argument_matches_schema(tool_name, schema)`

Passes if every call to the tool has arguments that validate against the given JSON Schema.

```python
schema = {
    "type": "object",
    "required": ["origin", "destination"],
    "properties": {
        "origin": {"type": "string", "minLength": 3},
        "destination": {"type": "string", "minLength": 3},
    },
}

result = (
    v.assert_trajectory(trajectory)
    .argument_matches_schema("search_flights", schema)
    .run()
)
```

Failure message includes the JSON Schema validation errors for each failing call.

### `custom_contract(tool_name, predicate_fn, description)`

Passes if a user-supplied function returns `True` for every call to the tool. The function receives a `ToolCall` object.

```python
def no_wildcard_search(call):
    return call.arguments.get("query", "") != "*"

result = (
    v.assert_trajectory(trajectory)
    .custom_contract("search", no_wildcard_search, "query must not be wildcard")
    .run()
)
```

## Combining with Trajectory Assertions

Contract assertions chain naturally with trajectory assertions:

```python
result = (
    v.assert_trajectory(trajectory)
    .calls_tools_in_order(["search", "book"])
    .does_not_call("cancel")
    .argument_not_empty("search", "query")
    .argument_value_in("book", "class", ["economy", "business"])
    .argument_matches_pattern("search", "date", r"^\d{4}-\d{2}-\d{2}$")
    .run()
)
```

## Skipping Tool Not Present

If a contract assertion names a tool that does not appear in the trajectory, the behavior depends on context: if the trajectory contains no calls to that tool, the assertion is skipped (passes vacuously). This allows defining broad contracts that apply only when a tool is actually used.
