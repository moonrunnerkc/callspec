# Getting Started

Install callspec and run your first tool-call contract test in under five minutes.

## Install

```bash
pip install callspec
```

Add a provider extra if you want to call a real LLM:

```bash
pip install "callspec[openai]"
pip install "callspec[anthropic]"
```

No account required. No API keys needed for the assertion library itself.

## First Test

Create a file called `test_agent.py`:

```python
from callspec import Callspec, ToolCallTrajectory
from callspec.providers.mock import MockProvider

provider = MockProvider(
    response_fn=lambda p, m: "Done",
    tool_calls=[
        {"name": "search", "arguments": {"query": "flights SFO to JFK"}},
        {"name": "book", "arguments": {"flight_id": "UA123", "seat": "12A"}},
    ],
)

v = Callspec(provider)
response = provider.call("Book a flight from SFO to JFK")
trajectory = ToolCallTrajectory.from_provider_response(response)

result = (
    v.assert_trajectory(trajectory)
    .calls_tools_in_order(["search", "book"])
    .does_not_call("cancel")
    .argument_not_empty("search", "query")
    .run()
)
assert result.passed
```

Run it:

```bash
python test_agent.py
```

Or wrap it in a test function for pytest:

```python
def test_booking_trajectory():
    # ... same code as above ...
    assert result.passed
```

```bash
pytest test_agent.py -v
```

## What Just Happened

1. `MockProvider` simulated an LLM that returns two tool calls.
2. `ToolCallTrajectory.from_provider_response()` extracted those calls into a normalized trajectory.
3. Three assertions ran against the trajectory: tool ordering, a negative check, and an argument presence check.
4. All three passed. `result.passed` is `True`.

## First Failure

Change `calls_tools_in_order(["search", "book"])` to `calls_tools_in_order(["book", "search"])` and run again. The failure message tells you exactly what went wrong:

```
CallsToolsInOrder failed: expected order ['book', 'search'], actual order ['search', 'book'].
Tool 'book' expected at position 0, first seen at position 1.
```

Every failure message includes the assertion name, expected value, actual value, and which tool call triggered it.

## Using a Real Provider

Swap MockProvider for any real provider. The assertions stay the same:

```python
import os
from callspec import Callspec, ToolCallTrajectory
from callspec.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
)

response = provider.call(
    "Book me a flight from SFO to JFK",
    tools=[{
        "type": "function",
        "function": {
            "name": "search_flights",
            "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "dest": {"type": "string"}}},
        },
    }],
)
trajectory = ToolCallTrajectory.from_provider_response(response)

result = (
    Callspec(provider)
    .assert_trajectory(trajectory)
    .calls_tool("search_flights")
    .argument_not_empty("search_flights", "origin")
    .run()
)
assert result.passed
```

Supported providers: OpenAI, Anthropic, Google, Mistral, Ollama, LiteLLM, MockProvider.

## Next

- [Trajectory Assertions](trajectory_assertions.md) for the full assertion reference.
- [Contract Assertions](contract_assertions.md) for argument validation.
- [Snapshots and Drift](snapshots_and_drift.md) for regression testing across model versions.
- [pytest and CI](pytest_and_ci.md) for CI pipeline integration.
