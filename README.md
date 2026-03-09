# callspec

Contract testing for LLM tool calls.

```bash
pip install callspec
```

```python
from callspec import Callspec, ToolCall, ToolCallTrajectory
from callspec.providers.mock import MockProvider

provider = MockProvider(
    response_fn=lambda p, m: "Booked flight",
    tool_calls=[
        {"name": "search_flights", "arguments": {"origin": "SFO", "dest": "JFK"}},
        {"name": "book_flight", "arguments": {"flight_id": "UA123"}},
    ],
)

v = Callspec(provider)
response = provider.call("Book me a flight from SFO to JFK")
trajectory = ToolCallTrajectory.from_provider_response(response)

result = (
    v.assert_trajectory(trajectory)
    .calls_tools_in_order(["search_flights", "book_flight"])
    .does_not_call("cancel_flight")
    .argument_not_empty("search_flights", "origin")
    .run()
)
assert result.passed
```

Your agent calls tools. Those calls are the contract between your code and the model. When you swap models, update a prompt, or change your retrieval pipeline, callspec tells you whether the agent still calls the right tools, in the right order, with the right arguments. No LLM-as-judge. No API calls for evaluation. Deterministic pass/fail that runs in CI.

## Docs

- [Getting Started](docs/getting_started.md)
- [Trajectory Assertions](docs/trajectory_assertions.md)
- [Contract Assertions](docs/contract_assertions.md)
- [Snapshots and Drift](docs/snapshots_and_drift.md)
- [pytest and CI](docs/pytest_and_ci.md)

## License

Apache 2.0
