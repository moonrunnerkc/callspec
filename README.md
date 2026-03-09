# callspec

[![PyPI version](https://img.shields.io/pypi/v/callspec)](https://pypi.org/project/callspec/)
[![Tests](https://github.com/moonrunnerkc/callspec/actions/workflows/test.yml/badge.svg)](https://github.com/moonrunnerkc/callspec/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/callspec)](https://pypi.org/project/callspec/)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

Contract testing for LLM tool calls.

```bash
pip install callspec
```

```python
from callspec import Callspec, ToolCallTrajectory
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

## Why callspec exists

Your agent calls tools. Those calls are the contract between your code and the model. When you swap models, update a prompt, or change your retrieval pipeline, the tool-call behavior can silently change: different tools get called, arguments go missing, the call order shifts. None of this throws an exception. Your code still runs. It just does the wrong thing.

You could write raw pytest assertions:

```python
assert response.tool_calls[0].function.name == "search_flights"
assert response.tool_calls[1].function.name == "book_flight"
assert response.tool_calls[0].function.arguments.get("origin")
```

That works until you need ordering semantics across five tool calls, negative assertions ("never calls `delete_account`"), argument validation across providers that return different response shapes, or drift detection against a recorded baseline. Then you are writing and maintaining a test harness. callspec is that harness.

**What you get over raw assertions:** Fluent chainable assertions for tool ordering, presence, absence, and argument shapes. Snapshot baselines that catch silent drift when you swap models or edit prompts, with diffs showing exactly what changed. Provider adapters that normalize responses across OpenAI, Anthropic, Google, Mistral, Ollama, and LiteLLM. A pytest plugin with fixtures, markers, and structured failure output. No LLM-as-judge. No API calls for evaluation. Deterministic pass/fail.

## Snapshot drift detection

The core value of callspec beyond basic assertions is catching behavioral drift across model versions.

```python
from callspec.snapshots.manager import SnapshotManager

manager = SnapshotManager(snapshot_dir="snapshots")

# Record a baseline once
manager.create_entry(
    snapshot_key="booking_flow",
    content="Booked flight",
    prompt="Book me a flight from SFO to JFK",
    tool_calls=[
        {"tool_name": "search_flights", "arguments": {"origin": "SFO", "dest": "JFK"}},
        {"tool_name": "book_flight", "arguments": {"flight_id": "UA123"}},
    ],
    model="gpt-4o-2024-11-20",
    provider="openai",
)

# On every future run, assert the trajectory still matches
result = (
    v.assert_trajectory(trajectory)
    .matches_baseline("booking_flow", manager)
    .run()
)
```

When it fails, the diff tells you exactly what changed: tools added, removed, or reordered, and argument keys that appeared or disappeared. Commit snapshots to version control. The git diff is your audit trail.

## Providers

callspec ships adapters for every major LLM provider. Install the one you use:

```bash
pip install "callspec[openai]"       # OpenAI (GPT-4o, o1, etc.)
pip install "callspec[anthropic]"    # Anthropic (Claude)
pip install "callspec[google]"       # Google (Gemini)
pip install "callspec[mistral]"      # Mistral
pip install "callspec[ollama]"       # Ollama (local models)
pip install "callspec[litellm]"      # LiteLLM (any provider)
```

The core library has zero provider dependencies. `MockProvider` is always available for offline testing.

## GitHub Action

callspec ships a [composite GitHub Action](action/action.yml) for CI integration. Add it to your workflow:

```yaml
- uses: moonrunnerkc/callspec@main
  with:
    suite: tests/contracts/booking.yml
    provider: openai
    callspec-extras: openai
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action installs callspec, runs your contract suite, and annotates PRs with failure details using GitHub workflow commands. See [pytest and CI](docs/pytest_and_ci.md) for the full integration guide.

## Docs

- [Getting Started](docs/getting_started.md) -- install, first test, first failure in under 5 minutes
- [Trajectory Assertions](docs/trajectory_assertions.md) -- full assertion reference
- [Contract Assertions](docs/contract_assertions.md) -- argument validation
- [Snapshots and Drift](docs/snapshots_and_drift.md) -- regression testing across model versions
- [pytest and CI](docs/pytest_and_ci.md) -- fixtures, markers, CI pipeline integration
- [Case Study: The Refund Agent](case_study/README.md) -- a model swap silently drops fraud checks, callspec catches it

## Status

callspec is alpha (v0.1.0), backed by 580+ tests across Python 3.9-3.13. The trajectory assertion API and snapshot system are stable. The API surface may change before 1.0 based on real-world usage.

## Contributing

File bugs and feature requests on [GitHub Issues](https://github.com/moonrunnerkc/callspec/issues). Pull requests welcome. Run the test suite before submitting:

```bash
pip install -e ".[dev]"
pytest
```

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

## License

Apache 2.0
