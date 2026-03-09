"""Tests for contract assertions (per-tool argument validation)."""


from callspec.assertions.contract import (
    ArgumentContainsKey,
    ArgumentMatchesPattern,
    ArgumentMatchesSchema,
    ArgumentNotEmpty,
    ArgumentValueIn,
    CustomContract,
)
from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCall, ToolCallTrajectory

CONFIG = CallspecConfig()


def _traj_with_calls(*calls_spec: tuple[str, dict]) -> ToolCallTrajectory:
    """Build trajectory from (tool_name, arguments) pairs."""
    calls = [
        ToolCall(tool_name=name, arguments=args, call_index=i)
        for i, (name, args) in enumerate(calls_spec)
    ]
    return ToolCallTrajectory(calls=calls, model="test", provider="mock")


EMPTY = ToolCallTrajectory(calls=[], model="test", provider="mock")


# ── ArgumentMatchesSchema ──

class TestArgumentMatchesSchema:
    SEARCH_SCHEMA = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
    }

    def test_passes_valid_args(self):
        traj = _traj_with_calls(("search", {"query": "hello", "limit": 5}))
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed
        assert result.assertion_type == "contract"

    def test_fails_missing_required_key(self):
        traj = _traj_with_calls(("search", {"limit": 5}))
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed
        assert "query" in result.message

    def test_fails_wrong_type(self):
        traj = _traj_with_calls(("search", {"query": 123}))
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_multiple_calls_all_must_pass(self):
        traj = _traj_with_calls(
            ("search", {"query": "a"}),
            ("search", {"query": "b"}),
        )
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_multiple_calls_one_fails(self):
        traj = _traj_with_calls(
            ("search", {"query": "a"}),
            ("search", {"limit": 5}),  # missing required "query"
        )
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {"id": 1}))
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed
        assert "not found" in result.message.lower()

    def test_ignores_other_tools(self):
        traj = _traj_with_calls(
            ("search", {"query": "ok"}),
            ("book", {"bad_field": True}),
        )
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_empty_trajectory(self):
        assertion = ArgumentMatchesSchema("search", self.SEARCH_SCHEMA)
        result = assertion.evaluate_trajectory(EMPTY, CONFIG)
        assert not result.passed


# ── ArgumentContainsKey ──

class TestArgumentContainsKey:
    def test_passes_key_present(self):
        traj = _traj_with_calls(("search", {"query": "hello"}))
        assertion = ArgumentContainsKey("search", "query")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_key_absent(self):
        traj = _traj_with_calls(("search", {"limit": 5}))
        assertion = ArgumentContainsKey("search", "query")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_multiple_calls_all_must_have_key(self):
        traj = _traj_with_calls(
            ("search", {"query": "a"}),
            ("search", {"limit": 5}),  # no "query"
        )
        assertion = ArgumentContainsKey("search", "query")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {"id": 1}))
        assertion = ArgumentContainsKey("search", "query")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed


# ── ArgumentValueIn ──

class TestArgumentValueIn:
    def test_passes_value_in_set(self):
        traj = _traj_with_calls(("search", {"engine": "google"}))
        assertion = ArgumentValueIn("search", "engine", ["google", "bing"])
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_value_not_in_set(self):
        traj = _traj_with_calls(("search", {"engine": "yahoo"}))
        assertion = ArgumentValueIn("search", "engine", ["google", "bing"])
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed
        assert "yahoo" in result.message

    def test_fails_key_absent(self):
        traj = _traj_with_calls(("search", {"query": "test"}))
        assertion = ArgumentValueIn("search", "engine", ["google"])
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_multiple_calls(self):
        traj = _traj_with_calls(
            ("search", {"engine": "google"}),
            ("search", {"engine": "bing"}),
        )
        assertion = ArgumentValueIn("search", "engine", ["google", "bing"])
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {}))
        assertion = ArgumentValueIn("search", "engine", ["google"])
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed


# ── ArgumentMatchesPattern ──

class TestArgumentMatchesPattern:
    def test_passes_pattern_match(self):
        traj = _traj_with_calls(("search", {"query": "python 3.12 features"}))
        assertion = ArgumentMatchesPattern("search", "query", r"python \d")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_no_match(self):
        traj = _traj_with_calls(("search", {"query": "hello world"}))
        assertion = ArgumentMatchesPattern("search", "query", r"^python")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_fails_non_string_value(self):
        traj = _traj_with_calls(("search", {"query": 42}))
        assertion = ArgumentMatchesPattern("search", "query", r"\d+")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed
        assert "not a string" in result.message.lower()

    def test_fails_key_absent(self):
        traj = _traj_with_calls(("search", {"limit": 5}))
        assertion = ArgumentMatchesPattern("search", "query", r".*")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {}))
        assertion = ArgumentMatchesPattern("search", "query", r".*")
        result = assertion.evaluate_trajectory(traj, CONFIG)
        assert not result.passed


# ── ArgumentNotEmpty ──

class TestArgumentNotEmpty:
    def test_passes_non_empty_string(self):
        traj = _traj_with_calls(("search", {"query": "hello"}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_empty_string(self):
        traj = _traj_with_calls(("search", {"query": ""}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_fails_whitespace_only(self):
        traj = _traj_with_calls(("search", {"query": "   "}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_fails_none_value(self):
        traj = _traj_with_calls(("search", {"query": None}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_fails_empty_list(self):
        traj = _traj_with_calls(("search", {"tags": []}))
        result = ArgumentNotEmpty("search", "tags").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_passes_non_empty_list(self):
        traj = _traj_with_calls(("search", {"tags": ["python"]}))
        result = ArgumentNotEmpty("search", "tags").evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_empty_dict(self):
        traj = _traj_with_calls(("search", {"metadata": {}}))
        result = ArgumentNotEmpty("search", "metadata").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_passes_non_empty_dict(self):
        traj = _traj_with_calls(("search", {"metadata": {"k": "v"}}))
        result = ArgumentNotEmpty("search", "metadata").evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_passes_numeric_value(self):
        # Numeric zero is not "empty" in the _is_empty sense
        traj = _traj_with_calls(("search", {"limit": 0}))
        result = ArgumentNotEmpty("search", "limit").evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_key_absent(self):
        traj = _traj_with_calls(("search", {"limit": 5}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {}))
        result = ArgumentNotEmpty("search", "query").evaluate_trajectory(traj, CONFIG)
        assert not result.passed


# ── CustomContract ──

class TestCustomContract:
    def test_passes_true_predicate(self):
        traj = _traj_with_calls(("search", {"query": "hello"}))
        result = CustomContract(
            "search",
            lambda call: len(call.arguments.get("query", "")) > 0,
            "query must not be blank",
        ).evaluate_trajectory(traj, CONFIG)
        assert result.passed

    def test_fails_false_predicate(self):
        traj = _traj_with_calls(("search", {"query": ""}))
        result = CustomContract(
            "search",
            lambda call: len(call.arguments.get("query", "")) > 0,
            "query must not be blank",
        ).evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_description_in_message(self):
        traj = _traj_with_calls(("search", {"query": ""}))
        result = CustomContract(
            "search",
            lambda call: False,
            "always fails for testing",
        ).evaluate_trajectory(traj, CONFIG)
        assert "always fails for testing" in result.message

    def test_applies_to_all_calls(self):
        traj = _traj_with_calls(
            ("search", {"query": "a"}),
            ("search", {"query": ""}),  # fails predicate
        )
        result = CustomContract(
            "search",
            lambda call: bool(call.arguments.get("query")),
            "query present",
        ).evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_tool_not_found(self):
        traj = _traj_with_calls(("book", {}))
        result = CustomContract(
            "search",
            lambda call: True,
            "trivial",
        ).evaluate_trajectory(traj, CONFIG)
        assert not result.passed

    def test_predicate_receives_tool_call(self):
        """Verify the predicate receives an actual ToolCall with correct fields."""
        received_calls = []

        def capture_predicate(call):
            received_calls.append(call)
            return True

        traj = _traj_with_calls(("search", {"query": "test"}))
        CustomContract("search", capture_predicate, "capture").evaluate_trajectory(traj, CONFIG)

        assert len(received_calls) == 1
        assert received_calls[0].tool_name == "search"
        assert received_calls[0].arguments == {"query": "test"}
