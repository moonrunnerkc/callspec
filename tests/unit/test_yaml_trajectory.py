"""Tests for YAML suite parsing with trajectory and contract assertions.

Verifies that the YAML parser correctly builds trajectory assertions,
contract assertions, and mixed cases. Also verifies the runner handles
trajectory cases in suites.
"""

from __future__ import annotations

import textwrap

import pytest

from llm_assert.core.yaml_suite import load_yaml_suite
from llm_assert.errors import SuiteParseError


def _write_suite(tmp_path, yaml_content: str):
    """Write YAML content to a suite file and return the path."""
    filepath = tmp_path / "suite.yml"
    filepath.write_text(textwrap.dedent(yaml_content))
    return filepath


class TestTrajectoryYAMLParsing:
    def test_calls_tools_in_order(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            name: "agent_contracts"
            cases:
              - name: "search_then_book"
                prompt: "Book a flight"
                trajectory:
                  - calls_tools_in_order: ["search_flights", "book_flight"]
        """)
        suite = load_yaml_suite(path)
        assert len(suite.cases) == 1
        case = suite.cases[0]
        assert case.has_trajectory_assertions
        assert not case.has_content_assertions
        assert len(case.trajectory_assertions) == 1

    def test_does_not_call(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "safe_agent"
                prompt: "Help me"
                trajectory:
                  - does_not_call: "delete_all"
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "DoesNotCall"

    def test_calls_tool(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "uses_search"
                prompt: "Find flights"
                trajectory:
                  - calls_tool: "search_flights"
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "CallsTool"

    def test_calls_exactly(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "exact_flow"
                prompt: "Book it"
                trajectory:
                  - calls_exactly: ["search", "select", "book"]
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "CallsExactly"

    def test_calls_subset(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "required_tools"
                prompt: "Do stuff"
                trajectory:
                  - calls_subset: ["search", "book"]
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "CallsSubset"

    def test_no_repeated_calls(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "no_dupes"
                prompt: "Do once"
                trajectory:
                  - no_repeated_calls: "charge_card"
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "NoRepeatedCalls"

    def test_call_count(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "count_check"
                prompt: "Retry search"
                trajectory:
                  - call_count:
                      tool_name: "search"
                      min_count: 1
                      max_count: 3
        """)
        suite = load_yaml_suite(path)
        assertion = suite.cases[0].trajectory_assertions[0]
        assert assertion.__class__.__name__ == "CallCount"

    def test_multiple_trajectory_assertions(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "complex_flow"
                prompt: "Complete booking"
                trajectory:
                  - calls_tools_in_order: ["search", "select", "book"]
                  - does_not_call: "cancel"
                  - no_repeated_calls: "charge_card"
        """)
        suite = load_yaml_suite(path)
        assert len(suite.cases[0].trajectory_assertions) == 3

    def test_unknown_trajectory_type_raises(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "bad"
                prompt: "test"
                trajectory:
                  - nonexistent_check: "foo"
        """)
        with pytest.raises(SuiteParseError, match="Unknown trajectory assertion"):
            load_yaml_suite(path)


class TestContractYAMLParsing:
    def test_not_empty_contract(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "origin_required"
                prompt: "Book flight"
                contracts:
                  search_flights:
                    - key: "origin"
                      not_empty: true
        """)
        suite = load_yaml_suite(path)
        case = suite.cases[0]
        assert case.has_trajectory_assertions
        assert len(case.trajectory_assertions) == 1
        assert case.trajectory_assertions[0].__class__.__name__ == "ArgumentNotEmpty"

    def test_matches_pattern_contract(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "iata_codes"
                prompt: "Book flight"
                contracts:
                  search_flights:
                    - key: "origin"
                      matches_pattern: "^[A-Z]{3}$"
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "ArgumentMatchesPattern"

    def test_value_in_contract(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "class_check"
                prompt: "Book flight"
                contracts:
                  book_flight:
                    - key: "class"
                      value_in: ["economy", "business", "first"]
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "ArgumentValueIn"

    def test_schema_contract(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "schema_check"
                prompt: "Book flight"
                contracts:
                  search_flights:
                    - schema:
                        type: "object"
                        required: ["origin", "destination"]
        """)
        suite = load_yaml_suite(path)
        assert suite.cases[0].trajectory_assertions[0].__class__.__name__ == "ArgumentMatchesSchema"

    def test_multiple_tools_multiple_contracts(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "full_contracts"
                prompt: "Book flight"
                contracts:
                  search_flights:
                    - key: "origin"
                      not_empty: true
                    - key: "destination"
                      not_empty: true
                  book_flight:
                    - key: "passenger_name"
                      not_empty: true
        """)
        suite = load_yaml_suite(path)
        # 2 for search_flights + 1 for book_flight
        assert len(suite.cases[0].trajectory_assertions) == 3

    def test_unrecognized_contract_raises(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "bad_contract"
                prompt: "test"
                contracts:
                  search:
                    - bogus_field: true
        """)
        with pytest.raises(SuiteParseError, match="no recognizable constraints"):
            load_yaml_suite(path)


class TestMixedCases:
    def test_trajectory_and_contracts_combined(self, tmp_path):
        """A case with both trajectory and contracts sections."""
        path = _write_suite(tmp_path, """
            version: "1.0"
            name: "booking_agent_contracts"
            cases:
              - name: "search_then_book"
                prompt: "Book a flight from NYC to London"
                trajectory:
                  - calls_tools_in_order: ["search_flights", "select_flight", "book_flight"]
                  - does_not_call: "cancel_booking"
                contracts:
                  search_flights:
                    - key: "origin"
                      not_empty: true
                    - key: "destination"
                      not_empty: true
                  book_flight:
                    - key: "passenger_name"
                      not_empty: true
        """)
        suite = load_yaml_suite(path)
        case = suite.cases[0]
        # 2 trajectory + 3 contract = 5 trajectory assertions total
        assert len(case.trajectory_assertions) == 5
        assert not case.has_content_assertions

    def test_content_and_trajectory_combined(self, tmp_path):
        """A case with both classic assertions and trajectory assertions."""
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "hybrid"
                prompt: "Search for flights and respond as JSON"
                assertions:
                  - type: is_valid_json
                trajectory:
                  - calls_tool: "search_flights"
        """)
        suite = load_yaml_suite(path)
        case = suite.cases[0]
        assert case.has_content_assertions
        assert case.has_trajectory_assertions
        assert len(case.assertions) == 1
        assert len(case.trajectory_assertions) == 1

    def test_multiple_cases_mixed(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "json_check"
                prompt: "Return JSON"
                assertions:
                  - type: is_valid_json
              - name: "tool_check"
                prompt: "Book a flight"
                trajectory:
                  - calls_tool: "search_flights"
                contracts:
                  search_flights:
                    - key: "query"
                      not_empty: true
        """)
        suite = load_yaml_suite(path)
        assert len(suite.cases) == 2

        content_case = suite.cases[0]
        assert content_case.has_content_assertions
        assert not content_case.has_trajectory_assertions

        traj_case = suite.cases[1]
        assert not traj_case.has_content_assertions
        assert traj_case.has_trajectory_assertions

    def test_no_assertions_at_all_raises(self, tmp_path):
        path = _write_suite(tmp_path, """
            version: "1.0"
            cases:
              - name: "empty_case"
                prompt: "Do something"
        """)
        with pytest.raises(SuiteParseError, match="no assertions"):
            load_yaml_suite(path)


class TestSuiteRunnerTrajectory:
    """Test that the runner executes trajectory assertions in suite cases."""

    def test_trajectory_case_passes_with_mock(self, tmp_path):
        """MockProvider returns the prompt as content with no tool calls.
        Trajectory assertions against empty tool calls should fail/pass
        according to assertion logic."""
        from llm_assert.core.runner import AssertionRunner
        from llm_assert.core.suite import AssertionCase, AssertionSuite
        from llm_assert.assertions.trajectory import DoesNotCall
        from llm_assert.providers.mock import MockProvider

        provider = MockProvider(
            response_fn=lambda p, m=None: "ok",
            tool_calls=[
                {"name": "search", "arguments": {"q": "test"}},
                {"name": "book", "arguments": {"id": 1}},
            ],
        )
        runner = AssertionRunner(provider=provider)

        case = AssertionCase(
            name="test_case",
            prompt="test",
            trajectory_assertions=[DoesNotCall("cancel")],
        )
        suite = AssertionSuite(name="test", cases=[case])
        result = runner.run_suite(suite)

        assert result.passed
        assert result.total_cases == 1
        assert result.passed_cases == 1

    def test_trajectory_case_fails_on_violation(self, tmp_path):
        from llm_assert.core.runner import AssertionRunner
        from llm_assert.core.suite import AssertionCase, AssertionSuite
        from llm_assert.assertions.trajectory import CallsTool
        from llm_assert.providers.mock import MockProvider

        provider = MockProvider(
            response_fn=lambda p, m=None: "ok",
            tool_calls=[
                {"name": "search", "arguments": {}},
            ],
        )
        runner = AssertionRunner(provider=provider)

        case = AssertionCase(
            name="missing_tool",
            prompt="test",
            trajectory_assertions=[CallsTool("book")],
        )
        suite = AssertionSuite(name="test", cases=[case])
        result = runner.run_suite(suite)

        assert not result.passed
        assert result.failed_cases == 1

    def test_mixed_content_and_trajectory_in_suite(self):
        from llm_assert.core.runner import AssertionRunner
        from llm_assert.core.suite import AssertionCase, AssertionSuite
        from llm_assert.assertions.structural import IsValidJson
        from llm_assert.assertions.trajectory import CallsTool
        from llm_assert.providers.mock import MockProvider

        provider = MockProvider(
            response_fn=lambda p, m=None: '{"status": "ok"}',
            tool_calls=[{"name": "search", "arguments": {}}],
        )
        runner = AssertionRunner(provider=provider)

        case = AssertionCase(
            name="hybrid",
            prompt="test",
            assertions=[IsValidJson()],
            trajectory_assertions=[CallsTool("search")],
        )
        suite = AssertionSuite(name="test", cases=[case])
        result = runner.run_suite(suite)

        assert result.passed
        # Both content and trajectory assertion results present
        case_result = result.case_results["hybrid"]
        assert len(case_result.assertions) == 2
