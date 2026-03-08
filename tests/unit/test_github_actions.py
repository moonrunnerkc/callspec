"""Tests for the GitHub Actions annotation formatter.

Validates annotation formatting, borderline pass detection, step summary
generation, output parameter setting, and the full emit_suite_result flow.
Tests run without a real GitHub Actions environment by controlling os.environ.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from llm_assert.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    SuiteResult,
)
from llm_assert.integrations.github_actions import (
    _is_borderline_pass,
    annotate_assertion_result,
    annotate_individual_result,
    annotate_suite_result,
    emit_annotations,
    emit_suite_result,
    format_annotation,
    is_github_actions,
    set_output,
    write_step_summary,
)

# ---------------------------------------------------------------------------
# Helpers for building test results
# ---------------------------------------------------------------------------

def _make_individual(
    passed: bool,
    name: str = "is_valid_json",
    assertion_type: str = "structural",
    message: str = "",
    score: float | None = None,
    threshold: float | None = None,
) -> IndividualAssertionResult:
    return IndividualAssertionResult(
        assertion_type=assertion_type,
        assertion_name=name,
        passed=passed,
        message=message or (f"{name} passed" if passed else f"{name} failed"),
        score=score,
        threshold=threshold,
    )


def _make_assertion_result(
    individuals: list[IndividualAssertionResult],
) -> AssertionResult:
    all_passed = all(i.passed for i in individuals)
    return AssertionResult(
        passed=all_passed,
        assertions=individuals,
        execution_time_ms=42,
        model="mock-v1",
    )


def _make_suite_result(
    case_map: dict[str, AssertionResult],
) -> SuiteResult:
    passed_count = sum(1 for r in case_map.values() if r.passed)
    failed_count = sum(1 for r in case_map.values() if not r.passed)
    return SuiteResult(
        passed=(failed_count == 0),
        case_results=case_map,
        total_cases=len(case_map),
        passed_cases=passed_count,
        failed_cases=failed_count,
        execution_time_ms=100,
    )


# ---------------------------------------------------------------------------
# is_github_actions
# ---------------------------------------------------------------------------

class TestIsGitHubActions:

    def test_detects_github_actions_env(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert is_github_actions() is True

    def test_returns_false_when_not_set(self):
        env = os.environ.copy()
        env.pop("GITHUB_ACTIONS", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_github_actions() is False

    def test_returns_false_for_non_true_value(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            assert is_github_actions() is False


# ---------------------------------------------------------------------------
# format_annotation
# ---------------------------------------------------------------------------

class TestFormatAnnotation:

    def test_error_with_all_params(self):
        result = format_annotation(
            level="error",
            message="something broke",
            file="tests/test_llm.py",
            line=42,
            col=8,
            title="LLMAssert: is_valid_json",
        )
        assert result == (
            "::error file=tests/test_llm.py,line=42,col=8,"
            "title=LLMAssert: is_valid_json::something broke"
        )

    def test_warning_with_file_only(self):
        result = format_annotation(
            level="warning",
            message="borderline pass",
            file="test.py",
        )
        assert result == "::warning file=test.py::borderline pass"

    def test_notice_with_no_params(self):
        result = format_annotation(level="notice", message="all good")
        assert result == "::notice::all good"

    def test_error_with_file_and_line(self):
        result = format_annotation(
            level="error",
            message="fail",
            file="src/app.py",
            line=10,
        )
        assert result == "::error file=src/app.py,line=10::fail"

    def test_title_without_file(self):
        result = format_annotation(
            level="notice",
            message="info",
            title="Custom Title",
        )
        assert result == "::notice title=Custom Title::info"


# ---------------------------------------------------------------------------
# _is_borderline_pass
# ---------------------------------------------------------------------------

class TestIsBorderlinePass:

    def test_borderline_when_score_near_threshold(self):
        # Score 0.76 with threshold 0.75: margin is 0.0375, score <= 0.7875
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.76,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is True

    def test_not_borderline_when_score_well_above(self):
        # Score 0.92 with threshold 0.75: 0.92 > 0.7875
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.92,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is False

    def test_not_borderline_when_no_score(self):
        result = _make_individual(passed=True, score=None, threshold=None)
        assert _is_borderline_pass(result) is False

    def test_not_borderline_when_failed(self):
        # Even if score is near threshold, a failure is not borderline
        result = _make_individual(
            passed=False,
            name="semantic_intent_matches",
            score=0.74,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is False

    def test_not_borderline_when_no_threshold(self):
        result = _make_individual(passed=True, score=0.80, threshold=None)
        assert _is_borderline_pass(result) is False

    def test_borderline_at_exact_threshold(self):
        # Score exactly at threshold: 0.75 <= 0.75 + 0.0375
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.75,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is True

    def test_borderline_at_upper_margin_boundary(self):
        # Score exactly at threshold + 5%: 0.7875 <= 0.7875
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.7875,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is True

    def test_not_borderline_just_above_margin(self):
        # Score just above threshold + 5%
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.7876,
            threshold=0.75,
        )
        assert _is_borderline_pass(result) is False


# ---------------------------------------------------------------------------
# annotate_individual_result
# ---------------------------------------------------------------------------

class TestAnnotateIndividualResult:

    def test_failed_assertion_produces_error(self):
        result = _make_individual(
            passed=False,
            name="is_valid_json",
            message="Expected valid JSON but got parse error at position 12",
        )
        annotation = annotate_individual_result(
            result, file="tests/test_llm.py", line=15
        )
        assert annotation.startswith("::error ")
        assert "is_valid_json" in annotation
        assert "parse error" in annotation

    def test_passing_assertion_produces_empty_string(self):
        result = _make_individual(passed=True)
        annotation = annotate_individual_result(result)
        assert annotation == ""

    def test_borderline_pass_produces_warning(self):
        result = _make_individual(
            passed=True,
            name="semantic_intent_matches",
            score=0.76,
            threshold=0.75,
        )
        annotation = annotate_individual_result(
            result, file="tests/test_sem.py", line=20
        )
        assert "::warning" in annotation
        assert "Borderline pass" in annotation
        assert "0.7600" in annotation

    def test_file_and_line_included_in_annotation(self):
        result = _make_individual(
            passed=False,
            name="matches_schema",
            message="Schema validation failed",
        )
        annotation = annotate_individual_result(
            result, file="tests/test_schema.py", line=99
        )
        assert "file=tests/test_schema.py" in annotation
        assert "line=99" in annotation


# ---------------------------------------------------------------------------
# annotate_assertion_result
# ---------------------------------------------------------------------------

class TestAnnotateAssertionResult:

    def test_all_passing_produces_no_annotations(self):
        chain_result = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
            _make_individual(passed=True, name="contains_keys"),
        ])
        annotations = annotate_assertion_result(chain_result, test_name="test_json")
        assert annotations == []

    def test_one_failure_produces_one_annotation(self):
        chain_result = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
            _make_individual(passed=False, name="contains_keys", message="Missing key: title"),
        ])
        annotations = annotate_assertion_result(
            chain_result, test_name="test_json", file="test.py"
        )
        assert len(annotations) == 1
        assert "Missing key: title" in annotations[0]

    def test_multiple_failures_produce_multiple_annotations(self):
        chain_result = _make_assertion_result([
            _make_individual(passed=False, name="is_valid_json", message="not json"),
            _make_individual(passed=False, name="length_between", message="too short"),
        ])
        annotations = annotate_assertion_result(chain_result, test_name="test_multi")
        assert len(annotations) == 2

    def test_borderline_pass_included_in_annotations(self):
        chain_result = _make_assertion_result([
            _make_individual(
                passed=True,
                name="semantic_intent_matches",
                score=0.76,
                threshold=0.75,
            ),
        ])
        annotations = annotate_assertion_result(chain_result)
        assert len(annotations) == 1
        assert "::warning" in annotations[0]


# ---------------------------------------------------------------------------
# annotate_suite_result
# ---------------------------------------------------------------------------

class TestAnnotateSuiteResult:

    def test_passing_suite_produces_summary_only(self):
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({"case_one": passing_case})
        annotations = annotate_suite_result(suite)
        # Only the summary annotation
        assert len(annotations) == 1
        assert "::notice" in annotations[0]
        assert "passed" in annotations[0]
        assert "1/1" in annotations[0]

    def test_failing_suite_produces_error_annotations_and_summary(self):
        failing_case = _make_assertion_result([
            _make_individual(passed=False, name="is_valid_json", message="not json"),
        ])
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({
            "case_fail": failing_case,
            "case_pass": passing_case,
        })
        annotations = annotate_suite_result(suite)
        # One error for the failure + one summary (also ::error for failing suite)
        assert len(annotations) == 2
        error_annotations = [a for a in annotations if a.startswith("::error")]
        assert len(error_annotations) == 2
        assert "FAILED" in annotations[-1]

    def test_file_propagated_to_case_annotations(self):
        failing_case = _make_assertion_result([
            _make_individual(passed=False, name="is_valid_json", message="fail"),
        ])
        suite = _make_suite_result({"case_one": failing_case})
        annotations = annotate_suite_result(suite, file="suite.yaml")
        for annotation in annotations:
            assert "file=suite.yaml" in annotation


# ---------------------------------------------------------------------------
# emit_annotations
# ---------------------------------------------------------------------------

class TestEmitAnnotations:

    def test_prints_each_annotation_to_stdout(self, capsys):
        annotations = [
            "::error file=test.py::failure one",
            "::warning::borderline",
            "::notice::summary",
        ]
        emit_annotations(annotations)
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "::error file=test.py::failure one"

    def test_skips_empty_strings(self, capsys):
        emit_annotations(["::error::fail", "", "::notice::ok"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# write_step_summary
# ---------------------------------------------------------------------------

class TestWriteStepSummary:

    def test_writes_markdown_to_summary_file(self, tmp_path):
        summary_file = tmp_path / "summary.md"
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({"json_check": passing_case})

        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}):
            write_step_summary(suite, suite_name="my_suite")

        content = summary_file.read_text()
        assert "## LLMAssert: my_suite" in content
        assert "1/1" in content
        assert "json_check" in content
        assert "| Case |" in content

    def test_includes_failure_details_in_table(self, tmp_path):
        summary_file = tmp_path / "summary.md"
        failing_case = _make_assertion_result([
            _make_individual(
                passed=False,
                name="matches_schema",
                message="Schema validation failed: missing required field 'title'",
            ),
        ])
        suite = _make_suite_result({"schema_check": failing_case})

        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}):
            write_step_summary(suite, suite_name="api_suite")

        content = summary_file.read_text()
        assert "Fail" in content
        assert "Schema validation failed" in content

    def test_no_op_when_env_var_missing(self, tmp_path):
        """No crash and no file creation when GITHUB_STEP_SUMMARY is not set."""
        env = os.environ.copy()
        env.pop("GITHUB_STEP_SUMMARY", None)
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({"case_one": passing_case})

        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            write_step_summary(suite)

    def test_handles_unwritable_path_gracefully(self):
        """OSError on write is caught, not propagated."""
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({"case_one": passing_case})

        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": "/nonexistent/dir/summary.md"}):
            # Should not raise
            write_step_summary(suite)


# ---------------------------------------------------------------------------
# set_output
# ---------------------------------------------------------------------------

class TestSetOutput:

    def test_writes_name_value_pair(self, tmp_path):
        output_file = tmp_path / "output.txt"

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            set_output("llm_assert_passed", "true")
            set_output("llm_assert_total_cases", "5")

        content = output_file.read_text()
        assert "llm_assert_passed=true\n" in content
        assert "llm_assert_total_cases=5\n" in content

    def test_no_op_when_env_var_missing(self):
        env = os.environ.copy()
        env.pop("GITHUB_OUTPUT", None)
        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            set_output("llm_assert_passed", "true")

    def test_handles_unwritable_path_gracefully(self):
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "/nonexistent/dir/output.txt"}):
            set_output("llm_assert_passed", "true")


# ---------------------------------------------------------------------------
# emit_suite_result (full integration)
# ---------------------------------------------------------------------------

class TestEmitSuiteResult:

    def test_emits_all_three_channels(self, tmp_path, capsys):
        """Annotations to stdout, summary to file, outputs to file."""
        summary_file = tmp_path / "summary.md"
        output_file = tmp_path / "output.txt"

        failing_case = _make_assertion_result([
            _make_individual(
                passed=False,
                name="is_valid_json",
                message="Expected valid JSON",
            ),
        ])
        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="contains_keys"),
        ])
        suite = _make_suite_result({
            "failing_case": failing_case,
            "passing_case": passing_case,
        })

        env_vars = {
            "GITHUB_STEP_SUMMARY": str(summary_file),
            "GITHUB_OUTPUT": str(output_file),
        }
        with patch.dict(os.environ, env_vars):
            emit_suite_result(suite, suite_name="test_suite", file="suite.yaml")

        # Annotations printed to stdout
        captured = capsys.readouterr()
        assert "::error" in captured.out
        assert "Expected valid JSON" in captured.out
        assert "LLMAssert Summary" in captured.out

        # Step summary written
        summary_content = summary_file.read_text()
        assert "## LLMAssert: test_suite" in summary_content
        assert "failing_case" in summary_content

        # Outputs written
        output_content = output_file.read_text()
        assert "llm_assert_passed=false" in output_content
        assert "llm_assert_passed_cases=1" in output_content
        assert "llm_assert_failed_cases=1" in output_content
        assert "llm_assert_total_cases=2" in output_content

    def test_works_without_github_env_vars(self, capsys):
        """Annotations still print to stdout even without summary/output files."""
        env = os.environ.copy()
        env.pop("GITHUB_STEP_SUMMARY", None)
        env.pop("GITHUB_OUTPUT", None)

        failing_case = _make_assertion_result([
            _make_individual(passed=False, name="is_valid_json", message="bad json"),
        ])
        suite = _make_suite_result({"case_one": failing_case})

        with patch.dict(os.environ, env, clear=True):
            emit_suite_result(suite)

        captured = capsys.readouterr()
        assert "::error" in captured.out

    def test_all_passing_suite_outputs(self, tmp_path, capsys):
        """Passing suite still produces summary and outputs."""
        summary_file = tmp_path / "summary.md"
        output_file = tmp_path / "output.txt"

        passing_case = _make_assertion_result([
            _make_individual(passed=True, name="is_valid_json"),
        ])
        suite = _make_suite_result({"case_one": passing_case})

        env_vars = {
            "GITHUB_STEP_SUMMARY": str(summary_file),
            "GITHUB_OUTPUT": str(output_file),
        }
        with patch.dict(os.environ, env_vars):
            emit_suite_result(suite, suite_name="passing_suite")

        captured = capsys.readouterr()
        assert "::notice" in captured.out
        assert "passed" in captured.out

        output_content = output_file.read_text()
        assert "llm_assert_passed=true" in output_content
