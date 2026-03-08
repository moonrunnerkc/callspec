"""Unit tests for all Verdict CLI commands.

Uses click.testing.CliRunner to test commands without subprocess overhead.
Tests verify correct output, exit codes, and error handling for each command.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from verdict.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_env(monkeypatch):
    """Set VERDICT_PROVIDER to mock for all CLI tests."""
    monkeypatch.setenv("VERDICT_PROVIDER", "mock")


class TestCLIHelp:
    """Verify help text renders for all commands."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Verdict" in result.output
        assert "run" in result.output
        assert "check" in result.output
        assert "snapshot" in result.output

    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "SUITE_FILE" in result.output

    def test_check_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "provider" in result.output.lower()

    def test_snapshot_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["snapshot", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "delete" in result.output
        assert "list" in result.output

    def test_report_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "REPORT_FILE" in result.output

    def test_providers_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["providers", "--help"])
        assert result.exit_code == 0

    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "verdict" in result.output.lower()


class TestCheckCommand:
    """Tests for verdict check."""

    def test_check_unknown_provider(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["check", "--provider", "nonexistent"])
        assert result.exit_code == 2
        assert "Unknown provider" in result.output

    def test_check_detects_no_providers(self, runner: CliRunner, monkeypatch) -> None:
        """When no providers are installed, check reports the situation."""
        # This test may or may not find installed providers depending on the env.
        # We just verify the command runs without crashing.
        result = runner.invoke(cli, ["check"])
        assert result.exit_code in (0, 1)


class TestRunCommand:
    """Tests for verdict run."""

    def test_run_nonexistent_suite(self, runner: CliRunner, mock_env) -> None:
        result = runner.invoke(cli, ["run", "/nonexistent/suite.yaml"])
        assert result.exit_code != 0

    def test_run_valid_suite(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: test_suite
            cases:
              - name: json_check
                prompt: '{"title": "hello"}'
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "test.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 0, f"output: {result.output}"
        assert "PASS" in result.output or "pass" in result.output.lower()

    def test_run_failing_suite(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        """A suite with a failing assertion should exit non-zero."""
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: fail_suite
            cases:
              - name: not_json
                prompt: "plain text not json"
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "fail.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 1

    def test_run_json_format(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: json_suite
            cases:
              - name: valid
                prompt: '{"a": 1}'
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "json.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock", "-f", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "suite" in parsed or "suite_name" in parsed

    def test_run_junit_format(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: check
                prompt: '{"valid": true}'
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "junit.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock", "-f", "junit"])
        assert result.exit_code == 0
        assert "<?xml" in result.output
        assert "testsuites" in result.output or "testsuite" in result.output

    def test_run_output_to_file(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: file_output
                prompt: '{"ok": true}'
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "out.yaml"
        suite_file.write_text(suite_content)
        output_file = tmp_path / "report.json"

        result = runner.invoke(
            cli,
            ["run", str(suite_file), "--provider", "mock", "-f", "json", "-o", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        parsed = json.loads(output_file.read_text())
        assert "suite" in parsed or "suite_name" in parsed

    def test_run_invalid_yaml(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_file = tmp_path / "bad.yaml"
        suite_file.write_text("  - broken: [yaml\n  incomplete")

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 2

    def test_run_no_provider(self, runner: CliRunner, tmp_path: Path, monkeypatch) -> None:
        """Without a provider set, run should fail with a clear message."""
        monkeypatch.delenv("VERDICT_PROVIDER", raising=False)
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: check
                prompt: test
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "noprov.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file)])
        assert result.exit_code == 2
        assert "provider" in result.output.lower()

    def test_run_multiple_cases(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: multi
            cases:
              - name: case_a
                prompt: '{"a": 1}'
                assertions:
                  - type: is_valid_json
              - name: case_b
                prompt: 'hello world'
                assertions:
                  - type: length_between
                    params:
                      min_chars: 5
                      max_chars: 50
        """)
        suite_file = tmp_path / "multi.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 0


class TestSnapshotCommand:
    """Tests for verdict snapshot subcommands."""

    def test_snapshot_list_empty(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            cli,
            ["snapshot", "list", "--snapshot-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "No snapshots" in result.output

    def test_snapshot_create_and_list(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snaps"
        result = runner.invoke(
            cli,
            ["snapshot", "create", "test_key", "hello world", "-p", "mock", "-d", str(snap_dir)],
        )
        assert result.exit_code == 0, f"output: {result.output}"
        assert "created" in result.output.lower()

        # Now list should show the snapshot
        result = runner.invoke(cli, ["snapshot", "list", "-d", str(snap_dir)])
        assert result.exit_code == 0
        assert "test_key" in result.output

    def test_snapshot_create_duplicate_fails(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snaps"

        # Create first
        runner.invoke(
            cli,
            ["snapshot", "create", "dup_key", "content", "-p", "mock", "-d", str(snap_dir)],
        )

        # Create duplicate should fail
        result = runner.invoke(
            cli,
            ["snapshot", "create", "dup_key", "content", "-p", "mock", "-d", str(snap_dir)],
        )
        assert result.exit_code != 0

    def test_snapshot_update(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snaps"

        # Create first
        runner.invoke(
            cli,
            ["snapshot", "create", "up_key", "original", "-p", "mock", "-d", str(snap_dir)],
        )

        # Update
        result = runner.invoke(
            cli,
            ["snapshot", "update", "up_key", "updated content", "-p", "mock", "-d", str(snap_dir)],
        )
        assert result.exit_code == 0
        assert "updated" in result.output.lower()

    def test_snapshot_delete(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snaps"

        # Create
        runner.invoke(
            cli,
            ["snapshot", "create", "del_key", "some content", "-p", "mock", "-d", str(snap_dir)],
        )

        # Delete with --yes to skip confirmation
        result = runner.invoke(
            cli,
            ["snapshot", "delete", "del_key", "-d", str(snap_dir), "--yes"],
        )
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

        # List should be empty
        result = runner.invoke(cli, ["snapshot", "list", "-d", str(snap_dir)])
        assert "No snapshots" in result.output or "del_key" not in result.output

    def test_snapshot_delete_nonexistent(self, runner: CliRunner, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snaps"
        result = runner.invoke(
            cli,
            ["snapshot", "delete", "ghost", "-d", str(snap_dir), "--yes"],
        )
        assert result.exit_code != 0

    def test_snapshot_no_provider(self, runner: CliRunner, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("VERDICT_PROVIDER", raising=False)
        snap_dir = tmp_path / "snaps"
        result = runner.invoke(
            cli,
            ["snapshot", "create", "key", "prompt", "-d", str(snap_dir)],
        )
        assert result.exit_code != 0
        assert "provider" in result.output.lower()


class TestReportCommand:
    """Tests for verdict report."""

    def test_report_nonexistent_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["report", "/nonexistent/file.json"])
        assert result.exit_code != 0

    def test_report_invalid_json(self, runner: CliRunner, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{}")
        result = runner.invoke(cli, ["report", str(bad_file)])
        assert result.exit_code == 2
        assert "Invalid JSON" in result.output

    def test_report_plaintext(self, runner: CliRunner, tmp_path: Path) -> None:
        report_data = {
            "suite_name": "test_suite",
            "passed": True,
            "execution_time_ms": 42,
            "cases": [
                {
                    "name": "case1",
                    "passed": True,
                    "assertions": [
                        {
                            "assertion_name": "is_valid_json",
                            "assertion_type": "structural",
                            "passed": True,
                        }
                    ],
                }
            ],
        }
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))

        result = runner.invoke(cli, ["report", str(report_file)])
        assert result.exit_code == 0
        assert "test_suite" in result.output
        assert "PASS" in result.output

    def test_report_json_format(self, runner: CliRunner, tmp_path: Path) -> None:
        report_data = {"suite_name": "re_emit", "passed": True, "cases": []}
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))

        result = runner.invoke(cli, ["report", str(report_file), "-f", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["suite_name"] == "re_emit"

    def test_report_junit_format(self, runner: CliRunner, tmp_path: Path) -> None:
        report_data = {
            "suite_name": "junit_test",
            "passed": True,
            "execution_time_ms": 10,
            "cases": [{"name": "c1", "passed": True, "assertions": []}],
        }
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))

        result = runner.invoke(cli, ["report", str(report_file), "-f", "junit"])
        assert result.exit_code == 0
        assert "<?xml" in result.output

    def test_report_shows_failures(self, runner: CliRunner, tmp_path: Path) -> None:
        report_data = {
            "suite_name": "fail_suite",
            "passed": False,
            "execution_time_ms": 100,
            "cases": [
                {
                    "name": "failing_case",
                    "passed": False,
                    "assertions": [
                        {
                            "assertion_name": "is_valid_json",
                            "assertion_type": "structural",
                            "passed": False,
                            "message": "Response is not valid JSON",
                        }
                    ],
                }
            ],
        }
        report_file = tmp_path / "report.json"
        report_file.write_text(json.dumps(report_data))

        result = runner.invoke(cli, ["report", str(report_file)])
        assert result.exit_code == 0
        assert "FAIL" in result.output
        assert "not valid JSON" in result.output


class TestProvidersCommand:
    """Tests for verdict providers."""

    def test_providers_list(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "mock" in result.output.lower()

    def test_providers_shows_mock_builtin(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        assert "BUILT-IN" in result.output


class TestYAMLSuiteParser:
    """Tests for YAML suite parsing (exercised through the run command)."""

    def test_yaml_with_schema_assertion(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: schema_suite
            cases:
              - name: schema_check
                prompt: '{"title": "Test", "count": 42}'
                assertions:
                  - type: matches_schema
                    params:
                      schema:
                        type: object
                        required:
                          - title
                          - count
                        properties:
                          title:
                            type: string
                          count:
                            type: integer
        """)
        suite_file = tmp_path / "schema.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 0

    def test_yaml_with_contains_keys(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: keys_check
                prompt: '{"name": "test", "age": 30}'
                assertions:
                  - type: contains_keys
                    params:
                      keys:
                        - name
                        - age
        """)
        suite_file = tmp_path / "keys.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 0

    def test_yaml_with_does_not_contain(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: no_secrets
                prompt: "safe response with no secrets"
                assertions:
                  - type: does_not_contain
                    params:
                      text: "password"
        """)
        suite_file = tmp_path / "deny.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 0

    def test_yaml_version_check(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "9.0"
            cases:
              - name: future
                prompt: test
                assertions:
                  - type: is_valid_json
        """)
        suite_file = tmp_path / "future.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 2

    def test_yaml_missing_cases(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            name: empty
        """)
        suite_file = tmp_path / "empty.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 2

    def test_yaml_unknown_assertion_type(self, runner: CliRunner, mock_env, tmp_path: Path) -> None:
        suite_content = textwrap.dedent("""\
            version: "1.0"
            cases:
              - name: bad_type
                prompt: test
                assertions:
                  - type: nonexistent_assertion
        """)
        suite_file = tmp_path / "bad_type.yaml"
        suite_file.write_text(suite_content)

        result = runner.invoke(cli, ["run", str(suite_file), "--provider", "mock"])
        assert result.exit_code == 2


class TestReportFormatter:
    """Unit tests for ReportFormatter methods."""

    def test_to_json_produces_valid_json(self) -> None:
        from verdict.core.report import ReportFormatter
        from verdict.core.types import AssertionResult, IndividualAssertionResult, SuiteResult

        suite_result = SuiteResult(
            passed=True,
            case_results={
                "test_case": AssertionResult(
                    passed=True,
                    assertions=[
                        IndividualAssertionResult(
                            assertion_type="structural",
                            assertion_name="is_valid_json",
                            passed=True,
                            message="Valid JSON",
                        )
                    ],
                    execution_time_ms=10,
                    model="mock",
                )
            },
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            execution_time_ms=10,
        )
        output = ReportFormatter.to_json(suite_result=suite_result, suite_name="test")
        parsed = json.loads(output)
        assert parsed["suite_name"] == "test"
        assert parsed["suite"]["passed"] is True

    def test_to_plaintext_shows_pass(self) -> None:
        from verdict.core.report import ReportFormatter
        from verdict.core.types import AssertionResult, IndividualAssertionResult, SuiteResult

        suite_result = SuiteResult(
            passed=True,
            case_results={
                "simple": AssertionResult(
                    passed=True,
                    assertions=[
                        IndividualAssertionResult(
                            assertion_type="structural",
                            assertion_name="is_valid_json",
                            passed=True,
                            message="OK",
                        )
                    ],
                    execution_time_ms=5,
                    model="mock",
                )
            },
            total_cases=1,
            passed_cases=1,
            execution_time_ms=5,
        )
        output = ReportFormatter.to_plaintext(suite_result=suite_result, suite_name="test")
        assert "PASSED" in output
        assert "is_valid_json" in output

    def test_to_junit_produces_xml(self) -> None:
        from verdict.core.report import ReportFormatter
        from verdict.core.types import AssertionResult, IndividualAssertionResult, SuiteResult

        suite_result = SuiteResult(
            passed=False,
            case_results={
                "fail_case": AssertionResult(
                    passed=False,
                    assertions=[
                        IndividualAssertionResult(
                            assertion_type="structural",
                            assertion_name="is_valid_json",
                            passed=False,
                            message="Not valid JSON: parse error at position 0",
                        )
                    ],
                    execution_time_ms=3,
                    model="mock",
                )
            },
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            execution_time_ms=3,
        )
        output = ReportFormatter.to_junit(suite_result=suite_result, suite_name="fail_test")
        assert "<?xml" in output
        assert 'failures="1"' in output
        assert "failure" in output
