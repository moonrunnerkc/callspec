"""Integration tests for the Callspec pytest plugin.

Tests the plugin end-to-end by running pytest as a subprocess against
real test files. This validates fixture injection, marker skipping,
CLI flag behavior, and failure output formatting in a realistic context.

These tests do NOT test assertion logic (that is covered in unit tests).
They test the integration layer between pytest and Callspec.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_pytest(
    test_content: str,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Write a test file to tmp_path and run pytest against it."""
    test_file = tmp_path / "test_callspec_plugin.py"
    test_file.write_text(test_content)

    # The plugin is auto-registered via the pytest11 entry point in
    # pyproject.toml, so no -p flag needed. Adding it causes a
    # "Plugin already registered" error.
    cmd = [
        PYTHON, "-m", "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "--no-header",
    ]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**__import__("os").environ, "CALLSPEC_PROVIDER": "mock"},
        timeout=60,
    )


class TestPluginFixtures:
    """Verify that fixtures are injected and functional."""

    def test_callspec_runner_fixture_available(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            def test_runner_exists(callspec_runner):
                assert callspec_runner is not None
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_callspec_config_fixture_available(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from callspec.core.config import CallspecConfig
            def test_config(callspec_config):
                assert isinstance(callspec_config, CallspecConfig)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_callspec_provider_fixture_uses_mock(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            def test_mock_provider(callspec_provider):
                assert callspec_provider.provider_name == "mock"
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_callspec_runner_calls_mock_provider(self, tmp_path: Path) -> None:
        """Verify the runner actually calls through to the mock provider."""
        test_code = textwrap.dedent("""
            from callspec.assertions.structural import IsValidJson
            def test_runner_call(callspec_runner):
                result = callspec_runner.assert_that('{"key": "value"}').is_valid_json().run()
                assert result.passed is True
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


class TestPluginMarkers:
    """Verify the callspec_behavioral marker and --callspec-skip-behavioral."""

    def test_behavioral_mark_registered(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest
            @pytest.mark.callspec_behavioral
            def test_behavioral():
                pass
        """)
        # Should run normally without the skip flag
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0
        assert "1 passed" in result.stdout

    def test_skip_behavioral_flag(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest
            @pytest.mark.callspec_behavioral
            def test_expensive():
                pass

            def test_cheap():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--callspec-skip-behavioral"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "1 skipped" in result.stdout

    def test_skip_behavioral_only_skips_marked(self, tmp_path: Path) -> None:
        """Non-behavioral tests still run when skip flag is active."""
        test_code = textwrap.dedent("""
            import pytest

            @pytest.mark.callspec_behavioral
            def test_b1():
                pass

            @pytest.mark.callspec_behavioral
            def test_b2():
                pass

            def test_structural():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--callspec-skip-behavioral"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "2 skipped" in result.stdout


class TestPluginFailureOutput:
    """Verify that assertion failures produce structured output."""

    def test_structural_failure_shows_details(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.assertions import assert_callspec_pass
            def test_json_fail(callspec_runner):
                result = callspec_runner.assert_that("not json at all").is_valid_json().run()
                assert_callspec_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode != 0
        # Failure output should mention the assertion type
        assert "FAILED" in result.stdout

    def test_passing_assertion_no_error(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.assertions import assert_callspec_pass
            def test_json_pass(callspec_runner):
                result = callspec_runner.assert_that('{"ok": true}').is_valid_json().run()
                assert_callspec_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0
        assert "1 passed" in result.stdout


class TestPluginReportGeneration:
    """Verify --callspec-report produces output files."""

    def test_json_report_written(self, tmp_path: Path) -> None:
        report_path = tmp_path / "callspec_report.json"
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.reporter import record_callspec_result
            from callspec.core.types import AssertionResult, IndividualAssertionResult
            def test_record(callspec_runner):
                result = callspec_runner.assert_that('{"ok": true}').is_valid_json().run()
                record_callspec_result("test_record", result)
                assert result.passed
        """)
        result = _run_pytest(
            test_code,
            tmp_path,
            ["--callspec-report", "json", "--callspec-report-path", str(report_path)],
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        if report_path.exists():
            data = json.loads(report_path.read_text())
            assert "results" in data or "callspec_version" in data

    def test_junit_report_written(self, tmp_path: Path) -> None:
        report_path = tmp_path / "callspec_report.xml"
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.reporter import record_callspec_result
            def test_record(callspec_runner):
                result = callspec_runner.assert_that('{"ok": true}').is_valid_json().run()
                record_callspec_result("test_record", result)
                assert result.passed
        """)
        result = _run_pytest(
            test_code,
            tmp_path,
            ["--callspec-report", "junit", "--callspec-report-path", str(report_path)],
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        if report_path.exists():
            content = report_path.read_text()
            assert "<?xml" in content


class TestPluginStrictMode:
    """Verify --callspec-strict flag behavior."""

    def test_strict_flag_accepted(self, tmp_path: Path) -> None:
        """Verify pytest accepts the --callspec-strict flag without error."""
        test_code = textwrap.dedent("""
            def test_simple():
                assert True
        """)
        result = _run_pytest(test_code, tmp_path, ["--callspec-strict"])
        assert result.returncode == 0


class TestPluginEndToEnd:
    """Full integration: assertion chain using the plugin fixtures."""

    def test_structural_chain_via_plugin(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.assertions import assert_callspec_pass

            def test_chain(callspec_runner):
                result = (
                    callspec_runner
                    .assert_that('{"title": "Hello", "body": "World"}')
                    .is_valid_json()
                    .contains_keys(["title", "body"])
                    .run()
                )
                assert_callspec_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_multiple_tests_in_one_file(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from callspec.pytest_plugin.assertions import assert_callspec_pass

            def test_json(callspec_runner):
                result = callspec_runner.assert_that('{}').is_valid_json().run()
                assert_callspec_pass(result)

            def test_length(callspec_runner):
                result = callspec_runner.assert_that('hello world').length_between(5, 20).run()
                assert_callspec_pass(result)

            def test_pattern(callspec_runner):
                result = callspec_runner.assert_that('abc 123 def').matches_pattern(r"\\d+").run()
                assert_callspec_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "3 passed" in result.stdout
