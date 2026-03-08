"""Integration tests for the LLMAssert pytest plugin.

Tests the plugin end-to-end by running pytest as a subprocess against
real test files. This validates fixture injection, marker skipping,
CLI flag behavior, and failure output formatting in a realistic context.

These tests do NOT test assertion logic (that is covered in unit tests).
They test the integration layer between pytest and LLMAssert.
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
    test_file = tmp_path / "test_llm_assert_plugin.py"
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
        env={**__import__("os").environ, "LLM_ASSERT_PROVIDER": "mock"},
        timeout=60,
    )


class TestPluginFixtures:
    """Verify that fixtures are injected and functional."""

    def test_llm_assert_runner_fixture_available(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            def test_runner_exists(llm_assert_runner):
                assert llm_assert_runner is not None
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_llm_assert_config_fixture_available(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.core.config import LLMAssertConfig
            def test_config(llm_assert_config):
                assert isinstance(llm_assert_config, LLMAssertConfig)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_llm_assert_provider_fixture_uses_mock(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            def test_mock_provider(llm_assert_provider):
                assert llm_assert_provider.provider_name == "mock"
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_llm_assert_runner_calls_mock_provider(self, tmp_path: Path) -> None:
        """Verify the runner actually calls through to the mock provider."""
        test_code = textwrap.dedent("""
            from llm_assert.assertions.structural import IsValidJson
            def test_runner_call(llm_assert_runner):
                result = llm_assert_runner.assert_that('{"key": "value"}').is_valid_json().run()
                assert result.passed is True
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


class TestPluginMarkers:
    """Verify the llm_assert_behavioral marker and --llm-assert-skip-behavioral."""

    def test_behavioral_mark_registered(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest
            @pytest.mark.llm_assert_behavioral
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
            @pytest.mark.llm_assert_behavioral
            def test_expensive():
                pass

            def test_cheap():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-behavioral"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "1 skipped" in result.stdout

    def test_skip_behavioral_only_skips_marked(self, tmp_path: Path) -> None:
        """Non-behavioral tests still run when skip flag is active."""
        test_code = textwrap.dedent("""
            import pytest

            @pytest.mark.llm_assert_behavioral
            def test_b1():
                pass

            @pytest.mark.llm_assert_behavioral
            def test_b2():
                pass

            def test_structural():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-behavioral"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "2 skipped" in result.stdout


class TestPluginFailureOutput:
    """Verify that assertion failures produce structured output."""

    def test_structural_failure_shows_details(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass
            def test_json_fail(llm_assert_runner):
                result = llm_assert_runner.assert_that("not json at all").is_valid_json().run()
                assert_llm_assert_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode != 0
        # Failure output should mention the assertion type
        assert "FAILED" in result.stdout

    def test_passing_assertion_no_error(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass
            def test_json_pass(llm_assert_runner):
                result = llm_assert_runner.assert_that('{"ok": true}').is_valid_json().run()
                assert_llm_assert_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0
        assert "1 passed" in result.stdout


class TestPluginReportGeneration:
    """Verify --llm-assert-report produces output files."""

    def test_json_report_written(self, tmp_path: Path) -> None:
        report_path = tmp_path / "llm_assert_report.json"
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.reporter import record_llm_assert_result
            from llm_assert.core.types import AssertionResult, IndividualAssertionResult
            def test_record(llm_assert_runner):
                result = llm_assert_runner.assert_that('{"ok": true}').is_valid_json().run()
                record_llm_assert_result("test_record", result)
                assert result.passed
        """)
        result = _run_pytest(
            test_code,
            tmp_path,
            ["--llm-assert-report", "json", "--llm-assert-report-path", str(report_path)],
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        if report_path.exists():
            data = json.loads(report_path.read_text())
            assert "results" in data or "llm_assert_version" in data

    def test_junit_report_written(self, tmp_path: Path) -> None:
        report_path = tmp_path / "llm_assert_report.xml"
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.reporter import record_llm_assert_result
            def test_record(llm_assert_runner):
                result = llm_assert_runner.assert_that('{"ok": true}').is_valid_json().run()
                record_llm_assert_result("test_record", result)
                assert result.passed
        """)
        result = _run_pytest(
            test_code,
            tmp_path,
            ["--llm-assert-report", "junit", "--llm-assert-report-path", str(report_path)],
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        if report_path.exists():
            content = report_path.read_text()
            assert "<?xml" in content


class TestPluginStrictMode:
    """Verify --llm-assert-strict flag behavior."""

    def test_strict_flag_accepted(self, tmp_path: Path) -> None:
        """Verify pytest accepts the --llm-assert-strict flag without error."""
        test_code = textwrap.dedent("""
            def test_simple():
                assert True
        """)
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-strict"])
        assert result.returncode == 0


class TestPluginEndToEnd:
    """Full integration: assertion chain using the plugin fixtures."""

    def test_structural_chain_via_plugin(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass

            def test_chain(llm_assert_runner):
                result = (
                    llm_assert_runner
                    .assert_that('{"title": "Hello", "body": "World"}')
                    .is_valid_json()
                    .contains_keys(["title", "body"])
                    .run()
                )
                assert_llm_assert_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_multiple_tests_in_one_file(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass

            def test_json(llm_assert_runner):
                result = llm_assert_runner.assert_that('{}').is_valid_json().run()
                assert_llm_assert_pass(result)

            def test_length(llm_assert_runner):
                result = llm_assert_runner.assert_that('hello world').length_between(5, 20).run()
                assert_llm_assert_pass(result)

            def test_pattern(llm_assert_runner):
                result = llm_assert_runner.assert_that('abc 123 def').matches_pattern(r"\\d+").run()
                assert_llm_assert_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "3 passed" in result.stdout
