"""Integration tests for trajectory_runner fixture and tool_contract marker.

Tests the plugin end-to-end by running pytest as a subprocess against
real test files that use the new trajectory fixtures.
"""

from __future__ import annotations

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
    test_file = tmp_path / "test_trajectory_plugin.py"
    test_file.write_text(test_content)

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


class TestTrajectoryRunnerFixture:
    def test_trajectory_runner_available(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            def test_fixture_exists(trajectory_runner):
                assert trajectory_runner is not None
                assert callable(trajectory_runner)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\\nstderr: {result.stderr}"
        assert "1 passed" in result.stdout

    def test_trajectory_runner_returns_builder(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            from llm_assert.core.trajectory_builder import TrajectoryBuilder
            def test_returns_builder(trajectory_runner):
                builder = trajectory_runner("test prompt")
                assert isinstance(builder, TrajectoryBuilder)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\\nstderr: {result.stderr}"

    def test_trajectory_runner_chain_and_run(self, tmp_path: Path) -> None:
        """Verify we can chain trajectory assertions and run them via the fixture."""
        test_code = textwrap.dedent("""
            def test_chain(trajectory_runner):
                # MockProvider returns empty tool_calls by default, so
                # does_not_call should always pass
                builder = trajectory_runner("test prompt")
                result = builder.does_not_call("nonexistent_tool").run()
                assert result.passed
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\\nstderr: {result.stderr}"


class TestToolContractMarker:
    def test_tool_contract_mark_registered(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest
            @pytest.mark.tool_contract
            def test_contract():
                pass
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode == 0
        assert "1 passed" in result.stdout

    def test_skip_contracts_flag(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest

            @pytest.mark.tool_contract
            def test_contract():
                pass

            def test_regular():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-contracts"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "1 skipped" in result.stdout

    def test_skip_contracts_only_skips_marked(self, tmp_path: Path) -> None:
        test_code = textwrap.dedent("""
            import pytest

            @pytest.mark.tool_contract
            def test_c1():
                pass

            @pytest.mark.tool_contract
            def test_c2():
                pass

            def test_normal():
                pass
        """)
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-contracts"])
        assert result.returncode == 0
        assert "1 passed" in result.stdout
        assert "2 skipped" in result.stdout

    def test_behavioral_and_contract_markers_independent(self, tmp_path: Path) -> None:
        """Skipping contracts should not skip behavioral tests and vice versa."""
        test_code = textwrap.dedent("""
            import pytest

            @pytest.mark.tool_contract
            def test_contract():
                pass

            @pytest.mark.llm_assert_behavioral
            def test_behavioral():
                pass

            def test_regular():
                pass
        """)
        # Skip only contracts
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-contracts"])
        assert result.returncode == 0
        assert "2 passed" in result.stdout
        assert "1 skipped" in result.stdout

        # Skip only behavioral
        result = _run_pytest(test_code, tmp_path, ["--llm-assert-skip-behavioral"])
        assert result.returncode == 0
        assert "2 passed" in result.stdout
        assert "1 skipped" in result.stdout


class TestTrajectoryAssertionOutput:
    def test_trajectory_failure_shows_tool_calls(self, tmp_path: Path) -> None:
        """When a trajectory assertion fails, the output should include context."""
        test_code = textwrap.dedent("""
            from llm_assert.pytest_plugin.assertions import assert_llm_assert_pass
            from llm_assert.core.trajectory import ToolCall, ToolCallTrajectory
            from llm_assert.core.trajectory_builder import TrajectoryBuilder

            def test_fail():
                trajectory = ToolCallTrajectory(
                    calls=[
                        ToolCall(tool_name="search", arguments={}, call_index=0),
                    ],
                    model="gpt-4o",
                    provider="openai",
                )
                result = TrajectoryBuilder(trajectory).calls_tool("book").run()
                assert_llm_assert_pass(result)
        """)
        result = _run_pytest(test_code, tmp_path)
        assert result.returncode != 0
        assert "FAILED" in result.stdout
