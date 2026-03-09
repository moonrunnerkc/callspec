"""The Refund Agent That Stopped Checking for Fraud.

A callspec case study demonstrating how tool-call contract testing
catches silent regressions that text-based tests miss.

Run:
    pytest case_study/ -v

No API keys required. No external services. All trajectories are
recorded JSON files loaded from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from callspec import ToolCallTrajectory
from callspec.core.trajectory import ToolCall
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.snapshots.diff import SnapshotDiff
from callspec.snapshots.manager import SnapshotManager

TRAJECTORY_DIR = Path(__file__).parent / "trajectories"
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def load_trajectory(filename: str) -> ToolCallTrajectory:
    """Load a recorded trajectory from a JSON file."""
    path = TRAJECTORY_DIR / filename
    data = json.loads(path.read_text())
    calls = [
        ToolCall(
            tool_name=c["tool_name"],
            arguments=c["arguments"],
            call_index=c["call_index"],
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )
        for c in data["calls"]
    ]
    return ToolCallTrajectory(
        calls=calls,
        model=data.get("model", ""),
        provider=data.get("provider", ""),
    )


# ── The contract ──
#
# A refund agent MUST follow this tool-call sequence for every request:
#
#   1. verify_customer  -- confirm the requester owns the account
#   2. check_order      -- pull the order details
#   3. fraud_check      -- assess fraud risk (non-negotiable)
#   4. issue_refund     -- process the refund
#
# Skipping fraud_check means issuing refunds on potentially stolen
# accounts. The contract is the same whether the refund is $5 or $5000.


REQUIRED_SEQUENCE = [
    "verify_customer",
    "check_order",
    "fraud_check",
    "issue_refund",
]


# ── Part 1: The baseline passes ──

class TestBaselineBehavior:
    """GPT-4o follows the contract on every request."""

    def test_low_value_refund_follows_contract(self):
        """$32 refund: all four tools called in order."""
        trajectory = load_trajectory("gpt4o_correct.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            .calls_tool("fraud_check")
            .argument_not_empty("check_order", "order_id")
            .argument_not_empty("fraud_check", "order_id")
            .run()
        )
        assert result.passed

    def test_high_value_refund_follows_contract(self):
        """$340 refund: identical contract, no special treatment."""
        trajectory = load_trajectory("gpt4o_high_value.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            .calls_tool("fraud_check")
            .argument_not_empty("fraud_check", "refund_amount")
            .run()
        )
        assert result.passed


# ── Part 2: The regression ──

class TestNewModelRegression:
    """After a model swap, low-value refunds skip fraud_check.

    The new model produces correct conversational output. The
    response is polite, references the right order, and formats
    the refund confirmation properly. Text-based tests see no
    regression. But the tool-call contract is broken.
    """

    def test_low_value_refund_skips_fraud_check(self):
        """$32 refund on new model: fraud_check is missing.

        This is the test that catches the regression. Without
        callspec, this failure ships to production.
        """
        trajectory = load_trajectory("new_model_regression.json")

        # The exact sequence assertion catches the missing tool
        result = (
            TrajectoryBuilder(trajectory)
            .calls_exactly(REQUIRED_SEQUENCE)
            .run()
        )
        assert not result.passed, "Should fail: fraud_check is missing"

        failure = result.assertions[0]
        assert "fraud_check" in failure.message

    def test_contract_catches_missing_fraud_check(self):
        """The ordering assertion also catches it."""
        trajectory = load_trajectory("new_model_regression.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            .run()
        )
        assert not result.passed

    def test_calls_tool_catches_missing_fraud_check(self):
        """Even a single assertion is enough."""
        trajectory = load_trajectory("new_model_regression.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tool("fraud_check")
            .run()
        )
        assert not result.passed
        assert "fraud_check" in result.assertions[0].message

    def test_high_value_refund_still_passes(self):
        """$340 refund on new model: fraud_check IS present.

        This is what makes the regression insidious. Large refunds
        still follow the contract. Only small refunds are affected.
        If you only test with one example, you might pick the one
        that passes.
        """
        trajectory = load_trajectory("new_model_high_value.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            .calls_tool("fraud_check")
            .run()
        )
        assert result.passed


# ── Part 3: Snapshot drift detection ──

class TestSnapshotDrift:
    """Demonstrate that snapshot baselines catch the drift automatically.

    If the team had recorded a baseline from GPT-4o and run callspec
    in CI after the model swap, the first run would have flagged
    the trajectory change before it shipped.
    """

    def test_baseline_matches_correct_behavior(self):
        """The baseline was recorded from GPT-4o. It matches itself."""
        manager = SnapshotManager(
            snapshot_dir=str(SNAPSHOT_DIR),
        )
        trajectory = load_trajectory("gpt4o_correct.json")

        result = (
            TrajectoryBuilder(trajectory)
            .sequence_matches_baseline("refund_low_value", manager)
            .run()
        )
        assert result.passed

    def test_regression_fails_baseline(self):
        """The new model trajectory does not match the baseline."""
        manager = SnapshotManager(
            snapshot_dir=str(SNAPSHOT_DIR),
        )
        trajectory = load_trajectory("new_model_regression.json")

        result = (
            TrajectoryBuilder(trajectory)
            .sequence_matches_baseline("refund_low_value", manager)
            .run()
        )
        assert not result.passed
        assert "sequence" in result.assertions[0].message.lower()

    def test_drift_diff_shows_removed_tool(self):
        """The diff report tells you exactly what changed."""
        baseline = load_trajectory("gpt4o_correct.json")
        current = load_trajectory("new_model_regression.json")

        diff = SnapshotDiff.compare_trajectories(
            snapshot_key="refund_low_value",
            baseline_calls=[c.to_dict() for c in baseline.calls],
            current_calls=[c.to_dict() for c in current.calls],
            baseline_model=baseline.model,
            current_model=current.model,
        )

        assert not diff.sequence_match
        assert "fraud_check" in diff.tools_removed
        assert diff.model_changed

        # The detailed report is what appears in CI output
        report = diff.detailed_report()
        assert "fraud_check" in report


# ── Part 4: Full contract with argument validation ──

class TestFullContract:
    """The complete contract a production refund agent should enforce.

    This is what you would put in your actual test suite. It
    covers tool ordering, required tools, forbidden tools, and
    argument validation in a single chainable assertion.
    """

    def test_correct_trajectory_passes_full_contract(self):
        trajectory = load_trajectory("gpt4o_correct.json")

        result = (
            TrajectoryBuilder(trajectory)
            # Tool ordering: the sequence is non-negotiable
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            # Every tool must appear exactly once
            .call_count("verify_customer", min_count=1, max_count=1)
            .call_count("fraud_check", min_count=1, max_count=1)
            .call_count("issue_refund", min_count=1, max_count=1)
            # Dangerous tools must never appear
            .does_not_call("delete_account")
            .does_not_call("override_fraud")
            # Arguments must be present and valid
            .argument_not_empty("verify_customer", "customer_id")
            .argument_not_empty("check_order", "order_id")
            .argument_not_empty("fraud_check", "order_id")
            .argument_not_empty("issue_refund", "order_id")
            .argument_not_empty("issue_refund", "amount")
            .run()
        )
        assert result.passed

    def test_regression_trajectory_fails_full_contract(self):
        trajectory = load_trajectory("new_model_regression.json")

        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(REQUIRED_SEQUENCE)
            .call_count("fraud_check", min_count=1, max_count=1)
            .argument_not_empty("fraud_check", "order_id")
            .run()
        )
        assert not result.passed

        # Collect the names of all failed assertions
        failed = [a.assertion_name for a in result.assertions if not a.passed]
        assert len(failed) >= 1
