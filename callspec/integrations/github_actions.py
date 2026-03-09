"""GitHub Actions annotation formatter for CI failure output.

Converts Callspec assertion results into GitHub workflow commands that
appear as annotations directly on the PR diff view. This is the highest-
leverage CI integration: a failing assertion annotates the exact test
file with structured failure context.

Supports both content assertion failures and tool-call contract failures.
Contract failures show which tool call failed, which contract was violated,
and the actual argument value vs. the expected constraint.

Workflow command format:
    ::error file=tests/test_my_llm.py,line=42::SemanticAssertion failed: ...

The Action configuration is minimal by design: provider name (from secrets),
suite file path, and optional flags. More than three config lines reduces
adoption.
"""

from __future__ import annotations

import os

from callspec.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    SuiteResult,
)

# GitHub Actions workflow command severity levels
_SEVERITY_ERROR = "error"
_SEVERITY_WARNING = "warning"
_SEVERITY_NOTICE = "notice"


def is_github_actions() -> bool:
    """Detect whether the current process is running inside GitHub Actions."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def format_annotation(
    level: str,
    message: str,
    file: str | None = None,
    line: int | None = None,
    col: int | None = None,
    title: str | None = None,
) -> str:
    """Build a single GitHub Actions workflow command annotation.

    The format is strict: ::level file=...,line=...,col=...::message
    Each parameter is optional except level and message.
    """
    params: list[str] = []
    if file:
        params.append(f"file={file}")
    if line is not None:
        params.append(f"line={line}")
    if col is not None:
        params.append(f"col={col}")
    if title:
        params.append(f"title={title}")

    param_str = ",".join(params)
    if param_str:
        return f"::{level} {param_str}::{message}"
    return f"::{level}::{message}"


def annotate_individual_result(
    result: IndividualAssertionResult,
    file: str | None = None,
    line: int | None = None,
) -> str:
    """Convert a single assertion result into a GitHub annotation string.

    Failed assertions produce ::error annotations. Borderline passes
    (score within 5% of threshold) produce ::warning annotations to
    flag fragile tests before they break.

    Contract and trajectory failures include tool-call-specific context:
    which tool, which argument, what was expected vs. actual.
    """
    if result.passed:
        if _is_borderline_pass(result):
            return format_annotation(
                level=_SEVERITY_WARNING,
                message=(
                    f"Borderline pass on {result.assertion_name}: "
                    f"score {result.score:.4f} is within 5% of threshold "
                    f"{result.threshold:.4f}. Consider tightening or loosening the threshold."
                ),
                file=file,
                line=line,
                title=f"Callspec: {result.assertion_name} (borderline)",
            )
        return ""

    # Build a detailed message for contract/trajectory failures
    message = _format_contract_failure(result) or result.message

    return format_annotation(
        level=_SEVERITY_ERROR,
        message=message,
        file=file,
        line=line,
        title=f"Callspec: {result.assertion_name}",
    )


def annotate_assertion_result(
    result: AssertionResult,
    test_name: str = "",
    file: str | None = None,
    line: int | None = None,
) -> list[str]:
    """Convert a full assertion chain result into GitHub annotation strings.

    Produces one annotation per failed (or borderline) assertion in the chain.
    Passing assertions produce no output to keep the PR clean.
    """
    annotations: list[str] = []
    for individual in result.assertions:
        annotation = annotate_individual_result(individual, file=file, line=line)
        if annotation:
            annotations.append(annotation)

    return annotations


def annotate_suite_result(
    suite_result: SuiteResult,
    file: str | None = None,
) -> list[str]:
    """Convert a full suite result into GitHub annotation strings.

    One annotation per failing assertion across all cases. Also emits a
    summary annotation with the overall pass/fail count.
    """
    annotations: list[str] = []

    for case_name, case_result in suite_result.case_results.items():
        case_annotations = annotate_assertion_result(
            case_result,
            test_name=case_name,
            file=file,
        )
        annotations.extend(case_annotations)

    # Summary annotation
    status = "passed" if suite_result.passed else "FAILED"
    summary_level = _SEVERITY_NOTICE if suite_result.passed else _SEVERITY_ERROR
    summary = (
        f"Callspec suite {status}: "
        f"{suite_result.passed_cases}/{suite_result.total_cases} cases passed "
        f"({suite_result.execution_time_ms}ms)"
    )
    annotations.append(
        format_annotation(
            level=summary_level,
            message=summary,
            file=file,
            title="Callspec Summary",
        )
    )

    return annotations


def emit_annotations(annotations: list[str]) -> None:
    """Write annotation strings to stdout where GitHub Actions picks them up.

    Each annotation is a single line written to stdout. GitHub Actions
    parses these from the log output automatically.
    """
    for annotation in annotations:
        if annotation:
            print(annotation, flush=True)


def write_step_summary(suite_result: SuiteResult, suite_name: str = "default") -> None:
    """Write a Markdown summary to $GITHUB_STEP_SUMMARY if available.

    The step summary appears in the workflow run UI as a rich Markdown block,
    giving reviewers a quick overview without digging through logs.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines: list[str] = []
    status_emoji = "Pass" if suite_result.passed else "Fail"
    lines.append(f"## Callspec: {suite_name} ({status_emoji})")
    lines.append("")
    lines.append(
        f"**{suite_result.passed_cases}/{suite_result.total_cases}** cases passed "
        f"in **{suite_result.execution_time_ms}ms**"
    )
    lines.append("")
    lines.append("| Case | Status | Details |")
    lines.append("|------|--------|---------|")

    for case_name, case_result in suite_result.case_results.items():
        case_status = "Pass" if case_result.passed else "Fail"
        failure_details = ""
        if not case_result.passed:
            failed_assertions = [a for a in case_result.assertions if not a.passed]
            if failed_assertions:
                failure_details = failed_assertions[0].message[:120]
        lines.append(f"| {case_name} | {case_status} | {failure_details} |")

    lines.append("")

    try:
        with open(summary_path, "a") as summary_file:
            summary_file.write("\n".join(lines) + "\n")
    except OSError:
        # Non-fatal: step summary is a convenience, not a requirement
        pass


def set_output(name: str, value: str) -> None:
    """Set a GitHub Actions output parameter via $GITHUB_OUTPUT.

    Used to pass structured data (pass/fail, score, report path) to
    subsequent workflow steps.
    """
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    try:
        with open(output_path, "a") as output_file:
            output_file.write(f"{name}={value}\n")
    except OSError:
        pass


def emit_suite_result(
    suite_result: SuiteResult,
    suite_name: str = "default",
    file: str | None = None,
) -> None:
    """Complete GitHub Actions integration: annotations, summary, and outputs.

    Call this once at the end of a Callspec run in CI. It handles all three
    output channels: workflow command annotations (inline on the PR),
    step summary (Markdown in the run UI), and output parameters (for
    downstream steps).
    """
    annotations = annotate_suite_result(suite_result, file=file)
    emit_annotations(annotations)
    write_step_summary(suite_result, suite_name)

    set_output("callspec_passed", str(suite_result.passed).lower())
    set_output("callspec_passed_cases", str(suite_result.passed_cases))
    set_output("callspec_failed_cases", str(suite_result.failed_cases))
    set_output("callspec_total_cases", str(suite_result.total_cases))


def _is_borderline_pass(result: IndividualAssertionResult) -> bool:
    """A pass is borderline if the score is within 5% of the threshold.

    Only applies to scored assertions with a defined threshold. Structural
    assertions (binary pass/fail) are never borderline.
    """
    if result.score is None or result.threshold is None:
        return False
    if not result.passed:
        return False

    margin = abs(result.threshold * 0.05)
    return result.score <= (result.threshold + margin)


def _format_contract_failure(result: IndividualAssertionResult) -> str | None:
    """Build a contract-specific failure message with tool call context.

    Returns None if the result doesn't contain contract/trajectory-specific
    details, falling back to result.message in the caller.
    """
    details = result.details
    if not details:
        return None

    parts = [result.message]

    # Trajectory assertion details: show actual vs expected tool names
    if "actual_tools" in details:
        parts.append(f"Actual tools: {details['actual_tools']}")
    if "expected_tools" in details:
        parts.append(f"Expected tools: {details['expected_tools']}")

    # Contract assertion details: show violations
    violations = details.get("violations")
    if violations and isinstance(violations, list):
        for violation in violations[:3]:  # cap at 3 to keep annotations readable
            if isinstance(violation, dict):
                tool = violation.get("tool_name", "?")
                key = violation.get("key", "?")
                actual = violation.get("actual_value", "?")
                parts.append(f"  {tool}.{key}: got {actual!r}")
            elif isinstance(violation, str):
                parts.append(f"  {violation}")

    # Diff detail from regression assertions
    if "tools_added" in details and details["tools_added"]:
        parts.append(f"Tools added: {details['tools_added']}")
    if "tools_removed" in details and details["tools_removed"]:
        parts.append(f"Tools removed: {details['tools_removed']}")

    return " | ".join(parts) if len(parts) > 1 else None
