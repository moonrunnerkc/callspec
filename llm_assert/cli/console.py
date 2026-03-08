"""Shared Rich console and style constants for LLMAssert CLI output.

All CLI commands import the console from here so terminal detection,
style choices, and markup behavior stay consistent across every surface.
Rich auto-disables markup when stdout is not a TTY, which makes piped
output clean without any conditional logic in the commands themselves.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# LLMAssert-specific semantic styles. Every color choice in the CLI
# traces back to this theme, so rebranding is a single-file edit.
LLM_ASSERT_THEME = Theme({
    "llm_assert.pass": "bold green",
    "llm_assert.fail": "bold red",
    "llm_assert.warn": "bold yellow",
    "llm_assert.skip": "dim",
    "llm_assert.score.good": "green",
    "llm_assert.score.borderline": "yellow",
    "llm_assert.score.bad": "red",
    "llm_assert.header": "bold cyan",
    "llm_assert.model": "dim",
    "llm_assert.key": "bold",
    "llm_assert.muted": "dim",
})

# Singleton console used by all CLI commands. The force_terminal
# kwarg is omitted so Rich auto-detects TTY vs pipe.
console = Console(theme=LLM_ASSERT_THEME)

# Unicode markers for pass/fail/skip status
PASS_MARKER = "[llm_assert.pass]\u2713[/llm_assert.pass]"
FAIL_MARKER = "[llm_assert.fail]\u2717[/llm_assert.fail]"
SKIP_MARKER = "[llm_assert.skip]--[/llm_assert.skip]"
WARN_MARKER = "[llm_assert.warn]![/llm_assert.warn]"


def score_style(score: float, threshold: float) -> str:
    """Return the Rich style name for a score relative to its threshold.

    Green if comfortably above threshold, yellow if within 5% margin,
    red if below threshold.
    """
    if score < threshold:
        return "llm_assert.score.bad"
    margin = threshold * 0.05
    if score < threshold + margin:
        return "llm_assert.score.borderline"
    return "llm_assert.score.good"


def format_score(score: float, threshold: float | None = None) -> str:
    """Format a numeric score with color based on proximity to threshold."""
    if threshold is not None:
        style = score_style(score, threshold)
        return f"[{style}]{score:.4f}[/{style}]"
    return f"{score:.4f}"
