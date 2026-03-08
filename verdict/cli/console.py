"""Shared Rich console and style constants for Verdict CLI output.

All CLI commands import the console from here so terminal detection,
style choices, and markup behavior stay consistent across every surface.
Rich auto-disables markup when stdout is not a TTY, which makes piped
output clean without any conditional logic in the commands themselves.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Verdict-specific semantic styles. Every color choice in the CLI
# traces back to this theme, so rebranding is a single-file edit.
VERDICT_THEME = Theme({
    "verdict.pass": "bold green",
    "verdict.fail": "bold red",
    "verdict.warn": "bold yellow",
    "verdict.skip": "dim",
    "verdict.score.good": "green",
    "verdict.score.borderline": "yellow",
    "verdict.score.bad": "red",
    "verdict.header": "bold cyan",
    "verdict.model": "dim",
    "verdict.key": "bold",
    "verdict.muted": "dim",
})

# Singleton console used by all CLI commands. The force_terminal
# kwarg is omitted so Rich auto-detects TTY vs pipe.
console = Console(theme=VERDICT_THEME)

# Unicode markers for pass/fail/skip status
PASS_MARKER = "[verdict.pass]\u2713[/verdict.pass]"
FAIL_MARKER = "[verdict.fail]\u2717[/verdict.fail]"
SKIP_MARKER = "[verdict.skip]--[/verdict.skip]"
WARN_MARKER = "[verdict.warn]![/verdict.warn]"


def score_style(score: float, threshold: float) -> str:
    """Return the Rich style name for a score relative to its threshold.

    Green if comfortably above threshold, yellow if within 5% margin,
    red if below threshold.
    """
    if score < threshold:
        return "verdict.score.bad"
    margin = threshold * 0.05
    if score < threshold + margin:
        return "verdict.score.borderline"
    return "verdict.score.good"


def format_score(score: float, threshold: float | None = None) -> str:
    """Format a numeric score with color based on proximity to threshold."""
    if threshold is not None:
        style = score_style(score, threshold)
        return f"[{style}]{score:.4f}[/{style}]"
    return f"{score:.4f}"
