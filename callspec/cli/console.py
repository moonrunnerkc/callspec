"""Shared Rich console and style constants for Callspec CLI output.

All CLI commands import the console from here so terminal detection,
style choices, and markup behavior stay consistent across every surface.
Rich auto-disables markup when stdout is not a TTY, which makes piped
output clean without any conditional logic in the commands themselves.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Callspec-specific semantic styles. Every color choice in the CLI
# traces back to this theme, so rebranding is a single-file edit.
CALLSPEC_THEME = Theme({
    "callspec.pass": "bold green",
    "callspec.fail": "bold red",
    "callspec.warn": "bold yellow",
    "callspec.skip": "dim",
    "callspec.score.good": "green",
    "callspec.score.borderline": "yellow",
    "callspec.score.bad": "red",
    "callspec.header": "bold cyan",
    "callspec.model": "dim",
    "callspec.key": "bold",
    "callspec.muted": "dim",
})

# Singleton console used by all CLI commands. The force_terminal
# kwarg is omitted so Rich auto-detects TTY vs pipe.
console = Console(theme=CALLSPEC_THEME)

# Unicode markers for pass/fail/skip status
PASS_MARKER = "[callspec.pass]\u2713[/callspec.pass]"
FAIL_MARKER = "[callspec.fail]\u2717[/callspec.fail]"
SKIP_MARKER = "[callspec.skip]--[/callspec.skip]"
WARN_MARKER = "[callspec.warn]![/callspec.warn]"


def score_style(score: float, threshold: float) -> str:
    """Return the Rich style name for a score relative to its threshold.

    Green if comfortably above threshold, yellow if within 5% margin,
    red if below threshold.
    """
    if score < threshold:
        return "callspec.score.bad"
    margin = threshold * 0.05
    if score < threshold + margin:
        return "callspec.score.borderline"
    return "callspec.score.good"


def format_score(score: float, threshold: float | None = None) -> str:
    """Format a numeric score with color based on proximity to threshold."""
    if threshold is not None:
        style = score_style(score, threshold)
        return f"[{style}]{score:.4f}[/{style}]"
    return f"{score:.4f}"
