"""callspec snapshot: manage baseline snapshots for regression assertions.

Subcommands: create, update, diff, delete, list.
Snapshots are versioned JSON files that live in the project repository
under callspec_snapshots/baselines.json.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import click

DEFAULT_SNAPSHOT_DIR = "callspec_snapshots"


@click.group()
def snapshot() -> None:
    """Manage baseline snapshots for regression assertions.

    Snapshots record provider outputs at a known-good point in time.
    Subsequent regression assertions compare current outputs against
    the stored baseline.
    """


@snapshot.command("create")
@click.argument("key")
@click.argument("prompt")
@click.option(
    "--provider", "-p",
    default=None,
    help="Provider to use. Falls back to CALLSPEC_PROVIDER env var.",
)
@click.option(
    "--snapshot-dir", "-d",
    default=DEFAULT_SNAPSHOT_DIR,
    help="Directory to store snapshot files.",
)
def snapshot_create(key: str, prompt: str, provider: str | None, snapshot_dir: str) -> None:
    """Create a new baseline snapshot.

    KEY is the identifier for this snapshot (used in regression assertions).
    PROMPT is the input prompt to send to the provider.
    """
    from callspec.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from callspec.errors import SnapshotError
    from callspec.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status(
            "[callspec.muted]Calling provider...[/callspec.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(
            "[callspec.fail]Provider call failed:[/callspec.fail] "
            f"{escape(str(call_err))}"
        )
        sys.exit(1)

    try:
        entry = manager.create_entry(
            snapshot_key=key,
            content=response.content,
            prompt=prompt,
            model=response.model,
            provider=response.provider,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[callspec.fail]Error:[/callspec.fail] {escape(str(snap_err))}")
        sys.exit(1)

    tool_info = f", {len(entry.tool_calls)} tool calls" if entry.tool_calls else ""
    console.print(
        f"{PASS_MARKER} Snapshot [callspec.key]'{key}'[/callspec.key] created "
        f"[callspec.muted]({entry.content_length} chars{tool_info}, model={entry.model})[/callspec.muted]"
    )


@snapshot.command("update")
@click.argument("key")
@click.argument("prompt")
@click.option("--provider", "-p", default=None, help="Provider to use.")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
def snapshot_update(key: str, prompt: str, provider: str | None, snapshot_dir: str) -> None:
    """Update an existing snapshot with a fresh provider response.

    Overwrites the previous baseline. The old version remains in git history.
    """
    from callspec.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from callspec.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status(
            "[callspec.muted]Calling provider...[/callspec.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(
            "[callspec.fail]Provider call failed:[/callspec.fail] "
            f"{escape(str(call_err))}"
        )
        sys.exit(1)

    entry = manager.update_entry(
        snapshot_key=key,
        content=response.content,
        prompt=prompt,
        model=response.model,
        provider=response.provider,
        tool_calls=response.tool_calls if response.tool_calls else None,
    )
    tool_info = f", {len(entry.tool_calls)} tool calls" if entry.tool_calls else ""
    console.print(
        f"{PASS_MARKER} Snapshot [callspec.key]'{key}'[/callspec.key] updated "
        f"[callspec.muted]({entry.content_length} chars{tool_info}, model={entry.model})[/callspec.muted]"
    )


@snapshot.command("diff")
@click.argument("key")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
@click.option(
    "--prompt",
    default=None,
    help="Prompt to generate fresh output for comparison. Uses stored prompt if omitted.",
)
@click.option("--provider", "-p", default=None, help="Provider to use.")
def snapshot_diff(key: str, snapshot_dir: str, prompt: str | None, provider: str | None) -> None:
    """Show diff between a stored snapshot and current provider output.

    Displays structural and semantic differences between the baseline
    and a fresh response.
    """
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    from callspec.cli.console import console, format_score
    from callspec.errors import SnapshotError
    from callspec.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        baseline_entry = manager.get_entry(key)
    except SnapshotError as snap_err:
        console.print(f"[callspec.fail]Error:[/callspec.fail] {escape(str(snap_err))}")
        sys.exit(1)

    effective_prompt = prompt or baseline_entry.prompt
    if not effective_prompt:
        console.print(
            "[callspec.fail]No prompt specified and snapshot "
            "has no stored prompt.[/callspec.fail] "
            "Provide --prompt to generate fresh output.",
        )
        sys.exit(1)

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    try:
        with console.status(
            "[callspec.muted]Calling provider...[/callspec.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(effective_prompt)
    except Exception as call_err:
        console.print(
            "[callspec.fail]Provider call failed:[/callspec.fail] "
            f"{escape(str(call_err))}"
        )
        sys.exit(1)

    # Structural comparison table
    comparison = Table(border_style="dim", title_style="callspec.header")
    comparison.add_column("", style="callspec.key", width=10)
    comparison.add_column("Baseline", style="callspec.muted")
    comparison.add_column("Current", style="callspec.muted")
    comparison.add_row("Model", baseline_entry.model, response.model)
    comparison.add_row(
        "Length",
        f"{baseline_entry.content_length} chars",
        f"{len(response.content)} chars",
    )

    if baseline_entry.content == response.content:
        comparison.add_row("Content", "[callspec.pass]IDENTICAL[/callspec.pass]", "")
    else:
        comparison.add_row("Content", "[callspec.warn]CHANGED[/callspec.warn]", "")

    console.print(Panel(
        comparison,
        title=f"[callspec.header]Snapshot diff: {key}[/callspec.header]",
        border_style="callspec.header",
        padding=(0, 1),
    ))

    if baseline_entry.content != response.content:
        console.print()
        console.print("[callspec.key]Baseline preview:[/callspec.key]")
        console.print(f"  [callspec.muted]{baseline_entry.content[:200]}[/callspec.muted]")
        console.print("[callspec.key]Current preview:[/callspec.key]")
        console.print(f"  [callspec.muted]{response.content[:200]}[/callspec.muted]")

    # Trajectory diff when baseline has tool-call data
    if baseline_entry.has_trajectory or response.tool_calls:
        from callspec.snapshots.diff import SnapshotDiff

        traj_diff = SnapshotDiff.compare_trajectories(
            snapshot_key=key,
            baseline_calls=baseline_entry.tool_calls,
            current_calls=response.tool_calls,
            baseline_model=baseline_entry.model,
            current_model=response.model,
            baseline_hash=baseline_entry.trajectory_hash,
        )

        console.print()
        if traj_diff.has_changes:
            console.print("[callspec.warn]Trajectory changes:[/callspec.warn]")
            console.print(f"  [callspec.muted]{traj_diff.detailed_report()}[/callspec.muted]")
        else:
            console.print("[callspec.pass]Trajectory unchanged.[/callspec.pass]")




@snapshot.command("delete")
@click.argument("key")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def snapshot_delete(key: str, snapshot_dir: str, yes: bool) -> None:
    """Delete a stored snapshot."""
    from callspec.cli.console import PASS_MARKER, console
    from callspec.errors import SnapshotError
    from callspec.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    if not yes:
        click.confirm(f"Delete snapshot '{key}'?", abort=True)

    try:
        manager.delete_entry(key)
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[callspec.fail]Error:[/callspec.fail] {escape(str(snap_err))}")
        sys.exit(1)

    console.print(f"{PASS_MARKER} Snapshot [callspec.key]'{key}'[/callspec.key] deleted.")


@snapshot.command("list")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
def snapshot_list(snapshot_dir: str) -> None:
    """List all stored snapshots."""
    from rich.table import Table

    from callspec.cli.console import console
    from callspec.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    keys = manager.list_keys()
    if not keys:
        console.print("[callspec.muted]No snapshots found.[/callspec.muted]")
        return

    table = Table(
        title=f"Snapshots ({len(keys)})",
        border_style="dim",
        title_style="callspec.header",
    )
    table.add_column("Key", style="callspec.key")
    table.add_column("Model", style="callspec.muted")
    table.add_column("Length", justify="right")
    table.add_column("Tool Calls", justify="right")

    for snapshot_key in sorted(keys):
        try:
            entry = manager.get_entry(snapshot_key)
            tool_count = str(len(entry.tool_calls)) if entry.tool_calls else "-"
            table.add_row(
                snapshot_key,
                entry.model,
                f"{entry.content_length} chars",
                tool_count,
            )
        except Exception as load_err:
            logger.warning(
                "Failed to load snapshot '%s': %s: %s",
                snapshot_key, type(load_err).__name__, load_err,
            )
            table.add_row(snapshot_key, "[callspec.fail]load error[/callspec.fail]", "", "")

    console.print(table)


def _get_provider(provider_name: str | None):
    """Resolve provider from name or CALLSPEC_PROVIDER env var."""
    import importlib
    import os

    from rich.markup import escape

    from callspec.cli.console import console

    name = provider_name or os.environ.get("CALLSPEC_PROVIDER")

    if not name:
        console.print(
            "[callspec.fail]No provider specified.[/callspec.fail] "
            "Use --provider flag or set CALLSPEC_PROVIDER env var.",
        )
        return None

    name = name.lower().strip()

    if name == "mock":
        from callspec.providers.mock import MockProvider
        return MockProvider(response_fn=lambda prompt, msgs=None: f"mock: {prompt}")

    provider_map = {
        "openai": ("callspec.providers.openai", "OpenAIProvider"),
        "anthropic": ("callspec.providers.anthropic", "AnthropicProvider"),
        "google": ("callspec.providers.google", "GoogleProvider"),
        "mistral": ("callspec.providers.mistral", "MistralProvider"),
        "ollama": ("callspec.providers.ollama", "OllamaProvider"),
        "litellm": ("callspec.providers.litellm", "LiteLLMProvider"),
    }

    if name not in provider_map:
        console.print(
            f"[callspec.fail]Unknown provider '{name}'.[/callspec.fail] "
            f"Available: {', '.join(sorted(provider_map.keys()))}, mock",
        )
        return None

    module_path, class_name = provider_map[name]
    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        return provider_class()
    except ImportError:
        console.print(
            f"[callspec.fail]Provider '{name}' not installed.[/callspec.fail] "
            f"Install with: pip install callspec[{name}]",
        )
        return None
    except Exception as init_err:
        console.print(
            f"[callspec.fail]Failed to initialize "
            f"'{escape(name)}':[/callspec.fail] "
            f"{escape(str(init_err))}"
        )
        return None
