"""verdict snapshot: manage baseline snapshots for regression assertions.

Subcommands: create, update, diff, delete, list.
Snapshots are versioned JSON files that live in the project repository
under verdict_snapshots/baselines.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

DEFAULT_SNAPSHOT_DIR = "verdict_snapshots"


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
    help="Provider to use. Falls back to VERDICT_PROVIDER env var.",
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
    from verdict.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from verdict.errors import SnapshotError
    from verdict.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status("[verdict.muted]Calling provider...[/verdict.muted]", spinner="dots"):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(f"[verdict.fail]Provider call failed:[/verdict.fail] {escape(str(call_err))}")
        sys.exit(1)

    try:
        entry = manager.create_entry(
            snapshot_key=key,
            content=response.content,
            prompt=prompt,
            model=response.model,
            provider=response.provider,
        )
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[verdict.fail]Error:[/verdict.fail] {escape(str(snap_err))}")
        sys.exit(1)

    console.print(
        f"{PASS_MARKER} Snapshot [verdict.key]'{key}'[/verdict.key] created "
        f"[verdict.muted]({entry.content_length} chars, model={entry.model})[/verdict.muted]"
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
    from verdict.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from verdict.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status("[verdict.muted]Calling provider...[/verdict.muted]", spinner="dots"):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(f"[verdict.fail]Provider call failed:[/verdict.fail] {escape(str(call_err))}")
        sys.exit(1)

    entry = manager.update_entry(
        snapshot_key=key,
        content=response.content,
        prompt=prompt,
        model=response.model,
        provider=response.provider,
    )
    console.print(
        f"{PASS_MARKER} Snapshot [verdict.key]'{key}'[/verdict.key] updated "
        f"[verdict.muted]({entry.content_length} chars, model={entry.model})[/verdict.muted]"
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

    from verdict.cli.console import console, format_score
    from verdict.errors import SnapshotError
    from verdict.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        baseline_entry = manager.get_entry(key)
    except SnapshotError as snap_err:
        console.print(f"[verdict.fail]Error:[/verdict.fail] {escape(str(snap_err))}")
        sys.exit(1)

    effective_prompt = prompt or baseline_entry.prompt
    if not effective_prompt:
        console.print(
            "[verdict.fail]No prompt specified and snapshot has no stored prompt.[/verdict.fail] "
            "Provide --prompt to generate fresh output.",
        )
        sys.exit(1)

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    try:
        with console.status("[verdict.muted]Calling provider...[/verdict.muted]", spinner="dots"):
            response = resolved_provider.call(effective_prompt)
    except Exception as call_err:
        console.print(f"[verdict.fail]Provider call failed:[/verdict.fail] {escape(str(call_err))}")
        sys.exit(1)

    # Structural comparison table
    comparison = Table(border_style="dim", title_style="verdict.header")
    comparison.add_column("", style="verdict.key", width=10)
    comparison.add_column("Baseline", style="verdict.muted")
    comparison.add_column("Current", style="verdict.muted")
    comparison.add_row("Model", baseline_entry.model, response.model)
    comparison.add_row(
        "Length",
        f"{baseline_entry.content_length} chars",
        f"{len(response.content)} chars",
    )

    if baseline_entry.content == response.content:
        comparison.add_row("Content", "[verdict.pass]IDENTICAL[/verdict.pass]", "")
    else:
        comparison.add_row("Content", "[verdict.warn]CHANGED[/verdict.warn]", "")

    console.print(Panel(
        comparison,
        title=f"[verdict.header]Snapshot diff: {key}[/verdict.header]",
        border_style="verdict.header",
        padding=(0, 1),
    ))

    if baseline_entry.content != response.content:
        console.print()
        console.print("[verdict.key]Baseline preview:[/verdict.key]")
        console.print(f"  [verdict.muted]{baseline_entry.content[:200]}[/verdict.muted]")
        console.print("[verdict.key]Current preview:[/verdict.key]")
        console.print(f"  [verdict.muted]{response.content[:200]}[/verdict.muted]")

    # Semantic comparison via embedding scorer if available
    try:
        from verdict.scoring.embeddings import EmbeddingScorer

        scorer = EmbeddingScorer()
        similarity = scorer.score(baseline_entry.content, response.content)
        drift = 1.0 - similarity

        console.print()
        console.print(
            f"  Semantic similarity: {format_score(similarity, 0.85)}"
        )
        drift_style = "verdict.score.good" if drift < 0.15 else "verdict.score.bad"
        console.print(
            f"  Semantic drift:      [{drift_style}]{drift:.4f}[/{drift_style}]"
        )
    except ImportError:
        console.print(
            "\n[verdict.muted]Semantic comparison unavailable "
            "(install verdict[semantic])[/verdict.muted]"
        )


@snapshot.command("delete")
@click.argument("key")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def snapshot_delete(key: str, snapshot_dir: str, yes: bool) -> None:
    """Delete a stored snapshot."""
    from verdict.cli.console import PASS_MARKER, console
    from verdict.errors import SnapshotError
    from verdict.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    if not yes:
        click.confirm(f"Delete snapshot '{key}'?", abort=True)

    try:
        manager.delete_entry(key)
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[verdict.fail]Error:[/verdict.fail] {escape(str(snap_err))}")
        sys.exit(1)

    console.print(f"{PASS_MARKER} Snapshot [verdict.key]'{key}'[/verdict.key] deleted.")


@snapshot.command("list")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
def snapshot_list(snapshot_dir: str) -> None:
    """List all stored snapshots."""
    from rich.table import Table

    from verdict.cli.console import console
    from verdict.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    keys = manager.list_keys()
    if not keys:
        console.print("[verdict.muted]No snapshots found.[/verdict.muted]")
        return

    table = Table(
        title=f"Snapshots ({len(keys)})",
        border_style="dim",
        title_style="verdict.header",
    )
    table.add_column("Key", style="verdict.key")
    table.add_column("Model", style="verdict.muted")
    table.add_column("Length", justify="right")

    for snapshot_key in sorted(keys):
        try:
            entry = manager.get_entry(snapshot_key)
            table.add_row(snapshot_key, entry.model, f"{entry.content_length} chars")
        except Exception:
            table.add_row(snapshot_key, "[verdict.fail]load error[/verdict.fail]", "")

    console.print(table)


def _get_provider(provider_name: str | None):
    """Resolve provider from name or VERDICT_PROVIDER env var."""
    import importlib
    import os

    from rich.markup import escape

    from verdict.cli.console import console

    name = provider_name or os.environ.get("VERDICT_PROVIDER")

    if not name:
        console.print(
            "[verdict.fail]No provider specified.[/verdict.fail] "
            "Use --provider flag or set VERDICT_PROVIDER env var.",
        )
        return None

    name = name.lower().strip()

    if name == "mock":
        from verdict.providers.mock import MockProvider
        return MockProvider(response_fn=lambda prompt, msgs=None: f"mock: {prompt}")

    provider_map = {
        "openai": ("verdict.providers.openai", "OpenAIProvider"),
        "anthropic": ("verdict.providers.anthropic", "AnthropicProvider"),
        "google": ("verdict.providers.google", "GoogleProvider"),
        "mistral": ("verdict.providers.mistral", "MistralProvider"),
        "ollama": ("verdict.providers.ollama", "OllamaProvider"),
        "litellm": ("verdict.providers.litellm", "LiteLLMProvider"),
    }

    if name not in provider_map:
        console.print(
            f"[verdict.fail]Unknown provider '{name}'.[/verdict.fail] "
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
            f"[verdict.fail]Provider '{name}' not installed.[/verdict.fail] "
            f"Install with: pip install verdict[{name}]",
        )
        return None
    except Exception as init_err:
        console.print(
            f"[verdict.fail]Failed to initialize "
            f"'{escape(name)}':[/verdict.fail] "
            f"{escape(str(init_err))}"
        )
        return None
