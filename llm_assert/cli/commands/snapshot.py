"""llm-assert snapshot: manage baseline snapshots for regression assertions.

Subcommands: create, update, diff, delete, list.
Snapshots are versioned JSON files that live in the project repository
under llm_assert_snapshots/baselines.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

DEFAULT_SNAPSHOT_DIR = "llm_assert_snapshots"


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
    help="Provider to use. Falls back to LLM_ASSERT_PROVIDER env var.",
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
    from llm_assert.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from llm_assert.errors import SnapshotError
    from llm_assert.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status(
            "[llm_assert.muted]Calling provider...[/llm_assert.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(
            "[llm_assert.fail]Provider call failed:[/llm_assert.fail] "
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
        )
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[llm_assert.fail]Error:[/llm_assert.fail] {escape(str(snap_err))}")
        sys.exit(1)

    console.print(
        f"{PASS_MARKER} Snapshot [llm_assert.key]'{key}'[/llm_assert.key] created "
        f"[llm_assert.muted]({entry.content_length} chars, model={entry.model})[/llm_assert.muted]"
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
    from llm_assert.cli.console import PASS_MARKER, console

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    from llm_assert.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        with console.status(
            "[llm_assert.muted]Calling provider...[/llm_assert.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(prompt)
    except Exception as call_err:
        from rich.markup import escape
        console.print(
            "[llm_assert.fail]Provider call failed:[/llm_assert.fail] "
            f"{escape(str(call_err))}"
        )
        sys.exit(1)

    entry = manager.update_entry(
        snapshot_key=key,
        content=response.content,
        prompt=prompt,
        model=response.model,
        provider=response.provider,
    )
    console.print(
        f"{PASS_MARKER} Snapshot [llm_assert.key]'{key}'[/llm_assert.key] updated "
        f"[llm_assert.muted]({entry.content_length} chars, model={entry.model})[/llm_assert.muted]"
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

    from llm_assert.cli.console import console, format_score
    from llm_assert.errors import SnapshotError
    from llm_assert.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    try:
        baseline_entry = manager.get_entry(key)
    except SnapshotError as snap_err:
        console.print(f"[llm_assert.fail]Error:[/llm_assert.fail] {escape(str(snap_err))}")
        sys.exit(1)

    effective_prompt = prompt or baseline_entry.prompt
    if not effective_prompt:
        console.print(
            "[llm_assert.fail]No prompt specified and snapshot "
            "has no stored prompt.[/llm_assert.fail] "
            "Provide --prompt to generate fresh output.",
        )
        sys.exit(1)

    resolved_provider = _get_provider(provider)
    if resolved_provider is None:
        sys.exit(1)

    try:
        with console.status(
            "[llm_assert.muted]Calling provider...[/llm_assert.muted]",
            spinner="dots",
        ):
            response = resolved_provider.call(effective_prompt)
    except Exception as call_err:
        console.print(
            "[llm_assert.fail]Provider call failed:[/llm_assert.fail] "
            f"{escape(str(call_err))}"
        )
        sys.exit(1)

    # Structural comparison table
    comparison = Table(border_style="dim", title_style="llm_assert.header")
    comparison.add_column("", style="llm_assert.key", width=10)
    comparison.add_column("Baseline", style="llm_assert.muted")
    comparison.add_column("Current", style="llm_assert.muted")
    comparison.add_row("Model", baseline_entry.model, response.model)
    comparison.add_row(
        "Length",
        f"{baseline_entry.content_length} chars",
        f"{len(response.content)} chars",
    )

    if baseline_entry.content == response.content:
        comparison.add_row("Content", "[llm_assert.pass]IDENTICAL[/llm_assert.pass]", "")
    else:
        comparison.add_row("Content", "[llm_assert.warn]CHANGED[/llm_assert.warn]", "")

    console.print(Panel(
        comparison,
        title=f"[llm_assert.header]Snapshot diff: {key}[/llm_assert.header]",
        border_style="llm_assert.header",
        padding=(0, 1),
    ))

    if baseline_entry.content != response.content:
        console.print()
        console.print("[llm_assert.key]Baseline preview:[/llm_assert.key]")
        console.print(f"  [llm_assert.muted]{baseline_entry.content[:200]}[/llm_assert.muted]")
        console.print("[llm_assert.key]Current preview:[/llm_assert.key]")
        console.print(f"  [llm_assert.muted]{response.content[:200]}[/llm_assert.muted]")




@snapshot.command("delete")
@click.argument("key")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def snapshot_delete(key: str, snapshot_dir: str, yes: bool) -> None:
    """Delete a stored snapshot."""
    from llm_assert.cli.console import PASS_MARKER, console
    from llm_assert.errors import SnapshotError
    from llm_assert.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    if not yes:
        click.confirm(f"Delete snapshot '{key}'?", abort=True)

    try:
        manager.delete_entry(key)
    except SnapshotError as snap_err:
        from rich.markup import escape
        console.print(f"[llm_assert.fail]Error:[/llm_assert.fail] {escape(str(snap_err))}")
        sys.exit(1)

    console.print(f"{PASS_MARKER} Snapshot [llm_assert.key]'{key}'[/llm_assert.key] deleted.")


@snapshot.command("list")
@click.option("--snapshot-dir", "-d", default=DEFAULT_SNAPSHOT_DIR)
def snapshot_list(snapshot_dir: str) -> None:
    """List all stored snapshots."""
    from rich.table import Table

    from llm_assert.cli.console import console
    from llm_assert.snapshots.manager import SnapshotManager

    manager = SnapshotManager(snapshot_dir=Path(snapshot_dir))

    keys = manager.list_keys()
    if not keys:
        console.print("[llm_assert.muted]No snapshots found.[/llm_assert.muted]")
        return

    table = Table(
        title=f"Snapshots ({len(keys)})",
        border_style="dim",
        title_style="llm_assert.header",
    )
    table.add_column("Key", style="llm_assert.key")
    table.add_column("Model", style="llm_assert.muted")
    table.add_column("Length", justify="right")

    for snapshot_key in sorted(keys):
        try:
            entry = manager.get_entry(snapshot_key)
            table.add_row(snapshot_key, entry.model, f"{entry.content_length} chars")
        except Exception:
            table.add_row(snapshot_key, "[llm_assert.fail]load error[/llm_assert.fail]", "")

    console.print(table)


def _get_provider(provider_name: str | None):
    """Resolve provider from name or LLM_ASSERT_PROVIDER env var."""
    import importlib
    import os

    from rich.markup import escape

    from llm_assert.cli.console import console

    name = provider_name or os.environ.get("LLM_ASSERT_PROVIDER")

    if not name:
        console.print(
            "[llm_assert.fail]No provider specified.[/llm_assert.fail] "
            "Use --provider flag or set LLM_ASSERT_PROVIDER env var.",
        )
        return None

    name = name.lower().strip()

    if name == "mock":
        from llm_assert.providers.mock import MockProvider
        return MockProvider(response_fn=lambda prompt, msgs=None: f"mock: {prompt}")

    provider_map = {
        "openai": ("llm_assert.providers.openai", "OpenAIProvider"),
        "anthropic": ("llm_assert.providers.anthropic", "AnthropicProvider"),
        "google": ("llm_assert.providers.google", "GoogleProvider"),
        "mistral": ("llm_assert.providers.mistral", "MistralProvider"),
        "ollama": ("llm_assert.providers.ollama", "OllamaProvider"),
        "litellm": ("llm_assert.providers.litellm", "LiteLLMProvider"),
    }

    if name not in provider_map:
        console.print(
            f"[llm_assert.fail]Unknown provider '{name}'.[/llm_assert.fail] "
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
            f"[llm_assert.fail]Provider '{name}' not installed.[/llm_assert.fail] "
            f"Install with: pip install llm-assert[{name}]",
        )
        return None
    except Exception as init_err:
        console.print(
            f"[llm_assert.fail]Failed to initialize "
            f"'{escape(name)}':[/llm_assert.fail] "
            f"{escape(str(init_err))}"
        )
        return None
