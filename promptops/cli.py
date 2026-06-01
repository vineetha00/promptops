"""Click CLI entrypoint for promptops."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__, ab, diff as diff_mod, eval as eval_mod, rollback as rb_mod, store


console = Console()


@click.group()
@click.version_option(__version__, prog_name="promptops")
def main() -> None:
    """Git-native version control for LLM prompts."""


@main.command("init")
def cmd_init() -> None:
    """Initialize a .promptops/ repo in the current directory."""
    po = store.init()
    console.print(f"[green]initialized[/green] {po}")


@main.command("commit")
@click.argument("name")
@click.argument("text")
@click.option("--tag", default=None, help="Version tag (e.g. v1). Auto-incremented if omitted.")
def cmd_commit(name: str, text: str, tag: str | None) -> None:
    """Commit a new version of a prompt."""
    try:
        v = store.commit(name, text, tag=tag)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    console.print(f"[green]committed[/green] {name}@{v.tag} → {v.path}")


@main.command("log")
@click.argument("name")
def cmd_log(name: str) -> None:
    """Show version history for a prompt."""
    try:
        versions = store.get_versions(name)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)

    table = Table(title=f"history: {name}")
    table.add_column("tag", style="cyan")
    table.add_column("date", style="dim")
    table.add_column("traffic", justify="right")
    table.add_column("alpha/beta", justify="right")
    table.add_column("score", justify="right")
    table.add_column("path", style="dim")

    active = store.load_registry().get(name, {}).get("active")
    for v in versions:
        date = datetime.fromtimestamp(v.timestamp).strftime("%Y-%m-%d %H:%M")
        score = v.scores.get("total", "") if isinstance(v.scores, dict) else ""
        score_str = f"{float(score):.2f}" if score != "" else "-"
        tag = f"[bold green]{v.tag}*[/bold green]" if v.tag == active else v.tag
        table.add_row(tag, date, str(v.traffic),
                      f"{v.alpha:.0f}/{v.beta:.0f}", score_str, v.path)
    console.print(table)


@main.command("diff")
@click.argument("name")
@click.argument("range_spec")
def cmd_diff(name: str, range_spec: str) -> None:
    """Semantic diff between two versions, given as tagA..tagB."""
    if ".." not in range_spec:
        console.print("[red]error:[/red] range must be tagA..tagB")
        sys.exit(1)
    a_tag, b_tag = range_spec.split("..", 1)
    try:
        a = store.read_text(name, a_tag)
        b = store.read_text(name, b_tag)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    res = diff_mod.semantic_diff(a, b)
    res.render(console, a_label=a_tag, b_label=b_tag)


@main.command("ab-test")
@click.argument("name")
@click.argument("tag_a")
@click.argument("tag_b")
@click.option("--metric", default="quality", help="Metric to compare (informational).")
@click.option("--trials", default=100, help="Number of simulated routing trials.")
@click.option("--rate-a", type=float, default=None, help="Override true win rate for tag_a.")
@click.option("--rate-b", type=float, default=None, help="Override true win rate for tag_b.")
@click.option("--seed", type=int, default=42)
def cmd_ab(name: str, tag_a: str, tag_b: str, metric: str, trials: int,
           rate_a: float | None, rate_b: float | None, seed: int) -> None:
    """Run Thompson Sampling A/B between two versions.

    Without an LLM judge, simulated trials use either explicit
    --rate-a/--rate-b or scores derived from the stored eval scores.
    """
    try:
        va = store.get_version(name, tag_a)
        vb = store.get_version(name, tag_b)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)

    def _derived_rate(v: store.Version, default: float) -> float:
        if isinstance(v.scores, dict) and v.scores.get("total") is not None:
            return min(0.99, max(0.01, float(v.scores["total"]) / 5.0))
        return default

    rates = {
        tag_a: rate_a if rate_a is not None else _derived_rate(va, 0.55),
        tag_b: rate_b if rate_b is not None else _derived_rate(vb, 0.60),
    }
    console.print(f"[bold]A/B test[/bold] {name}: {tag_a} vs {tag_b}  metric={metric}")
    console.print(f"  trials={trials}  rates={rates}  seed={seed}")
    result = ab.simulate(name, [tag_a, tag_b], trials, rates, seed=seed)

    table = Table(title="A/B result")
    table.add_column("tag", style="cyan")
    table.add_column("alpha", justify="right")
    table.add_column("beta", justify="right")
    table.add_column("traffic", justify="right")
    for tag, info in result.versions.items():
        table.add_row(tag, f"{info['alpha']:.0f}", f"{info['beta']:.0f}", str(info["traffic"]))
    console.print(table)

    if result.winner:
        console.print(
            f"[green]winner: {result.winner}[/green] "
            f"(p={result.probability:.3f}, samples={result.samples})"
        )
    else:
        console.print(
            f"[yellow]no winner yet[/yellow] "
            f"(top p={result.probability:.3f}, samples={result.samples}, threshold=0.95)"
        )


@main.command("rollback")
@click.argument("name")
@click.option("--to", "to_tag", required=True, help="Tag to roll back to.")
def cmd_rollback(name: str, to_tag: str) -> None:
    """Mark a prior version as the active one."""
    try:
        v = rb_mod.rollback(name, to_tag)
    except KeyError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)
    console.print(f"[green]rolled back[/green] {name} → {v.tag}")


@main.command("status")
def cmd_status() -> None:
    """List all tracked prompts and their active versions."""
    reg = store.load_registry()
    if not reg:
        console.print("[dim]no prompts tracked yet[/dim]")
        return
    table = Table(title="promptops status")
    table.add_column("prompt", style="cyan")
    table.add_column("active", style="green")
    table.add_column("versions", justify="right")
    table.add_column("total traffic", justify="right")
    for name, entry in reg.items():
        versions = entry.get("versions", [])
        traffic = sum(v.get("traffic", 0) for v in versions)
        table.add_row(name, entry.get("active") or "-", str(len(versions)), str(traffic))
    console.print(table)


@main.command("eval")
@click.option("--compare", "compare_spec", default=None,
              help="Compare two refs, e.g. main..HEAD")
@click.option("--prompt", "prompt_name", default=None,
              help="Evaluate a single prompt by name.")
@click.option("--fail-on-regression", is_flag=True,
              help="Exit code 1 if quality drops >10%.")
@click.option("--threshold", type=float, default=0.10,
              help="Regression threshold as a fraction (default 0.10).")
def cmd_eval(compare_spec: str | None, prompt_name: str | None,
             fail_on_regression: bool, threshold: float) -> None:
    """Run LLM-as-judge eval and optionally fail on regression."""
    if prompt_name:
        report = eval_mod.evaluate_prompt(prompt_name)
    else:
        base_ref, head_ref = "main", "HEAD"
        if compare_spec and ".." in compare_spec:
            base_ref, head_ref = compare_spec.split("..", 1)
        report = eval_mod.compare_branches(base_ref, head_ref,
                                           regression_threshold=threshold)

    table = Table(title="eval scores")
    table.add_column("version", style="cyan")
    table.add_column("quality", justify="right")
    table.add_column("relevance", justify="right")
    table.add_column("safety", justify="right")
    table.add_column("total", justify="right")
    for tag, score in report.versions.items():
        table.add_row(
            tag,
            f"{score.quality:.2f}",
            f"{score.relevance:.2f}",
            f"{score.safety:.2f}",
            f"{score.total:.2f}",
        )
    console.print(table)

    if report.regressions:
        console.print("[red]regressions detected:[/red]")
        for a, b, drop in report.regressions:
            console.print(f"  {a} → {b}: -{drop*100:.1f}%")
        if fail_on_regression:
            sys.exit(1)
    else:
        console.print("[green]no regressions[/green]")


@main.command("serve")
@click.option("--port", default=3333, help="(Informational — MCP uses stdio transport.)")
def cmd_serve(port: int) -> None:
    """Start the MCP server (stdio transport)."""
    from . import mcp_server

    console.print(f"[green]promptops MCP server[/green] starting (stdio, port hint={port})")
    mcp_server.run()


if __name__ == "__main__":
    main()
