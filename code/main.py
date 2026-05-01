"""
Multi-Domain Support Triage Agent — CLI entry point.

Usage examples:
  python main.py --ingest
  python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv
  python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv --debug
  python main.py --ticket "I lost my Visa card" --company Visa
  python main.py --evaluate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

# Ensure code/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

import config
from agent import TriageAgent
from ingestion import index_corpus
from retriever import Retriever, build_retriever
from schemas import TicketInput, TicketOutput
from evaluator import evaluate

console = Console()

BANNER = """[bold cyan]
╔══════════════════════════════════════════════════════════╗
║       Multi-Domain Support Triage Agent  v1.0            ║
║       HackerRank  •  Claude  •  Visa                     ║
╚══════════════════════════════════════════════════════════╝
[/bold cyan]"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    return "green" if status == "replied" else "yellow"


def _request_type_color(rt: str) -> str:
    colors = {
        "product_issue": "cyan",
        "feature_request": "blue",
        "bug": "red",
        "invalid": "dim",
    }
    return colors.get(rt, "white")


def _print_result(idx: int, ticket: TicketInput, output: TicketOutput, debug: bool = False) -> None:
    sc = _status_color(output.status)
    rc = _request_type_color(output.request_type)

    issue_preview = ticket.issue.strip().replace("\n", " ")[:100]
    title = f"[bold]#{idx}[/bold]  [{sc}]{output.status.upper()}[/{sc}]  [{rc}]{output.request_type}[/{rc}]  [dim]area: {output.product_area}[/dim]"
    body_lines = [
        f"[bold]Issue:[/bold] {issue_preview}{'…' if len(ticket.issue) > 100 else ''}",
        f"[bold]Response:[/bold] {output.response[:200]}{'…' if len(output.response) > 200 else ''}",
    ]
    if debug:
        body_lines.append(f"[dim][bold]Justification:[/bold] {output.justification}[/dim]")

    console.print(Panel("\n".join(body_lines), title=title, border_style=sc, expand=False))


def _print_summary(total: int, replied: int, escalated: int, elapsed: float) -> None:
    table = Table(title="Run Summary", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total tickets", str(total))
    table.add_row("Replied", f"[green]{replied}[/green]")
    table.add_row("Escalated", f"[yellow]{escalated}[/yellow]")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    table.add_row("Avg / ticket", f"{elapsed / max(total, 1):.1f}s")
    console.print(table)


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_batch(
    input_path: Path,
    output_path: Path,
    debug: bool = False,
    force_reindex: bool = False,
) -> None:
    console.print(BANNER)
    console.print(f"[dim]Input:[/dim]  {input_path}")
    console.print(f"[dim]Output:[/dim] {output_path}")
    console.print(f"[dim]LLM:[/dim]    {config.LLM_PROVIDER} / {config.ANTHROPIC_MODEL if config.LLM_PROVIDER == 'anthropic' else config.OPENAI_MODEL}")
    console.print()

    # Load tickets
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]
    tickets = [
        TicketInput(
            issue=str(row.get("Issue", row.get("issue", ""))),
            subject=str(row.get("Subject", row.get("subject", ""))),
            company=str(row.get("Company", row.get("company", "None"))),
        )
        for _, row in df.iterrows()
    ]
    console.print(f"[cyan]Loaded {len(tickets)} tickets[/cyan]")

    # Build retriever (indexes corpus if needed)
    console.print("[dim]Initialising retriever …[/dim]")
    retriever = build_retriever(force_reindex=force_reindex)
    console.print(f"[dim]Collection has {retriever._collection.count()} chunks[/dim]\n")

    agent = TriageAgent(retriever=retriever, debug=debug)

    results: list[dict] = []
    t0 = time.perf_counter()
    replied = escalated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Triaging tickets …", total=len(tickets))

        for idx, ticket in enumerate(tickets, 1):
            progress.update(task, description=f"[cyan]Ticket {idx}/{len(tickets)}[/cyan]")

            triage = agent.triage(ticket)
            out = triage.output

            if out.status == "replied":
                replied += 1
            else:
                escalated += 1

            results.append(
                {
                    "issue": ticket.issue,
                    "subject": ticket.subject,
                    "company": ticket.company,
                    "status": out.status,
                    "product_area": out.product_area,
                    "response": out.response,
                    "justification": out.justification,
                    "request_type": out.request_type,
                }
            )

            if debug:
                _print_result(idx, ticket, out, debug=True)

            progress.advance(task)

    elapsed = time.perf_counter() - t0

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)[["status", "product_area", "response", "justification", "request_type"]]
    out_df.to_csv(output_path, index=False)
    console.print(f"\n[bold green]✓ Output written to {output_path}[/bold green]")

    _print_summary(len(tickets), replied, escalated, elapsed)


def run_single(issue: str, company: str = "None", subject: str = "", debug: bool = False) -> None:
    console.print(BANNER)
    retriever = build_retriever()
    agent = TriageAgent(retriever=retriever, debug=debug)
    ticket = TicketInput(issue=issue, subject=subject, company=company)

    console.print(Panel(f"[bold]Issue:[/bold] {issue}\n[bold]Company:[/bold] {company}", title="Input Ticket"))

    with console.status("Triaging …"):
        triage = agent.triage(ticket)

    out = triage.output
    sc = _status_color(out.status)

    table = Table(title="Triage Result", box=box.ROUNDED, show_header=False)
    table.add_column("Field", style="bold", min_width=16)
    table.add_column("Value")
    table.add_row("Status", f"[{sc}]{out.status}[/{sc}]")
    table.add_row("Request Type", out.request_type)
    table.add_row("Product Area", out.product_area)
    table.add_row("Response", out.response)
    table.add_row("Justification", f"[dim]{out.justification}[/dim]")

    if debug and triage.retrieved_docs:
        table.add_row("─" * 16, "─" * 40)
        for i, doc in enumerate(triage.retrieved_docs, 1):
            table.add_row(
                f"Doc {i} ({doc.score:.2f})",
                f"[dim]{doc.title[:80]} [{doc.company}/{doc.category}][/dim]",
            )

    console.print(table)


def run_ingest(force: bool = False) -> None:
    console.print(BANNER)
    console.print("[cyan]Running corpus ingestion …[/cyan]")
    collection = index_corpus(force=force)
    console.print(f"[bold green]✓ Corpus indexed. {collection.count()} chunks in ChromaDB.[/bold green]")


def run_evaluate(
    sample_path: Path,
    pred_path: Path,
) -> None:
    console.print(BANNER)
    if not pred_path.exists():
        console.print(f"[red]Predictions file not found: {pred_path}[/red]")
        sys.exit(1)
    evaluate(sample_path, pred_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ingest", action="store_true", help="Index the support corpus into ChromaDB")
    mode.add_argument("--ticket", type=str, metavar="ISSUE", help="Triage a single ticket (interactive)")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate predictions against sample CSV")

    p.add_argument("--input", type=Path, default=config.SUPPORT_TICKETS_DIR / "support_tickets.csv",
                   help="Input CSV with Issue/Subject/Company columns")
    p.add_argument("--output", type=Path, default=config.SUPPORT_TICKETS_DIR / "output.csv",
                   help="Output CSV path")
    p.add_argument("--sample", type=Path, default=config.SUPPORT_TICKETS_DIR / "sample_support_tickets.csv",
                   help="Sample CSV for evaluation")
    p.add_argument("--company", type=str, default="None", help="Company for --ticket mode")
    p.add_argument("--subject", type=str, default="", help="Subject for --ticket mode")
    p.add_argument("--force-reindex", action="store_true", help="Force re-ingestion of corpus")
    p.add_argument("--debug", action="store_true", help="Show detailed retrieval and LLM reasoning")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.ingest:
        run_ingest(force=args.force_reindex)

    elif args.ticket:
        run_single(
            issue=args.ticket,
            company=args.company,
            subject=args.subject,
            debug=args.debug,
        )

    elif args.evaluate:
        run_evaluate(
            sample_path=args.sample,
            pred_path=args.output,
        )

    else:
        # Default: batch mode
        run_batch(
            input_path=args.input,
            output_path=args.output,
            debug=args.debug,
            force_reindex=args.force_reindex,
        )


if __name__ == "__main__":
    main()
