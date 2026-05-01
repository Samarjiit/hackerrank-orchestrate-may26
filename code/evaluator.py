from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

SAMPLE_CSV_COLS = {
    "input": ["Issue", "Subject", "Company"],
    "output": ["Response", "Product Area", "Status", "Request Type"],
}

PRED_COLS = ["status", "product_area", "response", "justification", "request_type"]


def load_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _normalize(val: str) -> str:
    return str(val).strip().lower()


def evaluate(sample_path: Path, pred_path: Path) -> dict:
    """Compare predictions against sample_support_tickets.csv ground truth."""
    sample = load_sample(sample_path)
    preds = load_predictions(pred_path)

    # Map sample columns to normalized names
    sample = sample.rename(
        columns={
            "Status": "status",
            "Product Area": "product_area",
            "Request Type": "request_type",
            "Response": "response",
        }
    )
    sample["status"] = sample["status"].str.strip().str.lower()
    sample["request_type"] = sample["request_type"].str.strip().str.lower()

    n = min(len(sample), len(preds))
    if n == 0:
        console.print("[red]No rows to evaluate.[/red]")
        return {}

    sample = sample.iloc[:n].reset_index(drop=True)
    preds = preds.iloc[:n].reset_index(drop=True)

    # ── Per-column accuracy ────────────────────────────────────────────────────
    results: dict = {}

    for col in ["status", "request_type"]:
        if col in preds.columns and col in sample.columns:
            correct = sum(
                _normalize(preds[col].iloc[i]) == _normalize(sample[col].iloc[i])
                for i in range(n)
            )
            results[col] = {"correct": correct, "total": n, "accuracy": correct / n}

    # ── Confusion matrix for status ───────────────────────────────────────────
    if "status" in preds.columns and "status" in sample.columns:
        tp = tn = fp = fn = 0
        for i in range(n):
            pred_s = _normalize(preds["status"].iloc[i])
            true_s = _normalize(sample["status"].iloc[i])
            if true_s == "escalated":
                if pred_s == "escalated":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_s == "escalated":
                    fp += 1
                else:
                    tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results["escalation_confusion"] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    results["n"] = n
    _print_results(results, sample, preds, n)
    return results


def _print_results(results: dict, sample: pd.DataFrame, preds: pd.DataFrame, n: int) -> None:
    console.rule("[bold cyan]Evaluation Report[/bold cyan]")

    # Accuracy table
    table = Table(title="Classification Accuracy", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy", style="green")

    for col in ["status", "request_type"]:
        if col in results:
            r = results[col]
            acc_pct = f"{r['accuracy'] * 100:.1f}%"
            table.add_row(col, str(r["correct"]), str(r["total"]), acc_pct)

    console.print(table)

    # Escalation confusion
    if "escalation_confusion" in results:
        ec = results["escalation_confusion"]
        cm = Table(title="Escalation Confusion Matrix", box=box.SIMPLE)
        cm.add_column("")
        cm.add_column("Pred: escalated")
        cm.add_column("Pred: replied")
        cm.add_row("True: escalated", f"[green]TP={ec['tp']}[/green]", f"[red]FN={ec['fn']}[/red]")
        cm.add_row("True: replied", f"[yellow]FP={ec['fp']}[/yellow]", f"[green]TN={ec['tn']}[/green]")
        console.print(cm)
        console.print(
            f"Escalation  Precision: [cyan]{ec['precision']}[/cyan]  "
            f"Recall: [cyan]{ec['recall']}[/cyan]  "
            f"F1: [cyan]{ec['f1']}[/cyan]"
        )

    # Row-level diff
    if "status" in preds.columns and "status" in sample.columns:
        console.rule("Row-level status diff")
        diff_table = Table(box=box.MINIMAL)
        diff_table.add_column("#", style="dim")
        diff_table.add_column("Issue (truncated)", max_width=50)
        diff_table.add_column("True status")
        diff_table.add_column("Pred status")
        diff_table.add_column("Match")

        for i in range(n):
            true_s = _normalize(sample["status"].iloc[i])
            pred_s = _normalize(preds["status"].iloc[i])
            match = "✓" if true_s == pred_s else "✗"
            color = "green" if match == "✓" else "red"
            issue_col = sample.get("Issue", sample.get("issue", pd.Series()))
            issue_text = str(issue_col.iloc[i])[:50] if len(issue_col) > i else ""
            diff_table.add_row(
                str(i + 1), issue_text, true_s, pred_s, f"[{color}]{match}[/{color}]"
            )
        console.print(diff_table)
