"""Writes the Markdown tables of the README from the Stage 5 result tables.

Usage (from the repository root):
    python scripts/make_tables.py [results_dir]
Prints the tables. Every number comes from results/*.csv. No number is typed in this script.
"""
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 and __name__ == "__main__" else Path("results")


def load(results_dir=RESULTS):
    return (pd.read_csv(Path(results_dir) / "final_results.csv"),
            pd.read_csv(Path(results_dir) / "paired_differences.csv"),
            pd.read_csv(Path(results_dir) / "per_event_test.csv"),
            pd.read_csv(Path(results_dir) / "difficult_chips.csv"))


def results_table(final, split):
    """Markdown table: method, seeds, IoU, 95% interval, seed range."""
    lines = ["| Method | Seeds | Water IoU | 95% interval | Seed range |", "|---|---|---|---|---|"]
    for _, r in final.iterrows():
        seed_range = "-" if r["seeds"] == 1 else f"{r[f'{split}_iou_seed_min']:.3f} to {r[f'{split}_iou_seed_max']:.3f}"
        lines.append(f"| {r['method']} | {int(r['seeds'])} | {r[f'{split}_iou_mean']:.3f} | "
                     f"{r[f'{split}_iou_ci_low']:.3f} to {r[f'{split}_iou_ci_high']:.3f} | {seed_range} |")
    return "\n".join(lines)


def paired_table(paired):
    """Markdown table of the paired differences (A minus B) with the 95% interval and the verdict."""
    lines = ["| Comparison (A against B) | Split | A minus B | 95% interval | Result |", "|---|---|---|---|---|"]
    for _, r in paired.iterrows():
        verdict = {"A better": "A better", "B better": "B better"}.get(r["verdict"], "not clear")
        lines.append(f"| {r['comparison']} | {r['split']} | {r['diff_A_minus_B']:+.3f} | "
                     f"{r['ci_low']:+.3f} to {r['ci_high']:+.3f} | {verdict} |")
    return "\n".join(lines)


if __name__ == "__main__":
    final, paired, per_event, difficult = load()
    print("TEST\n" + results_table(final, "test"))
    print("\nBOLIVIA\n" + results_table(final, "bolivia"))
    print("\nPAIRED\n" + paired_table(paired))
