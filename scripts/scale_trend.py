#!/usr/bin/env python3
"""Is there a monotone trend in the agent-count sweep, and in which metric?

The manuscript claimed accuracy improves with population size ``as reputation
gains statistical power to identify malicious actors''. That is two claims: a
trend in accuracy, and a mechanism. Both are testable on the sweep, and the
mechanism claim is testable directly, because the runner records the malicious
rejection rate per seed. Testing them was overdue: the accuracy column they were
written against was an interpolation from the ten-agent run.

The test is Jonckheere--Terpstra, which is the ordered-alternative analogue of
Kruskal--Wallis: it asks whether values tend to rise across groups placed in a
known order, rather than merely whether the groups differ. That is the right
instrument here because agent count is ordered and the alternative of interest is
monotone. Ties count as half, and the normal approximation carries the tie
correction for the group-size term only, which is adequate at five seeds per cell
and exact ties being unlikely in floating-point metrics.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import argparse
import csv
import json
import math

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "auc_roc",
    "sensitivity",
    "specificity",
    "malicious_rejection_rate",
    "honest_false_reject_rate",
)


def jonckheere(groups: list[list[float]]) -> tuple[float, float, float]:
    """JT statistic, its normal deviate, and a two-sided p-value."""
    u = 0.0
    for a, b in combinations(range(len(groups)), 2):
        for x in groups[a]:
            for y in groups[b]:
                u += 1.0 if y > x else (0.5 if y == x else 0.0)
    n = sum(len(g) for g in groups)
    squares = sum(len(g) ** 2 for g in groups)
    mean = (n * n - squares) / 4.0
    var = (n * n * (2 * n + 3) - sum(len(g) ** 2 * (2 * len(g) + 3) for g in groups)) / 72.0
    z = (u - mean) / math.sqrt(var)
    return u, z, math.erfc(abs(z) / math.sqrt(2.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", action="append", required=True, metavar="N:CSV",
        help="agent count and its ablation_runs.csv, e.g. 10:results/scale-scaled/n10/ablation_runs.csv")
    parser.add_argument("--output", default="results/scale-scaled/trend_manifest.json")
    args = parser.parse_args()

    cells: dict[int, list[dict[str, str]]] = {}
    for spec in args.runs:
        count, _, path = spec.partition(":")
        with Path(path).open(newline="", encoding="utf-8") as handle:
            cells[int(count)] = list(csv.DictReader(handle))
    order = sorted(cells)

    results = {}
    for metric in METRICS:
        groups = [[float(row[metric]) for row in cells[n]] for n in order]
        u, z, p = jonckheere(groups)
        results[metric] = {
            "statistic": u,
            "z": z,
            "p_value": p,
            "means": {str(n): sum(g) / len(g) for n, g in zip(order, groups)},
            "increasing": z > 0,
        }
        arrow = "increasing" if z > 0 else "decreasing"
        flag = "*" if p < 0.05 else " "
        print(f"{metric:26s} z={z:+5.2f} p={p:.4f}{flag} {arrow}")

    manifest = {
        "agent_counts": order,
        "seeds_per_cell": {str(n): len(cells[n]) for n in order},
        "test": "Jonckheere-Terpstra, ordered alternative, two-sided normal approximation",
        "metrics": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
