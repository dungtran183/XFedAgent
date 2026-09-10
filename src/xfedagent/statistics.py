"""Seed-level statistics for paired comparisons between two arms.

The statistical unit is the *seed*, not the round: 100 rounds of one seed share a
data partition, an initialisation and an attack schedule, so they are not 100
independent observations. Treating them as such shrinks every interval by roughly
a factor of ten.

Two arms that share seeds are *paired*, so the comparison is a Wilcoxon
signed-rank test over per-seed differences rather than a rank-sum test over two
independent groups. Multiplicity across a family of comparisons is controlled by
Holm, which is uniformly at least as powerful as Bonferroni at the same
family-wise error rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedComparison:
    """One paired comparison between two arms over a shared set of seeds."""

    metric: str
    arm_a: str
    arm_b: str
    seeds: tuple[int, ...]
    values_a: tuple[float, ...]
    values_b: tuple[float, ...]

    @property
    def differences(self) -> np.ndarray:
        return np.asarray(self.values_a, dtype=float) - np.asarray(self.values_b, dtype=float)

    def summary(self) -> dict:
        a = np.asarray(self.values_a, dtype=float)
        b = np.asarray(self.values_b, dtype=float)
        d = self.differences
        n = d.size

        mean_d = float(np.mean(d))
        sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
        # Paired-difference CI on the mean, at the t quantile for n-1 df.
        if n > 1 and sd_d > 0.0:
            half = float(stats.t.ppf(0.975, n - 1)) * sd_d / math.sqrt(n)
        else:
            half = 0.0

        # Wilcoxon is undefined when every difference is zero; report that plainly
        # rather than letting scipy raise.
        if n < 1 or np.allclose(d, 0.0):
            stat, p = float("nan"), 1.0
        else:
            result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            stat, p = float(result.statistic), float(result.pvalue)

        # Paired effect size: Cohen's d_z = mean(difference) / sd(difference).
        # This is the paired form; the pooled-sd form would understate the effect
        # by ignoring that the same seed appears in both arms.
        d_z = mean_d / sd_d if sd_d > 0.0 else float("nan")

        return {
            "metric": self.metric,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n_seeds": n,
            "mean_a": float(np.mean(a)),
            "sd_a": float(np.std(a, ddof=1)) if n > 1 else 0.0,
            "mean_b": float(np.mean(b)),
            "sd_b": float(np.std(b, ddof=1)) if n > 1 else 0.0,
            "mean_difference": mean_d,
            "sd_difference": sd_d,
            "ci_low": mean_d - half,
            "ci_high": mean_d + half,
            "wilcoxon_statistic": stat,
            "p_value": p,
            "effect_size_dz": d_z,
            "effect_size_form": "Cohen d_z = mean(a-b) / sd(a-b), paired",
            "test": "Wilcoxon signed-rank, two-sided, paired on seed",
        }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved, monotone enforced."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def compare_arms(
    runs_csv: str | Path,
    metric: str,
    arm_a: str,
    arm_b: str,
) -> PairedComparison:
    """Build a paired comparison from an ``ablation_runs.csv``-shaped table."""
    rows: list[dict[str, str]] = []
    with Path(runs_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def series(arm: str) -> dict[int, float]:
        return {int(r["seed"]): float(r[metric]) for r in rows if r["arm"] == arm}

    a_by_seed, b_by_seed = series(arm_a), series(arm_b)
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    if not shared:
        raise ValueError(f"no seeds shared between arms {arm_a!r} and {arm_b!r}; the arms are not paired")
    dropped = sorted((set(a_by_seed) | set(b_by_seed)) - set(shared))
    if dropped:
        print(f"warning: seeds present in only one arm, excluded from the paired test: {dropped}")

    return PairedComparison(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        seeds=tuple(shared),
        values_a=tuple(a_by_seed[s] for s in shared),
        values_b=tuple(b_by_seed[s] for s in shared),
    )


def compare_family(
    runs_csv: str | Path,
    comparisons: list[tuple[str, str, str]],
    output_dir: str | Path,
    family_label: str = "unnamed",
) -> dict:
    """Run a family of paired comparisons and apply Holm across the whole family.

    ``comparisons`` is a list of ``(metric, arm_a, arm_b)``. The family is exactly
    what is passed in, and it is recorded in the manifest: an adjusted p-value is
    meaningless without knowing which set it was adjusted over.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = [compare_arms(runs_csv, m, a, b).summary() for m, a, b in comparisons]
    adjusted = holm_adjust([s["p_value"] for s in summaries])
    for s, p_adj in zip(summaries, adjusted, strict=True):
        s["p_value_holm"] = p_adj
        s["significant_holm_0.05"] = bool(p_adj < 0.05)
        s["family"] = family_label
        s["family_size"] = len(summaries)

    path = out / "paired_comparisons.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "family": family_label,
        "family_members": [f"{m}: {a} vs {b}" for m, a, b in comparisons],
        "correction": "Holm step-down, family-wise alpha = 0.05",
        "unit_of_analysis": "seed",
        "csv": str(path),
        "comparisons": summaries,
    }
    with (out / "paired_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    for s in summaries:
        print(
            f"{s['metric']}: {s['arm_a']} vs {s['arm_b']}  "
            f"diff={s['mean_difference']:+.4f} "
            f"CI=[{s['ci_low']:+.4f},{s['ci_high']:+.4f}] "
            f"p={s['p_value']:.4f} p_holm={s['p_value_holm']:.4f} d_z={s['effect_size_dz']:.3f}",
            flush=True,
        )
    return manifest


def format_paired_table(manifest: dict, scale: float = 100.0) -> str:
    """Render the family as a LaTeX booktabs body, metrics scaled to percent."""
    lines = []
    for s in manifest["comparisons"]:
        stars = "" if not s["significant_holm_0.05"] else r"$^{*}$"
        lines.append(
            f"{s['metric'].replace('_', ' ')} & "
            f"{s['mean_a'] * scale:.1f} $\\pm$ {s['sd_a'] * scale:.1f} & "
            f"{s['mean_b'] * scale:.1f} $\\pm$ {s['sd_b'] * scale:.1f} & "
            f"{s['mean_difference'] * scale:+.1f} & "
            f"[{s['ci_low'] * scale:+.1f}, {s['ci_high'] * scale:+.1f}] & "
            f"{s['p_value_holm']:.3f}{stars} & {s['effect_size_dz']:+.2f} \\\\"
        )
    return "\n".join(lines)
