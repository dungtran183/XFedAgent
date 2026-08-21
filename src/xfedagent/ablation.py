"""Systematic component ablation over the XFedAgent pipeline.

The sweep re-runs one base configuration once per (arm, seed) pair, where an
*arm* switches off a named subset of the framework's components.  Everything
else -- data partition, model, attack schedule, and round count -- is held
fixed, so the difference between two arms isolates the marginal contribution of
the components that separate them.

Arms are addressed by the same labels that :class:`~.config.AblationConfig`
emits, so a results table can be keyed directly on ``summary.json['ablation']``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
import math

from .config import AblationConfig, ExperimentConfig
from .runner import ExperimentRunner

#: Components that may be switched off, in the order they are reported.
COMPONENTS: tuple[str, ...] = (
    "pov",
    "copy",
    "rep",
    "rot",
    "xchain",
)

_FIELD_FOR = {
    "pov": "pov_enabled",
    "copy": "copy_detector_enabled",
    "rep": "reputation_enabled",
    "rot": "rotation_enabled",
    "xchain": "cross_chain_enabled",
}

#: The default arm set reported in the paper's component-ablation table.
#: ``full`` is the complete framework; each remaining arm removes one component,
#: and ``no-pov+rep`` removes both factors of the two-factor design, so that
#: ``full``, ``no-pov``, ``no-rep`` and ``no-pov+rep`` form its four cells.
DEFAULT_ARMS: tuple[str, ...] = (
    "full",
    "no-pov",
    "no-rep",
    "no-pov+rep",
    "no-rot",
    "no-copy",
    "no-xchain",
)

METRIC_FIELDS: tuple[str, ...] = (
    "accuracy",
    "auc_roc",
    "f1",
    "malicious_rejection_rate",
    "honest_false_reject_rate",
)


def parse_arm(arm: str) -> AblationConfig:
    """Turn an arm label such as ``no-pov+rep`` into an :class:`AblationConfig`."""
    arm = arm.strip()
    if arm == "full":
        return AblationConfig()
    if not arm.startswith("no-"):
        raise ValueError(f"arm must be 'full' or start with 'no-': {arm!r}")
    disabled = [part for part in arm[3:].split("+") if part]
    if not disabled:
        raise ValueError(f"arm names no component to disable: {arm!r}")
    unknown = sorted(set(disabled) - set(COMPONENTS))
    if unknown:
        raise ValueError(
            f"unknown component(s) {', '.join(unknown)}; expected one of {', '.join(COMPONENTS)}"
        )
    return AblationConfig(**{_FIELD_FOR[name]: False for name in disabled})


def parse_seeds(spec: str) -> tuple[int, ...]:
    """Parse ``0-9`` or ``0,1,2`` into an ordered tuple of seeds."""
    seeds: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            lo_text, _, hi_text = chunk.partition("-")
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"empty seed range: {chunk!r}")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError("no seeds requested")
    return tuple(seeds)


def _arm_config(base: ExperimentConfig, ablation: AblationConfig, seed: int) -> ExperimentConfig:
    """Derive the configuration for one (arm, seed) cell of the sweep."""
    federation = replace(base.federation, seed=seed)
    relay = base.relay
    if not ablation.cross_chain_enabled:
        # A single-chain arm must actually stop relaying, not merely be labelled.
        relay = replace(relay, enabled=False)
    return replace(base, federation=federation, relay=relay, ablation=ablation)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, 0.0)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (mean, math.sqrt(variance))


def run_ablation(
    base: ExperimentConfig,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: str | Path,
) -> dict:
    """Execute the sweep and write tidy per-run and aggregated result tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for arm in arms:
        ablation = parse_arm(arm)
        for seed in seeds:
            cfg = _arm_config(base, ablation, seed)
            summary = ExperimentRunner(cfg).run()
            metrics = summary["final_metrics"]
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "auc_roc": metrics["auc_roc"],
                    "f1": metrics["f1"],
                    "malicious_rejection_rate": summary["malicious_rejection_rate"],
                    "honest_false_reject_rate": summary["honest_false_reject_rate"],
                    "accepted_updates": summary["accepted_updates"],
                    "rejected_updates": summary["rejected_updates"],
                    "run_name": summary["run_name"],
                    "config_digest": summary["config_digest"],
                }
            )
            print(
                f"arm={arm} seed={seed} accuracy={metrics['accuracy']:.4f} "
                f"auc={metrics['auc_roc']:.4f}",
                flush=True,
            )

    runs_path = out / "ablation_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)

    aggregate: list[dict] = []
    full_mean = None
    for arm in arms:
        cells = [r for r in runs if r["arm"] == arm]
        row: dict = {"arm": arm, "runs": len(cells)}
        for field in METRIC_FIELDS:
            mean, std = _mean_std([float(c[field]) for c in cells])
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        if arm == "full":
            full_mean = row["accuracy_mean"]
        aggregate.append(row)
    # Report each arm's accuracy cost relative to the complete framework, which is
    # the quantity the component-ablation table in the paper reports.
    for row in aggregate:
        row["accuracy_delta_pp"] = (
            (row["accuracy_mean"] - full_mean) * 100.0 if full_mean is not None else float("nan")
        )

    agg_path = out / "ablation_summary.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    manifest = {
        "base_config": base.name,
        "arms": list(arms),
        "seeds": list(seeds),
        "runs_csv": str(runs_path),
        "summary_csv": str(agg_path),
        "aggregate": aggregate,
    }
    with (out / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def format_latex_table(manifest: dict) -> str:
    """Render the aggregated sweep as a LaTeX booktabs body for the manuscript."""
    pretty = {
        "full": r"XFedAgent (full)",
        "no-pov": r"$-$ PoV utility gate",
        "no-rep": r"$-$ reputation weighting",
        "no-pov+rep": r"$-$ PoV $-$ reputation",
        "no-rot": r"$-$ validation rotation",
        "no-copy": r"$-$ copy detector",
        "no-xchain": r"$-$ cross-chain layer",
    }
    lines = []
    for row in manifest["aggregate"]:
        name = pretty.get(row["arm"], row["arm"])
        lines.append(
            f"{name} & "
            f"{row['accuracy_mean'] * 100:.1f} $\\pm$ {row['accuracy_std'] * 100:.1f} & "
            f"{row['auc_roc_mean']:.3f} & "
            f"{row['malicious_rejection_rate_mean'] * 100:.1f} & "
            f"{row['accuracy_delta_pp']:+.1f} \\\\"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Two-factor (2x2) analysis
# --------------------------------------------------------------------------
#: The four cells of the PoV-gate x reputation design, keyed by arm label.
FACTORIAL_CELLS: dict[str, tuple[bool, bool]] = {
    # arm label      -> (pov_on, rep_on)
    "no-pov+rep": (False, False),
    "no-pov": (False, True),
    "no-rep": (True, False),
    "full": (True, True),
}


def factorial_effects(aggregate: list[dict], metric: str = "accuracy_mean") -> dict:
    """Main effects and interaction for the PoV-gate x reputation design.

    A 2x2 design has four cells. Writing ``y[p][r]`` for the mean metric with the
    gate at level ``p`` and reputation at level ``r``, the main effect of a factor
    is its average simple effect across the levels of the other factor, and the
    interaction is the difference between the two simple effects:

        main(PoV) = ((y[1][0] - y[0][0]) + (y[1][1] - y[0][1])) / 2
        main(rep) = ((y[0][1] - y[0][0]) + (y[1][1] - y[1][0])) / 2
        interaction = y[1][1] - y[1][0] - y[0][1] + y[0][0]

    A negative interaction means the two mechanisms are partially redundant: the
    second one added recovers less than it would on its own. That is the expected
    sign whenever both defend against overlapping failure modes, or whenever the
    combined arm approaches a ceiling that caps additivity. Reporting the sum of
    the two simple effects as though it were the combined effect overstates the
    benefit, so this function returns the interaction alongside the main effects.
    """
    by_arm = {row["arm"]: row for row in aggregate}
    missing = sorted(set(FACTORIAL_CELLS) - set(by_arm))
    if missing:
        raise ValueError(
            "factorial analysis needs all four cells; missing arm(s): "
            + ", ".join(missing)
        )

    y = {}
    for arm, (pov_on, rep_on) in FACTORIAL_CELLS.items():
        if metric not in by_arm[arm]:
            raise ValueError(f"metric {metric!r} absent from arm {arm!r}")
        y[(pov_on, rep_on)] = float(by_arm[arm][metric])

    simple_pov_norep = y[(True, False)] - y[(False, False)]
    simple_pov_rep = y[(True, True)] - y[(False, True)]
    simple_rep_nopov = y[(False, True)] - y[(False, False)]
    simple_rep_pov = y[(True, True)] - y[(True, False)]
    interaction = y[(True, True)] - y[(True, False)] - y[(False, True)] + y[(False, False)]

    return {
        "metric": metric,
        "cells": {f"pov={int(p)},rep={int(r)}": v for (p, r), v in y.items()},
        "simple_effect_pov_without_reputation": simple_pov_norep,
        "simple_effect_pov_with_reputation": simple_pov_rep,
        "simple_effect_reputation_without_pov": simple_rep_nopov,
        "simple_effect_reputation_with_pov": simple_rep_pov,
        "main_effect_pov": (simple_pov_norep + simple_pov_rep) / 2.0,
        "main_effect_reputation": (simple_rep_nopov + simple_rep_pov) / 2.0,
        "interaction": interaction,
        "additive_prediction": (
            y[(False, False)] + simple_pov_norep + simple_rep_nopov
        ),
        "observed_both": y[(True, True)],
    }


def format_factorial_table(effects: dict, scale: float = 100.0) -> str:
    """Render the 2x2 effects as a LaTeX booktabs body, in the paper's units."""
    e = effects
    return "\n".join(
        [
            f"Main effect, PoV gate   & ${e['main_effect_pov'] * scale:+.1f}$ \\\\",
            f"Main effect, reputation & ${e['main_effect_reputation'] * scale:+.1f}$ \\\\",
            f"Interaction             & ${e['interaction'] * scale:+.1f}$ \\\\",
        ]
    )
