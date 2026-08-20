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
#: and ``no-pov+rep`` removes the two that the paper claims act synergistically.
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
