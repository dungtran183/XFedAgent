"""Threshold calibration for the class-aware gate.

This runs only when it has to. The trigger is a measured honest false-reject rate
above 15% at the operating point; below that the operating point stands and no
sweep is performed, because a threshold chosen after seeing several candidates is
a fitted parameter and has to be treated as one.

The selection rule is fixed here, in code, rather than applied by eye after the
table is printed:

    among the pairs whose malicious admission rate is at most 5%,
    take the one with the lowest honest false-reject rate;
    if no pair qualifies, keep (0.70, 0.70) and report the trade-off.

Both quantities come from the gate decisions taken during federated training on
the rotating validation pool — calibration data. The held-out test set is never
consulted, so the choice cannot be a test-set selection.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json

from .config import ExperimentConfig
from .runner import ExperimentRunner

#: A honest false-reject rate above this at the operating point triggers the sweep.
TRIGGER_HONEST_FALSE_REJECT: float = 0.15

#: A candidate pair is admissible only if it holds malicious admission at or below this.
MAX_MALICIOUS_ADMISSION: float = 0.05

#: The three candidate pairs the calibration rule sweeps.
DEFAULT_PAIRS: tuple[tuple[float, float], ...] = ((0.65, 0.65), (0.70, 0.70), (0.75, 0.75))

#: Where the rule lands when no candidate holds malicious admission low enough.
FALLBACK_PAIR: tuple[float, float] = (0.70, 0.70)


@dataclass(frozen=True)
class CalibrationCell:
    """Pooled gate behaviour for one threshold pair over the pilot seeds."""

    tau_sensitivity: float
    tau_specificity: float
    seeds: tuple[int, ...]
    honest_submitted: int
    honest_rejected: int
    malicious_submitted: int
    malicious_admitted: int

    @property
    def honest_false_reject_rate(self) -> float:
        """Pooled over submissions, so seeds with more submissions weigh more."""
        if self.honest_submitted == 0:
            return 0.0
        return self.honest_rejected / self.honest_submitted

    @property
    def malicious_admission_rate(self) -> float:
        if self.malicious_submitted == 0:
            return 0.0
        return self.malicious_admitted / self.malicious_submitted

    @property
    def label(self) -> str:
        return f"{self.tau_sensitivity:.2f}/{self.tau_specificity:.2f}"

    def to_row(self) -> dict:
        return {
            "tau_sensitivity": self.tau_sensitivity,
            "tau_specificity": self.tau_specificity,
            "seeds": " ".join(str(s) for s in self.seeds),
            "honest_submitted": self.honest_submitted,
            "honest_rejected": self.honest_rejected,
            "honest_false_reject_rate": self.honest_false_reject_rate,
            "malicious_submitted": self.malicious_submitted,
            "malicious_admitted": self.malicious_admitted,
            "malicious_admission_rate": self.malicious_admission_rate,
            "admissible": self.malicious_admission_rate <= MAX_MALICIOUS_ADMISSION,
        }


def parse_pairs(spec: str) -> tuple[tuple[float, float], ...]:
    """Parse ``0.65,0.70,0.75`` or ``0.65:0.70,0.75:0.75`` into threshold pairs.

    A bare value is a symmetric pair.
    """
    pairs: list[tuple[float, float]] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if ":" in token:
            sens, spec_ = token.split(":", 1)
            pairs.append((float(sens), float(spec_)))
        else:
            value = float(token)
            pairs.append((value, value))
    if not pairs:
        raise ValueError("no threshold pairs requested")
    return tuple(pairs)


def needs_calibration(honest_false_reject_rate: float) -> bool:
    """Whether the measured operating point crosses the calibration trigger."""
    return honest_false_reject_rate > TRIGGER_HONEST_FALSE_REJECT


def select_pair(cells: list[CalibrationCell]) -> dict:
    """Apply the pre-registered rule to the swept cells.

    Kept free of I/O so the rule itself can be tested directly, including the
    branch where nothing qualifies.
    """
    if not cells:
        raise ValueError("no calibration cells to choose between")

    admissible = [c for c in cells if c.malicious_admission_rate <= MAX_MALICIOUS_ADMISSION]
    if admissible:
        # Lowest honest false-reject rate wins; ties break towards the stricter
        # pair, so the choice is deterministic and never the laxer gate by accident.
        chosen = min(
            admissible,
            key=lambda c: (c.honest_false_reject_rate, -c.tau_sensitivity, -c.tau_specificity),
        )
        return {
            "chosen": (chosen.tau_sensitivity, chosen.tau_specificity),
            "rule_applied": (
                f"lowest honest false-reject among pairs with malicious admission "
                f"<= {MAX_MALICIOUS_ADMISSION:.0%}"
            ),
            "qualifying_pairs": [c.label for c in admissible],
            "fell_back": False,
            "trade_off": "",
        }

    kept = next(
        (c for c in cells if (c.tau_sensitivity, c.tau_specificity) == FALLBACK_PAIR), None
    )
    lowest = min(cells, key=lambda c: c.malicious_admission_rate)
    if kept is not None:
        retained = (
            f"At the retained operating point the honest false-reject rate is "
            f"{kept.honest_false_reject_rate:.1%} and malicious admission is "
            f"{kept.malicious_admission_rate:.1%}. "
        )
    else:
        # The operating point was not part of the sweep, so it has no measured row
        # here; saying so is better than printing a rate from a different pair.
        retained = (
            f"The operating point {FALLBACK_PAIR[0]:.2f}/{FALLBACK_PAIR[1]:.2f} was not "
            f"among the swept pairs, so no measured rate is quoted for it. "
        )
    return {
        "chosen": FALLBACK_PAIR,
        "rule_applied": (
            f"no pair held malicious admission <= {MAX_MALICIOUS_ADMISSION:.0%}; "
            f"kept the operating point {FALLBACK_PAIR[0]:.2f}/{FALLBACK_PAIR[1]:.2f}"
        ),
        "qualifying_pairs": [],
        "fell_back": True,
        "trade_off": (
            retained
            + f"The best admission rate over the sweep was "
            f"{lowest.malicious_admission_rate:.1%} at {lowest.label}, which still "
            f"exceeds the {MAX_MALICIOUS_ADMISSION:.0%} ceiling. Both numbers are "
            f"reported rather than resolved by moving the threshold."
        ),
    }


def sweep_pairs(
    base: ExperimentConfig,
    pairs: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
) -> list[CalibrationCell]:
    """Run the pilot for each threshold pair and pool its gate counts over seeds."""
    cells: list[CalibrationCell] = []
    for tau_sens, tau_spec in pairs:
        pooled = {"hs": 0, "hr": 0, "ms": 0, "ma": 0}
        for seed in seeds:
            pov = replace(
                base.pov,
                predicate="class_aware",
                threshold_sensitivity=tau_sens,
                threshold_specificity=tau_spec,
            )
            cfg = replace(base, pov=pov, federation=replace(base.federation, seed=seed))
            summary = ExperimentRunner(cfg).run()
            pooled["hs"] += int(summary["honest_submitted"])
            pooled["hr"] += int(summary["honest_rejected"])
            pooled["ms"] += int(summary["malicious_submitted"])
            pooled["ma"] += int(summary["malicious_admitted"])
            print(
                f"tau={tau_sens:.2f}/{tau_spec:.2f} seed={seed} "
                f"honest_reject={summary['honest_false_reject_rate']:.4f} "
                f"malicious_admit={1.0 - summary['malicious_rejection_rate']:.4f}",
                flush=True,
            )
        cells.append(
            CalibrationCell(
                tau_sensitivity=tau_sens,
                tau_specificity=tau_spec,
                seeds=tuple(seeds),
                honest_submitted=pooled["hs"],
                honest_rejected=pooled["hr"],
                malicious_submitted=pooled["ms"],
                malicious_admitted=pooled["ma"],
            )
        )
    return cells


def calibrate(
    base: ExperimentConfig,
    pairs: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    output_dir: str | Path,
) -> dict:
    """Sweep the candidate pairs and record the decision the rule produces."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cells = sweep_pairs(base, pairs, seeds)
    decision = select_pair(cells)

    rows = [c.to_row() for c in cells]
    path = out / "calibration_sweep.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    operating = next(
        (c for c in cells if (c.tau_sensitivity, c.tau_specificity) == FALLBACK_PAIR), None
    )
    manifest = {
        "selection_rule": decision["rule_applied"],
        "rule_fixed_before_results": True,
        "selected_on": "gate decisions over the rotating validation pool (calibration data)",
        "test_set_consulted": False,
        "trigger_honest_false_reject": TRIGGER_HONEST_FALSE_REJECT,
        "max_malicious_admission": MAX_MALICIOUS_ADMISSION,
        "seeds": list(seeds),
        "pairs": [list(p) for p in pairs],
        "chosen_pair": list(decision["chosen"]),
        "qualifying_pairs": decision["qualifying_pairs"],
        "fell_back_to_operating_point": decision["fell_back"],
        "trade_off": decision["trade_off"],
        "operating_point_honest_false_reject": (
            operating.honest_false_reject_rate if operating else None
        ),
        "csv": str(path),
        "cells": rows,
    }
    with (out / "calibration_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print()
    for c in cells:
        mark = "*" if list((c.tau_sensitivity, c.tau_specificity)) == manifest["chosen_pair"] else " "
        print(
            f"{mark} {c.label}  honest_false_reject={c.honest_false_reject_rate:.4f}  "
            f"malicious_admission={c.malicious_admission_rate:.4f}  "
            f"admissible={c.malicious_admission_rate <= MAX_MALICIOUS_ADMISSION}"
        )
    print(f"\nchosen={manifest['chosen_pair']}  rule={manifest['selection_rule']}")
    if manifest["trade_off"]:
        print(f"trade-off: {manifest['trade_off']}")
    return manifest


def format_calibration_table(manifest: dict, scale: float = 100.0) -> str:
    """Render the sweep as a LaTeX booktabs body, rates in percent."""
    lines = []
    for row in manifest["cells"]:
        chosen = [row["tau_sensitivity"], row["tau_specificity"]] == manifest["chosen_pair"]
        mark = r"$^{\dagger}$" if chosen else ""
        lines.append(
            f"{row['tau_sensitivity']:.2f} / {row['tau_specificity']:.2f}{mark} & "
            f"{row['honest_rejected']}/{row['honest_submitted']} & "
            f"{row['honest_false_reject_rate'] * scale:.1f} & "
            f"{row['malicious_admitted']}/{row['malicious_submitted']} & "
            f"{row['malicious_admission_rate'] * scale:.1f} & "
            f"{'yes' if row['admissible'] else 'no'} \\\\"
        )
    return "\n".join(lines)
