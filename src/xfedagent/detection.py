"""Per-attack detection rate p_det with Wilson confidence intervals.

Two things are deliberate here. First, the estimator is submission-weighted:

    p_det = 1 - (total malicious submissions admitted) / (total malicious submissions)

not a mean of per-round rejection rates. Each round contributes a different
denominator, so averaging per-round ratios weights sparse rounds equally with
dense ones and biases the estimate.

Second, the interval is Wilson rather than normal-approximation, because p_det
sits near 0 or 1 for most attacks and the Wald interval is badly behaved — and
can leave [0, 1] entirely — in exactly that regime.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import math

from .config import ExperimentConfig
from .runner import ExperimentRunner

#: Attacks measured one at a time. The schedule in a config may combine several;
#: p_det is only interpretable per attack, since a combined schedule cannot
#: attribute a rejection to any one of them.
DEFAULT_ATTACKS: tuple[str, ...] = (
    "majority_class",
    "label_flip",
    "random_gradient",
    "alie",
    "minmax",
    "backdoor",
    # The canonical free-rider vector. Listed last because it is the one attack the
    # utility predicate cannot see: a replayed global model attains exactly the global
    # model's utility, so it is refused by the copy-detector bound in the relation
    # rather than by the predicate.
    "replay",
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval at the confidence level implied by ``z``.

    Defaults to 95%. Returns ``(0.0, 1.0)`` for an empty sample, which is the
    honest statement when nothing was observed.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class AttackDetection:
    attack: str
    seed: int
    submitted: int
    admitted: int

    @property
    def detected(self) -> int:
        return self.submitted - self.admitted

    @property
    def p_det(self) -> float:
        return self.detected / self.submitted if self.submitted else 0.0

    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.detected, self.submitted)


def measure_p_det(
    base: ExperimentConfig,
    attacks: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: str | Path,
) -> dict:
    """Run one single-attack experiment per (attack, seed) and tabulate p_det."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[AttackDetection] = []

    for attack in attacks:
        for seed in seeds:
            federation = replace(base.federation, attacks=(attack,), seed=seed)
            cfg = replace(base, federation=federation)
            summary = ExperimentRunner(cfg).run()
            record = AttackDetection(
                attack=attack,
                seed=seed,
                submitted=int(summary["malicious_submitted"]),
                admitted=int(summary["malicious_admitted"]),
            )
            records.append(record)
            lo, hi = record.interval()
            print(
                f"attack={attack} seed={seed} submitted={record.submitted} "
                f"admitted={record.admitted} p_det={record.p_det:.4f} "
                f"CI=[{lo:.4f},{hi:.4f}]",
                flush=True,
            )

    per_seed = out / "pdet_by_attack.csv"
    with per_seed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack", "seed", "submitted", "admitted", "p_det", "ci_low", "ci_high"])
        for r in records:
            lo, hi = r.interval()
            writer.writerow([r.attack, r.seed, r.submitted, r.admitted, f"{r.p_det:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    # Pooled across seeds, again by summing submissions rather than averaging rates.
    pooled: list[dict] = []
    for attack in attacks:
        cells = [r for r in records if r.attack == attack]
        submitted = sum(r.submitted for r in cells)
        admitted = sum(r.admitted for r in cells)
        detected = submitted - admitted
        p = detected / submitted if submitted else 0.0
        lo, hi = wilson_interval(detected, submitted)
        pooled.append(
            {
                "attack": attack,
                "seeds": len(cells),
                "submitted": submitted,
                "admitted": admitted,
                "detected": detected,
                "p_det": p,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    pooled_path = out / "pdet_pooled.csv"
    with pooled_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pooled[0].keys()))
        writer.writeheader()
        writer.writerows(pooled)

    manifest = {
        "base_config": base.name,
        "attacks": list(attacks),
        "seeds": list(seeds),
        "estimator": "submission-weighted; p_det = 1 - admitted/submitted",
        "interval": "Wilson score, 95%",
        "per_seed_csv": str(per_seed),
        "pooled_csv": str(pooled_path),
        "pooled": pooled,
    }
    with (out / "pdet_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def format_pdet_table(manifest: dict) -> str:
    """Render the pooled table as a LaTeX booktabs body."""
    pretty = {
        "majority_class": "Majority-class collapse",
        "label_flip": "Label flipping",
        "random_gradient": "Random gradient",
        "alie": "ALIE",
        "minmax": "Min-Max",
        "backdoor": "Backdoor (trigger)",
        "replay": "Replay (free-rider)",
    }
    lines = []
    for row in manifest["pooled"]:
        name = pretty.get(row["attack"], row["attack"].replace("_", r"\_"))
        lines.append(
            f"{name} & {row['submitted']} & {row['admitted']} & "
            f"{row['p_det']:.3f} & [{row['ci_low']:.3f}, {row['ci_high']:.3f}] \\\\"
        )
    return "\n".join(lines)
