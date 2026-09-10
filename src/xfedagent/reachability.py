"""Empirical validation diagnostics for pooled and round-0 fitted models.

A pooled fit is a reference, not an upper bound on all possible models. The
per-client by shared-challenge pass matrix estimates empty gate-admission rounds
under uniform client sampling without replacement. It preserves dependence
induced by a shared challenge and does not extrapolate to later reputation or
optimization trajectories. No observed failure proves that a cohort is
unsatisfiable, and an observed pass does not guarantee that a full run succeeds.

Threshold searches and binormal AUC calculations are validation diagnostics.
Suggested operating points require a separate calibration assessment. The
held-out test split is never scored here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from .calibration import TRIGGER_HONEST_FALSE_REJECT
from .config import ExperimentConfig, resolved_predicted_positive_rate
from .data import load_dataset
from .metrics import (
    ConfusionCounts,
    confusion_counts,
    confusion_counts_at_rate,
    passes_class_aware,
    passes_raw_accuracy,
)
from .model import TorchModel
from .pov import ValidationRotator

#: Rotated challenges the verdict is averaged over, since the gate draws a fresh
#: validation subset every round.
DEFAULT_CHALLENGES: int = 8


@dataclass(frozen=True)
class Bound:
    """Fitted-model observations; the historical class name denotes no analytic bound."""

    label: str
    trials: int
    train_rows: int
    train_prevalence: float
    validation_rows: int
    auc: float
    sensitivity: float
    specificity: float
    passes: float
    reachable_min: float
    reachable_threshold: float
    mean_score: float
    #: Models scored. One pooled reference; one per client for the cold start.
    states: int = 1
    #: Rows the *median* fitted model saw; for the cold start, one client's shard.
    #: The pooled total is reported beside it rather than in its place.
    train_rows_total: int = 0
    #: Per-model pass rate over the rotated challenges, in client order.
    per_state_passes: tuple[float, ...] = ()
    #: Trials where the rate rule admitted no witness: equal scores straddled the
    #: k-th place, so no threshold declares exactly k rows positive.
    witness_failures: int = 0
    #: Rows are fitted clients; columns are the same rotated challenges.
    pass_matrix: tuple[tuple[bool, ...], ...] = ()
    threshold_search_passes: float = 0.0

    def to_row(self) -> dict[str, float | int | str]:
        return {
            "bound": self.label,
            "train_rows": self.train_rows,
            "train_rows_total": self.train_rows_total,
            "states": self.states,
            "trials": self.trials,
            "train_prevalence": round(self.train_prevalence, 6),
            "validation_rows": self.validation_rows,
            "auc_roc": round(self.auc, 6),
            "sensitivity": round(self.sensitivity, 6),
            "specificity": round(self.specificity, 6),
            "pass_rate": round(self.passes, 6),
            "reachable_min_se_sp": round(self.reachable_min, 6),
            "predicted_min_se_sp": round(predicted_reachable_min(self.auc), 6),
            "reachable_threshold": round(self.reachable_threshold, 6),
            "mean_score": round(self.mean_score, 6),
            "states_never_admitted": sum(1 for p in self.per_state_passes if p == 0.0),
            "witness_failures": self.witness_failures,
            "threshold_search_pass_rate": round(self.threshold_search_passes, 6),
        }

    def stall_probability(self, drawn: int) -> float:
        """Empirical empty-round probability over shared challenges.

        Conditional on a challenge with f failing clients among n, uniform
        sampling of m distinct clients fails jointly with C(f,m)/C(n,m). Average
        that quantity over the observed challenge columns, without assuming
        independent client decisions or predicting later rounds.
        """
        matrix = np.asarray(self.pass_matrix, dtype=bool)
        if matrix.ndim != 2 or not matrix.size:
            raise ValueError("shared-challenge pass matrix is required")
        population = matrix.shape[0]
        if not 1 <= drawn <= population:
            raise ValueError("drawn clients must lie between one and the fitted population")
        denominator = math.comb(population, drawn)
        failures = np.sum(~matrix, axis=0)
        return float(np.mean([math.comb(int(f), drawn) / denominator if f >= drawn else 0.0 for f in failures]))


def predicted_reachable_min(auc: float) -> float:
    """The reachable operating point a given AUC implies, under a binormal ROC.

    If the two score distributions are normal with equal variance then the ROC is
    fixed by AUC alone, and the best the class-aware predicate can do at any
    decision threshold is the symmetric point where sensitivity equals
    specificity: ``Phi(Phi^-1(AUC) / sqrt(2))``. Notably this does not involve
    prevalence, so a cohort's AUC alone says how far the predicate could get.

    It is an estimate, not a bound: an empirical ROC need not be binormal. Once
    scores are available, ``reachable_min_se_sp`` reports their observed optimum.
    """

    clipped = min(max(float(auc), 1e-9), 1.0 - 1e-9)
    return float(norm.cdf(norm.ppf(clipped) / np.sqrt(2.0)))


def auc_floor(required: float) -> float:
    """AUC threshold under the equal-variance binormal score model only.

    This inverts :func:`predicted_reachable_min`, not a distribution-free
    impossibility test. For an arbitrary monotone ROC passing through
    (1-required, required), the necessary area bound is required**2. Concavity
    strengthens it to required; neither assumption may be inferred from AUC alone.
    """

    clipped = min(max(float(required), 1e-9), 1.0 - 1e-9)
    return float(norm.cdf(np.sqrt(2.0) * norm.ppf(clipped)))


def _reachable_point(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """``max_t min(Se(t), Sp(t))`` and the threshold attaining it.

    The predicate needs both rates above their thresholds, so the quantity that
    decides admission at a given threshold is the weaker of the two. Maximising it
    over the ROC gives the operating point the predicate would be judged at if the
    decision threshold were chosen for this cohort instead of assumed.
    """

    if np.unique(y).size < 2:
        return 0.0, 0.5
    false_positive, true_positive, thresholds = roc_curve(y, scores)
    weaker = np.minimum(true_positive, 1.0 - false_positive)
    best = int(np.argmax(weaker))
    threshold = float(thresholds[best])
    # roc_curve prepends an unreachable +inf threshold for the all-negative point.
    if not np.isfinite(threshold):
        threshold = 1.0
    return float(weaker[best]), threshold


def _gate_counts(
    cfg: ExperimentConfig, y: np.ndarray, scores: np.ndarray, rate: float
) -> ConfusionCounts | None:
    """The four integers the configured gate would read, or ``None`` for no witness.

    The pre-flight has to measure the gate that will run, not the gate that used to
    run, so the binarisation rule is dispatched here exactly as it is in the backend.
    """

    if cfg.pov.decision_rule == "rate":
        return confusion_counts_at_rate(y, scores, rate)
    return confusion_counts(y, scores, cfg.pov.decision_threshold)


def _gate_verdict(cfg: ExperimentConfig, counts: ConfusionCounts) -> bool:
    if cfg.pov.predicate == "class_aware":
        return passes_class_aware(
            counts,
            cfg.pov.threshold_sensitivity,
            cfg.pov.threshold_specificity,
            cfg.pov.tolerance,
        )
    return passes_raw_accuracy(counts, cfg.pov.threshold, cfg.pov.tolerance)


def _threshold_can_pass(cfg: ExperimentConfig, y: np.ndarray, scores: np.ndarray) -> bool:
    """Whether this fitted model passes somewhere on this observed ROC."""
    if not len(y) or not np.isfinite(scores).all():
        return False
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not positives or not negatives:
        return False
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    for fp_rate, tp_rate in zip(fpr, tpr, strict=True):
        tp, fp = round(tp_rate * positives), round(fp_rate * negatives)
        if _gate_verdict(cfg, ConfusionCounts(tp, negatives - fp, fp, positives - tp)):
            return True
    return False


def _measure(
    label: str,
    cfg: ExperimentConfig,
    model: TorchModel,
    states: list,
    train_y: np.ndarray,
    rotator: ValidationRotator,
    challenges: int,
    rate: float = 0.5,
    row_counts: list[int] | None = None,
) -> Bound:
    """Score trained states against ``challenges`` rotated validation subsets.

    ``pass_rate`` averages (state, challenge) trials. The full pass matrix is
    retained because this pooled rate alone does not determine the chance that
    a shared challenge excludes every selected client.
    """

    if challenges <= 0 or not states:
        raise ValueError("at least one fitted state and one validation challenge are required")
    aucs, sens, specs, passes, mins, thresholds, means = [], [], [], [], [], [], []
    rows = 0
    per_state: list[float] = []
    pass_matrix: list[tuple[bool, ...]] = []
    searched_passes: list[bool] = []
    witness_failures = 0
    for state in states:
        state_passes: list[float] = []
        for index in range(challenges):
            challenge = rotator.challenge(index, cfg.federation.seed)
            validation_x, validation_y = rotator.data_for(challenge)
            scores = model.predict_probabilities(state, validation_x)
            gate_counts = _gate_counts(cfg, validation_y, scores, rate)
            if gate_counts is None:
                witness_failures += 1
            counts = gate_counts or confusion_counts(
                validation_y, scores, cfg.pov.decision_threshold
            )
            rows = len(validation_y)
            aucs.append(
                float(roc_auc_score(validation_y, scores))
                if np.unique(validation_y).size > 1
                else 0.5
            )
            sens.append(counts.sensitivity)
            specs.append(counts.specificity)
            admitted = gate_counts is not None and _gate_verdict(cfg, counts)
            searched_passes.append(_threshold_can_pass(cfg, validation_y, scores))
            state_passes.append(1.0 if admitted else 0.0)
            reachable, threshold = _reachable_point(validation_y, scores)
            mins.append(reachable)
            thresholds.append(threshold)
            means.append(float(np.mean(scores)) if scores.size else 0.0)
        passes.extend(state_passes)
        per_state.append(float(np.mean(state_passes)) if state_passes else 0.0)
        pass_matrix.append(tuple(bool(p) for p in state_passes))
    counts_per_state = row_counts or [int(len(train_y))]
    return Bound(
        label=label,
        trials=len(passes),
        # One fitted model's rows; for the cold start, one shard.
        train_rows=int(np.median(counts_per_state)),
        train_rows_total=int(len(train_y)),
        states=len(states),
        train_prevalence=float(np.mean(train_y)) if len(train_y) else 0.0,
        validation_rows=rows,
        auc=float(np.mean(aucs)),
        sensitivity=float(np.mean(sens)),
        specificity=float(np.mean(specs)),
        passes=float(np.mean(passes)),
        reachable_min=float(np.mean(mins)),
        reachable_threshold=float(np.median(thresholds)),
        mean_score=float(np.mean(means)),
        per_state_passes=tuple(per_state),
        witness_failures=witness_failures,
        pass_matrix=tuple(pass_matrix),
        threshold_search_passes=float(np.mean(searched_passes)),
    )


def measure_reachability(
    cfg: ExperimentConfig,
    challenges: int = DEFAULT_CHALLENGES,
    output: str | None = None,
) -> dict:
    """Empirical diagnostics; no impossibility or future-training certificate."""

    if challenges <= 0:
        raise ValueError("challenges must be positive")

    rate = resolved_predicted_positive_rate(cfg)
    bundle = load_dataset(cfg.data, cfg.federation.seed)
    features = bundle.clients[0].train_x.shape[2]
    rotator = ValidationRotator(cfg.pov, bundle.validation_pool_x, bundle.validation_pool_y)

    model = TorchModel(
        cfg.model, features, cfg.data.timesteps, cfg.federation.seed, cfg.pov.decision_threshold
    )
    initial = model.clone_state()

    # Cold start: every client trains its own shard from the initialisation, which
    # is what round 0 submits.
    cold_states = [
        model.train_local(initial, client.train_x, client.train_y, cfg.federation.seed + index).state
        for index, client in enumerate(bundle.clients)
    ]
    cold_rows = np.concatenate([client.train_y for client in bundle.clients])
    cold_bound = _measure(
        "cold_start",
        cfg,
        model,
        cold_states,
        cold_rows,
        rotator,
        challenges,
        rate=rate,
        row_counts=[int(client.train_y.size) for client in bundle.clients],
    )

    # A budgeted pooled reference. More data/epochs do not make a fitted model an
    # oracle or an upper bound on attainable predictive performance.
    pooled_x = np.concatenate([client.train_x for client in bundle.clients])
    pooled_y = np.concatenate([client.train_y for client in bundle.clients])
    pooled_model = TorchModel(
        cfg.model, features, cfg.data.timesteps, cfg.federation.seed, cfg.pov.decision_threshold
    )
    object.__setattr__(
        pooled_model, "cfg", replace(cfg.model, local_epochs=max(cfg.model.local_epochs, 20))
    )
    pooled = pooled_model.train_local(
        pooled_model.clone_state(), pooled_x, pooled_y, cfg.federation.seed
    )
    pooled_reference = _measure(
        "pooled_reference", cfg, pooled_model, [pooled.state], pooled_y, rotator, challenges, rate=rate
    )

    required = max(cfg.pov.threshold_sensitivity, cfg.pov.threshold_specificity) - cfg.pov.tolerance

    drawn = cfg.federation.clients_per_round
    stall = cold_bound.stall_probability(drawn)
    if (cfg.pov.decision_rule == "threshold" and pooled_reference.passes == 0.0
            and pooled_reference.threshold_search_passes > 0.0):
        verdict = "threshold_sensitive_reference"
    elif cold_bound.passes == 0.0 and pooled_reference.passes == 0.0:
        verdict = "no_admission_observed"
    elif stall > TRIGGER_HONEST_FALSE_REJECT:
        verdict = "elevated_empty_round_risk"
    else:
        verdict = "observed_admission"

    # These are candidates on observed validation scores, not calibrated policies.
    binding = pooled_reference if pooled_reference.passes == 0.0 else cold_bound
    threshold_gap = abs(pooled_reference.reachable_threshold - cold_bound.reachable_threshold)
    manifest = {
        "config": cfg.name,
        "predicate": cfg.pov.predicate,
        "decision_threshold": cfg.pov.decision_threshold,
        "threshold_sensitivity": cfg.pov.threshold_sensitivity,
        "threshold_specificity": cfg.pov.threshold_specificity,
        "tolerance": cfg.pov.tolerance,
        "effective_threshold": round(required, 6),
        # Historical key retained, explicitly scoped to the binormal model.
        "auc_floor": round(auc_floor(required), 6),
        "auc_floor_assumption": "equal-variance binormal scores; diagnostic, not a distribution-free bound",
        "distribution_free_auc_floor": round(
            max(0.0, cfg.pov.threshold_sensitivity - cfg.pov.tolerance)
            * max(0.0, cfg.pov.threshold_specificity - cfg.pov.tolerance), 6
        ) if cfg.pov.predicate == "class_aware" else None,
        "challenges": challenges,
        "decision_rule": cfg.pov.decision_rule,
        "predicted_positive_rate": round(rate, 6),
        "verdict": verdict,
        "diagnostic_only": True,
        "clients_per_round": drawn,
        # P(round 0 admits nothing), which the verdict is decided on.
        "round0_stall_probability": round(stall, 6),
        "round0_stall_method": "empirical mean of C(failing clients,drawn)/C(all clients,drawn) on shared challenges",
        "cold_start_pass_matrix": [list(row) for row in cold_bound.pass_matrix],
        "clients_never_admitted": sum(1 for p in cold_bound.per_state_passes if p == 0.0),
        # The rate at which an honest round-0 client is turned away, which a cohort
        # can raise while still clearing the stall test. The reputation rule charges
        # 0.5 per rejection against a 0.1 reward, so a sustained rate drives honest
        # clients under reputation_min. Compared against the pre-registered trigger.
        "implied_honest_false_reject": round(1.0 - cold_bound.passes, 6),
        "honest_false_reject_above_trigger": bool(
            (1.0 - cold_bound.passes) > TRIGGER_HONEST_FALSE_REJECT
        ),
        "trigger_honest_false_reject": TRIGGER_HONEST_FALSE_REJECT,
        # The threshold the cohort itself implies, read off the validation pool only.
        "suggested_decision_threshold": round(binding.reachable_threshold, 4),
        "binding_bound": binding.label,
        "reachable_threshold_gap": round(threshold_gap, 6),
        # Candidate optima may differ even when their passing intervals overlap.
        # Utility tolerance must not be compared to a gap in score units.
        "thresholds_disagree": bool(not np.isclose(
            pooled_reference.reachable_threshold, cold_bound.reachable_threshold
        )),
        "threshold_comparison_is_heuristic": True,
        # The closed-form rate rule for this challenge, reported unconditionally so a
        # stalled threshold run carries its own alternative.
        "suggested_predicted_positive_rate": round(resolved_predicted_positive_rate(cfg), 6),
        "bounds": [cold_bound.to_row(), pooled_reference.to_row()],
    }
    if output is not None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "reachability.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cold_bound.to_row().keys()))
            writer.writeheader()
            writer.writerows(manifest["bounds"])
        (directory / "reachability_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifest["csv"] = str(csv_path)
    return manifest


def format_reachability_table(manifest: dict) -> str:
    """A booktabs body, in the same shape as the other pre-flight tables."""

    lines = []
    for row in manifest["bounds"]:
        lines.append(
            f"{row['bound'].replace('_', ' ')} & {row['train_rows']} & "
            f"{row['auc_roc']:.3f} & {row['sensitivity']:.3f} & {row['specificity']:.3f} & "
            f"{row['reachable_min_se_sp']:.3f} & {row['predicted_min_se_sp']:.3f} & "
            f"{row['reachable_threshold']:.3f} & "
            f"{row['pass_rate']:.2f} \\\\"
        )
    return "\n".join(lines)
