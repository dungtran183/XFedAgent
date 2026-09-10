from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    auc_roc: float
    precision: float
    recall: float
    f1: float
    samples: int
    # Under a class-aware gate both sensitivity and specificity are quantities the
    # protocol acts on, so both are reported. ``recall`` is sensitivity; the
    # complementary rate is kept alongside it rather than left to be recomputed.
    specificity: float = 0.0
    balanced_accuracy: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)



def _scored_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Model outputs as float64, with non-finite entries scored as no signal.

    A norm-matched random update that the gate admits perturbs the global model
    in an arbitrary direction; enough admitted rounds of it drive the logits to
    ``inf`` and the softmax to ``nan``. Such a submission has no measurable
    utility, so it is scored as a confident negative prediction instead of being
    allowed to abort the measurement. The class-aware predicate then rejects it
    for a zero sensitivity numerator, which is the verdict the gate should reach
    anyway; doing the substitution here rather than special-casing the gate keeps
    a single definition of the predicate, and keeps the confusion counts the
    circuit is stated over consistent with the metrics reported beside them.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size and not np.isfinite(probabilities).all():
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    return probabilities


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, decision_threshold: float = 0.5
) -> BinaryMetrics:
    """Metrics at a stated decision threshold; 0.5 unless a config names another.

    The threshold is a parameter rather than a literal because on a cohort whose
    prevalence is far from a half the model's score mass sits well below 0.5, so
    ``0.5`` measures sensitivity at an operating point the model was never fitted
    for. It defaults to 0.5, which is what every earlier run used.
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = _scored_probabilities(probabilities)
    y_pred = (probabilities >= decision_threshold).astype(int)
    if np.unique(y_true).size == 1:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, probabilities))
    negatives = int(np.sum(y_true == 0))
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    specificity = float(true_negatives / negatives) if negatives else 0.0
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc_roc=auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=sensitivity,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        samples=int(y_true.size),
        specificity=specificity,
        balanced_accuracy=0.5 * (sensitivity + specificity),
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }



@dataclass(frozen=True)
class ConfusionCounts:
    """Integer confusion counts, the quantities a PoV circuit can expose cheaply.

    A ZK circuit cannot divide, so a class-aware admission predicate is stated over
    these four integers and enforced by cross-multiplication. Exposing the counts
    rather than a ratio also keeps the public instance integral, which matters
    because every public input must be a field element.
    """

    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def sensitivity(self) -> float:
        return self.tp / self.positives if self.positives else 0.0

    @property
    def specificity(self) -> float:
        return self.tn / self.negatives if self.negatives else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.sensitivity + self.specificity)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def confusion_counts(
    y_true: np.ndarray, probabilities: np.ndarray, decision_threshold: float = 0.5
) -> ConfusionCounts:
    """Integer confusion counts, binarised at ``decision_threshold``.

    This configurable probability threshold belongs to the software evaluator.
    The shipped binary-linear circuit derives its own predictions from a fixed
    zero-logit comparison; it does not prove arbitrary software thresholds.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = (_scored_probabilities(probabilities) >= decision_threshold).astype(int)
    return ConfusionCounts(
        tp=int(np.sum((y_pred == 1) & (y_true == 1))),
        tn=int(np.sum((y_pred == 0) & (y_true == 0))),
        fp=int(np.sum((y_pred == 1) & (y_true == 0))),
        fn=int(np.sum((y_pred == 0) & (y_true == 1))),
    )


def positive_count_for_rate(predicted_positive_rate: float, rows: int) -> int:
    """How many of ``rows`` challenge rows a stated predicted-positive rate declares.

    Rounded half up rather than with numpy's banker's rounding, because the prover
    and the verifier have to land on the same integer from the same public rate and
    the same row count, and half-up is the rule that is trivial to restate.
    """

    if rows <= 0:
        return 0
    count = int(np.floor(float(predicted_positive_rate) * rows + 0.5))
    return int(min(max(count, 0), rows))


def confusion_counts_at_rate(
    y_true: np.ndarray, probabilities: np.ndarray, predicted_positive_rate: float
) -> ConfusionCounts | None:
    """Integer confusion counts under a stated predicted-positive *rate*.

    The threshold rule fixes the probability above which a row counts as positive
    and lets the number of positives fall where it may. This fixes the number and
    lets the threshold fall where it may: ``k = round(rate * n)`` rows are declared
    positive, and which ones follows from the model's own ranking. A single public
    rate therefore means the same thing to every prover regardless of how its local
    score scale sits, which a single public threshold does not.

    This rate rule is implemented in software only. A possible circuit extension
    would compare each score with a private threshold and enforce ``TP + FP = k``
    for a public ``k``. That extension would require a revised public instance,
    relation and matching proving/verifying keys; it is not implemented by the
    shipped zero-logit binary-linear circuit.

    ``None`` is returned when no ``t`` declares exactly ``k`` rows positive, which
    happens when equal scores straddle the boundary. A constant-output model can
    realise only counts 0 and n, so no threshold selects an interior ``k``. Row
    index tie-breaking would select predictions unsupported by a score threshold
    and is deliberately excluded.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = _scored_probabilities(probabilities)
    rows = int(scores.size)
    count = positive_count_for_rate(predicted_positive_rate, rows)
    if rows == 0 or count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    threshold = float(scores[order[count - 1]])
    # A witness exists only when the k-th highest score is strictly above the
    # (k+1)-th, so that ``score >= t`` selects exactly k rows and not more.
    if count < rows and scores[order[count]] >= threshold:
        return None
    y_pred = (scores >= threshold).astype(int)
    if int(y_pred.sum()) != count:
        return None
    return ConfusionCounts(
        tp=int(np.sum((y_pred == 1) & (y_true == 1))),
        tn=int(np.sum((y_pred == 0) & (y_true == 0))),
        fp=int(np.sum((y_pred == 1) & (y_true == 0))),
        fn=int(np.sum((y_pred == 0) & (y_true == 1))),
    )


def passes_raw_accuracy(counts: ConfusionCounts, tau: float, tolerance: float = 0.0) -> bool:
    """The original PoV predicate: raw accuracy at or above ``tau``.

    On an imbalanced validation set this is satisfiable by a constant classifier:
    predicting the majority class always yields accuracy equal to the majority
    prevalence, which exceeds ``tau`` whenever ``tau`` is below that prevalence.
    """
    return counts.accuracy + tolerance >= tau


def passes_class_aware(
    counts: ConfusionCounts,
    tau_sens: float,
    tau_spec: float,
    tolerance: float = 0.0,
) -> bool:
    """Class-aware predicate: sensitivity and specificity thresholds, both enforced.

    Stated over integer counts by cross-multiplication so that no division is
    required, which is how the circuit enforces it:

        TP * denom_s >= ceil((tau_sens - tol) * denom_s) * (TP + FN)   [sensitivity]
        TN * denom_p >= ceil((tau_spec - tol) * denom_p) * (TN + FP)   [specificity]

    Here the equivalent rational comparisons are performed in floating point; the
    circuit uses the fixed-point numerator/denominator form of the same inequality.
    A constant classifier fails whichever inequality corresponds to the class it
    never predicts, because that class contributes a zero numerator.
    """
    if counts.positives == 0 or counts.negatives == 0:
        # An all-one-class validation subset cannot certify both directions.
        return False
    sens_ok = counts.tp >= (tau_sens - tolerance) * counts.positives
    spec_ok = counts.tn >= (tau_spec - tolerance) * counts.negatives
    return bool(sens_ok and spec_ok)


def predicate_margin(
    predicate: str,
    accuracy: float,
    sensitivity: float,
    specificity: float,
    threshold: float,
    threshold_sensitivity: float,
    threshold_specificity: float,
) -> float:
    """Nonnegative utility margin above the nominal predicate thresholds.

    Mirrors Eq. (2) of the paper. Under ``raw_accuracy`` the margin is the
    accuracy surplus over ``threshold``. Under ``class_aware`` it is the
    *smaller* of the two per-class surpluses, so an agent cannot bank reputation
    by excelling on the majority class alone. Admission tolerance does not lower
    these reward thresholds. The caller applies the margin only after admission;
    a submission within the tolerance band is accepted with zero reward.
    """
    if predicate == "class_aware":
        return max(
            0.0,
            min(
                sensitivity - threshold_sensitivity,
                specificity - threshold_specificity,
            ),
        )
    return max(0.0, accuracy - threshold)


def max_predicate_margin(
    predicate: str,
    threshold: float,
    threshold_sensitivity: float,
    threshold_specificity: float,
) -> float:
    """Delta_max of Eq. (3): the largest attainable predicate margin."""
    if predicate == "class_aware":
        return 1.0 - max(threshold_sensitivity, threshold_specificity)
    return 1.0 - threshold
