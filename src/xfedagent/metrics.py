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

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = (probabilities >= 0.5).astype(int)
    if np.unique(y_true).size == 1:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, probabilities))
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc_roc=auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        samples=int(y_true.size),
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


def confusion_counts(y_true: np.ndarray, probabilities: np.ndarray) -> ConfusionCounts:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(probabilities, dtype=np.float64) >= 0.5).astype(int)
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
