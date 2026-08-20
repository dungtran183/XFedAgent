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

