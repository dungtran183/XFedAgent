"""The PoV admission predicate under class imbalance.

A raw-accuracy threshold set below the majority prevalence is satisfiable by a
constant classifier, which is the failure these tests pin down.
"""
from __future__ import annotations

import pytest

from xfedagent.metrics import (
    ConfusionCounts,
    binary_metrics,
    confusion_counts,
    passes_class_aware,
    passes_raw_accuracy,
)
import numpy as np

#: MIMIC-III's reported positive-class prevalence.
PREVALENCE = 0.137
N = 100
POS = round(N * PREVALENCE)
NEG = N - POS

#: A classifier that always predicts the negative (majority) class.
MAJORITY = ConfusionCounts(tp=0, tn=NEG, fp=0, fn=POS)
#: A useful classifier: most of both classes correct.
USEFUL = ConfusionCounts(tp=11, tn=78, fp=8, fn=3)


def test_majority_classifier_beats_the_raw_threshold():
    """This is the vulnerability: 1 - prevalence exceeds tau."""
    assert MAJORITY.accuracy == pytest.approx(1 - PREVALENCE, abs=0.005)
    assert passes_raw_accuracy(MAJORITY, tau=0.75) is True


def test_majority_classifier_fails_the_class_aware_predicate():
    assert MAJORITY.sensitivity == 0.0
    assert passes_class_aware(MAJORITY, 0.60, 0.60) is False


def test_class_aware_predicate_admits_a_useful_classifier():
    assert passes_class_aware(USEFUL, 0.60, 0.60) is True


def test_balanced_accuracy_of_a_constant_classifier_is_one_half():
    assert MAJORITY.balanced_accuracy == pytest.approx(0.5)


def test_raw_gate_is_safe_only_above_the_majority_prevalence():
    """Documents the exact condition under which the original gate is sound."""
    assert passes_raw_accuracy(MAJORITY, tau=0.90) is False
    assert passes_raw_accuracy(MAJORITY, tau=0.80) is True


def test_all_one_class_subset_cannot_certify_both_directions():
    degenerate = ConfusionCounts(tp=0, tn=N, fp=0, fn=0)
    assert passes_class_aware(degenerate, 0.60, 0.60) is False


def test_cross_multiplied_form_matches_the_rational_comparison():
    """The circuit enforces integer cross-multiplication rather than division."""
    c = USEFUL
    for tau in (0.5, 0.6, 0.7, 0.8):
        rational = (c.sensitivity >= tau) and (c.specificity >= tau)
        integral = (c.tp >= tau * c.positives) and (c.tn >= tau * c.negatives)
        assert rational == integral


def test_confusion_counts_partition_the_sample():
    rng = np.random.default_rng(0)
    y = (rng.random(N) < PREVALENCE).astype(int)
    p = rng.random(N)
    c = confusion_counts(y, p)
    assert c.tp + c.tn + c.fp + c.fn == N
    assert c.positives == int(y.sum())


# --------------------------------------------------------------------------
# Balanced validation sampling
# --------------------------------------------------------------------------
import json  # noqa: E402
import pathlib  # noqa: E402
from xfedagent.config import parse_config  # noqa: E402
from xfedagent.pov import ValidationRotator  # noqa: E402


def _rotator(prevalence: float, balanced: bool, pool: int = 1000):
    cfg = parse_config(json.loads((pathlib.Path("configs/full.json")).read_text()))
    object.__setattr__(cfg.pov, "balanced_validation", balanced)
    rng = np.random.default_rng(0)
    y = (rng.random(pool) < prevalence).astype(int)
    x = rng.random((pool, 24, 17)).astype("float32")
    return ValidationRotator(cfg.pov, x, y), cfg


def test_balanced_sampling_ignores_pool_prevalence():
    rot, cfg = _rotator(PREVALENCE, balanced=True)
    _, y = rot.data_for(rot.challenge(0, 42))
    assert y.size == cfg.pov.validation_size
    assert int(y.sum()) == cfg.pov.validation_size // 2


def test_proportional_sampling_inherits_pool_prevalence():
    rot, cfg = _rotator(PREVALENCE, balanced=False)
    _, y = rot.data_for(rot.challenge(0, 42))
    assert int(y.sum()) < cfg.pov.validation_size // 2


def test_balanced_sampling_degrades_gracefully_when_a_class_is_scarce():
    """With too few positives the draw takes all of them and tops up."""
    rot, cfg = _rotator(0.01, balanced=True, pool=1000)
    _, y = rot.data_for(rot.challenge(0, 42))
    assert y.size == cfg.pov.validation_size


def test_rotation_changes_the_subset_between_rounds():
    rot, _ = _rotator(PREVALENCE, balanced=True)
    a = set(rot.challenge(0, 42).indices.tolist())
    b = set(rot.challenge(1, 42).indices.tolist())
    assert a != b


def test_predicate_margin_class_aware_takes_the_weaker_direction():
    from xfedagent.metrics import predicate_margin

    # Strong specificity cannot compensate for weak sensitivity.
    margin = predicate_margin(
        "class_aware",
        accuracy=0.95,
        sensitivity=0.72,
        specificity=0.99,
        threshold=0.75,
        threshold_sensitivity=0.70,
        threshold_specificity=0.70,
    )
    assert margin == pytest.approx(0.02)


def test_predicate_margin_is_zero_when_either_direction_fails():
    from xfedagent.metrics import predicate_margin

    assert predicate_margin(
        "class_aware",
        accuracy=0.95,
        sensitivity=0.60,
        specificity=0.99,
        threshold=0.75,
        threshold_sensitivity=0.70,
        threshold_specificity=0.70,
    ) == 0.0


def test_predicate_margin_raw_accuracy_matches_the_former_rule():
    from xfedagent.metrics import predicate_margin

    assert predicate_margin(
        "raw_accuracy",
        accuracy=0.86,
        sensitivity=0.0,
        specificity=1.0,
        threshold=0.75,
        threshold_sensitivity=0.70,
        threshold_specificity=0.70,
    ) == pytest.approx(0.11)


def test_max_predicate_margin_matches_the_reported_values():
    from xfedagent.metrics import max_predicate_margin

    assert max_predicate_margin("raw_accuracy", 0.75, 0.70, 0.70) == pytest.approx(0.25)
    assert max_predicate_margin("class_aware", 0.75, 0.70, 0.70) == pytest.approx(0.30)


def test_exclusion_threshold_from_max_margin():
    """q* = alpha*Delta_max / (beta + alpha*Delta_max), Eq. (10)."""
    from xfedagent.metrics import max_predicate_margin

    alpha, beta = 0.1, 0.5
    for predicate, expected in (("raw_accuracy", 0.048), ("class_aware", 0.057)):
        d = max_predicate_margin(predicate, 0.75, 0.70, 0.70)
        assert alpha * d / (beta + alpha * d) == pytest.approx(expected, abs=5e-4)


def test_a_diverged_update_is_scored_rather_than_crashing():
    """Non-finite model outputs must reach the gate as a verdict, not an exception.

    A norm-matched random update is admitted often enough that the global model
    can diverge; once the logits overflow, the softmax returns ``nan``. The
    measurement has to survive that and score the submission, because a run that
    aborts yields no detection rate at all.
    """
    y_true = np.array([0, 1, 0, 1])
    diverged = np.array([np.nan, np.nan, np.nan, np.nan])
    metrics = binary_metrics(y_true, diverged)
    assert metrics.samples == 4
    # Scored as a constant negative prediction: nothing positive is recovered.
    assert metrics.recall == 0.0
    assert metrics.specificity == 1.0
    assert metrics.auc_roc == pytest.approx(0.5)


def test_a_diverged_update_fails_the_class_aware_gate():
    y_true = np.array([0, 1] * 25)
    counts = confusion_counts(y_true, np.full(50, np.nan))
    assert counts.tp == 0 and counts.fn == 25
    assert not passes_class_aware(counts, 0.65, 0.65)


def test_metrics_and_counts_agree_on_non_finite_output():
    """The gate reads counts and the tables read metrics; they must not disagree."""
    y_true = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.9, np.inf, np.nan, -np.inf, 0.8])
    counts = confusion_counts(y_true, probabilities)
    metrics = binary_metrics(y_true, probabilities)
    assert counts.sensitivity == pytest.approx(metrics.recall)
    assert counts.specificity == pytest.approx(metrics.specificity)


# --------------------------------------------------------------------------
# The decision threshold the counts are taken at
# --------------------------------------------------------------------------


def test_default_decision_threshold_is_the_historical_half():
    """The field exists to be stated, not to change what earlier runs measured."""
    rng = np.random.default_rng(7)
    y = (rng.random(N) < PREVALENCE).astype(int)
    p = rng.random(N)
    assert confusion_counts(y, p) == confusion_counts(y, p, 0.5)
    assert binary_metrics(y, p).to_dict() == binary_metrics(y, p, 0.5).to_dict()


def test_a_low_prevalence_model_can_pass_only_at_a_placed_threshold():
    """The measured failure: score mass below 0.5 makes the gate read Se = 0.

    A model that separates the classes well can still be rejected by a gate that
    binarises at 0.5, because on a cohort at 13.7% prevalence the positive scores
    need not cross a half. The same counts, taken where the scores actually lie,
    satisfy the same predicate.
    """
    y = np.array([1] * 20 + [0] * 80)
    # Ranked correctly -- positives score above negatives -- but all below 0.5.
    p = np.concatenate([np.linspace(0.24, 0.40, 20), np.linspace(0.02, 0.22, 80)])
    at_half = confusion_counts(y, p, 0.5)
    assert at_half.tp == 0 and at_half.sensitivity == 0.0
    assert not passes_class_aware(at_half, 0.65, 0.65)
    placed = confusion_counts(y, p, 0.23)
    assert passes_class_aware(placed, 0.65, 0.65)
    # Neither predicate nor counts changed shape: still a partition of the sample.
    assert placed.tp + placed.tn + placed.fp + placed.fn == 100


def test_config_rejects_a_decision_threshold_outside_the_unit_interval():
    from xfedagent.config import validate_config

    cfg = parse_config(json.loads((pathlib.Path("configs/full.json")).read_text()))
    assert cfg.pov.decision_threshold == 0.5
    for bad in (0.0, 1.0, -0.1, 1.5):
        object.__setattr__(cfg.pov, "decision_threshold", bad)
        with pytest.raises(ValueError, match="decision_threshold"):
            validate_config(cfg)
