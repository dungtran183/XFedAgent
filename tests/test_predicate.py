"""The PoV admission predicate under class imbalance.

A raw-accuracy threshold set below the majority prevalence is satisfiable by a
constant classifier, which is the failure these tests pin down.
"""
from __future__ import annotations

import pytest

from xfedagent.metrics import (
    ConfusionCounts,
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
import json, pathlib  # noqa: E402
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
