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
