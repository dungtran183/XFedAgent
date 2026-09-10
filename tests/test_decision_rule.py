"""A public predicted-positive rate in place of a public decision threshold.

The gate binarises the model's scores before the four integers reach the circuit,
and the rule it binarises by is a public challenge parameter. A *threshold* fixes
the probability at which a row becomes positive; a *rate* fixes how many rows
become positive and lets each prover's threshold fall where its own scores sit.

Under a non-IID partition that difference decides whether the federation starts at
all. A client's score scale follows its shard's prevalence, so on a Dirichlet(0.5)
split the min(Se, Sp)-optimal threshold ranges over most of the unit interval and no
single public value sits near more than a couple of clients. A single public rate
sits at the same operating point for all of them. These tests pin that property, the
closed form that says where the rate belongs, and the tie rule that keeps a
constant-output model refused.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from xfedagent.config import (
    PAIRED_ALLOWED_DIFFERENCES,
    challenge_prevalence,
    parse_config,
    predicted_positive_rate_for,
    resolved_predicted_positive_rate,
    validate_config_pair,
)
from xfedagent.metrics import (
    confusion_counts,
    confusion_counts_at_rate,
    passes_class_aware,
    positive_count_for_rate,
)

#: A hundred-row balanced challenge, which is what ``balanced_validation`` draws.
N = 100
TAU = 0.65
TOLERANCE = 0.03
REQUIRED = TAU - TOLERANCE


def ranked(rng: np.random.Generator, prevalence: float, separation: float, scale: float,
           offset: float, rows: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores whose *ranking* is fixed but whose scale is not.

    ``scale`` and ``offset`` compress the scores into a narrow band, which is what a
    client trained on a shard of 0.7% prevalence actually produces: the ranking is
    informative, every value is small. A rule stated over ranks cannot see the
    difference; a rule stated over probabilities sees nothing else.
    """

    y = (rng.random(rows) < prevalence).astype(int)
    latent = rng.normal(separation * y, 1.0)
    unit = 1.0 / (1.0 + np.exp(-latent))
    return y, offset + scale * unit


def test_the_rate_declares_exactly_k_rows_positive():
    rng = np.random.default_rng(0)
    y, scores = ranked(rng, 0.5, 1.4, 1.0, 0.0)
    for rate in (0.2, 0.35, 0.5, 0.62, 0.8):
        counts = confusion_counts_at_rate(y, scores, rate)
        assert counts is not None
        # The one constraint the circuit gains, over integers it already carries.
        assert counts.tp + counts.fp == positive_count_for_rate(rate, N)


def test_the_count_is_rounded_half_up_and_clipped():
    assert positive_count_for_rate(0.5, 100) == 50
    assert positive_count_for_rate(0.3923, 100) == 39
    assert positive_count_for_rate(0.3923, 1000) == 392
    # Half-up rather than numpy's banker's rounding, which would send 0.395*100 to 39.
    assert positive_count_for_rate(0.395, 100) == 40
    assert positive_count_for_rate(0.005, 100) == 1
    assert positive_count_for_rate(0.0, 100) == 0
    assert positive_count_for_rate(1.0, 100) == 100
    assert positive_count_for_rate(0.5, 0) == 0


def test_a_rescaled_model_is_judged_identically_by_rate_and_differently_by_threshold():
    """The property the whole change rests on.

    Two clients with the same ranking and different score scales are the same model
    as far as the predicate's own quantities go: sensitivity and specificity are
    within-class rates over an ordering. The rate rule reads the ordering and returns
    the same four integers. The threshold rule reads the probabilities and returns
    two different verdicts, which is the deadlock, restated in six lines.
    """

    rng = np.random.default_rng(11)
    y, wide = ranked(rng, 0.5, 1.4, 1.0, 0.0)
    narrow = 0.02 + 0.06 * (wide - wide.min()) / (wide.max() - wide.min())

    by_rate = [confusion_counts_at_rate(y, s, 0.5) for s in (wide, narrow)]
    assert by_rate[0] == by_rate[1]

    by_threshold = [confusion_counts(y, s, 0.5) for s in (wide, narrow)]
    assert by_threshold[0] != by_threshold[1]
    # The compressed client is not a worse model; the threshold simply sits above
    # every score it produces, so it is scored as having predicted nothing positive.
    assert by_threshold[1].tp + by_threshold[1].fp == 0
    assert by_threshold[1].sensitivity == 0.0


def test_a_constant_output_model_has_no_witness_at_any_interior_rate():
    """The security property, and the reason ties are not broken by row index.

    A constant model can realise only two predicted-positive counts, 0 and n, so no
    ``t`` satisfies ``pred = (score >= t)`` together with ``sum(pred) = k`` for any k
    between them: the proof does not exist. Breaking the tie by index instead would
    hand it whichever half the ordering happened to give, which on a balanced
    hundred-row challenge admits it about once in eighty-four draws.
    """

    y = np.array([0, 1] * (N // 2))
    for value in (0.0, 0.137, 0.5, 0.8607, 1.0):
        scores = np.full(N, value)
        for rate in (0.01, 0.2, 0.5, 0.62, 0.99):
            assert confusion_counts_at_rate(y, scores, rate) is None
    # And the bound that holds even if a tiebreak were introduced: whatever k is,
    # a model with no ordering cannot get both rates above a half.
    for k in range(1, N):
        assert min(k / N, 1.0 - k / N) <= 0.5 < REQUIRED


def test_the_closed_form_rate_is_where_the_predicate_target_is_realisable():
    """``rate*(pi, tau) = pi*tau + (1-pi)(1-tau)``, and on a balanced challenge, a half.

    A model sitting exactly at Se = Sp = tau declares ``tau*P + (1-tau)*N`` rows
    positive, so that fraction is where the gate has to stand for the target to be
    attainable at all. It needs no sweep, and on a balanced challenge it collapses to
    0.5 for every tau -- the rule there is simply "the better-scoring half".
    """

    for pi in (0.007, 0.1397, 0.5, 0.744):
        for tau in (0.60, 0.65, 0.75):
            expected = pi * tau + (1.0 - pi) * (1.0 - tau)
            y = (np.arange(10000) < round(pi * 10000)).astype(int)
            # A model at exactly Se = Sp = tau: send tau of each class to its own side.
            scores = np.where(y == 1, 0.9, 0.1)
            flip = np.concatenate([
                np.flatnonzero(y == 1)[: round((1 - tau) * y.sum())],
                np.flatnonzero(y == 0)[: round((1 - tau) * (y.size - y.sum()))],
            ])
            scores[flip] = 1.0 - scores[flip]
            assert (scores > 0.5).mean() == pytest.approx(expected, abs=2e-4)
    assert predicted_positive_rate_for(
        parse_config(_rate_config()).pov, 0.5
    ) == pytest.approx(0.5)


def _rate_config(**pov: object) -> dict:
    """The synthetic config with the class-aware gate under the rate rule."""

    from tests.conftest import _SYNTHETIC_CONFIG

    raw = deepcopy(_SYNTHETIC_CONFIG)
    raw["pov"].update(
        {
            "predicate": "class_aware",
            "threshold_sensitivity": TAU,
            "threshold_specificity": TAU,
            "tolerance": TOLERANCE,
            "balanced_validation": True,
            "decision_rule": "rate",
        }
    )
    raw["pov"].update(pov)
    return raw


def test_the_default_rule_is_the_threshold_so_earlier_runs_are_unchanged():
    from tests.conftest import _SYNTHETIC_CONFIG

    cfg = parse_config(deepcopy(_SYNTHETIC_CONFIG))
    assert cfg.pov.decision_rule == "threshold"
    assert cfg.pov.predicted_positive_rate is None


def test_an_unknown_rule_or_an_out_of_range_rate_is_refused():
    with pytest.raises(ValueError, match="decision_rule"):
        parse_config(_rate_config(decision_rule="top_k"))
    with pytest.raises(ValueError, match="predicted_positive_rate"):
        parse_config(_rate_config(predicted_positive_rate=0.0))
    with pytest.raises(ValueError, match="predicted_positive_rate"):
        parse_config(_rate_config(predicted_positive_rate=1.0))


def test_a_rate_that_rounds_to_nothing_or_to_every_row_is_refused():
    """Not a verdict about the cohort, so it is caught before it can pose as one.

    A rate rounding to 0 declares nothing positive and a rate rounding to n declares
    everything positive; either way one of the two rates is zero for every submission
    and the gate refuses the whole run. Left uncaught this arrives later looking like
    a cohort that cannot meet its thresholds.
    """

    size = _rate_config()["pov"]["validation_size"]
    with pytest.raises(ValueError, match="between 1 and"):
        parse_config(_rate_config(predicted_positive_rate=0.4 / size))
    with pytest.raises(ValueError, match="between 1 and"):
        parse_config(_rate_config(predicted_positive_rate=1.0 - 0.4 / size))


def test_the_two_arms_must_agree_on_the_decision_rule():
    """The rule is a shared challenge parameter, not part of the gate being compared.

    The arms may differ on the predicate and its thresholds and on nothing else. A
    rate rule used by one arm and not the other would move the operating point along
    with the predicate, and the measured difference would be the sum of the two.
    """

    assert "pov.decision_rule" not in PAIRED_ALLOWED_DIFFERENCES
    assert "pov.predicted_positive_rate" not in PAIRED_ALLOWED_DIFFERENCES
    left = _rate_config()
    right = deepcopy(left)
    right["pov"]["predicate"] = "raw_accuracy"
    validate_config_pair(left, right)          # the predicate alone is admissible
    right["pov"]["decision_rule"] = "threshold"
    with pytest.raises(ValueError, match="pov.decision_rule"):
        validate_config_pair(left, right)


def test_the_challenge_prevalence_and_not_the_cohort_prevalence_sets_the_rate():
    """The mistake worth a test of its own.

    ``balanced_validation`` draws half the challenge from each class, so the challenge
    is 50/50 however skewed the cohort is. Deriving the rate from the cohort's own
    prevalence would put the gate at 0.32 on a challenge whose target is 0.50, and the
    measured admission at 0.40 was 0.15 against 0.80 at the correct value.
    """

    balanced = parse_config(_rate_config())
    assert challenge_prevalence(balanced) == 0.5
    assert resolved_predicted_positive_rate(balanced) == pytest.approx(0.5)

    raw = _rate_config(balanced_validation=False)
    raw["data"]["positive_rate"] = 0.1397
    proportional = parse_config(raw)
    assert challenge_prevalence(proportional) == pytest.approx(0.1397)
    assert resolved_predicted_positive_rate(proportional) == pytest.approx(
        0.1397 * TAU + (1 - 0.1397) * (1 - TAU)
    )


def test_the_backend_refuses_a_constant_model_and_admits_a_ranked_one():
    """End to end through the gate the runner actually calls."""

    from xfedagent.pov import SoftwarePoVBackend

    cfg = parse_config(_rate_config())
    backend = SoftwarePoVBackend(cfg.pov, cfg.model, predicted_positive_rate=0.5)
    assert backend.predicted_positive_rate == pytest.approx(0.5)

    rng = np.random.default_rng(5)
    y, scores = ranked(rng, 0.5, 2.6, 1.0, 0.0)
    counts = confusion_counts_at_rate(y, scores, backend.predicted_positive_rate)
    assert counts is not None
    assert passes_class_aware(counts, TAU, TAU, TOLERANCE)

    flat = confusion_counts_at_rate(y, np.full(y.size, 0.4), backend.predicted_positive_rate)
    assert flat is None


def test_on_a_balanced_challenge_the_rate_rule_lands_exactly_on_se_equals_sp():
    """The rule attains ``max_t min(Se, Sp)``; it does not approximate it.

    With ``P = N`` positives and negatives and ``k = P`` rows declared positive,
    ``TP + FP = P`` and ``TP + FN = P`` force ``FP = FN``, so ``TN = N - FP = P - FN``
    and ``Sp = TN/N = (P - FN)/P = TP/P = Se``. Sensitivity falls and specificity rises
    as the threshold rises, so the two cross exactly where their minimum is largest.
    A single public rate of one half therefore places *every* prover at its own
    reachable optimum, whatever its score scale -- which is the whole difference from
    a public threshold, and the reason the measured round-0 admission moved from 1.25%
    to 80% on a cohort whose clients' optima span 0.002 to 0.729.
    """

    rng = np.random.default_rng(17)
    for separation in (0.2, 0.8, 1.5, 2.4):
        for scale, offset in ((1.0, 0.0), (0.06, 0.02), (0.3, 0.68)):
            y = np.array([0, 1] * (N // 2))
            latent = rng.normal(separation * y, 1.0)
            scores = offset + scale / (1.0 + np.exp(-latent))
            counts = confusion_counts_at_rate(y, scores, 0.5)
            assert counts is not None
            assert counts.fp == counts.fn
            assert counts.sensitivity == pytest.approx(counts.specificity, abs=1e-12)
            # And that shared value is the ROC's own best minimum.
            from xfedagent.reachability import _reachable_point

            reachable, _ = _reachable_point(y, scores)
            assert min(counts.sensitivity, counts.specificity) == pytest.approx(
                reachable, abs=1e-12
            )


def test_the_derived_rate_declares_exactly_as_many_rows_as_the_challenge_has_positives():
    """``k == P`` is what the optimality above rests on, so it holds for odd sizes too.

    Balanced stratification takes ``validation_size // 2`` positives, so on an odd
    challenge the prevalence is just under a half. Rounding that to 0.5 would declare
    one row too many and give up the exact equality for nothing.
    """

    from dataclasses import replace

    base = parse_config(_rate_config())
    for size in (100, 101, 64, 63, 33):
        cfg = replace(base, pov=replace(base.pov, validation_size=size))
        rate = resolved_predicted_positive_rate(cfg)
        assert positive_count_for_rate(rate, size) == size // 2, size


def test_the_gate_under_the_rate_rule_is_a_threshold_on_auc():
    """The two closed forms compose into a screen that needs no fit.

    The rate rule puts every prover at ``max_t min(Se, Sp)``; under binormal scores
    that quantity is ``Phi(Phi^-1(AUC)/sqrt(2))``, a function of AUC alone. So the
    class-aware gate admits a prover exactly when its AUC clears
    ``auc_floor(tau - tol)`` -- 0.6671 at the configured 0.62 -- and a published AUC
    decides admission before any model exists. Measured on the ten round-0 shards of
    the real cohort the screen agreed with the gate on all ten, across shard
    prevalences from 0.007 to 0.744.
    """

    from xfedagent.reachability import auc_floor

    floor = auc_floor(REQUIRED)
    assert floor == pytest.approx(0.6671, abs=5e-5)

    rng = np.random.default_rng(23)
    agreements = 0
    trials = 0
    for separation in np.linspace(0.1, 2.6, 14):
        y = np.array([0, 1] * (N // 2))
        # Scale and offset vary per trial: the screen must not see them.
        scale = float(rng.uniform(0.05, 1.0))
        offset = float(rng.uniform(0.0, 1.0 - scale))
        latent = rng.normal(separation * y, 1.0)
        scores = offset + scale / (1.0 + np.exp(-latent))
        counts = confusion_counts_at_rate(y, scores, 0.5)
        admitted = passes_class_aware(counts, TAU, TAU, TOLERANCE)
        auc = float(np.mean([(scores[y == 1][:, None] > scores[y == 0][None, :]).mean()]))
        screened = auc >= floor
        trials += 1
        # Disagreement is possible within a couple of points of the floor, where the
        # finite challenge and the binormal idealisation part company.
        agreements += admitted == screened or abs(auc - floor) < 0.03
    assert agreements == trials, (agreements, trials)


def test_fixing_the_count_leaves_the_confusion_matrix_one_degree_of_freedom():
    """``TP`` determines all four integers once ``k`` is public.

    ``FP = k - TP``, ``FN = P - TP``, ``TN = N - k + TP``. Both rates are then
    increasing in TP and in nothing else, so every monotone predicate over the four
    counts is a threshold on TP: fixing the rate collapses the predicate design space
    to the choice of one integer. This is the structural reason the rule closes the
    constant-classifier hole -- not that the predicate changed, but that the degree of
    freedom the constant classifier lived in is gone.
    """

    rng = np.random.default_rng(41)
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(400):
        y = np.array([1] * 14 + [0] * 86)
        rng.shuffle(y)
        scores = 1.0 / (1.0 + np.exp(-rng.normal(rng.uniform(0.2, 2.5) * y, 1.0)))
        counts = confusion_counts_at_rate(y, scores, 0.32)
        assert counts is not None
        k, p, n = positive_count_for_rate(0.32, 100), int(y.sum()), y.size
        assert counts.fp == k - counts.tp
        assert counts.fn == p - counts.tp
        assert counts.tn == (n - p) - k + counts.tp
        # Raw accuracy is then an affine function of TP, so a raw-accuracy gate is a
        # sensitivity gate: at tau - tol = 0.72 on this geometry, TP >= 9 of 14.
        assert counts.accuracy == pytest.approx(((n - p) - k + 2 * counts.tp) / n)
        seen.add((counts.tp, counts.tn, counts.fp, counts.fn))
    assert len(seen) == len({matrix[0] for matrix in seen})


def test_on_a_balanced_challenge_at_k_equals_p_every_predicate_is_the_same_predicate():
    """The cost of the rule, stated where it cannot be missed.

    With ``P = N`` and ``k = P``: ``accuracy = (N - k + 2*TP)/n = TP/P = Se = Sp``. The
    four quantities are one number, so the raw-accuracy and class-aware predicates
    coincide -- ``TP >= 31`` for both at n = 100. A paired comparison of the two
    predicates is therefore not measurable under this rule on a balanced challenge:
    the rule dissolves the distinction rather than favouring either side. Which is why
    the paper's predicate argument stays scoped to the threshold rule it was made
    about, and this rule stays out of the manuscript.
    """

    rng = np.random.default_rng(43)
    for separation in (0.2, 0.9, 1.8, 2.7):
        y = np.array([0, 1] * (N // 2))
        scores = 1.0 / (1.0 + np.exp(-rng.normal(separation * y, 1.0)))
        counts = confusion_counts_at_rate(y, scores, 0.5)
        rates = (counts.accuracy, counts.sensitivity, counts.specificity,
                 counts.balanced_accuracy)
        assert max(rates) - min(rates) == 0.0, rates
        # Both predicates therefore admit exactly the same submissions.
        from xfedagent.metrics import passes_raw_accuracy

        assert passes_class_aware(counts, REQUIRED, REQUIRED, 0.0) is passes_raw_accuracy(
            counts, REQUIRED, 0.0
        )
