"""Empirical gate diagnostics and the assumptions of their analytic screens."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from xfedagent.config import load_config, parse_config
from xfedagent.metrics import ConfusionCounts, confusion_counts
from xfedagent.reachability import (
    _gate_verdict,
    _reachable_point,
    auc_floor,
    format_reachability_table,
    measure_reachability,
    predicted_reachable_min,
)


def test_separable_scores_are_fully_reachable():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.01, 0.30, 50), np.linspace(0.70, 0.99, 50)])
    reachable, threshold = _reachable_point(y, scores)
    assert reachable == pytest.approx(1.0)
    assert 0.30 < threshold <= 0.70


@pytest.fixture(autouse=True)
def no_training_in_diagnostic_tests(monkeypatch):
    """Exercise scoring/report logic without fitting a pooled or federated model."""
    from types import SimpleNamespace
    from xfedagent.model import TorchModel

    monkeypatch.setattr(TorchModel, "train_local",
        lambda self, state, *args, **kwargs: SimpleNamespace(state=state))


def test_a_ranked_model_below_the_conventional_threshold_is_still_reachable():
    """The measured failure mode: good ranking, all of it under 0.5.

    ``confusion_counts`` at 0.5 reports zero sensitivity, so the gate rejects. The
    same scores are perfectly separable, which is what makes "too weak" the wrong
    diagnosis and the threshold the thing to state.
    """
    y = np.array([1] * 20 + [0] * 80)
    scores = np.concatenate([np.linspace(0.26, 0.40, 20), np.linspace(0.02, 0.24, 80)])
    assert confusion_counts(y, scores, 0.5).sensitivity == 0.0
    reachable, threshold = _reachable_point(y, scores)
    assert reachable == pytest.approx(1.0)
    assert threshold < 0.5


def test_a_single_class_challenge_carries_no_information():
    reachable, threshold = _reachable_point(np.zeros(20, dtype=int), np.random.default_rng(0).random(20))
    assert reachable == 0.0 and threshold == 0.5


def test_reachable_point_is_an_upper_bound_on_every_fixed_threshold():
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.2).astype(int)
    scores = np.clip(0.15 * y + rng.normal(0.2, 0.1, 200), 0.0, 1.0)
    reachable, _ = _reachable_point(y, scores)
    for threshold in np.linspace(0.05, 0.95, 19):
        counts = confusion_counts(y, scores, float(threshold))
        assert min(counts.sensitivity, counts.specificity) <= reachable + 1e-9


def test_the_verdict_reads_the_configured_predicate():
    cfg = parse_config(json.loads(pathlib.Path("configs/full.json").read_text()))
    object.__setattr__(cfg.pov, "predicate", "class_aware")
    object.__setattr__(cfg.pov, "threshold_sensitivity", 0.65)
    object.__setattr__(cfg.pov, "threshold_specificity", 0.65)
    object.__setattr__(cfg.pov, "tolerance", 0.0)
    majority = ConfusionCounts(tp=0, tn=86, fp=0, fn=14)
    useful = ConfusionCounts(tp=11, tn=70, fp=16, fn=3)
    assert not _gate_verdict(cfg, majority)
    assert _gate_verdict(cfg, useful)
    # The raw-accuracy gate is the one the majority classifier walks through.
    object.__setattr__(cfg.pov, "predicate", "raw_accuracy")
    object.__setattr__(cfg.pov, "threshold", 0.75)
    assert _gate_verdict(cfg, majority)


def test_end_to_end_reports_both_bounds_and_a_verdict(tmp_path):
    cfg = load_config("configs/synthetic-smoke.json")
    manifest = measure_reachability(cfg, challenges=2, output=str(tmp_path))
    assert manifest["verdict"] in {
        "observed_admission",
        "threshold_sensitive_reference",
        "elevated_empty_round_risk",
        "no_admission_observed",
    }
    labels = [row["bound"] for row in manifest["bounds"]]
    assert labels == ["cold_start", "pooled_reference"]
    assert manifest["diagnostic_only"] is True
    matrix = np.asarray(manifest["cold_start_pass_matrix"], dtype=bool)
    assert matrix.shape == (cfg.data.clients, 2)
    if cfg.federation.clients_per_round == cfg.data.clients:
        assert manifest["round0_stall_probability"] == pytest.approx(np.mean(~matrix.any(axis=0)))
    assert "binormal" in manifest["auc_floor_assumption"]
    for row in manifest["bounds"]:
        assert 0.0 <= row["pass_rate"] <= 1.0
        assert 0.0 <= row["reachable_min_se_sp"] <= 1.0
        assert row["trials"] > 0
    assert (tmp_path / "reachability.csv").exists()
    assert (tmp_path / "reachability_manifest.json").exists()
    assert format_reachability_table(manifest).count("\\\\") == 2
    # The held-out test set must not be consulted by a pre-flight: no key or bound
    # column reports anything measured on it.
    reported = {key for key in manifest if key != "csv"} | {
        column for row in manifest["bounds"] for column in row
    }
    assert not any("test" in name for name in reported)


def test_the_closed_form_agrees_with_the_measured_reachable_point():
    """On binormal scores the AUC-only prediction should track the measurement.

    The point of the closed form is that it needs no fit: it turns a published AUC
    into the best the class-aware predicate could do. Here the generating process
    really is binormal, so agreement is the specification, not a coincidence -- and
    it must hold across prevalences, because the formula does not involve one.
    """

    rng = np.random.default_rng(3)
    worst = 0.0
    for prevalence in (0.10, 0.14, 0.35, 0.50):
        for separation in (0.3, 0.9, 1.6, 2.6):
            labels = (rng.random(20000) < prevalence).astype(int)
            scores = rng.normal(separation * labels, 1.0)
            measured, _ = _reachable_point(labels, scores)
            predicted = predicted_reachable_min(roc_auc_score(labels, scores))
            worst = max(worst, abs(measured - predicted))
    assert worst < 0.02, worst


def test_auc_floor_inverts_the_prediction_and_brackets_the_requirement():
    """``auc_floor`` is the AUC at which the predicted reachable point equals it."""

    for required in (0.55, 0.62, 0.70, 0.85):
        floor = auc_floor(required)
        assert predicted_reachable_min(floor) == pytest.approx(required, abs=1e-6)
        # A gate is never easier than the rate it demands, and demanding both rates
        # at once is strictly harder than the rate alone unless it is a half.
        assert floor >= required
    assert auc_floor(0.5) == pytest.approx(0.5, abs=1e-9)


def test_a_weak_binormal_score_example_fails_the_threshold_search():
    """One binormal example; this does not assert a distribution-free exclusion."""

    rng = np.random.default_rng(4)
    labels = (rng.random(8000) < 0.14).astype(int)
    # Separation chosen so AUC lands under the floor for 0.62/0.62.
    scores = rng.normal(0.25 * labels, 1.0)
    auc = roc_auc_score(labels, scores)
    assert auc < auc_floor(0.62)
    best = max(
        min(
            confusion_counts(labels, scores, threshold).sensitivity,
            confusion_counts(labels, scores, threshold).specificity,
        )
        for threshold in np.linspace(0.01, 0.99, 99)
    )
    assert best < 0.62


def test_every_bound_reports_the_prediction_beside_the_measurement(tmp_path):
    manifest = measure_reachability(
        load_config("configs/synthetic-smoke.json"), challenges=2, output=str(tmp_path)
    )
    assert 0.0 < manifest["auc_floor"] < 1.0
    for row in manifest["bounds"]:
        assert 0.0 <= row["predicted_min_se_sp"] <= 1.0


def test_a_misplaced_threshold_is_not_reported_as_an_unreachable_cohort():
    """The two ways a ceiling can fail have to be told apart.

    Both produce ``pass_rate = 0`` on the ceiling, and before this distinction
    existed both were reported as ``cohort_cannot_reach_thresholds`` -- which sends
    the reader looking for a better cohort when the fix was to name the threshold
    the scores actually live at.
    """

    required = 0.62
    # A ranked model whose scores all sit below the configured 0.5: at that
    # threshold nothing is called positive, so sensitivity is zero, yet the ranking
    # separates the classes well enough to clear both rates somewhere.
    labels = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([np.linspace(0.01, 0.14, 500), np.linspace(0.16, 0.30, 500)])
    at_half = confusion_counts(labels, scores, 0.5)
    assert at_half.sensitivity == 0.0
    reachable, _ = _reachable_point(labels, scores)
    assert reachable >= required
    # So the diagnosis is the threshold, and the AUC screen agrees the cohort is fine.
    assert roc_auc_score(labels, scores) >= auc_floor(required)


def test_the_two_ceiling_failures_are_distinguished_by_the_reachable_point():
    """Same pass_rate, opposite diagnosis, decided by reachable_min alone."""

    required = 0.62
    rng = np.random.default_rng(5)
    labels = (rng.random(6000) < 0.14).astype(int)
    # Too weak to pass anywhere.
    weak = rng.normal(0.2 * labels, 1.0)
    assert _reachable_point(labels, weak)[0] < required
    # Strong enough somewhere, but shifted off the configured threshold.
    strong = rng.normal(1.6 * labels, 1.0) * 0.05 + 0.10
    assert _reachable_point(labels, strong)[0] >= required
    assert confusion_counts(labels, strong, 0.5).sensitivity == 0.0


def test_general_monotone_roc_attains_tau_squared_below_the_binormal_screen():
    """A concrete nonconcave ROC refutes the former universal AUC >= tau claim."""
    for required in (0.55, 0.62, 0.70, 0.75):
        high = round(required * 100)
        labels = np.array([0] * (100 - high) + [1] * high + [0] * high + [1] * (100 - high))
        scores = np.array([0.9] * (100 - high) + [0.8] * high + [0.2] * high + [0.1] * (100 - high))
        counts = confusion_counts(labels, scores, 0.5)
        assert counts.sensitivity == counts.specificity == pytest.approx(required)
        assert roc_auc_score(labels, scores) == pytest.approx(required ** 2)
        assert roc_auc_score(labels, scores) < auc_floor(required)
        # The stronger tau area requires concavity of the ROC.
        polyline = np.trapezoid([0.0, required, 1.0], [0.0, 1.0 - required, 1.0])
        assert polyline == pytest.approx(required, abs=1e-12)
        assert auc_floor(required) > required
    assert auc_floor(0.62) == pytest.approx(0.6671, abs=5e-4)


def test_empty_round_probability_retains_shared_challenge_dependence():
    from xfedagent.reachability import Bound

    def bound(matrix) -> Bound:
        return Bound(
            label="cold_start", trials=80, train_rows=1914, train_prevalence=0.14,
            validation_rows=100, auc=0.70, sensitivity=0.2, specificity=0.9,
            passes=float(np.mean(matrix)) if len(matrix) else 0.0,
            reachable_min=0.649, reachable_threshold=0.112, mean_score=0.1,
            pass_matrix=tuple(tuple(row) for row in matrix),
        )

    correlated = bound([[True, False]] * 4)
    complementary = bound([[True, False], [False, True]] * 2)
    assert correlated.passes == complementary.passes == 0.5
    assert correlated.stall_probability(4) == 0.5
    assert complementary.stall_probability(4) == 0.0
    assert complementary.stall_probability(2) == pytest.approx(1 / 6)
    assert correlated.stall_probability(4) != (1 - correlated.passes) ** 4
    with pytest.raises(ValueError, match="population"):
        correlated.stall_probability(5)
    with pytest.raises(ValueError, match="matrix"):
        bound([]).stall_probability(1)


def test_the_suggested_threshold_serves_the_binding_bound(tmp_path, synthetic_config):
    """Read off the ceiling in both cases, the suggestion can serve nobody.

    Once the ceiling passes the gate, the run waits on the cold start, so the cold
    start's own operating point is the one to report. On the MIMIC cohort the ceiling
    wanted 0.167 and the round-0 clients wanted 0.112; recommending 0.167 is how a
    threshold that admitted 1.25% of round-0 submissions came to be written into a
    config as the pre-flight's suggestion.
    """

    manifest = measure_reachability(synthetic_config, challenges=2, output=str(tmp_path))
    bounds = {row["bound"]: row for row in manifest["bounds"]}
    assert manifest["binding_bound"] in bounds
    expected = "pooled_reference" if bounds["pooled_reference"]["pass_rate"] == 0.0 else "cold_start"
    assert manifest["binding_bound"] == expected
    assert manifest["suggested_decision_threshold"] == pytest.approx(
        round(bounds[expected]["reachable_threshold"], 4)
    )
    # Candidate separation is reported without asserting disjoint passing sets.
    gap = abs(bounds["pooled_reference"]["reachable_threshold"] - bounds["cold_start"]["reachable_threshold"])
    assert manifest["reachable_threshold_gap"] == pytest.approx(gap, abs=1e-6)
    assert manifest["threshold_comparison_is_heuristic"] is True
    # The alternative travels with the diagnosis rather than needing a second run.
    assert 0.0 < manifest["suggested_predicted_positive_rate"] < 1.0


def test_the_cold_start_reports_one_shard_and_not_the_concatenation(tmp_path, synthetic_config):
    """``train_rows`` is what the bound is about: the rows one round-0 model saw.

    Reporting the pooled 19,140 beside a docstring that says "a single client's shard"
    is how a Dirichlet partition spanning 121 to 5,274 rows goes unread. The pooled
    figure is kept, under its own name.
    """

    manifest = measure_reachability(synthetic_config, challenges=2, output=str(tmp_path))
    cold, ceiling = manifest["bounds"]
    assert cold["states"] == synthetic_config.data.clients
    assert cold["train_rows"] < cold["train_rows_total"]
    assert cold["train_rows_total"] == ceiling["train_rows_total"] == ceiling["train_rows"]
    assert ceiling["states"] == 1
    assert 0 <= cold["states_never_admitted"] <= cold["states"]


def test_the_preflight_measures_the_rule_the_run_will_use(tmp_path, synthetic_config):
    """A pre-flight that measured a threshold gate for a rate run would answer nothing.

    Same cohort, same fits, same challenges; only the binarisation rule differs. The
    manifest has to say which one it measured, and the witness-failure column has to
    exist so a run refused for want of a witness is not read as a cohort that cannot
    meet its thresholds.
    """

    from dataclasses import replace

    threshold_run = measure_reachability(synthetic_config, challenges=2, output=str(tmp_path / "t"))
    assert threshold_run["decision_rule"] == "threshold"

    rate_cfg = replace(
        synthetic_config,
        pov=replace(
            synthetic_config.pov, decision_rule="rate", predicate="class_aware",
            balanced_validation=True,
        ),
    )
    rate_run = measure_reachability(rate_cfg, challenges=2, output=str(tmp_path / "r"))
    assert rate_run["decision_rule"] == "rate"
    assert rate_run["predicted_positive_rate"] == pytest.approx(0.5)
    for row in rate_run["bounds"]:
        # Every trial either produced a witness or is counted as having failed to.
        assert 0 <= row["witness_failures"] <= row["trials"]
        assert row["pass_rate"] <= 1.0 - row["witness_failures"] / row["trials"]


def test_the_preflight_reports_the_honest_false_reject_rate_it_implies(tmp_path, synthetic_config):
    """The stall test and the false-reject trigger answer different questions.

    ``round0_stall_probability`` asks whether the run can begin; the pre-registered
    ``TRIGGER_HONEST_FALSE_REJECT`` asks what fraction of honest work is thrown away
    while it runs. A cohort can pass the first and fail the second, and the measured
    case is exactly that: the raw-accuracy arm under the rate rule stalled with
    probability 0.005 -- comfortably reachable -- while its cold-start pass rate of
    0.41 implied a 0.59 honest false-reject rate, and the full run measured 0.55 to
    0.68 and accepted 8, 1 and 5 updates of 600 as the reputation penalty compounded.
    Both numbers now leave the pre-flight.
    """

    from xfedagent.calibration import TRIGGER_HONEST_FALSE_REJECT

    manifest = measure_reachability(synthetic_config, challenges=2, output=str(tmp_path))
    cold = next(row for row in manifest["bounds"] if row["bound"] == "cold_start")
    assert manifest["implied_honest_false_reject"] == pytest.approx(
        1.0 - cold["pass_rate"], abs=1e-6
    )
    assert manifest["trigger_honest_false_reject"] == TRIGGER_HONEST_FALSE_REJECT
    assert manifest["honest_false_reject_above_trigger"] is (
        manifest["implied_honest_false_reject"] > TRIGGER_HONEST_FALSE_REJECT
    )
    # The two readings are independent: a low stall probability does not bound the
    # false-reject rate, which is the whole reason both are reported.
    assert 0.0 <= manifest["implied_honest_false_reject"] <= 1.0


def test_threshold_search_uses_both_asymmetric_requirements(synthetic_config):
    from dataclasses import replace
    from xfedagent.reachability import _threshold_can_pass

    cfg = replace(synthetic_config, pov=replace(synthetic_config.pov,
        predicate="class_aware", threshold_sensitivity=0.45,
        threshold_specificity=0.95, tolerance=0.0))
    y = np.array([1] * 10 + [0] * 10)
    scores = np.array([0.4] * 5 + [0.1] * 5 + [0.2] * 10)
    assert _reachable_point(y, scores)[0] < 0.95
    assert _threshold_can_pass(cfg, y, scores)
