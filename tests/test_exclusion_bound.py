"""Tests for the Theorem 2 verification script.

The script's job is to be able to *fail*. A checker that reports agreement
regardless of the recursion it is given would have signed off on the bound the
response letter originally quoted, which is violated on four fifths of the rates
it covers. These tests therefore pin the direction of each check: q* must move
when the parameters move, the published form must be shown failing, and the
hypothesis-free form must be shown holding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_exclusion_bound", ROOT / "scripts" / "verify_exclusion_bound.py"
)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


@pytest.fixture
def rec():
    return verify.Recursion(verify.parameters(ROOT / "configs" / "predicate-class-aware.json"))


def test_delta_max_follows_from_the_predicate_not_from_a_constant():
    """Under the class-aware gate the largest margin is 1 - max(tau_se, tau_sp).

    Hard-coding 0.35 would make the checker agree with the manuscript by
    construction on any config, including one whose thresholds had moved.
    """
    p = verify.parameters(ROOT / "configs" / "predicate-class-aware.json")
    assert p["predicate"] == "class_aware"
    assert p["delta_max"] == pytest.approx(0.35)
    raw = verify.parameters(ROOT / "configs" / "predicate-raw-accuracy.json")
    assert raw["predicate"] == "raw_accuracy"
    assert raw["delta_max"] == pytest.approx(0.25)


def test_qstar_matches_the_closed_form(rec):
    assert rec.q_star == pytest.approx(0.0654205607476635, abs=1e-12)


def test_the_bisected_boundary_lands_on_qstar(rec):
    boundary = verify.bisect_boundary(rec, rounds=20000)
    assert boundary == pytest.approx(rec.q_star, rel=1e-3)


def test_a_single_detection_at_the_initial_reputation_excludes(rec):
    """beta = r_0 in the shipped config, so the first penalty is fatal.

    This is the direct answer to ``adaptive attackers maintain their reputation'':
    they cannot, until they have banked to r_min + beta first.
    """
    assert rec.step(rec.r0, detected=True) < rec.r_min
    assert rec.replay([1]) == 1
    banked = rec.r_min + rec.beta
    assert banked == pytest.approx(0.7)


def test_the_published_form_is_shown_failing_and_the_slack_form_holding(rec):
    report = verify.check_bounds(rec, grid=200, rounds=20000)
    assert report["rates_excluded"] == 200
    assert report["published_violations"] > 0
    assert report["slack_violations"] == 0
    assert report["slack_mean_looseness"] > 1.0


def test_order_independence_is_checked_in_both_directions(rec):
    """Above q* nothing survives; below it something must, or the test is vacuous."""
    report = verify.check_order_independence(rec, trials=4000)
    assert report["survivors_above_qstar"] == 0
    assert report["survivors_below_qstar"] > 0
