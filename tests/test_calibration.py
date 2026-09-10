"""Tests for the threshold-calibration rule.

The rule has to be mechanical, because the whole point of fixing it in advance is
that the numbers cannot influence it. These tests pin each branch: a qualifying
sweep, a sweep where nothing qualifies, the deterministic tie-break, and the
trigger that decides whether a sweep is licensed at all.
"""
from __future__ import annotations

import pytest

from xfedagent.calibration import (
    FALLBACK_PAIR,
    MAX_MALICIOUS_ADMISSION,
    TRIGGER_HONEST_FALSE_REJECT,
    CalibrationCell,
    format_calibration_table,
    needs_calibration,
    parse_pairs,
    select_pair,
)


def cell(tau, honest_rejected, malicious_admitted, honest_submitted=400, malicious_submitted=180):
    return CalibrationCell(
        tau_sensitivity=tau,
        tau_specificity=tau,
        seeds=(0, 1, 2),
        honest_submitted=honest_submitted,
        honest_rejected=honest_rejected,
        malicious_submitted=malicious_submitted,
        malicious_admitted=malicious_admitted,
    )


def test_rates_are_pooled_over_submissions():
    c = cell(0.70, honest_rejected=100, malicious_admitted=9)
    assert c.honest_false_reject_rate == pytest.approx(0.25)
    assert c.malicious_admission_rate == pytest.approx(0.05)


def test_zero_denominators_do_not_divide_by_zero():
    c = CalibrationCell(0.70, 0.70, (0,), 0, 0, 0, 0)
    assert c.honest_false_reject_rate == 0.0
    assert c.malicious_admission_rate == 0.0


def test_trigger_is_strictly_above_fifteen_percent():
    assert not needs_calibration(TRIGGER_HONEST_FALSE_REJECT)
    assert needs_calibration(TRIGGER_HONEST_FALSE_REJECT + 1e-9)
    assert needs_calibration(0.28)
    assert not needs_calibration(0.023)


def test_lowest_false_reject_wins_among_admissible_pairs():
    cells = [
        cell(0.65, honest_rejected=40, malicious_admitted=0),    # 10.0%, admissible
        cell(0.70, honest_rejected=112, malicious_admitted=0),   # 28.0%, admissible
        cell(0.75, honest_rejected=200, malicious_admitted=0),   # 50.0%, admissible
    ]
    decision = select_pair(cells)
    assert decision["chosen"] == (0.65, 0.65)
    assert decision["fell_back"] is False
    assert len(decision["qualifying_pairs"]) == 3


def test_a_pair_that_admits_too_much_is_excluded_even_if_it_rejects_least():
    # 0.65 has the lowest honest false-reject but admits 10% of malicious updates,
    # so the rule must pass it over rather than reward it.
    cells = [
        cell(0.65, honest_rejected=20, malicious_admitted=18),   # 5.0% honest, 10% admission
        cell(0.70, honest_rejected=112, malicious_admitted=9),   # 28.0% honest, 5% admission
    ]
    decision = select_pair(cells)
    assert decision["chosen"] == (0.70, 0.70)
    assert decision["qualifying_pairs"] == ["0.70/0.70"]


def test_admission_ceiling_is_inclusive():
    at_ceiling = cell(0.70, honest_rejected=112, malicious_admitted=9)
    assert at_ceiling.malicious_admission_rate == pytest.approx(MAX_MALICIOUS_ADMISSION)
    assert select_pair([at_ceiling])["chosen"] == (0.70, 0.70)


def test_falls_back_to_the_operating_point_when_nothing_qualifies():
    cells = [
        cell(0.65, honest_rejected=20, malicious_admitted=30),
        cell(0.70, honest_rejected=112, malicious_admitted=20),
        cell(0.75, honest_rejected=200, malicious_admitted=18),
    ]
    decision = select_pair(cells)
    assert decision["chosen"] == FALLBACK_PAIR
    assert decision["fell_back"] is True
    assert decision["qualifying_pairs"] == []
    # The trade-off must be stated, not left implicit.
    assert "exceeds" in decision["trade_off"]
    assert "10.0%" in decision["trade_off"]      # best observed admission, 18/180


def test_fallback_says_so_when_the_operating_point_was_not_swept():
    cells = [cell(0.60, honest_rejected=10, malicious_admitted=40)]
    decision = select_pair(cells)
    assert decision["chosen"] == FALLBACK_PAIR
    assert "was not among the swept pairs" in decision["trade_off"]


def test_ties_break_towards_the_stricter_pair():
    cells = [
        cell(0.65, honest_rejected=40, malicious_admitted=0),
        cell(0.75, honest_rejected=40, malicious_admitted=0),
    ]
    assert select_pair(cells)["chosen"] == (0.75, 0.75)


def test_empty_sweep_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        select_pair([])


def test_parse_pairs_accepts_symmetric_and_asymmetric_forms():
    assert parse_pairs("0.65,0.70,0.75") == ((0.65, 0.65), (0.70, 0.70), (0.75, 0.75))
    assert parse_pairs("0.65:0.80") == ((0.65, 0.80),)
    with pytest.raises(ValueError):
        parse_pairs("")


def test_latex_table_marks_the_chosen_row():
    cells = [cell(0.65, 40, 0), cell(0.70, 112, 0)]
    manifest = {
        "cells": [c.to_row() for c in cells],
        "chosen_pair": [0.65, 0.65],
    }
    body = format_calibration_table(manifest)
    lines = body.splitlines()
    assert r"$^{\dagger}$" in lines[0]
    assert r"$^{\dagger}$" not in lines[1]
    assert "10.0" in lines[0]        # honest false-reject, percent
