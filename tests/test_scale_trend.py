"""Tests for the agent-count trend test.

Jonckheere--Terpstra is used because the alternative of interest is monotone in
an ordered factor. The risk is a checker that reports a trend from noise, or one
whose sign convention is reversed -- which would have confirmed the manuscript's
withdrawn claim that detection improves with population size. Both directions are
pinned on constructed data before the real sweep is trusted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scale_trend", ROOT / "scripts" / "scale_trend.py")
trend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trend)


def test_a_strictly_increasing_sweep_is_significant_and_positive():
    groups = [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]]
    u, z, p = trend.jonckheere(groups)
    assert z > 0
    assert p < 0.01
    assert u == 54  # every cross-group pair concordant: 6 pairs x 9 comparisons


def test_the_sign_reverses_with_the_order():
    rising = [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
    falling = list(reversed(rising))
    _, z_up, p_up = trend.jonckheere(rising)
    _, z_down, p_down = trend.jonckheere(falling)
    assert z_up == pytest.approx(-z_down)
    assert p_up == pytest.approx(p_down)


def test_a_flat_sweep_is_not_reported_as_a_trend():
    groups = [[1.0, 2.0, 3.0]] * 4
    _, z, p = trend.jonckheere(groups)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p > 0.9


def test_ties_count_as_half():
    _, z, _ = trend.jonckheere([[1.0], [1.0]])
    assert z == pytest.approx(0.0)
