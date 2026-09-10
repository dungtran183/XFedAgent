"""Tests for the seed-level paired statistics."""
from __future__ import annotations

import csv
import math

import pytest

from xfedagent.statistics import (
    PairedComparison,
    compare_arms,
    compare_family,
    holm_adjust,
)


def _write_runs(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "seed", "accuracy", "auc_roc"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def runs_csv(tmp_path):
    path = tmp_path / "ablation_runs.csv"
    rows = []
    for seed in range(10):
        # `full` beats `no-pov` on every seed, by a margin that varies with seed.
        rows.append({"arm": "full", "seed": seed, "accuracy": 0.84 + 0.002 * seed, "auc_roc": 0.86})
        rows.append({"arm": "no-pov", "seed": seed, "accuracy": 0.63 + 0.002 * seed, "auc_roc": 0.71})
        rows.append({"arm": "no-rep", "seed": seed, "accuracy": 0.71 + 0.001 * seed, "auc_roc": 0.79})
    _write_runs(path, rows)
    return path


def test_holm_is_monotone_and_at_least_as_large_as_the_raw_p():
    raw = [0.001, 0.02, 0.04]
    adjusted = holm_adjust(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert adjusted == sorted(adjusted)


def test_holm_matches_the_closed_form_on_three_hypotheses():
    # Holm multiplies the k-th smallest p by (m - k + 1), then enforces monotonicity.
    raw = [0.01, 0.02, 0.03]
    assert holm_adjust(raw) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_never_exceeds_one():
    assert all(p <= 1.0 for p in holm_adjust([0.6, 0.7, 0.8]))


def test_holm_of_empty_family_is_empty():
    assert holm_adjust([]) == []


def test_comparison_pairs_on_shared_seeds(runs_csv):
    comparison = compare_arms(runs_csv, "accuracy", "full", "no-pov")
    assert comparison.seeds == tuple(range(10))
    assert comparison.differences.size == 10


def test_unpaired_arms_are_rejected(tmp_path):
    path = tmp_path / "runs.csv"
    _write_runs(
        path,
        [
            {"arm": "full", "seed": 0, "accuracy": 0.8, "auc_roc": 0.9},
            {"arm": "no-pov", "seed": 1, "accuracy": 0.6, "auc_roc": 0.7},
        ],
    )
    with pytest.raises(ValueError, match="not paired"):
        compare_arms(path, "accuracy", "full", "no-pov")


def test_effect_size_is_the_paired_form(runs_csv):
    summary = compare_arms(runs_csv, "accuracy", "full", "no-pov").summary()
    # Differences are a constant 0.21 here, so sd(difference) is 0 and d_z is
    # undefined; the summary must say so rather than reporting a finite value.
    assert math.isclose(summary["mean_difference"], 0.21, abs_tol=1e-9)
    assert math.isnan(summary["effect_size_dz"])
    assert "paired" in summary["effect_size_form"]


def test_seed_is_the_unit_of_analysis(runs_csv):
    summary = compare_arms(runs_csv, "accuracy", "full", "no-rep").summary()
    assert summary["n_seeds"] == 10
    assert "seed" in summary["test"]


def test_family_records_its_own_membership(tmp_path, runs_csv):
    manifest = compare_family(
        runs_csv,
        [("accuracy", "full", "no-pov"), ("accuracy", "full", "no-rep")],
        tmp_path / "stats",
        family_label="gate-vs-reputation",
    )
    assert manifest["family"] == "gate-vs-reputation"
    assert len(manifest["family_members"]) == 2
    for row in manifest["comparisons"]:
        assert row["family_size"] == 2
        assert row["p_value_holm"] >= row["p_value"]


def test_identical_arms_give_no_significant_difference(tmp_path):
    path = tmp_path / "runs.csv"
    rows = []
    for seed in range(6):
        rows.append({"arm": "a", "seed": seed, "accuracy": 0.8, "auc_roc": 0.9})
        rows.append({"arm": "b", "seed": seed, "accuracy": 0.8, "auc_roc": 0.9})
    _write_runs(path, rows)
    summary = compare_arms(path, "accuracy", "a", "b").summary()
    assert summary["mean_difference"] == 0.0
    assert summary["p_value"] == 1.0


def test_wilcoxon_detects_a_consistent_paired_shift(tmp_path):
    path = tmp_path / "runs.csv"
    rows = []
    for seed in range(10):
        rows.append({"arm": "a", "seed": seed, "accuracy": 0.80 + 0.01 * seed, "auc_roc": 0.9})
        rows.append({"arm": "b", "seed": seed, "accuracy": 0.75 + 0.012 * seed, "auc_roc": 0.9})
    _write_runs(path, rows)
    summary = compare_arms(path, "accuracy", "a", "b").summary()
    assert summary["p_value"] < 0.05
    assert summary["mean_difference"] > 0
    assert summary["ci_low"] > 0


def test_paired_comparison_accepts_direct_construction():
    comparison = PairedComparison(
        metric="accuracy",
        arm_a="x",
        arm_b="y",
        seeds=(0, 1, 2),
        values_a=(0.9, 0.8, 0.85),
        values_b=(0.7, 0.65, 0.75),
    )
    summary = comparison.summary()
    assert summary["n_seeds"] == 3
    assert summary["mean_difference"] > 0
