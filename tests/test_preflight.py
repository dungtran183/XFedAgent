"""Preflight has to demonstrate the cohort, not assume it.

The preflight report states the prevalence, confirms that no ``subject_id``
straddles two splits, and stops rather than runs when the prevalence has drifted
away from the cohort the paper describes.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from xfedagent.cli import PREVALENCE_TOLERANCE, preflight_checks


def _cohort(path, patients: int = 40, positives: int = 10) -> None:
    rows = []
    for patient_id in range(patients):
        label = 1 if patient_id < positives else 0
        rows.append({"patient_id": patient_id, "label": label, "t00_f00": float(patient_id)})
    pd.DataFrame(rows).to_csv(path, index=False)


def _csv_config(synthetic_config, path, *, positive_rate: float = 0.25):
    data = replace(
        synthetic_config.data,
        source="csv_timeseries",
        csv_path=str(path),
        timesteps=1,
        features=1,
        clients=4,
        validation_pool_size=6,
        validation_fraction=0.2,
        test_fraction=0.2,
        positive_rate=positive_rate,
    )
    return replace(synthetic_config, data=data)


def test_preflight_reports_prevalence_and_a_patient_disjoint_split(synthetic_config, tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    _cohort(path)
    checks, cohort = preflight_checks(_csv_config(synthetic_config, path))
    assert checks["dataset"] is True
    assert checks["dataset_patient_disjoint"] is True
    assert checks["dataset_prevalence_matches_config"] is True
    assert cohort["prevalence"] == pytest.approx(0.25)
    assert cohort["patients"] == 40
    assert cohort["shared_patients"] == 0
    assert cohort["train_patients"] + cohort["validation_patients"] + cohort["test_patients"] == 40


def test_preflight_fails_when_the_cohort_prevalence_has_drifted(synthetic_config, tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    _cohort(path, positives=30)
    cfg = _csv_config(synthetic_config, path, positive_rate=0.25)
    checks, cohort = preflight_checks(cfg)
    assert cohort["prevalence"] == pytest.approx(0.75)
    assert checks["dataset_prevalence_matches_config"] is False


def test_prevalence_tolerance_admits_a_rebuild_of_the_same_cohort(synthetic_config, tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    _cohort(path, patients=100, positives=25)
    cfg = _csv_config(synthetic_config, path, positive_rate=0.25 + PREVALENCE_TOLERANCE / 2)
    checks, _ = preflight_checks(cfg)
    assert checks["dataset_prevalence_matches_config"] is True


def test_preflight_flags_a_missing_cohort_without_touching_the_disk(synthetic_config, tmp_path) -> None:
    cfg = _csv_config(synthetic_config, tmp_path / "absent.csv")
    checks, cohort = preflight_checks(cfg)
    assert checks["dataset"] is False
    assert cohort == {}


def test_synthetic_cohort_needs_no_dataset_checks(synthetic_config) -> None:
    checks, cohort = preflight_checks(synthetic_config)
    assert not any(name.startswith("dataset") for name in checks)
    assert cohort == {}
