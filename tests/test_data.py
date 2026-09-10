from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xfedagent.config import DataConfig
from xfedagent.data import audit_csv_dataset, audit_csv_splits, csv_timeseries_dataset


def _config(path: str, *, clients: int = 4) -> DataConfig:
    return DataConfig(
        source="csv_timeseries",
        samples=60,
        clients=clients,
        timesteps=1,
        features=1,
        validation_pool_size=8,
        validation_fraction=0.15,
        test_fraction=0.15,
        dirichlet_alpha=0.5,
        csv_path=path,
        label_column="label",
        patient_column="subject_id",
    )


def test_csv_split_keeps_each_patient_in_one_partition(tmp_path) -> None:
    rows = []
    for patient_id in range(30):
        for _ in range(2):
            rows.append(
                {
                    "subject_id": patient_id,
                    "label": patient_id % 2,
                    # The feature lets the test recover membership without adding
                    # patient identifiers to experiment outputs.
                    "feature": float(patient_id),
                }
            )
    path = tmp_path / "cohort.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    bundle = csv_timeseries_dataset(_config(str(path)), seed=7)
    memberships: list[set[int]] = [
        set(bundle.test_x[:, 0, 0]),
        set(bundle.validation_pool_x[:, 0, 0]),
    ]
    memberships.extend(
        set(client.train_x[:, 0, 0]) for client in bundle.clients
    )

    for left in range(len(memberships)):
        for right in range(left + 1, len(memberships)):
            assert memberships[left].isdisjoint(memberships[right])
    assert len(set().union(*memberships)) == 30


def test_csv_split_keeps_mixed_outcome_admissions_for_a_patient_together(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(12), 2),
            "label": np.tile([0, 1], 12),
            "feature": np.repeat(np.arange(12), 2),
        }
    )
    path = tmp_path / "cohort.csv"
    frame.to_csv(path, index=False)

    bundle = csv_timeseries_dataset(_config(str(path), clients=2), seed=0)
    memberships = [
        set(bundle.test_x[:, 0, 0]),
        set(bundle.validation_pool_x[:, 0, 0]),
        *(set(client.train_x[:, 0, 0]) for client in bundle.clients),
    ]
    for left in range(len(memberships)):
        for right in range(left + 1, len(memberships)):
            assert memberships[left].isdisjoint(memberships[right])


def test_csv_split_requires_patient_column(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    pd.DataFrame({"label": [0, 1], "feature": [0.0, 1.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing patient column"):
        csv_timeseries_dataset(replace(_config(str(path)), clients=1), seed=0)


def test_csv_audit_reports_only_aggregate_cohort_properties(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "subject_id": np.arange(12),
            "label": [0] * 9 + [1] * 3,
            "feature": np.arange(12, dtype=float),
        }
    ).to_csv(path, index=False)

    report = audit_csv_dataset(_config(str(path), clients=2))

    assert report["rows"] == 12
    assert report["patients"] == 12
    assert report["positives"] == 3
    assert report["prevalence"] == pytest.approx(0.25)
    assert report["schema_valid"] is True
    assert report["patients_with_mixed_labels"] == 0
    assert "patient_ids" not in report


def _write_cohort(path: Path, patients: int = 40, rows_each: int = 2) -> None:
    rows = []
    for patient_id in range(patients):
        for _ in range(rows_each):
            rows.append(
                {
                    "subject_id": patient_id,
                    "label": patient_id % 2,
                    "feature": float(patient_id),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_audit_csv_splits_reports_a_patient_disjoint_split(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    _write_cohort(path)
    report = audit_csv_splits(_config(str(path)), seed=0)
    assert report["patient_disjoint"] is True
    assert report["shared_patients"] == 0
    assert report["rows_assigned_once"] is True
    assert report["validation_pool_satisfied"] is True
    assert report["rows"] == 80
    assert (
        report["train_rows"] + report["validation_rows"] + report["test_rows"] == report["rows"]
    )
    # Aggregate figures only: nothing in the report identifies a patient.
    assert not any(isinstance(value, str) for value in report.values())


def test_audit_csv_splits_matches_the_split_the_loader_actually_uses(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    _write_cohort(path)
    cfg = _config(str(path))
    report = audit_csv_splits(cfg, seed=3)
    bundle = csv_timeseries_dataset(cfg, seed=3)
    assert report["test_rows"] == len(bundle.test_y)
    assert report["train_rows"] == sum(len(client.train_y) for client in bundle.clients)


def test_changing_heldout_features_cannot_change_training_normalisation(tmp_path):
    from xfedagent.data import _patient_disjoint_split, _split_counts

    path = tmp_path / "cohort.csv"
    _write_cohort(path)
    cfg, seed = _config(str(path)), 7
    before = csv_timeseries_dataset(cfg, seed)
    frame = pd.read_csv(path)
    test_count, validation_count = _split_counts(cfg, len(frame))
    train, val, test = _patient_disjoint_split(frame.subject_id.astype(str).to_numpy(),
        test_count=test_count, validation_count=validation_count,
        minimum_train_patients=cfg.clients, rng=np.random.default_rng(seed))
    frame.loc[np.concatenate([val, test]), "feature"] += 1000000.0
    frame.to_csv(path, index=False)
    after = csv_timeseries_dataset(cfg, seed)
    for a, b in zip(before.clients, after.clients, strict=True):
        assert np.array_equal(a.train_x, b.train_x)
        assert np.array_equal(a.train_y, b.train_y)
    pooled_train = np.concatenate([c.train_x for c in after.clients])
    assert pooled_train.mean() == pytest.approx(0.0, abs=1e-7)
    assert pooled_train.std() == pytest.approx(1.0, abs=1e-7)
    assert np.all(after.test_x > before.test_x)


def test_synthetic_standardisation_is_fitted_after_the_split(synthetic_config):
    from xfedagent.data import synthetic_mimic_dataset

    bundle = synthetic_mimic_dataset(synthetic_config.data, synthetic_config.federation.seed)
    pooled = np.concatenate([c.train_x for c in bundle.clients])
    np.testing.assert_allclose(pooled.mean(axis=(0, 1)), 0.0, atol=1e-6)
    np.testing.assert_allclose(pooled.std(axis=(0, 1)), 1.0, atol=2e-6)
