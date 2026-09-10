from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig


@dataclass(frozen=True)
class ClientSplit:
    client_id: int
    train_x: np.ndarray
    train_y: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    clients: tuple[ClientSplit, ...]
    validation_pool_x: np.ndarray
    validation_pool_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_names: tuple[str, ...]


def load_dataset(cfg: DataConfig, seed: int) -> DatasetBundle:
    if cfg.source == "synthetic_mimic":
        return synthetic_mimic_dataset(cfg, seed)
    if cfg.source == "csv_timeseries":
        return csv_timeseries_dataset(cfg, seed)
    raise ValueError(f"unsupported data source: {cfg.source}")


def audit_csv_dataset(cfg: DataConfig) -> dict[str, bool | float | int]:
    """Return aggregate-only cohort checks suitable for preflight logs."""

    import pandas as pd

    if cfg.csv_path is None:
        raise ValueError("csv_timeseries requires data.csv_path")
    path = Path(cfg.csv_path)
    frame = pd.read_csv(path)
    required = {cfg.patient_column, cfg.label_column}
    feature_columns = [column for column in frame.columns if column not in required]
    schema_valid = required.issubset(frame.columns) and (
        len(feature_columns) == cfg.timesteps * cfg.features
    )
    if not required.issubset(frame.columns):
        return {
            "rows": int(len(frame)),
            "patients": 0,
            "positives": 0,
            "prevalence": 0.0,
            "schema_valid": False,
            "patient_ids_complete": False,
            "binary_labels": False,
            "patients_with_mixed_labels": 0,
            "enough_patients": False,
        }

    patient_ids_complete = not bool(frame[cfg.patient_column].isna().any())
    labels = pd.to_numeric(frame[cfg.label_column], errors="coerce")
    binary_labels = bool(labels.notna().all() and set(labels.unique()).issubset({0, 1}))
    labels_per_patient = (
        frame.assign(_audit_label=labels)
        .groupby(cfg.patient_column, dropna=False)["_audit_label"]
        .nunique(dropna=False)
    )
    patients = int(frame[cfg.patient_column].nunique(dropna=True))
    positives = int(labels.sum()) if binary_labels else 0
    return {
        "rows": int(len(frame)),
        "patients": patients,
        "positives": positives,
        "prevalence": float(positives / len(frame)) if binary_labels and len(frame) else 0.0,
        "schema_valid": bool(schema_valid),
        "patient_ids_complete": patient_ids_complete,
        "binary_labels": binary_labels,
        "patients_with_mixed_labels": int(labels_per_patient.gt(1).sum()),
        "enough_patients": patients >= cfg.clients + 2,
    }


def audit_csv_splits(cfg: DataConfig, seed: int = 0) -> dict[str, bool | int]:
    """Confirm the production splitter puts every patient in exactly one split.

    Preflight demonstrates rather than assumes that no ``subject_id`` straddles
    train, validation and test. The check runs
    the same splitter the loader runs, on the same seed, and reports only counts,
    so it is safe to keep in a log alongside credentialed data.
    """

    import pandas as pd

    if cfg.csv_path is None:
        raise ValueError("csv_timeseries requires data.csv_path")
    frame = pd.read_csv(Path(cfg.csv_path))
    if cfg.patient_column not in frame.columns:
        raise ValueError(f"missing patient column: {cfg.patient_column}")
    patient_ids = frame[cfg.patient_column].astype(str).to_numpy()
    rows = len(frame)
    test_count, validation_count = _split_counts(cfg, rows)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _patient_disjoint_split(
        patient_ids,
        test_count=test_count,
        validation_count=validation_count,
        minimum_train_patients=cfg.clients,
        rng=rng,
    )
    groups = {
        "train": set(patient_ids[train_idx]),
        "validation": set(patient_ids[val_idx]),
        "test": set(patient_ids[test_idx]),
    }
    shared = set()
    names = sorted(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared |= groups[left] & groups[right]
    assigned = len(train_idx) + len(val_idx) + len(test_idx)
    return {
        "rows": rows,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_patients": len(groups["train"]),
        "validation_patients": len(groups["validation"]),
        "test_patients": len(groups["test"]),
        "shared_patients": len(shared),
        "patient_disjoint": not shared,
        "validation_pool_satisfied": int(len(val_idx)) >= cfg.validation_pool_size,
        "rows_assigned_once": assigned == len({*train_idx.tolist(), *val_idx.tolist(), *test_idx.tolist()}),
    }


def synthetic_mimic_dataset(cfg: DataConfig, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n = cfg.samples
    timesteps = cfg.timesteps
    features = cfg.features
    if features < 8:
        raise ValueError("synthetic_mimic requires at least 8 features")

    severity = rng.normal(0.0, 1.0, size=n)
    chronic = rng.normal(0.0, 0.7, size=n)
    hospital_group = rng.integers(0, cfg.clients, size=n)
    group_shift = rng.normal(0.0, 0.35, size=(cfg.clients, features))
    slopes = rng.normal(0.0, 0.08, size=(n, features))
    base_coeff = np.linspace(-0.6, 0.9, features)

    x = np.empty((n, timesteps, features), dtype=np.float32)
    state = rng.normal(0.0, 0.4, size=(n, features))
    for step in range(timesteps):
        innovation = rng.normal(0.0, 0.35, size=(n, features))
        trend = (step / max(1, timesteps - 1)) * slopes
        latent = np.outer(severity, base_coeff) + chronic[:, None] * 0.2 + group_shift[hospital_group]
        state = 0.72 * state + 0.28 * latent + trend + innovation
        x[:, step, :] = state.astype(np.float32)

    risk_features = x[:, -6:, :].mean(axis=1)
    risk = (
        1.35 * severity
        + 0.70 * chronic
        + 0.85 * risk_features[:, 0]
        - 0.65 * risk_features[:, 2]
        + 0.50 * risk_features[:, 5 % features]
        + rng.normal(0.0, 0.25, size=n)
    )
    # Quantile at 1 - positive_rate makes the positive class as rare as the
    # configured prevalence, so an imbalanced surrogate can be generated.
    threshold = np.quantile(risk, 1.0 - cfg.positive_rate)
    y = (risk > threshold).astype(np.int64)

    indices = rng.permutation(n)
    test_count = max(1, int(round(cfg.test_fraction * n)))
    validation_count = max(cfg.validation_pool_size, int(round(cfg.validation_fraction * n)))
    if test_count + validation_count >= n:
        raise ValueError("test and validation splits leave no training samples")
    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + validation_count]
    train_idx = indices[test_count + validation_count :]
    x = _zscore(x, x[train_idx])

    client_indices = _dirichlet_partition(y[train_idx], cfg.clients, cfg.dirichlet_alpha, rng)
    clients = tuple(
        ClientSplit(client_id=i, train_x=x[train_idx[part]], train_y=y[train_idx[part]]) for i, part in enumerate(client_indices)
    )
    val_select = val_idx[: cfg.validation_pool_size]
    names = tuple(f"feature_{i:02d}" for i in range(features))
    return DatasetBundle(
        clients=clients,
        validation_pool_x=x[val_select],
        validation_pool_y=y[val_select],
        test_x=x[test_idx],
        test_y=y[test_idx],
        feature_names=names,
    )


def csv_timeseries_dataset(cfg: DataConfig, seed: int) -> DatasetBundle:
    import pandas as pd

    if cfg.csv_path is None:
        raise ValueError("csv_timeseries requires data.csv_path")
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if cfg.label_column not in frame.columns:
        raise ValueError(f"missing label column: {cfg.label_column}")
    if cfg.patient_column not in frame.columns:
        raise ValueError(f"missing patient column: {cfg.patient_column}")
    if frame[cfg.patient_column].isna().any():
        raise ValueError(f"patient column contains missing values: {cfg.patient_column}")
    feature_columns = [c for c in frame.columns if c not in {cfg.label_column, cfg.patient_column}]
    if len(feature_columns) != cfg.timesteps * cfg.features:
        raise ValueError("csv_timeseries expects flattened timesteps * features columns")
    x = frame[feature_columns].to_numpy(dtype=np.float32).reshape(-1, cfg.timesteps, cfg.features)
    y = frame[cfg.label_column].to_numpy(dtype=np.int64)
    patient_ids = frame[cfg.patient_column].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    test_count, validation_count = _split_counts(cfg, int(y.size))
    train_idx, val_idx, test_idx = _patient_disjoint_split(
        patient_ids,
        test_count=test_count,
        validation_count=validation_count,
        minimum_train_patients=cfg.clients,
        rng=rng,
    )
    # Fit every learned preprocessing statistic after the patient split. The same
    # training-only transformation is applied to all clients and held-out rows.
    x = _zscore(x, x[train_idx])
    client_indices = _dirichlet_patient_partition(
        y[train_idx],
        patient_ids[train_idx],
        cfg.clients,
        cfg.dirichlet_alpha,
        rng,
    )
    clients = tuple(
        ClientSplit(client_id=i, train_x=x[train_idx[part]], train_y=y[train_idx[part]]) for i, part in enumerate(client_indices)
    )
    return DatasetBundle(
        clients=clients,
        validation_pool_x=x[val_idx[: cfg.validation_pool_size]],
        validation_pool_y=y[val_idx[: cfg.validation_pool_size]],
        test_x=x[test_idx],
        test_y=y[test_idx],
        feature_names=tuple(feature_columns),
    )


def _split_counts(cfg: DataConfig, rows: int) -> tuple[int, int]:
    """Split sizes for the CSV cohort, shared by the loader and its audit."""

    test_count = max(1, int(round(cfg.test_fraction * rows)))
    validation_count = max(cfg.validation_pool_size, int(round(cfg.validation_fraction * rows)))
    return test_count, validation_count


def _patient_disjoint_split(
    patient_ids: np.ndarray,
    *,
    test_count: int,
    validation_count: int,
    minimum_train_patients: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows by patient, never by observation.

    Targets are expressed in rows because the public configuration uses sample
    fractions and a minimum validation-pool size. Whole patient groups are added
    until each target is met, while reserving enough patients for training.
    """

    ids = np.asarray(patient_ids)
    unique_ids = np.unique(ids)
    required_patients = minimum_train_patients + 2
    if unique_ids.size < required_patients:
        raise ValueError(
            "csv_timeseries requires at least "
            f"{required_patients} distinct patients for patient-disjoint splits"
        )

    shuffled = unique_ids[rng.permutation(unique_ids.size)]
    groups = [np.flatnonzero(ids == patient_id) for patient_id in shuffled]

    def take_groups(start: int, target_rows: int, patients_to_reserve: int) -> tuple[list[np.ndarray], int]:
        selected: list[np.ndarray] = []
        rows = 0
        cursor = start
        while rows < target_rows and len(groups) - cursor > patients_to_reserve:
            group = groups[cursor]
            selected.append(group)
            rows += group.size
            cursor += 1
        if rows < target_rows:
            raise ValueError(
                "patient-disjoint split cannot satisfy the configured row target "
                "while preserving the requested training patients"
            )
        return selected, cursor

    test_groups, cursor = take_groups(0, test_count, minimum_train_patients + 1)
    validation_groups, cursor = take_groups(
        cursor, validation_count, minimum_train_patients
    )
    train_groups = groups[cursor:]
    if len(train_groups) < minimum_train_patients:
        raise ValueError("patient-disjoint split leaves too few training patients")

    def flatten(selected: list[np.ndarray]) -> np.ndarray:
        result = np.concatenate(selected).astype(np.int64, copy=False)
        rng.shuffle(result)
        return result

    return flatten(train_groups), flatten(validation_groups), flatten(test_groups)


def _dirichlet_patient_partition(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Partition complete patient groups across clients using patient labels."""

    ids = np.asarray(patient_ids)
    unique_ids = np.unique(ids)
    if unique_ids.size < clients:
        raise ValueError("training split has fewer patients than clients")

    patient_labels = np.empty(unique_ids.size, dtype=np.int64)
    row_groups: list[np.ndarray] = []
    for index, patient_id in enumerate(unique_ids):
        rows = np.flatnonzero(ids == patient_id)
        # A patient may have several admissions with different outcomes. All of them
        # stay in one client, and the patient group is stratified on whether any
        # admission is positive, so mixed-outcome patients are kept.
        patient_labels[index] = int(np.max(labels[rows]))
        row_groups.append(rows)

    patient_partitions = _dirichlet_partition(patient_labels, clients, alpha, rng)
    result: list[np.ndarray] = []
    for partition in patient_partitions:
        rows = np.concatenate([row_groups[index] for index in partition]).astype(
            np.int64, copy=False
        )
        rng.shuffle(rows)
        result.append(rows)
    return tuple(result)


def _zscore(x: np.ndarray, training_reference: np.ndarray) -> np.ndarray:
    """Standardise from training rows only; the reference must be explicit."""
    reference = np.asarray(training_reference, dtype=np.float64)
    if not reference.size or not np.isfinite(reference).all() or not np.isfinite(x).all():
        raise ValueError("standardisation requires nonempty finite training features and finite inputs")
    mean = reference.mean(axis=(0, 1), keepdims=True)
    std = reference.std(axis=(0, 1), keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _dirichlet_partition(labels: np.ndarray, clients: int, alpha: float, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    partitions: list[list[int]] = [[] for _ in range(clients)]
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.full(clients, alpha))
        cuts = np.cumsum(proportions)[:-1]
        split_points = np.floor(cuts * label_indices.size).astype(int)
        for client_id, chunk in enumerate(np.split(label_indices, split_points)):
            partitions[client_id].extend(int(v) for v in chunk)
    result = []
    for part in partitions:
        if not part:
            donor = max(partitions, key=len)
            part.append(donor.pop())
        arr = np.array(part, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)
    return tuple(result)
