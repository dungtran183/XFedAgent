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
    threshold = np.quantile(risk, 0.50)
    y = (risk > threshold).astype(np.int64)

    x = _zscore(x)
    indices = rng.permutation(n)
    test_count = max(1, int(round(cfg.test_fraction * n)))
    validation_count = max(cfg.validation_pool_size, int(round(cfg.validation_fraction * n)))
    if test_count + validation_count >= n:
        raise ValueError("test and validation splits leave no training samples")
    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + validation_count]
    train_idx = indices[test_count + validation_count :]

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
    feature_columns = [c for c in frame.columns if c not in {cfg.label_column, cfg.patient_column}]
    if len(feature_columns) != cfg.timesteps * cfg.features:
        raise ValueError("csv_timeseries expects flattened timesteps * features columns")
    x = frame[feature_columns].to_numpy(dtype=np.float32).reshape(-1, cfg.timesteps, cfg.features)
    y = frame[cfg.label_column].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(y.size)
    test_count = max(1, int(round(cfg.test_fraction * y.size)))
    validation_count = max(cfg.validation_pool_size, int(round(cfg.validation_fraction * y.size)))
    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + validation_count]
    train_idx = indices[test_count + validation_count :]
    client_indices = _dirichlet_partition(y[train_idx], cfg.clients, cfg.dirichlet_alpha, rng)
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


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
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
