from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig
from .metrics import BinaryMetrics, binary_metrics


class TimeSeriesCNN(nn.Module):
    def __init__(self, features: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = features
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_channels, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.encoder(x))


@dataclass(frozen=True)
class TrainResult:
    state: OrderedDict[str, torch.Tensor]
    metrics: BinaryMetrics
    seconds: float


class TorchModel:
    def __init__(self, cfg: ModelConfig, features: int, timesteps: int, seed: int) -> None:
        if cfg.kind != "cnn1d":
            raise ValueError(f"unsupported model kind: {cfg.kind}")
        torch.manual_seed(seed)
        self.cfg = cfg
        self.features = features
        self.timesteps = timesteps
        self.device = torch.device("cpu")
        self.net = TimeSeriesCNN(features, cfg.hidden_channels).to(self.device)

    def clone_state(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in self.net.state_dict().items())

    def load_state(self, state: OrderedDict[str, torch.Tensor]) -> None:
        self.net.load_state_dict(OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state.items()))

    def train_local(self, state: OrderedDict[str, torch.Tensor], x: np.ndarray, y: np.ndarray, seed: int) -> TrainResult:
        self.load_state(state)
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, generator=generator)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()
        start = time.perf_counter()
        self.net.train()
        for _ in range(self.cfg.local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.net(batch_x.to(self.device)), batch_y.to(self.device))
                loss.backward()
                optimizer.step()
        seconds = time.perf_counter() - start
        metrics = self.evaluate_arrays(x, y)
        return TrainResult(state=self.clone_state(), metrics=metrics, seconds=seconds)

    def evaluate_state(self, state: OrderedDict[str, torch.Tensor], x: np.ndarray, y: np.ndarray) -> BinaryMetrics:
        self.load_state(state)
        return self.evaluate_arrays(x, y)

    def predict_probabilities(self, state: OrderedDict[str, torch.Tensor], x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.load_state(state)
        self.net.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(0, x.shape[0], batch_size):
                batch = torch.from_numpy(x[offset : offset + batch_size]).float().to(self.device)
                probs = torch.softmax(self.net(batch), dim=1)[:, 1]
                outputs.append(probs.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0,), dtype=np.float32)

    def evaluate_arrays(self, x: np.ndarray, y: np.ndarray) -> BinaryMetrics:
        probabilities = self.predict_probabilities(self.clone_state(), x)
        return binary_metrics(y, probabilities)


def state_to_vector(state: OrderedDict[str, torch.Tensor]) -> np.ndarray:
    values = []
    for tensor in state.values():
        if torch.is_floating_point(tensor):
            values.append(tensor.detach().cpu().numpy().ravel().astype(np.float64))
    return np.concatenate(values) if values else np.empty((0,), dtype=np.float64)


def vector_update(global_state: OrderedDict[str, torch.Tensor], client_state: OrderedDict[str, torch.Tensor]) -> np.ndarray:
    return state_to_vector(client_state) - state_to_vector(global_state)


def apply_vector_update(global_state: OrderedDict[str, torch.Tensor], update: np.ndarray) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    offset = 0
    for name, tensor in global_state.items():
        base = tensor.detach().cpu().clone()
        if torch.is_floating_point(base):
            count = base.numel()
            delta = torch.from_numpy(update[offset : offset + count].reshape(tuple(base.shape))).to(dtype=base.dtype)
            result[name] = base + delta
            offset += count
        else:
            result[name] = base
    if offset != update.size:
        raise ValueError("update vector length does not match model state")
    return result


def aggregate_states(
    global_state: OrderedDict[str, torch.Tensor],
    client_states: list[OrderedDict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    if not client_states:
        return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in global_state.items())
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("aggregation weights must sum to a positive value")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, base_tensor in global_state.items():
        if torch.is_floating_point(base_tensor):
            acc = torch.zeros_like(base_tensor.detach().cpu(), dtype=base_tensor.dtype)
            for state, weight in zip(client_states, weights, strict=True):
                acc = acc + state[name].detach().cpu().to(dtype=base_tensor.dtype) * (float(weight) / total)
            result[name] = acc
        else:
            result[name] = base_tensor.detach().cpu().clone()
    return result

