from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from .model import apply_vector_update, vector_update


def poison_labels(y: np.ndarray, attack_names: tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    result = y.copy()
    if "label_flip" in attack_names:
        mask = rng.random(result.size) < 0.40
        result[mask] = 1 - result[mask]
    if "majority_class" in attack_names:
        # Collapse every local label onto the majority class. Training on this
        # yields a constant classifier, which is the cheapest model that still
        # clears a raw-accuracy gate set below the majority prevalence.
        counts = np.bincount(result, minlength=2)
        result[:] = int(np.argmax(counts))
    return result


def poison_inputs(x: np.ndarray, y: np.ndarray, attack_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if "backdoor" not in attack_names:
        return x, y
    poisoned_x = x.copy()
    poisoned_y = y.copy()
    window = min(4, poisoned_x.shape[1])
    poisoned_x[:, -window:, -1] = np.maximum(poisoned_x[:, -window:, -1], 3.0)
    poisoned_y[:] = 1
    return poisoned_x, poisoned_y


def apply_update_attacks(
    global_state: OrderedDict[str, torch.Tensor],
    client_states: dict[int, OrderedDict[str, torch.Tensor]],
    malicious_clients: set[int],
    attack_names: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[int, OrderedDict[str, torch.Tensor]]:
    if not malicious_clients:
        return client_states
    honest_updates = [
        vector_update(global_state, state) for cid, state in client_states.items() if cid not in malicious_clients
    ]
    if not honest_updates:
        return client_states
    honest = np.vstack(honest_updates)
    mean_update = honest.mean(axis=0)
    std_update = honest.std(axis=0)
    median_norm = float(np.median(np.linalg.norm(honest, axis=1)))
    result = dict(client_states)

    for client_id in malicious_clients:
        if client_id not in result:
            continue
        if "alie" in attack_names:
            crafted = mean_update - 1.5 * std_update
        elif "minmax" in attack_names:
            crafted = -2.0 * mean_update
        elif "random_gradient" in attack_names:
            crafted = rng.normal(0.0, 1.0, size=mean_update.size)
            crafted_norm = float(np.linalg.norm(crafted))
            if crafted_norm > 0.0:
                crafted = crafted * (median_norm / crafted_norm)
        else:
            crafted = vector_update(global_state, result[client_id])
        result[client_id] = apply_vector_update(global_state, crafted)
    return result

