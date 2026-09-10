from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from .model import apply_vector_update, vector_update


def minmax_update(honest_updates: np.ndarray) -> np.ndarray:
    """Maximal perturbation along negative coordinate-wise standard deviation.

    Implements the distance constraint of Shejwalkar and Houmansadr, NDSS 2021,
    Eq. (6): max_i ||u - h_i||_2 <= max_ij ||h_i - h_j||_2. With the direction
    fixed, each ball gives a quadratic bound on the ray length; their minimum
    solves the one-dimensional optimization without a tuned search ceiling.
    https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf
    """
    honest = np.asarray(honest_updates, dtype=np.float64)
    if honest.ndim != 2 or not honest.shape[0] or not np.isfinite(honest).all():
        raise ValueError("MinMax requires a nonempty matrix of finite honest updates")
    scale = float(np.max(np.abs(honest))) if honest.size else 0.0
    if scale == 0.0:
        return np.zeros(honest.shape[1], dtype=np.float64)
    # Rescaling prevents the distance squares from overflowing for large updates.
    scaled = honest / scale
    center = scaled.mean(axis=0)
    direction = scaled.std(axis=0)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        return honest.mean(axis=0)
    direction /= direction_norm
    diameter_sq = max(
        float(np.max(np.sum((scaled - row) ** 2, axis=1))) for row in scaled
    )
    offsets = center - scaled
    projection = offsets @ direction
    slack = np.maximum(0.0, diameter_sq - np.sum(offsets ** 2, axis=1))
    roots = projection + np.sqrt(projection ** 2 + slack)
    length = max(0.0, float(np.min(roots)))
    # Stay on the feasible side of the inclusive boundary after float rounding.
    length = float(np.nextafter(length, 0.0))
    return (center - length * direction) * scale


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
    if "replay" in attack_names:
        # Submit the current global model unchanged. Its utility is whatever the
        # global model already attains, so a utility predicate alone admits it and the
        # relation carries the copy-detector bound as well. Handled before the honest
        # update statistics, which a replay does not read.
        result = dict(client_states)
        for client_id in malicious_clients:
            if client_id in result:
                result[client_id] = OrderedDict(
                    (name, tensor.detach().cpu().clone()) for name, tensor in global_state.items()
                )
        return result
    honest_updates = [
        vector_update(global_state, state) for cid, state in client_states.items() if cid not in malicious_clients
    ]
    if not honest_updates:
        return client_states
    honest = np.vstack(honest_updates)
    mean_update = honest.mean(axis=0)
    std_update = honest.std(axis=0)
    median_norm = float(np.median(np.linalg.norm(honest, axis=1)))
    minmax = minmax_update(honest) if "minmax" in attack_names and "alie" not in attack_names else None
    result = dict(client_states)

    for client_id in malicious_clients:
        if client_id not in result:
            continue
        if "alie" in attack_names:
            crafted = mean_update - 1.5 * std_update
        elif "minmax" in attack_names:
            crafted = minmax
        elif "random_gradient" in attack_names:
            crafted = rng.normal(0.0, 1.0, size=mean_update.size)
            crafted_norm = float(np.linalg.norm(crafted))
            if crafted_norm > 0.0:
                crafted = crafted * (median_norm / crafted_norm)
        else:
            crafted = vector_update(global_state, result[client_id])
        result[client_id] = apply_vector_update(global_state, crafted)
    return result
