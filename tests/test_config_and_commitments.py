from pathlib import Path

import numpy as np
import pytest
import torch
from collections import OrderedDict

from xfedagent.commitments import MerkleCommitter, quantize_vector
from xfedagent.config import load_config
from xfedagent.energy import estimate_energy
from xfedagent.config import EnergyConfig


def test_full_config_loads() -> None:
    cfg = load_config(Path("configs") / "full.json")
    assert cfg.name == "full-mimic-cloud"
    assert cfg.data.clients == 10
    assert cfg.pov.validation_size <= cfg.data.validation_pool_size


def test_merkle_commitment_is_deterministic() -> None:
    data = np.arange(64, dtype=np.int64)
    committer = MerkleCommitter("sha3_256", chunk_bytes=32)
    left = committer.commit_array(data)
    right = committer.commit_array(data.copy())
    assert left.root == right.root
    assert len(left.leaves) > 1


def test_quantization_bounds() -> None:
    vector = np.array([-2.0, -0.1, 0.0, 0.25, 1.8])
    quantized, scale = quantize_vector(vector, bits=8)
    assert scale > 0
    assert quantized.min() >= -127
    assert quantized.max() <= 127


def test_energy_model_includes_idle() -> None:
    cfg = EnergyConfig(
        enabled=True,
        local_training_watts=5.2,
        proving_watts=5.8,
        communication_watts=3.8,
        idle_watts=2.7,
    )
    breakdown = estimate_energy(cfg, training_seconds=3600.0, proving_seconds=3600.0, communication_seconds=3600.0, idle_seconds=3600.0)
    assert abs(breakdown.local_training_wh - 5.2) < 1e-9
    assert abs(breakdown.proving_wh - 5.8) < 1e-9
    assert abs(breakdown.communication_wh - 3.8) < 1e-9
    assert abs(breakdown.idle_wh - 2.7) < 1e-9
    assert abs(breakdown.total_wh - (5.2 + 5.8 + 3.8 + 2.7)) < 1e-9


def test_energy_disabled_is_zero() -> None:
    cfg = EnergyConfig(False, 5.2, 5.8, 3.8, 2.7)
    breakdown = estimate_energy(cfg, 10.0, 10.0, 10.0, 10.0)
    assert breakdown.total_wh == 0.0


def test_exact_state_binding_distinguishes_models_with_the_same_quantization():
    from xfedagent.model import state_commitment, state_to_vector

    a = OrderedDict(weight=torch.tensor([1.0, 2.0]), counter=torch.tensor(1))
    b = OrderedDict(weight=torch.tensor([1.00001, 2.0]), counter=torch.tensor(1))
    qa, sa = quantize_vector(state_to_vector(a), 8)
    qb, sb = quantize_vector(state_to_vector(b), 8)
    assert np.array_equal(qa, qb) and sa == sb
    assert state_commitment(a) != state_commitment(b)
    b = OrderedDict(weight=a["weight"], counter=torch.tensor(2))
    assert state_commitment(a) != state_commitment(b), "integer buffers also bind"


def test_aggregation_binding_includes_models_round_order_and_actual_weights():
    from xfedagent.model import aggregation_input_commitment

    entries = [(1, "model-a", 0.5), (2, "model-b", 0.6)]
    baseline = aggregation_input_commitment(2, "global-a", entries)
    assert baseline == aggregation_input_commitment(2, "global-a", entries.copy())
    variants = [
        (3, "global-a", entries), (2, "global-b", entries),
        (2, "global-a", list(reversed(entries))),
        (2, "global-a", [(1, "changed-model", 0.5), entries[1]]),
        (2, "global-a", [(1, "model-a", np.nextafter(0.5, 1.0)), entries[1]]),
    ]
    assert all(aggregation_input_commitment(*variant) != baseline for variant in variants)
    with pytest.raises(ValueError, match="duplicate"):
        aggregation_input_commitment(2, "g", [entries[0], entries[0]])


@pytest.mark.parametrize("weights", [[float("nan")], [float("inf")], [-1.0], [0.0], []])
def test_aggregation_rejects_invalid_weights(weights):
    from xfedagent.model import aggregate_states

    state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="weight"):
        aggregate_states(state, [state], weights)
