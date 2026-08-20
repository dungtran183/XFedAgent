"""Shared test fixtures.

The runner tests exercise the full pipeline on a small, deterministic
synthetic configuration built in-process. This keeps the test suite fast and
self-contained without shipping an auxiliary configuration file or requiring
the external dataset that the production configuration expects.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from xfedagent.config import parse_config

_SYNTHETIC_CONFIG = {
    "name": "synthetic-test",
    "data": {
        "source": "synthetic_mimic",
        "samples": 480,
        "clients": 4,
        "timesteps": 24,
        "features": 17,
        "validation_pool_size": 64,
        "validation_fraction": 0.18,
        "test_fraction": 0.18,
        "dirichlet_alpha": 0.8,
        "csv_path": None,
        "label_column": "label",
        "patient_column": "patient_id",
    },
    "model": {
        "kind": "cnn1d",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "local_epochs": 2,
        "batch_size": 16,
        "hidden_channels": [8, 16],
        "quantization_bits": 8,
        "copy_epsilon": 0.0001,
    },
    "pov": {
        "backend": "software",
        "threshold": 0.56,
        "validation_size": 32,
        "tolerance": 0.02,
        "rotation": "per_round",
        "hash_algorithm": "sha3_256",
        "proof_seconds_per_sample": 0.18,
        "snarkjs_bin": "snarkjs",
        "circuit_wasm": None,
        "proving_key": None,
    },
    "federation": {
        "rounds": 4,
        "clients_per_round": 4,
        "byzantine_fraction": 0.25,
        "attacks": ["label_flip", "random_gradient"],
        "seed": 7,
        "reputation_initial": 0.5,
        "reputation_min": 0.2,
        "reputation_reward_alpha": 0.1,
        "reputation_penalty_beta": 0.2,
    },
    "relay": {
        "enabled": True,
        "chains": ["chain-a", "chain-b"],
        "relayers": 4,
        "faulty_relayers": 1,
        "source_finality_blocks": 2,
        "source_block_seconds": 0.2,
        "destination_finality_seconds": 0.1,
        "gas_verify_base": 207000,
        "gas_reputation_update": 78000,
    },
    "energy": {
        "enabled": True,
        "local_training_watts": 5.2,
        "proving_watts": 5.8,
        "communication_watts": 3.8,
        "idle_watts": 2.7,
    },
    "output": {"directory": "results", "save_round_models": False},
    "toolchain": {
        "require_snarkjs": False,
        "require_qemu": False,
        "require_ipfs": False,
        "require_hardhat": False,
    },
}


@pytest.fixture
def synthetic_config():
    """Return a freshly parsed, validated synthetic experiment configuration."""
    return parse_config(deepcopy(_SYNTHETIC_CONFIG))
