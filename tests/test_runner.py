import json
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pytest
import torch

from xfedagent.runner import ExperimentRunner


def test_runner_produces_artifacts(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "rounds.csv").exists()
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["config_digest"] == summary["config_digest"]


def test_summary_reports_full_framework(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()

    # Tier 3: global-model integrity commitment is published every round.
    assert isinstance(summary["final_model_root"], str) and summary["final_model_root"]

    # Tier 2: cross-chain relay accounting is surfaced, with a Byzantine quorum
    # and a positive gas budget per relayed message.
    relay = summary["relay"]
    assert relay["enabled"] is True
    assert relay["quorum"] > 2 * relay["faulty_relayers"]
    assert relay["total_gas"] > 0
    assert relay["messages_relayed"] > 0

    # Energy: the four-component model yields a non-negative per-update mean.
    assert summary["energy_wh"]["mean"] >= 0.0


def test_rounds_csv_carries_commitments(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    header = (run_dir / "rounds.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "global_model_root" in header
    assert "input_binding_hash" in header


def test_save_round_models_flag(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    object.__setattr__(cfg.output, "save_round_models", True)
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    saved = sorted((run_dir / "models").glob("round_*.pt"))
    assert len(saved) == cfg.federation.rounds


def _admission_case(cfg):
    from xfedagent.commitments import array_payload, digest_bytes
    from xfedagent.metrics import binary_metrics
    from xfedagent.pov import SoftwarePoVBackend, ValidationChallenge

    x = np.arange(cfg.pov.validation_size, dtype=np.float32)[:, None, None]
    y = np.arange(cfg.pov.validation_size) % 2
    probabilities = y * 0.8 + 0.1

    class FixedModel:
        def evaluate_state(self, state, features, labels):
            return binary_metrics(labels, probabilities)

        def predict_probabilities(self, state, features):
            return probabilities

    challenge = ValidationChallenge(0, np.arange(y.size), "test-seed",
        digest_bytes(array_payload(x) + array_payload(y), cfg.pov.hash_algorithm))
    backend = SoftwarePoVBackend(cfg.pov, cfg.model)
    args = dict(client_id=1, round_index=0, model=FixedModel(),
        global_state=OrderedDict(weight=torch.tensor([0.0, 0.0])),
        client_state=OrderedDict(weight=torch.tensor([1.0, 2.0])),
        challenge=challenge, validation_x=x, validation_y=y)
    return backend, args


def test_pov_rejects_substituted_model_challenge_data_and_round(synthetic_config):
    from xfedagent.model import state_commitment

    backend, args = _admission_case(synthetic_config)
    committed = state_commitment(args["client_state"])
    assert backend.prove(**args, committed_model_root=committed)[0].accepted
    with pytest.raises(ValueError, match="changed after commitment"):
        backend.prove(**{**args, "client_state": OrderedDict(weight=torch.tensor([1.00001, 2.0]))}, committed_model_root=committed)
    with pytest.raises(ValueError, match="does not open"):
        backend.prove(**{**args, "validation_y": 1 - args["validation_y"]})
    with pytest.raises(ValueError, match="round"):
        backend.prove(**{**args, "round_index": 1})


def test_pov_transcript_binds_predicate_policy_and_unambiguous_identity(synthetic_config):
    from xfedagent.pov import SoftwarePoVBackend

    backend, args = _admission_case(synthetic_config)
    original, _ = backend.prove(**args)
    modified = SoftwarePoVBackend(replace(synthetic_config.pov, threshold_sensitivity=0.61), synthetic_config.model)
    assert modified.prove(**args)[0].proof_hash != original.proof_hash
    # Decimal concatenation formerly aliased client=1, round=23 with 12, 3.
    a = backend.prove(**{**args, "client_id": 1, "round_index": 23,
        "challenge": replace(args["challenge"], round_index=23)})[0]
    b = backend.prove(**{**args, "client_id": 12, "round_index": 3,
        "challenge": replace(args["challenge"], round_index=3)})[0]
    assert a.proof_hash != b.proof_hash


def test_relay_rejects_duplicate_equivocating_and_stale_source_rounds(synthetic_config):
    from xfedagent.relay import CrossChainRelaySimulator

    backend, args = _admission_case(synthetic_config)
    transcript, _ = backend.prove(**args)
    relay = CrossChainRelaySimulator(synthetic_config.relay)
    assert relay.relay(transcript).nonce == 0
    for duplicated in (transcript, replace(transcript, proof_hash="different-proof")):
        with pytest.raises(ValueError, match="replay"):
            relay.relay(duplicated)
    assert relay.relay(replace(transcript, round_index=2)).nonce == 1
    with pytest.raises(ValueError, match="stale"):
        relay.relay(replace(transcript, round_index=1))
    assert relay.replays_detected == 3
    assert relay.nonces[1] == 2, "refused replays must not consume fresh nonces"
