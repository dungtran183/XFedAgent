from __future__ import annotations

from collections import OrderedDict
import os
import random

import numpy as np
import torch

from .attacks import apply_update_attacks, poison_inputs, poison_labels
from .commitments import MerkleCommitter, array_payload, digest_bytes, stable_config_digest
from .config import ExperimentConfig
from .data import load_dataset
from .energy import estimate_energy
from .metrics import predicate_margin
from .metrics import summarize
from .model import TorchModel, aggregate_states, state_to_vector
from .pov import SoftwarePoVBackend, ValidationRotator
from .relay import CrossChainRelaySimulator
from .results import prepare_run_dir, save_round_model, write_json, write_rounds_csv


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.config_digest = stable_config_digest(cfg.to_dict())
        self.run_name = f"{cfg.name}-{cfg.ablation.label}-{self.config_digest[:12]}"

    def run(self) -> dict:
        self._seed_everything(self.cfg.federation.seed)
        data = load_dataset(self.cfg.data, self.cfg.federation.seed)
        model = TorchModel(self.cfg.model, self.cfg.data.features, self.cfg.data.timesteps, self.cfg.federation.seed)
        ablation = self.cfg.ablation
        pov = SoftwarePoVBackend(self.cfg.pov, self.cfg.model, ablation)
        rotator = ValidationRotator(
            self.cfg.pov, data.validation_pool_x, data.validation_pool_y, ablation
        )
        relay = CrossChainRelaySimulator(self.cfg.relay)
        committer = MerkleCommitter(self.cfg.pov.hash_algorithm)
        rng = np.random.default_rng(self.cfg.federation.seed)
        malicious = self._select_malicious_clients(rng)
        reputations = {client.client_id: self.cfg.federation.reputation_initial for client in data.clients}
        global_state = model.clone_state()
        run_dir = prepare_run_dir(self.cfg.output.directory, self.run_name)
        round_rows: list[dict] = []
        proof_seconds: list[float] = []
        modeled_proof_seconds: list[float] = []
        energy_wh: list[float] = []
        finality_seconds: list[float] = []
        relay_gas: list[int] = []
        accepted_updates = 0
        rejected_updates = 0
        malicious_rejections = 0
        malicious_attempts = 0
        honest_false_rejects = 0
        honest_attempts = 0
        relay_quorum = 0

        for round_index in range(self.cfg.federation.rounds):
            selected = self._select_round_clients(round_index, rng)
            challenge = rotator.challenge(round_index, self.cfg.federation.seed)
            val_x, val_y = rotator.data_for(challenge)
            trained_states: dict[int, OrderedDict[str, torch.Tensor]] = {}
            training_seconds: dict[int, float] = {}

            for client_id in selected:
                client = data.clients[client_id]
                local_x, local_y = client.train_x, client.train_y
                if client_id in malicious:
                    local_x, local_y = poison_inputs(local_x, local_y, self.cfg.federation.attacks)
                    local_y = poison_labels(local_y, self.cfg.federation.attacks, rng)
                train = model.train_local(global_state, local_x, local_y, seed=self.cfg.federation.seed + round_index * 1009 + client_id)
                trained_states[client_id] = train.state
                training_seconds[client_id] = train.seconds

            trained_states = apply_update_attacks(global_state, trained_states, malicious, self.cfg.federation.attacks, rng)
            accepted_states: list[OrderedDict[str, torch.Tensor]] = []
            weights: list[float] = []
            accepted_clients: list[int] = []
            round_accepted = 0
            round_rejected = 0
            round_honest_attempts = 0
            round_honest_rejects = 0
            round_mal_attempts = 0
            round_mal_admitted = 0

            for client_id in selected:
                transcript, _ = pov.prove(
                    client_id=client_id,
                    round_index=round_index,
                    model=model,
                    global_state=global_state,
                    client_state=trained_states[client_id],
                    challenge=challenge,
                    validation_x=val_x,
                    validation_y=val_y,
                )
                receipt = relay.relay(transcript)
                relay_quorum = receipt.quorum
                is_malicious = client_id in malicious
                if is_malicious:
                    malicious_attempts += 1
                    round_mal_attempts += 1
                else:
                    honest_attempts += 1
                    round_honest_attempts += 1
                if transcript.accepted and is_malicious:
                    round_mal_admitted += 1
                if not transcript.accepted and not is_malicious:
                    round_honest_rejects += 1
                if transcript.accepted:
                    if ablation.reputation_enabled:
                        margin = predicate_margin(
                            self.cfg.pov.predicate,
                            transcript.accuracy,
                            transcript.sensitivity,
                            transcript.specificity,
                            self.cfg.pov.threshold,
                            self.cfg.pov.threshold_sensitivity,
                            self.cfg.pov.threshold_specificity,
                        )
                        reputations[client_id] = min(
                            1.0,
                            reputations[client_id]
                            + self.cfg.federation.reputation_reward_alpha * margin,
                        )
                        admitted = reputations[client_id] >= self.cfg.federation.reputation_min
                        weight = reputations[client_id]
                    else:
                        # Without reputation the aggregator is a plain unweighted mean
                        # over PoV-admitted updates, and no agent is ever excluded.
                        admitted = True
                        weight = 1.0
                    if admitted:
                        accepted_states.append(trained_states[client_id])
                        weights.append(weight)
                        accepted_clients.append(client_id)
                        accepted_updates += 1
                        round_accepted += 1
                else:
                    if ablation.reputation_enabled:
                        reputations[client_id] = max(
                            0.0,
                            reputations[client_id] - self.cfg.federation.reputation_penalty_beta,
                        )
                    rejected_updates += 1
                    round_rejected += 1
                    if is_malicious:
                        malicious_rejections += 1
                    else:
                        honest_false_rejects += 1

                energy = estimate_energy(
                    self.cfg.energy,
                    training_seconds.get(client_id, 0.0),
                    transcript.modeled_proof_seconds,
                    receipt.source_finality_seconds,
                    receipt.destination_finality_seconds,
                )
                proof_seconds.append(transcript.proof_seconds)
                modeled_proof_seconds.append(transcript.modeled_proof_seconds)
                energy_wh.append(energy.total_wh)
                finality_seconds.append(receipt.finality_seconds)
                relay_gas.append(receipt.gas_used)

            global_state = aggregate_states(global_state, accepted_states, weights)
            global_model_root = committer.build(array_payload(state_to_vector(global_state))).root
            input_binding_hash = digest_bytes(
                "|".join(
                    f"{cid}:{reputations[cid]:.12f}" for cid in sorted(accepted_clients)
                ).encode("utf-8"),
                self.cfg.pov.hash_algorithm,
            )
            test_metrics = model.evaluate_state(global_state, data.test_x, data.test_y)
            if self.cfg.output.save_round_models:
                save_round_model(run_dir, round_index, global_state)
            round_rows.append(
                {
                    "round": round_index,
                    "accuracy": test_metrics.accuracy,
                    "auc_roc": test_metrics.auc_roc,
                    "f1": test_metrics.f1,
                    "accepted": round_accepted,
                    "rejected": round_rejected,
                    "honest_false_reject_rate": (
                        round_honest_rejects / round_honest_attempts if round_honest_attempts else 0.0
                    ),
                    "malicious_admission_rate": (
                        round_mal_admitted / round_mal_attempts if round_mal_attempts else 0.0
                    ),
                    "mean_reputation": float(np.mean(list(reputations.values()))),
                    "active_reputations": int(sum(v >= self.cfg.federation.reputation_min for v in reputations.values())),
                    "global_model_root": global_model_root,
                    "input_binding_hash": input_binding_hash,
                }
            )

        final_metrics = model.evaluate_state(global_state, data.test_x, data.test_y)
        final_model_root = committer.build(array_payload(state_to_vector(global_state))).root
        summary = {
            "run_name": self.run_name,
            "config_digest": self.config_digest,
            "ablation": {**self.cfg.ablation.__dict__, "label": self.cfg.ablation.label},
            "final_metrics": final_metrics.to_dict(),
            "final_model_root": final_model_root,
            "accepted_updates": accepted_updates,
            "rejected_updates": rejected_updates,
            "malicious_clients": sorted(malicious),
            "malicious_rejection_rate": malicious_rejections / malicious_attempts if malicious_attempts else 0.0,
            "honest_false_reject_rate": honest_false_rejects / honest_attempts if honest_attempts else 0.0,
            "proof_seconds": summarize(proof_seconds),
            "modeled_proof_seconds": summarize(modeled_proof_seconds),
            "energy_wh": summarize(energy_wh),
            "relay": {
                "enabled": self.cfg.relay.enabled,
                "quorum": relay_quorum,
                "relayers": self.cfg.relay.relayers,
                "faulty_relayers": self.cfg.relay.faulty_relayers,
                "finality_seconds": summarize(finality_seconds),
                "total_gas": int(sum(relay_gas)),
                "gas_per_update": summarize([float(g) for g in relay_gas]),
                "messages_relayed": len(relay_gas),
                "replays_detected": 0,
            },
            "final_reputations": {str(k): v for k, v in sorted(reputations.items())},
        }
        self._write_outputs(summary, round_rows, self.cfg.to_dict())
        return summary

    def _select_malicious_clients(self, rng: np.random.Generator) -> set[int]:
        count = int(np.floor(self.cfg.data.clients * self.cfg.federation.byzantine_fraction))
        if count == 0:
            return set()
        selected = rng.choice(self.cfg.data.clients, size=count, replace=False)
        return {int(v) for v in selected}

    def _select_round_clients(self, round_index: int, rng: np.random.Generator) -> list[int]:
        if self.cfg.federation.clients_per_round == self.cfg.data.clients:
            return list(range(self.cfg.data.clients))
        selected = rng.choice(self.cfg.data.clients, size=self.cfg.federation.clients_per_round, replace=False)
        return sorted(int(v) for v in selected)

    def _write_outputs(self, summary: dict, rounds: list[dict], config: dict) -> None:
        run_dir = prepare_run_dir(self.cfg.output.directory, self.run_name)
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "summary.json", summary)
        write_rounds_csv(run_dir / "rounds.csv", rounds)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
