"""Software admission evaluator with commitments to the exact evaluated tensors.

The transcript is an integrity record, not a zero-knowledge proof. The separate
Circom artifact evaluates an integer binary-linear model and uses Poseidon.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import time
import json

import numpy as np
import torch

from .commitments import array_payload, digest_bytes
from .config import AblationConfig, ModelConfig, PoVConfig, predicted_positive_rate_for
from .metrics import (
    BinaryMetrics,
    confusion_counts,
    confusion_counts_at_rate,
    passes_class_aware,
    passes_raw_accuracy,
)
from .model import TorchModel, state_commitment, state_to_vector


@dataclass(frozen=True)
class ValidationChallenge:
    round_index: int
    indices: np.ndarray
    seed_material: str
    validation_root: str


@dataclass(frozen=True)
class ProofTranscript:
    client_id: int
    round_index: int
    model_root: str
    validation_root: str
    proof_hash: str
    accuracy: float
    threshold: float
    copy_distance: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    predicate: str
    accepted: bool
    backend: str
    proof_seconds: float
    modeled_proof_seconds: float
    global_root: str
    confusion: dict[str, int]
    policy: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ValidationRotator:
    def __init__(
        self,
        cfg: PoVConfig,
        pool_x: np.ndarray,
        pool_y: np.ndarray,
        ablation: AblationConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.pool_x = pool_x
        self.pool_y = pool_y
        self.ablation = ablation or AblationConfig()

    def challenge(self, round_index: int, master_seed: int) -> ValidationChallenge:
        # With rotation ablated the challenge is frozen to the round-0 subset, so
        # every round proves against one static validation set.
        effective_round = round_index if self.ablation.rotation_enabled else 0
        seed_material = digest_bytes(f"{master_seed}:{effective_round}:{self.cfg.rotation}".encode("utf-8"), self.cfg.hash_algorithm)
        seed = int(seed_material[:16], 16)
        rng = np.random.default_rng(seed)
        if self.cfg.balanced_validation:
            indices = self._balanced_indices(rng)
        else:
            indices = rng.choice(self.pool_y.size, size=self.cfg.validation_size, replace=False)
        payload = array_payload(self.pool_x[indices]) + array_payload(self.pool_y[indices])
        root = digest_bytes(payload, self.cfg.hash_algorithm)
        return ValidationChallenge(round_index=round_index, indices=indices, seed_material=seed_material, validation_root=root)

    def _balanced_indices(self, rng: np.random.Generator) -> np.ndarray:
        """Draw the subset with equal class counts, falling back if a class is short.

        The scarcer class caps how balanced the draw can be; when the pool cannot
        supply half the subset from one class we take all of it and top up from the
        other, which is the closest achievable balance.
        """
        half = self.cfg.validation_size // 2
        pos = np.flatnonzero(self.pool_y == 1)
        neg = np.flatnonzero(self.pool_y == 0)
        take_pos = min(half, pos.size)
        take_neg = min(self.cfg.validation_size - take_pos, neg.size)
        take_pos = min(pos.size, self.cfg.validation_size - take_neg)
        chosen = np.concatenate(
            [
                rng.choice(pos, size=take_pos, replace=False),
                rng.choice(neg, size=take_neg, replace=False),
            ]
        )
        rng.shuffle(chosen)
        return chosen

    def data_for(self, challenge: ValidationChallenge) -> tuple[np.ndarray, np.ndarray]:
        return self.pool_x[challenge.indices], self.pool_y[challenge.indices]


class SoftwarePoVBackend:
    def __init__(
        self,
        cfg: PoVConfig,
        model_cfg: ModelConfig,
        ablation: AblationConfig | None = None,
        predicted_positive_rate: float | None = None,
    ) -> None:
        if cfg.backend != "software":
            raise ValueError("SoftwarePoVBackend requires backend=software")
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.ablation = ablation or AblationConfig()
        # Under decision_rule="rate" the rate is a public challenge parameter, so it
        # is resolved once here rather than per proof. The closed form reads the
        # challenge prevalence, which for an unbalanced challenge comes from the
        # cohort; the balanced form is the fallback when the caller supplies none.
        self.predicted_positive_rate = (
            float(predicted_positive_rate)
            if predicted_positive_rate is not None
            else predicted_positive_rate_for(cfg, 0.5 if cfg.balanced_validation else 0.5)
        )

    def prove(
        self,
        client_id: int,
        round_index: int,
        model: TorchModel,
        global_state: OrderedDict[str, torch.Tensor],
        client_state: OrderedDict[str, torch.Tensor],
        challenge: ValidationChallenge,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        committed_model_root: str | None = None,
    ) -> tuple[ProofTranscript, BinaryMetrics]:
        start = time.perf_counter()
        if round_index != challenge.round_index:
            raise ValueError("proof round does not match validation challenge")
        if len(validation_y) != self.cfg.validation_size or len(validation_x) != len(validation_y):
            raise ValueError("validation challenge has an unexpected sample count")
        actual_validation_root = digest_bytes(
            array_payload(validation_x) + array_payload(validation_y), self.cfg.hash_algorithm
        )
        if actual_validation_root != challenge.validation_root:
            raise ValueError("validation data does not open the challenge commitment")
        model_root = state_commitment(client_state, self.cfg.hash_algorithm)
        if committed_model_root is not None and model_root != committed_model_root:
            raise ValueError("client model changed after commitment")
        global_root = state_commitment(global_state, self.cfg.hash_algorithm)
        metrics = model.evaluate_state(client_state, validation_x, validation_y)
        probabilities = model.predict_probabilities(client_state, validation_x)
        # The software evaluator's four counts. Under the rate rule no threshold need
        # exist -- equal scores straddling the k-th place leave no threshold that
        # declares exactly k rows positive -- and that is a refusal rather than a
        # fallback. Threshold-rule counts are recorded either way, so a rejected
        # submission still carries a comparable operating point.
        witness_exists = True
        if self.cfg.decision_rule == "rate":
            rate_counts = confusion_counts_at_rate(
                validation_y, probabilities, self.predicted_positive_rate
            )
            witness_exists = rate_counts is not None
        else:
            rate_counts = None
        counts = (
            rate_counts
            if rate_counts is not None
            else confusion_counts(validation_y, probabilities, self.cfg.decision_threshold)
        )
        vector = state_to_vector(client_state)
        copy_distance = float(np.linalg.norm(vector - state_to_vector(global_state)))
        # Ablations: without the PoV utility gate every update is admitted; without
        # the in-circuit copy detector a replayed global model is no longer rejected.
        if not self.ablation.pov_enabled:
            meets_threshold = True
        elif not witness_exists:
            meets_threshold = False
        elif self.cfg.predicate == "class_aware":
            meets_threshold = passes_class_aware(
                counts,
                self.cfg.threshold_sensitivity,
                self.cfg.threshold_specificity,
                self.cfg.tolerance,
            )
        else:
            meets_threshold = passes_raw_accuracy(
                counts, self.cfg.threshold, self.cfg.tolerance
            )
        not_a_copy = (
            copy_distance >= self.model_cfg.copy_epsilon
            if self.ablation.copy_detector_enabled
            else True
        )
        accepted = bool(meets_threshold and not_a_copy and np.isfinite(vector).all() and np.isfinite(probabilities).all())
        policy = {
            "predicate": self.cfg.predicate,
            "threshold": self.cfg.threshold,
            "threshold_sensitivity": self.cfg.threshold_sensitivity,
            "threshold_specificity": self.cfg.threshold_specificity,
            "tolerance": self.cfg.tolerance,
            "decision_rule": self.cfg.decision_rule,
            "decision_threshold": self.cfg.decision_threshold,
            "predicted_positive_rate": self.predicted_positive_rate,
            "copy_epsilon": self.model_cfg.copy_epsilon,
            "pov_enabled": self.ablation.pov_enabled,
            "copy_detector_enabled": self.ablation.copy_detector_enabled,
        }
        transcript_payload = json.dumps({
            "domain": "XFedAgent:software-pov:v1",
            "client_id": client_id,
            "round_index": round_index,
            "model_root": model_root,
            "global_root": global_root,
            "validation_root": challenge.validation_root,
            "counts": counts.to_dict(),
            "policy": policy,
            "accepted": accepted,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        proof_hash = digest_bytes(transcript_payload, self.cfg.hash_algorithm)
        proof_seconds = time.perf_counter() - start
        modeled = self.cfg.proof_seconds_per_sample * self.cfg.validation_size
        transcript = ProofTranscript(
            client_id=client_id,
            round_index=round_index,
            model_root=model_root,
            validation_root=challenge.validation_root,
            proof_hash=proof_hash,
            accuracy=counts.accuracy,
            threshold=self.cfg.threshold,
            copy_distance=copy_distance,
            sensitivity=counts.sensitivity,
            specificity=counts.specificity,
            balanced_accuracy=counts.balanced_accuracy,
            predicate=self.cfg.predicate,
            accepted=accepted,
            backend=self.cfg.backend,
            proof_seconds=proof_seconds,
            modeled_proof_seconds=modeled,
            global_root=global_root,
            confusion=counts.to_dict(),
            policy=policy,
        )
        return transcript, metrics
