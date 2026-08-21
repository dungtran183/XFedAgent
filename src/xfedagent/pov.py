"""Software Proof-of-Validation backend.

The model weights and the validation subset are each bound to a single
commitment over the whole quantised vector, matching the in-circuit
construction (one streaming hash, no Merkle indirection or selective opening).
This software backend uses SHA3-256 as a stand-in for the Poseidon-2 sponge
that the Circom circuit uses; both are collision-resistant 256-bit commitments,
and the substitution keeps the simulation free of a native field-arithmetic
dependency without changing the commitment's binding semantics.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from .commitments import array_payload, digest_bytes, quantize_vector
from .config import AblationConfig, ModelConfig, PoVConfig
from .metrics import BinaryMetrics, confusion_counts, passes_class_aware, passes_raw_accuracy
from .model import TorchModel, state_to_vector


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
    ) -> None:
        if cfg.backend != "software":
            raise ValueError("SoftwarePoVBackend requires backend=software")
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.ablation = ablation or AblationConfig()

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
    ) -> tuple[ProofTranscript, BinaryMetrics]:
        start = time.perf_counter()
        metrics = model.evaluate_state(client_state, validation_x, validation_y)
        probabilities = model.predict_probabilities(client_state, validation_x)
        counts = confusion_counts(validation_y, probabilities)
        vector = state_to_vector(client_state)
        quantized, scale = quantize_vector(vector, self.model_cfg.quantization_bits)
        model_payload = array_payload(quantized) + np.asarray([scale], dtype=np.float64).tobytes()
        model_root = digest_bytes(model_payload, self.cfg.hash_algorithm)
        copy_distance = float(np.linalg.norm(vector - state_to_vector(global_state)))
        # Ablations: without the PoV utility gate every update is admitted; without
        # the in-circuit copy detector a replayed global model is no longer rejected.
        if not self.ablation.pov_enabled:
            meets_threshold = True
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
        accepted = bool(meets_threshold and not_a_copy)
        transcript_payload = (
            str(client_id).encode("ascii")
            + str(round_index).encode("ascii")
            + bytes.fromhex(model_root)
            + bytes.fromhex(challenge.validation_root)
            + f"{metrics.accuracy:.12f}:{self.cfg.threshold:.12f}:{copy_distance:.12f}".encode("ascii")
        )
        proof_hash = digest_bytes(transcript_payload, self.cfg.hash_algorithm)
        proof_seconds = time.perf_counter() - start
        modeled = self.cfg.proof_seconds_per_sample * self.cfg.validation_size
        transcript = ProofTranscript(
            client_id=client_id,
            round_index=round_index,
            model_root=model_root,
            validation_root=challenge.validation_root,
            proof_hash=proof_hash,
            accuracy=metrics.accuracy,
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
        )
        return transcript, metrics
