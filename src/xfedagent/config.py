from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DataConfig:
    source: str
    samples: int
    clients: int
    timesteps: int
    features: int
    validation_pool_size: int
    validation_fraction: float
    test_fraction: float
    dirichlet_alpha: float
    csv_path: str | None = None
    label_column: str = "label"
    patient_column: str = "patient_id"
    #: Positive-class prevalence for the synthetic surrogate. 0.5 gives a balanced
    #: cohort; set it to the real cohort's prevalence to reproduce its imbalance.
    positive_rate: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    learning_rate: float
    weight_decay: float
    local_epochs: int
    batch_size: int
    hidden_channels: tuple[int, ...]
    quantization_bits: int
    copy_epsilon: float


@dataclass(frozen=True)
class PoVConfig:
    backend: str
    threshold: float
    validation_size: int
    tolerance: float
    rotation: str
    hash_algorithm: str
    proof_seconds_per_sample: float
    snarkjs_bin: str = "snarkjs"
    circuit_wasm: str | None = None
    proving_key: str | None = None
    #: "raw_accuracy" reproduces the original gate; "class_aware" enforces
    #: sensitivity and specificity separately. The class-aware predicate is the
    #: default because a raw-accuracy threshold below the majority prevalence is
    #: satisfiable by a constant classifier on an imbalanced validation set.
    predicate: str = "class_aware"
    threshold_sensitivity: float = 0.60
    threshold_specificity: float = 0.60
    #: Draw the per-round validation subset with equal numbers of each class.
    #: The concentration bound on the class-aware predicate is governed by the
    #: smaller class count, so sampling in proportion to a skewed pool makes it
    #: nearly vacuous; balanced stratification restores it.
    balanced_validation: bool = True


@dataclass(frozen=True)
class FederationConfig:
    rounds: int
    clients_per_round: int
    byzantine_fraction: float
    attacks: tuple[str, ...]
    seed: int
    reputation_initial: float
    reputation_min: float
    reputation_reward_alpha: float
    reputation_penalty_beta: float


@dataclass(frozen=True)
class RelayConfig:
    enabled: bool
    chains: tuple[str, ...]
    relayers: int
    faulty_relayers: int
    source_finality_blocks: int
    source_block_seconds: float
    destination_finality_seconds: float
    gas_verify_base: int
    gas_reputation_update: int


@dataclass(frozen=True)
class EnergyConfig:
    enabled: bool
    local_training_watts: float
    proving_watts: float
    communication_watts: float
    idle_watts: float


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    save_round_models: bool


@dataclass(frozen=True)
class AblationConfig:
    """Switches that disable individual XFedAgent components.

    Every field defaults to True, so an existing configuration file that omits
    the ``ablation`` section reproduces the full framework unchanged.
    """

    pov_enabled: bool = True
    copy_detector_enabled: bool = True
    reputation_enabled: bool = True
    rotation_enabled: bool = True
    cross_chain_enabled: bool = True

    @property
    def label(self) -> str:
        """Compact identifier used in ablation result tables."""
        if self.is_full:
            return "full"
        off = [
            name
            for name, on in (
                ("pov", self.pov_enabled),
                ("copy", self.copy_detector_enabled),
                ("rep", self.reputation_enabled),
                ("rot", self.rotation_enabled),
                ("xchain", self.cross_chain_enabled),
            )
            if not on
        ]
        return "no-" + "+".join(off)

    @property
    def is_full(self) -> bool:
        return all(
            (
                self.pov_enabled,
                self.copy_detector_enabled,
                self.reputation_enabled,
                self.rotation_enabled,
                self.cross_chain_enabled,
            )
        )


@dataclass(frozen=True)
class ToolchainConfig:
    require_snarkjs: bool = False
    require_qemu: bool = False
    require_ipfs: bool = False
    require_hardhat: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    pov: PoVConfig
    federation: FederationConfig
    relay: RelayConfig
    energy: EnergyConfig
    output: OutputConfig
    toolchain: ToolchainConfig = field(default_factory=ToolchainConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    required = {"name", "data", "model", "pov", "federation", "relay", "energy", "output"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing config sections: {', '.join(missing)}")

    data = DataConfig(**raw["data"])
    model_raw = dict(raw["model"])
    model_raw["hidden_channels"] = tuple(int(v) for v in model_raw["hidden_channels"])
    model = ModelConfig(**model_raw)
    pov = PoVConfig(**raw["pov"])
    fed_raw = dict(raw["federation"])
    fed_raw["attacks"] = tuple(str(v) for v in fed_raw.get("attacks", []))
    federation = FederationConfig(**fed_raw)
    relay_raw = dict(raw["relay"])
    relay_raw["chains"] = tuple(str(v) for v in relay_raw.get("chains", []))
    relay = RelayConfig(**relay_raw)
    energy = EnergyConfig(**raw["energy"])
    output = OutputConfig(**raw["output"])
    toolchain = ToolchainConfig(**raw.get("toolchain", {}))
    ablation = AblationConfig(**raw.get("ablation", {}))

    cfg = ExperimentConfig(
        name=str(raw["name"]),
        data=data,
        model=model,
        pov=pov,
        federation=federation,
        relay=relay,
        energy=energy,
        output=output,
        toolchain=toolchain,
        ablation=ablation,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.data.clients <= 0:
        raise ValueError("data.clients must be positive")
    if cfg.federation.clients_per_round > cfg.data.clients:
        raise ValueError("clients_per_round cannot exceed data.clients")
    if not 0.0 <= cfg.federation.byzantine_fraction < 1.0:
        raise ValueError("byzantine_fraction must be in [0, 1)")
    if cfg.pov.validation_size > cfg.data.validation_pool_size:
        raise ValueError("validation_size cannot exceed validation_pool_size")
    if not 0.0 < cfg.pov.threshold < 1.0:
        raise ValueError("pov.threshold must be in (0, 1)")
    if cfg.pov.predicate not in {"raw_accuracy", "class_aware"}:
        raise ValueError("pov.predicate must be raw_accuracy or class_aware")
    for name in ("threshold_sensitivity", "threshold_specificity"):
        value = getattr(cfg.pov, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"pov.{name} must be in (0, 1)")
    if cfg.pov.backend not in {"software", "snarkjs"}:
        raise ValueError("pov.backend must be software or snarkjs")
    if cfg.pov.backend == "snarkjs" and (cfg.pov.circuit_wasm is None or cfg.pov.proving_key is None):
        raise ValueError("snarkjs backend requires pov.circuit_wasm and pov.proving_key")
    if cfg.relay.enabled and cfg.relay.relayers <= 3 * cfg.relay.faulty_relayers:
        raise ValueError("relay requires relayers > 3 * faulty_relayers")
    if not 0.0 < cfg.data.positive_rate < 1.0:
        raise ValueError("data.positive_rate must be in (0, 1)")
    if cfg.model.quantization_bits < 2 or cfg.model.quantization_bits > 16:
        raise ValueError("quantization_bits must be in [2, 16]")
    if cfg.ablation.cross_chain_enabled and not cfg.relay.enabled:
        raise ValueError(
            "ablation.cross_chain_enabled requires relay.enabled; disable both to run single-chain"
        )
