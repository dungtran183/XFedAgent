from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from .metrics import positive_count_for_rate


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
    #: The probability above which a score counts as a positive prediction when the
    #: confusion counts are formed. 0.5 is the right operating point only when the
    #: cohort's prevalence is near a half, so the gate names it rather than leaving
    #: it implicit. This is a software-evaluation policy; the binary-linear
    #: reference circuit has a fixed zero-logit prediction rule.
    decision_threshold: float = 0.5
    #: How the four integer confusion counts are formed from the model's scores.
    #: Both rules are software-only. "threshold" is the default. "rate" fixes the
    #: number of rows declared positive
    #: instead of the probability at which a row becomes positive: under a non-IID
    #: partition each client's score scale follows its own shard, so one public
    #: threshold means a different operating point to every prover while one public
    #: rate means the same one to all of them.
    decision_rule: str = "threshold"
    #: The fraction of challenge rows declared positive when decision_rule="rate".
    #: ``None`` derives it in closed form from the challenge prevalence and the
    #: thresholds: a model sitting at Se = Sp = tau produces
    #: ``rate = pi * tau + (1 - pi) * (1 - tau)``, so that is where the gate should
    #: stand. Name it explicitly only to override that.
    predicted_positive_rate: float | None = None
    #: Draw the per-round validation subset with equal numbers of each class.
    #: The concentration bound on the class-aware predicate is governed by the
    #: smaller class count, so sampling in proportion to a skewed pool makes it
    #: nearly vacuous; balanced stratification restores it.
    balanced_validation: bool = True


def challenge_prevalence(cfg: "ExperimentConfig") -> float:
    """The positive-class fraction of the validation subset the gate will draw.

    Balanced stratification takes half the challenge from each class, so the
    challenge prevalence is a half however skewed the cohort is; without it the
    subset is drawn in proportion and the cohort's prevalence carries through. The
    distinction matters because the rate the gate should stand at is a function of
    the *challenge* prevalence, not the cohort's -- reading the cohort's 0.14 off the
    config where the challenge is 50/50 puts the gate 10 points from where it
    belongs. When the pool holds fewer positives than half the challenge the
    realised fraction is lower than a half, in which case name the rate explicitly.

    Balanced stratification takes ``validation_size // 2`` positives, so for an odd
    challenge size the fraction is just under a half rather than exactly a half. The
    floor division is reproduced here rather than rounded away, because the rate rule
    attains the reachable optimum exactly when the count it declares equals the number
    of positives, and one row is the whole of that margin on a hundred-row challenge.
    """

    if cfg.pov.balanced_validation:
        return (cfg.pov.validation_size // 2) / cfg.pov.validation_size
    return float(cfg.data.positive_rate)


def predicted_positive_rate_for(pov: "PoVConfig", prevalence: float) -> float:
    """Where the gate stands under the rate rule: the configured value or the closed form.

    A model whose sensitivity and specificity both sit at ``tau`` declares
    ``tau * P + (1 - tau) * N`` of the ``n`` rows positive, so as a fraction

        rate*(pi, tau) = pi * tau + (1 - pi) * (1 - tau).

    That is the rate at which the predicate's own target is realisable, and it needs
    no sweep. On a balanced challenge it collapses to a half for any tau, which is
    the whole of the rule there: declare the better-scoring half positive.
    """

    if pov.predicted_positive_rate is not None:
        return float(pov.predicted_positive_rate)
    tau = (
        max(pov.threshold_sensitivity, pov.threshold_specificity)
        if pov.predicate == "class_aware"
        else pov.threshold
    )
    pi = float(prevalence)
    return float(pi * tau + (1.0 - pi) * (1.0 - tau))


def resolved_predicted_positive_rate(cfg: "ExperimentConfig") -> float:
    """The rate the configured gate will use, with the challenge prevalence supplied."""

    return predicted_positive_rate_for(cfg.pov, challenge_prevalence(cfg))


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
    for name in ("threshold_sensitivity", "threshold_specificity", "decision_threshold"):
        value = getattr(cfg.pov, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"pov.{name} must be in (0, 1)")
    if cfg.pov.decision_rule not in {"threshold", "rate"}:
        raise ValueError("pov.decision_rule must be threshold or rate")
    if cfg.pov.predicted_positive_rate is not None and not 0.0 < cfg.pov.predicted_positive_rate < 1.0:
        raise ValueError("pov.predicted_positive_rate must be in (0, 1)")
    if cfg.pov.decision_rule == "rate":
        # A rate that rounds to no rows or to every row leaves one of the two rates
        # zero by construction, so the gate would refuse every submission. Caught
        # here rather than surfacing later as a cohort that fails the thresholds.
        count = positive_count_for_rate(
            resolved_predicted_positive_rate(cfg), cfg.pov.validation_size
        )
        if not 0 < count < cfg.pov.validation_size:
            raise ValueError(
                "pov.decision_rule=rate needs a rate that declares between 1 and "
                f"validation_size-1 rows positive; got {count} of {cfg.pov.validation_size}"
            )
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


# The two predicate arms are a paired comparison, so they must share the seed, the
# data split, the model initialisation, the client partition and the attack
# schedule; only the gate itself may move. The raw mappings are compared rather
# than the parsed dataclasses, so an unmodelled field cannot drift between arms.
PAIRED_ALLOWED_DIFFERENCES = frozenset(
    {
        "name",
        "pov.predicate",
        "pov.balanced_validation",
        "pov.threshold",
        "pov.threshold_sensitivity",
        "pov.threshold_specificity",
    }
)

_MISSING = object()


def flatten_config(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a config mapping to dotted paths for field-by-field comparison."""

    flat: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def paired_differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return every dotted path whose value differs between two raw configs."""

    first, second = flatten_config(left), flatten_config(right)
    differences: dict[str, tuple[Any, Any]] = {}
    for path in sorted(set(first) | set(second)):
        a, b = first.get(path, _MISSING), second.get(path, _MISSING)
        if a != b:
            differences[path] = (None if a is _MISSING else a, None if b is _MISSING else b)
    return differences


def validate_config_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    allowed: frozenset[str] = PAIRED_ALLOWED_DIFFERENCES,
) -> dict[str, tuple[Any, Any]]:
    """Raise unless two configs form an admissible paired comparison.

    Returns the differences that were allowed, so a caller can log exactly what
    the two arms disagree on.
    """

    differences = paired_differences(left, right)
    illegal = {path: value for path, value in differences.items() if path not in allowed}
    if illegal:
        detail = "; ".join(f"{path}: {a!r} vs {b!r}" for path, (a, b) in sorted(illegal.items()))
        raise ValueError(f"paired configs differ outside the gate: {detail}")
    if "pov.predicate" not in differences:
        raise ValueError("paired configs enforce the same pov.predicate, so they are not a comparison")
    return differences
