from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import EnergyConfig


@dataclass(frozen=True)
class EnergyBreakdown:
    local_training_wh: float
    proving_wh: float
    communication_wh: float
    idle_wh: float
    total_wh: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_energy(
    cfg: EnergyConfig,
    training_seconds: float,
    proving_seconds: float,
    communication_seconds: float,
    idle_seconds: float,
) -> EnergyBreakdown:
    """Per-update energy from a calibrated four-component power model.

    Each phase draws a distinct constant power: active local training, ZK
    proving, cross-chain message propagation, and the idle baseline the device
    draws while awaiting destination-chain finality. Energy is the product of
    phase power and phase duration, expressed in watt-hours.
    """
    if not cfg.enabled:
        return EnergyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)
    local = training_seconds * cfg.local_training_watts / 3600.0
    proving = proving_seconds * cfg.proving_watts / 3600.0
    communication = communication_seconds * cfg.communication_watts / 3600.0
    idle = idle_seconds * cfg.idle_watts / 3600.0
    return EnergyBreakdown(
        local_training_wh=local,
        proving_wh=proving,
        communication_wh=communication,
        idle_wh=idle,
        total_wh=local + proving + communication + idle,
    )
