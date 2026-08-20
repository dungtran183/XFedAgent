from __future__ import annotations

from dataclasses import asdict, dataclass

from .commitments import digest_bytes
from .config import RelayConfig
from .pov import ProofTranscript


@dataclass(frozen=True)
class RelayReceipt:
    client_id: int
    round_index: int
    source_chain: str
    destination_chain: str
    nonce: int
    quorum: int
    source_finality_seconds: float
    destination_finality_seconds: float
    finality_seconds: float
    gas_used: int
    message_hash: str
    accepted: bool

    def to_dict(self) -> dict:
        return asdict(self)


class CrossChainRelaySimulator:
    def __init__(self, cfg: RelayConfig) -> None:
        self.cfg = cfg
        self.nonces: dict[int, int] = {}
        self.seen: set[str] = set()

    def relay(self, transcript: ProofTranscript) -> RelayReceipt:
        if not self.cfg.enabled:
            return RelayReceipt(
                client_id=transcript.client_id,
                round_index=transcript.round_index,
                source_chain="local",
                destination_chain="local",
                nonce=self.nonces.get(transcript.client_id, 0),
                quorum=0,
                source_finality_seconds=0.0,
                destination_finality_seconds=0.0,
                finality_seconds=0.0,
                gas_used=0,
                message_hash=transcript.proof_hash,
                accepted=transcript.accepted,
            )
        source = self.cfg.chains[0]
        destination = self.cfg.chains[-1]
        nonce = self.nonces.get(transcript.client_id, 0)
        payload = f"{source}:{destination}:{transcript.client_id}:{transcript.round_index}:{nonce}:{transcript.proof_hash}".encode("utf-8")
        message_hash = digest_bytes(payload)
        if message_hash in self.seen:
            raise ValueError("relay replay detected")
        self.seen.add(message_hash)
        self.nonces[transcript.client_id] = nonce + 1
        quorum = (2 * self.cfg.relayers) // 3 + 1
        source_finality = self.cfg.source_finality_blocks * self.cfg.source_block_seconds
        destination_finality = self.cfg.destination_finality_seconds
        finality = source_finality + destination_finality
        gas = self.cfg.gas_verify_base + self.cfg.gas_reputation_update
        return RelayReceipt(
            client_id=transcript.client_id,
            round_index=transcript.round_index,
            source_chain=source,
            destination_chain=destination,
            nonce=nonce,
            quorum=quorum,
            source_finality_seconds=source_finality,
            destination_finality_seconds=destination_finality,
            finality_seconds=finality,
            gas_used=gas,
            message_hash=message_hash,
            accepted=transcript.accepted,
        )

