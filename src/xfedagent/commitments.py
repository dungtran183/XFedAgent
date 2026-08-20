from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib

import numpy as np


def digest_bytes(payload: bytes, algorithm: str = "sha3_256") -> str:
    if algorithm == "sha3_256":
        return hashlib.sha3_256(payload).hexdigest()
    if algorithm == "blake2b":
        return hashlib.blake2b(payload, digest_size=32).hexdigest()
    raise ValueError(f"unsupported hash algorithm: {algorithm}")


def array_payload(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    return contiguous.dtype.str.encode("ascii") + b":" + str(contiguous.shape).encode("ascii") + b":" + contiguous.tobytes()


@dataclass(frozen=True)
class MerkleTree:
    root: str
    leaves: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]


class MerkleCommitter:
    def __init__(self, algorithm: str = "sha3_256", chunk_bytes: int = 1024) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.algorithm = algorithm
        self.chunk_bytes = chunk_bytes

    def build(self, payload: bytes) -> MerkleTree:
        leaves = tuple(self._leaf_hash(chunk) for chunk in self._chunks(payload))
        if not leaves:
            leaves = (self._leaf_hash(b""),)
        levels: list[tuple[str, ...]] = [leaves]
        current = leaves
        while len(current) > 1:
            next_level = []
            for left, right in self._pairs(current):
                next_level.append(digest_bytes(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right), self.algorithm))
            current = tuple(next_level)
            levels.append(current)
        return MerkleTree(root=current[0], leaves=leaves, levels=tuple(levels))

    def commit_array(self, array: np.ndarray) -> MerkleTree:
        return self.build(array_payload(array))

    def _chunks(self, payload: bytes) -> Iterable[bytes]:
        for offset in range(0, len(payload), self.chunk_bytes):
            yield payload[offset : offset + self.chunk_bytes]

    def _leaf_hash(self, chunk: bytes) -> str:
        return digest_bytes(b"\x00" + chunk, self.algorithm)

    @staticmethod
    def _pairs(values: tuple[str, ...]) -> Iterable[tuple[str, str]]:
        count = len(values)
        for index in range(0, count, 2):
            left = values[index]
            right = values[index + 1] if index + 1 < count else left
            yield left, right


def quantize_vector(vector: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    vector = np.asarray(vector, dtype=np.float64)
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs == 0.0:
        return np.zeros_like(vector, dtype=np.int64), 1.0
    max_int = 2 ** (bits - 1) - 1
    scale = max_abs / max_int
    quantized = np.clip(np.rint(vector / scale), -max_int, max_int).astype(np.int64)
    return quantized, scale


def stable_config_digest(mapping: dict) -> str:
    import json

    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest_bytes(payload)

