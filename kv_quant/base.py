from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class QuantizerStats:
    quantize_time_s: float = 0.0
    dequantize_time_s: float = 0.0
    quantize_calls: int = 0
    dequantize_calls: int = 0
    bf16_kv_bytes: int = 0
    compressed_kv_bytes: int = 0


class KVQuantizer(ABC):
    def __init__(
        self,
        bits: int = 4,
        block_size: int = 16,
        name: str = "BASE",
        key_bits: int | None = None,
        value_bits: int | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        if key_bits not in (2, 4):
            raise ValueError(f"Unsupported key_bits={key_bits}; expected one of [2, 4].")
        if value_bits not in (2, 4):
            raise ValueError(f"Unsupported value_bits={value_bits}; expected one of [2, 4].")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.bits = bits
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.block_size = block_size
        self._name = name
        self.stats = QuantizerStats()

    def name(self) -> str:
        return self._name

    def reset_stats(self) -> None:
        self.stats = QuantizerStats()

    @abstractmethod
    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def memory_bytes(self, state: Dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        raise NotImplementedError
