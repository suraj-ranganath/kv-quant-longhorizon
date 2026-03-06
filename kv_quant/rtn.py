from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_sym, quantize_sym, timed


class RTNQuantizer(KVQuantizer):
    def __init__(self, bits: int = 4, block_size: int = 16) -> None:
        super().__init__(bits=bits, block_size=block_size, name=f"RTN_INT{bits}")

    def _quantize_tensor(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        q, scale = quantize_sym(xb, bits=self.bits, reduce_dims=(2,))
        return {
            "q": q,
            "scale": scale,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "block_size": self.block_size,
        }

    def _dequantize_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        scale = state["scale"]
        pad_len = state["pad_len"]
        orig_len = state["orig_shape"][1]
        x = dequantize_sym(q, scale)
        return _unshape_blocks(x, pad_len, orig_len)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            k_state = self._quantize_tensor(k)
            v_state = self._quantize_tensor(v)
            state = {"k": k_state, "v": v_state}
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            k = self._dequantize_tensor(state["k"])
            v = self._dequantize_tensor(state["v"])
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        def _bytes_for_tensor(s: Dict[str, Any]) -> int:
            q = s["q"]
            scale = s["scale"]
            return packed_bytes(q.numel(), self.bits) + scale.numel() * 2

        return _bytes_for_tensor(state["k"]) + _bytes_for_tensor(state["v"])

    def estimate_active_kv_bytes(
        self,
        active_tokens: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        num_blocks = (max(active_tokens, 0) + self.block_size - 1) // self.block_size
        q_values = batch_size * num_blocks * self.block_size * num_heads * head_dim
        scale_values = batch_size * num_blocks * num_heads * head_dim
        per_tensor = packed_bytes(q_values, self.bits) + scale_values * 2
        return per_tensor * 2
