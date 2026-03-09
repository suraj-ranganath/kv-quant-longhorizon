from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_sym, quantize_sym, timed


class RTNQuantizer(KVQuantizer):
    def __init__(
        self,
        bits: int = 4,
        block_size: int = 16,
        key_bits: int | None = None,
        value_bits: int | None = None,
        name: str | None = None,
    ) -> None:
        key_bits = bits if key_bits is None else key_bits
        value_bits = bits if value_bits is None else value_bits
        resolved_name = name or (f"RTN_INT{bits}" if key_bits == value_bits == bits else f"RTN_K{key_bits}_V{value_bits}")
        super().__init__(
            bits=bits,
            block_size=block_size,
            name=resolved_name,
            key_bits=key_bits,
            value_bits=value_bits,
        )

    def _quantize_tensor(self, x: torch.Tensor, bits: int) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        q, scale = quantize_sym(xb, bits=bits, reduce_dims=(2,))
        return {
            "q": q,
            "scale": scale,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": bits,
            "block_size": self.block_size,
        }

    def _dequantize_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        q = state["q"]
        scale = state["scale"]
        pad_len = state["pad_len"]
        orig_len = state["orig_shape"][1]
        dtype = state.get("tensor_dtype", torch.bfloat16)
        x = dequantize_sym(q, scale, dtype=dtype)
        return _unshape_blocks(x, pad_len, orig_len)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            k_state = self._quantize_tensor(k, self.key_bits)
            v_state = self._quantize_tensor(v, self.value_bits)
            tensor_dtype = (meta or {}).get("tensor_dtype", k.dtype)
            k_state["tensor_dtype"] = tensor_dtype
            v_state["tensor_dtype"] = tensor_dtype
            state = {"k": k_state, "v": v_state}
        self.stats.quantize_time_s += t[0]
        self.stats.quantize_calls += 1
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            k = self._dequantize_tensor(state["k"])
            v = self._dequantize_tensor(state["v"])
        self.stats.dequantize_time_s += t[0]
        self.stats.dequantize_calls += 1
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        def _bytes_for_tensor(s: Dict[str, Any]) -> int:
            q = s["q"]
            scale = s["scale"]
            return packed_bytes(q.numel(), int(s.get("bits", self.bits))) + scale.numel() * 2

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
        key_bytes = packed_bytes(q_values, self.key_bits) + scale_values * 2
        value_bytes = packed_bytes(q_values, self.value_bits) + scale_values * 2
        return key_bytes + value_bytes
