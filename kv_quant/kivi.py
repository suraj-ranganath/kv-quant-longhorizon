from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_asym, quantize_asym, timed


class KIVIQuantizer(KVQuantizer):
    """
    KIVI-style asymmetric quantization:
    - keys: per-channel along sequence blocks
    - values: per-token along channel dimensions
    """

    def __init__(self, bits: int = 4, block_size: int = 16) -> None:
        super().__init__(bits=bits, block_size=block_size, name=f"KIVI_INT{bits}")

    def _quantize_keys(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        # per-channel over block tokens
        q, scale, zp = quantize_asym(xb, bits=self.bits, reduce_dims=(2,))
        return {
            "q": q,
            "scale": scale,
            "zp": zp,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "block_size": self.block_size,
        }

    def _quantize_values(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        # per-token (single scale/zero-point per token over head and channel)
        q, scale, zp = quantize_asym(xb, bits=self.bits, reduce_dims=(3, 4))
        return {
            "q": q,
            "scale": scale,
            "zp": zp,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "bits": self.bits,
            "block_size": self.block_size,
        }

    def _dequantize(self, state: Dict[str, Any]) -> torch.Tensor:
        x = dequantize_asym(state["q"], state["scale"], state["zp"])
        return _unshape_blocks(x, state["pad_len"], state["orig_shape"][1])

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            k_state = self._quantize_keys(k)
            v_state = self._quantize_values(v)
            state = {"k": k_state, "v": v_state}
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            k = self._dequantize(state["k"])
            v = self._dequantize(state["v"])
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        def _bytes_for_tensor(s: Dict[str, Any]) -> int:
            q = s["q"]
            scale = s["scale"]
            zp = s["zp"]
            return packed_bytes(q.numel(), self.bits) + scale.numel() * 2 + zp.numel() * 2

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
        key_scale_values = batch_size * num_blocks * num_heads * head_dim
        value_scale_values = batch_size * num_blocks * self.block_size
        key_bytes = packed_bytes(q_values, self.bits) + key_scale_values * 2 + key_scale_values * 2
        value_bytes = packed_bytes(q_values, self.bits) + value_scale_values * 2 + value_scale_values * 2
        return key_bytes + value_bytes
