from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_sym, fwht_last_dim, quantize_sym, timed


class QuaRotKVQuantizer(KVQuantizer):
    """
    KV-cache-only QuaRot-style baseline:
    - Apply orthogonal Hadamard rotation on channel axis
    - RTN quantize in rotated space
    - Dequantize then inverse-rotate on read
    """

    def __init__(self, bits: int = 4, block_size: int = 16) -> None:
        super().__init__(bits=bits, block_size=block_size, name=f"QUAROT_KV_INT{bits}")

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht_last_dim(x)

    def _inv_rotate(self, x: torch.Tensor) -> torch.Tensor:
        # normalized Hadamard is involutory: H^-1 == H
        return fwht_last_dim(x)

    def _quantize_tensor(self, x: torch.Tensor) -> Dict[str, Any]:
        xr = self._rotate(x)
        xb, pad_len = _reshape_blocks(xr, self.block_size)
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
        x = dequantize_sym(state["q"], state["scale"])
        x = _unshape_blocks(x, state["pad_len"], state["orig_shape"][1])
        return self._inv_rotate(x)

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
