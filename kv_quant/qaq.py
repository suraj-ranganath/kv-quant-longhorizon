from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_asym, quantize_asym, timed


class QAQQuantizer(KVQuantizer):
    """
    QAQ-style quantization with distinct K/V handling and explicit outlier preservation.
    """

    def __init__(self, bits: int = 2, block_size: int = 16, outlier_threshold: float = 6.0) -> None:
        if outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be > 0")
        self.outlier_threshold = float(outlier_threshold)
        super().__init__(bits=bits, block_size=block_size, name=f"QAQ_INT{bits}")

    def _split_outliers(self, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = xb.std(unbiased=False).clamp_min(1e-8)
        bound = self.outlier_threshold * std
        mask = xb.abs() > bound
        if not mask.any():
            empty_idx = torch.empty(0, device=xb.device, dtype=torch.int32)
            empty_vals = torch.empty(0, device=xb.device, dtype=torch.float16)
            return xb, empty_idx, empty_vals
        flat = xb.reshape(-1)
        idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        vals = flat.index_select(0, idx.to(torch.int64)).to(torch.float16)
        clamped = xb.clamp(min=-bound, max=bound)
        return clamped, idx, vals

    def _quantize_tensor(self, x: torch.Tensor, reduce_dims: Tuple[int, ...]) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        clipped, outlier_idx, outlier_vals = self._split_outliers(xb)
        q, scale, zp = quantize_asym(clipped, bits=self.bits, reduce_dims=reduce_dims)
        return {
            "q": q,
            "scale": scale,
            "zp": zp,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
            "outlier_idx": outlier_idx,
            "outlier_vals": outlier_vals,
        }

    def _dequantize_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        x = dequantize_asym(state["q"], state["scale"], state["zp"])
        outlier_idx: torch.Tensor = state["outlier_idx"]
        if outlier_idx.numel() > 0:
            flat = x.reshape(-1)
            flat.index_copy_(0, outlier_idx.to(torch.int64), state["outlier_vals"].to(torch.float32))
            x = flat.reshape_as(x)
        return _unshape_blocks(x, state["pad_len"], state["orig_shape"][1])

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            k_state = self._quantize_tensor(k, reduce_dims=(2,))
            v_state = self._quantize_tensor(v, reduce_dims=(3, 4))
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
            zp = s["zp"]
            outlier_idx = s["outlier_idx"]
            outlier_vals = s["outlier_vals"]
            return (
                packed_bytes(q.numel(), self.bits)
                + scale.numel() * 2
                + zp.numel() * 2
                + outlier_idx.numel() * 4
                + outlier_vals.numel() * 2
            )

        return _bytes_for_tensor(state["k"]) + _bytes_for_tensor(state["v"])
