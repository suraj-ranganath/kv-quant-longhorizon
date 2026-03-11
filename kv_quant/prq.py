from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .packing import packed_bytes
from .utils import _reshape_blocks, _unshape_blocks, dequantize_sym, quantize_sym, timed


class PRQQuantizer(KVQuantizer):
    """
    Progressive residual quantization:
    - Stage 1 quantizes the original tensor.
    - Stage 2 quantizes the residual after stage 1 reconstruction.
    """

    def __init__(self, bits: int = 2, block_size: int = 16, residual_bits: int = 4) -> None:
        if residual_bits not in (2, 4):
            raise ValueError("residual_bits must be 2 or 4")
        self.residual_bits = int(residual_bits)
        self.quant_chunk_blocks = 64
        super().__init__(bits=bits, block_size=block_size, name=f"PRQ_INT{bits}")

    def _quantize_tensor(self, x: torch.Tensor) -> Dict[str, Any]:
        xb, pad_len = _reshape_blocks(x, self.block_size)
        scale_shape = list(xb.shape)
        scale_shape[2] = 1
        q_base = torch.empty_like(xb, dtype=torch.int8)
        scale_base = torch.empty(scale_shape, device=xb.device, dtype=torch.float16)
        q_res = torch.empty_like(xb, dtype=torch.int8)
        scale_res = torch.empty(scale_shape, device=xb.device, dtype=torch.float16)

        for start in range(0, xb.shape[1], self.quant_chunk_blocks):
            end = start + self.quant_chunk_blocks
            xb_chunk = xb[:, start:end]
            q_base_chunk, scale_base_chunk = quantize_sym(xb_chunk, bits=self.bits, reduce_dims=(2,))
            x_base_chunk = dequantize_sym(q_base_chunk, scale_base_chunk)
            residual_chunk = xb_chunk - x_base_chunk
            q_res_chunk, scale_res_chunk = quantize_sym(residual_chunk, bits=self.residual_bits, reduce_dims=(2,))
            q_base[:, start:end].copy_(q_base_chunk)
            scale_base[:, start:end].copy_(scale_base_chunk)
            q_res[:, start:end].copy_(q_res_chunk)
            scale_res[:, start:end].copy_(scale_res_chunk)
        return {
            "q_base": q_base,
            "scale_base": scale_base,
            "q_res": q_res,
            "scale_res": scale_res,
            "pad_len": pad_len,
            "orig_shape": tuple(x.shape),
        }

    def _dequantize_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        x_base = dequantize_sym(state["q_base"], state["scale_base"])
        x_res = dequantize_sym(state["q_res"], state["scale_res"])
        x = x_base + x_res
        return _unshape_blocks(x, state["pad_len"], state["orig_shape"][1])

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
            base = packed_bytes(s["q_base"].numel(), self.bits) + s["scale_base"].numel() * 2
            residual = packed_bytes(s["q_res"].numel(), self.residual_bits) + s["scale_res"].numel() * 2
            return base + residual

        return _bytes_for_tensor(state["k"]) + _bytes_for_tensor(state["v"])
