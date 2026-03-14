from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .kivi import KIVIQuantizer
from .prq import PRQQuantizer
from .quarot_kv import QuaRotKVQuantizer
from .rtn import RTNQuantizer
from .utils import timed


class TPTQQuantizer(KVQuantizer):
    """
    Temporal Progressive Tiered Quantization:
    - Recent tokens use higher precision quantizer.
    - Older tokens use PRQ low-bit quantization.
    - Old-key outliers are preserved in higher precision.
    """

    _METHOD_TO_CLASS = {
        "RTN": RTNQuantizer,
        "KIVI": KIVIQuantizer,
        "QUAROT_KV": QuaRotKVQuantizer,
    }

    def __init__(
        self,
        bits: int = 2,
        block_size: int = 16,
        recent_ratio: float = 0.3,
        recent_bits: int = 4,
        recent_method: str = "RTN",
        residual_bits: int = 2,
        outlier_threshold: float = 6.0,
        outlier_max_ratio: float = 0.005,
    ) -> None:
        if not (0.0 < recent_ratio <= 1.0):
            raise ValueError("recent_ratio must be in (0, 1].")
        if recent_bits not in (2, 4):
            raise ValueError("recent_bits must be 2 or 4.")
        if residual_bits not in (2, 4):
            raise ValueError("residual_bits must be 2 or 4.")
        if outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be > 0.")
        if not (0.0 <= outlier_max_ratio <= 1.0):
            raise ValueError("outlier_max_ratio must be in [0, 1].")

        recent_method = recent_method.upper()
        if recent_method not in self._METHOD_TO_CLASS:
            raise ValueError(f"Unsupported recent_method={recent_method}")

        self.recent_ratio = float(recent_ratio)
        self.recent_bits = int(recent_bits)
        self.recent_method = recent_method
        self.residual_bits = int(residual_bits)
        self.outlier_threshold = float(outlier_threshold)
        self.outlier_max_ratio = float(outlier_max_ratio)
        self._recent_ratio_history: list[float] = []
        self._outlier_ratio_history: list[float] = []

        super().__init__(bits=bits, block_size=block_size, name=f"TPTQ_INT{bits}")
        self._recent_quantizer = self._METHOD_TO_CLASS[self.recent_method](bits=self.recent_bits, block_size=block_size)
        self._old_quantizer = PRQQuantizer(bits=self.bits, block_size=block_size, residual_bits=self.residual_bits)

    def reset_stats(self) -> None:
        super().reset_stats()
        self._recent_ratio_history.clear()
        self._outlier_ratio_history.clear()

    def _build_recent_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        recent_tokens = max(1, int(round(self.recent_ratio * seq_len)))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[-recent_tokens:] = True
        return mask

    def _extract_old_key_outliers(self, old_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if old_k.numel() == 0:
            return (
                torch.empty(0, device=old_k.device, dtype=torch.int32),
                torch.empty(0, device=old_k.device, dtype=torch.float16),
                0.0,
            )
        max_outliers = int(self.outlier_max_ratio * old_k.numel())
        if max_outliers == 0:
            return (
                torch.empty(0, device=old_k.device, dtype=torch.int32),
                torch.empty(0, device=old_k.device, dtype=torch.float16),
                0.0,
            )
        std = old_k.std(unbiased=False).clamp_min(1e-8)
        bound = self.outlier_threshold * std
        mask = old_k.abs() > bound
        if not mask.any():
            return (
                torch.empty(0, device=old_k.device, dtype=torch.int32),
                torch.empty(0, device=old_k.device, dtype=torch.float16),
                0.0,
            )
        flat = old_k.reshape(-1)
        idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        if idx.numel() > max_outliers:
            idx64 = idx.to(torch.int64)
            outlier_scores = flat.abs().index_select(0, idx64)
            selected = torch.topk(outlier_scores, k=max_outliers, largest=True, sorted=False).indices
            idx = idx.index_select(0, selected.to(torch.int64))
        vals = flat.index_select(0, idx.to(torch.int64)).to(torch.float16)
        ratio = float(idx.numel()) / float(max(1, old_k.numel()))
        return idx, vals, ratio

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            recent_mask = self._build_recent_mask(int(k.shape[1]), k.device)
            recent_idx = recent_mask.nonzero(as_tuple=False).squeeze(-1)
            old_idx = (~recent_mask).nonzero(as_tuple=False).squeeze(-1)

            state: Dict[str, Any] = {
                "recent_mask": recent_mask,
                "recent_idx": recent_idx,
                "old_idx": old_idx,
                "orig_shape": tuple(k.shape),
                "recent_state": None,
                "old_state": None,
                "old_key_outlier_idx": None,
                "old_key_outlier_vals": None,
            }

            if recent_idx.numel() > 0:
                rk = k.index_select(1, recent_idx)
                rv = v.index_select(1, recent_idx)
                state["recent_state"] = self._recent_quantizer.quantize_kv(rk, rv, meta=meta)

            if old_idx.numel() > 0:
                ok = k.index_select(1, old_idx)
                ov = v.index_select(1, old_idx)
                outlier_idx, outlier_vals, outlier_ratio = self._extract_old_key_outliers(ok)
                state["old_key_outlier_idx"] = outlier_idx
                state["old_key_outlier_vals"] = outlier_vals
                state["old_state"] = self._old_quantizer.quantize_kv(ok, ov, meta=meta)
            else:
                outlier_ratio = 0.0

        self._recent_ratio_history.append(float(recent_mask.float().mean().item()))
        self._outlier_ratio_history.append(float(outlier_ratio))
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            recent_mask: torch.Tensor = state["recent_mask"]
            b, l, h, d = state["orig_shape"]
            device = recent_mask.device
            tensor_dtype = meta.get("tensor_dtype") if meta else None
            target_dtype = tensor_dtype if tensor_dtype is not None else torch.float32
            k = torch.zeros((b, l, h, d), device=device, dtype=target_dtype)
            v = torch.zeros_like(k)

            recent_idx = state.get("recent_idx")
            old_idx = state.get("old_idx")
            if recent_idx is None:
                recent_idx = recent_mask.nonzero(as_tuple=False).squeeze(-1)
            if old_idx is None:
                old_idx = (~recent_mask).nonzero(as_tuple=False).squeeze(-1)

            if state["recent_state"] is not None and recent_idx.numel() > 0:
                rk, rv = self._recent_quantizer.dequantize_kv(state["recent_state"], meta=meta)
                k.index_copy_(1, recent_idx, rk.to(device=device, dtype=target_dtype))
                v.index_copy_(1, recent_idx, rv.to(device=device, dtype=target_dtype))

            if state["old_state"] is not None and old_idx.numel() > 0:
                ok, ov = self._old_quantizer.dequantize_kv(state["old_state"], meta=meta)
                outlier_idx = state.get("old_key_outlier_idx")
                outlier_vals = state.get("old_key_outlier_vals")
                if outlier_idx is not None and outlier_vals is not None and outlier_idx.numel() > 0:
                    flat_ok = ok.reshape(-1)
                    flat_ok.index_copy_(0, outlier_idx.to(torch.int64), outlier_vals.to(torch.float32))
                    ok = flat_ok.reshape_as(ok)
                k.index_copy_(1, old_idx, ok.to(device=device, dtype=target_dtype))
                v.index_copy_(1, old_idx, ov.to(device=device, dtype=target_dtype))
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        total = int(state["recent_mask"].numel())
        if state.get("recent_idx") is not None:
            total += int(state["recent_idx"].numel() * 8)
        if state.get("old_idx") is not None:
            total += int(state["old_idx"].numel() * 8)
        if state.get("recent_state") is not None:
            total += int(self._recent_quantizer.memory_bytes(state["recent_state"]))
        if state.get("old_state") is not None:
            total += int(self._old_quantizer.memory_bytes(state["old_state"]))
        outlier_idx = state.get("old_key_outlier_idx")
        outlier_vals = state.get("old_key_outlier_vals")
        if outlier_idx is not None:
            total += int(outlier_idx.numel() * 4)
        if outlier_vals is not None:
            total += int(outlier_vals.numel() * 2)
        return total

    def diagnostics(self) -> Dict[str, float]:
        events = len(self._recent_ratio_history)
        if events == 0:
            return {}
        return {
            "tptq_events": float(events),
            "tptq_avg_recent_ratio": float(sum(self._recent_ratio_history) / events),
            "tptq_avg_outlier_ratio": float(sum(self._outlier_ratio_history) / events),
            "tptq_config_recent_ratio": float(self.recent_ratio),
            "tptq_recent_bits": float(self.recent_bits),
            "tptq_old_bits": float(self.bits),
            "tptq_residual_bits": float(self.residual_bits),
            "tptq_outlier_threshold": float(self.outlier_threshold),
            "tptq_outlier_max_ratio": float(self.outlier_max_ratio),
        }
