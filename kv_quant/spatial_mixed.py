from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .kivi import KIVIQuantizer
from .quarot_kv import QuaRotKVQuantizer
from .rtn import RTNQuantizer
from .utils import timed


class SpatialMixedQuantizer(KVQuantizer):
    """
    Spatially-aware mixed-precision quantization:
    - Build a foreground/background token mask from temporal variance.
    - Quantize foreground and background partitions with different quantizers.
    """

    _METHOD_TO_CLASS = {
        "RTN": RTNQuantizer,
        "KIVI": KIVIQuantizer,
        "QUAROT_KV": QuaRotKVQuantizer,
    }

    def __init__(
        self,
        block_size: int = 16,
        fg_method: str = "RTN",
        fg_bits: int = 4,
        bg_method: str = "RTN",
        bg_bits: int = 2,
        mask_policy: str = "hybrid",
        variance_threshold: float = 0.02,
        min_foreground_ratio: float = 0.45,
        max_foreground_ratio: float = 0.85,
        target_foreground_ratio: float = 0.65,
    ) -> None:
        fg_method = fg_method.upper()
        bg_method = bg_method.upper()
        mask_policy = mask_policy.lower()
        if fg_method not in self._METHOD_TO_CLASS:
            raise ValueError(f"Unsupported fg_method={fg_method}")
        if bg_method not in self._METHOD_TO_CLASS:
            raise ValueError(f"Unsupported bg_method={bg_method}")
        if mask_policy not in {"threshold", "topk", "hybrid"}:
            raise ValueError(f"Unsupported mask_policy={mask_policy}")
        if not (0.0 <= min_foreground_ratio <= 1.0 and 0.0 <= max_foreground_ratio <= 1.0):
            raise ValueError("Foreground ratios must be in [0, 1].")
        if min_foreground_ratio > max_foreground_ratio:
            raise ValueError("min_foreground_ratio cannot exceed max_foreground_ratio.")
        if not (0.0 <= target_foreground_ratio <= 1.0):
            raise ValueError("target_foreground_ratio must be in [0, 1].")

        self.fg_method = fg_method
        self.fg_bits = int(fg_bits)
        self.bg_method = bg_method
        self.bg_bits = int(bg_bits)
        self.mask_policy = mask_policy
        self.variance_threshold = float(variance_threshold)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.max_foreground_ratio = float(max_foreground_ratio)
        self.target_foreground_ratio = float(target_foreground_ratio)
        self.frame_seq_length: int | None = None
        self._fg_ratio_history: list[float] = []
        self._threshold_fg_ratio_history: list[float] = []
        self._topk_fallback_history: list[float] = []

        name = f"SPATIAL_MIXED_FG_{self.fg_method}_INT{self.fg_bits}_BG_{self.bg_method}_INT{self.bg_bits}"
        super().__init__(bits=self.fg_bits, block_size=block_size, name=name)
        self._fg_quantizer = self._METHOD_TO_CLASS[self.fg_method](bits=self.fg_bits, block_size=block_size)
        self._bg_quantizer = self._METHOD_TO_CLASS[self.bg_method](bits=self.bg_bits, block_size=block_size)

    def set_runtime_context(self, frame_seq_length: int | None = None) -> None:
        if frame_seq_length is not None and frame_seq_length > 0:
            self.frame_seq_length = int(frame_seq_length)

    def reset_stats(self) -> None:
        super().reset_stats()
        self._fg_ratio_history.clear()
        self._threshold_fg_ratio_history.clear()
        self._topk_fallback_history.clear()

    def _build_foreground_spatial_mask(self, variances: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        token_count = int(variances.numel())
        if token_count == 0:
            return torch.zeros_like(variances, dtype=torch.bool), {
                "final_fg_ratio": 0.0,
                "threshold_fg_ratio": 0.0,
                "used_topk_fallback": 0.0,
            }

        min_fg = max(1, int(round(self.min_foreground_ratio * token_count)))
        max_fg = int(round(self.max_foreground_ratio * token_count))
        max_fg = max(min_fg, min(token_count, max_fg if max_fg > 0 else token_count))
        target_fg = int(round(self.target_foreground_ratio * token_count))
        target_fg = max(min_fg, min(max_fg, target_fg if target_fg > 0 else min_fg))

        threshold_mask = variances > self.variance_threshold
        threshold_count = int(threshold_mask.sum().item())
        used_topk_fallback = 0.0

        if self.mask_policy == "topk":
            keep = target_fg
            used_topk_fallback = 1.0
        elif self.mask_policy == "threshold":
            if threshold_count < min_fg:
                keep = min_fg
                used_topk_fallback = 1.0
            elif threshold_count > max_fg:
                keep = max_fg
                used_topk_fallback = 1.0
            else:
                keep = threshold_count
        else:  # hybrid
            if threshold_count < min_fg or threshold_count > max_fg:
                keep = target_fg
                used_topk_fallback = 1.0
            else:
                keep = threshold_count

        if keep <= 0:
            fg_spatial = torch.zeros_like(threshold_mask, dtype=torch.bool)
        elif keep >= token_count:
            fg_spatial = torch.ones_like(threshold_mask, dtype=torch.bool)
        elif keep == threshold_count and used_topk_fallback == 0.0:
            fg_spatial = threshold_mask
        else:
            _, top_idx = torch.topk(variances, k=keep, largest=True)
            fg_spatial = torch.zeros_like(threshold_mask, dtype=torch.bool)
            fg_spatial[top_idx] = True

        return fg_spatial, {
            "final_fg_ratio": float(fg_spatial.float().mean().item()),
            "threshold_fg_ratio": float(threshold_count / token_count),
            "used_topk_fallback": used_topk_fallback,
        }

    def _build_foreground_mask(self, v: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        seq_len = int(v.shape[1])
        mask = torch.ones(seq_len, dtype=torch.bool, device=v.device)
        diagnostics: Dict[str, float] = {
            "mask_policy": 1.0 if self.mask_policy == "topk" else (2.0 if self.mask_policy == "hybrid" else 0.0),
            "final_fg_ratio": 1.0,
            "threshold_fg_ratio": 1.0,
            "used_topk_fallback": 0.0,
            "variance_mean": 0.0,
            "variance_std": 0.0,
        }
        if self.frame_seq_length is None or self.frame_seq_length <= 0:
            return mask, diagnostics

        spatial_tokens = int(self.frame_seq_length)
        if spatial_tokens <= 0:
            return mask, diagnostics
        num_frames = seq_len // spatial_tokens
        usable = num_frames * spatial_tokens
        if num_frames < 2 or usable == 0:
            return mask, diagnostics

        v_usable = v[:, :usable].reshape(v.shape[0], num_frames, spatial_tokens, v.shape[2], v.shape[3]).float()
        per_frame_token = v_usable.abs().mean(dim=(0, 3, 4))  # [T, S]
        variances = per_frame_token.var(dim=0, unbiased=False)  # [S]
        fg_spatial, mask_stats = self._build_foreground_spatial_mask(variances)
        diagnostics.update(mask_stats)
        diagnostics["variance_mean"] = float(variances.mean().item())
        diagnostics["variance_std"] = float(variances.std(unbiased=False).item())

        mask = torch.zeros(seq_len, dtype=torch.bool, device=v.device)
        mask[:usable] = fg_spatial.repeat(num_frames)
        if usable < seq_len:
            mask[usable:] = True
        diagnostics["final_fg_ratio"] = float(mask.float().mean().item())
        return mask, diagnostics

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        with timed() as t:
            fg_mask, diagnostics = self._build_foreground_mask(v)
            fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(-1)
            bg_idx = (~fg_mask).nonzero(as_tuple=False).squeeze(-1)
            state: Dict[str, Any] = {
                "fg_mask": fg_mask,
                "fg_idx": fg_idx,
                "bg_idx": bg_idx,
                "orig_shape": tuple(k.shape),
                "fg_state": None,
                "bg_state": None,
                "diagnostics": diagnostics,
            }
            if fg_idx.numel() > 0:
                state["fg_state"] = self._fg_quantizer.quantize_kv(
                    k.index_select(1, fg_idx), v.index_select(1, fg_idx), meta=meta
                )
            if bg_idx.numel() > 0:
                state["bg_state"] = self._bg_quantizer.quantize_kv(
                    k.index_select(1, bg_idx), v.index_select(1, bg_idx), meta=meta
                )
        self._fg_ratio_history.append(float(diagnostics.get("final_fg_ratio", 1.0)))
        self._threshold_fg_ratio_history.append(float(diagnostics.get("threshold_fg_ratio", 1.0)))
        self._topk_fallback_history.append(float(diagnostics.get("used_topk_fallback", 0.0)))
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            fg_mask: torch.Tensor = state["fg_mask"]
            b, l, h, d = state["orig_shape"]
            device = fg_mask.device
            tensor_dtype = meta.get("tensor_dtype") if meta else None
            target_dtype = tensor_dtype if tensor_dtype is not None else torch.float32
            k = torch.zeros((b, l, h, d), device=device, dtype=target_dtype)
            v = torch.zeros_like(k)

            fg_idx = state.get("fg_idx")
            if fg_idx is None:
                fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(-1)
            if state["fg_state"] is not None and fg_idx.numel() > 0:
                fg_k, fg_v = self._fg_quantizer.dequantize_kv(state["fg_state"], meta=meta)
                k.index_copy_(1, fg_idx, fg_k.to(device=device, dtype=target_dtype))
                v.index_copy_(1, fg_idx, fg_v.to(device=device, dtype=target_dtype))

            bg_idx = state.get("bg_idx")
            if bg_idx is None:
                bg_idx = (~fg_mask).nonzero(as_tuple=False).squeeze(-1)
            if state["bg_state"] is not None and bg_idx.numel() > 0:
                bg_k, bg_v = self._bg_quantizer.dequantize_kv(state["bg_state"], meta=meta)
                k.index_copy_(1, bg_idx, bg_k.to(device=device, dtype=target_dtype))
                v.index_copy_(1, bg_idx, bg_v.to(device=device, dtype=target_dtype))
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        total = int(state["fg_mask"].numel())
        if state.get("fg_idx") is not None:
            total += int(state["fg_idx"].numel() * 8)
        if state.get("bg_idx") is not None:
            total += int(state["bg_idx"].numel() * 8)
        if state.get("fg_state") is not None:
            total += int(self._fg_quantizer.memory_bytes(state["fg_state"]))
        if state.get("bg_state") is not None:
            total += int(self._bg_quantizer.memory_bytes(state["bg_state"]))
        return total

    def diagnostics(self) -> Dict[str, Any]:
        events = len(self._fg_ratio_history)
        if events == 0:
            return {}

        return {
            "spatial_mask_events": float(events),
            "spatial_avg_foreground_ratio": float(sum(self._fg_ratio_history) / events),
            "spatial_avg_threshold_foreground_ratio": float(sum(self._threshold_fg_ratio_history) / events),
            "spatial_topk_fallback_rate": float(sum(self._topk_fallback_history) / events),
            "spatial_target_foreground_ratio": float(self.target_foreground_ratio),
            "spatial_min_foreground_ratio": float(self.min_foreground_ratio),
            "spatial_max_foreground_ratio": float(self.max_foreground_ratio),
            "spatial_variance_threshold": float(self.variance_threshold),
            "spatial_mask_policy": self.mask_policy,
            "spatial_mask_policy_code": 1.0 if self.mask_policy == "topk" else (2.0 if self.mask_policy == "hybrid" else 0.0),
        }
