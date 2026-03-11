"""
FlowCache Hybrid: Temporal + Layer-aware KV-cache Quantization

Combines:
- FlowCache inspiration: Chunk-wise temporal heterogeneity (recent chunks → higher precision)
- ScaleKV inspiration: Layer-role-aware budgets (early vs. late decoder layers)

For Self-Forcing long autoregressive video generation with frame_seq_length=1560.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .kivi import KIVIQuantizer
from .quarot_kv import QuaRotKVQuantizer
from .rtn import RTNQuantizer
from .utils import timed


class FlowCacheHybridQuantizer(KVQuantizer):
    """
    Frame-aware, layer-role-aware KV-cache quantizer.
    
    Uses AgeTier-style masking but with temporal chunk awareness
    and per-layer budget allocation.
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
        chunk_recent_ratio: float = 0.5,
        recent_method: str = "RTN",
        old_method: str = "RTN",
        recent_bits: int | None = None,
        layer_budget_table: Dict[int, float] | None = None,
        min_layer_budget_scale: float = 0.75,
        max_layer_budget_scale: float = 1.25,
    ) -> None:
        """
        Args:
            bits: Low-bit precision for old chunks (2 or 4)
            block_size: Quantization block size
            chunk_recent_ratio: Fraction of recent chunks to keep at high precision (0-1)
            recent_method: Quantizer for recent chunks ("RTN", "KIVI", "QUAROT_KV")
            old_method: Quantizer for old chunks
            recent_bits: Precision for recent chunks (default: 4)
            layer_budget_table: Optional explicit Dict[layer_id] -> budget_scale.
            min_layer_budget_scale: Default minimum scale for early layers.
            max_layer_budget_scale: Default maximum scale for late layers.
        """
        recent_method = recent_method.upper()
        old_method = old_method.upper()
        
        if recent_method not in self._METHOD_TO_CLASS:
            raise ValueError(f"Unsupported recent_method={recent_method}")
        if old_method not in self._METHOD_TO_CLASS:
            raise ValueError(f"Unsupported old_method={old_method}")
        if not (0.0 < chunk_recent_ratio <= 1.0):
            raise ValueError("chunk_recent_ratio must be in (0, 1]")
        if recent_bits is not None and recent_bits not in (2, 4):
            raise ValueError("recent_bits must be 2 or 4")
        if min_layer_budget_scale <= 0 or max_layer_budget_scale <= 0:
            raise ValueError("layer budget scales must be > 0")

        self.chunk_recent_ratio = float(chunk_recent_ratio)
        self.recent_bits = int(recent_bits) if recent_bits is not None else 4
        self.recent_method = recent_method
        self.old_method = old_method
        self.layer_budget_table = layer_budget_table or {}
        self.min_layer_budget_scale = float(min_layer_budget_scale)
        self.max_layer_budget_scale = float(max_layer_budget_scale)
        self.frame_seq_length: int | None = None
        self.num_layers: int | None = None
        
        # Tracking for diagnostics
        self._chunk_ratio_history: list[float] = []
        self._layer_budget_history: list[float] = []

        name = f"FLOWCACHE_HYBRID_INT{bits}"
        super().__init__(bits=bits, block_size=block_size, name=name)
        
        self._recent_quantizer = self._METHOD_TO_CLASS[self.recent_method](
            bits=self.recent_bits, block_size=block_size
        )
        self._old_quantizer = self._METHOD_TO_CLASS[self.old_method](
            bits=self.bits, block_size=block_size
        )

    def set_runtime_context(self, frame_seq_length: int | None = None, num_layers: int | None = None) -> None:
        """Set frame sequence length for chunk boundary detection."""
        if frame_seq_length is not None and frame_seq_length > 0:
            self.frame_seq_length = int(frame_seq_length)
        if num_layers is not None and num_layers > 0:
            self.num_layers = int(num_layers)

    def _active_seq_len(self, full_seq_len: int, meta: Dict[str, Any] | None = None) -> int:
        if meta is not None:
            for key in ("local_end_index", "current_end", "global_end_index"):
                raw = meta.get(key)
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return max(1, min(full_seq_len, value))
        return int(full_seq_len)

    def reset_stats(self) -> None:
        super().reset_stats()
        self._chunk_ratio_history.clear()
        self._layer_budget_history.clear()

    def _compute_chunk_boundaries(
        self, seq_len: int, frame_seq_length: int
    ) -> list[Tuple[int, int]]:
        """
        Map token sequence into frame-aligned chunks.
        
        Returns list of (start_idx, end_idx) tuples.
        """
        if frame_seq_length <= 0:
            # Fallback: treat entire sequence as one chunk
            return [(0, seq_len)]
        
        chunks = []
        for chunk_id in range((seq_len + frame_seq_length - 1) // frame_seq_length):
            start = chunk_id * frame_seq_length
            end = min((chunk_id + 1) * frame_seq_length, seq_len)
            chunks.append((start, end))
        return chunks

    def _resolve_layer_budget(self, layer_id: int, num_layers: int | None) -> float:
        if layer_id in self.layer_budget_table:
            return float(self.layer_budget_table[layer_id])
        resolved_num_layers = int(num_layers or self.num_layers or 0)
        if resolved_num_layers <= 1:
            return 1.0
        progress = float(max(0, min(layer_id, resolved_num_layers - 1))) / float(resolved_num_layers - 1)
        return self.min_layer_budget_scale + progress * (self.max_layer_budget_scale - self.min_layer_budget_scale)

    def _build_chunk_aware_mask(
        self, seq_len: int, device: torch.device, layer_id: int = 0, num_layers: int | None = None
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Build recency mask based on frame chunks and layer-aware budget.
        
        Recent chunks → True (high precision)
        Old chunks → False (low precision)
        """
        # Compute layer-aware budget scaling
        layer_scale = self._resolve_layer_budget(layer_id, num_layers)
        if seq_len <= 0:
            return torch.zeros(seq_len, dtype=torch.bool, device=device), 0.0, float(layer_scale)

        frame_seq_length = self.frame_seq_length or 1560
        chunks = self._compute_chunk_boundaries(seq_len, frame_seq_length)
        effective_recent_ratio = min(self.chunk_recent_ratio * layer_scale, 1.0)

        # Determine how many recent chunks to keep at high precision
        num_chunks = len(chunks)
        if num_chunks == 0:
            return torch.zeros(seq_len, dtype=torch.bool, device=device), 0.0, float(layer_scale)
        num_recent_chunks = max(1, int(round(effective_recent_ratio * num_chunks)))
        num_recent_chunks = min(num_recent_chunks, num_chunks)

        # Mark recent chunks as True
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for chunk_idx in range(num_chunks - num_recent_chunks, num_chunks):
            start, end = chunks[chunk_idx]
            mask[start:end] = True

        return mask, float(mask.float().mean().item()), float(layer_scale)

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Quantize k, v with chunk-aware and layer-aware budget allocation.
        
        Args:
            k: Keys tensor [B, seq_len, num_heads, head_dim]
            v: Values tensor [B, seq_len, num_heads, head_dim]
            meta: Optional metadata dict with keys:
                  - "layer_id": int (0-31 for 32-layer model)
                  - "frame_seq_length": int (default 1560)
                  - "tensor_dtype": torch.dtype
        
        Returns:
            Quantized state dict
        """
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        
        # Extract context from meta
        layer_id = meta.get("layer_id", 0) if meta else 0
        frame_seq_length = meta.get("frame_seq_length", None) if meta else None
        num_layers = meta.get("num_layers", None) if meta else None
        if frame_seq_length is not None:
            self.frame_seq_length = int(frame_seq_length)
        if num_layers is not None:
            self.num_layers = int(num_layers)
        
        with timed() as t:
            full_seq_len = int(k.shape[1])
            active_seq_len = self._active_seq_len(full_seq_len, meta)

            # Build chunk-aware, layer-aware mask
            recent_mask_active, chunk_ratio, layer_budget = self._build_chunk_aware_mask(
                active_seq_len, k.device, int(layer_id), self.num_layers
            )
            recent_mask = torch.zeros(full_seq_len, dtype=torch.bool, device=k.device)
            recent_mask[:active_seq_len] = recent_mask_active
            recent_idx = recent_mask.nonzero(as_tuple=False).squeeze(-1)
            active_mask = torch.zeros(full_seq_len, dtype=torch.bool, device=k.device)
            active_mask[:active_seq_len] = True
            old_idx = (active_mask & (~recent_mask)).nonzero(as_tuple=False).squeeze(-1)
            
            state: Dict[str, Any] = {
                "recent_mask": recent_mask,
                "recent_idx": recent_idx,
                "old_idx": old_idx,
                "orig_shape": tuple(k.shape),
                "recent_state": None,
                "old_state": None,
                "metadata": {
                    "layer_id": int(layer_id),
                    "num_layers": int(self.num_layers or 0),
                    "frame_seq_length": self.frame_seq_length or 1560,
                    "active_seq_len": int(active_seq_len),
                    "chunk_ratio": float(chunk_ratio),
                    "layer_budget": float(layer_budget),
                },
            }
            
            # Quantize recent chunks with high precision
            if recent_idx.numel() > 0:
                state["recent_state"] = self._recent_quantizer.quantize_kv(
                    k.index_select(1, recent_idx), v.index_select(1, recent_idx), meta=meta
                )
            
            # Quantize old chunks with low precision
            if old_idx.numel() > 0:
                state["old_state"] = self._old_quantizer.quantize_kv(
                    k.index_select(1, old_idx), v.index_select(1, old_idx), meta=meta
                )
        
        self._chunk_ratio_history.append(float(chunk_ratio))
        self._layer_budget_history.append(float(layer_budget))
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        """Reconstruct k, v from quantized state."""
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

            # Reconstruct recent chunks
            if state["recent_state"] is not None and recent_idx.numel() > 0:
                rk, rv = self._recent_quantizer.dequantize_kv(state["recent_state"], meta=meta)
                k.index_copy_(1, recent_idx, rk.to(device=device, dtype=target_dtype))
                v.index_copy_(1, recent_idx, rv.to(device=device, dtype=target_dtype))

            # Reconstruct old chunks
            if state["old_state"] is not None and old_idx.numel() > 0:
                ok, ov = self._old_quantizer.dequantize_kv(state["old_state"], meta=meta)
                k.index_copy_(1, old_idx, ok.to(device=device, dtype=target_dtype))
                v.index_copy_(1, old_idx, ov.to(device=device, dtype=target_dtype))
        
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        """Compute total compressed size in bytes."""
        total = int(state["recent_mask"].numel())
        if state.get("recent_idx") is not None:
            total += int(state["recent_idx"].numel() * 8)
        if state.get("old_idx") is not None:
            total += int(state["old_idx"].numel() * 8)
        if state.get("recent_state") is not None:
            total += int(self._recent_quantizer.memory_bytes(state["recent_state"]))
        if state.get("old_state") is not None:
            total += int(self._old_quantizer.memory_bytes(state["old_state"]))
        return total

    def diagnostics(self) -> Dict[str, float]:
        """Return quantizer diagnostics."""
        events = len(self._chunk_ratio_history)
        if events == 0:
            return {}
        
        return {
            "flowcache_events": float(events),
            "flowcache_avg_chunk_ratio": float(sum(self._chunk_ratio_history) / events),
            "flowcache_avg_layer_budget": float(sum(self._layer_budget_history) / events),
            "flowcache_config_chunk_recent_ratio": float(self.chunk_recent_ratio),
            "flowcache_recent_bits": float(self.recent_bits),
            "flowcache_old_bits": float(self.bits),
            "flowcache_min_layer_budget_scale": float(self.min_layer_budget_scale),
            "flowcache_max_layer_budget_scale": float(self.max_layer_budget_scale),
        }
