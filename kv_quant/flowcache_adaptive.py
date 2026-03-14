from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .flowcache_hybrid import FlowCacheHybridQuantizer
from .utils import timed


class FlowCacheAdaptiveQuantizer(FlowCacheHybridQuantizer):
    """
    Adaptive FlowCache variant:
    - recent chunks use INT4,
    - important old chunks use INT4,
    - remaining old chunks use low-bit INT2,
    with profiled layer budgets and chunk importance from relative-L1 cache deltas.
    """

    def __init__(
        self,
        bits: int = 2,
        block_size: int = 16,
        chunk_recent_ratio: float = 0.25,
        recent_method: str = "RTN",
        old_method: str = "RTN",
        recent_bits: int | None = None,
        layer_budget_table: Dict[int, float] | None = None,
        min_layer_budget_scale: float = 0.75,
        max_layer_budget_scale: float = 1.25,
        important_old_ratio: float = 0.20,
        importance_alpha: float = 0.7,
        importance_beta: float = 0.3,
    ) -> None:
        if not (0.0 <= important_old_ratio <= 1.0):
            raise ValueError("important_old_ratio must be in [0, 1].")
        if importance_alpha < 0 or importance_beta < 0 or (importance_alpha + importance_beta) <= 0:
            raise ValueError("importance_alpha and importance_beta must be non-negative and not both zero.")
        super().__init__(
            bits=bits,
            block_size=block_size,
            chunk_recent_ratio=chunk_recent_ratio,
            recent_method=recent_method,
            old_method=old_method,
            recent_bits=recent_bits,
            layer_budget_table=layer_budget_table,
            min_layer_budget_scale=min_layer_budget_scale,
            max_layer_budget_scale=max_layer_budget_scale,
        )
        self.important_old_ratio = float(important_old_ratio)
        self.importance_alpha = float(importance_alpha)
        self.importance_beta = float(importance_beta)
        self._important_quantizer = self._METHOD_TO_CLASS[self.recent_method](bits=self.recent_bits, block_size=block_size)
        self._prev_chunk_summaries: dict[int, torch.Tensor] = {}
        self._important_ratio_history: list[float] = []
        self._delta_history: list[float] = []
        self._name = f"FLOWCACHE_ADAPTIVE_INT{bits}"

    def reset_stats(self) -> None:
        super().reset_stats()
        self.reset_prompt_state()
        self._important_ratio_history.clear()
        self._delta_history.clear()

    def reset_prompt_state(self) -> None:
        self._prev_chunk_summaries.clear()

    def _compute_chunk_summaries(self, k: torch.Tensor, v: torch.Tensor, chunks: list[Tuple[int, int]]) -> torch.Tensor:
        if not chunks:
            return torch.empty((0, 1), dtype=torch.float32)
        summaries = []
        for start, end in chunks:
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            summary = (
                k_chunk.float().abs().mean(dim=(0, 1, 3)) + v_chunk.float().abs().mean(dim=(0, 1, 3))
            ).cpu()
            summaries.append(summary)
        return torch.stack(summaries, dim=0)

    def _compute_delta_scores(self, layer_id: int, summaries: torch.Tensor) -> torch.Tensor:
        num_chunks = int(summaries.shape[0])
        scores = torch.ones(num_chunks, dtype=torch.float32)
        prev = self._prev_chunk_summaries.get(int(layer_id))
        if prev is None or prev.ndim != 2 or prev.shape[1] != summaries.shape[1]:
            return scores
        overlap = min(int(prev.shape[0]), num_chunks)
        if overlap <= 0:
            return scores
        curr_tail = summaries[-overlap:]
        prev_tail = prev[-overlap:]
        denom = prev_tail.abs().mean(dim=1).clamp_min(1e-6)
        tail_scores = (curr_tail - prev_tail).abs().mean(dim=1) / denom
        scores[-overlap:] = tail_scores
        return scores

    def _build_tier_indices(
        self,
        seq_len: int,
        device: torch.device,
        layer_id: int,
        num_layers: int | None,
        delta_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        frame_seq_length = self.frame_seq_length or 1560
        chunks = self._compute_chunk_boundaries(seq_len, frame_seq_length)
        num_chunks = len(chunks)
        layer_budget = self._resolve_layer_budget(layer_id, num_layers)
        effective_recent_ratio = min(self.chunk_recent_ratio * layer_budget, 1.0)
        num_recent_chunks = max(1, int(round(effective_recent_ratio * num_chunks)))
        num_recent_chunks = min(num_recent_chunks, num_chunks)

        recent_chunk_ids = list(range(num_chunks - num_recent_chunks, num_chunks))
        old_chunk_ids = list(range(0, num_chunks - num_recent_chunks))

        important_old_count = 0
        important_old_ids: set[int] = set()
        recency_scores = torch.linspace(0.0, 1.0, steps=max(num_chunks, 1), dtype=torch.float32)
        combined_scores = self.importance_alpha * delta_scores + self.importance_beta * recency_scores[:num_chunks]
        if old_chunk_ids and self.important_old_ratio > 0:
            important_old_count = int(round(self.important_old_ratio * layer_budget * len(old_chunk_ids)))
            important_old_count = min(len(old_chunk_ids), max(1, important_old_count))
            old_scores = combined_scores[old_chunk_ids]
            topk_local = torch.topk(old_scores, k=important_old_count, largest=True, sorted=False).indices.tolist()
            important_old_ids = {old_chunk_ids[idx] for idx in topk_local}

        recent_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        important_mask = torch.zeros_like(recent_mask)
        old_mask = torch.zeros_like(recent_mask)
        for chunk_idx, (start, end) in enumerate(chunks):
            if chunk_idx in recent_chunk_ids:
                recent_mask[start:end] = True
            elif chunk_idx in important_old_ids:
                important_mask[start:end] = True
            else:
                old_mask[start:end] = True

        important_ratio = float(important_mask.float().mean().item()) if seq_len > 0 else 0.0
        avg_delta = float(delta_scores.mean().item()) if delta_scores.numel() > 0 else 0.0
        return (
            recent_mask.nonzero(as_tuple=False).squeeze(-1),
            important_mask.nonzero(as_tuple=False).squeeze(-1),
            old_mask.nonzero(as_tuple=False).squeeze(-1),
            float(recent_mask.float().mean().item()) if seq_len > 0 else 0.0,
            important_ratio,
            float(layer_budget),
            avg_delta,
        )

    def quantize_kv(self, k, v, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bf16_bytes = int(k.numel() * 2 + v.numel() * 2)
        layer_id = int(meta.get("layer_id", 0)) if meta else 0
        frame_seq_length = meta.get("frame_seq_length") if meta else None
        num_layers = meta.get("num_layers") if meta else None
        if frame_seq_length is not None:
            self.frame_seq_length = int(frame_seq_length)
        if num_layers is not None:
            self.num_layers = int(num_layers)

        with timed() as t:
            full_seq_len = int(k.shape[1])
            active_seq_len = self._active_seq_len(full_seq_len, meta)
            active_k = k[:, :active_seq_len]
            active_v = v[:, :active_seq_len]
            chunks = self._compute_chunk_boundaries(active_seq_len, int(self.frame_seq_length or 1560))
            summaries = self._compute_chunk_summaries(active_k, active_v, chunks)
            delta_scores = self._compute_delta_scores(layer_id, summaries)
            recent_idx, important_idx, old_idx, recent_ratio, important_ratio, layer_budget, avg_delta = self._build_tier_indices(
                active_seq_len, k.device, layer_id, self.num_layers, delta_scores
            )
            state: Dict[str, Any] = {
                "recent_idx": recent_idx,
                "important_idx": important_idx,
                "old_idx": old_idx,
                "orig_shape": tuple(k.shape),
                "recent_state": None,
                "important_state": None,
                "old_state": None,
                "metadata": {
                    "layer_id": int(layer_id),
                    "num_layers": int(self.num_layers or 0),
                    "frame_seq_length": int(self.frame_seq_length or 1560),
                    "active_seq_len": int(active_seq_len),
                    "recent_ratio": float(recent_ratio),
                    "important_ratio": float(important_ratio),
                    "layer_budget": float(layer_budget),
                    "avg_delta": float(avg_delta),
                },
            }
            if recent_idx.numel() > 0:
                state["recent_state"] = self._recent_quantizer.quantize_kv(
                    k.index_select(1, recent_idx), v.index_select(1, recent_idx), meta=meta
                )
            if important_idx.numel() > 0:
                state["important_state"] = self._important_quantizer.quantize_kv(
                    k.index_select(1, important_idx), v.index_select(1, important_idx), meta=meta
                )
            if old_idx.numel() > 0:
                state["old_state"] = self._old_quantizer.quantize_kv(
                    k.index_select(1, old_idx), v.index_select(1, old_idx), meta=meta
                )
            self._prev_chunk_summaries[layer_id] = summaries

        self._chunk_ratio_history.append(float(recent_ratio))
        self._important_ratio_history.append(float(important_ratio))
        self._layer_budget_history.append(float(layer_budget))
        self._delta_history.append(float(avg_delta))
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            b, l, h, d = state["orig_shape"]
            tensor_dtype = meta.get("tensor_dtype") if meta else None
            target_dtype = tensor_dtype if tensor_dtype is not None else torch.float32
            candidate_idx = next(
                (
                    idx
                    for idx in (state.get("recent_idx"), state.get("important_idx"), state.get("old_idx"))
                    if idx is not None
                ),
                None,
            )
            device = candidate_idx.device if candidate_idx is not None else torch.device("cpu")
            k = torch.zeros((b, l, h, d), device=device, dtype=target_dtype)
            v = torch.zeros_like(k)

            recent_idx = state.get("recent_idx")
            important_idx = state.get("important_idx")
            old_idx = state.get("old_idx")
            if state["recent_state"] is not None and recent_idx is not None and recent_idx.numel() > 0:
                rk, rv = self._recent_quantizer.dequantize_kv(state["recent_state"], meta=meta)
                k.index_copy_(1, recent_idx, rk.to(device=device, dtype=target_dtype))
                v.index_copy_(1, recent_idx, rv.to(device=device, dtype=target_dtype))
            if state["important_state"] is not None and important_idx is not None and important_idx.numel() > 0:
                ik, iv = self._important_quantizer.dequantize_kv(state["important_state"], meta=meta)
                k.index_copy_(1, important_idx, ik.to(device=device, dtype=target_dtype))
                v.index_copy_(1, important_idx, iv.to(device=device, dtype=target_dtype))
            if state["old_state"] is not None and old_idx is not None and old_idx.numel() > 0:
                ok, ov = self._old_quantizer.dequantize_kv(state["old_state"], meta=meta)
                k.index_copy_(1, old_idx, ok.to(device=device, dtype=target_dtype))
                v.index_copy_(1, old_idx, ov.to(device=device, dtype=target_dtype))
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        total = 0
        for key in ("recent_idx", "important_idx", "old_idx"):
            idx = state.get(key)
            if idx is not None:
                total += int(idx.numel() * 8)
        if state.get("recent_state") is not None:
            total += int(self._recent_quantizer.memory_bytes(state["recent_state"]))
        if state.get("important_state") is not None:
            total += int(self._important_quantizer.memory_bytes(state["important_state"]))
        if state.get("old_state") is not None:
            total += int(self._old_quantizer.memory_bytes(state["old_state"]))
        return total

    def diagnostics(self) -> Dict[str, float]:
        events = len(self._chunk_ratio_history)
        if events == 0:
            return {}
        return {
            "flowcache_adaptive_events": float(events),
            "flowcache_adaptive_avg_recent_ratio": float(sum(self._chunk_ratio_history) / events),
            "flowcache_adaptive_avg_important_ratio": float(sum(self._important_ratio_history) / events),
            "flowcache_adaptive_avg_layer_budget": float(sum(self._layer_budget_history) / events),
            "flowcache_adaptive_avg_delta": float(sum(self._delta_history) / events),
            "flowcache_adaptive_config_chunk_recent_ratio": float(self.chunk_recent_ratio),
            "flowcache_adaptive_config_important_old_ratio": float(self.important_old_ratio),
            "flowcache_adaptive_recent_bits": float(self.recent_bits),
            "flowcache_adaptive_old_bits": float(self.bits),
            "flowcache_adaptive_importance_alpha": float(self.importance_alpha),
            "flowcache_adaptive_importance_beta": float(self.importance_beta),
        }
