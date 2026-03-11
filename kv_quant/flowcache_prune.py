from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .flowcache_adaptive import FlowCacheAdaptiveQuantizer
from .utils import timed


class FlowCachePruneQuantizer(FlowCacheAdaptiveQuantizer):
    """
    Profiled prune/evict variant:
    - recent chunks use INT4,
    - a small important-old subset uses INT4,
    - a limited retained-old subset uses low-bit INT2,
    - the least important old chunks are evicted and reconstructed as zeros.

    Importance plans are refreshed only when the number of frame-aligned chunks grows,
    which reduces scoring overhead relative to fully step-adaptive selection.
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
        important_old_ratio: float = 0.10,
        retained_old_ratio: float = 0.30,
        importance_alpha: float = 0.7,
        importance_beta: float = 0.3,
        refresh_gap_chunks: int = 1,
    ) -> None:
        if not (0.0 <= retained_old_ratio <= 1.0):
            raise ValueError("retained_old_ratio must be in [0, 1].")
        if refresh_gap_chunks <= 0:
            raise ValueError("refresh_gap_chunks must be >= 1.")
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
            important_old_ratio=important_old_ratio,
            importance_alpha=importance_alpha,
            importance_beta=importance_beta,
        )
        self.retained_old_ratio = float(retained_old_ratio)
        self.refresh_gap_chunks = int(refresh_gap_chunks)
        self._cached_chunk_plans: dict[int, Dict[str, Any]] = {}
        self._retained_ratio_history: list[float] = []
        self._pruned_ratio_history: list[float] = []
        self._refresh_history: list[float] = []
        self._name = f"FLOWCACHE_PRUNE_INT{bits}"

    def reset_stats(self) -> None:
        super().reset_stats()
        self._retained_ratio_history.clear()
        self._pruned_ratio_history.clear()
        self._refresh_history.clear()

    def reset_prompt_state(self) -> None:
        super().reset_prompt_state()
        self._cached_chunk_plans.clear()

    def _select_chunk_plan(
        self,
        num_chunks: int,
        layer_id: int,
        num_layers: int | None,
        delta_scores: torch.Tensor,
    ) -> Dict[str, Any]:
        layer_budget = self._resolve_layer_budget(layer_id, num_layers)
        effective_recent_ratio = min(self.chunk_recent_ratio * layer_budget, 1.0)
        num_recent_chunks = max(1, int(round(effective_recent_ratio * num_chunks)))
        num_recent_chunks = min(num_recent_chunks, num_chunks)

        recent_chunk_ids = list(range(num_chunks - num_recent_chunks, num_chunks))
        old_chunk_ids = list(range(0, num_chunks - num_recent_chunks))
        recency_scores = torch.linspace(0.0, 1.0, steps=max(num_chunks, 1), dtype=torch.float32)
        combined_scores = self.importance_alpha * delta_scores + self.importance_beta * recency_scores[:num_chunks]

        important_old_ids: set[int] = set()
        retained_old_ids: set[int] = set()
        important_old_count = 0
        retained_old_count = 0

        if old_chunk_ids and self.important_old_ratio > 0.0:
            important_old_count = int(round(self.important_old_ratio * layer_budget * len(old_chunk_ids)))
            important_old_count = min(len(old_chunk_ids), max(1, important_old_count))
            old_scores = combined_scores[old_chunk_ids]
            topk_local = torch.topk(old_scores, k=important_old_count, largest=True, sorted=False).indices.tolist()
            important_old_ids = {old_chunk_ids[idx] for idx in topk_local}

        remaining_old_ids = [idx for idx in old_chunk_ids if idx not in important_old_ids]
        if remaining_old_ids and self.retained_old_ratio > 0.0:
            retained_old_count = int(round(self.retained_old_ratio * layer_budget * len(remaining_old_ids)))
            retained_old_count = min(len(remaining_old_ids), max(1, retained_old_count))
            remaining_scores = combined_scores[remaining_old_ids]
            topk_local = torch.topk(remaining_scores, k=retained_old_count, largest=True, sorted=False).indices.tolist()
            retained_old_ids = {remaining_old_ids[idx] for idx in topk_local}

        important_ratio = float(important_old_count / max(num_chunks, 1))
        retained_ratio = float(retained_old_count / max(num_chunks, 1))
        pruned_ratio = float(max(len(remaining_old_ids) - retained_old_count, 0) / max(num_chunks, 1))
        avg_delta = float(delta_scores.mean().item()) if delta_scores.numel() > 0 else 0.0
        return {
            "num_chunks": int(num_chunks),
            "recent_chunk_ids": tuple(recent_chunk_ids),
            "important_old_ids": tuple(sorted(important_old_ids)),
            "retained_old_ids": tuple(sorted(retained_old_ids)),
            "layer_budget": float(layer_budget),
            "recent_ratio": float(num_recent_chunks / max(num_chunks, 1)),
            "important_ratio": important_ratio,
            "retained_ratio": retained_ratio,
            "pruned_ratio": pruned_ratio,
            "avg_delta": avg_delta,
        }

    def _materialize_chunk_spans(
        self,
        seq_len: int,
        plan: Dict[str, Any],
    ) -> tuple[list[Tuple[int, int]], list[Tuple[int, int]], list[Tuple[int, int]], list[Tuple[int, int]]]:
        chunks = self._compute_chunk_boundaries(seq_len, int(self.frame_seq_length or 1560))
        recent_chunk_ids = set(plan["recent_chunk_ids"])
        important_old_ids = set(plan["important_old_ids"])
        retained_old_ids = set(plan["retained_old_ids"])

        recent_spans: list[Tuple[int, int]] = []
        important_spans: list[Tuple[int, int]] = []
        retained_spans: list[Tuple[int, int]] = []
        pruned_spans: list[Tuple[int, int]] = []
        for chunk_idx, span in enumerate(chunks):
            if chunk_idx in recent_chunk_ids:
                recent_spans.append(span)
            elif chunk_idx in important_old_ids:
                important_spans.append(span)
            elif chunk_idx in retained_old_ids:
                retained_spans.append(span)
            else:
                pruned_spans.append(span)
        return recent_spans, important_spans, retained_spans, pruned_spans

    @staticmethod
    def _spans_to_indices(spans: list[Tuple[int, int]], device: torch.device) -> torch.Tensor:
        parts = [torch.arange(start, end, device=device, dtype=torch.long) for start, end in spans if end > start]
        if not parts:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(parts, dim=0)

    def _materialize_indices(
        self,
        seq_len: int,
        device: torch.device,
        plan: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, list[Tuple[int, int]]]:
        recent_spans, important_spans, retained_spans, pruned_spans = self._materialize_chunk_spans(seq_len, plan)
        recent_idx = self._spans_to_indices(recent_spans, device)
        important_idx = self._spans_to_indices(important_spans, device)
        retained_idx = self._spans_to_indices(retained_spans, device)
        recent_ratio = float(recent_idx.numel() / max(seq_len, 1))
        important_ratio = float(important_idx.numel() / max(seq_len, 1))
        retained_ratio = float(retained_idx.numel() / max(seq_len, 1))
        return (
            recent_idx,
            important_idx,
            retained_idx,
            recent_ratio,
            important_ratio,
            retained_ratio,
            pruned_spans,
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
            num_chunks = len(chunks)
            cached_plan = self._cached_chunk_plans.get(layer_id)
            refresh_plan = (
                cached_plan is None
                or abs(int(cached_plan["num_chunks"]) - num_chunks) >= self.refresh_gap_chunks
                or int(cached_plan["num_chunks"]) != num_chunks
            )
            if refresh_plan:
                summaries = self._compute_chunk_summaries(active_k, active_v, chunks)
                delta_scores = self._compute_delta_scores(layer_id, summaries)
                cached_plan = self._select_chunk_plan(num_chunks, layer_id, self.num_layers, delta_scores)
                self._cached_chunk_plans[layer_id] = cached_plan
                self._prev_chunk_summaries[layer_id] = summaries

            recent_idx, important_idx, old_idx, recent_ratio, important_ratio, retained_ratio, pruned_spans = self._materialize_indices(
                active_seq_len, k.device, cached_plan
            )
            pruned_ratio = float(sum(max(0, end - start) for start, end in pruned_spans) / max(active_seq_len, 1))
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
                    "retained_ratio": float(retained_ratio),
                    "pruned_ratio": float(pruned_ratio),
                    "layer_budget": float(cached_plan["layer_budget"]),
                    "avg_delta": float(cached_plan["avg_delta"]),
                    "plan_refreshed": bool(refresh_plan),
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

        self._chunk_ratio_history.append(float(recent_ratio))
        self._important_ratio_history.append(float(important_ratio))
        self._retained_ratio_history.append(float(retained_ratio))
        self._pruned_ratio_history.append(float(pruned_ratio))
        self._layer_budget_history.append(float(cached_plan["layer_budget"]))
        self._delta_history.append(float(cached_plan["avg_delta"]))
        self._refresh_history.append(1.0 if refresh_plan else 0.0)
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

    def diagnostics(self) -> Dict[str, float]:
        events = len(self._chunk_ratio_history)
        if events == 0:
            return {}
        refresh_rate = float(sum(self._refresh_history) / events) if self._refresh_history else 0.0
        return {
            "flowcache_prune_events": float(events),
            "flowcache_prune_avg_recent_ratio": float(sum(self._chunk_ratio_history) / events),
            "flowcache_prune_avg_important_ratio": float(sum(self._important_ratio_history) / events),
            "flowcache_prune_avg_retained_ratio": float(sum(self._retained_ratio_history) / events),
            "flowcache_prune_avg_pruned_ratio": float(sum(self._pruned_ratio_history) / events),
            "flowcache_prune_avg_layer_budget": float(sum(self._layer_budget_history) / events),
            "flowcache_prune_avg_delta": float(sum(self._delta_history) / events),
            "flowcache_prune_refresh_rate": refresh_rate,
            "flowcache_prune_config_chunk_recent_ratio": float(self.chunk_recent_ratio),
            "flowcache_prune_config_important_old_ratio": float(self.important_old_ratio),
            "flowcache_prune_config_retained_old_ratio": float(self.retained_old_ratio),
            "flowcache_prune_refresh_gap_chunks": float(self.refresh_gap_chunks),
            "flowcache_prune_recent_bits": float(self.recent_bits),
            "flowcache_prune_old_bits": float(self.bits),
            "flowcache_prune_importance_alpha": float(self.importance_alpha),
            "flowcache_prune_importance_beta": float(self.importance_beta),
        }
