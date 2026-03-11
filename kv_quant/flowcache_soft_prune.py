from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .flowcache_prune import FlowCachePruneQuantizer
from .utils import timed


class FlowCacheSoftPruneQuantizer(FlowCachePruneQuantizer):
    """
    Soft-eviction variant of FlowCache prune:
    - recent chunks use INT4,
    - important old chunks use INT4,
    - retained old chunks use low-bit quantization,
    - evicted chunks fall back to a pooled BF16 summary token repeated across the chunk.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = f"FLOWCACHE_SOFT_PRUNE_INT{self.bits}"

    @staticmethod
    def _pooled_chunk_summaries(k: torch.Tensor, v: torch.Tensor, spans: list[Tuple[int, int]]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not spans:
            return None, None
        k_summaries = []
        v_summaries = []
        for start, end in spans:
            chunk_k = k[:, start:end]
            chunk_v = v[:, start:end]
            k_summaries.append(chunk_k.float().mean(dim=1).to(dtype=k.dtype))
            v_summaries.append(chunk_v.float().mean(dim=1).to(dtype=v.dtype))
        return torch.stack(k_summaries, dim=0), torch.stack(v_summaries, dim=0)

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
            pruned_k_summary, pruned_v_summary = self._pooled_chunk_summaries(active_k, active_v, pruned_spans)
            state: Dict[str, Any] = {
                "recent_idx": recent_idx,
                "important_idx": important_idx,
                "old_idx": old_idx,
                "pruned_spans": [(int(start), int(end)) for start, end in pruned_spans],
                "pruned_k_summary": pruned_k_summary,
                "pruned_v_summary": pruned_v_summary,
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

            pruned_spans = state.get("pruned_spans") or []
            pruned_k_summary = state.get("pruned_k_summary")
            pruned_v_summary = state.get("pruned_v_summary")
            if pruned_spans and pruned_k_summary is not None and pruned_v_summary is not None:
                for idx, (start, end) in enumerate(pruned_spans):
                    length = max(0, int(end) - int(start))
                    if length <= 0:
                        continue
                    chunk_k = pruned_k_summary[idx].to(device=device, dtype=target_dtype).unsqueeze(1).expand(-1, length, -1, -1)
                    chunk_v = pruned_v_summary[idx].to(device=device, dtype=target_dtype).unsqueeze(1).expand(-1, length, -1, -1)
                    k[:, start:end] = chunk_k
                    v[:, start:end] = chunk_v
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        total = super().memory_bytes(state)
        if state.get("pruned_k_summary") is not None:
            total += int(state["pruned_k_summary"].numel() * state["pruned_k_summary"].element_size())
        if state.get("pruned_v_summary") is not None:
            total += int(state["pruned_v_summary"].numel() * state["pruned_v_summary"].element_size())
        if state.get("pruned_spans"):
            total += int(len(state["pruned_spans"]) * 16)
        return total

    def diagnostics(self) -> Dict[str, float]:
        events = len(self._chunk_ratio_history)
        if events == 0:
            return {}
        refresh_rate = float(sum(self._refresh_history) / events) if self._refresh_history else 0.0
        return {
            "flowcache_soft_prune_events": float(events),
            "flowcache_soft_prune_avg_recent_ratio": float(sum(self._chunk_ratio_history) / events),
            "flowcache_soft_prune_avg_important_ratio": float(sum(self._important_ratio_history) / events),
            "flowcache_soft_prune_avg_retained_ratio": float(sum(self._retained_ratio_history) / events),
            "flowcache_soft_prune_avg_pruned_ratio": float(sum(self._pruned_ratio_history) / events),
            "flowcache_soft_prune_avg_layer_budget": float(sum(self._layer_budget_history) / events),
            "flowcache_soft_prune_avg_delta": float(sum(self._delta_history) / events),
            "flowcache_soft_prune_refresh_rate": refresh_rate,
            "flowcache_soft_prune_config_chunk_recent_ratio": float(self.chunk_recent_ratio),
            "flowcache_soft_prune_config_important_old_ratio": float(self.important_old_ratio),
            "flowcache_soft_prune_config_retained_old_ratio": float(self.retained_old_ratio),
            "flowcache_soft_prune_refresh_gap_chunks": float(self.refresh_gap_chunks),
            "flowcache_soft_prune_recent_bits": float(self.recent_bits),
            "flowcache_soft_prune_old_bits": float(self.bits),
            "flowcache_soft_prune_importance_alpha": float(self.importance_alpha),
            "flowcache_soft_prune_importance_beta": float(self.importance_beta),
            "flowcache_soft_prune_summary_tokens_per_chunk": 1.0,
        }
