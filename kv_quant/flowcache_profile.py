from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .base import KVQuantizer
from .utils import timed


class FlowCacheProfileQuantizer(KVQuantizer):
    """
    BF16-preserving cache profiler for chunkwise and layerwise KV dynamics.

    This quantizer does not compress the cache. Instead, it stores BF16 KV state
    exactly while logging per-layer chunk delta statistics that can be converted
    into a non-monotonic layer budget table for FlowCacheAdaptive.
    """

    def __init__(self, block_size: int = 16, profile_recent_ratio: float = 0.25) -> None:
        if not (0.0 < profile_recent_ratio <= 1.0):
            raise ValueError("profile_recent_ratio must be in (0, 1].")
        super().__init__(bits=4, block_size=block_size, name="FLOWCACHE_PROFILE")
        self.profile_recent_ratio = float(profile_recent_ratio)
        self.frame_seq_length: int | None = None
        self.num_layers: int | None = None
        self._prev_chunk_summaries: dict[int, torch.Tensor] = {}
        self._layer_old_delta_sum: dict[int, float] = {}
        self._layer_old_delta_count: dict[int, int] = {}
        self._layer_all_delta_sum: dict[int, float] = {}
        self._layer_all_delta_count: dict[int, int] = {}

    def set_runtime_context(self, frame_seq_length: int | None = None, num_layers: int | None = None) -> None:
        if frame_seq_length is not None and frame_seq_length > 0:
            self.frame_seq_length = int(frame_seq_length)
        if num_layers is not None and num_layers > 0:
            self.num_layers = int(num_layers)

    def reset_stats(self) -> None:
        super().reset_stats()
        self.reset_prompt_state()
        self._layer_old_delta_sum.clear()
        self._layer_old_delta_count.clear()
        self._layer_all_delta_sum.clear()
        self._layer_all_delta_count.clear()

    def reset_prompt_state(self) -> None:
        self._prev_chunk_summaries.clear()

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

    def _compute_chunk_boundaries(self, seq_len: int) -> list[Tuple[int, int]]:
        frame_seq_length = int(self.frame_seq_length or 1560)
        if frame_seq_length <= 0:
            return [(0, seq_len)]
        chunks: list[Tuple[int, int]] = []
        for chunk_id in range((seq_len + frame_seq_length - 1) // frame_seq_length):
            start = chunk_id * frame_seq_length
            end = min((chunk_id + 1) * frame_seq_length, seq_len)
            chunks.append((start, end))
        return chunks

    def _compute_chunk_summaries(self, k: torch.Tensor, v: torch.Tensor, chunks: list[Tuple[int, int]]) -> torch.Tensor:
        if not chunks:
            return torch.empty((0, 1), dtype=torch.float32)
        summaries = []
        for start, end in chunks:
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            # Head-level activation sketch: tiny but stable enough for relative-L1 tracking.
            summary = (
                k_chunk.float().abs().mean(dim=(0, 1, 3)) + v_chunk.float().abs().mean(dim=(0, 1, 3))
            ).cpu()
            summaries.append(summary)
        return torch.stack(summaries, dim=0)

    def _compute_chunk_deltas(self, layer_id: int, summaries: torch.Tensor) -> torch.Tensor:
        num_chunks = int(summaries.shape[0])
        deltas = torch.ones(num_chunks, dtype=torch.float32)
        prev = self._prev_chunk_summaries.get(int(layer_id))
        if prev is None or prev.ndim != 2 or prev.shape[1] != summaries.shape[1]:
            return deltas
        overlap = min(int(prev.shape[0]), num_chunks)
        if overlap <= 0:
            return deltas
        curr_tail = summaries[-overlap:]
        prev_tail = prev[-overlap:]
        denom = prev_tail.abs().mean(dim=1).clamp_min(1e-6)
        tail_deltas = (curr_tail - prev_tail).abs().mean(dim=1) / denom
        deltas[-overlap:] = tail_deltas
        return deltas

    def _update_layer_stats(self, layer_id: int, deltas: torch.Tensor) -> None:
        if deltas.numel() == 0:
            return
        layer_id = int(layer_id)
        total_sum = float(deltas.sum().item())
        total_count = int(deltas.numel())
        self._layer_all_delta_sum[layer_id] = self._layer_all_delta_sum.get(layer_id, 0.0) + total_sum
        self._layer_all_delta_count[layer_id] = self._layer_all_delta_count.get(layer_id, 0) + total_count

        num_recent_chunks = max(1, int(round(self.profile_recent_ratio * total_count)))
        old_count = max(0, total_count - num_recent_chunks)
        if old_count > 0:
            old_deltas = deltas[:old_count]
            self._layer_old_delta_sum[layer_id] = self._layer_old_delta_sum.get(layer_id, 0.0) + float(old_deltas.sum().item())
            self._layer_old_delta_count[layer_id] = self._layer_old_delta_count.get(layer_id, 0) + int(old_deltas.numel())

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
            active_seq_len = self._active_seq_len(int(k.shape[1]), meta)
            active_k = k[:, :active_seq_len]
            active_v = v[:, :active_seq_len]
            chunks = self._compute_chunk_boundaries(active_seq_len)
            summaries = self._compute_chunk_summaries(active_k, active_v, chunks)
            deltas = self._compute_chunk_deltas(layer_id, summaries)
            self._update_layer_stats(layer_id, deltas)
            self._prev_chunk_summaries[layer_id] = summaries
            state = {
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "orig_shape": tuple(k.shape),
            }
        self.stats.quantize_time_s += t[0]
        self.stats.bf16_kv_bytes = bf16_bytes
        self.stats.compressed_kv_bytes = self.memory_bytes(state)
        return state

    def dequantize_kv(self, state: Dict[str, Any], meta: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        with timed() as t:
            tensor_dtype = meta.get("tensor_dtype") if meta else None
            target_dtype = tensor_dtype if tensor_dtype is not None else state["k"].dtype
            k = state["k"].to(dtype=target_dtype)
            v = state["v"].to(dtype=target_dtype)
        self.stats.dequantize_time_s += t[0]
        return k, v

    def memory_bytes(self, state: Dict[str, Any]) -> int:
        return int(state["k"].numel() * state["k"].element_size() + state["v"].numel() * state["v"].element_size())

    def diagnostics(self) -> Dict[str, Any]:
        layer_scores = {
            str(layer_id): self._layer_old_delta_sum[layer_id] / max(self._layer_old_delta_count.get(layer_id, 1), 1)
            for layer_id in sorted(self._layer_old_delta_sum)
            if self._layer_old_delta_count.get(layer_id, 0) > 0
        }
        all_layer_scores = {
            str(layer_id): self._layer_all_delta_sum[layer_id] / max(self._layer_all_delta_count.get(layer_id, 1), 1)
            for layer_id in sorted(self._layer_all_delta_sum)
            if self._layer_all_delta_count.get(layer_id, 0) > 0
        }
        avg_old_delta = (
            sum(self._layer_old_delta_sum.values()) / max(sum(self._layer_old_delta_count.values()), 1)
            if self._layer_old_delta_count
            else 0.0
        )
        return {
            "flowcache_profile_recent_ratio": float(self.profile_recent_ratio),
            "flowcache_profile_num_layers": int(self.num_layers or 0),
            "flowcache_profile_avg_old_delta": float(avg_old_delta),
            "flowcache_profile_layer_scores": layer_scores,
            "flowcache_profile_layer_scores_all_chunks": all_layer_scores,
        }
