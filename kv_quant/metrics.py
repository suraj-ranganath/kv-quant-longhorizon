from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EfficiencyMetrics:
    method: str
    total_runtime_s: float
    peak_vram_bytes: int
    quantize_time_s: float
    dequantize_time_s: float
    bf16_kv_bytes: int
    compressed_kv_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.compressed_kv_bytes <= 0:
            return 0.0
        return self.bf16_kv_bytes / self.compressed_kv_bytes

    def to_dict(self):
        data = asdict(self)
        data["compression_ratio"] = self.compression_ratio
        return data
