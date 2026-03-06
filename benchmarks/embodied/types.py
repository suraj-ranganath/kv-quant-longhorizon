from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EmbodiedSample:
    sample_id: str
    prompt: str
    cond_image_path: str | None
    qa_pairs: list[dict[str, Any]]
    meta: dict[str, Any]


@dataclass
class RunRecord:
    run_id: str
    method: str
    config: dict[str, Any]
    per_sample_dir: str
    videos_dir: str
    summary_path: str
