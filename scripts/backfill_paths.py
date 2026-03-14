from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKFILL_ROOT = REPO_ROOT / "results" / "combined" / "backfills"


def metric_backfill_key(*, benchmark: str, metric_kind: str, run_root: Path | str, method: str) -> str:
    payload = f"{benchmark}|{metric_kind}|{Path(run_root).resolve()}|{method}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def metric_backfill_path(*, benchmark: str, metric_kind: str, run_root: Path | str, method: str) -> Path:
    key = metric_backfill_key(benchmark=benchmark, metric_kind=metric_kind, run_root=run_root, method=method)
    if metric_kind == "storyeval_system":
        return BACKFILL_ROOT / "storyeval_system_metrics" / f"{key}_{method}.json"
    return BACKFILL_ROOT / metric_kind / benchmark / f"{key}_{method}.json"


def metric_backfill_aux_dir(*, benchmark: str, metric_kind: str, run_root: Path | str, method: str) -> Path:
    key = metric_backfill_key(benchmark=benchmark, metric_kind=metric_kind, run_root=run_root, method=method)
    return BACKFILL_ROOT / metric_kind / benchmark / f"{key}_{method}"
