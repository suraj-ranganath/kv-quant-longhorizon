#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StoryEval benchmark run artifacts.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to results/benchmarks/storyeval/<run_id>")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    per_prompt_dir = run_dir / "per_prompt"
    metrics_dir = run_dir / "metrics"
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for p in sorted(per_prompt_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    success = [r for r in records if not r.get("error")]
    failed = [r for r in records if r.get("error")]
    runtimes = [float(r["wall_time_sec"]) for r in success if r.get("wall_time_sec") is not None]
    peaks = [float(r["peak_vram_mb"]) for r in success if r.get("peak_vram_mb") is not None]
    unique_prompt_ids = sorted({str(r.get("prompt_id")) for r in records if r.get("prompt_id") is not None})

    vbench = _load_json(metrics_dir / "vbench.json") or {}
    drift = _load_json(metrics_dir / "drift_imaging_quality.json") or {}
    vbench_agg = vbench.get("aggregate", {}) if isinstance(vbench.get("aggregate"), dict) else {}
    drift_curve = drift.get("curve", []) if isinstance(drift.get("curve"), list) else []
    drift_last = drift_curve[-1].get("imaging_quality") if drift_curve else None

    summary = {
        "benchmark": "storyeval",
        "run_id": run_dir.name,
        "num_records": len(records),
        "num_prompts": len(unique_prompt_ids),
        "num_success": len(success),
        "num_failed": len(failed),
        "avg_runtime_sec": _safe_mean(runtimes),
        "avg_peak_vram_mb": _safe_mean(peaks),
        "max_peak_vram_mb": max(peaks) if peaks else None,
        "vbench_background_consistency": vbench_agg.get("background_consistency"),
        "vbench_imaging_quality": vbench_agg.get("imaging_quality"),
        "vbench_subject_consistency": vbench_agg.get("subject_consistency"),
        "vbench_aesthetic_quality": vbench_agg.get("aesthetic_quality"),
        "drift_points": len(drift_curve),
        "drift_last_imaging_quality": drift_last,
    }

    summary_json = summary_dir / "summary.json"
    summary_csv = summary_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

