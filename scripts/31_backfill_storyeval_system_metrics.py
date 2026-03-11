#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.t2v.storyeval import storyeval_video_name
from scripts.backfill_paths import BACKFILL_ROOT, metric_backfill_path

DEFAULT_DATASET_CSV = REPO_ROOT / "results" / "combined" / "combined_comparison_dataset.csv"
DEFAULT_SUMMARY_JSON = BACKFILL_ROOT / "storyeval_system_metrics_summary.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def mean_or_none(values: list[float | int | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def selected_storyeval_targets(dataset_csv: Path) -> list[dict[str, str]]:
    targets: dict[tuple[str, str, str], dict[str, str]] = {}
    with dataset_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("benchmark") != "storyeval":
                continue
            run_root = row.get("run_root")
            method = row.get("method")
            source_user = row.get("source_user")
            if not run_root or not method or not source_user:
                continue
            targets[(source_user, run_root, method)] = {
                "source_user": source_user,
                "run_root": run_root,
                "method": method,
            }
    return sorted(targets.values(), key=lambda row: (row["source_user"], row["method"], row["run_root"]))


def extract_trace_metrics(
    *,
    method: str,
    per_prompt_records: dict[str, dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    per_video: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        prompt_id = row.get("prompt_id")
        seed = safe_int(row.get("seed"))
        if not isinstance(prompt_id, str) or seed is None:
            continue
        video_name = storyeval_video_name(prompt_id, seed)
        record = per_prompt_records.get(video_name, {})
        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        bf16_kv_bytes = max((safe_int(sample.get("bf16_kv_bytes")) or 0) for sample in samples) if samples else None
        compressed_kv_bytes = max((safe_int(sample.get("compressed_kv_bytes")) or 0) for sample in samples) if samples else None
        if bf16_kv_bytes == 0:
            bf16_kv_bytes = None
        if compressed_kv_bytes == 0:
            compressed_kv_bytes = None
        if method == "BF16" and bf16_kv_bytes is not None and compressed_kv_bytes is None:
            compressed_kv_bytes = bf16_kv_bytes
        compression_ratio = None
        if bf16_kv_bytes is not None and compressed_kv_bytes not in (None, 0):
            compression_ratio = float(bf16_kv_bytes) / float(compressed_kv_bytes)

        peak_vram_bytes = first_non_null(
            safe_int(row.get("peak_vram_bytes")),
            safe_int(record.get("peak_vram_bytes")),
        )
        wall_time_sec = first_non_null(
            safe_float(row.get("runtime_s")),
            safe_float(record.get("wall_time_sec")),
        )
        per_video[video_name] = {
            "prompt_id": prompt_id,
            "seed": seed,
            "bf16_kv_bytes": bf16_kv_bytes,
            "compressed_kv_bytes": compressed_kv_bytes,
            "compression_ratio": compression_ratio,
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_mb": (float(peak_vram_bytes) / (1024.0 * 1024.0)) if peak_vram_bytes is not None else None,
            "wall_time_sec": wall_time_sec,
        }
    return per_video


def build_per_prompt_record_map(run_root: Path) -> dict[str, dict[str, Any]]:
    per_prompt: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "per_prompt").glob("*.json")):
        record = load_json(path)
        video_rel = record.get("generated_video_path")
        video_name = Path(video_rel).name if isinstance(video_rel, str) and video_rel else None
        if video_name is None:
            prompt_id = record.get("prompt_id")
            seed = safe_int(record.get("seed"))
            if isinstance(prompt_id, str) and seed is not None:
                video_name = storyeval_video_name(prompt_id, seed)
        if video_name is None:
            continue
        per_prompt[video_name] = record
    return per_prompt


def build_backfill_payload(
    *,
    source_user: str,
    run_root: Path,
    method: str,
) -> dict[str, Any]:
    per_prompt_records = build_per_prompt_record_map(run_root)
    if not per_prompt_records:
        raise RuntimeError(f"Missing StoryEval per_prompt records under {run_root}")

    trace_path = run_root / "logs" / "vram_trace_storyeval.jsonl"
    if not trace_path.exists():
        raise RuntimeError(f"Missing StoryEval trace log at {trace_path}")
    trace_rows = load_jsonl(trace_path)
    if not trace_rows:
        raise RuntimeError(f"StoryEval trace log is empty at {trace_path}")

    summary_path = run_root / "summary" / "summary.json"
    if not summary_path.exists():
        summary_path = run_root / "summary" / "runner_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}

    efficiency_path = run_root / "metrics" / f"efficiency_{method}.json"
    efficiency = load_json(efficiency_path) if efficiency_path.exists() else {}

    per_video = extract_trace_metrics(method=method, per_prompt_records=per_prompt_records, trace_rows=trace_rows)
    if not per_video:
        raise RuntimeError(f"Unable to derive per-video StoryEval system metrics for {run_root}")

    bf16_values = [safe_int(row.get("bf16_kv_bytes")) for row in per_video.values()]
    compressed_values = [safe_int(row.get("compressed_kv_bytes")) for row in per_video.values()]
    runtime_values = [
        first_non_null(safe_float(row.get("wall_time_sec")), safe_float(record.get("wall_time_sec")))
        for video_name, row in per_video.items()
        for record in [per_prompt_records.get(video_name, {})]
    ]
    peak_bytes_values = [safe_int(row.get("peak_vram_bytes")) for row in per_video.values()]

    agg_bf16_kv_bytes = first_non_null(
        safe_int(efficiency.get("bf16_kv_bytes")),
        max((value for value in bf16_values if value is not None), default=None),
    )
    agg_compressed_kv_bytes = first_non_null(
        safe_int(efficiency.get("compressed_kv_bytes")),
        max((value for value in compressed_values if value is not None), default=None),
    )
    agg_compression_ratio = safe_float(efficiency.get("compression_ratio"))
    if agg_compression_ratio is None and agg_bf16_kv_bytes is not None and agg_compressed_kv_bytes not in (None, 0):
        agg_compression_ratio = float(agg_bf16_kv_bytes) / float(agg_compressed_kv_bytes)

    aggregate = {
        "bf16_kv_bytes": agg_bf16_kv_bytes,
        "compressed_kv_bytes": agg_compressed_kv_bytes,
        "compression_ratio": agg_compression_ratio,
        "quantize_time_s": first_non_null(safe_float(efficiency.get("quantize_time_s")), 0.0 if method == "BF16" else None),
        "dequantize_time_s": first_non_null(safe_float(efficiency.get("dequantize_time_s")), 0.0 if method == "BF16" else None),
        "total_runtime_s": first_non_null(
            safe_float(efficiency.get("total_runtime_s")),
            sum(value for value in runtime_values if value is not None) if any(value is not None for value in runtime_values) else None,
        ),
        "avg_runtime_s_per_prompt": first_non_null(
            safe_float(efficiency.get("avg_runtime_s_per_prompt")),
            mean_or_none(runtime_values),
            safe_float(summary.get("avg_runtime_sec")),
        ),
        "avg_peak_vram_mb": first_non_null(
            safe_float(summary.get("avg_peak_vram_mb")),
            mean_or_none([(float(value) / (1024.0 * 1024.0)) if value is not None else None for value in peak_bytes_values]),
        ),
        "max_peak_vram_mb": first_non_null(
            safe_float(summary.get("max_peak_vram_mb")),
            max(((float(value) / (1024.0 * 1024.0)) for value in peak_bytes_values if value is not None), default=None),
        ),
        "num_videos": len(per_video),
        "num_trace_rows": len(trace_rows),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_user": source_user,
        "benchmark": "storyeval",
        "run_root": str(run_root),
        "method": method,
        "source_trace_path": str(trace_path),
        "source_efficiency_path": str(efficiency_path) if efficiency_path.exists() else None,
        "aggregate": aggregate,
        "per_video": per_video,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill StoryEval system metrics from existing traces and efficiency logs.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = selected_storyeval_targets(args.dataset_csv)
    if not targets:
        raise RuntimeError(f"No StoryEval targets found in {args.dataset_csv}")

    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        run_root = Path(target["run_root"]).resolve()
        method = target["method"]
        output_path = metric_backfill_path(
            benchmark="storyeval",
            metric_kind="storyeval_system",
            run_root=run_root,
            method=method,
        )
        payload = build_backfill_payload(
            source_user=target["source_user"],
            run_root=run_root,
            method=method,
        )
        write_json(output_path, payload)
        print(f"[storyeval-system] wrote {output_path}")
        summary_rows.append(
            {
                "source_user": target["source_user"],
                "run_root": str(run_root),
                "method": method,
                "output_path": str(output_path),
                "num_videos": payload["aggregate"]["num_videos"],
                "compression_ratio": payload["aggregate"]["compression_ratio"],
            }
        )

    write_json(
        args.summary_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_targets": len(summary_rows),
            "targets": summary_rows,
        },
    )
    print(f"[storyeval-system] summary {args.summary_json}")


if __name__ == "__main__":
    main()
