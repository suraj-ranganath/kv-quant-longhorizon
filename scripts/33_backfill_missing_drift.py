#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backfill_paths import BACKFILL_ROOT, metric_backfill_aux_dir, metric_backfill_path

DEFAULT_DATASET_CSV = REPO_ROOT / "results" / "combined" / "combined_comparison_dataset.csv"
DEFAULT_EVAL_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python")
DEFAULT_MOVIEGEN_PROMPTS = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
DEFAULT_SUMMARY_JSON = BACKFILL_ROOT / "drift_backfill_summary.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def has_metric_value(row: dict[str, str], key: str) -> bool:
    return row.get(key) not in (None, "")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def deterministic_master_port(*, benchmark: str, run_root: Path, method: str) -> str:
    payload = f"{benchmark}|{run_root.resolve()}|{method}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return str(15000 + (int(digest[:8], 16) % 20000))


def ffprobe_frame_count(video_path: Path) -> int:
    probes = [
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
    ]
    for cmd in probes:
        try:
            out = subprocess.check_output(cmd).decode("utf-8").strip()
            if out and out != "N/A":
                return int(float(out))
        except Exception:
            continue
    return 0


def collect_targets(
    dataset_csv: Path,
    *,
    selected_benchmarks: set[str],
    selected_methods: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with dataset_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            benchmark = row.get("benchmark")
            source_user = row.get("source_user")
            run_root = row.get("run_root")
            method = row.get("method")
            if not benchmark or not source_user or not run_root or not method:
                continue
            if benchmark not in selected_benchmarks:
                continue
            if selected_methods and method not in selected_methods:
                continue
            is_ten_second = str(row.get("is_ten_second")).lower() in {"1", "true", "yes"}
            if not is_ten_second:
                continue
            drift_key = "moviegen_drift_last_imaging_quality" if benchmark == "moviegen" else "storyeval_drift_last_imaging_quality"
            if has_metric_value(row, drift_key):
                continue
            key = (benchmark, source_user, run_root, method)
            grouped.setdefault(
                key,
                {
                    "benchmark": benchmark,
                    "source_user": source_user,
                    "run_root": run_root,
                    "method": method,
                },
            )
    return sorted(grouped.values(), key=lambda row: (row["benchmark"], row["source_user"], row["method"], row["run_root"]))


def moviegen_videos_dir(run_root: Path, method: str) -> Path:
    return run_root / "videos" / method


def detect_common_frame_count(videos_dir: Path) -> int:
    counts = [ffprobe_frame_count(path) for path in sorted(videos_dir.glob("*.mp4"))]
    counts = [count for count in counts if count > 0]
    if not counts:
        raise RuntimeError(f"Unable to determine frame counts in {videos_dir}")
    return min(counts)


def backfill_moviegen_drift(
    *,
    target: dict[str, Any],
    eval_python: Path,
    prompt_file: Path,
) -> dict[str, Any]:
    run_root = Path(target["run_root"]).resolve()
    method = str(target["method"])
    videos_dir = moviegen_videos_dir(run_root, method)
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory for {method}: {videos_dir}")

    terminal_frame_cap = detect_common_frame_count(videos_dir)
    output_json = metric_backfill_path(
        benchmark="moviegen",
        metric_kind="drift",
        run_root=run_root,
        method=method,
    )
    work_dir = metric_backfill_aux_dir(
        benchmark="moviegen",
        metric_kind="drift",
        run_root=run_root,
        method=method,
    )
    env = os.environ.copy()
    env["PATH"] = f"{eval_python.parent}:{env.get('PATH', '')}"
    env["MASTER_PORT"] = deterministic_master_port(benchmark="moviegen", run_root=run_root, method=method)
    cmd = [
        str(eval_python),
        str(REPO_ROOT / "scripts" / "04_eval_drift_curve.py"),
        "--method",
        method,
        "--videos-dir",
        str(videos_dir),
        "--prompt-file",
        str(prompt_file),
        "--frame-step",
        str(terminal_frame_cap),
        "--max-frames",
        str(terminal_frame_cap),
        "--work-dir",
        str(work_dir),
        "--output",
        str(output_json),
    ]
    run_cmd(cmd, env=env)
    payload = load_json(output_json)
    return {
        "benchmark": "moviegen",
        "source_user": target["source_user"],
        "run_root": str(run_root),
        "method": method,
        "output_json": str(output_json),
        "num_videos": sum(1 for _ in videos_dir.glob("*.mp4")),
        "terminal_frame_cap": terminal_frame_cap,
        "drift_points": len(payload.get("curve", [])) if isinstance(payload.get("curve"), list) else None,
    }


def backfill_storyeval_drift(
    *,
    target: dict[str, Any],
    eval_python: Path,
) -> dict[str, Any]:
    run_root = Path(target["run_root"]).resolve()
    method = str(target["method"])
    output_json = metric_backfill_path(
        benchmark="storyeval",
        metric_kind="drift",
        run_root=run_root,
        method=method,
    )
    env = os.environ.copy()
    env["PATH"] = f"{eval_python.parent}:{env.get('PATH', '')}"
    env["MASTER_PORT"] = deterministic_master_port(benchmark="storyeval", run_root=run_root, method=method)
    cmd = [
        str(eval_python),
        str(REPO_ROOT / "scripts" / "eval_storyeval_drift.py"),
        "--run_dir",
        str(run_root),
        "--python_bin",
        str(eval_python),
    ]
    run_cmd(cmd, env=env)
    local_json = run_root / "metrics" / "drift_imaging_quality.json"
    if not local_json.exists():
        raise RuntimeError(f"StoryEval drift script did not produce {local_json}")
    write_json(output_json, load_json(local_json))
    payload = load_json(output_json)
    return {
        "benchmark": "storyeval",
        "source_user": target["source_user"],
        "run_root": str(run_root),
        "method": method,
        "output_json": str(output_json),
        "drift_points": len(payload.get("curve", [])) if isinstance(payload.get("curve"), list) else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill missing MovieGen/StoryEval drift metrics from existing videos only.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--eval-python", type=Path, default=DEFAULT_EVAL_PYTHON)
    parser.add_argument("--moviegen-prompts", type=Path, default=DEFAULT_MOVIEGEN_PROMPTS)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("moviegen", "storyeval"),
        help="Restrict drift backfills to one or more benchmarks.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Restrict drift backfills to one or more methods.",
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_benchmarks = set(args.benchmark or ["moviegen", "storyeval"])
    selected_methods = set(args.method or [])
    targets = collect_targets(
        args.dataset_csv,
        selected_benchmarks=selected_benchmarks,
        selected_methods=selected_methods,
    )
    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        if target["benchmark"] == "moviegen":
            summary_rows.append(
                backfill_moviegen_drift(
                    target=target,
                    eval_python=args.eval_python,
                    prompt_file=args.moviegen_prompts,
                )
            )
        else:
            summary_rows.append(
                backfill_storyeval_drift(
                    target=target,
                    eval_python=args.eval_python,
                )
            )
    write_json(
        args.summary_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_targets": len(summary_rows),
            "targets": summary_rows,
        },
    )
    print(f"[drift-backfill] summary {args.summary_json}")


if __name__ == "__main__":
    main()
