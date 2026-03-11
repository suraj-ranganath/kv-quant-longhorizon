#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


VIDEO_NAME_RE = re.compile(r"^(?P<prompt_id>.+)_seed(?P<seed>\d+)\.mp4$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_seeded_video_name(name: str) -> tuple[str | None, int | None]:
    m = VIDEO_NAME_RE.match(name)
    if not m:
        return None, None
    return m.group("prompt_id"), int(m.group("seed"))


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _extract_aggregate(value: Any) -> float | None:
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBench for StoryEval videos and save canonical metrics JSON.")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to results/benchmarks/storyeval/<run_id>")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--vbench_eval_path", type=Path, default=Path("third_party/VBench/evaluate.py"))
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["background_consistency", "imaging_quality", "subject_consistency", "aesthetic_quality"],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    per_prompt_dir = run_dir / "per_prompt"
    videos_dir = run_dir / "videos"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    vbench_raw_dir = metrics_dir / "vbench_raw"
    vbench_raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_map_path = metrics_dir / "vbench_prompt_map.json"

    prompt_map: dict[str, str] = {}
    prompt_records_by_video: dict[str, dict[str, Any]] = {}
    for rec_path in sorted(per_prompt_dir.glob("*.json")):
        rec = _load_json(rec_path)
        if rec.get("error"):
            continue
        video_rel = rec.get("generated_video_path")
        if not isinstance(video_rel, str):
            continue
        video_abs = (repo_root / video_rel)
        if not video_abs.exists():
            continue
        prompt = rec.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt_map[str(video_abs)] = prompt
        prompt_map[str(video_abs.resolve())] = prompt
        prompt_records_by_video[video_abs.name] = rec

    if not prompt_map:
        raise RuntimeError(f"No successful StoryEval per_prompt records found in {per_prompt_dir}")

    prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd = [
        args.python_bin,
        str((repo_root / args.vbench_eval_path) if not args.vbench_eval_path.is_absolute() else args.vbench_eval_path),
        "--videos_path",
        str(videos_dir),
        "--mode",
        "custom_input",
        "--prompt_file",
        str(prompt_map_path),
        "--dimension",
        *args.dimensions,
        "--output_path",
        str(vbench_raw_dir),
    ]
    _run_cmd(cmd, cwd=repo_root)

    candidates = sorted(vbench_raw_dir.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"VBench output not found in {vbench_raw_dir}")
    latest_raw = candidates[0]
    raw_payload = _load_json(latest_raw)
    shutil.copy2(latest_raw, metrics_dir / "vbench_raw_latest.json")

    aggregate: dict[str, float | None] = {}
    per_video: dict[str, dict[str, Any]] = {}
    for dim in args.dimensions:
        dim_payload = raw_payload.get(dim)
        aggregate[dim] = _extract_aggregate(dim_payload)
        details = dim_payload[1] if isinstance(dim_payload, list) and len(dim_payload) > 1 and isinstance(dim_payload[1], list) else []
        for entry in details:
            if not isinstance(entry, dict):
                continue
            video_path = entry.get("video_path")
            if not isinstance(video_path, str):
                continue
            video_name = Path(video_path).name
            rec = per_video.setdefault(video_name, {})
            rec[dim] = entry.get("video_results")

    for video_name, rec in per_video.items():
        prompt_id, seed = _parse_seeded_video_name(video_name)
        meta = prompt_records_by_video.get(video_name, {})
        rec["prompt_id"] = prompt_id or meta.get("prompt_id")
        rec["seed"] = seed if seed is not None else meta.get("seed")
        rec["prompt"] = meta.get("prompt")
        rec["video_path"] = str((videos_dir / video_name).resolve())

    out_payload = {
        "benchmark": "storyeval",
        "run_id": run_dir.name,
        "dimensions": args.dimensions,
        "aggregate": aggregate,
        "per_video": per_video,
        "raw_vbench_path": str(latest_raw),
    }
    out_path = metrics_dir / "vbench.json"
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
