#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


VIDEO_NAME_RE = re.compile(r"^(?P<prompt_id>.+)_seed(?P<seed>\d+)\.mp4$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _ffprobe_frame_count(video_path: Path) -> int:
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


def _extract_aggregate(value: Any) -> float | None:
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _prompt_seed_from_name(name: str) -> tuple[str | None, int | None]:
    m = VIDEO_NAME_RE.match(name)
    if not m:
        return None, None
    return m.group("prompt_id"), int(m.group("seed"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute StoryEval long-horizon drift using VBench imaging_quality.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--vbench_eval_path", type=Path, default=Path("third_party/VBench/evaluate.py"))
    parser.add_argument("--frame_step", type=int, default=50)
    parser.add_argument("--max_prompts_for_drift", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    per_prompt_dir = run_dir / "per_prompt"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    drift_work_dir = metrics_dir / "drift_work"
    drift_work_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for rec_path in sorted(per_prompt_dir.glob("*.json")):
        rec = _load_json(rec_path)
        if rec.get("error"):
            continue
        video_rel = rec.get("generated_video_path")
        if not isinstance(video_rel, str):
            continue
        video_path = (repo_root / video_rel).resolve()
        if not video_path.exists():
            continue
        records.append((rec, video_path))
    if not records:
        raise RuntimeError(f"No successful per_prompt records found in {per_prompt_dir}")

    by_prompt: dict[str, tuple[dict[str, Any], Path]] = {}
    for rec, video_path in records:
        prompt_id = rec.get("prompt_id")
        if not isinstance(prompt_id, str):
            prompt_id, _ = _prompt_seed_from_name(video_path.name)
        if not isinstance(prompt_id, str):
            continue
        if prompt_id not in by_prompt:
            by_prompt[prompt_id] = (rec, video_path)

    selected = [by_prompt[k] for k in sorted(by_prompt.keys())[: max(args.max_prompts_for_drift, 1)]]
    if not selected:
        raise RuntimeError("Unable to select any prompts for drift evaluation.")

    fps = int(selected[0][0].get("fps") or 16)
    min_frames = min(_ffprobe_frame_count(video_path) for _, video_path in selected)
    if min_frames <= 0:
        raise RuntimeError("Unable to determine frame count for drift evaluation videos.")

    max_frame_cap = int(math.floor(min_frames / args.frame_step) * args.frame_step)
    if max_frame_cap <= 0:
        raise RuntimeError(
            f"frame_step={args.frame_step} is too large for min_frames={min_frames}. "
            "Use a smaller frame step."
        )

    curve: list[dict[str, Any]] = []
    for frame_cap in range(args.frame_step, max_frame_cap + 1, args.frame_step):
        clip_dir = drift_work_dir / f"clips_{frame_cap}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        prompt_map: dict[str, str] = {}
        for rec, src_video in selected:
            dst_video = clip_dir / src_video.name
            _run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(src_video),
                    "-frames:v",
                    str(frame_cap),
                    str(dst_video),
                ]
            )
            prompt_map[str(dst_video.resolve())] = str(rec.get("prompt", ""))

        prompt_map_path = drift_work_dir / f"prompt_map_{frame_cap}.json"
        prompt_map_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")

        vbench_out = drift_work_dir / f"vbench_imaging_quality_{frame_cap}"
        vbench_out.mkdir(parents=True, exist_ok=True)
        _run_cmd(
            [
                args.python_bin,
                str((repo_root / args.vbench_eval_path) if not args.vbench_eval_path.is_absolute() else args.vbench_eval_path),
                "--videos_path",
                str(clip_dir),
                "--mode",
                "custom_input",
                "--prompt_file",
                str(prompt_map_path),
                "--dimension",
                "imaging_quality",
                "--output_path",
                str(vbench_out),
            ],
            cwd=repo_root,
        )

        candidates = sorted(vbench_out.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"Missing VBench output for frame_cap={frame_cap}")
        payload = _load_json(candidates[0])
        iq = _extract_aggregate(payload.get("imaging_quality"))
        curve.append(
            {
                "frame_cap": frame_cap,
                "seconds": frame_cap / float(fps),
                "imaging_quality": iq,
            }
        )

    drift_payload = {
        "benchmark": "storyeval",
        "run_id": run_dir.name,
        "frame_step": args.frame_step,
        "max_prompts_for_drift": args.max_prompts_for_drift,
        "num_prompts_evaluated": len(selected),
        "fps": fps,
        "min_frames_across_selected": min_frames,
        "curve": curve,
    }
    drift_json = metrics_dir / "drift_imaging_quality.json"
    drift_json.write_text(json.dumps(drift_payload, indent=2), encoding="utf-8")

    x = [p["seconds"] for p in curve]
    y = [p["imaging_quality"] for p in curve]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(x, y, marker="o")
    plt.title("StoryEval Drift: Imaging Quality vs Prefix Duration")
    plt.xlabel("Prefix Duration (s)")
    plt.ylabel("VBench Imaging Quality")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = plots_dir / "drift_imaging_quality.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Wrote {drift_json}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

