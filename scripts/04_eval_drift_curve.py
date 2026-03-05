#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    subprocess.check_call(cmd, cwd=cwd)


def build_prompt_map(videos_dir: Path, prompt_file: Path, out_json: Path) -> None:
    prompts = [x.rstrip("\n") for x in prompt_file.read_text(encoding="utf-8").splitlines()]
    mapping = {}
    for p in sorted(videos_dir.glob("prompt_*_seed_*.mp4")):
        m = re.search(r"prompt_(\d+)_seed_", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < len(prompts):
            mapping[str(p)] = prompts[idx]
    out_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def truncate_videos(src_dir: Path, dst_dir: Path, num_frames: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.mp4")):
        dst = dst_dir / src.name
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-frames:v",
            str(num_frames),
            str(dst),
        ]
        run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate long-horizon drift using VBench imaging_quality every N frames.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/MovieGenVideoBench_extended.txt"))
    parser.add_argument("--frame-step", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=700)
    parser.add_argument("--work-dir", type=Path, default=Path("results/metrics/drift_work"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--vbench-eval", type=Path, default=Path("third_party/VBench/evaluate.py"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = args.output or (root / "results" / "metrics" / f"drift_{args.method}.json")
    work_dir = root / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for frame_cap in range(args.frame_step, args.max_frames + 1, args.frame_step):
        clip_dir = work_dir / f"{args.method}_frames_{frame_cap}"
        truncate_videos(root / args.videos_dir, clip_dir, frame_cap)

        prompt_map = work_dir / f"prompt_map_{frame_cap}.json"
        build_prompt_map(clip_dir, root / args.prompt_file, prompt_map)

        vbench_out = work_dir / f"vbench_{args.method}_{frame_cap}"
        vbench_out.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                "python",
                str(root / args.vbench_eval),
                "--videos_path",
                str(clip_dir),
                "--mode",
                "custom_input",
                "--prompt_file",
                str(prompt_map),
                "--dimension",
                "imaging_quality",
                "--output_path",
                str(vbench_out),
            ],
            cwd=str(root),
        )

        latest = sorted(vbench_out.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
        payload = json.loads(latest.read_text(encoding="utf-8"))
        score = payload.get("imaging_quality", None)
        scores.append({"frame_cap": frame_cap, "imaging_quality": score})

    out = {
        "method": args.method,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "curve": scores,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
