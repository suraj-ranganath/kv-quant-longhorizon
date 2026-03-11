#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.t2v.storyeval import StoryEvalLoader, storyeval_video_name

DEFAULT_PROMPT_FILE = REPO_ROOT / "data" / "prompts" / "storyeval" / "all_prompts.txt"
DEFAULT_INFER_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python")
DEFAULT_EVAL_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def safe_symlink(src: Path, dst: Path, *, force: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_generated_videos(video_dir: Path) -> int:
    if not video_dir.exists():
        return 0
    return sum(1 for _ in video_dir.glob("prompt_*_seed_*.mp4"))


def deterministic_master_port(namespace: str) -> str:
    digest = hashlib.sha1(namespace.encode("utf-8")).hexdigest()
    return str(15000 + (int(digest[:8], 16) % 20000))


def build_generate_cmd(
    run_root: Path,
    method: str,
    prompt_file: Path,
    passthrough: list[str],
    *,
    infer_python: Path,
    max_prompts: int,
    num_output_frames: int,
    seed: int,
    fps: int,
) -> list[str]:
    cmd = [
        str(infer_python),
        str(REPO_ROOT / "scripts" / "01_generate.py"),
        "--results-root",
        str(run_root),
        "--method",
        method,
        "--prompt-path",
        str(prompt_file),
        "--max-prompts",
        str(max_prompts),
        "--num-output-frames",
        str(num_output_frames),
        "--seed",
        str(seed),
        "--fps",
        str(fps),
    ]
    cmd.extend(passthrough)
    return cmd


def build_flowcache_profile_cmd(
    run_root: Path,
    prompt_file: Path,
    passthrough: list[str],
    *,
    infer_python: Path,
    max_prompts: int,
    num_output_frames: int,
    seed: int,
    fps: int,
    recent_ratio: float,
) -> list[str]:
    filtered_passthrough: list[str] = []
    idx = 0
    while idx < len(passthrough):
        arg = passthrough[idx]
        if arg == "--flowcache-layer-budget-path":
            idx += 2
            continue
        if arg == "--device":
            idx += 2
            continue
        if arg == "--use-ema":
            idx += 1
            continue
        filtered_passthrough.append(arg)
        idx += 1
    cmd = [
        str(infer_python),
        str(REPO_ROOT / "scripts" / "01_generate.py"),
        "--results-root",
        str(run_root / "profile_pass"),
        "--method",
        "FLOWCACHE_PROFILE",
        "--prompt-path",
        str(prompt_file),
        "--max-prompts",
        str(max_prompts),
        "--num-output-frames",
        str(num_output_frames),
        "--seed",
        str(seed),
        "--fps",
        str(fps),
        "--flowcache-recent-ratio",
        str(recent_ratio),
        "--device",
        "cuda:0",
        "--use-ema",
    ]
    cmd.extend(filtered_passthrough)
    return cmd


def write_flowcache_layer_budget(
    profile_efficiency_path: Path,
    output_path: Path,
    *,
    min_scale: float,
    max_scale: float,
) -> None:
    payload = load_json(profile_efficiency_path)
    scores = payload.get("flowcache_profile_layer_scores", {})
    if not isinstance(scores, dict) or not scores:
        table: dict[int, float] = {}
    else:
        items = {int(key): float(value) for key, value in scores.items()}
        min_score = min(items.values())
        max_score = max(items.values())
        table = {}
        for layer_id, score in sorted(items.items()):
            if max_score > min_score:
                norm = (score - min_score) / (max_score - min_score)
            else:
                norm = 0.5
            table[layer_id] = min_scale + norm * (max_scale - min_scale)
    write_json(output_path, table)


def expected_storyeval_prompts(prompt_file: Path, *, max_prompts: int) -> list[Any]:
    prompts = StoryEvalLoader(prompt_path=prompt_file).load()
    return prompts[:max_prompts]


def augment_vram_samples(samples: list[dict[str, Any]], efficiency: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    bf16_kv_bytes = efficiency.get("bf16_kv_bytes")
    compressed_kv_bytes = efficiency.get("compressed_kv_bytes")
    for sample in samples:
        row = dict(sample)
        if bf16_kv_bytes is not None:
            row.setdefault("bf16_kv_bytes", bf16_kv_bytes)
        if compressed_kv_bytes is not None:
            row.setdefault("compressed_kv_bytes", compressed_kv_bytes)
        out.append(row)
    return out


def rewrite_storyeval_layout(
    run_root: Path,
    *,
    run_name: str,
    method: str,
    config_id: str | None,
    prompt_file: Path,
    max_prompts: int,
    fps: int,
    duration_sec_requested: float,
    seed_base: int,
    source_moviegen_run_root: str | None,
    force: bool,
) -> dict[str, Any]:
    prompts = expected_storyeval_prompts(prompt_file, max_prompts=max_prompts)
    prompt_by_line_index = {
        int(prompt.meta.get("line_index", idx)): prompt
        for idx, prompt in enumerate(prompts)
    }

    generation_log_path = run_root / "logs" / f"generation_{method}.jsonl"
    trace_log_path = run_root / "logs" / f"vram_trace_{method}.jsonl"
    generation_records = load_jsonl(generation_log_path)
    if not generation_records:
        raise RuntimeError(f"Missing generation records at {generation_log_path}")

    efficiency_path = run_root / "metrics" / f"efficiency_{method}.json"
    efficiency = load_json(efficiency_path) if efficiency_path.exists() else {}

    canonical_records: list[dict[str, Any]] = []
    storyeval_traces: list[dict[str, Any]] = []
    per_prompt_dir = run_root / "per_prompt"
    videos_dir = run_root / "videos"
    per_prompt_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in per_prompt_dir.glob("*.json"):
            remove_path(path)
        for path in videos_dir.glob("*.mp4"):
            remove_path(path)

    first_total_frames: int | None = None
    first_target_latent_frames: int | None = None
    first_resolution: list[int] | None = None

    for record in generation_records:
        prompt_idx = int(record["prompt_id"])
        prompt_obj = prompt_by_line_index.get(prompt_idx)
        if prompt_obj is None:
            raise KeyError(f"StoryEval prompt index {prompt_idx} not found in {prompt_file}")
        seed = int(record["seed"])
        source_video_rel = record["output_video"]
        source_video_path = (REPO_ROOT / source_video_rel).resolve()
        if not source_video_path.exists():
            raise FileNotFoundError(f"Generated video missing: {source_video_path}")

        target_name = storyeval_video_name(prompt_obj.prompt_id, seed)
        target_video_path = videos_dir / target_name
        safe_symlink(source_video_path, target_video_path, force=force)

        total_frames = int(record["total_frames"]) if record.get("total_frames") is not None else None
        latents_shape = record.get("latents_shape")
        target_latent_frames = (
            int(latents_shape[1])
            if isinstance(latents_shape, list) and len(latents_shape) > 1 and isinstance(latents_shape[1], (int, float))
            else None
        )
        resolution = record.get("resolution")
        peak_vram_bytes = int(record["peak_vram_bytes"]) if record.get("peak_vram_bytes") is not None else None
        first_total_frames = first_total_frames or total_frames
        first_target_latent_frames = first_target_latent_frames or target_latent_frames
        first_resolution = first_resolution or (resolution if isinstance(resolution, list) else None)

        payload = {
            "benchmark": "storyeval",
            "method": method,
            "config_id": config_id,
            "run_id": run_name,
            "prompt_id": prompt_obj.prompt_id,
            "prompt": prompt_obj.prompt,
            "seed": seed,
            "fps": fps,
            "duration_sec_requested": duration_sec_requested,
            "effective_duration_sec": (float(total_frames) / float(fps)) if total_frames is not None and fps > 0 else None,
            "target_latent_frames": target_latent_frames,
            "target_frames": total_frames,
            "total_frames": total_frames,
            "resolution": resolution,
            "wall_time_sec": record.get("wall_clock_runtime_s"),
            "peak_vram_mb": (float(peak_vram_bytes) / (1024.0 * 1024.0)) if peak_vram_bytes is not None else None,
            "peak_vram_bytes": peak_vram_bytes,
            "generated_video_path": str(target_video_path.relative_to(REPO_ROOT)),
            "sf_config_path": record.get("model_config"),
            "sf_checkpoint_path": record.get("checkpoint_path"),
            "git_commit_hash": record.get("git_commit_hash"),
            "line_index": prompt_obj.meta.get("line_index"),
            "source_moviegen_run_root": source_moviegen_run_root,
            "error": None,
        }
        out_json = per_prompt_dir / f"{prompt_obj.prompt_id}_seed{seed}.json"
        write_json(out_json, payload)
        canonical_records.append(payload)

    trace_records = load_jsonl(trace_log_path)
    for record in trace_records:
        prompt_idx = int(record["prompt_id"])
        prompt_obj = prompt_by_line_index.get(prompt_idx)
        if prompt_obj is None:
            continue
        storyeval_traces.append(
            {
                "method": method,
                "config_id": config_id,
                "prompt_id": prompt_obj.prompt_id,
                "prompt": prompt_obj.prompt,
                "line_index": prompt_obj.meta.get("line_index"),
                "seed": int(record["seed"]),
                "runtime_s": record.get("runtime_s"),
                "peak_vram_bytes": record.get("peak_vram_bytes"),
                "samples": augment_vram_samples(record.get("samples", []), efficiency),
            }
        )
    write_jsonl(run_root / "logs" / "vram_trace_storyeval.jsonl", storyeval_traces)

    config_payload = {
        "benchmark": "storyeval",
        "method": method,
        "config_id": config_id,
        "run_id": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_file": str(prompt_file),
        "num_prompts_selected": len(prompts),
        "start_idx": 0,
        "end_idx": None,
        "max_prompts": max_prompts,
        "seed": seed_base,
        "seeds_per_prompt": 1,
        "fps": fps,
        "duration_sec_requested": duration_sec_requested,
        "target_latent_frames": first_target_latent_frames,
        "target_frames": first_total_frames,
        "effective_duration_sec": (float(first_total_frames) / float(fps)) if first_total_frames is not None and fps > 0 else None,
        "resolution": first_resolution,
        "source_moviegen_run_root": source_moviegen_run_root,
    }
    write_json(run_root / "summary" / "config.json", config_payload)
    return config_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run StoryEval parity for a combined-workspace method using canonical StoryEval output layout."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Target run directory under results/benchmarks/storyeval/")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config-id", type=str, default=None)
    parser.add_argument("--source-moviegen-run-root", type=str, default=None)
    parser.add_argument("--infer-python", type=Path, default=DEFAULT_INFER_PYTHON)
    parser.add_argument("--eval-python", type=Path, default=DEFAULT_EVAL_PYTHON)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--num-output-frames", type=int, default=42)
    parser.add_argument("--duration-sec-requested", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--profile-flowcache", action="store_true")
    parser.add_argument("--flowcache-profile-recent-ratio", type=float, default=None)
    parser.add_argument("--flowcache-profile-min-scale", type=float, default=0.7)
    parser.add_argument("--flowcache-profile-max-scale", type=float, default=1.3)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-vbench", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip generation when expected videos already exist.")
    parser.add_argument("--force", action="store_true", help="Delete the existing run root before regenerating.")
    parser.add_argument(
        "generate_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to scripts/01_generate.py after '--'.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    passthrough = list(args.generate_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    passthrough = [
        arg
        for arg in passthrough
        if arg not in {"--resume", "--force", "--skip-generate", "--skip-vbench", "--skip-drift", "--skip-summary"}
    ]

    run_root = args.run_root if args.run_root.is_absolute() else (REPO_ROOT / args.run_root)
    prompt_file = args.prompt_file if args.prompt_file.is_absolute() else (REPO_ROOT / args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(f"StoryEval prompt file not found: {prompt_file}")

    if args.force and run_root.exists():
        remove_path(run_root)

    for sub in ("logs", "metrics", "plots", "summary", "per_prompt", "videos"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    expected_prompts = expected_storyeval_prompts(prompt_file, max_prompts=args.max_prompts)
    method_video_dir = run_root / "videos" / args.method
    generation_log_path = run_root / "logs" / f"generation_{args.method}.jsonl"
    generation_ready = (
        method_video_dir.exists()
        and count_generated_videos(method_video_dir) == len(expected_prompts)
        and generation_log_path.exists()
    )

    effective_passthrough = list(passthrough)
    if "--low-memory" not in effective_passthrough:
        effective_passthrough.append("--low-memory")
    if args.profile_flowcache:
        layer_budget_path = run_root / "metrics" / "flowcache_profile_layer_budget.json"
        if "--flowcache-layer-budget-path" not in effective_passthrough:
            effective_passthrough.extend(["--flowcache-layer-budget-path", str(layer_budget_path)])

    if not args.skip_generate:
        if not (args.resume and generation_ready):
            if args.profile_flowcache:
                profile_efficiency_path = run_root / "profile_pass" / "metrics" / "efficiency_FLOWCACHE_PROFILE.json"
                profile_ready = profile_efficiency_path.exists()
                if not (args.resume and profile_ready):
                    profile_cmd = build_flowcache_profile_cmd(
                        run_root,
                        prompt_file,
                        effective_passthrough,
                        infer_python=args.infer_python,
                        max_prompts=args.max_prompts,
                        num_output_frames=args.num_output_frames,
                        seed=args.seed,
                        fps=args.fps,
                        recent_ratio=args.flowcache_profile_recent_ratio or 0.25,
                    )
                    run_cmd(profile_cmd)
                write_flowcache_layer_budget(
                    profile_efficiency_path,
                    run_root / "metrics" / "flowcache_profile_layer_budget.json",
                    min_scale=args.flowcache_profile_min_scale,
                    max_scale=args.flowcache_profile_max_scale,
                )

            generate_cmd = build_generate_cmd(
                run_root,
                args.method,
                prompt_file,
                effective_passthrough,
                infer_python=args.infer_python,
                max_prompts=args.max_prompts,
                num_output_frames=args.num_output_frames,
                seed=args.seed,
                fps=args.fps,
            )
            run_cmd(generate_cmd)
        else:
            print(f"Skipping generation for {args.method}: found {len(expected_prompts)} existing StoryEval videos.")

    config_payload = rewrite_storyeval_layout(
        run_root,
        run_name=args.run_name,
        method=args.method,
        config_id=args.config_id,
        prompt_file=prompt_file,
        max_prompts=args.max_prompts,
        fps=args.fps,
        duration_sec_requested=args.duration_sec_requested,
        seed_base=args.seed,
        source_moviegen_run_root=args.source_moviegen_run_root,
        force=args.force,
    )

    run_meta = {
        "run_name": args.run_name,
        "run_id": args.run_name,
        "run_timestamp_unix": int(time.time()),
        "run_root": str(run_root),
        "benchmark": "storyeval",
        "experiment_type": "combined_storyeval_backfill",
        "method": args.method,
        "config_id": args.config_id,
        "source_moviegen_run_root": args.source_moviegen_run_root,
        "profiled_flowcache": bool(args.profile_flowcache),
        "prompt_file": str(prompt_file),
        "target_max_prompts": args.max_prompts,
        "target_num_output_frames": args.num_output_frames,
        "target_seed_base": args.seed,
    }
    write_json(run_root / "run_meta.json", run_meta)

    if not args.skip_vbench:
        env = os.environ.copy()
        env["MASTER_PORT"] = deterministic_master_port(f"{run_root}:{args.method}:storyeval_vbench")
        run_cmd(
            [
                str(args.eval_python),
                str(REPO_ROOT / "scripts" / "eval_storyeval_vbench.py"),
                "--run_dir",
                str(run_root),
                "--python_bin",
                str(args.eval_python),
            ],
            env=env,
        )

    if not args.skip_drift:
        env = os.environ.copy()
        env["MASTER_PORT"] = deterministic_master_port(f"{run_root}:{args.method}:storyeval_drift")
        run_cmd(
            [
                str(args.eval_python),
                str(REPO_ROOT / "scripts" / "eval_storyeval_drift.py"),
                "--run_dir",
                str(run_root),
                "--python_bin",
                str(args.eval_python),
            ],
            env=env,
        )

    if not args.skip_summary:
        run_cmd(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "summarize_storyeval.py"),
                "--run_dir",
                str(run_root),
            ]
        )

    print(f"StoryEval backfill completed for {args.method} under {run_root}")
    print(json.dumps(config_payload, indent=2))


if __name__ == "__main__":
    main()
