#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_RUN_ROOT = Path(
    "/data/vaishak/kv-quant-longhorizon/results/runs/1772751420_baseline10s_10prompts_v3"
)
DEFAULT_INFER_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python")
DEFAULT_EVAL_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def count_video_outputs(video_dir: Path) -> int:
    if not video_dir.exists():
        return 0
    return sum(1 for _ in video_dir.glob("prompt_*_seed_*.mp4"))


def expected_video_outputs(prompt_file: Path, *, max_prompts: int, num_samples: int) -> int:
    prompts = [
        line.rstrip("\n")
        for line in prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return min(max_prompts, len(prompts)) * num_samples


def deterministic_master_port(namespace: str) -> str:
    digest = hashlib.sha1(namespace.encode("utf-8")).hexdigest()
    return str(15000 + (int(digest[:8], 16) % 20000))


def link_baseline_artifacts(run_root: Path, baseline_run_root: Path, methods: list[str]) -> None:
    for method in methods:
        src_video_dir = baseline_run_root / "videos" / method
        if src_video_dir.exists():
            safe_symlink(src_video_dir, run_root / "videos" / method)

        for prefix in ("generation", "vram_trace"):
            src = baseline_run_root / "logs" / f"{prefix}_{method}.jsonl"
            if src.exists():
                safe_symlink(src, run_root / "logs" / src.name)

        for prefix in ("efficiency", "fidelity", "vbench", "drift"):
            src = baseline_run_root / "metrics" / f"{prefix}_{method}.json"
            if src.exists():
                safe_symlink(src, run_root / "metrics" / src.name)

        vbench_dir = baseline_run_root / "metrics" / f"vbench_{method}"
        if vbench_dir.exists():
            safe_symlink(vbench_dir, run_root / "metrics" / vbench_dir.name)


def list_methods_in_run(run_root: Path) -> list[str]:
    methods: set[str] = set()
    for metric_dir in [run_root / "metrics"]:
        if not metric_dir.exists():
            continue
        for prefix in ("efficiency", "fidelity", "vbench", "drift"):
            for path in metric_dir.glob(f"{prefix}_*.json"):
                methods.add(path.stem[len(prefix) + 1 :])
    videos_root = run_root / "videos"
    if videos_root.exists():
        for path in videos_root.iterdir():
            if path.is_dir():
                methods.add(path.name)
    return sorted(methods)


def write_or_update_run_meta(
    run_root: Path,
    *,
    run_name: str,
    config_id: str | None,
    method: str,
    baseline_run_root: Path,
    extra_meta: dict[str, Any],
) -> None:
    run_meta_path = run_root / "run_meta.json"
    payload: dict[str, Any] = {}
    if run_meta_path.exists():
        try:
            payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if not payload:
        payload = {
            "run_name": run_name,
            "run_id": run_root.name,
            "run_timestamp_unix": int(time.time()),
            "run_root": str(run_root),
            "experiment_type": "combined_backfill",
        }
    payload.setdefault("linked_baseline_run_root", str(baseline_run_root))
    payload.setdefault("backfilled_methods", [])
    if method not in payload["backfilled_methods"]:
        payload["backfilled_methods"].append(method)
    if config_id:
        payload.setdefault("backfill_config_ids", [])
        if config_id not in payload["backfill_config_ids"]:
            payload["backfill_config_ids"].append(config_id)
    payload.update(extra_meta)
    run_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_generate_cmd(
    run_root: Path,
    method: str,
    passthrough: list[str],
    *,
    infer_python: Path,
    max_prompts: int,
    num_output_frames: int,
    seed: int,
) -> list[str]:
    cmd = [
        str(infer_python),
        str(REPO_ROOT / "scripts" / "01_generate.py"),
        "--results-root",
        str(run_root),
        "--method",
        method,
        "--max-prompts",
        str(max_prompts),
        "--num-output-frames",
        str(num_output_frames),
        "--seed",
        str(seed),
    ]
    cmd.extend(passthrough)
    return cmd


def build_flowcache_profile_cmd(
    run_root: Path,
    passthrough: list[str],
    *,
    infer_python: Path,
    max_prompts: int,
    num_output_frames: int,
    seed: int,
    recent_ratio: float,
) -> list[str]:
    cmd = [
        str(infer_python),
        str(REPO_ROOT / "scripts" / "01_generate.py"),
        "--results-root",
        str(run_root / "profile_pass"),
        "--method",
        "FLOWCACHE_PROFILE",
        "--max-prompts",
        str(max_prompts),
        "--num-output-frames",
        str(num_output_frames),
        "--seed",
        str(seed),
        "--flowcache-recent-ratio",
        str(recent_ratio),
        "--device",
        "cuda:0",
        "--use-ema",
    ]
    cmd.extend(passthrough)
    return cmd


def write_flowcache_layer_budget(
    profile_efficiency_path: Path,
    output_path: Path,
    *,
    min_scale: float,
    max_scale: float,
) -> None:
    payload = json.loads(profile_efficiency_path.read_text(encoding="utf-8"))
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single MovieGen backfill config inside a combined-workspace run root.")
    parser.add_argument("--run-root", type=Path, required=True, help="Target run directory, e.g. results/runs/<ts>_<name>")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config-id", type=str, default=None)
    parser.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_RUN_ROOT)
    parser.add_argument("--infer-python", type=Path, default=DEFAULT_INFER_PYTHON)
    parser.add_argument("--eval-python", type=Path, default=DEFAULT_EVAL_PYTHON)
    parser.add_argument("--link-method", action="append", default=["BF16", "RTN_INT4"], help="Existing methods to symlink from the baseline run.")
    parser.add_argument("--bf16-reference-method", type=str, default="BF16")
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--num-output-frames", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-flowcache", action="store_true")
    parser.add_argument("--flowcache-profile-recent-ratio", type=float, default=None)
    parser.add_argument("--flowcache-profile-min-scale", type=float, default=0.7)
    parser.add_argument("--flowcache-profile-max-scale", type=float, default=1.3)
    parser.add_argument("--compute-drift", action="store_true", help="Also run 04_eval_drift_curve.py after VBench.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-vbench", action="store_true")
    parser.add_argument("--skip-fidelity", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing method artifacts instead of skipping generation.")
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

    run_root = args.run_root if args.run_root.is_absolute() else (REPO_ROOT / args.run_root)
    baseline_run_root = args.baseline_run_root
    if not baseline_run_root.exists():
        raise FileNotFoundError(f"Baseline run root not found: {baseline_run_root}")

    for sub in ("videos", "logs", "metrics", "tables", "plots"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    link_baseline_artifacts(run_root, baseline_run_root, args.link_method)

    candidate_video_dir = run_root / "videos" / args.method
    candidate_generation_log = run_root / "logs" / f"generation_{args.method}.jsonl"
    candidate_vram_trace_log = run_root / "logs" / f"vram_trace_{args.method}.jsonl"
    candidate_efficiency_json = run_root / "metrics" / f"efficiency_{args.method}.json"
    candidate_vbench_json = run_root / "metrics" / f"vbench_{args.method}.json"
    candidate_vbench_dir = run_root / "metrics" / f"vbench_{args.method}"
    candidate_fidelity_json = run_root / "metrics" / f"fidelity_{args.method}.json"
    candidate_drift_json = run_root / "metrics" / f"drift_{args.method}.json"
    flowcache_layer_budget_path = run_root / "metrics" / "flowcache_profile_layer_budget.json"
    flowcache_profile_efficiency_path = run_root / "profile_pass" / "metrics" / "efficiency_FLOWCACHE_PROFILE.json"
    prompt_file = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
    expected_outputs = expected_video_outputs(
        prompt_file,
        max_prompts=args.max_prompts,
        num_samples=1,
    )

    if args.force:
        if candidate_video_dir.exists() and not candidate_video_dir.is_symlink():
            shutil.rmtree(candidate_video_dir)
        for path in (
            candidate_generation_log,
            candidate_vram_trace_log,
            candidate_efficiency_json,
            candidate_vbench_json,
            candidate_fidelity_json,
            candidate_drift_json,
            flowcache_layer_budget_path,
            candidate_vbench_dir,
        ):
            if path.exists() or path.is_symlink():
                remove_path(path)
        profile_root = run_root / "profile_pass"
        if profile_root.exists() and not profile_root.is_symlink():
            shutil.rmtree(profile_root)

    if args.profile_flowcache and not flowcache_layer_budget_path.exists():
        if args.flowcache_profile_recent_ratio is None:
            raise ValueError("--flowcache-profile-recent-ratio is required when --profile-flowcache is set")
        run_cmd(
            build_flowcache_profile_cmd(
                run_root,
                passthrough,
                infer_python=args.infer_python,
                max_prompts=args.max_prompts,
                num_output_frames=args.num_output_frames,
                seed=args.seed,
                recent_ratio=args.flowcache_profile_recent_ratio,
            )
        )
        if not flowcache_profile_efficiency_path.exists():
            raise FileNotFoundError(
                f"Expected FlowCache profile efficiency output at {flowcache_profile_efficiency_path}"
            )
        write_flowcache_layer_budget(
            flowcache_profile_efficiency_path,
            flowcache_layer_budget_path,
            min_scale=args.flowcache_profile_min_scale,
            max_scale=args.flowcache_profile_max_scale,
        )

    if args.profile_flowcache and "--flowcache-layer-budget-path" not in passthrough:
        passthrough.extend(["--flowcache-layer-budget-path", str(flowcache_layer_budget_path)])

    current_video_outputs = count_video_outputs(candidate_video_dir)
    if not args.force and current_video_outputs and current_video_outputs < expected_outputs:
        print(
            f"Incomplete video outputs detected for {args.method}: "
            f"{current_video_outputs}/{expected_outputs}. Regenerating cleanly."
        )
        if candidate_video_dir.exists() and not candidate_video_dir.is_symlink():
            shutil.rmtree(candidate_video_dir)
        for path in (
            candidate_generation_log,
            candidate_vram_trace_log,
            candidate_efficiency_json,
            candidate_vbench_json,
            candidate_fidelity_json,
            candidate_drift_json,
            candidate_vbench_dir,
        ):
            if path.exists() or path.is_symlink():
                remove_path(path)
        current_video_outputs = 0

    if not args.skip_generate and current_video_outputs < expected_outputs:
        cmd = build_generate_cmd(
            run_root,
            args.method,
            passthrough,
            infer_python=args.infer_python,
            max_prompts=args.max_prompts,
            num_output_frames=args.num_output_frames,
            seed=args.seed,
        )
        run_cmd(cmd)
        current_video_outputs = count_video_outputs(candidate_video_dir)
        if current_video_outputs < expected_outputs:
            raise FileNotFoundError(
                f"Expected {expected_outputs} generated videos for {args.method}, "
                f"but found {current_video_outputs} in {candidate_video_dir}"
            )

    bf16_dir = baseline_run_root / "videos" / args.bf16_reference_method
    if not args.skip_fidelity and args.method != args.bf16_reference_method and not candidate_fidelity_json.exists():
        run_cmd(
            [
                str(args.infer_python),
                str(REPO_ROOT / "scripts" / "02_eval_fidelity.py"),
                "--bf16-dir",
                str(bf16_dir),
                "--candidate-dir",
                str(candidate_video_dir),
                "--output",
                str(candidate_fidelity_json),
                "--device",
                "cpu",
            ]
        )

    if not args.skip_vbench and not candidate_vbench_json.exists():
        env = os.environ.copy()
        env["RUN_ROOT"] = str(run_root)
        env["MASTER_PORT"] = deterministic_master_port(f"{run_root}:{args.method}:vbench")
        run_cmd(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "03_eval_vbench.sh"),
                args.method,
                str(candidate_video_dir),
                str(prompt_file),
                str(candidate_vbench_dir),
                str(candidate_vbench_json),
            ],
            env={**env, "PYTHON_BIN": str(args.eval_python)},
        )

    if args.compute_drift and not candidate_drift_json.exists():
        run_cmd(
            [
                str(args.eval_python),
                str(REPO_ROOT / "scripts" / "04_eval_drift_curve.py"),
                "--method",
                args.method,
                "--videos-dir",
                str(candidate_video_dir.relative_to(REPO_ROOT)),
                "--prompt-file",
                str((REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt").relative_to(REPO_ROOT)),
                "--output",
                str(candidate_drift_json.relative_to(REPO_ROOT)),
            ]
        )

    if not args.skip_summary:
        methods = list_methods_in_run(run_root)
        run_cmd(
            [
                str(args.infer_python),
                str(REPO_ROOT / "scripts" / "05_summarize_results.py"),
                "--results-root",
                str(run_root),
                "--methods",
                *methods,
            ]
        )

    write_or_update_run_meta(
        run_root,
        run_name=args.run_name,
        config_id=args.config_id,
        method=args.method,
        baseline_run_root=baseline_run_root,
        extra_meta={
            "target_num_output_frames": args.num_output_frames,
            "target_max_prompts": args.max_prompts,
            "target_seed_base": args.seed,
            "profiled_flowcache": args.profile_flowcache,
        },
    )

    print(f"Backfill completed for {args.method} under {run_root}")


if __name__ == "__main__":
    main()
