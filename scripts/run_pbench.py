#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.embodied.evaluator import (
    BinaryQAEvaluator,
    HeuristicEvaluator,
    ManualEvaluator,
    VLMRemoteEvaluator,
)
from benchmarks.embodied.io import (
    SelfForcingConfig,
    SelfForcingGenerator,
    generate_video_self_forcing,
    load_json,
    summarize_per_sample_records,
    write_json,
)
from benchmarks.embodied.pbench import load_pbench_samples, save_normalized_samples
from benchmarks.embodied.types import RunRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PBench evaluation using Self-Forcing-Wan-1.3B.")
    parser.add_argument("--run_id", type=str, default=str(int(time.time())))
    parser.add_argument("--out_root", type=Path, default=REPO_ROOT / "results" / "pbench")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_per_sample", type=int, default=1)
    parser.add_argument("--evaluator", choices=["heuristic", "manual", "vlm_remote"], default="heuristic")
    parser.add_argument("--heuristic_mode", choices=["always_false", "random"], default="always_false")
    parser.add_argument("--sf_config_path", type=Path, default=REPO_ROOT / "third_party" / "Self-Forcing" / "configs" / "self_forcing_dmd.yaml")
    parser.add_argument("--sf_default_config_path", type=Path, default=REPO_ROOT / "third_party" / "Self-Forcing" / "configs" / "default_config.yaml")
    parser.add_argument("--sf_checkpoint_path", type=Path, default=REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--prompt_override", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    parser.add_argument("--vlm_endpoint", type=str, default=None)
    parser.add_argument("--vlm_api_key_env", type=str, default="VLM_API_KEY")
    return parser


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_pbench")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_evaluator(args: argparse.Namespace, run_root: Path) -> BinaryQAEvaluator:
    if args.evaluator == "heuristic":
        return HeuristicEvaluator(mode=args.heuristic_mode, seed=args.seed)
    if args.evaluator == "manual":
        return ManualEvaluator(queue_dir=run_root / "manual_queue")
    if args.evaluator == "vlm_remote":
        if not args.vlm_endpoint:
            raise ValueError("--vlm_endpoint is required when --evaluator=vlm_remote")
        api_key = os.environ.get(args.vlm_api_key_env)
        return VLMRemoteEvaluator(
            endpoint=args.vlm_endpoint,
            api_key=api_key,
            trace_dir=run_root / "evaluator_traces",
        )
    raise ValueError(f"Unsupported evaluator: {args.evaluator}")


def _record_domain(meta: dict[str, Any]) -> str:
    for key in ("domain", "task", "type", "category"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sample_seed(base_seed: int, sample_idx: int, seed_idx: int) -> int:
    return int(base_seed + sample_idx * 1000 + seed_idx)


def main() -> None:
    args = build_parser().parse_args()

    run_root = args.out_root / args.run_id
    videos_dir = run_root / "videos"
    per_sample_dir = run_root / "per_sample"
    run_root.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_root / "logs.txt")
    logger.info("Starting PBench run_id=%s", args.run_id)

    hf_token = os.environ.get(args.hf_token_env)
    try:
        samples, split_used = load_pbench_samples(
            split=args.split,
            max_samples=args.max_samples,
            cache_root=REPO_ROOT / "data_cache" / "pbench",
            hf_token=hf_token,
        )
    except Exception as exc:
        logger.error("Failed to load PBench dataset: %s", exc)
        logger.error("If the dataset is gated, request access and run `huggingface-cli login` in the inference env.")
        raise SystemExit(1)
    if args.prompt_override:
        for s in samples:
            s.prompt = args.prompt_override

    normalized_cache = REPO_ROOT / "data_cache" / "pbench" / f"normalized_{split_used}.jsonl"
    save_normalized_samples(samples, normalized_cache)
    save_normalized_samples(samples, run_root / "normalized_samples.jsonl")

    evaluator = build_evaluator(args, run_root)
    sf_cfg = SelfForcingConfig(
        config_path=args.sf_config_path,
        default_config_path=args.sf_default_config_path,
        checkpoint_path=args.sf_checkpoint_path,
        use_ema=args.use_ema,
        device=args.device,
        num_output_frames=args.num_output_frames,
        fps=args.fps,
        low_memory=args.low_memory,
    )
    generator = SelfForcingGenerator(sf_cfg)

    config_payload = {
        "run_id": args.run_id,
        "split": split_used,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "seeds_per_sample": args.seeds_per_sample,
        "evaluator": args.evaluator,
        "heuristic_mode": args.heuristic_mode,
        "sf_config_path": str(args.sf_config_path),
        "sf_default_config_path": str(args.sf_default_config_path),
        "sf_checkpoint_path": str(args.sf_checkpoint_path),
        "num_output_frames": args.num_output_frames,
        "fps": args.fps,
        "device": args.device,
        "use_ema": args.use_ema,
        "low_memory": args.low_memory,
        "created_at_unix": int(time.time()),
    }
    write_json(run_root / "config.json", config_payload)

    logger.info("Loaded %d samples from split=%s", len(samples), split_used)

    records: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        for seed_idx in range(args.seeds_per_sample):
            seed = _sample_seed(args.seed, sample_idx, seed_idx)
            stem = f"{sample.sample_id}_seed{seed}"
            video_path = videos_dir / f"{stem}.mp4"
            sample_json_path = per_sample_dir / f"{stem}.json"

            if args.resume and sample_json_path.exists():
                existing = load_json(sample_json_path)
                if isinstance(existing, dict):
                    records.append(existing)
                    logger.info("[resume] %s", sample_json_path.name)
                    continue

            prompt = sample.prompt
            record = {
                "sample_id": sample.sample_id,
                "seed": seed,
                "prompt": prompt,
                "cond_image_path": sample.cond_image_path,
                "generated_video_path": str(video_path),
                "qa_results": [],
                "runtime": math.nan,
                "runtime_s": math.nan,
                "peak_vram": None,
                "peak_vram_bytes": None,
                "errors": [],
                "meta": {**sample.meta, "domain": _record_domain(sample.meta)},
            }

            try:
                gen_meta = generate_video_self_forcing(
                    prompt=prompt,
                    cond_image_path=sample.cond_image_path,
                    seed=seed,
                    out_path=str(video_path),
                    config={
                        "config_path": str(args.sf_config_path),
                        "default_config_path": str(args.sf_default_config_path),
                        "checkpoint_path": str(args.sf_checkpoint_path),
                        "use_ema": args.use_ema,
                        "device": args.device,
                        "num_output_frames": args.num_output_frames,
                        "fps": args.fps,
                        "low_memory": args.low_memory,
                    },
                    generator=generator,
                )
                record["runtime"] = gen_meta["runtime_s"]
                record["runtime_s"] = gen_meta["runtime_s"]
                record["peak_vram"] = gen_meta["peak_vram_bytes"]
                record["peak_vram_bytes"] = gen_meta["peak_vram_bytes"]
                record["generation_meta"] = gen_meta

                for qa in sample.qa_pairs:
                    question = str(qa.get("question", "")).strip()
                    if not question:
                        continue
                    gt_answer = bool(qa.get("answer", False))
                    eval_result = evaluator.predict_with_trace(
                        video_path=str(video_path),
                        prompt=prompt,
                        cond_image_path=sample.cond_image_path,
                        question=question,
                    )
                    pred_answer = None if eval_result.pending else bool(eval_result.pred_answer)
                    qa_record = {
                        "question": question,
                        "gt_answer": gt_answer,
                        "pred_answer": pred_answer,
                        "correct": (pred_answer == gt_answer) if pred_answer is not None else None,
                        "pending": bool(eval_result.pending),
                        "evaluator_trace_path": eval_result.trace_path,
                    }
                    if eval_result.meta:
                        qa_record["evaluator_meta"] = eval_result.meta
                    record["qa_results"].append(qa_record)
            except Exception as exc:
                record["errors"].append(str(exc))
                logger.exception("Failed sample=%s seed=%s", sample.sample_id, seed)

            write_json(sample_json_path, record)
            records.append(record)
            logger.info(
                "done sample=%s seed=%d qa=%d errors=%d",
                sample.sample_id,
                seed,
                len(record["qa_results"]),
                len(record["errors"]),
            )

    summary = summarize_per_sample_records(records, run_id=args.run_id, config=config_payload)
    write_json(run_root / "summary.json", summary)

    run_record = RunRecord(
        run_id=args.run_id,
        method="self_forcing_wan_1.3b",
        config=config_payload,
        per_sample_dir=str(per_sample_dir),
        videos_dir=str(videos_dir),
        summary_path=str(run_root / "summary.json"),
    )
    write_json(run_root / "run_record.json", run_record.__dict__)

    logger.info("Run complete: %s", args.run_id)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
