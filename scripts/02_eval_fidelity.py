#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torchvision.io import read_video

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-image is required for PSNR/SSIM. Install with `pip install scikit-image`.") from exc

try:
    import lpips
except Exception:
    lpips = None


def load_video(path: Path):
    frames, _, _ = read_video(str(path), pts_unit="sec")
    # frames: [T, H, W, C], uint8
    if frames.numel() == 0:
        raise RuntimeError(f"Empty video: {path}")
    return frames


def compute_lpips_per_frame(ref: torch.Tensor, cand: torch.Tensor, model) -> float:
    # Input frames uint8 [T, H, W, C]
    t = ref.shape[0]
    vals: List[float] = []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(t):
            r = ref[i].permute(2, 0, 1).float().to(model_device) / 127.5 - 1.0
            c = cand[i].permute(2, 0, 1).float().to(model_device) / 127.5 - 1.0
            vals.append(float(model(r.unsqueeze(0), c.unsqueeze(0)).item()))
    return float(np.mean(vals))


def evaluate_pair(ref_path: Path, cand_path: Path, lpips_model) -> Dict:
    ref = load_video(ref_path)
    cand = load_video(cand_path)

    if ref.shape[0] != cand.shape[0]:
        raise ValueError(f"Frame-count mismatch: {ref_path.name} ({ref.shape[0]}) vs {cand_path.name} ({cand.shape[0]})")
    if ref.shape[1:3] != cand.shape[1:3]:
        raise ValueError(f"Resolution mismatch: {ref_path.name} ({tuple(ref.shape[1:3])}) vs {cand_path.name} ({tuple(cand.shape[1:3])})")

    ref_np = ref.numpy()
    cand_np = cand.numpy()

    psnr_vals = []
    ssim_vals = []
    for i in range(ref_np.shape[0]):
        if np.array_equal(ref_np[i], cand_np[i]):
            psnr_vals.append(float("inf"))
        else:
            psnr_vals.append(peak_signal_noise_ratio(ref_np[i], cand_np[i], data_range=255))
        ssim_vals.append(structural_similarity(ref_np[i], cand_np[i], channel_axis=-1, data_range=255))

    result = {
        "video": cand_path.name,
        "num_frames": int(ref.shape[0]),
        "resolution": [int(ref.shape[1]), int(ref.shape[2])],
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }

    if lpips_model is None:
        result["lpips"] = None
    else:
        result["lpips"] = compute_lpips_per_frame(ref, cand, lpips_model)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fidelity metrics (PSNR/SSIM/LPIPS) against BF16.")
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--match-intersection",
        action="store_true",
        help="Evaluate only videos present in both directories instead of requiring the full BF16 set.",
    )
    args = parser.parse_args()

    bf16_videos = sorted(args.bf16_dir.glob("*.mp4"))
    cand_map = {p.name: p for p in sorted(args.candidate_dir.glob("*.mp4"))}
    if not bf16_videos:
        raise RuntimeError(f"No BF16 videos found in {args.bf16_dir}")
    bf16_map = {p.name: p for p in bf16_videos}

    missing_in_candidate = sorted(set(bf16_map) - set(cand_map))
    missing_in_bf16 = sorted(set(cand_map) - set(bf16_map))
    if args.match_intersection:
        matched_names = sorted(set(bf16_map) & set(cand_map))
        if not matched_names:
            raise RuntimeError(
                f"No overlapping videos between {args.bf16_dir} and {args.candidate_dir}"
            )
    else:
        if missing_in_candidate:
            raise FileNotFoundError(
                f"Missing candidate video(s) for {len(missing_in_candidate)} BF16 reference files; "
                f"first missing file: {missing_in_candidate[0]}"
            )
        matched_names = sorted(bf16_map)

    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net="alex")
        lpips_model = lpips_model.to(args.device)
        lpips_model.eval()

    per_video = []
    for video_name in matched_names:
        ref_path = bf16_map[video_name]
        cand_path = cand_map[video_name]
        metrics = evaluate_pair(ref_path, cand_path, lpips_model)
        per_video.append(metrics)

    psnr_vals = [x["psnr"] for x in per_video]
    agg = {
        "num_videos": len(per_video),
        "psnr": float(np.mean(psnr_vals)) if not any(math.isinf(v) for v in psnr_vals) else float("inf"),
        "ssim": float(np.mean([x["ssim"] for x in per_video])),
        "lpips": None,
    }
    lpips_vals = [x["lpips"] for x in per_video if x["lpips"] is not None]
    if lpips_vals:
        agg["lpips"] = float(np.mean(lpips_vals))

    output = {
        "bf16_dir": str(args.bf16_dir),
        "candidate_dir": str(args.candidate_dir),
        "match_policy": "intersection" if args.match_intersection else "strict_bf16_reference",
        "num_bf16_videos": len(bf16_map),
        "num_candidate_videos": len(cand_map),
        "matched_videos": matched_names,
        "missing_in_candidate": missing_in_candidate,
        "missing_in_bf16": missing_in_bf16,
        "aggregate": agg,
        "per_video": per_video,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
