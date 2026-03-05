#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json_or_none(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_vbench_score(v):
    # VBench output format is usually: [aggregate_score, details]
    if isinstance(v, list) and v:
        return v[0]
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline metrics into CSV and Markdown tables.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["BF16", "RTN_INT4", "RTN_INT2", "KIVI_INT4", "KIVI_INT2", "QUAROT_KV_INT4", "QUAROT_KV_INT2"],
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/baseline_summary.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("tables/baseline_summary.md"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = args.results_root if args.results_root.is_absolute() else (root / args.results_root)
    metrics_dir = args.metrics_dir if args.metrics_dir.is_absolute() else (results_root / args.metrics_dir)
    out_csv = args.out_csv if args.out_csv.is_absolute() else (results_root / args.out_csv)
    out_md = args.out_md if args.out_md.is_absolute() else (results_root / args.out_md)
    rows = []

    for method in args.methods:
        fidelity = load_json_or_none(metrics_dir / f"fidelity_{method}.json")
        vbench = load_json_or_none(metrics_dir / f"vbench_{method}.json")
        efficiency = load_json_or_none(metrics_dir / f"efficiency_{method}.json")

        row = {
            "method": method,
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "background_consistency": None,
            "imaging_quality": None,
            "subject_consistency": None,
            "aesthetic_quality": None,
            "compression_ratio": None,
            "total_runtime_s": None,
            "peak_vram_bytes": None,
            "quantize_time_s": None,
            "dequantize_time_s": None,
        }

        if fidelity is not None:
            agg = fidelity.get("aggregate", {})
            row["psnr"] = agg.get("psnr")
            row["ssim"] = agg.get("ssim")
            row["lpips"] = agg.get("lpips")

        if vbench is not None:
            row["background_consistency"] = extract_vbench_score(vbench.get("background_consistency"))
            row["imaging_quality"] = extract_vbench_score(vbench.get("imaging_quality"))
            row["subject_consistency"] = extract_vbench_score(vbench.get("subject_consistency"))
            row["aesthetic_quality"] = extract_vbench_score(vbench.get("aesthetic_quality"))

        if efficiency is not None:
            row["compression_ratio"] = efficiency.get("compression_ratio")
            row["total_runtime_s"] = efficiency.get("total_runtime_s")
            row["peak_vram_bytes"] = efficiency.get("peak_vram_bytes")
            row["quantize_time_s"] = efficiency.get("quantize_time_s")
            row["dequantize_time_s"] = efficiency.get("dequantize_time_s")

        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| method | psnr | ssim | lpips | background_consistency | imaging_quality | subject_consistency | aesthetic_quality | compression_ratio | total_runtime_s | peak_vram_bytes | quantize_time_s | dequantize_time_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            "| {method} | {psnr} | {ssim} | {lpips} | {background_consistency} | {imaging_quality} | {subject_consistency} | {aesthetic_quality} | {compression_ratio} | {total_runtime_s} | {peak_vram_bytes} | {quantize_time_s} | {dequantize_time_s} |".format(
                **{k: ("" if v is None else v) for k, v in r.items()}
            )
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
