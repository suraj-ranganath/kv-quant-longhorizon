# QVG Baseline Replication on Self-Forcing-Wan-1.3B

## 1) Project Summary
This repository is a replication-first project for the **Self-Forcing-Wan-1.3B** slice of the QVG paper.

Primary objective:
- Faithfully reproduce baseline KV-cache quantization comparisons on **Self-Forcing-Wan-1.3B**.
- Baseline-first order: **BF16 -> RTN -> KIVI -> QuaRot-KV-only**.
- Only after baseline pipeline and benchmarking are complete, start **QVG/QVG-Pro** design and implementation.

## 2) Reference Links
- QVG paper (arXiv): https://arxiv.org/abs/2602.02958
- QVG HTML: https://arxiv.org/html/2602.02958v2
- Self-Forcing repo: https://github.com/guandeh17/Self-Forcing
- Self-Forcing project page: https://self-forcing.github.io/
- VBench repo: https://github.com/Vchitect/VBench
- VBench project page: https://vchitect.github.io/VBench-project/
- flash-kmeans repo: https://github.com/svg-project/flash-kmeans
- KIVI repo: https://github.com/jy-yuan/KIVI
- KIVI paper: https://proceedings.mlr.press/v235/liu24bz.html
- QuaRot repo: https://github.com/spcl/QuaRot
- QuaRot paper: https://arxiv.org/abs/2404.00456

## 3) Experimental Scope
In-scope replication target:
- Model family slice: **Self-Forcing-Wan-1.3B** (base weights: **Wan2.1-T2V-1.3B**).
- Prompt suite: `prompts/MovieGenVideoBench_extended.txt` from Self-Forcing flow.
- Target output resolution: **480p** (smaller smoke tests allowed for debugging only).
- Generation mode: preserve official Self-Forcing chunk-wise autoregressive behavior.
- Modifications allowed: KV-cache quantization boundary only.

Out of scope for initial phase:
- LongCat-13B replication.
- HY-WorldPlay-8B replication.

Methods to reproduce in order:
- BF16 baseline.
- RTN (INT4/INT2, block size 16).
- KIVI (INT4/INT2 if feasible, block size 16, asymmetric key/value treatment).
- QuaRot-KV-only (INT4 priority; INT2 optional if principled).
- Then QVG / QVG-Pro planning and implementation.

## 4) Hardware Assumptions
Target hardware:
- **1x NVIDIA RTX A5000 24GB**.

Expected constraints vs H100-class setups:
- Lower throughput and potentially tighter memory headroom.
- Long-horizon runs (up to 700 frames) may take significantly longer.
- Some sweeps may require staged smoke tests before full-scale execution.

Policy:
- Prioritize correctness and faithful behavior over aggressive optimization.

## 5) Environment Plan (Two Environments)
To reduce dependency conflicts, maintain separate environments.

### Inference environment
Purpose:
- Self-Forcing generation and KV-cache quantization baselines.

Expected packages:
- PyTorch + CUDA build compatible with local driver.
- flash-attn (if supported in the local stack).
- Self-Forcing dependencies.
- LPIPS/SSIM/PSNR utility dependencies needed during generation-side checks.
- Optional flash-kmeans (primarily for later QVG acceleration experiments).

### Evaluation environment
Purpose:
- VBench metric evaluation and tooling that may conflict with inference stack.

Expected packages:
- VBench dependencies.
- detectron2 and pinned transitive dependencies required by VBench.

## 6) Benchmark Plan
All baselines are evaluated against BF16 reference.

### Fidelity vs BF16
- PSNR
- SSIM
- LPIPS

### Perceptual quality (VBench)
- background_consistency
- imaging_quality
- subject_consistency
- aesthetic_quality

### Systems / efficiency
- KV-cache compression ratio.
- End-to-end latency overhead.
- Peak GPU memory (if measurable reliably in this environment).
- Quantize/dequantize time breakdown.

### Long-horizon drift
- `imaging_quality` sampled every 50 frames.
- Stretch target: up to 700 frames if feasible on A5000.

## 7) Method Roadmap (Non-Negotiable Order)
1. **Step 1: BF16 baseline**
2. **Step 2: RTN**
3. **Step 3: KIVI**
4. **Step 4: QuaRot-KV-only**
5. **Step 5: Benchmark baseline suite end-to-end**
6. **Step 6: QVG design / implementation**

QVG implementation is blocked until baseline suite is green and benchmarked.

## 8) Deliverables
- Runnable generation scripts for BF16 and baseline quantization methods.
- Baseline KV quantizer implementations: RTN, KIVI, QuaRot-KV-only.
- Evaluation scripts for fidelity, VBench, efficiency, and drift curve.
- Saved result artifacts:
  - videos
  - raw metrics JSON
  - summary CSV/Markdown tables
  - plots
- Replication notes documenting deviations, assumptions, and A5000 constraints.

## 9) Known Risks / Deviations
- No public QVG reference implementation; design must be inferred from paper text.
- A5000 is slower and smaller-memory than H100; long runs can be expensive.
- flash-kmeans compatibility may require fallback path.
- VBench dependency stack likely needs a separate environment.
- 700-frame long-horizon experiments may need staged execution or reduced batch parallelism.

## 10) Planned Repository Layout
```text
.
├── README.md
├── third_party/
│   ├── Self-Forcing/
│   └── VBench/
├── checkpoints/
├── wan_models/
├── prompts/
├── kv_quant/
│   ├── __init__.py
│   ├── base.py
│   ├── rtn.py
│   ├── kivi.py
│   ├── quarot_kv.py
│   ├── qvg.py
│   ├── packing.py
│   ├── utils.py
│   └── metrics.py
├── scripts/
│   ├── 00_env_info.sh
│   ├── 01_generate.py
│   ├── 02_eval_fidelity.py
│   ├── 03_eval_vbench.sh
│   ├── 04_eval_drift_curve.py
│   └── 05_summarize_results.py
├── results/
│   ├── videos/
│   ├── metrics/
│   ├── logs/
│   ├── tables/
│   └── plots/
└── docs/
```

## 11) Setup and Execution Plan
### Phase 0: Documentation gate
- Update README with full plan.
- Commit README.
- Push branch.

### Phase 1: Repo/environment setup
- Create scaffold directories and baseline script entry points.
- Clone Self-Forcing and VBench into `third_party/`.
- Prepare separate inference/eval environments.

### Phase 2: BF16 baseline first
- Run official Self-Forcing generation path at target settings.
- Save outputs as `results/videos/BF16/prompt_{id}_seed_{seed}.mp4`.
- Log metadata per run:
  - prompt_id
  - seed
  - model config
  - checkpoint path
  - git commit hash
  - frame count
  - resolution
  - wall-clock runtime
  - peak VRAM (if available)

### Phase 3: KV quantization abstraction
- Introduce `KVQuantizer` interface:
  - `name()`
  - `quantize_kv(k, v, meta)`
  - `dequantize_kv(state, meta)`
  - `memory_bytes(state)`
- Hook quantization only where cache is appended/read.

### Phase 4: Baseline implementations
- RTN first (INT4/INT2, block=16).
- KIVI second (asymmetric keys/values, block=16).
- QuaRot-KV-only third (INT4 priority; INT2 optional).

### Phase 5: Evaluation harness
- `scripts/02_eval_fidelity.py`: PSNR/SSIM/LPIPS.
- `scripts/03_eval_vbench.sh`: selected VBench dimensions.
- Efficiency logging: runtime, memory, quant/dequant time, cache bytes, compression ratio.

### Phase 6: Required run matrix
Minimum methods:
- BF16
- RTN_INT4
- RTN_INT2
- KIVI_INT4
- KIVI_INT2
- QUAROT_KV_INT4
- QUAROT_KV_INT2 (if implemented)

Workflow:
- Smoke test on 3-5 prompts.
- Scale to full MovieGen prompt list.
- Fixed seeds.
- Aggregate results into:
  - `results/tables/baseline_summary.csv`
  - `results/tables/baseline_summary.md`

### Phase 7: QVG gate
Do not start QVG implementation until all baseline criteria are complete:
- BF16/RTN/KIVI/QuaRot-KV-only are functioning.
- Fidelity and VBench metrics are produced for all baseline methods.
- Efficiency metrics and summary tables exist.

Only then:
- Add `kv_quant/qvg.py` implementation plan/stub.
- Lock hook points and design choices.
- Implement QVG correctness-first.

## 12) Git Workflow
Minimum checkpoint commits:
1. README plan commit.
2. BF16 baseline integration.
3. RTN.
4. KIVI.
5. QuaRot-KV-only.
6. Evaluation harness.
7. Baseline summary and replication notes.
8. QVG planning stub.

Push after each meaningful checkpoint.

Suggested initial commit message:
- `docs: add full replication plan for QVG on Self-Forcing-Wan-1.3B`
