# Environment Setup

## Inference environment (`qvg_sf_infer`)

```bash
./scripts/10_clone_deps.sh
./scripts/11_apply_self_forcing_patch.sh
conda create -n qvg_sf_infer python=3.10 -y
conda activate qvg_sf_infer
pip install --upgrade pip
pip install -r requirements-inference.txt
pip install flash-attn --no-build-isolation
cd third_party/Self-Forcing && python setup.py develop && cd -
```

Checkpoint download:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir checkpoints
```

## Evaluation environment (`qvg_sf_eval`)

```bash
./scripts/10_clone_deps.sh
conda create -n qvg_sf_eval python=3.10 -y
conda activate qvg_sf_eval
pip install --upgrade pip
pip install -r requirements-eval.txt
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

Optional VBench local install path:

```bash
cd third_party/VBench
pip install -e .
cd -
```

## Dashboard environment (optional)

Use either `qvg_sf_infer` or a separate lightweight env:

```bash
conda create -n qvg_sf_dashboard python=3.10 -y
conda activate qvg_sf_dashboard
pip install --upgrade pip
pip install -r requirements-dashboard.txt
```

Launch:

```bash
./scripts/13_launch_dashboard.sh
```
