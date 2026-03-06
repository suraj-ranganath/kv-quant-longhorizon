from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_FORCING_ROOT = REPO_ROOT / "third_party" / "Self-Forcing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SELF_FORCING_ROOT) not in sys.path:
    sys.path.insert(0, str(SELF_FORCING_ROOT))

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.misc import set_seed


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])  # noqa: S603
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _iter_kv_blocks(pipeline: Any) -> list[dict[str, Any]]:
    if hasattr(pipeline, "kv_cache1") and isinstance(pipeline.kv_cache1, list):
        return pipeline.kv_cache1
    if hasattr(pipeline, "kv_cache_pos") and isinstance(pipeline.kv_cache_pos, list):
        return pipeline.kv_cache_pos
    return []


def reset_kv_state(pipeline: Any) -> None:
    for block in _iter_kv_blocks(pipeline):
        if "global_end_index" in block:
            block["global_end_index"].fill_(0)
        if "local_end_index" in block:
            block["local_end_index"].fill_(0)
        if isinstance(block.get("k"), torch.Tensor) and block["k"].numel() > 0:
            block["k"].zero_()
        if isinstance(block.get("v"), torch.Tensor) and block["v"].numel() > 0:
            block["v"].zero_()

    if hasattr(pipeline, "crossattn_cache") and isinstance(pipeline.crossattn_cache, list):
        for block in pipeline.crossattn_cache:
            if "is_init" in block:
                block["is_init"] = False
    if hasattr(pipeline, "crossattn_cache_pos") and isinstance(pipeline.crossattn_cache_pos, list):
        for block in pipeline.crossattn_cache_pos:
            if "is_init" in block:
                block["is_init"] = False
    if hasattr(pipeline, "crossattn_cache_neg") and isinstance(pipeline.crossattn_cache_neg, list):
        for block in pipeline.crossattn_cache_neg:
            if "is_init" in block:
                block["is_init"] = False


def ensure_kv_cache_capacity(pipeline: Any, num_output_frames: int, dtype: torch.dtype, device: torch.device) -> None:
    frame_seq_length = int(getattr(pipeline, "frame_seq_length", 0))
    if frame_seq_length <= 0:
        return
    required_tokens = int(num_output_frames) * frame_seq_length
    for block in _iter_kv_blocks(pipeline):
        k = block.get("k")
        v = block.get("v")
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.ndim != 4 or v.ndim != 4:
            continue
        current_tokens = int(k.shape[1])
        if required_tokens <= current_tokens:
            continue
        batch_size, _, num_heads, head_dim = k.shape
        new_k = torch.zeros([batch_size, required_tokens, num_heads, head_dim], dtype=dtype, device=device)
        new_v = torch.zeros_like(new_k)
        if current_tokens > 0:
            new_k[:, :current_tokens] = k
            new_v[:, :current_tokens] = v
        block["k"] = new_k
        block["v"] = new_v


@dataclass
class SelfForcingConfig:
    config_path: Path
    default_config_path: Path
    checkpoint_path: Path
    use_ema: bool = True
    device: str = "cuda:0"
    num_output_frames: int = 21
    fps: int = 16
    low_memory: bool = False


class SelfForcingGenerator:
    def __init__(self, cfg: SelfForcingConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pipeline: Any | None = None
        self.low_memory_runtime = bool(cfg.low_memory)

    def _initialize_pipeline(self) -> Any:
        config = OmegaConf.load(str(self.cfg.default_config_path))
        config = OmegaConf.merge(config, OmegaConf.load(str(self.cfg.config_path)))
        if hasattr(config, "denoising_step_list"):
            pipeline = CausalInferencePipeline(config, device=self.device)
        else:
            pipeline = CausalDiffusionInferencePipeline(config, device=self.device)

        if not self.cfg.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.cfg.checkpoint_path}")
        state_dict = torch.load(str(self.cfg.checkpoint_path), map_location="cpu")
        key = "generator_ema" if self.cfg.use_ema else "generator"
        if key not in state_dict:
            raise KeyError(f"Checkpoint missing key `{key}`. Available keys: {list(state_dict.keys())[:10]}")
        pipeline.generator.load_state_dict(state_dict[key])

        pipeline = pipeline.to(dtype=torch.bfloat16)
        self.low_memory_runtime = self.cfg.low_memory or get_cuda_free_memory_gb(self.device) < 40
        if self.low_memory_runtime:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=self.device)
        else:
            pipeline.text_encoder.to(self.device)
        pipeline.generator.to(self.device)
        pipeline.vae.to(self.device)

        pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=self.device)
        ensure_kv_cache_capacity(
            pipeline,
            num_output_frames=self.cfg.num_output_frames,
            dtype=torch.bfloat16,
            device=self.device,
        )
        return pipeline

    def _get_pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = self._initialize_pipeline()
        return self.pipeline

    @staticmethod
    def _load_initial_latent_if_available(
        pipeline: Any,
        cond_image_path: str | None,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], str]:
        if not cond_image_path:
            return None, "no_conditioning_image"
        try:
            image = Image.open(cond_image_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize((480, 832)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            image_tensor = transform(image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            if not hasattr(pipeline, "vae") or not hasattr(pipeline.vae, "encode_to_latent"):
                return None, "conditioning_image_ignored_vae_encode_missing"
            initial_latent = pipeline.vae.encode_to_latent(image_tensor).to(device=device, dtype=torch.bfloat16)
            return initial_latent, "conditioning_image_applied"
        except Exception as exc:
            return None, f"conditioning_image_ignored_error:{exc}"

    @staticmethod
    def _align_noise_frames(
        noise_frames: int,
        num_frame_per_block: int,
        independent_first_frame: bool,
        has_initial_latent: bool,
    ) -> tuple[int, str | None]:
        if num_frame_per_block <= 1:
            return max(1, int(noise_frames)), None

        noise_frames = max(1, int(noise_frames))
        if has_initial_latent or not independent_first_frame:
            remainder = noise_frames % num_frame_per_block
            if remainder == 0:
                return noise_frames, None
            lower = noise_frames - remainder
            upper = lower + num_frame_per_block
            candidate = lower if lower > 0 else upper
            return int(candidate), f"noise_frames_aligned:{noise_frames}->{candidate}"

        remainder = (noise_frames - 1) % num_frame_per_block
        if remainder == 0:
            return noise_frames, None
        lower = noise_frames - remainder
        upper = lower + num_frame_per_block
        candidate = lower if lower > 0 else upper
        return int(candidate), f"noise_frames_aligned:{noise_frames}->{candidate}"

    def generate_video(
        self,
        prompt: str,
        cond_image_path: str | None,
        seed: int,
        out_path: Path,
    ) -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Self-Forcing generation.")
        pipeline = self._get_pipeline()
        set_seed(seed)
        reset_kv_state(pipeline)

        initial_latent, cond_note = self._load_initial_latent_if_available(
            pipeline=pipeline,
            cond_image_path=cond_image_path,
            device=self.device,
        )
        num_output_frames = int(self.cfg.num_output_frames)
        noise_frames = num_output_frames
        if initial_latent is not None:
            noise_frames = max(1, num_output_frames - int(initial_latent.shape[1]))

        num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
        independent_first_frame = bool(getattr(pipeline, "independent_first_frame", False))

        # Some Self-Forcing configs cannot accept single-frame i2v conditioning unless
        # `independent_first_frame` is enabled. Fall back to text-only and log it.
        if initial_latent is not None and (not independent_first_frame):
            num_input_frames = int(initial_latent.shape[1])
            if num_input_frames % num_frame_per_block != 0:
                initial_latent = None
                noise_frames = num_output_frames
                cond_note = (
                    f"{cond_note};conditioning_image_ignored_incompatible_num_input_frames:"
                    f"{num_input_frames}_mod_{num_frame_per_block}"
                )

        noise_frames, align_note = self._align_noise_frames(
            noise_frames=noise_frames,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            has_initial_latent=initial_latent is not None,
        )

        sampled_noise = torch.randn([1, noise_frames, 16, 60, 104], device=self.device, dtype=torch.bfloat16)

        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            kwargs = {
                "noise": sampled_noise,
                "text_prompts": [prompt],
                "return_latents": True,
                "initial_latent": initial_latent,
            }
            # CausalInferencePipeline supports low_memory; diffusion pipeline does not.
            if "low_memory" in pipeline.inference.__code__.co_varnames:
                kwargs["low_memory"] = self.low_memory_runtime
            video, latents = pipeline.inference(**kwargs)
        runtime_s = time.perf_counter() - start
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(self.device))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        video_uint8 = (255.0 * rearrange(video, "b t c h w -> b t h w c")).clamp(0, 255).to(torch.uint8).cpu()
        write_video(str(out_path), video_uint8[0], fps=int(self.cfg.fps))

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()

        _, _, _, h, w = video.shape
        return {
            "runtime_s": runtime_s,
            "peak_vram_bytes": peak_vram_bytes,
            "total_frames": int(video.shape[1]),
            "resolution": [int(h), int(w)],
            "latents_shape": list(latents.shape),
            "conditioning_status": cond_note,
            "conditioning_applied": initial_latent is not None,
            "noise_frames_used": int(noise_frames),
            "num_frame_per_block": int(num_frame_per_block),
            "independent_first_frame": independent_first_frame,
            "frame_alignment_note": align_note,
        }


def generate_video_self_forcing(
    prompt: str,
    cond_image_path: Optional[str],
    seed: int,
    out_path: str,
    config: dict[str, Any],
    generator: SelfForcingGenerator | None = None,
) -> dict[str, Any]:
    if generator is None:
        cfg = SelfForcingConfig(
            config_path=Path(config["config_path"]),
            default_config_path=Path(config["default_config_path"]),
            checkpoint_path=Path(config["checkpoint_path"]),
            use_ema=bool(config.get("use_ema", True)),
            device=str(config.get("device", "cuda:0")),
            num_output_frames=int(config.get("num_output_frames", 21)),
            fps=int(config.get("fps", 16)),
            low_memory=bool(config.get("low_memory", False)),
        )
        generator = SelfForcingGenerator(cfg)
    meta = generator.generate_video(prompt=prompt, cond_image_path=cond_image_path, seed=seed, out_path=Path(out_path))
    meta["git_commit_hash"] = git_commit_hash()
    meta["generator_method"] = "self_forcing_wan_1.3b"
    return meta


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_per_sample_records(records: list[dict[str, Any]], run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    completed = 0
    failed = 0
    total_runtime = 0.0
    total_questions = 0
    answered_questions = 0
    correct_answers = 0
    pending_answers = 0
    by_domain: dict[str, dict[str, int]] = {}

    for rec in records:
        errors = rec.get("errors", [])
        if errors:
            failed += 1
        else:
            completed += 1
            runtime = rec.get("runtime_s")
            if runtime is None:
                runtime = rec.get("runtime")
            if isinstance(runtime, (int, float)):
                total_runtime += float(runtime)

        domain = (
            rec.get("meta", {}).get("domain")
            or rec.get("meta", {}).get("task")
            or rec.get("meta", {}).get("type")
            or "unknown"
        )
        dom = by_domain.setdefault(str(domain), {"total": 0, "answered": 0, "correct": 0, "pending": 0})

        for qa in rec.get("qa_results", []):
            total_questions += 1
            dom["total"] += 1
            pred = qa.get("pred_answer")
            is_pending = bool(qa.get("pending", False))
            if is_pending:
                pending_answers += 1
                dom["pending"] += 1
                continue
            if isinstance(pred, bool):
                answered_questions += 1
                dom["answered"] += 1
                if bool(qa.get("correct", False)):
                    correct_answers += 1
                    dom["correct"] += 1

    accuracy_overall = (correct_answers / answered_questions) if answered_questions > 0 else 0.0
    accuracy_by_domain = {
        k: ((v["correct"] / v["answered"]) if v["answered"] > 0 else 0.0) for k, v in by_domain.items()
    }
    avg_runtime = total_runtime / completed if completed > 0 else math.nan

    return {
        "run_id": run_id,
        "method": "self_forcing_wan_1.3b",
        "config": config,
        "counts": {
            "records": len(records),
            "completed": completed,
            "failed": failed,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "pending_answers": pending_answers,
            "correct_answers": correct_answers,
        },
        "accuracy_overall": accuracy_overall,
        "accuracy_by_domain": accuracy_by_domain,
        "avg_runtime_s": avg_runtime,
    }
