from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .types import EmbodiedSample

DATASET_NAME = "nvidia/PBench"
DEFAULT_CACHE_ROOT = Path("data_cache/pbench")


def _normalize_sample_id(value: Any, fallback_index: int) -> str:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        raw = f"sample_{fallback_index:06d}"
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return raw or f"sample_{fallback_index:06d}"


def _pick_first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "true", "1", "y", "t"}:
            return True
        if v in {"no", "false", "0", "n", "f"}:
            return False
    return None


def _iter_question_answer_pairs(obj: Any) -> Iterable[tuple[str, bool]]:
    if isinstance(obj, str):
        text = obj.strip()
        if text and text[0] in "[{":
            try:
                parsed = json.loads(text)
                yield from _iter_question_answer_pairs(parsed)
            except Exception:
                pass
        return
    if isinstance(obj, dict):
        keys = {k.lower(): k for k in obj.keys()}
        if "question" in keys and ("answer" in keys or "label" in keys or "target" in keys):
            q_key = keys["question"]
            a_key = keys.get("answer") or keys.get("label") or keys.get("target")
            q = obj.get(q_key)
            a = _to_bool(obj.get(a_key))
            if isinstance(q, str) and q.strip() and a is not None:
                yield q.strip(), a
        for v in obj.values():
            yield from _iter_question_answer_pairs(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_question_answer_pairs(item)


def _extract_qa_pairs(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, bool]] = set()

    # Fast-path common layouts.
    questions = _pick_first(row, ["questions", "qa_questions"])
    answers = _pick_first(row, ["answers", "qa_answers"])
    if isinstance(questions, list) and isinstance(answers, list):
        for q, a in zip(questions, answers):
            a_bool = _to_bool(a)
            if isinstance(q, str) and q.strip() and a_bool is not None:
                key = (q.strip(), a_bool)
                if key not in seen:
                    seen.add(key)
                    out.append({"question": q.strip(), "answer": a_bool})

    single_q = _pick_first(row, ["question", "qa_question"])
    single_a = _pick_first(row, ["answer", "qa_answer"])
    single_a_bool = _to_bool(single_a)
    if isinstance(single_q, str) and single_q.strip() and single_a_bool is not None:
        key = (single_q.strip(), single_a_bool)
        if key not in seen:
            seen.add(key)
            out.append({"question": single_q.strip(), "answer": single_a_bool})

    for q, a in _iter_question_answer_pairs(row):
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q, "answer": a})

    # Common PBench packaging: `qa_pairs` as serialized JSON string.
    qa_pairs_raw = row.get("qa_pairs")
    if isinstance(qa_pairs_raw, str):
        try:
            parsed = json.loads(qa_pairs_raw)
            for q, a in _iter_question_answer_pairs(parsed):
                key = (q, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"question": q, "answer": a})
        except Exception:
            pass

    return out


def _clean_meta(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, (bytes, bytearray)):
                cleaned[k] = f"<bytes:{len(v)}>"
            elif isinstance(v, Image.Image):
                cleaned[k] = "<image>"
            else:
                cleaned[k] = _clean_meta(v)
        return cleaned
    if isinstance(obj, list):
        return [_clean_meta(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_prompt(row: dict[str, Any]) -> str:
    prompt = _pick_first(
        row,
        [
            "prompt",
            "extended_prompt",
            "caption",
            "text",
            "instruction",
            "description",
        ],
    )
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    # last resort: find first non-empty string with prompt-like key
    for k, v in row.items():
        kl = k.lower()
        if isinstance(v, str) and v.strip() and any(tok in kl for tok in ("prompt", "caption", "text", "instruction")):
            return v.strip()
    return ""


def _decode_image_like(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = value.get("path")
        if isinstance(path, str) and path:
            p = Path(path)
            if p.exists():
                return Image.open(p).convert("RGB")
    if isinstance(value, str):
        p = Path(value)
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def _extract_conditioning_image(row: dict[str, Any]) -> Image.Image | None:
    direct_keys = [
        "cond_image",
        "conditioning_image",
        "image",
        "input_image",
        "source_image",
        "reference_image",
    ]
    for key in direct_keys:
        if key in row:
            img = _decode_image_like(row[key])
            if img is not None:
                return img
    for k, v in row.items():
        if "image" in k.lower():
            img = _decode_image_like(v)
            if img is not None:
                return img
    return None


def load_pbench_samples(
    split: str | None = None,
    max_samples: int | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    hf_token: str | None = None,
) -> tuple[list[EmbodiedSample], str]:
    try:
        from datasets import get_dataset_split_names, load_dataset
    except Exception as exc:
        raise RuntimeError("Missing dependency: `datasets`. Install it in the inference env.") from exc

    try:
        split_names = get_dataset_split_names(DATASET_NAME, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            "Unable to access nvidia/PBench. Ensure HF access is granted and run `huggingface-cli login`."
        ) from exc
    if not split_names:
        raise RuntimeError("Dataset nvidia/PBench returned no splits.")

    target_split = split or split_names[0]
    if target_split not in split_names:
        raise ValueError(f"Unknown split `{target_split}`. Available: {split_names}")

    split_expr = f"{target_split}[:{max_samples}]" if max_samples is not None else target_split
    dataset = load_dataset(DATASET_NAME, split=split_expr, token=hf_token)

    cache_root = Path(cache_root)
    image_root = cache_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    samples: list[EmbodiedSample] = []
    used_ids: set[str] = set()

    for idx in range(len(dataset)):
        row = dict(dataset[idx])

        sample_id_raw = _pick_first(row, ["sample_id", "id", "uid", "example_id", "instance_id"])
        sample_id = _normalize_sample_id(sample_id_raw, idx)
        if sample_id in used_ids:
            sample_id = f"{sample_id}_{idx:06d}"
        used_ids.add(sample_id)

        prompt = _extract_prompt(row)
        if not prompt:
            prompt = f"[missing_prompt] sample_id={sample_id}"

        qa_pairs = _extract_qa_pairs(row)
        image = _extract_conditioning_image(row)
        cond_image_path: str | None = None
        if image is not None:
            out_path = image_root / f"{sample_id}.png"
            image.convert("RGB").save(out_path)
            cond_image_path = str(out_path)

        samples.append(
            EmbodiedSample(
                sample_id=sample_id,
                prompt=prompt,
                cond_image_path=cond_image_path,
                qa_pairs=qa_pairs,
                meta=_clean_meta(row),
            )
        )

    return samples, target_split


def save_normalized_samples(samples: list[EmbodiedSample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "prompt": s.prompt,
                        "cond_image_path": s.cond_image_path,
                        "qa_pairs": s.qa_pairs,
                        "meta": s.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_normalized_samples(path: Path) -> list[EmbodiedSample]:
    out: list[EmbodiedSample] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out.append(
                    EmbodiedSample(
                        sample_id=str(row.get("sample_id")),
                        prompt=str(row.get("prompt", "")),
                        cond_image_path=row.get("cond_image_path"),
                        qa_pairs=list(row.get("qa_pairs", [])),
                        meta=dict(row.get("meta", {})),
                    )
                )
            except Exception:
                continue
    return out
