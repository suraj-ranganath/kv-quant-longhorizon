from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluatorResult:
    pred_answer: bool
    pending: bool = False
    trace_path: str | None = None
    meta: dict[str, Any] | None = None


class BinaryQAEvaluator(ABC):
    @abstractmethod
    def predict(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> bool:
        raise NotImplementedError

    def predict_with_trace(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> EvaluatorResult:
        return EvaluatorResult(pred_answer=self.predict(video_path, prompt, cond_image_path, question))


class HeuristicEvaluator(BinaryQAEvaluator):
    def __init__(self, mode: str = "always_false", seed: int = 0) -> None:
        self.mode = mode
        self._rng = random.Random(seed)

    def predict(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> bool:
        if self.mode == "random":
            return bool(self._rng.getrandbits(1))
        return False


class ManualEvaluator(BinaryQAEvaluator):
    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _trace_key(video_path: str, question: str) -> str:
        safe_video = Path(video_path).stem.replace(" ", "_")
        safe_question = "".join(c if c.isalnum() else "_" for c in question)[:80]
        safe_question = safe_question.strip("_") or "question"
        return f"{safe_video}__{safe_question}"

    def _trace_path(self, video_path: str, question: str) -> Path:
        return self.queue_dir / f"{self._trace_key(video_path, question)}.json"

    def predict_with_trace(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> EvaluatorResult:
        path = self._trace_path(video_path, question)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                pred = payload.get("pred_answer")
                if isinstance(pred, bool):
                    return EvaluatorResult(pred_answer=pred, pending=False, trace_path=str(path))
            except Exception:
                pass

        payload = {
            "created_at_unix": int(time.time()),
            "video_path": video_path,
            "prompt": prompt,
            "cond_image_path": cond_image_path,
            "question": question,
            "pred_answer": None,
            "status": "pending_manual",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return EvaluatorResult(pred_answer=False, pending=True, trace_path=str(path))

    def predict(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> bool:
        return self.predict_with_trace(video_path, prompt, cond_image_path, question).pred_answer


class VLMRemoteEvaluator(BinaryQAEvaluator):
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        trace_dir: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def predict_with_trace(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> EvaluatorResult:
        body = {
            "video_path": video_path,
            "prompt": prompt,
            "cond_image_path": cond_image_path,
            "question": question,
            "task": "binary_qa",
        }
        raw_response = {}
        pred_answer = False
        err: str | None = None
        try:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read().decode("utf-8")
                raw_response = json.loads(payload)
            parsed = raw_response.get("pred_answer")
            if isinstance(parsed, bool):
                pred_answer = parsed
            elif isinstance(parsed, str):
                parsed_lower = parsed.strip().lower()
                pred_answer = parsed_lower in {"yes", "true", "1"}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            err = str(exc)

        trace_path = None
        if self.trace_dir is not None:
            trace_path = self.trace_dir / f"{int(time.time() * 1000)}.json"
            trace_payload = {
                "request": body,
                "response": raw_response,
                "error": err,
                "pred_answer": pred_answer,
            }
            trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return EvaluatorResult(
            pred_answer=pred_answer,
            pending=False,
            trace_path=str(trace_path) if trace_path else None,
            meta={"error": err} if err else None,
        )

    def predict(self, video_path: str, prompt: str, cond_image_path: str | None, question: str) -> bool:
        return self.predict_with_trace(video_path, prompt, cond_image_path, question).pred_answer
