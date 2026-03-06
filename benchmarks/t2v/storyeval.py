from __future__ import annotations

import re
from pathlib import Path

from .base import BenchmarkLoader, BenchmarkPrompt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = REPO_ROOT / "data" / "prompts" / "storyeval" / "all_prompts.txt"


# Ported from third_party/StoryEval/utils.py.
def sentence_to_filename(sentence: str, max_length: int = 198) -> str:
    sentence = re.sub(r"[^\w\s]", "", sentence)
    words = sentence.split()
    filename = "_".join(words)
    filename = filename[:max_length]
    return filename


def storyeval_video_name(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_seed{seed}.mp4"


class StoryEvalLoader(BenchmarkLoader):
    def __init__(self, prompt_path: Path = DEFAULT_PROMPT_PATH) -> None:
        self.prompt_path = prompt_path

    def name(self) -> str:
        return "storyeval"

    def load(self) -> list[BenchmarkPrompt]:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"StoryEval prompt file not found: {self.prompt_path}")

        prompts: list[BenchmarkPrompt] = []
        seen_ids: set[str] = set()
        with self.prompt_path.open("r", encoding="utf-8") as f:
            for idx, raw_line in enumerate(f):
                prompt = raw_line.strip()
                if not prompt:
                    continue
                prompt_id = sentence_to_filename(prompt)
                if prompt_id in seen_ids:
                    raise ValueError(
                        f"Duplicate StoryEval prompt_id '{prompt_id}' at line {idx + 1}. "
                        "This would overwrite videos."
                    )
                seen_ids.add(prompt_id)
                prompts.append(
                    BenchmarkPrompt(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        meta={
                            "line_index": idx,
                            "source_path": str(self.prompt_path),
                        },
                    )
                )
        return prompts

