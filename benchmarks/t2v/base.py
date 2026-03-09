from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkPrompt:
    prompt_id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


class BenchmarkLoader(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> list[BenchmarkPrompt]:
        raise NotImplementedError

