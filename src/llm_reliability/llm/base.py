"""The common interface every LLM backend implements.

The rest of the pipeline only ever calls ``client.complete(prompt, ...)``, so
swapping Claude for a local model (or the offline simulator) is a one-line
config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None  # includes hidden reasoning tokens where the API counts them
    stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(ABC):
    """Abstract LLM client.

    ``purpose`` is "answer", "sample" or "self_check" and ``meta`` carries the
    question id / sample index. Real backends ignore ``meta``; the offline
    simulator uses it to stay deterministic.
    """

    provider: str = "base"
    model: str = "unknown"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        purpose: str = "answer",
        meta: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
