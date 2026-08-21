"""Small backend-neutral value objects used by the vLLM evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SamplingParams:
    """One request-local sampling contract."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    logprobs: int = 0
    sample_in_vocab_order: bool = False


@dataclass
class TextCompletion:
    """One decoded completion plus the evidence needed by scorers."""

    text: str
    token_ids: list[int]
    finish_reason: str
    cache_fallback_count: int = 0
    cache_fallback_steps: list[int] = field(default_factory=list)
    kv_stats: dict[str, Any] = field(default_factory=dict)


class ConceptLMInferencer:
    """Removed legacy backend retained only as an explicit failure boundary."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("ncp-olmo-eval only ships the vLLM inference backend")
