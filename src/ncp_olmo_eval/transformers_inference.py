"""Compatibility boundary for the intentionally omitted Transformers backend."""

from __future__ import annotations

from typing import Any


class TransformersInferencer:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("Transformers inference is not included in ncp-olmo-eval")
