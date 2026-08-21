"""Compatibility boundary for the intentionally omitted Megatron backend."""

from __future__ import annotations

from typing import Any

NATIVE_MEGATRON_BACKEND = "native_megatron"


def build_native_megatron_inferencer(*_: Any, **__: Any) -> Any:
    raise RuntimeError("Megatron inference is not included in ncp-olmo-eval")
