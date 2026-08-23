"""Compatibility boundary for the intentionally omitted LMDeploy backend."""

from __future__ import annotations

from typing import Any

LMDEPLOY_BACKEND = "lmdeploy"


def add_lmdeploy_args(_: Any) -> None:
    return None


def validate_lmdeploy_args(args: Any, *_: Any, **__: Any) -> None:
    if getattr(args, "hf_backend", None) != LMDEPLOY_BACKEND:
        return
    raise RuntimeError("LMDeploy is not included in ncp-olmo-eval")


def lmdeploy_engine_policy(*_: Any, **__: Any) -> dict[str, Any]:
    raise RuntimeError("LMDeploy is not included in ncp-olmo-eval")


class LMDeployInferencer:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("LMDeploy is not included in ncp-olmo-eval")
