"""Pinned lm-evaluation-harness runtime contract for GSM8K/Core88."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

LM_EVAL_DISTRIBUTION = "lm_eval"
LM_EVAL_VERSION = "0.4.13.dev0"
LM_EVAL_COMMIT = "95d580638385578c1c07fa554cf16ad7f5b5f460"
LM_EVAL_REPOSITORY = "https://github.com/EleutherAI/lm-evaluation-harness.git"
LM_EVAL_REQUIREMENT = f"lm_eval @ git+{LM_EVAL_REPOSITORY}@{LM_EVAL_COMMIT}"


def _installed_version() -> str:
    try:
        return importlib.metadata.version(LM_EVAL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "GSM8K/Core88 requires the pinned lm-evaluation-harness runtime; "
            "install ncp-olmo-eval[vllm] for inference or "
            "ncp-olmo-eval[gsm8k] for protocol preparation"
        ) from error


def task_manager_class() -> type[Any]:
    """Return the pinned TaskManager API or fail before reading task data."""

    version = _installed_version()
    if version != LM_EVAL_VERSION:
        raise RuntimeError(
            "unsupported lm-evaluation-harness version: "
            f"installed={version!r}, expected={LM_EVAL_VERSION!r} "
            f"from commit {LM_EVAL_COMMIT}"
        )
    try:
        module = importlib.import_module("lm_eval.tasks")
    except ImportError as error:
        raise RuntimeError(
            "the pinned lm-evaluation-harness distribution is installed but "
            "lm_eval.tasks cannot be imported"
        ) from error
    task_manager = getattr(module, "TaskManager", None)
    if task_manager is None:
        raise RuntimeError("the pinned lm_eval.tasks module has no TaskManager")
    return task_manager


def runtime_identity() -> dict[str, str]:
    """Return the dependency identity sealed into GSM8K input manifests."""

    task_manager_class()
    return {
        "distribution": LM_EVAL_DISTRIBUTION,
        "version": LM_EVAL_VERSION,
        "repository": LM_EVAL_REPOSITORY,
        "commit": LM_EVAL_COMMIT,
    }
