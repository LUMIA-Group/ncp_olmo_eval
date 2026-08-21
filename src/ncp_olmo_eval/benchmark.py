"""Artifact integrity helpers shared by the standalone evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _hf_artifact_fingerprint(model_path: str | Path) -> dict[str, Any]:
    """Fingerprint metadata and weight file stats without reading all weights."""

    root = Path(model_path).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    relative_paths = {"config.json"}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid SafeTensors index: {index_path}")
        relative_paths.update({"model.safetensors.index.json", *weight_map.values()})
    elif (root / "model.safetensors").is_file():
        relative_paths.add("model.safetensors")
    else:
        raise FileNotFoundError(f"model has no SafeTensors weights or index: {root}")

    metadata_paths = [
        name
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        )
        if (root / name).is_file()
    ]
    relative_paths.update(metadata_paths)
    files: dict[str, dict[str, int]] = {}
    for relative_path in sorted(relative_paths):
        path = root / relative_path
        stat = path.stat()
        files[str(relative_path)] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    digest = hashlib.sha256()
    for relative_path in metadata_paths:
        with (root / relative_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": str(root),
        "metadata_sha256": digest.hexdigest(),
        "metadata_files": metadata_paths,
        "files": files,
    }


def _artifact_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint the only supported artifact kind: an HF-compatible directory."""

    model_path = getattr(args, "hf_model_path", "")
    if not model_path:
        raise ValueError("vLLM evaluation requires hf_model_path")
    return _hf_artifact_fingerprint(model_path)


def _peak_memory_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024**3)


def load_eval_model(_: argparse.Namespace) -> Any:
    raise RuntimeError("ncp-olmo-eval only ships the vLLM inference backend")
