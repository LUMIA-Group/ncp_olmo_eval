#!/usr/bin/env python3
"""Create an immutable, weight-symlinked long-context model overlay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .long_context_protocol import file_sha256, model_context_capability, write_json_atomic

OVERLAY_STATUS = "LONG_CONTEXT_MODEL_OVERLAY_READY"
OVERLAY_MANIFEST = "long_context_overlay_manifest.json"


def create_long_context_model_overlay(
    source_model: Path, output_dir: Path, target_context_length: int
) -> dict[str, Any]:
    """Link one model and replace only its declared context ceiling.

    The source artifact is never modified. Existing destinations are rejected
    so a previous runtime cannot be silently repointed to different weights.
    This operation declares unvalidated extrapolation; it does not change RoPE
    or YaRN parameters and therefore makes no quality claim beyond the native
    context supported by the source checkpoint.
    """

    source = source_model.resolve()
    output = output_dir.resolve()
    target = int(target_context_length)
    if not source.is_dir() or not (source / "config.json").is_file():
        raise FileNotFoundError(f"source model config is missing: {source / 'config.json'}")
    if output.exists():
        raise FileExistsError(f"overlay destination already exists: {output}")
    if target <= 0:
        raise ValueError("target_context_length must be positive")

    source_capability = model_context_capability(source / "config.json")
    native = int(source_capability["native_context_length"])
    if target <= native:
        raise ValueError(f"overlay target must exceed native context: {target} <= {native}")

    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    source_config["max_sequence_length"] = target
    source_config["max_position_embeddings"] = target

    output.mkdir(parents=True)
    linked_entries = []
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        if entry.name in {"config.json", OVERLAY_MANIFEST}:
            continue
        destination = output / entry.name
        os.symlink(str(entry.resolve()), destination, target_is_directory=entry.is_dir())
        linked_entries.append(entry.name)
    write_json_atomic(output / "config.json", source_config)

    manifest = {
        "status": OVERLAY_STATUS,
        "source_model": str(source),
        "source_model_config_sha256": source_capability["model_config_sha256"],
        "source_native_context_length": native,
        "target_context_length": target,
        "context_mode": "explicit_unvalidated_extrapolation",
        "rope_scaling_unchanged": True,
        "linked_entries": linked_entries,
        "overlay_config_sha256": file_sha256(output / "config.json"),
    }
    write_json_atomic(output / OVERLAY_MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-context-length", type=int, required=True)
    args = parser.parse_args()
    manifest = create_long_context_model_overlay(
        args.source_model, args.output_dir, args.target_context_length
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
