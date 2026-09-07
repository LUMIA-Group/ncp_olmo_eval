"""Validation and immutable identity helpers for NCP DFlash checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

DRAFT_CONFIG_NAME = "config.json"
DRAFT_SINGLE_WEIGHTS_NAME = "model.safetensors"
DRAFT_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_weight_name(raw_name: object, *, index_path: Path) -> str:
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError(f"DFlash weight_map contains an invalid shard name: {index_path}")
    if "\\" in raw_name:
        raise ValueError(f"DFlash weight_map contains a non-POSIX shard path: {raw_name}")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"DFlash weight_map contains an unsafe shard path: {raw_name}")
    if relative.suffix != ".safetensors":
        raise ValueError(f"DFlash weight_map references a non-safetensors shard: {raw_name}")
    return relative.as_posix()


def dflash_weight_files(checkpoint: Path) -> tuple[Path | None, tuple[Path, ...]]:
    """Resolve one-file or Hugging Face-indexed DFlash SafeTensors weights."""

    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NCP DFlash checkpoint does not exist: {root}")
    config_path = root / DRAFT_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"NCP DFlash checkpoint is missing config.json: {root}")

    index_path = root / DRAFT_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid DFlash SafeTensors index: {index_path}: {error}") from error
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"DFlash SafeTensors index has no non-empty weight_map: {index_path}")
        shard_names: set[str] = set()
        for tensor_name, raw_shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError(f"DFlash weight_map contains an invalid tensor name: {index_path}")
            shard_names.add(_safe_weight_name(raw_shard_name, index_path=index_path))
        shard_paths = tuple(root / name for name in sorted(shard_names))
        missing = [path.relative_to(root).as_posix() for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"NCP DFlash checkpoint is missing indexed SafeTensors shards: {missing}"
            )
        return index_path, shard_paths

    single_path = root / DRAFT_SINGLE_WEIGHTS_NAME
    if not single_path.is_file():
        raise FileNotFoundError(
            "NCP DFlash checkpoint is missing model.safetensors or "
            f"model.safetensors.index.json: {root}"
        )
    return None, (single_path,)


def dflash_checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Seal the config, index, and every referenced weight shard without loading tensors."""

    root = checkpoint.expanduser().resolve()
    index_path, weight_paths = dflash_weight_files(root)
    relative_names = [path.relative_to(root).as_posix() for path in weight_paths]
    stats = {name: path.stat() for name, path in zip(relative_names, weight_paths, strict=True)}
    sizes = {name: stat.st_size for name, stat in stats.items()}
    mtimes = {name: stat.st_mtime_ns for name, stat in stats.items()}
    return {
        "path": str(root),
        "config_sha256": _sha256(root / DRAFT_CONFIG_NAME),
        "weight_index": index_path.name if index_path is not None else None,
        "weight_index_sha256": _sha256(index_path) if index_path is not None else None,
        "weight_files": relative_names,
        "weight_file_count": len(relative_names),
        "weight_sizes": sizes,
        "weight_mtimes_ns": mtimes,
        # Preserve the legacy scalar fields. For a sharded checkpoint they are
        # deterministic aggregates; the per-file maps remain authoritative.
        "weights_size": sum(sizes.values()),
        "weights_mtime_ns": max(mtimes.values()),
    }


def dflash_identity_matches(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare identities while accepting legacy metadata for one-file drafts."""

    legacy_fields = ("path", "config_sha256", "weights_size", "weights_mtime_ns")
    if any(observed.get(field) != expected.get(field) for field in legacy_fields):
        return False
    extended_fields = (
        "weight_index",
        "weight_index_sha256",
        "weight_files",
        "weight_file_count",
        "weight_sizes",
        "weight_mtimes_ns",
    )
    expected_count = int(expected.get("weight_file_count", 1))
    observed_has_extended_identity = any(field in observed for field in extended_fields)
    if expected_count == 1 and not observed_has_extended_identity:
        return True
    return all(observed.get(field) == expected.get(field) for field in extended_fields)
