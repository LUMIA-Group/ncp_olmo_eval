from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncp_olmo_eval.dflash_checkpoint import (
    dflash_checkpoint_identity,
    dflash_identity_matches,
    dflash_weight_files,
)


def _config(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"conceptlm_dflash"}\n', encoding="utf-8")


def _sharded_checkpoint(root: Path, *, shard_count: int = 21) -> Path:
    _config(root)
    weight_map: dict[str, str] = {}
    for index in range(1, shard_count + 1):
        name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        (root / name).write_bytes(f"shard-{index}".encode())
        weight_map[f"layer.{index}.weight"] = name
        weight_map[f"layer.{index}.bias"] = name
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return root


def test_single_file_identity_remains_legacy_compatible(tmp_path: Path) -> None:
    root = tmp_path / "draft"
    _config(root)
    weights = root / "model.safetensors"
    weights.write_bytes(b"single")

    identity = dflash_checkpoint_identity(root)
    legacy = {
        key: identity[key]
        for key in ("path", "config_sha256", "weights_size", "weights_mtime_ns")
    }

    assert identity["weight_index"] is None
    assert identity["weight_files"] == ["model.safetensors"]
    assert identity["weight_file_count"] == 1
    assert dflash_identity_matches(legacy, identity)


def test_hf_index_resolves_each_of_21_unique_shards_once(tmp_path: Path) -> None:
    root = _sharded_checkpoint(tmp_path / "draft")
    (root / "model.safetensors").write_bytes(b"stale-single-file")

    index_path, shards = dflash_weight_files(root)
    identity = dflash_checkpoint_identity(root)

    assert index_path == root / "model.safetensors.index.json"
    assert len(shards) == 21
    assert identity["weight_file_count"] == 21
    assert identity["weight_files"][0] == "model-00001-of-00021.safetensors"
    assert identity["weight_files"][-1] == "model-00021-of-00021.safetensors"
    assert identity["weights_size"] == sum(path.stat().st_size for path in shards)
    assert identity["weight_index_sha256"]


def test_sharded_identity_requires_full_shard_metadata(tmp_path: Path) -> None:
    root = _sharded_checkpoint(tmp_path / "draft", shard_count=2)
    identity = dflash_checkpoint_identity(root)
    legacy = {
        key: identity[key]
        for key in ("path", "config_sha256", "weights_size", "weights_mtime_ns")
    }

    assert not dflash_identity_matches(legacy, identity)
    assert dflash_identity_matches(identity, identity)


def test_missing_indexed_shard_is_rejected(tmp_path: Path) -> None:
    root = _sharded_checkpoint(tmp_path / "draft", shard_count=2)
    (root / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="missing indexed SafeTensors shards"):
        dflash_weight_files(root)


def test_unsafe_indexed_shard_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "draft"
    _config(root)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "../escape.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe shard path"):
        dflash_weight_files(root)
