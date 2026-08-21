"""Prepare and verify sealed Hugging Face assets for offline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ASSET_SCHEMA = "ncp-olmo-eval-assets-v1"
LOCK_SCHEMA = "ncp-olmo-eval-assets-lock-v1"
_FULL_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != ASSET_SCHEMA:
        raise ValueError(f"unsupported asset manifest: {path}")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("asset manifest must contain a non-empty assets list")
    normalized = []
    names: set[str] = set()
    for item in assets:
        if not isinstance(item, dict):
            raise ValueError("each asset must be an object")
        name = str(item.get("name", ""))
        if not name or name in names or "/" in name or name.startswith("."):
            raise ValueError(f"invalid or duplicate asset name: {name!r}")
        revision = str(item.get("revision", ""))
        if not _FULL_COMMIT_SHA.fullmatch(revision):
            raise ValueError(f"asset {name} must pin a full 40-character Hub commit SHA")
        repo_id = str(item.get("repo_id", ""))
        repo_type = str(item.get("repo_type", "model"))
        if not repo_id or repo_type not in {"model", "dataset", "space"}:
            raise ValueError(f"asset {name} has an invalid repo_id or repo_type")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "allow_patterns": [str(value) for value in item.get("allow_patterns", [])],
                "ignore_patterns": [str(value) for value in item.get("ignore_patterns", [])],
            }
        )
    return normalized


def _contract(root: Path, lock_root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _files(root)
    ]
    if not files:
        raise ValueError(f"asset has no files: {root}")
    return {**asset, "root": str(root.resolve().relative_to(lock_root.resolve())), "files": files}


def prepare(manifest: Path, output_root: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    assets = _read_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    contracts = []
    for asset in assets:
        destination = output_root / asset["name"]
        snapshot_download(
            repo_id=asset["repo_id"],
            repo_type=asset["repo_type"],
            revision=asset["revision"],
            local_dir=destination,
            allow_patterns=asset["allow_patterns"] or None,
            ignore_patterns=asset["ignore_patterns"] or None,
        )
        contracts.append(_contract(destination, output_root, asset))
    lock = {
        "schema_version": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": manifest.name,
        "source_manifest_sha256": _sha256(manifest),
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "assets": contracts,
    }
    lock_path = output_root / "assets.lock.json"
    temporary = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(lock_path)
    return lock


def verify(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError(f"unsupported asset lock: {lock_path}")
    checked = 0
    for asset in lock.get("assets", []):
        root = lock_path.resolve().parent / str(asset["root"])
        for item in asset.get("files", []):
            path = root / str(item["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(item["size"]) or _sha256(path) != item["sha256"]:
                raise ValueError(f"asset changed after sealing: {path}")
            checked += 1
    return {"status": "ASSET_CACHE_VERIFIED", "lock": str(lock_path), "files_checked": checked}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = (
        prepare(args.manifest, args.output_root) if args.command == "prepare" else verify(args.lock)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
