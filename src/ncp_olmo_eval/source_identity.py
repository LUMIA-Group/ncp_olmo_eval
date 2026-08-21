"""Immutable evaluator source identity for checkouts, wheels, and OCI images."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

_FULL_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def _source_files(package_root: Path) -> Iterable[Path]:
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".typed"} and "__pycache__" not in path.parts:
            yield path


def package_tree_sha256(package_root: Path | None = None) -> str:
    """Hash installed evaluator sources independently of their absolute path."""

    root = (package_root or Path(__file__).resolve().parent).resolve()
    digest = hashlib.sha256()
    count = 0
    for path in _source_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
        count += 1
    if count == 0:
        raise RuntimeError(f"evaluator package has no source files: {root}")
    return digest.hexdigest()


def source_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Return a fail-closed identity for a Git checkout or installed package."""

    package_root = Path(__file__).resolve().parent
    tree_sha256 = package_tree_sha256(package_root)
    if repo_root is not None:
        root = repo_root.resolve()
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not _FULL_REVISION.fullmatch(commit):
                raise RuntimeError(f"Git returned a non-full revision: {commit!r}")
            return {
                "source_kind": "git",
                "repo_root": str(root),
                "repo_commit": commit,
                "repo_dirty": bool(dirty),
                "source_tree_sha256": tree_sha256,
            }
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    try:
        package_version = version("ncp-olmo-eval")
    except PackageNotFoundError:
        package_version = "unknown"
    declared_revision = os.environ.get("NCP_OLMO_SOURCE_REVISION", "")
    revision = (
        declared_revision if _FULL_REVISION.fullmatch(declared_revision) else tree_sha256[:40]
    )
    return {
        "source_kind": "installed-package",
        "repo_root": str(package_root),
        "repo_commit": revision,
        "repo_dirty": False,
        "source_tree_sha256": tree_sha256,
        "package_version": package_version,
    }


def evaluator_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Return source identity using the legacy aggregate report field names."""

    state = source_state(repo_root)
    return {
        "evaluator_repo_root": state["repo_root"],
        "evaluator_repo_commit": state["repo_commit"],
        "evaluator_repo_dirty": state["repo_dirty"],
        "evaluator_source_kind": state["source_kind"],
        "evaluator_source_tree_sha256": state["source_tree_sha256"],
        "evaluator_package_version": state.get("package_version"),
    }
