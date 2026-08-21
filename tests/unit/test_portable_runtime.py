from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from ncp_olmo_eval import assets
from ncp_olmo_eval.source_identity import package_tree_sha256, source_state
from ncp_olmo_eval.task_runner import run_task
from ncp_olmo_eval.task_spec import Resources, TaskSpec, read_status, write_task


def _task(tmp_path: Path, *, returncode: int) -> tuple[Path, TaskSpec]:
    task_root = tmp_path / f"task-{returncode}"
    spec = TaskSpec(
        task_id=f"portable-task-{returncode}",
        phase="test",
        benchmark="gsm8k",
        role="contract",
        argv=(sys.executable, "-c", f"raise SystemExit({returncode})"),
        cwd=str(tmp_path.resolve()),
        env={"NCP_TEST_VALUE": "portable"},
        resources=Resources(),
        output_root=str((task_root / "output").resolve()),
        status_path=str((task_root / "status.json").resolve()),
        log_path=str((task_root / "task.log").resolve()),
    )
    path = task_root / "task.json"
    write_task(path, spec)
    return path, spec


@pytest.mark.parametrize(("returncode", "state"), [(0, "Succeeded"), (7, "Failed")])
def test_task_runner_persists_terminal_status(tmp_path: Path, returncode: int, state: str) -> None:
    task_path, spec = _task(tmp_path, returncode=returncode)
    result = run_task(task_path)
    assert result["state"] == state
    assert result["returncode"] == returncode
    assert read_status(Path(spec.status_path))["state"] == state
    assert Path(spec.log_path).is_file()


def test_asset_manifest_requires_full_hub_commit(tmp_path: Path) -> None:
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": assets.ASSET_SCHEMA,
                "assets": [
                    {
                        "name": "model",
                        "repo_id": "org/model",
                        "revision": "main",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="40-character"):
        assets._read_manifest(manifest)


def test_asset_lock_is_relocatable_and_detects_changes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    model = bundle / "model"
    model.mkdir(parents=True)
    weight = model / "config.json"
    weight.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    lock = bundle / "assets.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": assets.LOCK_SCHEMA,
                "assets": [
                    {
                        "name": "model",
                        "root": "model",
                        "files": [
                            {
                                "path": "config.json",
                                "size": weight.stat().st_size,
                                "sha256": digest,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert assets.verify(lock)["files_checked"] == 1

    moved = tmp_path / "moved"
    bundle.rename(moved)
    moved_lock = moved / "assets.lock.json"
    assert assets.verify(moved_lock)["files_checked"] == 1
    (moved / "model" / "config.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after sealing"):
        assets.verify(moved_lock)


def test_installed_source_identity_is_path_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    monkeypatch.setenv("NCP_OLMO_SOURCE_REVISION", revision)
    state = source_state(tmp_path / "not-a-checkout")
    assert state["source_kind"] == "installed-package"
    assert state["repo_commit"] == revision
    assert state["source_tree_sha256"] == package_tree_sha256()
