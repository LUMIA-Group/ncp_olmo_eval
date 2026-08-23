from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ncp_olmo_eval import evaluation_cli
from ncp_olmo_eval.task_spec import Resources, TaskSpec, read_status, read_task


def _checkpoint(root: Path, *, model_type: str = "ncp_olmo3") -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": ["NcpOlmoForCausalLM"]}),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fixture")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _clean_git_state(root: Path) -> dict[str, object]:
    return {
        "source_kind": "git",
        "repo_root": str(root),
        "repo_commit": "1" * 40,
        "repo_dirty": False,
        "source_tree_sha256": "2" * 64,
    }


def test_registration_is_vllm_only_and_versioned(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    first = evaluation_cli.register_model(
        root=evaluation_root, checkpoint=checkpoint, backend="vllm", new_version=False
    )
    assert first["registration_name"].startswith("vllm-")
    assert first["registration_name"].endswith("-v1")
    with pytest.raises(evaluation_cli.EvaluationError, match="已经注册"):
        evaluation_cli.register_model(
            root=evaluation_root, checkpoint=checkpoint, backend="vllm", new_version=False
        )
    second = evaluation_cli.register_model(
        root=evaluation_root, checkpoint=checkpoint, backend="vllm", new_version=True
    )
    assert second["registration_name"].endswith("-v2")
    with pytest.raises(evaluation_cli.EvaluationError, match="不支持"):
        evaluation_cli.register_model(
            root=evaluation_root, checkpoint=checkpoint, backend="hf", new_version=False
        )


def test_protocol_matrix_is_frozen() -> None:
    core = evaluation_cli.protocol_for("vllm", "core88")
    assert core["global_seed"] == 42
    assert core["batch_size"] == 8
    assert core["generation_samples_cap"] == 0
    assert core["vllm_max_model_len"] == 8192
    ruler = evaluation_cli.protocol_for("vllm", "ruler")
    assert ruler["batch_size"] == 4
    assert ruler["lengths"] == [4096, 8192, 16384, 32768, 65536]
    assert ruler["eos_stopping"] is True
    helmet = evaluation_cli.protocol_for("vllm", "helmet")
    assert helmet["batch_size"] == 4
    assert helmet["lengths"] == [8192, 16384, 32768, 65536]


def test_task_names_are_generic_and_dns_safe() -> None:
    registration = "vllm-abcdef-v12345678901234567890"
    names = (
        evaluation_cli._job_name("pred", registration, "core88", "m3-a1r2"),
        evaluation_cli._job_name("eval", registration, "core88", "p55-a1"),
        evaluation_cli._final_job_name(registration, "helmet", "a1"),
    )
    assert all(name.startswith("ncp-eval-") for name in names)
    assert all(len(name) < 50 for name in names)


def test_dry_run_builds_five_portable_core88_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_root = tmp_path / "evaluations"
    registration = evaluation_cli.register_model(
        root=evaluation_root,
        checkpoint=_checkpoint(tmp_path),
        backend="vllm",
        new_version=False,
    )
    monkeypatch.setattr(evaluation_cli, "_git_state", lambda _: _clean_git_state(tmp_path))
    result = evaluation_cli.submit_inference(
        root=evaluation_root,
        registration_name=registration["registration_name"],
        benchmark="core88",
        data_root=None,
        executor="emit",
        force=False,
        dry_run=True,
    )
    assert len(result["submitted"]) == 5
    assert [task["role"] for task in result["jobs"]] == [
        "core88-m0",
        "core88-m1",
        "core88-m2",
        "core88-m3",
        "gsm8k-m4",
    ]
    assert all(task["status"] == "DryRun" for task in result["jobs"])
    assert all(task["command"]["argv"][0] == evaluation_cli.PYTHON_BIN for task in result["jobs"])
    for task in result["jobs"][:4]:
        argv = task["command"]["argv"]
        assert "--no-allow-unverified-lmdeploy" not in argv
    assert not (evaluation_root / registration["registration_name"] / "core88").exists()


def test_emit_materializes_scheduler_neutral_gsm8k_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_root = tmp_path / "evaluations"
    registration = evaluation_cli.register_model(
        root=evaluation_root,
        checkpoint=_checkpoint(tmp_path),
        backend="vllm",
        new_version=False,
    )
    monkeypatch.setattr(evaluation_cli, "_git_state", lambda _: _clean_git_state(tmp_path))
    result = evaluation_cli.submit_inference(
        root=evaluation_root,
        registration_name=registration["registration_name"],
        benchmark="gsm8k",
        data_root=None,
        executor="emit",
        force=False,
        dry_run=False,
    )
    task = result["jobs"][0]
    spec = read_task(Path(task["task_path"]))
    status = read_status(Path(task["status_path"]))
    assert spec.resources == Resources(gpus=8, cpus=64, memory_gib=256)
    assert spec.argv[:4] == (
        evaluation_cli.PYTHON_BIN,
        "-m",
        "ncp_olmo_eval.portable_tasks",
        "gsm8k",
    )
    assert status["state"] == "Planned"
    assert result["task_plan"] == str(Path(task["task_path"]).parent.parent / "plan.json")
    plan = json.loads(Path(result["task_plan"]).read_text())
    assert plan["tasks"] == [task["task_path"]]


def test_task_specs_reject_serialized_secrets(tmp_path: Path) -> None:
    spec = TaskSpec(
        task_id="safe-task",
        phase="test",
        benchmark="gsm8k",
        role="test",
        argv=("true",),
        cwd=str(tmp_path.resolve()),
        env={"OPENAI_API_KEY": "must-not-be-serialized"},
        resources=Resources(),
        output_root=str((tmp_path / "out").resolve()),
        status_path=str((tmp_path / "status.json").resolve()),
        log_path=str((tmp_path / "task.log").resolve()),
    )
    with pytest.raises(ValueError, match="credentials"):
        spec.as_json()


def test_task_specs_reject_mutable_container_tags(tmp_path: Path) -> None:
    spec = TaskSpec(
        task_id="unsafe-image-task",
        phase="test",
        benchmark="gsm8k",
        role="test",
        argv=("true",),
        cwd=str(tmp_path.resolve()),
        env={},
        resources=Resources(),
        output_root=str((tmp_path / "out").resolve()),
        status_path=str((tmp_path / "status.json").resolve()),
        log_path=str((tmp_path / "task.log").resolve()),
        container_image="example.org/evaluator:latest",
    )
    with pytest.raises(ValueError, match="immutable OCI"):
        spec.as_json()


def test_portable_scripts_have_valid_syntax() -> None:
    root = Path(__file__).resolve().parents[2]
    for script in sorted((root / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(
        ["python", "-m", "py_compile", str(root / "scripts/render-kubernetes-jobs.py")],
        check=True,
    )


def test_published_tree_has_no_site_specific_submission_or_storage_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "r" + "job",
        "h" + "cluster",
        "kube" + "brain",
        "/mnt/shared" + "-storage",
        "registry.h." + "pj" + "lab",
        "http" + "proxy-headless",
        "gp" + "fs://gpfs2",
    )
    paths = [
        path
        for parent in (root / "src", root / "scripts", root / "configs", root / "docs")
        for path in parent.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert not any(value.lower() in text for value in forbidden), path
