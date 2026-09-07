from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ncp_olmo_eval import evaluation_cli
from ncp_olmo_eval.dflash_checkpoint import dflash_checkpoint_identity
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


def _draft_checkpoint(root: Path) -> Path:
    draft = root / "draft"
    draft.mkdir()
    config = {
        "model_type": "conceptlm_dflash",
        "proposal_method": "path_selector",
        "hlm_conditioning": "causal_residual",
        "concept_chunk_size": 4,
        "target_layer_ids": [1, 4, 7, 10, 13],
        "block_size": 16,
    }
    (draft / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"draft-weights")
    return draft


def _sharded_draft_checkpoint(root: Path, *, shard_count: int = 21) -> Path:
    draft = _draft_checkpoint(root)
    (draft / "model.safetensors").unlink()
    weight_map = {}
    for index in range(1, shard_count + 1):
        name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        (draft / name).write_bytes(f"draft-shard-{index}".encode())
        weight_map[f"draft.layer.{index}.weight"] = name
    (draft / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return draft


def _verification_artifact(
    root: Path,
    *,
    target: Path,
    draft: Path,
    mode: str = "sequential_exact",
) -> Path:
    approximate = mode == "segmented_kv_approx"
    operating_point = {
        "schema_version": "ncp-dflash-operating-point-v1",
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "scheduler_queue_size": 32,
        "continuous_batching_enabled": True,
        "tensor_parallel_size": 1,
        "execution_mode": "eager",
        "gpu_memory_utilization": 0.8,
        "attention_backend": "FLASH_ATTN",
        "flash_attn_version": 3,
        "hlm_attention_impl": "legacy_mixed",
        "vllm_use_v2_model_runner": "0",
        "speculative_num_tokens": 8,
        "speculative_verification_mode": mode,
        "draft_attention_backend": "flash_varlen",
        "context_kv_cache": True,
        "sparse_context_projection": True,
        "min_eligible_batch": 1,
        "min_proposal_tokens_per_row": 1,
        "min_proposal_tokens_per_batch": 1,
        "runtime_block_size": 0,
        "active_batch_widths": "1:8,2:8,4:4,8:2",
        "dynamic_runtime_block_size": True,
        "runtime_layer_count": 5,
        "runtime_local_mixer": "full",
        "mixer_compile_mode": "default",
        "chunk_size": 4,
        "target_layers": "1,4,7,10,13",
        "telemetry_flush_interval": 8,
    }
    payload = {
        "status": (
            "NCP_DFLASH_VLLM_APPROXIMATE_AB_OK"
            if approximate
            else "NCP_DFLASH_VLLM_EXACT_MATCH_OK"
        ),
        "vllm_version": "0.13.0",
        "speculative_verification_mode": mode,
        "speculative_output_contract": "approximate" if approximate else "target_exact",
        "downstream_score_required": approximate,
        "exact_token_match_count": 31 if approximate else 32,
        "comparison_count": 32,
        "generated_token_count": 4096,
        "throughput_speedup": 1.2 if approximate else 1.0,
        "benchmark_contract": {
            "seed": 42,
            "prompt_count": 32,
            "max_new_tokens": 128,
            "batch_size": 8,
            "scheduler_queue_size": 32,
            "gpu_memory_utilization": 0.8,
            "vllm_use_v2_model_runner": "0",
            "ignore_eos": True,
        },
        "speculative_operating_point": operating_point,
        "target_model_identity": {"source_model": str(target.resolve())},
        "draft_model_identity": dflash_checkpoint_identity(draft),
    }
    path = root / f"comparison-{mode}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_speculative_registration_is_target_and_draft_bound(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _draft_checkpoint(tmp_path)
    verification = _verification_artifact(
        tmp_path, target=checkpoint, draft=draft
    )

    registration = evaluation_cli.register_model(
        root=evaluation_root,
        checkpoint=checkpoint,
        backend="vllm",
        new_version=False,
        vllm_speculative_draft_model=draft,
        vllm_speculative_verification=verification,
    )

    assert registration["vllm_speculative_correctness_status"] == "EXACT_MATCH_VERIFIED"
    assert registration["vllm_speculative_verification"]["vllm_version"] == "0.13.0"
    operating_point = registration["vllm_speculative_verification"][
        "speculative_operating_point"
    ]
    assert operating_point["max_num_seqs"] == 8
    assert operating_point["scheduler_queue_size"] == 32
    assert operating_point["active_batch_widths"] == "1:8,2:8,4:4,8:2"
    _, loaded = evaluation_cli.load_registration(
        evaluation_root, registration["registration_name"]
    )
    assert loaded["vllm_speculative_draft_model"] == str(draft.resolve())


def test_speculative_registration_accepts_21_shard_hf_draft(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _sharded_draft_checkpoint(tmp_path)
    verification = _verification_artifact(tmp_path, target=checkpoint, draft=draft)

    registration = evaluation_cli.register_model(
        root=evaluation_root,
        checkpoint=checkpoint,
        backend="vllm",
        new_version=False,
        vllm_speculative_draft_model=draft,
        vllm_speculative_verification=verification,
    )

    sealed = registration["vllm_speculative_draft_validation"]
    assert sealed["weight_file_count"] == 21
    assert len(sealed["weight_files"]) == 21
    _, loaded = evaluation_cli.load_registration(
        evaluation_root, registration["registration_name"]
    )
    assert loaded["vllm_speculative_draft_validation"]["weight_index_sha256"]


def test_approximate_speculative_registration_requires_opt_in(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _draft_checkpoint(tmp_path)
    verification = _verification_artifact(
        tmp_path,
        target=checkpoint,
        draft=draft,
        mode="segmented_kv_approx",
    )

    with pytest.raises(evaluation_cli.EvaluationError, match="显式设置"):
        evaluation_cli.register_model(
            root=evaluation_root,
            checkpoint=checkpoint,
            backend="vllm",
            new_version=False,
            vllm_speculative_draft_model=draft,
            vllm_speculative_verification=verification,
        )


def test_speculative_registration_rejects_unsealed_legacy_artifact(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _draft_checkpoint(tmp_path)
    verification = _verification_artifact(tmp_path, target=checkpoint, draft=draft)
    payload = json.loads(verification.read_text(encoding="utf-8"))
    del payload["speculative_operating_point"]
    verification.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(evaluation_cli.EvaluationError, match="operating point"):
        evaluation_cli.register_model(
            root=evaluation_root,
            checkpoint=checkpoint,
            backend="vllm",
            new_version=False,
            vllm_speculative_draft_model=draft,
            vllm_speculative_verification=verification,
        )


def test_continuous_batch_registration_requires_a_full_verified_queue(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _draft_checkpoint(tmp_path)
    verification = _verification_artifact(tmp_path, target=checkpoint, draft=draft)
    payload = json.loads(verification.read_text(encoding="utf-8"))
    payload["benchmark_contract"]["prompt_count"] = 8
    verification.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(evaluation_cli.EvaluationError, match="full scheduler queue"):
        evaluation_cli.register_model(
            root=evaluation_root,
            checkpoint=checkpoint,
            backend="vllm",
            new_version=False,
            vllm_speculative_draft_model=draft,
            vllm_speculative_verification=verification,
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


def test_speculative_core88_dry_run_propagates_sealed_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_root = tmp_path / "evaluations"
    checkpoint = _checkpoint(tmp_path)
    draft = _draft_checkpoint(tmp_path)
    registration = evaluation_cli.register_model(
        root=evaluation_root,
        checkpoint=checkpoint,
        backend="vllm",
        new_version=False,
        vllm_speculative_draft_model=draft,
        vllm_speculative_verification=_verification_artifact(
            tmp_path, target=checkpoint, draft=draft
        ),
    )
    monkeypatch.setattr(evaluation_cli, "_git_state", lambda _: _clean_git_state(tmp_path))
    monkeypatch.setenv("CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS", "8:0")
    monkeypatch.setenv("CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT", "1")

    result = evaluation_cli.submit_inference(
        root=evaluation_root,
        registration_name=registration["registration_name"],
        benchmark="core88",
        data_root=None,
        executor="emit",
        force=False,
        dry_run=True,
    )

    for task in result["jobs"][:4]:
        argv = task["command"]["argv"]
        assert argv[argv.index("--score-batch-size") + 1] == "8"
        assert argv[argv.index("--generation-batch-size") + 1] == "8"
        assert argv[argv.index("--vllm-scheduler-queue-size") + 1] == "32"
        assert argv[argv.index("--vllm-speculative-num-tokens") + 1] == "8"
        assert argv[argv.index("--vllm-speculative-draft-model") + 1] == str(
            draft.resolve()
        )
        env = task["command"]["env"]
        assert env["CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS"] == "1:8,2:8,4:4,8:2"
        assert env["CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT"] == "5"
    gsm8k_argv = result["jobs"][4]["command"]["argv"]
    assert gsm8k_argv[gsm8k_argv.index("--batch-size") + 1] == "8"
    assert gsm8k_argv[gsm8k_argv.index("--scheduler-queue-size") + 1] == "32"
    assert gsm8k_argv[gsm8k_argv.index("--speculative-num-tokens") + 1] == "8"
    assert gsm8k_argv[gsm8k_argv.index("--speculative-draft-model") + 1] == str(
        draft.resolve()
    )
    with pytest.raises(evaluation_cli.EvaluationError, match="拒绝"):
        evaluation_cli._protocol_for_registration(registration, "ruler")


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


def test_bigcodebench_score_task_seals_explicit_olmo_eval_root(tmp_path: Path) -> None:
    group = next(
        item for item in evaluation_cli.CORE_SCORER_GROUPS if int(item["index"]) == 1
    )
    command = evaluation_cli._score_command(
        {},
        "core88",
        {"attempt_root": str(tmp_path / "inference")},
        tmp_path / "scores",
        "score-bigcodebench",
        group=group,
        partition_index=0,
        partition_count=int(group["partition_count"]),
    )
    assert command.env["OLMO_EVAL_COMMIT"] == evaluation_cli.OLMO_EVAL_COMMIT
    assert command.env["OLMO_EVAL_ROOT"] == str(evaluation_cli.OLMO_EVAL_ROOT)


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
