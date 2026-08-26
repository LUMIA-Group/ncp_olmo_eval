from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from ncp_olmo_eval import evaluation_cli, unified_eval_results


def _checkpoint(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "ncp_olmo3", "architectures": ["NcpOlmoForCausalLM"]}),
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


def test_sciq_protocol_matches_release_contract() -> None:
    protocol = evaluation_cli.protocol_for("vllm", "sciq")

    assert protocol == {
        "backend": "vllm",
        "runtime_backend": "native_vllm",
        "batch_size": 8,
        "sample_setting": 0,
        "resume": True,
        "gpus_per_job": 8,
        "request_type": "loglikelihood",
        "metric": "acc",
        "num_fewshot": 0,
        "samples_per_example": 1,
        "sampling": False,
        "global_seed": 42,
        "machine_count": 1,
        "source_profile": "all_supported_local",
        "task_order": 348,
        "task_name": "olmo_eval_sciq",
        "example_count": 1000,
        "source_file": "all_supported_local/348_olmo_eval_sciq.jsonl.gz",
        "source_sha256": evaluation_cli.SCIQ_SOURCE_SHA256,
    }


def test_sciq_dry_run_is_one_eight_gpu_pool_job(
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
        benchmark="sciq",
        data_root=None,
        executor="emit",
        force=False,
        dry_run=True,
    )

    assert len(result["submitted"]) == 1
    job = result["jobs"][0]
    assert job["role"] == "sciq"
    assert job["command"]["resources"] == {
        "gpus": 8,
        "cpus": 96,
        "memory_gib": 512,
        "nodes": 1,
    }
    argv = job["command"]["argv"]
    assert argv[:3] == [evaluation_cli.PYTHON_BIN, "-m", "ncp_olmo_eval.core_native_pool"]
    assert argv[argv.index("--profile") + 1] == "all_supported_local"
    assert argv[argv.index("--machine-count") + 1] == "1"
    assert argv[argv.index("--global-seed") + 1] == "42"
    assert argv[argv.index("--score-batch-size") + 1] == "8"
    assert argv[argv.index("--vllm-max-model-len") + 1] == "8192"
    assert not (evaluation_root / registration["registration_name"] / "sciq").exists()


def test_official_metric_list_selects_raw_accuracy() -> None:
    pytest.importorskip("torch")
    from ncp_olmo_eval.core_native_eval import _primary_metric

    contract = [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
    ]

    assert _primary_metric(contract) == "acc"


def test_manifest_accepts_reordered_full_profile_only_for_sealed_subset(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ncp_olmo_eval.core_native_eval import _load_manifest

    metric_contract = [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True},
    ]
    relative = "all_supported_local/348_olmo_eval_sciq.jsonl.gz"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task_order": 348,
                    "task": "olmo_eval_sciq",
                    "example_id": "sciq-0",
                    "metric": metric_contract,
                    "request_type": "loglikelihood",
                }
            )
            + "\n"
        )
    tasks = [
        {
            "task_order": order,
            "task": "olmo_eval_sciq" if order == 348 else f"task-{order}",
            "file": relative if order == 348 else f"unused/{order:03d}.jsonl.gz",
        }
        for order in range(1, 352)
    ]
    tasks = [task for task in tasks if task["task_order"] not in {32, 33}] + [
        task for task in tasks if task["task_order"] in {32, 33}
    ]
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {"all_supported_local": {"task_count": 351, "tasks": tasks, "failures": []}}
        ),
        encoding="utf-8",
    )

    _, selected = _load_manifest(tmp_path, "all_supported_local", {348})

    assert selected[0]["metric"] == "acc"
    assert selected[0]["metric_definitions"] == metric_contract
    with pytest.raises(RuntimeError, match="task order is not the fixed"):
        _load_manifest(tmp_path, "all_supported_local", set())


def test_aggregate_gold_loader_can_select_one_task(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ncp_olmo_eval.core_native_aggregate import _load_gold

    tasks = []
    for order in (1, 2):
        relative = f"profile/{order:03d}.jsonl.gz"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_order": order,
                        "task": f"task-{order}",
                        "example_id": f"example-{order}",
                        "metric": "acc",
                        "request_type": "loglikelihood",
                    }
                )
                + "\n"
            )
        tasks.append(
            {
                "task_order": order,
                "task": f"task-{order}",
                "file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    gold, selected = _load_gold(
        tmp_path,
        "profile",
        {"profile": {"task_count": 2, "tasks": tasks, "failures": []}},
        {2},
    )

    assert [task["task_order"] for task in selected] == [2]
    assert set(gold) == {(2, '"example-2"')}


def _write_sciq_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    source_task = {
        "task_order": evaluation_cli.SCIQ_TASK_ORDER,
        "task": evaluation_cli.SCIQ_TASK_NAME,
        "file": evaluation_cli.SCIQ_SOURCE_FILE,
        "sha256": evaluation_cli.SCIQ_SOURCE_SHA256,
        "num_examples": evaluation_cli.SCIQ_EXAMPLE_COUNT,
        "metric": "acc",
        "request_type": "loglikelihood",
    }
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "profile": evaluation_cli.SCIQ_SOURCE_PROFILE,
                "task_orders": [evaluation_cli.SCIQ_TASK_ORDER],
                "global_seed": 42,
                "machine_count": 1,
                "gpus_per_machine": 8,
                "hf_backend_requested": "native_vllm",
                "score_batch_size": 8,
                "generation_batch_size": 8,
                "processes_per_gpu": 1,
                "tasks": [source_task],
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "CORE_NATIVE_POOL_OK",
                "task_count": 1,
                "prediction_task_count_complete": 1,
                "score_task_count_complete": 1,
                "expected_predictions": 1000,
                "observed_predictions": 1000,
                "tasks": [
                    {
                        "task_order": 348,
                        "task": "olmo_eval_sciq",
                        "metric": "acc",
                        "request_type": "loglikelihood",
                        "score_status": "SCORED",
                        "primary_score": 0.75,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    machine = root / "machine-00"
    predictions = machine / "predictions"
    predictions.mkdir(parents=True)
    (machine / "result.json").write_text(
        json.dumps(
            {
                "status": "CORE_NATIVE_MACHINE_OK",
                "artifact_mutated": False,
                "source_artifact_mutated": False,
                "planned_work_items": 1000,
                "completed_work_items": 1000,
            }
        ),
        encoding="utf-8",
    )
    with (predictions / "348_olmo_eval_sciq.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(1000):
            handle.write(
                json.dumps(
                    {"task_order": 348, "task": "olmo_eval_sciq", "example_id": index}
                )
                + "\n"
            )


def test_sciq_artifact_validation_requires_full_coverage(tmp_path: Path) -> None:
    inference = tmp_path / "inference"
    _write_sciq_fixture(inference)

    artifacts = evaluation_cli._artifact_status("sciq", {"attempt_root": str(inference)})

    assert artifacts["complete"] is True
    assert artifacts["prediction_count"] == 1000
    predictions = inference / "machine-00/predictions/348_olmo_eval_sciq.jsonl"
    rows = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    assert evaluation_cli._artifact_status(
        "sciq", {"attempt_root": str(inference)}
    )["complete"] is False


def test_sciq_score_and_final_artifacts_require_raw_acc(tmp_path: Path) -> None:
    score_root = tmp_path / "score"
    score_root.mkdir()
    payload = {
        "status": "CORE_NATIVE_FULL_OK",
        "profile": "all_supported_local",
        "task_orders": [348],
        "expected_predictions": 1000,
        "observed_predictions": 1000,
        "tasks": [
            {
                "task_order": 348,
                "task": "olmo_eval_sciq",
                "metric": "acc",
                "request_type": "loglikelihood",
                "score_status": "SCORED",
                "primary_score": 0.75,
            }
        ],
    }
    (score_root / "score.json").write_text(json.dumps(payload), encoding="utf-8")
    (score_root / "sciq-score.csv").write_text(
        "task_order,task,primary_score\n348,olmo_eval_sciq,0.75\n", encoding="utf-8"
    )
    (score_root / "_SUCCESS").touch()

    assert evaluation_cli._score_artifacts(
        "sciq", {"score_root": str(score_root)}
    )["complete"] is True
    result = unified_eval_results.finalize_result("sciq", score_root, tmp_path / "final")
    assert result["benchmark"] == "sciq"
    assert (tmp_path / "final/sciq-score.csv").is_file()

    payload["tasks"][0]["metric"] = "acc_norm"
    (score_root / "score.json").write_text(json.dumps(payload), encoding="utf-8")
    assert evaluation_cli._score_artifacts(
        "sciq", {"score_root": str(score_root)}
    )["complete"] is False
