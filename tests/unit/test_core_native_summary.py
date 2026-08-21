import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ncp_olmo_eval import core88_workflow as workflow_module
from ncp_olmo_eval.core88_workflow import create_workflow, seal_companion
from ncp_olmo_eval.core_native_aggregate import _validate_evaluator_repo_state
from ncp_olmo_eval.core_native_aggregate import parse_args as parse_aggregate_args
from ncp_olmo_eval.core_native_answer_eval import OFFICIAL_OLMO_EVAL_COMMIT
from ncp_olmo_eval.core_native_summary import (
    SUMMARY_HEADERS,
    _same_run_request_seed,
    build_core88_summary,
    write_core88_summary_csv,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_core88_aggregate_keeps_detailed_88_only_cli_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_native_aggregate",
            "--data-root",
            str(tmp_path / "data"),
            "--profile",
            "core88",
            "--input-root",
            str(tmp_path / "input"),
            "--output-json",
            str(tmp_path / "result.json"),
        ],
    )

    args = parse_aggregate_args()

    assert args.output_summary_csv is None
    assert args.workflow_manifest is None
    assert args.gsm8k_results_root is None


def test_core88_aggregate_final_summary_requires_same_run_gsm8k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core_native_aggregate",
            "--data-root",
            str(tmp_path / "data"),
            "--profile",
            "core88",
            "--input-root",
            str(tmp_path / "input"),
            "--output-json",
            str(tmp_path / "result.json"),
            "--output-summary-csv",
            str(tmp_path / "summary.csv"),
        ],
    )

    with pytest.raises(SystemExit):
        parse_aggregate_args()


def test_evaluator_repo_state_must_match_workflow(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    workflow = {"repo_root": str(root), "repo_commit": "a" * 40}
    state = {
        "evaluator_repo_root": str(root),
        "evaluator_repo_commit": "a" * 40,
        "evaluator_repo_dirty": False,
    }

    _validate_evaluator_repo_state(workflow, state)

    with pytest.raises(RuntimeError, match="differs"):
        _validate_evaluator_repo_state(workflow, {**state, "evaluator_repo_commit": "b" * 40})
    with pytest.raises(RuntimeError, match="clean"):
        _validate_evaluator_repo_state(workflow, {**state, "evaluator_repo_dirty": True})


def test_evaluator_repo_allows_only_audited_scoring_descendant(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    scoring_file = root / "experiments/vllm_inference/core_native_summary.py"
    scoring_file.parent.mkdir(parents=True)
    scoring_file.write_text("base = True\n", encoding="utf-8")
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "core88@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Core88 Test"], check=True
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "base"], check=True)
    base_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    scoring_file.write_text("base = True\nfixed = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "scoring fix"], check=True)
    scoring_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    workflow = {"repo_root": str(root), "repo_commit": base_commit}
    state = {
        "evaluator_repo_root": str(root),
        "evaluator_repo_commit": scoring_commit,
        "evaluator_repo_dirty": False,
    }

    proof = _validate_evaluator_repo_state(workflow, state, allow_descendant=True)

    assert proof["status"] == workflow_module.SCORING_DESCENDANT_STATUS
    assert proof["changed_files"] == [
        "experiments/vllm_inference/core_native_summary.py"
    ]

    (root / "README.md").write_text("not a scoring-only change\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "unrelated"], check=True)
    unrelated_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(RuntimeError, match="non-scoring"):
        _validate_evaluator_repo_state(
            workflow,
            {**state, "evaluator_repo_commit": unrelated_commit},
            allow_descendant=True,
        )


def _core88_report(
    tmp_path: Path,
    *,
    input_root: Path | None = None,
    model_path: str | Path = "/model",
    workflow_id: str | None = None,
    repo_commit: str | None = None,
) -> dict:
    input_root = input_root or tmp_path / "core88-input"
    prediction_path = input_root / "shard-00" / "predictions" / "derived.jsonl"
    (input_root / "run_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (input_root / "run_manifest.json").write_text(
        json.dumps({"workflow_id": workflow_id, "repo_commit": repo_commit}) + "\n",
        encoding="utf-8",
    )

    mmlu_layout = {33: ("humanities", 13), 37: ("other", 14), 41: ("social", 12), 45: ("stem", 18)}
    rows = []
    observed_predictions = {}
    for task_order, (prefix, subject_count) in mmlu_layout.items():
        observed_predictions[task_order] = subject_count
        for subject_index in range(subject_count):
            rows.append(
                {
                    "task_order": task_order,
                    "example_id": f"{prefix}_{subject_index}:0",
                    "sample_index": 0,
                    "primary_correct": True,
                }
            )
    rows.extend(
        [
            {
                "task_order": 23,
                "example_id": "lambada-0",
                "sample_index": 0,
                "candidates": [{"is_greedy": True}],
            },
            {
                "task_order": 23,
                "example_id": "lambada-1",
                "sample_index": 0,
                "candidates": [{"is_greedy": False}],
            },
        ]
    )
    observed_predictions[23] = 2
    _write_jsonl(prediction_path, rows)

    scores = {order: 0.5 for order in range(1, 89)}
    scores[75] = 0.3
    for order in range(26, 33):
        scores[order] = 0.2
    for index, order in enumerate(range(81, 88), start=1):
        scores[order] = index / 10
    tasks = [
        {
            "task_order": order,
            "task": f"task-{order}",
            "primary_score": scores[order],
            "score_status": "SCORED",
            "score_complete": True,
            "observed_predictions": observed_predictions.get(order, 1),
        }
        for order in range(1, 89)
    ]
    return {
        "status": "CORE_NATIVE_FULL_OK",
        "profile": "core88",
        "model_label": "test-model",
        "workflow_id": workflow_id,
        "repo_commit": repo_commit,
        "hf_model_path": str(model_path),
        "runtime_hf_model_paths": [str(model_path)],
        "input_roots": [str(input_root)],
        "tasks": tasks,
    }


def _gsm8k_root(tmp_path: Path) -> Path:
    root = tmp_path / "gsm8k"
    shard_rows = {
        0: [
            (0, True, "Therefore the answer is 42.", "42"),
            (2, False, "This also gives 42.", "42"),
        ],
        1: [(1, False, "The answer is 41.", "42"), (3, True, "We obtain 42.", "42")],
    }
    for shard_index, rows in shard_rows.items():
        shard_root = root / f"shard-{shard_index:02d}"
        predictions_path = shard_root / "predictions.jsonl"
        _write_jsonl(
            predictions_path,
            [
                {
                    "doc_index": doc_index,
                    "standard_correct": correct,
                    "output": output,
                    "gold": gold,
                }
                for doc_index, correct, output, gold in rows
            ],
        )
        result = {
            "status": "GSM8K_EVAL_OK",
            "hf_model_path": "/model",
            "artifact_mutated": False,
            "source_model_mutated": False,
            "shard_count": 2,
            "shard_index": shard_index,
            "dataset_sample_count": 4,
            "sample_count": 2,
            "pass_at_1_correct": sum(int(correct) for _, correct, _, _ in rows),
            "predictions_jsonl": str(predictions_path),
        }
        (shard_root / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    return root


def _write_dispatch_plan(output_root: Path, tmp_path: Path) -> Path:
    plan_root = output_root / "dispatch-plan"
    plan_root.mkdir(parents=True)
    plan = {
        "plan_id": f"plan-{output_root.name}",
        "profile": "core88",
        "data_root": str((tmp_path / "data").resolve()),
        "data_summary_sha256": "b" * 64,
        "task_orders": list(range(1, 89)),
        "limit_per_task": 0,
        "generation_samples_cap": 8,
        "max_gen_tokens_cap": 0,
        "machine_count": 4,
        "global_seed": 20260727,
        "local_dispatch": "dynamic-request-batch-pull-v2",
    }
    plan_json = plan_root / "plan.json"
    plan_json.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    return plan_json


def _same_run_gsm8k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, backend: str = "native_vllm"
) -> tuple[dict, Path, Path]:
    standard_input = tmp_path / "standard-input.json"
    dataset = tmp_path / "gsm8k-test.parquet"
    standard_input.write_text("{}\n", encoding="utf-8")
    dataset.write_bytes(b"test dataset")
    input_rows = [
        {
            "doc_index": doc_index,
            "prompt": f"Question {doc_index}",
            "question": f"Question {doc_index}",
            "gold_answer": "42",
        }
        for doc_index in range(16)
    ]
    rendered_inputs = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in input_rows)
    fixed_values = {
        "CANONICAL_STANDARD_INPUT_CONFIG": standard_input,
        "CANONICAL_STANDARD_INPUT_CONFIG_SHA256": hashlib.sha256(
            standard_input.read_bytes()
        ).hexdigest(),
        "CANONICAL_DATASET_TEST_FILE": dataset,
        "CANONICAL_DATASET_TEST_SHA256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "CANONICAL_PROMPT_SHA256": workflow_module._prompt_sha256(
            [str(row["prompt"]) for row in input_rows]
        ),
        "CANONICAL_INPUTS_JSONL_SHA256": hashlib.sha256(rendered_inputs.encode()).hexdigest(),
        "GSM8K_EXAMPLE_COUNT": len(input_rows),
    }
    for name, value in fixed_values.items():
        monkeypatch.setattr(workflow_module, name, value)
    monkeypatch.setattr(workflow_module, "_validate_repo_state", lambda *_: None)

    model = tmp_path / "model"
    runtime_model = tmp_path / "runtime-model"
    output_root = tmp_path / "workflow"
    workflow = create_workflow(
        output_root=output_root,
        repo_root=tmp_path / "repo",
        repo_commit="a" * 40,
        hf_model_path=model,
        model_identity_path=model,
        model_label="test-model",
        run_tag="0804v1",
        core_job_names=[f"core-{rank}" for rank in range(4)],
        gsm8k_job_name="gsm8k",
        global_seed=20260727,
        core_plan_json=_write_dispatch_plan(output_root, tmp_path),
        hf_backend=backend,
        processes_per_gpu=1 if backend == "lmdeploy" else 4,
        score_batch_size=8 if backend == "lmdeploy" else 1,
        generation_batch_size=8 if backend == "lmdeploy" else 1,
        row_chunk_size=8 if backend == "lmdeploy" else 1,
    )
    workflow_manifest = Path(workflow["output_root"]) / "workflow.json"
    protocol = workflow["gsm8k_protocol"]
    example_count = int(protocol["dataset_sample_count"])
    report = _core88_report(
        tmp_path,
        input_root=Path(workflow["core_results_root"]),
        model_path=model,
        workflow_id=workflow["workflow_id"],
        repo_commit="a" * 40,
    )
    report.update(
        evaluator_repo_root=str(Path(workflow["repo_root"]).resolve()),
        evaluator_repo_commit="a" * 40,
        evaluator_repo_dirty=False,
    )
    root = Path(workflow["gsm8k_results_root"])
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    inputs_jsonl = inputs / "inputs.jsonl"
    inputs_jsonl.write_text(rendered_inputs, encoding="utf-8")
    input_manifest = {
        "status": "GSM8K_INPUTS_READY",
        "task": protocol["task"],
        "task_group": protocol["task_group"],
        "standard_input_config": protocol["standard_input_config"],
        "standard_input_config_sha256": protocol["standard_input_config_sha256"],
        "dataset_test_file": protocol["dataset_test_file"],
        "dataset_test_sha256": protocol["dataset_test_sha256"],
        "dataset_sample_count": example_count,
        "fewshot_seed": protocol["fewshot_seed"],
        "prompt_sha256": protocol["prompt_sha256"],
        "inputs_jsonl": str(inputs_jsonl.resolve()),
        "inputs_jsonl_sha256": protocol["inputs_jsonl_sha256"],
    }
    (inputs / "manifest.json").write_text(json.dumps(input_manifest) + "\n", encoding="utf-8")
    (inputs / "_SUCCESS").touch()

    predictions = [
        {
            "doc_index": doc_index,
            "sample_index": 0,
            "sample_seed": _same_run_request_seed(doc_index),
            "output": "Therefore the answer is 42.",
            "gold_answer": "42",
        }
        for doc_index in range(example_count)
    ]
    _write_jsonl(root / "predictions.jsonl", predictions)
    for rank in range(8):
        shard = root / f"shard-{rank:02d}"
        shard.mkdir()
        result = {
            "status": (
                "GSM8K_LMDEPLOY_GENERATION_OK"
                if backend == "lmdeploy"
                else "GSM8K_VLLM_GENERATION_OK"
            ),
            "rank": rank,
            "world_size": 8,
            "dataset_sample_count": example_count,
            "sample_count": len(range(rank, example_count, 8)),
            "samples_per_doc": 1,
            "batch_size": 8,
            "max_num_seqs": 8,
            "max_model_len": 65536 if backend == "lmdeploy" else 2048,
            "sampling_seed": 1234,
            "model_family": "conceptlm" if backend == "lmdeploy" else "auto",
        }
        (shard / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
        (shard / "_SUCCESS").touch()
    aggregate = {
        "status": ("GSM8K_LMDEPLOY_EVAL_OK" if backend == "lmdeploy" else "GSM8K_VLLM_EVAL_OK"),
        "model_identifier": str(runtime_model.resolve()),
        "model_family": "conceptlm" if backend == "lmdeploy" else "auto",
        "backend": backend,
        "task": "olmo_eval_paper_gsm8k_main",
        "task_group": "olmo_eval_paper_math_gsm_8shot",
        "task_contract": {
            "num_fewshot": 8,
            "fewshot_sampler": "first_n",
            "fixed_fewshot_samples": True,
            "generation_kwargs": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.6,
                "max_gen_toks": 512,
                "until": ["Question:", "</s>", "<|im_end|>"],
            },
            "runtime_repeats": 1,
        },
        "standard_input_config": protocol["standard_input_config"],
        "standard_input_config_sha256": protocol["standard_input_config_sha256"],
        "dataset_test_file": protocol["dataset_test_file"],
        "dataset_test_sha256": protocol["dataset_test_sha256"],
        "prompt_sha256": protocol["prompt_sha256"],
        "dataset_sample_count": example_count,
        "sample_count": example_count,
        "samples_per_doc": 1,
        "num_fewshot": 8,
        "fewshot_sampler": "first_n",
        "fixed_fewshot_samples": True,
        "fewshot_seed": 1234,
        "sampling_seed": 1234,
        "batch_size": 8,
        "max_num_seqs": 8,
        "max_model_len": 65536 if backend == "lmdeploy" else 2048,
        "gpu_count": 8,
        "shard_count": 8,
        "schedule_position_rule": "doc_sample_pairs_doc_major[rank::world_size]",
        "temperature": 0.6,
        "top_p": 0.6,
        "max_new_tokens": 512,
        "stop_strings": ["Question:", "</s>", "<|im_end|>"],
        "prefix_caching": False,
        "speculative_decoding": False,
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "pass_at_1_correct": example_count,
        "pass_at_1": 1.0,
        "predictions_jsonl": str((root / "predictions.jsonl").resolve()),
    }
    if backend == "lmdeploy":
        aggregate["top_k"] = 0
        aggregate["lmdeploy_runtime"] = dict(protocol["lmdeploy_config"])
        aggregate["lmdeploy_runtime"]["session_len"] = 65536
    (root / "aggregate.json").write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
    (root / "_SUCCESS").touch()
    preparation_manifest = root / "model-preparation.json"
    preparation_manifest.write_text(
        json.dumps(
            {
                "source_model": str(model.resolve()),
                "destination_model": str(runtime_model.resolve()),
                "weight_files_mutated": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seal_companion(
        workflow_path=workflow_manifest,
        gsm8k_root=root,
        repo_commit="a" * 40,
        model_identity_path=model,
        runtime_model_path=runtime_model,
        model_family="conceptlm" if backend == "lmdeploy" else "auto",
        model_preparation_manifest=preparation_manifest,
    )
    return report, root, workflow_manifest


def test_fixed_core88_summary_writes_two_evidence_backed_external_columns(tmp_path: Path) -> None:
    report = _core88_report(tmp_path)
    output_csv = tmp_path / "core88-summary-30cols.csv"

    metadata = write_core88_summary_csv(
        output_csv, report, gsm8k_results_root=_gsm8k_root(tmp_path), allow_diagnostic_gsm8k=True
    )

    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(SUMMARY_HEADERS) == 30
    assert len(rows) == 1
    assert list(rows[0]) == list(SUMMARY_HEADERS)
    assert rows[0]["MMLU（acc；57 subjects等权）"] == "100.000000"
    assert rows[0]["GSM8k（pass@1）"] == "75.000000"
    assert rows[0]["MATH Avg（3项 pass@1 等权）"] == "41.666667"
    assert rows[0]["Code avg（7项执行 pass@1 等权）"] == "40.000000"
    assert rows[0]["LAMBADA-OpenAI（greedy acc）"] == "50.000000"
    assert rows[0]["SciQ（acc）"] == "NA"
    assert metadata["status"] == "CORE88_FIXED_SUMMARY_DIAGNOSTIC"
    assert metadata["final_output_eligible"] is False
    assert metadata["gsm8k"]["pass_at_1_correct"] == 3
    assert metadata["gsm8k"]["legacy_declared_correct"] == 2
    assert metadata["gsm8k"]["legacy_scorer_disagreements"] == 1
    assert metadata["gsm8k"]["status"] == "GSM8K_EXTERNAL_RESULT_OFFICIAL_RESCORE_OK"
    assert metadata["sciq"]["status"] == "NOT_PROVIDED_CORE88_HAS_NO_SCIQ"


def test_fixed_core88_summary_marks_unprovided_external_metrics_as_na(tmp_path: Path) -> None:
    row, metadata = build_core88_summary(_core88_report(tmp_path), allow_diagnostic_gsm8k=True)

    assert row["GSM8k（pass@1）"] == "NA"
    assert row["MATH Avg（3项 pass@1 等权）"] == "NA"
    assert row["SciQ（acc）"] == "NA"
    assert metadata["gsm8k"]["status"] == "NOT_PROVIDED"
    assert metadata["status"] == "CORE88_FIXED_SUMMARY_DIAGNOSTIC"
    assert metadata["final_output_eligible"] is False


def test_fixed_core88_summary_requires_same_run_gsm8k_by_default(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="same-workflow GSM8K"):
        build_core88_summary(_core88_report(tmp_path))

    with pytest.raises(RuntimeError, match="same-workflow GSM8K"):
        build_core88_summary(_core88_report(tmp_path), gsm8k_results_root=_gsm8k_root(tmp_path))


def test_fixed_core88_summary_rejects_incomplete_scores(tmp_path: Path) -> None:
    report = _core88_report(tmp_path)
    report["status"] = "CORE_NATIVE_FULL_PREDICTIONS_OK"

    with pytest.raises(RuntimeError, match="fully scored"):
        build_core88_summary(report, allow_diagnostic_gsm8k=True)


def test_same_run_vllm_gsm8k_populates_final_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(tmp_path, monkeypatch)
    row, metadata = build_core88_summary(
        report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
    )

    assert row["GSM8k（pass@1）"] == "100.000000"
    assert metadata["gsm8k"]["status"] == "GSM8K_SAME_RUN_OFFICIAL_RESCORE_OK"
    assert metadata["gsm8k"]["workflow_id"] == report["workflow_id"]
    assert metadata["status"] == "CORE88_FIXED_SUMMARY_OK"
    assert metadata["final_output_eligible"] is True


def test_same_run_lmdeploy_gsm8k_populates_final_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(
        tmp_path, monkeypatch, backend="lmdeploy"
    )
    row, metadata = build_core88_summary(
        report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
    )

    assert row["GSM8k（pass@1）"] == "100.000000"
    assert metadata["gsm8k"]["status"] == "GSM8K_SAME_RUN_OFFICIAL_RESCORE_OK"
    assert metadata["status"] == "CORE88_FIXED_SUMMARY_OK"
    assert metadata["final_output_eligible"] is True
    companion = json.loads(
        (gsm8k_root / "core88-companion.json").read_text(encoding="utf-8")
    )
    assert companion["max_model_len"] == 65536
    assert companion["max_model_len_policy"] == "lmdeploy_runtime_default"


def test_same_run_vllm_gsm8k_rejects_another_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, _ = _same_run_gsm8k(tmp_path, monkeypatch)
    other = create_workflow(
        output_root=tmp_path / "other-workflow",
        repo_root=tmp_path / "repo",
        repo_commit="a" * 40,
        hf_model_path=tmp_path / "model",
        model_identity_path=tmp_path / "model",
        model_label="test-model",
        run_tag="0804v2",
        core_job_names=[f"other-{rank}" for rank in range(4)],
        gsm8k_job_name="other-gsm8k",
        global_seed=20260727,
        core_plan_json=_write_dispatch_plan(tmp_path / "other-workflow", tmp_path),
    )

    with pytest.raises(
        RuntimeError, match="workflow id|result root|does not belong to this workflow"
    ):
        build_core88_summary(
            report,
            gsm8k_results_root=gsm8k_root,
            workflow_manifest=Path(other["output_root"]) / "workflow.json",
        )


def test_same_run_vllm_gsm8k_rejects_post_seal_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(tmp_path, monkeypatch)
    aggregate_path = gsm8k_root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["batch_size"] = 1
    aggregate_path.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after same-run sealing"):
        build_core88_summary(
            report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
        )


def test_same_run_vllm_gsm8k_rejects_post_seal_prediction_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(tmp_path, monkeypatch)
    predictions_path = gsm8k_root / "predictions.jsonl"
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="root predictions changed"):
        build_core88_summary(
            report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
        )


def test_same_run_vllm_gsm8k_binds_prediction_gold_to_sealed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(tmp_path, monkeypatch)
    predictions_path = gsm8k_root / "predictions.jsonl"
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    predictions[0]["gold_answer"] = "41"
    _write_jsonl(predictions_path, predictions)
    companion_path = gsm8k_root / "core88-companion.json"
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    companion["predictions_jsonl_sha256"] = hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    companion_path.write_text(json.dumps(companion) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="gold differs from sealed inputs"):
        build_core88_summary(
            report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
        )


def test_degraded_math_runtime_cannot_produce_final_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, gsm8k_root, workflow_manifest = _same_run_gsm8k(tmp_path, monkeypatch)
    report["allow_degraded_math_runtime"] = True
    report["status"] = "CORE_NATIVE_FULL_DIAGNOSTIC"

    _, metadata = build_core88_summary(
        report, gsm8k_results_root=gsm8k_root, workflow_manifest=workflow_manifest
    )

    assert metadata["status"] == "CORE88_FIXED_SUMMARY_DIAGNOSTIC"
    assert metadata["final_output_eligible"] is False
    assert metadata["allow_degraded_math_runtime"] is True
