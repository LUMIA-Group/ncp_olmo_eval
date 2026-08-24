#!/usr/bin/env python3
"""Build the fixed 30-column Core88 comparison table from verified artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core88_workflow import (
    COMPANION_STATUS,
    GSM8K_SEED,
    file_sha256,
    load_workflow,
    require_formal_workflow,
    validate_gsm8k_max_model_len,
    validate_scoring_evaluator_repo,
    verify_inputs,
)
from .core_native_answer_eval import OFFICIAL_OLMO_EVAL_COMMIT, score_gsm_answer

SUMMARY_HEADERS = (
    "MMLU（acc；57 subjects等权）",
    "GSM8k（pass@1）",
    "GSM-Symbolic avg（pass@1；3子集等权）",
    "Minerva均值（pass@1；7类等权）",
    "MATH Avg（3项 pass@1 等权）",
    "MATH-500（pass@1）",
    "BigCodeBench（执行 pass@1）",
    "HumanEval（执行 pass@1）",
    "DeepSeek LeetCode（执行 pass@1）",
    "DS-1000（执行 pass@1）",
    "MBPP（执行 pass@1）",
    "MultiPL-E HumanEval（执行 pass@1；6语言等权）",
    "MultiPL-E MBPP（执行 pass@1；6语言等权）",
    "Code avg（7项执行 pass@1 等权）",
    "ARC-E（acc）",
    "ARC-C（acc）",
    "MMLU-STEM（acc；subject等权）",
    "SciQ（acc）",
    "MMLU-Humanities（acc；subject等权）",
    "MMLU-Social Sciences（acc；subject等权）",
    "MMLU-Other（acc；subject等权）",
    "PiQA（acc）",
    "HellaSwag（acc_per_char）",
    "WinoGrande（acc_raw）",
    "LAMBADA-Standard（greedy acc）",
    "LAMBADA-OpenAI（greedy acc）",
    "MMLU-Pro（acc）",
    "DeepMind-Math（exact match）",
    "BBH（exact match）",
    "LBPP（执行 pass@1）",
)

_MMLU_CATEGORY_BY_TASK_ORDER = {33: "humanities", 37: "other", 41: "social_sciences", 45: "stem"}
_EXPECTED_MMLU_SUBJECT_COUNTS = {"humanities": 13, "other": 14, "social_sciences": 12, "stem": 18}
_MINERVA_TASK_ORDERS = tuple(range(26, 33))
_CODE_TASK_ORDERS = tuple(range(81, 88))
FINAL_SUMMARY_STATUS = "CORE88_FIXED_SUMMARY_OK"
DIAGNOSTIC_SUMMARY_STATUS = "CORE88_FIXED_SUMMARY_DIAGNOSTIC"


def _stable_id(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse the standalone Core88 summary CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--output-metadata-json",
        type=Path,
        default=None,
        help="Optional machine-readable evidence for every derived field.",
    )
    parser.add_argument(
        "--gsm8k-results-root",
        type=Path,
        default=None,
        help="Optional separately validated paper-8-shot GSM8K result root.",
    )
    parser.add_argument(
        "--workflow-manifest",
        type=Path,
        default=None,
        help="Same-run Core88 workflow manifest for production GSM8K evidence.",
    )
    parser.add_argument(
        "--allow-diagnostic-gsm8k",
        action="store_true",
        help=(
            "Allow a missing or legacy GSM8K artifact for diagnostic output. "
            "Such output is never marked as a final Core88 summary."
        ),
    )
    parser.add_argument(
        "--sciq-score",
        type=float,
        default=None,
        help=(
            "Optional SciQ accuracy in [0,1]. Core88 does not contain SciQ, so the "
            "column is NA unless this independent score is supplied."
        ),
    )
    args = parser.parse_args()
    has_same_run_gsm8k = args.gsm8k_results_root is not None and args.workflow_manifest is not None
    if not has_same_run_gsm8k and not args.allow_diagnostic_gsm8k:
        parser.error(
            "final Core88 summary requires --workflow-manifest and "
            "--gsm8k-results-root; use --allow-diagnostic-gsm8k only for "
            "non-final diagnostics"
        )
    return args


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise RuntimeError("cannot average an empty score collection")
    value = sum(materialized) / len(materialized)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite score: {value}")
    return value


def _validate_unit_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not a numeric score: {value!r}")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"{label} must be finite and in [0,1], got {score}")
    return score


def _validate_finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not a numeric score: {value!r}")
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{label} must be finite, got {score}")
    return score


def _task_scores(report: dict[str, Any]) -> dict[int, float]:
    if report.get("profile") != "core88":
        raise RuntimeError(
            f"30-column summary requires profile=core88, got {report.get('profile')}"
        )
    accepted_statuses = {"CORE_NATIVE_FULL_OK"}
    if report.get("allow_degraded_math_runtime") is True:
        accepted_statuses.add("CORE_NATIVE_FULL_DIAGNOSTIC")
    if report.get("status") not in accepted_statuses:
        raise RuntimeError("30-column summary requires a fully scored Core88 report")
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 88:
        raise RuntimeError(f"30-column summary requires exactly 88 tasks, got {len(tasks or [])}")
    by_order: dict[int, float] = {}
    for task in tasks:
        task_order = int(task["task_order"])
        if task_order in by_order:
            raise RuntimeError(f"duplicate task order in merged report: {task_order}")
        if task.get("score_status") != "SCORED" or task.get("score_complete") is not True:
            raise RuntimeError(f"task {task_order} is not completely scored")
        by_order[task_order] = _validate_finite_score(
            task.get("primary_score"), f"task {task_order} primary_score"
        )
    if sorted(by_order) != list(range(1, 89)):
        raise RuntimeError("merged report does not contain the fixed Core88 task order 1..88")
    return by_order


def _prediction_files(report: dict[str, Any]) -> Iterable[Path]:
    input_roots = report.get("input_roots")
    if not isinstance(input_roots, list) or not input_roots:
        raise RuntimeError("merged report contains no input_roots for derived score validation")
    for raw_root in input_roots:
        root = Path(str(raw_root))
        if not (root / "run_manifest.json").is_file():
            raise RuntimeError(f"missing run manifest in Core88 input root: {root}")
        yield from sorted(root.glob("**/predictions/*.jsonl"))


def _derive_mmlu_and_lambada(report: dict[str, Any]) -> dict[str, Any]:
    selected_orders = set(_MMLU_CATEGORY_BY_TASK_ORDER) | {23}
    subject_correctness: dict[str, list[bool]] = defaultdict(list)
    subject_categories: dict[str, str] = {}
    seen: set[tuple[int, str, int]] = set()
    observed_by_task: dict[int, int] = defaultdict(int)
    lambada_greedy_correct = 0

    for path in _prediction_files(report):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                task_order = int(record["task_order"])
                if task_order not in selected_orders:
                    continue
                key = (
                    task_order,
                    _stable_id(record["example_id"]),
                    int(record.get("sample_index", 0)),
                )
                if key in seen:
                    raise RuntimeError(
                        f"duplicate derived-metric prediction at {path}:{line_number}"
                    )
                seen.add(key)
                observed_by_task[task_order] += 1
                if task_order == 23:
                    candidates = record.get("candidates")
                    if not isinstance(candidates, list) or len(candidates) != 1:
                        raise RuntimeError(
                            f"LAMBADA-OpenAI requires one saved candidate at {path}:{line_number}"
                        )
                    is_greedy = candidates[0].get("is_greedy")
                    if not isinstance(is_greedy, bool):
                        raise RuntimeError(
                            f"LAMBADA-OpenAI is missing a boolean greedy flag at {path}:{line_number}"
                        )
                    lambada_greedy_correct += int(is_greedy)
                    continue

                example_id = record.get("example_id")
                if not isinstance(example_id, str):
                    raise RuntimeError(
                        f"MMLU example_id must encode subject:index at {path}:{line_number}"
                    )
                subject, separator, _ = example_id.partition(":")
                if not separator or not subject:
                    raise RuntimeError(
                        f"MMLU example_id must encode subject:index at {path}:{line_number}"
                    )
                correct = record.get("primary_correct")
                if not isinstance(correct, bool):
                    raise RuntimeError(
                        f"MMLU prediction is missing primary_correct at {path}:{line_number}"
                    )
                category = _MMLU_CATEGORY_BY_TASK_ORDER[task_order]
                previous_category = subject_categories.setdefault(subject, category)
                if previous_category != category:
                    raise RuntimeError(f"MMLU subject {subject!r} appears in multiple categories")
                subject_correctness[subject].append(correct)

    report_tasks = {int(task["task_order"]): task for task in report["tasks"]}
    for task_order in sorted(selected_orders):
        expected = int(report_tasks[task_order]["observed_predictions"])
        if observed_by_task[task_order] != expected:
            raise RuntimeError(
                f"derived task {task_order} prediction count mismatch: "
                f"{observed_by_task[task_order]} != {expected}"
            )

    if len(subject_correctness) != 57:
        raise RuntimeError(f"MMLU requires exactly 57 subjects, got {len(subject_correctness)}")
    subject_scores = {
        subject: _mean(float(correct) for correct in correctness)
        for subject, correctness in sorted(subject_correctness.items())
    }
    category_subject_scores: dict[str, float] = {}
    observed_subject_counts: dict[str, int] = {}
    for category in sorted(_EXPECTED_MMLU_SUBJECT_COUNTS):
        values = [
            subject_scores[subject]
            for subject, subject_category in subject_categories.items()
            if subject_category == category
        ]
        observed_subject_counts[category] = len(values)
        category_subject_scores[category] = _mean(values)
    if observed_subject_counts != _EXPECTED_MMLU_SUBJECT_COUNTS:
        raise RuntimeError(
            "MMLU category subject counts changed: "
            f"{observed_subject_counts} != {_EXPECTED_MMLU_SUBJECT_COUNTS}"
        )

    lambada_count = observed_by_task[23]
    if lambada_count <= 0:
        raise RuntimeError("LAMBADA-OpenAI contains no predictions")
    return {
        "mmlu_subject_equal": _mean(subject_scores.values()),
        "mmlu_category_subject_equal": category_subject_scores,
        "mmlu_subject_scores": subject_scores,
        "mmlu_subject_counts": observed_subject_counts,
        "lambada_openai_greedy": lambada_greedy_correct / lambada_count,
        "lambada_openai_greedy_correct": lambada_greedy_correct,
        "lambada_openai_example_count": lambada_count,
    }


def _same_path(first: str, second: str) -> bool:
    if first == second:
        return True
    return Path(first).resolve() == Path(second).resolve()


def _same_run_request_seed(doc_index: int, base_seed: int = GSM8K_SEED) -> int:
    payload = f"gsm8k\0{base_seed}\0{doc_index}\0{0}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _reported_evaluator_state(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve the path-independent scorer identity recorded by aggregation."""

    return {
        "evaluator_repo_root": report.get("evaluator_repo_root"),
        "evaluator_repo_commit": report.get("evaluator_repo_commit"),
        "evaluator_repo_dirty": report.get("evaluator_repo_dirty"),
        "evaluator_source_kind": report.get("evaluator_source_kind"),
        "evaluator_source_tree_sha256": report.get("evaluator_source_tree_sha256"),
        "evaluator_package_version": report.get("evaluator_package_version"),
    }


def _validate_reported_evaluator(
    workflow: dict[str, Any], report: dict[str, Any], *, allow_descendant: bool = False
) -> dict[str, Any]:
    """Validate the evaluator identity preserved in a scored Core88 report."""

    return validate_scoring_evaluator_repo(
        workflow,
        _reported_evaluator_state(report),
        allow_descendant=allow_descendant,
    )


def _load_same_run_gsm8k_score(
    root: Path, report: dict[str, Any], workflow_manifest: Path
) -> dict[str, Any]:
    """Validate the backend-matched GSM8K artifact produced by this run."""

    workflow = load_workflow(workflow_manifest)
    require_formal_workflow(workflow)
    protocol = workflow["gsm8k_protocol"]
    example_count = int(protocol["dataset_sample_count"])
    gpu_count = int(protocol["gpu_count"])
    sampling_seed = int(protocol["sampling_seed"])
    backend = str(protocol["backend"])
    verify_inputs(workflow_manifest, root)
    root = root.resolve()
    workflow_id = str(workflow["workflow_id"])
    if report.get("workflow_id") != workflow_id:
        raise RuntimeError("Core88 and GSM8K do not share one workflow id")
    if report.get("repo_commit") != workflow.get("repo_commit"):
        raise RuntimeError("Core88 and GSM8K do not share one repository commit")
    compatibility = _validate_reported_evaluator(
        workflow,
        report,
        allow_descendant=(
            isinstance(report.get("evaluator_compatibility"), dict)
            and report["evaluator_compatibility"].get("status")
            == "CORE88_SCORING_DESCENDANT_OK"
        ),
    )
    recorded_compatibility = report.get("evaluator_compatibility")
    if recorded_compatibility is not None and recorded_compatibility != compatibility:
        raise RuntimeError("Core88 evaluator compatibility proof changed")
    if (
        compatibility["status"] == "CORE88_SCORING_DESCENDANT_OK"
        and recorded_compatibility != compatibility
    ):
        raise RuntimeError("Core88 scoring descendant is missing its compatibility proof")
    if not _same_path(str(root), str(workflow["gsm8k_results_root"])):
        raise RuntimeError("GSM8K result root does not belong to this workflow")
    if not _same_path(str(report["hf_model_path"]), str(workflow["model_identity_path"])):
        raise RuntimeError("Core88 model does not match its workflow model identity")
    if report.get("model_label") != workflow.get("model_label"):
        raise RuntimeError("Core88 model label does not match its workflow")
    report_roots = {Path(path).resolve() for path in report.get("input_roots") or []}
    if report_roots != {Path(workflow["core_results_root"]).resolve()}:
        raise RuntimeError("Core88 input roots do not match their workflow root")
    for marker in (root / "_SUCCESS", root / "_CORE88_SUCCESS"):
        if not marker.is_file():
            raise RuntimeError(f"missing same-run GSM8K success marker: {marker}")

    companion_path = root / "core88-companion.json"
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    if companion.get("status") != COMPANION_STATUS:
        raise RuntimeError("same-run GSM8K companion is not successful")
    if companion.get("backend") != backend:
        raise RuntimeError("same-run GSM8K companion backend changed")
    if companion.get("workflow_id") != workflow_id:
        raise RuntimeError("GSM8K companion belongs to another workflow")
    if not _same_path(
        str(companion.get("workflow_manifest", "")), str(workflow_manifest)
    ) or companion.get("workflow_manifest_sha256") != file_sha256(workflow_manifest):
        raise RuntimeError("GSM8K companion workflow manifest changed")
    if companion.get("repo_commit") != workflow.get("repo_commit"):
        raise RuntimeError("GSM8K companion used another repository commit")
    if not _same_path(
        str(companion.get("model_identity_path", "")), str(workflow["model_identity_path"])
    ):
        raise RuntimeError("GSM8K companion used another model")
    preparation_path = Path(str(companion["model_preparation_manifest"])).resolve()
    if preparation_path != root / "model-preparation.json":
        raise RuntimeError("GSM8K preparation manifest escapes its result root")
    if companion.get("model_preparation_manifest_sha256") != file_sha256(preparation_path):
        raise RuntimeError("GSM8K model preparation manifest changed")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("weight_files_mutated") is not False:
        raise RuntimeError("GSM8K source model mutation was not ruled out")
    if not _same_path(str(preparation.get("source_model", "")), str(workflow["hf_model_path"])):
        raise RuntimeError("GSM8K prepared another HF model source")
    inputs_manifest_path = Path(str(companion["inputs_manifest"])).resolve()
    if inputs_manifest_path != root / "inputs" / "manifest.json" or companion.get(
        "inputs_manifest_sha256"
    ) != file_sha256(inputs_manifest_path):
        raise RuntimeError("GSM8K prepared input manifest changed")
    inputs_jsonl_path = Path(str(companion.get("inputs_jsonl", ""))).resolve()
    if inputs_jsonl_path != root / "inputs" / "inputs.jsonl" or companion.get(
        "inputs_jsonl_sha256"
    ) != file_sha256(inputs_jsonl_path):
        raise RuntimeError("GSM8K prepared inputs changed after sealing")
    if companion.get("inputs_jsonl_sha256") != protocol["inputs_jsonl_sha256"]:
        raise RuntimeError("GSM8K prepared inputs differ from the fixed workflow")
    sealed_gold_by_doc: dict[int, str] = {}
    with inputs_jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            input_record = json.loads(line)
            doc_index = int(input_record["doc_index"])
            if doc_index in sealed_gold_by_doc:
                raise RuntimeError(
                    "duplicate doc_index in sealed GSM8K inputs at line " f"{line_number}"
                )
            sealed_gold_by_doc[doc_index] = str(input_record["gold_answer"])
    if set(sealed_gold_by_doc) != set(range(example_count)):
        raise RuntimeError("sealed GSM8K input gold coverage is incomplete")

    aggregate_path = root / "aggregate.json"
    if not _same_path(
        str(companion.get("aggregate_json", "")), str(aggregate_path)
    ) or companion.get("aggregate_json_sha256") != file_sha256(aggregate_path):
        raise RuntimeError("GSM8K aggregate changed after same-run sealing")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    effective_max_model_len = validate_gsm8k_max_model_len(protocol, aggregate)
    expected_fields = {
        "status": (
            "GSM8K_LMDEPLOY_EVAL_OK"
            if backend == "lmdeploy"
            else "GSM8K_VLLM_EVAL_OK"
        ),
        "backend": backend,
        "task": protocol["task"],
        "task_group": protocol["task_group"],
        "standard_input_config_sha256": protocol["standard_input_config_sha256"],
        "dataset_test_sha256": protocol["dataset_test_sha256"],
        "prompt_sha256": protocol["prompt_sha256"],
        "dataset_sample_count": example_count,
        "sample_count": example_count,
        "samples_per_doc": protocol["samples_per_doc"],
        "num_fewshot": protocol["num_fewshot"],
        "fewshot_sampler": "first_n",
        "fixed_fewshot_samples": True,
        "fewshot_seed": protocol["fewshot_seed"],
        "sampling_seed": sampling_seed,
        "batch_size": protocol["batch_size"],
        "max_num_seqs": protocol["batch_size"],
        "gpu_count": gpu_count,
        "shard_count": gpu_count,
        "temperature": 0.6,
        "top_p": 0.6,
        "max_new_tokens": 512,
        "stop_strings": ["Question:", "</s>", "<|im_end|>"],
        "prefix_caching": False,
        "speculative_decoding": False,
        "schedule_position_rule": "doc_sample_pairs_doc_major[rank::world_size]",
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
    }
    if backend == "lmdeploy":
        expected_fields["top_k"] = 0
    else:
        expected_fields["max_model_len"] = effective_max_model_len
    for field, expected in expected_fields.items():
        if aggregate.get(field) != expected:
            raise RuntimeError(
                f"same-run GSM8K {field} changed: " f"{aggregate.get(field)!r} != {expected!r}"
            )
    if backend == "lmdeploy":
        lmdeploy_runtime = aggregate.get("lmdeploy_runtime")
        expected_runtime = protocol.get("lmdeploy_config")
        if not isinstance(lmdeploy_runtime, dict) or not isinstance(
            expected_runtime, dict
        ):
            raise RuntimeError("same-run GSM8K LMDeploy runtime metadata is missing")
        for field, expected in expected_runtime.items():
            if lmdeploy_runtime.get(field) != expected:
                raise RuntimeError(
                    f"same-run GSM8K LMDeploy {field} changed: "
                    f"{lmdeploy_runtime.get(field)!r} != {expected!r}"
                )
    if not _same_path(
        str(aggregate.get("standard_input_config", "")), str(protocol["standard_input_config"])
    ):
        raise RuntimeError("same-run GSM8K used another standard input config")
    if not _same_path(
        str(aggregate.get("dataset_test_file", "")), str(protocol["dataset_test_file"])
    ):
        raise RuntimeError("same-run GSM8K used another dataset test file")
    task_contract = aggregate.get("task_contract")
    expected_task_contract = {
        "num_fewshot": protocol["num_fewshot"],
        "fewshot_sampler": "first_n",
        "fixed_fewshot_samples": True,
        "runtime_repeats": 1,
    }
    if not isinstance(task_contract, dict) or any(
        task_contract.get(field) != expected for field, expected in expected_task_contract.items()
    ):
        raise RuntimeError("same-run GSM8K task contract changed")
    generation_kwargs = task_contract.get("generation_kwargs")
    expected_generation_kwargs = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.6,
        "max_gen_toks": 512,
        "until": ["Question:", "</s>", "<|im_end|>"],
    }
    if not isinstance(generation_kwargs, dict) or any(
        generation_kwargs.get(field) != expected
        for field, expected in expected_generation_kwargs.items()
    ):
        raise RuntimeError("same-run GSM8K generation contract changed")
    if aggregate.get("model_family") != companion.get("model_family"):
        raise RuntimeError("same-run GSM8K model family changed")
    if not _same_path(
        str(aggregate.get("model_identifier", "")), str(companion.get("runtime_model_path", ""))
    ):
        raise RuntimeError("same-run GSM8K runtime model changed")

    declared_samples = 0
    for rank in range(gpu_count):
        shard_root = root / f"shard-{rank:02d}"
        if not (shard_root / "_SUCCESS").is_file():
            raise RuntimeError(f"missing GSM8K shard success marker: {rank}")
        result = json.loads((shard_root / "result.json").read_text(encoding="utf-8"))
        shard_expected = {
            "status": (
                "GSM8K_LMDEPLOY_GENERATION_OK"
                if backend == "lmdeploy"
                else "GSM8K_VLLM_GENERATION_OK"
            ),
            "rank": rank,
            "world_size": gpu_count,
            "dataset_sample_count": example_count,
            "samples_per_doc": protocol["samples_per_doc"],
            "batch_size": protocol["batch_size"],
            "max_num_seqs": protocol["batch_size"],
            "max_model_len": effective_max_model_len,
            "sampling_seed": sampling_seed,
            "model_family": companion["model_family"],
        }
        for field, expected in shard_expected.items():
            if result.get(field) != expected:
                raise RuntimeError(f"same-run GSM8K shard {rank} {field} changed")
        declared_samples += int(result["sample_count"])
    if declared_samples != example_count:
        raise RuntimeError("same-run GSM8K shard sample coverage is incomplete")

    predictions_path = Path(str(companion.get("predictions_jsonl", ""))).resolve()
    if predictions_path != root / "predictions.jsonl" or companion.get(
        "predictions_jsonl_sha256"
    ) != file_sha256(predictions_path):
        raise RuntimeError("same-run GSM8K root predictions changed after sealing")
    if not _same_path(str(aggregate.get("predictions_jsonl", "")), str(predictions_path)):
        raise RuntimeError("same-run GSM8K aggregate points outside its result root")
    seen_docs: set[int] = set()
    correct = 0
    with predictions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            doc_index = int(record["doc_index"])
            if doc_index in seen_docs or int(record.get("sample_index", -1)) != 0:
                raise RuntimeError(f"invalid same-run GSM8K request key at line {line_number}")
            seen_docs.add(doc_index)
            if int(record.get("sample_seed", -1)) != _same_run_request_seed(
                doc_index, sampling_seed
            ):
                raise RuntimeError(f"same-run GSM8K request seed changed at line {line_number}")
            sealed_gold = sealed_gold_by_doc.get(doc_index)
            if sealed_gold is None or str(record.get("gold_answer", "")) != sealed_gold:
                raise RuntimeError(
                    "same-run GSM8K prediction gold differs from sealed inputs "
                    f"for doc_index {doc_index}"
                )
            score = score_gsm_answer(str(record["output"]), sealed_gold)
            correct += int(bool(score["primary_correct"]))
    if seen_docs != set(range(example_count)):
        raise RuntimeError(f"same-run GSM8K prediction coverage is incomplete: {len(seen_docs)}")
    if int(aggregate.get("pass_at_1_correct", -1)) != correct:
        raise RuntimeError("same-run GSM8K official rescore differs from aggregate")
    score = correct / example_count
    if not math.isclose(float(aggregate.get("pass_at_1", -1)), score):
        raise RuntimeError("same-run GSM8K pass@1 differs from official rescore")
    return {
        "status": "GSM8K_SAME_RUN_OFFICIAL_RESCORE_OK",
        "workflow_id": workflow_id,
        "result_root": str(root),
        "aggregate_json": str(aggregate_path),
        "aggregate_json_sha256": file_sha256(aggregate_path),
        "prediction_files": [str(predictions_path)],
        "shard_count": gpu_count,
        "example_count": example_count,
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "pass_at_1_correct": correct,
        "pass_at_1": score,
    }


def _load_gsm8k_score(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    result_paths = sorted(root.glob("shard-*/result.json"))
    if not result_paths and (root / "result.json").is_file():
        result_paths = [root / "result.json"]
    if not result_paths:
        raise RuntimeError(f"GSM8K result root contains no result.json: {root}")

    allowed_models = {
        str(report["hf_model_path"]),
        *(str(path) for path in report.get("runtime_hf_model_paths") or []),
    }
    shard_count: int | None = None
    dataset_sample_count: int | None = None
    shard_indices: set[int] = set()
    declared_correct = 0
    declared_samples = 0
    prediction_paths: list[Path] = []
    root_resolved = root.resolve()
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "GSM8K_EVAL_OK":
            raise RuntimeError(f"GSM8K result is not successful: {result_path}")
        source_mutated = result.get("source_model_mutated")
        checkpoint_mutated = result.get("checkpoint_mutated")
        if result.get("artifact_mutated") is not False or source_mutated is True:
            raise RuntimeError(f"GSM8K model artifact mutation was not ruled out: {result_path}")
        if result.get("model_source") == "dcp" and checkpoint_mutated is not False:
            raise RuntimeError(
                f"GSM8K checkpoint mutation was not ruled out: {result_path}"
            )
        result_model = str(
            result.get("hf_model_path")
            or result.get("model_identity_path")
            or result.get("checkpoint_root")
            or ""
        )
        if not result_model or not any(_same_path(result_model, model) for model in allowed_models):
            raise RuntimeError(
                f"GSM8K model {result_model!r} does not match Core88 model {sorted(allowed_models)}"
            )
        current_shard_count = int(result["shard_count"])
        current_dataset_count = int(result["dataset_sample_count"])
        if shard_count is None:
            shard_count = current_shard_count
        if dataset_sample_count is None:
            dataset_sample_count = current_dataset_count
        if current_shard_count != shard_count or current_dataset_count != dataset_sample_count:
            raise RuntimeError("GSM8K result shards disagree on shard or dataset counts")
        shard_index = int(result["shard_index"])
        if shard_index in shard_indices:
            raise RuntimeError(f"duplicate GSM8K shard index: {shard_index}")
        shard_indices.add(shard_index)
        declared_correct += int(result["pass_at_1_correct"])
        declared_samples += int(result["sample_count"])
        prediction_path = Path(str(result["predictions_jsonl"])).resolve()
        if not prediction_path.is_relative_to(root_resolved):
            raise RuntimeError(f"GSM8K prediction path escapes the result root: {prediction_path}")
        prediction_paths.append(prediction_path)

    if shard_count is None or dataset_sample_count is None:
        raise RuntimeError("GSM8K result set is empty")
    if shard_indices != set(range(shard_count)):
        raise RuntimeError(f"GSM8K shard coverage is incomplete: {sorted(shard_indices)}")
    if len(result_paths) != shard_count:
        raise RuntimeError(
            f"GSM8K result file count {len(result_paths)} != shard_count {shard_count}"
        )
    if declared_samples != dataset_sample_count:
        raise RuntimeError(
            f"GSM8K declared sample coverage {declared_samples} != {dataset_sample_count}"
        )

    seen_docs: set[int] = set()
    prediction_correct = 0
    legacy_scorer_disagreements = 0
    for prediction_path in prediction_paths:
        with prediction_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_index = int(record["doc_index"])
                if doc_index in seen_docs:
                    raise RuntimeError(
                        f"duplicate GSM8K doc_index {doc_index} at {prediction_path}:{line_number}"
                    )
                seen_docs.add(doc_index)
                if "output" not in record or "gold" not in record:
                    raise RuntimeError(
                        "GSM8K prediction lacks output/gold required for official "
                        f"rescoring at {prediction_path}:{line_number}"
                    )
                official_score = score_gsm_answer(str(record["output"]), str(record["gold"]))
                official_correct = bool(official_score["primary_correct"])
                prediction_correct += int(official_correct)
                legacy_correct = record.get("standard_correct")
                if isinstance(legacy_correct, bool) and legacy_correct != official_correct:
                    legacy_scorer_disagreements += 1
    if seen_docs != set(range(dataset_sample_count)):
        raise RuntimeError(
            f"GSM8K prediction coverage is incomplete: {len(seen_docs)}/{dataset_sample_count}"
        )
    return {
        "status": "GSM8K_EXTERNAL_RESULT_OFFICIAL_RESCORE_OK",
        "result_root": str(root_resolved),
        "result_files": [str(path.resolve()) for path in result_paths],
        "prediction_files": [str(path) for path in prediction_paths],
        "shard_count": shard_count,
        "example_count": dataset_sample_count,
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "pass_at_1_correct": prediction_correct,
        "pass_at_1": prediction_correct / dataset_sample_count,
        "legacy_declared_correct": declared_correct,
        "legacy_scorer_disagreements": legacy_scorer_disagreements,
    }


def _format_percentage(value: float | None) -> str:
    return "NA" if value is None else f"{value * 100.0:.6f}"


def build_core88_summary(
    report: dict[str, Any],
    *,
    gsm8k_results_root: Path | None = None,
    workflow_manifest: Path | None = None,
    sciq_score: float | None = None,
    allow_diagnostic_gsm8k: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build one fixed-schema percentage row and its derivation evidence.

    A production summary is fail-closed: it requires a standalone GSM8K result
    sealed into the same workflow.  Missing or legacy GSM8K evidence is only
    available through the explicit diagnostic opt-in and receives a distinct
    non-final status.
    """

    task_scores = _task_scores(report)
    derived = _derive_mmlu_and_lambada(report)
    unit_task_orders = {1, 3, 19, 67, 74, *range(26, 33), *range(75, 89)}
    for task_order in unit_task_orders:
        task_scores[task_order] = _validate_unit_score(
            task_scores[task_order], f"task {task_order} primary_score"
        )
    validated_sciq = None if sciq_score is None else _validate_unit_score(sciq_score, "SciQ score")
    same_run_gsm8k = gsm8k_results_root is not None and workflow_manifest is not None
    if not same_run_gsm8k and not allow_diagnostic_gsm8k:
        raise RuntimeError("final Core88 summary requires same-workflow GSM8K results")
    gsm8k = None
    if same_run_gsm8k:
        gsm8k = _load_same_run_gsm8k_score(gsm8k_results_root, report, workflow_manifest)
    elif gsm8k_results_root is not None:
        gsm8k = _load_gsm8k_score(gsm8k_results_root, report)
    gsm8k_score = None if gsm8k is None else float(gsm8k["pass_at_1"])
    gsm_symbolic = task_scores[75]
    minerva = _mean(task_scores[order] for order in _MINERVA_TASK_ORDERS)
    math_average = None if gsm8k_score is None else _mean((gsm8k_score, gsm_symbolic, minerva))
    code_average = _mean(task_scores[order] for order in _CODE_TASK_ORDERS)
    categories = derived["mmlu_category_subject_equal"]
    unit_scores: tuple[float | None, ...] = (
        float(derived["mmlu_subject_equal"]),
        gsm8k_score,
        gsm_symbolic,
        minerva,
        math_average,
        task_scores[80],
        task_scores[81],
        task_scores[82],
        task_scores[83],
        task_scores[84],
        task_scores[85],
        task_scores[86],
        task_scores[87],
        code_average,
        task_scores[3],
        task_scores[1],
        float(categories["stem"]),
        validated_sciq,
        float(categories["humanities"]),
        float(categories["social_sciences"]),
        float(categories["other"]),
        task_scores[67],
        task_scores[19],
        task_scores[74],
        task_scores[88],
        float(derived["lambada_openai_greedy"]),
        task_scores[76],
        task_scores[77],
        task_scores[78],
        task_scores[79],
    )
    if len(unit_scores) != len(SUMMARY_HEADERS):
        raise AssertionError("summary schema and value count diverged")
    row = {
        header: _format_percentage(value)
        for header, value in zip(SUMMARY_HEADERS, unit_scores, strict=True)
    }
    degraded_math_runtime = bool(report.get("allow_degraded_math_runtime", False))
    final_output_eligible = same_run_gsm8k and not degraded_math_runtime
    metadata = {
        "schema_version": "core88-fixed-summary-v1",
        "status": (FINAL_SUMMARY_STATUS if final_output_eligible else DIAGNOSTIC_SUMMARY_STATUS),
        "final_output_eligible": final_output_eligible,
        "allow_degraded_math_runtime": degraded_math_runtime,
        "gsm8k_evidence_mode": (
            "same_workflow"
            if same_run_gsm8k
            else ("legacy_external" if gsm8k is not None else "missing")
        ),
        "units": "percentage_points",
        "model_label": report.get("model_label", ""),
        "hf_model_path": report["hf_model_path"],
        "profile": report["profile"],
        "detailed_report_status": report["status"],
        "mmlu": {"aggregation": "equal_mean_over_57_subjects", "subject_count": 57, **derived},
        "gsm8k": ({"status": "NOT_PROVIDED", "score": None} if gsm8k is None else gsm8k),
        "sciq": {
            "status": (
                "NOT_PROVIDED_CORE88_HAS_NO_SCIQ"
                if validated_sciq is None
                else "EXTERNAL_SCORE_PROVIDED"
            ),
            "score": validated_sciq,
        },
        "gsm_symbolic": {
            "task_order": 75,
            "aggregation": "equal_mean_over_3_subsets",
            "score": gsm_symbolic,
        },
        "minerva": {
            "task_orders": list(_MINERVA_TASK_ORDERS),
            "aggregation": "equal_mean_over_7_categories",
            "score": minerva,
        },
        "math_average": {"components": ["GSM8K", "GSM-Symbolic", "Minerva"], "score": math_average},
        "code_average": {
            "task_orders": list(_CODE_TASK_ORDERS),
            "aggregation": "equal_mean_over_7_execution_tasks",
            "score": code_average,
        },
        "lambada_openai": {
            "task_order": 23,
            "aggregation": "greedy_exact_continuation_from_saved_candidate_flags",
            "score": derived["lambada_openai_greedy"],
            "correct": derived["lambada_openai_greedy_correct"],
            "example_count": derived["lambada_openai_example_count"],
        },
        "headers": list(SUMMARY_HEADERS),
        "row": row,
    }
    return row, metadata


def write_core88_summary_csv(
    output_csv: Path,
    report: dict[str, Any],
    *,
    gsm8k_results_root: Path | None = None,
    workflow_manifest: Path | None = None,
    sciq_score: float | None = None,
    allow_diagnostic_gsm8k: bool = False,
) -> dict[str, Any]:
    """Atomically write the fixed 30-column Core88 comparison table."""

    row, metadata = build_core88_summary(
        report,
        gsm8k_results_root=gsm8k_results_root,
        workflow_manifest=workflow_manifest,
        sciq_score=sciq_score,
        allow_diagnostic_gsm8k=allow_diagnostic_gsm8k,
    )
    return write_prepared_core88_summary_csv(output_csv, row, metadata)


def write_prepared_core88_summary_csv(
    output_csv: Path, row: dict[str, str], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Write a summary that was already fully validated and prepared."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_HEADERS))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(output_csv)
    metadata["output_csv"] = str(output_csv.resolve())
    return metadata


def main() -> None:
    args = parse_args()
    report = json.loads(args.merged_json.read_text(encoding="utf-8"))
    metadata = write_core88_summary_csv(
        args.output_csv,
        report,
        gsm8k_results_root=args.gsm8k_results_root,
        workflow_manifest=args.workflow_manifest,
        sciq_score=args.sciq_score,
        allow_diagnostic_gsm8k=args.allow_diagnostic_gsm8k,
    )
    metadata["merged_json"] = str(args.merged_json.resolve())
    if args.output_metadata_json is not None:
        _write_json_atomic(args.output_metadata_json, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
