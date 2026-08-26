#!/usr/bin/env python3
"""Merge and rescore saved Hugging Face OLMo Core prediction artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core88_workflow import (
    load_workflow,
    validate_core_run_manifest,
    validate_scoring_evaluator_repo,
    verify_inputs,
)
from .core_native_eval import (
    ACCURACY_METRICS,
    GREEDY_EXACT_CONTINUATION_SCORING_CONTRACT,
    _file_sha256,
    _generation_score,
    _pass_at_k,
    _primary_metric,
    _stable_id,
    _task_is_generation,
    _task_sample_count,
    _text_sha256,
    _write_json_atomic,
    _write_scores_csv,
)
from .core_native_summary import (
    DIAGNOSTIC_SUMMARY_STATUS,
    build_core88_summary,
    write_prepared_core88_summary_csv,
)
from .source_identity import evaluator_state

_COMPATIBLE_SOURCE_PROFILES = {
    "core_native": {"core_native"},
    "core79": {"core_native", "core79"},
    "core88": {"core_native", "core79", "core88"},
}
_EVALUATOR_REPO_ROOT = Path(__file__).resolve().parents[2]


def _evaluator_repo_state(repo_root: Path = _EVALUATOR_REPO_ROOT) -> dict[str, Any]:
    """Return the exact source state of the code performing final aggregation."""

    return evaluator_state(repo_root)


def _validate_evaluator_repo_state(
    workflow: dict[str, Any], state: dict[str, Any], *, allow_descendant: bool = False
) -> dict[str, Any]:
    """Validate the final scorer and return its compatibility proof."""

    return validate_scoring_evaluator_repo(workflow, state, allow_descendant=allow_descendant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--profile", default="core88")
    parser.add_argument(
        "--task-orders",
        default="",
        help="Optional comma-separated subset of task orders from --profile.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every job partition belonging to one model.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=("Ordered score CSV path. Defaults to --output-json with a .csv suffix."),
    )
    parser.add_argument(
        "--code-results",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional isolated-sandbox JSONL results with task_order, example_id, sample_index, and passed."
        ),
    )
    parser.add_argument(
        "--code-results-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional directory containing sharded code-results*.jsonl files. Error JSONL files are ignored."
        ),
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=None,
        help="Optional fixed 30-column comparison CSV; requires all 88 scores to be complete.",
    )
    parser.add_argument(
        "--output-summary-metadata-json",
        type=Path,
        default=None,
        help="Optional derivation evidence for --output-summary-csv.",
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
        help="Fresh workflow manifest binding Core88 to standalone GSM8K.",
    )
    parser.add_argument(
        "--allow-missing-workflow-gsm8k",
        action="store_true",
        help="Diagnostic compatibility mode; never use for final Core88 outputs.",
    )
    parser.add_argument(
        "--sciq-score",
        type=float,
        default=None,
        help="Optional independent SciQ accuracy in [0,1]; Core88 itself has no SciQ task.",
    )
    parser.add_argument(
        "--allow-degraded-math-runtime",
        action="store_true",
        help=(
            "Allow Minerva/MATH scoring without the official SymPy+ANTLR 4.11 "
            "runtime. Diagnostic only; never use for final scores."
        ),
    )
    parser.add_argument(
        "--allow-scoring-evaluator-descendant",
        action="store_true",
        help=(
            "Allow a clean descendant scorer commit when its complete diff is "
            "restricted to the audited Core88 scoring/sealing allowlist."
        ),
    )
    args = parser.parse_args()
    if args.output_summary_metadata_json is not None and args.output_summary_csv is None:
        parser.error("--output-summary-metadata-json requires --output-summary-csv")
    if (
        args.gsm8k_results_root is not None
        or args.workflow_manifest is not None
        or args.sciq_score is not None
    ) and args.output_summary_csv is None:
        parser.error(
            "--gsm8k-results-root/--workflow-manifest/--sciq-score require " "--output-summary-csv"
        )
    if args.output_summary_csv is not None and not args.allow_missing_workflow_gsm8k:
        if args.gsm8k_results_root is None or args.workflow_manifest is None:
            parser.error(
                "final Core88 aggregation requires --workflow-manifest and " "--gsm8k-results-root"
            )
    return args


def _prediction_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("**/predictions/*.jsonl"))


def _load_gold(
    data_root: Path,
    profile: str,
    summary: dict[str, Any] | None = None,
    task_orders: set[int] | None = None,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    if summary is None:
        summary = json.loads((data_root / "summary.json").read_text(encoding="utf-8"))
    section = summary.get(profile)
    if not isinstance(section, dict) or section.get("failures"):
        raise RuntimeError(f"{profile} export is missing or reports failures")
    tasks = section.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"{profile} contains no tasks")
    if task_orders:
        selected = [task for task in tasks if int(task["task_order"]) in task_orders]
        selected_orders = {int(task["task_order"]) for task in selected}
        if selected_orders != task_orders:
            raise RuntimeError(
                f"{profile} does not contain requested task orders: "
                f"{sorted(task_orders - selected_orders)}"
            )
        tasks = selected
    gold: dict[tuple[int, str], dict[str, Any]] = {}
    normalized_tasks: list[dict[str, Any]] = []
    for source_task in tasks:
        task = dict(source_task)
        path = data_root / task["file"]
        if _file_sha256(path) != task["sha256"]:
            raise RuntimeError(f"dataset SHA256 mismatch: {path}")
        first_row: dict[str, Any] | None = None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if first_row is None:
                    first_row = row
                metric_contract = row.get("metric")
                if not isinstance(metric_contract, str):
                    row["metric_definitions"] = metric_contract
                row["metric"] = _primary_metric(metric_contract)
                key = (int(row["task_order"]), _stable_id(row["example_id"]))
                if key in gold:
                    raise RuntimeError(f"duplicate gold key: {key}")
                gold[key] = row
        if first_row is None:
            raise RuntimeError(f"dataset is empty: {path}")
        metric_contract = task.get("metric", first_row.get("metric"))
        if not isinstance(metric_contract, str):
            task["metric_definitions"] = metric_contract
        task["metric"] = _primary_metric(metric_contract)
        if (
            task.get("metric_definitions") is None
            and first_row.get("metric_definitions") is not None
        ):
            task["metric_definitions"] = first_row["metric_definitions"]
        if task.get("request_type") is None:
            task["request_type"] = first_row.get("request_type")
        normalized_tasks.append(task)
    return gold, normalized_tasks


def _compatible_source_task_orders(
    summary: dict[str, Any], target_profile: str, source_profile: str
) -> set[int]:
    compatible_sources = _COMPATIBLE_SOURCE_PROFILES.get(target_profile, {target_profile})
    if source_profile not in compatible_sources:
        raise RuntimeError(
            f"profile {source_profile!r} is not a compatible source for {target_profile!r}"
        )
    target_section = summary.get(target_profile)
    source_section = summary.get(source_profile)
    if not isinstance(target_section, dict) or target_section.get("failures"):
        raise RuntimeError(f"{target_profile} export is missing or reports failures")
    if not isinstance(source_section, dict) or source_section.get("failures"):
        raise RuntimeError(f"{source_profile} export is missing or reports failures")
    target_tasks = {int(task["task_order"]): task for task in target_section.get("tasks") or []}
    source_tasks = source_section.get("tasks") or []
    if not target_tasks or not source_tasks:
        raise RuntimeError(
            f"cannot validate profile compatibility: {source_profile!r} -> {target_profile!r}"
        )
    compatible_orders: set[int] = set()
    signature_fields = (
        "task_order",
        "task",
        "sha256",
        "num_examples",
        "sample_count",
        "metric",
        "request_type",
    )
    for source_task in source_tasks:
        task_order = int(source_task["task_order"])
        target_task = target_tasks.get(task_order)
        if target_task is None:
            raise RuntimeError(
                f"{source_profile} task order {task_order} is absent from {target_profile}"
            )
        source_signature = tuple(source_task.get(field) for field in signature_fields)
        target_signature = tuple(target_task.get(field) for field in signature_fields)
        if source_signature != target_signature:
            raise RuntimeError(
                f"incompatible task {task_order} between {source_profile} and {target_profile}"
            )
        compatible_orders.add(task_order)
    return compatible_orders


def _load_code_results(paths: list[Path]) -> dict[tuple[int, str, int], bool]:
    results: dict[tuple[int, str, int], bool] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record.get("status") not in (None, "PASS", "FAIL"):
                    raise RuntimeError(
                        f"non-score code result at {path}:{line_number}: {record.get('status')}"
                    )
                if record.get("sandbox_self_test") not in (
                    None,
                    "CORE_NATIVE_SANDBOX_SELF_TEST_OK",
                ):
                    raise RuntimeError(f"unverified sandbox result at {path}:{line_number}")
                key = (
                    int(record["task_order"]),
                    _stable_id(record["example_id"]),
                    int(record["sample_index"]),
                )
                if key in results:
                    raise RuntimeError(f"duplicate code result at {path}:{line_number}: {key}")
                results[key] = bool(record["passed"])
    return results


def _validate_prediction_gold(record: dict[str, Any], gold: dict[str, Any], path: Path) -> None:
    if record.get("input_sha256") != _text_sha256(str(gold["input"])):
        raise RuntimeError(
            f"prediction input SHA256 mismatch for task {gold['task_order']} example {gold['example_id']} in {path}"
        )
    if record.get("task") != gold.get("task"):
        raise RuntimeError(f"prediction task mismatch for order {gold['task_order']} in {path}")


def _resolve_code_result_paths(paths: list[Path], directories: list[Path]) -> list[Path]:
    resolved = [path.resolve() for path in paths]
    for directory in directories:
        resolved.extend(
            path.resolve()
            for path in sorted(directory.rglob("code-results*.jsonl"))
            if not path.name.endswith("-errors.jsonl")
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _apply_manifest_evaluation_sample_counts(
    tasks: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restore per-run sampling caps when aggregating partitioned predictions."""

    samples_by_task_order: dict[int, int] = {}
    for manifest in manifests:
        for task in manifest.get("tasks") or []:
            task_order = int(task["task_order"])
            samples_per_example = _task_sample_count(task)
            if samples_per_example <= 0:
                raise RuntimeError(
                    f"invalid evaluation sample count for task {task_order}: "
                    f"{samples_per_example}"
                )
            previous = samples_by_task_order.setdefault(task_order, samples_per_example)
            if previous != samples_per_example:
                raise RuntimeError(
                    f"inconsistent evaluation sample count for task {task_order}: "
                    f"{previous} != {samples_per_example}"
                )

    effective_tasks = []
    for task in tasks:
        effective_task = dict(task)
        task_order = int(task["task_order"])
        if task_order in samples_by_task_order:
            effective_task["evaluation_sample_count"] = samples_by_task_order[task_order]
        effective_tasks.append(effective_task)
    return effective_tasks


def _manifest_parallelism_metadata(manifests: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Normalize legacy partitions and pooled machine manifests for reports."""

    partition_indices: set[int] = set()
    partition_counts: set[int] = set()
    machine_indices: set[int] = set()
    machine_counts: set[int] = set()
    for manifest in manifests:
        if "partition_index" in manifest:
            partition_indices.add(int(manifest["partition_index"]))
        if "partition_count" in manifest:
            partition_counts.add(int(manifest["partition_count"]))
        if "machine_index" in manifest:
            machine_indices.add(int(manifest["machine_index"]))
        if "machine_count" in manifest:
            machine_count = int(manifest["machine_count"])
            machine_counts.add(machine_count)
            if manifest.get("schema_version") == "hf-core-native-pool-run-v1":
                machine_indices.update(range(machine_count))
    return {
        "partition_indices": sorted(partition_indices),
        "partition_counts": sorted(partition_counts),
        "machine_indices": sorted(machine_indices),
        "machine_counts": sorted(machine_counts),
    }


def _rescore(
    record: dict[str, Any],
    gold: dict[str, Any],
    code_results: dict[tuple[int, str, int], bool],
    *,
    require_official_math_runtime: bool = True,
) -> dict[str, Any]:
    metric = str(gold["metric"])
    if _task_is_generation(gold):
        if gold.get("execution_contract"):
            code_key = (
                int(gold["task_order"]),
                _stable_id(gold["example_id"]),
                int(record.get("sample_index", 0)),
            )
            passed = code_results.get(code_key)
            return {
                "primary_value": (float(passed) if passed is not None else None),
                "primary_correct": passed,
                "score_status": (
                    "SCORED" if passed is not None else "PENDING_EXTERNAL_CODE_EXECUTION"
                ),
            }
        score = _generation_score(
            gold,
            str(record["completion"]),
            require_official_math_runtime=require_official_math_runtime,
        )
        return {"primary_value": float(bool(score["primary_correct"])), **score}
    candidates = record["candidates"]
    target_index = gold.get("target_index")
    effective_target = (
        int(target_index)
        if isinstance(target_index, int) and 0 <= target_index < len(candidates)
        else 0
    )
    predicted = {
        "raw": max(range(len(candidates)), key=lambda index: candidates[index]["sum_logprob"]),
        "per_token": max(
            range(len(candidates)), key=lambda index: candidates[index]["logprob_per_token"]
        ),
        "per_char": max(
            range(len(candidates)), key=lambda index: candidates[index]["logprob_per_char"]
        ),
        "per_byte": min(
            range(len(candidates)), key=lambda index: candidates[index]["bits_per_byte"]
        ),
    }
    normalization = {
        "acc": "raw",
        "acc_raw": "raw",
        "acc_per_token": "per_token",
        "acc_per_char": "per_char",
    }.get(metric)
    if gold.get("scoring_contract") == GREEDY_EXACT_CONTINUATION_SCORING_CONTRACT:
        correct = bool(candidates[effective_target]["is_greedy"])
        return {
            "primary_value": float(correct),
            "primary_correct": correct,
            "predicted_index": predicted,
            "effective_target_index": effective_target,
        }
    if normalization is not None:
        correct = predicted[normalization] == effective_target
        return {
            "primary_value": float(correct),
            "primary_correct": correct,
            "predicted_index": predicted,
            "effective_target_index": effective_target,
        }
    return {
        "primary_value": float(candidates[effective_target]["bits_per_byte"]),
        "primary_correct": None,
        "predicted_index": predicted,
        "effective_target_index": effective_target,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    data_summary_path = data_root / "summary.json"
    data_summary = json.loads(data_summary_path.read_text(encoding="utf-8"))
    selected_task_orders = {
        int(value.strip()) for value in args.task_orders.split(",") if value.strip()
    }
    if any(value <= 0 for value in selected_task_orders):
        raise ValueError(f"task orders must be positive: {sorted(selected_task_orders)}")
    gold, source_tasks = _load_gold(
        data_root, args.profile, data_summary, selected_task_orders
    )
    code_result_paths = _resolve_code_result_paths(args.code_results, args.code_results_dir)
    code_results = _load_code_results(code_result_paths)
    manifests = []
    seen: set[tuple[int, str, int]] = set()
    records_by_task: dict[
        str, dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]
    ] = defaultdict(lambda: defaultdict(list))
    prediction_files: list[str] = []
    models: set[str] = set()
    runtime_model_paths: set[str] = set()
    model_labels: set[str] = set()
    source_profiles: set[str] = set()
    workflow_ids: set[str] = set()
    repo_commits: set[str] = set()
    for root in args.input_root:
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        if manifest.get("workflow_id"):
            workflow_ids.add(str(manifest["workflow_id"]))
        if manifest.get("repo_commit"):
            repo_commits.add(str(manifest["repo_commit"]))
        models.add(str(manifest["hf_model_path"]))
        runtime_model_paths.add(
            str(manifest.get("runtime_hf_model_path", manifest["hf_model_path"]))
        )
        if manifest.get("model_label"):
            model_labels.add(str(manifest["model_label"]))
        if Path(manifest["data_root"]).resolve() != data_root:
            raise RuntimeError(f"data root mismatch in {manifest_path}")
        source_profile = str(manifest.get("profile", "core_native"))
        source_profiles.add(source_profile)
        compatible_task_orders = _compatible_source_task_orders(
            data_summary, args.profile, source_profile
        )
        for path in _prediction_files(root):
            prediction_files.append(str(path))
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = (
                        int(record["task_order"]),
                        _stable_id(record["example_id"]),
                        int(record.get("sample_index", 0)),
                    )
                    if key in seen:
                        raise RuntimeError(f"duplicate prediction key: {key}")
                    if key[0] not in compatible_task_orders:
                        raise RuntimeError(
                            f"prediction task {key[0]} is outside source profile {source_profile} in {path}"
                        )
                    seen.add(key)
                    gold_key = (key[0], key[1])
                    if gold_key not in gold:
                        raise RuntimeError(f"prediction has no gold record: {key}")
                    gold_row = gold[gold_key]
                    _validate_prediction_gold(record, gold_row, path)
                    rescored = _rescore(
                        record,
                        gold_row,
                        code_results,
                        require_official_math_runtime=(not args.allow_degraded_math_runtime),
                    )
                    task_name = str(gold_row["task"])
                    records_by_task[task_name][key[1]].append((record, gold_row, rescored))
    if len(models) != 1:
        raise RuntimeError(f"input roots contain multiple models: {sorted(models)}")
    if len(model_labels) > 1:
        raise RuntimeError(f"input roots contain multiple model labels: {sorted(model_labels)}")

    source_tasks = _apply_manifest_evaluation_sample_counts(source_tasks, manifests)
    tasks = []
    for task in source_tasks:
        task_name = str(task["task"])
        by_example = records_by_task.get(task_name, {})
        matching_gold = next(
            (row for (task_order, _), row in gold.items() if task_order == int(task["task_order"])),
            None,
        )
        metric = str(
            task.get("metric") if task.get("metric") is not None else matching_gold["metric"]
        )
        is_generation = _task_is_generation({**task, "metric": metric})
        example_values: list[float] = []
        subset_values: dict[str, list[float]] = defaultdict(list)
        pass_at_4_values: list[float] = []
        pending_external_scores = 0
        correct_samples = 0
        scored_samples = 0
        answer_scorers: set[str] = set()
        scorer_commits: set[str] = set()
        scorer_source_sha256s: set[str] = set()
        math_runtimes: dict[str, dict[str, Any]] = {}
        for triples in by_example.values():
            for _, _, rescored in triples:
                if rescored.get("answer_scorer"):
                    answer_scorers.add(str(rescored["answer_scorer"]))
                if rescored.get("olmo_eval_commit"):
                    scorer_commits.add(str(rescored["olmo_eval_commit"]))
                scorer_source_sha256s.update(
                    str(value) for value in rescored.get("official_source_sha256s") or []
                )
                if isinstance(rescored.get("official_math_runtime"), dict):
                    runtime = dict(rescored["official_math_runtime"])
                    math_runtimes[_stable_id(runtime)] = runtime
            rescored_values = [triple[2]["primary_value"] for triple in triples]
            pending_external_scores += sum(value is None for value in rescored_values)
            available = [float(value) for value in rescored_values if value is not None]
            if not available:
                continue
            if is_generation:
                correctness = [bool(value) for value in available]
                value = sum(correctness) / len(correctness)
                correct_samples += sum(correctness)
                scored_samples += len(correctness)
                if "pass@4" in (task.get("secondary_metrics") or []):
                    pass_at_4 = _pass_at_k(correctness, 4)
                    if pass_at_4 is not None:
                        pass_at_4_values.append(pass_at_4)
            else:
                value = available[0]
            subset = str(triples[0][1].get("subset") or "__all__")
            example_values.append(value)
            subset_values[subset].append(value)
        subset_scores = {
            subset: sum(values) / len(values)
            for subset, values in sorted(subset_values.items())
            if values
        }
        if task.get("aggregation") == "macro_mean_over_subsets":
            primary_score = (
                sum(subset_scores.values()) / len(subset_scores) if subset_scores else None
            )
        else:
            primary_score = sum(example_values) / len(example_values) if example_values else None
        samples_per_example = _task_sample_count(task)
        expected_examples = int(task["num_examples"])
        expected_predictions = expected_examples * samples_per_example
        observed_examples = len(by_example)
        observed_predictions = sum(len(values) for values in by_example.values())
        prediction_complete = (
            observed_examples == expected_examples and observed_predictions == expected_predictions
        )
        score_complete = (
            prediction_complete
            and pending_external_scores == 0
            and len(example_values) == observed_examples
        )
        task_result = {
            "task_order": int(task["task_order"]),
            "task": task_name,
            "name": task.get("name"),
            "spec": task.get("spec"),
            "metric": metric,
            "secondary_metrics": list(task.get("secondary_metrics") or []),
            "request_type": task.get("request_type"),
            "num_fewshot": task["num_fewshot"],
            "aggregation": task.get("aggregation", "mean_over_examples"),
            "samples_per_example": samples_per_example,
            "expected_examples": expected_examples,
            "observed_examples": observed_examples,
            "expected_predictions": expected_predictions,
            "observed_predictions": observed_predictions,
            "prediction_complete": prediction_complete,
            "score_complete": score_complete,
            "pending_external_scores": pending_external_scores,
            "primary_score": primary_score,
            "subset_scores": subset_scores,
        }
        task_result["score_status"] = (
            "INCOMPLETE_PREDICTIONS"
            if not prediction_complete
            else (
                "SCORED"
                if score_complete
                else (
                    "PENDING_EXTERNAL_CODE_EXECUTION"
                    if pending_external_scores
                    else "INCOMPLETE_SCORE"
                )
            )
        )
        if metric in ACCURACY_METRICS or is_generation:
            task_result["accuracy"] = primary_score
            task_result["correct_samples"] = correct_samples
            task_result["scored_samples"] = scored_samples
        else:
            task_result["mean_target_bits_per_byte"] = primary_score
        if answer_scorers:
            task_result["answer_scorers"] = sorted(answer_scorers)
        if scorer_commits:
            task_result["olmo_eval_commits"] = sorted(scorer_commits)
        if scorer_source_sha256s:
            task_result["official_scorer_source_sha256s"] = sorted(scorer_source_sha256s)
        if math_runtimes:
            if len(math_runtimes) != 1:
                raise RuntimeError(
                    f"task {task_name} used multiple math runtimes: " f"{sorted(math_runtimes)}"
                )
            task_result["official_math_runtime"] = next(iter(math_runtimes.values()))
        if pass_at_4_values:
            task_result["pass@4"] = sum(pass_at_4_values) / len(pass_at_4_values)
        tasks.append(task_result)

    predictions_complete = all(bool(task["prediction_complete"]) for task in tasks)
    scores_complete = all(bool(task["score_complete"]) for task in tasks)
    parallelism_metadata = _manifest_parallelism_metadata(manifests)
    report_status = (
        "CORE_NATIVE_FULL_DIAGNOSTIC"
        if predictions_complete and scores_complete and args.allow_degraded_math_runtime
        else (
            "CORE_NATIVE_FULL_OK"
            if predictions_complete and scores_complete
            else (
                "CORE_NATIVE_FULL_PREDICTIONS_OK"
                if predictions_complete
                else "CORE_NATIVE_FULL_INCOMPLETE"
            )
        )
    )
    report = {
        "schema_version": "hf-core-native-merged-v2",
        "status": report_status,
        "allow_degraded_math_runtime": bool(args.allow_degraded_math_runtime),
        "model_label": next(iter(model_labels), ""),
        "workflow_id": next(iter(workflow_ids), None),
        "repo_commit": next(iter(repo_commits), None),
        "hf_model_path": next(iter(models)),
        "runtime_hf_model_paths": sorted(runtime_model_paths),
        "data_root": str(data_root),
        "profile": args.profile,
        "task_orders": [int(task["task_order"]) for task in source_tasks],
        "source_profiles": sorted(source_profiles),
        "data_summary_sha256": _file_sha256(data_summary_path),
        "input_roots": [str(path.resolve()) for path in args.input_root],
        **parallelism_metadata,
        "prediction_files": prediction_files,
        "expected_examples": len(gold),
        "observed_examples": sum(int(task["observed_examples"]) for task in tasks),
        "expected_predictions": sum(int(task["expected_predictions"]) for task in tasks),
        "observed_predictions": len(seen),
        "task_count": len(tasks),
        "prediction_task_count_complete": sum(bool(task["prediction_complete"]) for task in tasks),
        "score_task_count_complete": sum(bool(task["score_complete"]) for task in tasks),
        "code_result_count": len(code_results),
        "code_result_files": [str(path) for path in code_result_paths],
        "tasks": tasks,
    }
    if len(workflow_ids) > 1:
        raise RuntimeError(
            f"Core88 input roots contain multiple workflow ids: {sorted(workflow_ids)}"
        )
    if len(repo_commits) > 1:
        raise RuntimeError(
            f"Core88 input roots contain multiple repo commits: {sorted(repo_commits)}"
        )
    evaluator_state = _evaluator_repo_state()
    report.update(evaluator_state)
    if args.workflow_manifest is not None:
        workflow = load_workflow(args.workflow_manifest)
        report["evaluator_compatibility"] = _validate_evaluator_repo_state(
            workflow, evaluator_state, allow_descendant=args.allow_scoring_evaluator_descendant
        )
        for manifest in manifests:
            validate_core_run_manifest(workflow, manifest)
        if args.gsm8k_results_root is not None:
            verify_inputs(args.workflow_manifest, args.gsm8k_results_root)
        if workflow_ids != {str(workflow["workflow_id"])}:
            raise RuntimeError("Core88 run manifests do not match the requested workflow id")
        if repo_commits != {str(workflow["repo_commit"])}:
            raise RuntimeError("Core88 run manifests do not match the workflow repository commit")

    prepared_summary: tuple[dict[str, str], dict[str, Any]] | None = None
    if args.output_summary_csv is not None:
        prepared_summary = build_core88_summary(
            report,
            gsm8k_results_root=args.gsm8k_results_root,
            workflow_manifest=args.workflow_manifest,
            sciq_score=args.sciq_score,
            allow_diagnostic_gsm8k=args.allow_missing_workflow_gsm8k,
        )
        summary_row, summary_metadata = prepared_summary
        report["standalone_gsm8k"] = summary_metadata["gsm8k"]
        report["summary_csv"] = str(args.output_summary_csv.resolve())
        report["summary_status"] = summary_metadata["status"]
        report["final_output_eligible"] = bool(summary_metadata["final_output_eligible"])
        if summary_metadata["status"] == DIAGNOSTIC_SUMMARY_STATUS:
            report["status"] = "CORE_NATIVE_FULL_DIAGNOSTIC"
            summary_metadata["detailed_report_status"] = report["status"]
        if args.output_summary_metadata_json is not None:
            report["summary_metadata_json"] = str(args.output_summary_metadata_json.resolve())

    output_csv = args.output_csv or args.output_json.with_suffix(".csv")
    report["scores_csv"] = str(output_csv.resolve())
    _write_scores_csv(output_csv, report)
    if prepared_summary is not None:
        summary_row, summary_metadata = prepared_summary
        summary_metadata = write_prepared_core88_summary_csv(
            args.output_summary_csv, summary_row, summary_metadata
        )
        if args.output_summary_metadata_json is not None:
            summary_metadata["merged_json"] = str(args.output_json.resolve())
            _write_json_atomic(args.output_summary_metadata_json, summary_metadata)
    _write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] == "CORE_NATIVE_FULL_INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
