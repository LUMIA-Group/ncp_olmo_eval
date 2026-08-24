#!/usr/bin/env python3
"""Materialize Core88 reports from sealed prediction and code-score summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core88_workflow import load_workflow, validate_core_run_manifest, verify_inputs
from .core_native_aggregate import _evaluator_repo_state
from .core_native_code_eval import MULTIPLE_CORE88_LANGUAGES
from .core_native_eval import _file_sha256, _pass_at_k, _write_json_atomic, _write_scores_csv
from .core_native_summary import (
    DIAGNOSTIC_SUMMARY_STATUS,
    _reported_evaluator_state,
    _validate_reported_evaluator,
    build_core88_summary,
    write_prepared_core88_summary_csv,
)

CODE_SCORER_CONTRACT = {(79, 82, 83, 85): 8, (81,): 8, (84,): 8, (86, 87): 32}
CODE_TASK_ORDERS = frozenset(order for orders in CODE_SCORER_CONTRACT for order in orders)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _summary_paths(directories: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        paths.extend(sorted(directory.rglob("code-results-*-summary.json")))
    return sorted({path.resolve() for path in paths})


def _merge_code_summaries(
    report: dict[str, Any], summary_paths: list[Path], *, verify_code_result_hashes: bool
) -> tuple[
    dict[int, dict[str, int]],
    dict[int, dict[str, dict[str, int]]],
    dict[int, dict[str, dict[str, Any]]],
]:
    expected_input_roots = sorted(str(Path(path).resolve()) for path in report["input_roots"])
    observed_partitions: dict[tuple[int, ...], set[int]] = defaultdict(set)
    task_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"selected": 0, "passed": 0, "failed": 0}
    )
    subset_counts: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"selected": 0, "passed": 0, "failed": 0})
    )
    example_counts: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in summary_paths:
        summary = _read_json(path)
        if summary.get("schema_version") != "core-native-code-sandbox-v2":
            raise RuntimeError(f"legacy code summary cannot use cached finalize: {path}")
        if summary.get("status") != "CORE_NATIVE_CODE_RESULTS_OK":
            raise RuntimeError(f"incomplete code summary: {path}")
        if summary.get("profile") != report.get("profile"):
            raise RuntimeError(f"profile mismatch in {path}")
        if summary.get("data_summary_sha256") != report.get("data_summary_sha256"):
            raise RuntimeError(f"dataset identity mismatch in {path}")
        input_roots = sorted(str(Path(value).resolve()) for value in summary.get("input_roots", []))
        if input_roots != expected_input_roots:
            raise RuntimeError(f"input root mismatch in {path}")
        task_orders = tuple(int(value) for value in summary.get("task_orders", []))
        expected_partition_count = CODE_SCORER_CONTRACT.get(task_orders)
        partition_count = int(summary.get("partition_count", -1))
        partition_index = int(summary.get("partition_index", -1))
        if expected_partition_count is None or partition_count != expected_partition_count:
            raise RuntimeError(f"unsupported code scorer partition contract in {path}")
        if not 0 <= partition_index < partition_count:
            raise RuntimeError(f"invalid code scorer partition index in {path}")
        if partition_index in observed_partitions[task_orders]:
            raise RuntimeError(
                f"duplicate code scorer partition for {task_orders}: {partition_index}"
            )
        observed_partitions[task_orders].add(partition_index)

        aggregate = summary.get("result_aggregate")
        if not isinstance(aggregate, dict) or aggregate.get("status") != (
            "CORE_NATIVE_CODE_RESULT_AGGREGATE_OK"
        ):
            raise RuntimeError(f"incomplete code result aggregate in {path}")
        output_jsonl = Path(str(summary["output_jsonl"])).resolve()
        if verify_code_result_hashes:
            if not output_jsonl.is_file():
                raise RuntimeError(f"code result file is missing: {output_jsonl}")
            if _file_sha256(output_jsonl) != aggregate.get("output_jsonl_sha256"):
                raise RuntimeError(f"code result SHA256 mismatch: {output_jsonl}")
        for key, counts in (aggregate.get("task_counts") or {}).items():
            task_order = int(key)
            if task_order not in task_orders:
                raise RuntimeError(f"unexpected task {task_order} in {path}")
            for field in ("selected", "passed", "failed"):
                task_counts[task_order][field] += int(counts[field])
        for key, subsets in (aggregate.get("subset_counts") or {}).items():
            task_order = int(key)
            if task_order not in task_orders:
                raise RuntimeError(f"unexpected subset task {task_order} in {path}")
            for subset, counts in subsets.items():
                for field in ("selected", "passed", "failed"):
                    subset_counts[task_order][str(subset)][field] += int(counts[field])
        for key, examples in (aggregate.get("example_counts") or {}).items():
            task_order = int(key)
            if task_order not in task_orders:
                raise RuntimeError(f"unexpected example task {task_order} in {path}")
            for example_id, counts in examples.items():
                subset = str(counts["subset"])
                target = example_counts[task_order].setdefault(
                    str(example_id), {"subset": subset, "selected": 0, "passed": 0, "failed": 0}
                )
                if target["subset"] != subset:
                    raise RuntimeError(f"example subset changed across scorer partitions in {path}")
                for field in ("selected", "passed", "failed"):
                    target[field] += int(counts[field])

    for task_orders, partition_count in CODE_SCORER_CONTRACT.items():
        expected = set(range(partition_count))
        if observed_partitions.get(task_orders, set()) != expected:
            raise RuntimeError(
                f"code scorer partitions are incomplete for {task_orders}: "
                f"{sorted(observed_partitions.get(task_orders, set()))}"
            )
    return (
        dict(task_counts),
        {task_order: dict(subsets) for task_order, subsets in subset_counts.items()},
        {task_order: dict(examples) for task_order, examples in example_counts.items()},
    )


def merge_cached_report(
    base_report: dict[str, Any],
    summary_paths: list[Path],
    *,
    verify_code_result_hashes: bool = False,
) -> dict[str, Any]:
    """Replace pending code scores without reopening raw prediction JSONL."""

    report = json.loads(json.dumps(base_report))
    if report.get("schema_version") != "hf-core-native-merged-v2":
        raise RuntimeError("unsupported Core88 prediction score snapshot schema")
    if report.get("status") != "CORE_NATIVE_FULL_PREDICTIONS_OK":
        raise RuntimeError("Core88 prediction score snapshot is not prediction-complete")
    if report.get("prediction_task_count_complete") != report.get("task_count"):
        raise RuntimeError("Core88 prediction score snapshot has incomplete predictions")
    reported_evaluator = _reported_evaluator_state(report)
    if reported_evaluator["evaluator_repo_dirty"] is not False:
        raise RuntimeError("prediction score snapshot records a dirty evaluator repository")
    if not str(reported_evaluator["evaluator_repo_commit"] or ""):
        raise RuntimeError("prediction score snapshot is missing its evaluator commit")
    if len(str(reported_evaluator["evaluator_source_tree_sha256"] or "")) != 64:
        raise RuntimeError("prediction score snapshot is missing its evaluator source-tree digest")

    task_counts, subset_counts, example_counts = _merge_code_summaries(
        report, summary_paths, verify_code_result_hashes=verify_code_result_hashes
    )
    tasks_by_order = {int(task["task_order"]): task for task in report["tasks"]}
    pending_code_orders = {
        task_order
        for task_order, task in tasks_by_order.items()
        if int(task.get("pending_external_scores", 0)) > 0
    }
    if pending_code_orders != CODE_TASK_ORDERS:
        raise RuntimeError(f"unexpected pending Core88 code tasks: {sorted(pending_code_orders)}")
    for task_order in sorted(CODE_TASK_ORDERS):
        task = tasks_by_order[task_order]
        counts = task_counts.get(task_order, {"selected": 0, "passed": 0, "failed": 0})
        selected = int(counts["selected"])
        passed = int(counts["passed"])
        failed = int(counts["failed"])
        expected = int(task["expected_predictions"])
        if selected != expected or passed + failed != selected:
            raise RuntimeError(
                f"task {task_order} code result coverage mismatch: "
                f"selected={selected}, passed={passed}, failed={failed}, expected={expected}"
            )
        task_example_counts = example_counts.get(task_order, {})
        expected_examples = int(task["expected_examples"])
        if len(task_example_counts) != expected_examples:
            raise RuntimeError(
                f"task {task_order} code example coverage mismatch: "
                f"{len(task_example_counts)} != {expected_examples}"
            )
        example_values: list[float] = []
        values_by_subset: dict[str, list[float]] = defaultdict(list)
        pass_at_4_values: list[float] = []
        for values in task_example_counts.values():
            if values["selected"] != values["passed"] + values["failed"]:
                raise RuntimeError(f"task {task_order} has inconsistent per-example counts")
            value = values["passed"] / values["selected"]
            example_values.append(value)
            values_by_subset[str(values["subset"])].append(value)
            if "pass@4" in (task.get("secondary_metrics") or []):
                pass_at_4 = _pass_at_k([True] * values["passed"] + [False] * values["failed"], 4)
                if pass_at_4 is not None:
                    pass_at_4_values.append(pass_at_4)
        subset_scores = {
            subset: sum(values) / len(values) for subset, values in sorted(values_by_subset.items())
        }
        task_subset_counts = subset_counts.get(task_order, {})
        if set(task_subset_counts) != set(values_by_subset):
            raise RuntimeError(f"task {task_order} subset aggregate coverage mismatch")
        for subset, aggregate_counts in task_subset_counts.items():
            matching_examples = [
                values for values in task_example_counts.values() if values["subset"] == subset
            ]
            for field in ("selected", "passed", "failed"):
                if int(aggregate_counts[field]) != sum(
                    int(values[field]) for values in matching_examples
                ):
                    raise RuntimeError(
                        f"task {task_order} subset {subset} {field} aggregate mismatch"
                    )
        if task.get("aggregation") == "macro_mean_over_subsets":
            if set(subset_scores) != set(MULTIPLE_CORE88_LANGUAGES):
                raise RuntimeError(
                    f"task {task_order} language coverage mismatch: {sorted(subset_scores)}"
                )
            primary_score = sum(subset_scores.values()) / len(subset_scores)
        else:
            primary_score = sum(example_values) / len(example_values)
        task.update(
            {
                "score_complete": True,
                "pending_external_scores": 0,
                "primary_score": primary_score,
                "subset_scores": subset_scores,
                "score_status": "SCORED",
                "accuracy": primary_score,
                "correct_samples": passed,
                "scored_samples": selected,
            }
        )
        if pass_at_4_values:
            task["pass@4"] = sum(pass_at_4_values) / len(pass_at_4_values)

    code_result_files = [str(_read_json(path)["output_jsonl"]) for path in summary_paths]
    report.update(
        {
            "status": "CORE_NATIVE_FULL_OK",
            "score_task_count_complete": len(report["tasks"]),
            "code_result_count": sum(values["selected"] for values in task_counts.values()),
            "code_result_files": code_result_files,
            "code_result_summary_files": [str(path) for path in summary_paths],
            "cached_finalize": {
                "status": "CORE_NATIVE_CACHED_FINALIZE_OK",
                "raw_prediction_files_reopened": 0,
                "raw_code_result_files_reopened": (
                    len(summary_paths) if verify_code_result_hashes else 0
                ),
                "code_result_hashes_verified": verify_code_result_hashes,
            },
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--code-summary-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-summary-metadata-json", type=Path, required=True)
    parser.add_argument("--gsm8k-results-root", type=Path, required=True)
    parser.add_argument("--workflow-manifest", type=Path, default=None)
    parser.add_argument("--allow-missing-workflow-gsm8k", action="store_true")
    parser.add_argument("--allow-scoring-evaluator-descendant", action="store_true")
    parser.add_argument("--sciq-score", type=float, default=None)
    parser.add_argument("--verify-code-result-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_report = _read_json(args.base_report.resolve())
    summary_paths = _summary_paths(args.code_summary_dir)
    report = merge_cached_report(
        base_report, summary_paths, verify_code_result_hashes=args.verify_code_result_hashes
    )
    finalizer_state = _evaluator_repo_state()
    if finalizer_state["evaluator_repo_dirty"] is not False:
        raise RuntimeError("cached Core88 finalizer repository must be clean")
    report["finalizer_evaluator_state"] = finalizer_state
    if args.workflow_manifest is not None:
        workflow = load_workflow(args.workflow_manifest)
        report["evaluator_compatibility"] = _validate_reported_evaluator(
            workflow,
            report,
            allow_descendant=args.allow_scoring_evaluator_descendant,
        )
        if report.get("workflow_id") != workflow.get("workflow_id"):
            raise RuntimeError("prediction score snapshot has a different workflow id")
        if report.get("repo_commit") != workflow.get("repo_commit"):
            raise RuntimeError(
                "prediction score snapshot has a different workflow repository commit"
            )
        for input_root in report["input_roots"]:
            validate_core_run_manifest(workflow, _read_json(Path(input_root) / "run_manifest.json"))
        verify_inputs(args.workflow_manifest, args.gsm8k_results_root)

    summary_row, summary_metadata = build_core88_summary(
        report,
        gsm8k_results_root=args.gsm8k_results_root,
        workflow_manifest=args.workflow_manifest,
        sciq_score=args.sciq_score,
        allow_diagnostic_gsm8k=args.allow_missing_workflow_gsm8k,
    )
    report["standalone_gsm8k"] = summary_metadata["gsm8k"]
    report["summary_csv"] = str(args.output_summary_csv.resolve())
    report["summary_status"] = summary_metadata["status"]
    report["final_output_eligible"] = bool(summary_metadata["final_output_eligible"])
    report["summary_metadata_json"] = str(args.output_summary_metadata_json.resolve())
    if summary_metadata["status"] == DIAGNOSTIC_SUMMARY_STATUS:
        report["status"] = "CORE_NATIVE_FULL_DIAGNOSTIC"
        summary_metadata["detailed_report_status"] = report["status"]
    report["scores_csv"] = str(args.output_csv.resolve())
    _write_scores_csv(args.output_csv, report)
    summary_metadata = write_prepared_core88_summary_csv(
        args.output_summary_csv, summary_row, summary_metadata
    )
    summary_metadata["merged_json"] = str(args.output_json.resolve())
    _write_json_atomic(args.output_summary_metadata_json, summary_metadata)
    _write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
