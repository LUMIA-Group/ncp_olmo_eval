import json
from pathlib import Path

import pytest

from ncp_olmo_eval import core_native_cached_finalize
from ncp_olmo_eval.core_native_cached_finalize import (
    CODE_SCORER_CONTRACT,
    merge_cached_report,
)
from ncp_olmo_eval.core_native_code_eval import MULTIPLE_CORE88_LANGUAGES


def _base_report(tmp_path: Path) -> dict:
    code_orders = {order for orders in CODE_SCORER_CONTRACT for order in orders}
    tasks = []
    for task_order in range(1, 89):
        code_task = task_order in code_orders
        expected = 6 if task_order in {86, 87} else 1
        tasks.append(
            {
                "task_order": task_order,
                "task": f"task-{task_order}",
                "aggregation": (
                    "macro_mean_over_subsets" if task_order in {86, 87} else "mean_over_examples"
                ),
                "expected_examples": expected,
                "expected_predictions": expected,
                "observed_predictions": expected,
                "prediction_complete": True,
                "score_complete": not code_task,
                "pending_external_scores": expected if code_task else 0,
                "primary_score": None if code_task else 1.0,
                "score_status": "PENDING_EXTERNAL_CODE_EXECUTION" if code_task else "SCORED",
            }
        )
    return {
        "schema_version": "hf-core-native-merged-v2",
        "status": "CORE_NATIVE_FULL_PREDICTIONS_OK",
        "profile": "core88",
        "data_summary_sha256": "data-sha",
        "input_roots": [str(tmp_path / "inference")],
        "evaluator_repo_commit": "evaluator-commit",
        "task_count": 88,
        "prediction_task_count_complete": 88,
        "score_task_count_complete": 80,
        "tasks": tasks,
    }


def _code_summaries(tmp_path: Path, report: dict) -> list[Path]:
    paths = []
    for task_orders, partition_count in CODE_SCORER_CONTRACT.items():
        for partition_index in range(partition_count):
            task_counts = {}
            subset_counts = {}
            example_counts = {}
            if partition_index == 0:
                for task_order in task_orders:
                    if task_order in {86, 87}:
                        task_counts[str(task_order)] = {"selected": 6, "passed": 3, "failed": 3}
                        subset_counts[str(task_order)] = {
                            language: {
                                "selected": 1,
                                "passed": int(index % 2 == 0),
                                "failed": int(index % 2 != 0),
                            }
                            for index, language in enumerate(sorted(MULTIPLE_CORE88_LANGUAGES))
                        }
                        example_counts[str(task_order)] = {
                            f'"example-{language}"': {
                                "subset": language,
                                "selected": 1,
                                "passed": int(index % 2 == 0),
                                "failed": int(index % 2 != 0),
                            }
                            for index, language in enumerate(sorted(MULTIPLE_CORE88_LANGUAGES))
                        }
                    else:
                        task_counts[str(task_order)] = {"selected": 1, "passed": 1, "failed": 0}
                        subset_counts[str(task_order)] = {
                            "__all__": {"selected": 1, "passed": 1, "failed": 0}
                        }
                        example_counts[str(task_order)] = {
                            '"example"': {
                                "subset": "__all__",
                                "selected": 1,
                                "passed": 1,
                                "failed": 0,
                            }
                        }
            root = tmp_path / f"g{task_orders[0]}-p{partition_index}"
            root.mkdir(parents=True)
            path = root / f"code-results-p{partition_index}of{partition_count}-summary.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "core-native-code-sandbox-v2",
                        "status": "CORE_NATIVE_CODE_RESULTS_OK",
                        "profile": "core88",
                        "data_summary_sha256": report["data_summary_sha256"],
                        "input_roots": report["input_roots"],
                        "task_orders": list(task_orders),
                        "partition_index": partition_index,
                        "partition_count": partition_count,
                        "output_jsonl": str(root / "code-results.jsonl"),
                        "result_aggregate": {
                            "status": "CORE_NATIVE_CODE_RESULT_AGGREGATE_OK",
                            "task_counts": task_counts,
                            "subset_counts": subset_counts,
                            "example_counts": example_counts,
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def test_cached_finalize_merges_code_summaries_without_raw_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _base_report(tmp_path)
    summaries = _code_summaries(tmp_path, report)
    monkeypatch.setattr(
        core_native_cached_finalize,
        "_evaluator_repo_state",
        lambda: {
            "evaluator_repo_root": "/repo",
            "evaluator_repo_commit": "evaluator-commit",
            "evaluator_repo_dirty": False,
        },
    )

    merged = merge_cached_report(report, summaries)

    assert merged["status"] == "CORE_NATIVE_FULL_OK"
    assert merged["score_task_count_complete"] == 88
    assert merged["code_result_count"] == 18
    assert merged["cached_finalize"] == {
        "status": "CORE_NATIVE_CACHED_FINALIZE_OK",
        "raw_prediction_files_reopened": 0,
        "raw_code_result_files_reopened": 0,
        "code_result_hashes_verified": False,
    }
    tasks = {task["task_order"]: task for task in merged["tasks"]}
    assert tasks[79]["primary_score"] == 1.0
    assert tasks[86]["primary_score"] == 0.5
    assert set(tasks[86]["subset_scores"]) == set(MULTIPLE_CORE88_LANGUAGES)


def test_cached_finalize_rejects_a_stale_prediction_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _base_report(tmp_path)
    monkeypatch.setattr(
        core_native_cached_finalize,
        "_evaluator_repo_state",
        lambda: {
            "evaluator_repo_root": "/repo",
            "evaluator_repo_commit": "new-commit",
            "evaluator_repo_dirty": False,
        },
    )

    with pytest.raises(RuntimeError, match="different evaluator commit"):
        merge_cached_report(report, [])
