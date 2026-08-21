#!/usr/bin/env python3
"""Score sealed HELMET predictions with the pinned official implementations."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .helmet_protocol import validate_official_helmet_checkout, write_official_compatible_results
from .helmet_trec_eval_compat import compatibility_metadata, install_if_missing
from .long_context_protocol import (
    HELMET_BENCHMARK,
    SCORE_STATUS,
    file_sha256,
    validate_prepared_dataset,
    write_csv,
    write_json_atomic,
    write_jsonl_atomic,
)
from .long_context_score import _load_shards

HELMET_PRIMARY_METRICS = {
    "json_kv": ("substring_exact_match",),
    "ruler_niah_mk_2": ("ruler_recall",),
    "ruler_niah_mk_3": ("ruler_recall",),
    "ruler_niah_mv": ("ruler_recall",),
    "kilt_nq": ("substring_exact_match",),
    "kilt_hotpotqa": ("substring_exact_match",),
    "kilt_popqa_3": ("substring_exact_match",),
    "kilt_triviaqa": ("substring_exact_match",),
    "msmarco_rerank_psg": ("NDCG@10",),
    "icl_trec_coarse": ("exact_match",),
    "icl_trec_fine": ("exact_match",),
    "icl_banking77": ("exact_match",),
    "icl_clinic150": ("exact_match",),
    "icl_nlu": ("exact_match",),
    "alce_asqa": ("str_em", "citation_rec", "citation_prec"),
    "alce_qampari": ("qampari_rec_top5", "citation_rec", "citation_prec"),
    "narrativeqa": ("gpt-4-score",),
    "infbench_qa_eng": ("rougeL_f1",),
    "infbench_choice_eng": ("exact_match",),
    "infbench_sum_eng": ("gpt-4-f1",),
    "multi_lexsum": ("gpt-4-f1",),
}
HELMET_CATEGORY_PRIMARY_METRICS = {
    "Recall": (
        ("json_kv", "substring_exact_match"),
        ("ruler_niah_mk_2", "ruler_recall"),
        ("ruler_niah_mk_3", "ruler_recall"),
        ("ruler_niah_mv", "ruler_recall"),
    ),
    "RAG": (
        ("kilt_nq", "substring_exact_match"),
        ("kilt_hotpotqa", "substring_exact_match"),
        ("kilt_popqa_3", "substring_exact_match"),
        ("kilt_triviaqa", "substring_exact_match"),
    ),
    "ICL": (
        ("icl_trec_coarse", "exact_match"),
        ("icl_trec_fine", "exact_match"),
        ("icl_banking77", "exact_match"),
        ("icl_clinic150", "exact_match"),
        ("icl_nlu", "exact_match"),
    ),
    "Cite": (
        ("alce_asqa", "str_em"),
        ("alce_asqa", "citation_rec"),
        ("alce_asqa", "citation_prec"),
        ("alce_qampari", "qampari_rec_top5"),
        ("alce_qampari", "citation_rec"),
        ("alce_qampari", "citation_prec"),
    ),
    "Re-rank": (("msmarco_rerank_psg", "NDCG@10"),),
    "LongQA": (
        ("narrativeqa", "gpt-4-score"),
        ("infbench_qa_eng", "rougeL_f1"),
        ("infbench_choice_eng", "exact_match"),
    ),
    "Summ": (("infbench_sum_eng", "gpt-4-f1"), ("multi_lexsum", "gpt-4-f1")),
}

HELMET_TASK_FAMILY_PATTERNS = (
    (re.compile(r"^icl_trec_coarse_\d+shot_balance$"), "icl_trec_coarse"),
    (re.compile(r"^icl_trec_fine_\d+shot_balance$"), "icl_trec_fine"),
    (re.compile(r"^icl_banking77_\d+shot_balance$"), "icl_banking77"),
    (re.compile(r"^icl_clinic150_\d+shot_balance$"), "icl_clinic150"),
    (re.compile(r"^icl_nlu_\d+shot_balance$"), "icl_nlu"),
    (re.compile(r"^alce_asqa_\d+$"), "alce_asqa"),
    (re.compile(r"^alce_qampari_\d+$"), "alce_qampari"),
    (re.compile(r"^narrativeqa_\d+$"), "narrativeqa"),
    (re.compile(r"^infbench_qa_eng_\d+$"), "infbench_qa_eng"),
    (re.compile(r"^infbench_choice_eng_\d+$"), "infbench_choice_eng"),
    (re.compile(r"^infbench_sum_eng_\d+$"), "infbench_sum_eng"),
    (re.compile(r"^multi_lexsum_\d+$"), "multi_lexsum"),
)


def _helmet_task_family(task: str) -> str:
    """Map one profile-specific HELMET task ID to its stable metric family."""

    if task in HELMET_PRIMARY_METRICS:
        return task
    matches = [family for pattern, family in HELMET_TASK_FAMILY_PATTERNS if pattern.fullmatch(task)]
    if len(matches) != 1:
        raise ValueError(f"unsupported HELMET task family: {task}")
    return matches[0]


def _tasks_by_family(
    rows: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, str]], dict[tuple[int, str], str]]:
    """Index exact profile task IDs without collapsing distinct task lengths."""

    tasks: dict[int, dict[str, str]] = defaultdict(dict)
    categories: dict[tuple[int, str], str] = {}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("HELMET row lacks scoring metadata")
        input_max_length = int(metadata["input_max_length"])
        task = str(row["task"])
        family = _helmet_task_family(task)
        previous = tasks[input_max_length].setdefault(family, task)
        if previous != task:
            raise ValueError(
                "multiple HELMET task IDs resolve to one family at one length: "
                f"length={input_max_length} family={family} tasks={previous},{task}"
            )
        category = str(metadata["category"])
        category_key = (input_max_length, task)
        previous_category = categories.setdefault(category_key, category)
        if previous_category != category:
            raise ValueError(f"HELMET task category differs across rows: {category_key}")
    return {length: dict(mapping) for length, mapping in tasks.items()}, categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-citation-nli", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _configure_official_scoring_runtime(eval_alce: Any) -> dict[str, Any]:
    """Apply explicit local AutoAIS and NLTK cache overrides when configured."""

    runtime: dict[str, Any] = {
        "autoais_model": str(getattr(eval_alce, "AUTOAIS_MODEL", "")),
        "autoais_model_source": "official_default",
        "nltk_data": None,
    }
    autoais_model = os.environ.get("HELMET_AUTOAIS_MODEL")
    if autoais_model:
        autoais_path = Path(autoais_model).expanduser()
        if not autoais_path.is_dir():
            raise FileNotFoundError(
                f"HELMET_AUTOAIS_MODEL is not a local model directory: {autoais_path}"
            )
        resolved_autoais = autoais_path.resolve()
        eval_alce.AUTOAIS_MODEL = str(resolved_autoais)
        runtime["autoais_model"] = str(resolved_autoais)
        runtime["autoais_model_source"] = "local_override"

    nltk_data = os.environ.get("NLTK_DATA")
    if nltk_data:
        nltk_path = Path(nltk_data).expanduser()
        if not nltk_path.is_dir():
            raise FileNotFoundError(f"NLTK_DATA is not a local cache directory: {nltk_path}")
        resolved_nltk = str(nltk_path.resolve())
        runtime["nltk_data"] = resolved_nltk
        nltk_module = sys.modules.get("nltk")
        nltk_search_path = getattr(getattr(nltk_module, "data", None), "path", None)
        if isinstance(nltk_search_path, list) and resolved_nltk not in nltk_search_path:
            nltk_search_path.insert(0, resolved_nltk)
    return runtime


def _load_official_modules(official_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    install_if_missing()
    root_string = str(official_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    utils = importlib.import_module("utils")
    eval_alce = importlib.import_module("eval_alce")
    if Path(utils.__file__).resolve() != official_root.resolve() / "utils.py":
        raise RuntimeError(f"imported wrong HELMET utils module: {utils.__file__}")
    if Path(eval_alce.__file__).resolve() != official_root.resolve() / "eval_alce.py":
        raise RuntimeError(f"imported wrong HELMET eval_alce module: {eval_alce.__file__}")
    citation_runtime = _configure_official_scoring_runtime(eval_alce)
    return utils, eval_alce, citation_runtime


def _max_default_metrics(output: str, answer: Any, utils: Any, prefix: str) -> dict[str, float]:
    raw = utils.calculate_metrics(output, answer)
    parsed = utils.parse_output(output, prefix)
    if parsed is None:
        return {str(key): float(value) for key, value in raw.items()}
    parsed_metrics = utils.calculate_metrics(parsed, answer)
    return {str(key): float(max(value, parsed_metrics[key])) for key, value in raw.items()}


def _score_one(
    task: str, output: str, payload: dict[str, Any], utils: Any
) -> tuple[dict[str, float], str | None]:
    answer = payload.get("answer")
    family = _helmet_task_family(task)
    if family == "json_kv":
        metrics = _max_default_metrics(output, answer, utils, "corresponding value:")
        return metrics, utils.parse_output(output, "corresponding value:")
    if family.startswith("ruler_"):
        answers = answer if isinstance(answer, list) else [answer]
        recall = sum(str(value).lower() in output.lower() for value in answers) / len(answers)
        return {"ruler_recall": float(recall)}, output
    if family.startswith("icl_"):
        parsed = utils.parse_output(output, "label:")
        return utils.calculate_metrics(parsed, answer), parsed
    if family == "infbench_choice_eng":
        metrics = _max_default_metrics(output, answer, utils, "Answer:")
        metrics.pop("substring_exact_match", None)
        metrics["substring_exact_match"] = 0.0
        answers = answer if isinstance(answer, list) else [answer]
        if len(answers) > 1 and str(answers[1]).lower() in output.lower():
            metrics["substring_exact_match"] = 1.0
            metrics["exact_match"] = 1.0
        return metrics, utils.parse_output(output, "Answer:")
    if family == "msmarco_rerank_psg":
        return {}, None
    if family in {"alce_asqa", "alce_qampari"}:
        return {}, None
    return _max_default_metrics(output, answer, utils, "Answer:"), utils.parse_output(
        output, "Answer:"
    )


def _length_label(input_max_length: int) -> str:
    if input_max_length % 1024 != 0:
        return str(input_max_length)
    return f"{input_max_length // 1024}k"


def _judge_metrics(
    path: Path | None, input_lengths: tuple[int, ...]
) -> dict[tuple[int, str, str], float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("metrics", payload)
    if not isinstance(rows, list):
        raise ValueError("judge results must contain a metrics list")
    result: dict[tuple[int, str, str], float] = {}
    for row in rows:
        declared_length = row.get("input_max_length")
        if declared_length is None:
            if len(input_lengths) != 1:
                raise ValueError(
                    "multi-length HELMET judge results require input_max_length on every row"
                )
            input_max_length = input_lengths[0]
        else:
            input_max_length = int(declared_length)
        if input_max_length not in input_lengths:
            raise ValueError(
                f"judge result length is not present in prepared data: {input_max_length}"
            )
        key = (input_max_length, str(row["dataset"]), str(row["metric"]))
        if key in result:
            raise ValueError(f"duplicate judge metric: {key}")
        value = float(row["score_percent"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite judge metric: {key}")
        result[key] = value
    return result


def score_run(args: argparse.Namespace) -> dict[str, Any]:
    official_contract = validate_official_helmet_checkout(args.official_root.resolve())
    utils, eval_alce, citation_runtime = _load_official_modules(args.official_root.resolve())
    data_root = args.data_root.resolve()
    manifest, rows = validate_prepared_dataset(data_root)
    if manifest.get("benchmark") != HELMET_BENCHMARK:
        raise ValueError("prepared data is not HELMET")
    results, predictions = _load_shards(
        args.inference_root.resolve(),
        prepared_manifest_sha256=file_sha256(data_root / "manifest.json"),
        prepared_inputs_sha256=file_sha256(data_root / "inputs.jsonl"),
    )
    row_by_id = {str(row["example_id"]): row for row in rows}
    input_lengths = tuple(sorted({int(row["metadata"]["input_max_length"]) for row in rows}))
    protocol_input_lengths = tuple(
        sorted(int(value) for value in manifest.get("protocol", {}).get("input_lengths", []))
    )
    formal_example_count = int(manifest.get("protocol", {}).get("formal_example_count", -1))
    formal_prepared_dataset = (
        manifest.get("protocol", {}).get("formal_data_validation_passed") is True
        and len(row_by_id) == formal_example_count
        and input_lengths == protocol_input_lengths
    )
    tasks_by_length, category_by_task = _tasks_by_family(list(row_by_id.values()))
    if formal_prepared_dataset:
        expected_families = set(HELMET_PRIMARY_METRICS)
        for input_max_length in input_lengths:
            observed_families = set(tasks_by_length.get(input_max_length, {}))
            if observed_families != expected_families:
                raise ValueError(
                    "formal HELMET task family coverage mismatch: "
                    f"length={input_max_length} "
                    f"missing={sorted(expected_families - observed_families)} "
                    f"extra={sorted(observed_families - expected_families)}"
                )
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        example_id = str(prediction["example_id"])
        if int(prediction.get("sample_index", -1)) != 0:
            raise ValueError("formal HELMET scoring requires sample_index=0")
        if example_id in prediction_by_id:
            raise ValueError(f"duplicate HELMET prediction: {example_id}")
        prediction_by_id[example_id] = prediction
    if args.allow_partial:
        rows = [row for row in rows if str(row["example_id"]) in prediction_by_id]
    elif set(prediction_by_id) != set(row_by_id):
        missing = sorted(set(row_by_id) - set(prediction_by_id))
        extra = sorted(set(prediction_by_id) - set(row_by_id))
        raise ValueError(
            f"HELMET prediction coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    scored_rows: list[dict[str, Any]] = []
    per_dataset_values: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    citation_data: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    rerank_results: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    rerank_qrels: dict[int, dict[str, dict[str, int]]] = defaultdict(dict)
    for row in rows:
        example_id = str(row["example_id"])
        prediction = prediction_by_id[example_id]
        generation = str(prediction["generation"])
        metadata = row["metadata"]
        task = str(row["task"])
        family = _helmet_task_family(task)
        input_max_length = int(metadata["input_max_length"])
        output = str(metadata["completion_prefix"]) + generation
        payload = copy.deepcopy(metadata["scoring_payload"])
        metrics, parsed = _score_one(task, output, payload, utils)
        if family == "msmarco_rerank_psg":
            qid = str(payload["qid"])
            rerank_results[input_max_length][qid] = utils.parse_rankings(output)
            rerank_qrels[input_max_length][qid] = {
                str(item[0]): int(item[1]) for item in payload["qrel"]
            }
        if family in {"alce_asqa", "alce_qampari"}:
            payload["output"] = output
            citation_data[(input_max_length, task)].append(payload)
        for metric, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite HELMET metric: {example_id} {metric}")
            per_dataset_values[(input_max_length, task)][str(metric)].append(numeric * 100.0)
        scored_rows.append(
            {
                "example_id": example_id,
                "input_max_length": input_max_length,
                "input_length_label": _length_label(input_max_length),
                "category": metadata["category"],
                "dataset": task,
                "generation": generation,
                "official_output": output,
                "parsed_output": parsed,
                "metrics": metrics,
            }
        )
    for input_max_length, length_results in rerank_results.items():
        length_qrels = rerank_qrels[input_max_length]
        k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        k_values = [value for value in k_values if value <= min(map(len, length_qrels.values()))]
        retrieval = utils.calculate_retrieval_metrics(
            results=length_results, qrels=length_qrels, k_values=k_values
        )
        for metric, value in retrieval.items():
            per_dataset_values[(input_max_length, "msmarco_rerank_psg")][str(metric)] = [
                float(value) * 100.0
            ]
    for (input_max_length, task), data in citation_data.items():
        family = _helmet_task_family(task)
        normalized = copy.deepcopy(data)
        for item in normalized:
            item["output"] = re.sub(r"\n+", " ", str(item["output"])).replace("<|im_end|>", "")
        without_citations = copy.deepcopy(normalized)
        for item in without_citations:
            item["output"] = utils.remove_citations(item["output"])
        if family == "alce_asqa":
            str_em, _ = eval_alce.compute_str_em(without_citations)
            per_dataset_values[(input_max_length, task)]["str_em"] = [float(str_em)]
        elif family == "alce_qampari":
            qampari = eval_alce.compute_qampari_f1(without_citations)
            for metric, value in qampari.items():
                per_dataset_values[(input_max_length, task)][str(metric)] = [float(value)]
        if args.run_citation_nli:
            citation_metrics = eval_alce.compute_autoais(
                normalized, qampari=family == "alce_qampari", at_most_citations=3
            )
            for metric in ("citation_rec", "citation_prec"):
                per_dataset_values[(input_max_length, task)][metric] = [
                    float(citation_metrics[metric])
                ]

    judge_metrics = _judge_metrics(args.judge_results, input_lengths)
    dataset_scores: dict[tuple[int, str, str], float] = {}
    for (input_max_length, task), metrics in per_dataset_values.items():
        for metric, values in metrics.items():
            dataset_scores[(input_max_length, task, metric)] = mean(values)
    dataset_scores.update(judge_metrics)
    dataset_rows: list[dict[str, Any]] = []
    for input_max_length in input_lengths:
        for family, primary_metrics in HELMET_PRIMARY_METRICS.items():
            task = tasks_by_length.get(input_max_length, {}).get(family)
            if task is None:
                continue
            category = category_by_task[(input_max_length, task)]
            for metric in primary_metrics:
                score_value = dataset_scores.get((input_max_length, task, metric))
                dataset_rows.append(
                    {
                        "input_max_length": input_max_length,
                        "input_length_label": _length_label(input_max_length),
                        "category": category,
                        "dataset": task,
                        "metric": metric,
                        "score_percent": score_value,
                        "status": (
                            "available" if score_value is not None else "pending_external_judge"
                        ),
                        "example_count": sum(
                            str(row["task"]) == task
                            and int(row["metadata"]["input_max_length"]) == input_max_length
                            for row in rows
                        ),
                    }
                )

    category_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for input_max_length in input_lengths:
        complete_category_scores: list[float] = []
        for category, metric_keys in HELMET_CATEGORY_PRIMARY_METRICS.items():
            values = [
                dataset_scores.get(
                    (
                        input_max_length,
                        tasks_by_length.get(input_max_length, {}).get(family, ""),
                        metric,
                    )
                )
                for family, metric in metric_keys
            ]
            complete = all(value is not None for value in values)
            category_score = (
                mean(value for value in values if value is not None) if complete else None
            )
            if category_score is not None:
                complete_category_scores.append(category_score)
            category_rows.append(
                {
                    "input_max_length": input_max_length,
                    "input_length_label": _length_label(input_max_length),
                    "category": category,
                    "score_percent": category_score,
                    "status": "complete" if complete else "pending_external_judge",
                    "available_metric_count": sum(value is not None for value in values),
                    "required_metric_count": len(values),
                }
            )
        length_score = (
            mean(complete_category_scores)
            if formal_prepared_dataset
            and not args.allow_partial
            and len(complete_category_scores) == 7
            else None
        )
        length_rows.append(
            {
                "input_max_length": input_max_length,
                "input_length_label": _length_label(input_max_length),
                "official_primary_score_percent": length_score,
                "official_primary_score_complete": length_score is not None,
                "prepared_example_count": sum(
                    int(row["metadata"]["input_max_length"]) == input_max_length for row in rows
                ),
                "prediction_count": sum(
                    int(row["metadata"]["input_max_length"]) == input_max_length
                    and str(row["example_id"]) in prediction_by_id
                    for row in rows
                ),
            }
        )
    complete_length_scores = [
        float(row["official_primary_score_percent"])
        for row in length_rows
        if row["official_primary_score_percent"] is not None
    ]
    single_length_official_score = (
        complete_length_scores[0]
        if len(input_lengths) == 1 and len(complete_length_scores) == 1
        else None
    )
    sweep_macro_average = (
        mean(complete_length_scores) if len(complete_length_scores) == len(input_lengths) else None
    )
    score = {
        "schema_version": manifest["schema_version"],
        "status": SCORE_STATUS,
        "benchmark": HELMET_BENCHMARK,
        "model_label": results[0]["model_label"],
        "backend": results[0]["backend"],
        "resolved_backend": results[0]["resolved_backend"],
        "batch_size": results[0]["batch_size"],
        "global_seed": results[0]["global_seed"],
        "prediction_count": len(predictions),
        "prepared_example_count": len(row_by_id),
        "input_lengths": list(input_lengths),
        "input_length_labels": [_length_label(value) for value in input_lengths],
        "formal_complete_dataset": formal_prepared_dataset and not args.allow_partial,
        "inference_official_protocol_compatible": formal_prepared_dataset
        and all(result.get("official_protocol_compatible") is True for result in results),
        "citation_nli_complete": bool(args.run_citation_nli),
        "model_judge_complete": all(
            (input_max_length, tasks_by_length.get(input_max_length, {}).get(family, ""), metric)
            in dataset_scores
            for input_max_length in input_lengths
            for family, metric in (
                ("narrativeqa", "gpt-4-score"),
                ("infbench_sum_eng", "gpt-4-f1"),
                ("multi_lexsum", "gpt-4-f1"),
            )
        ),
        "official_primary_score_percent": single_length_official_score,
        "official_primary_score_complete": single_length_official_score is not None,
        "all_length_primary_scores_complete": len(complete_length_scores) == len(input_lengths),
        "sweep_macro_average_score_percent": sweep_macro_average,
        "sweep_macro_average_is_official_primary_score": False,
        "length_rows": length_rows,
        "dataset_rows": dataset_rows,
        "category_rows": category_rows,
        "official_source": official_contract,
        "citation_runtime": citation_runtime,
        "trec_eval": compatibility_metadata(),
        "prepared_manifest_sha256": file_sha256(data_root / "manifest.json"),
        "prepared_inputs_sha256": file_sha256(data_root / "inputs.jsonl"),
        "inference_shard_count": len(results),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "score.json", score)
    write_csv(output_root / "helmet-dataset-summary.csv", dataset_rows)
    write_csv(output_root / "helmet-category-summary.csv", category_rows)
    write_csv(output_root / "helmet-length-summary.csv", length_rows)
    write_jsonl_atomic(output_root / "helmet-scored.jsonl", scored_rows)
    generations = {
        example_id: str(prediction["generation"])
        for example_id, prediction in prediction_by_id.items()
    }
    for input_max_length in input_lengths:
        length_rows_for_judge = [
            row for row in rows if int(row["metadata"]["input_max_length"]) == input_max_length
        ]
        judge_path = output_root / (
            f"helmet-official-judge-input-{_length_label(input_max_length)}.json"
        )
        write_official_compatible_results(judge_path, length_rows_for_judge, generations)
    if len(input_lengths) == 1:
        write_official_compatible_results(
            output_root / "helmet-official-judge-input.json", rows, generations
        )
    return score


def main() -> None:
    result = score_run(parse_args())
    print(f"status={result['status']}")
    print(f"benchmark={result['benchmark']}")
    print(f"official_primary_score_complete={result['official_primary_score_complete']}")


if __name__ == "__main__":
    main()
