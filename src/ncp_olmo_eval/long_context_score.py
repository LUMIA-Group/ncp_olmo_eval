#!/usr/bin/env python3
"""Validate, score, and report RULER or LongBench v2 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .long_context_protocol import (
    LONGBENCH_V2_BENCHMARK,
    RULER_BENCHMARK,
    SHARD_STATUS,
    assert_finite_scores,
    file_sha256,
    read_json_or_jsonl,
    score_predictions,
    validate_prepared_dataset,
    write_csv,
    write_json_atomic,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    """Parse the long-context scoring command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow fixture/smoke subsets; formal reports must leave this disabled.",
    )
    return parser.parse_args()


def _load_shards(
    inference_root: Path, *, prepared_manifest_sha256: str, prepared_inputs_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_paths = sorted(inference_root.glob("shard-*-of-*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"no completed inference shards under {inference_root}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    shard_counts = {int(result.get("shard_count", -1)) for result in results}
    if len(shard_counts) != 1:
        raise ValueError("inference shards disagree on shard_count")
    shard_count = next(iter(shard_counts))
    indices = {int(result.get("shard_index", -1)) for result in results}
    if indices != set(range(shard_count)):
        raise ValueError(
            f"inference shard coverage mismatch: {sorted(indices)} != {list(range(shard_count))}"
        )
    comparable_fields = (
        "benchmark",
        "model_label",
        "backend",
        "resolved_backend",
        "batch_size",
        "samples_per_example",
        "global_seed",
        "sampling",
    )
    baseline = results[0]
    for result, path in zip(results, result_paths, strict=True):
        if result.get("status") != SHARD_STATUS:
            raise ValueError(f"incomplete inference shard: {path}")
        if result.get("artifact_mutated") is not False:
            raise ValueError(f"model artifact changed during inference: {path}")
        if result.get("prepared_manifest_sha256") != prepared_manifest_sha256:
            raise ValueError(f"prepared manifest mismatch: {path}")
        if result.get("prepared_inputs_sha256") != prepared_inputs_sha256:
            raise ValueError(f"prepared inputs mismatch: {path}")
        for field in comparable_fields:
            if result.get(field) != baseline.get(field):
                raise ValueError(f"inference shards disagree on {field}: {path}")
    predictions: list[dict[str, Any]] = []
    for result_path in result_paths:
        predictions.extend(read_json_or_jsonl(result_path.parent / "predictions.jsonl"))
    return results, predictions


def _longbench_official_summary(score: dict[str, Any]) -> list[dict[str, Any]]:
    by_slice = {
        (row["slice"], row["value"]): row["official_sample0_accuracy_percent"]
        for row in score["breakdown_rows"]
    }

    def rounded(slice_name: str, value: str) -> float | None:
        accuracy = by_slice.get((slice_name, value))
        return None if accuracy is None else round(accuracy, 1)

    return [
        {
            "Model": score["model_label"],
            "Overall": rounded("overall", "all"),
            "Easy": rounded("difficulty", "easy"),
            "Hard": rounded("difficulty", "hard"),
            "Short": rounded("length", "short"),
            "Medium": rounded("length", "medium"),
            "Long": rounded("length", "long"),
        }
    ]


def score_run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate complete inference artifacts and emit JSON/CSV reports."""

    data_root = args.data_root.resolve()
    inference_root = args.inference_root.resolve()
    output_root = args.output_root.resolve()
    manifest, rows = validate_prepared_dataset(data_root)
    results, predictions = _load_shards(
        inference_root,
        prepared_manifest_sha256=file_sha256(data_root / "manifest.json"),
        prepared_inputs_sha256=file_sha256(data_root / "inputs.jsonl"),
    )
    if args.allow_partial:
        predicted_ids = {str(prediction["example_id"]) for prediction in predictions}
        rows = [row for row in rows if str(row["example_id"]) in predicted_ids]
        if not rows:
            raise ValueError("partial scoring selected no prepared rows")
    score = score_predictions(manifest, rows, predictions)
    score.update(
        {
            "model_label": results[0]["model_label"],
            "backend": results[0]["backend"],
            "resolved_backend": results[0]["resolved_backend"],
            "batch_size": results[0]["batch_size"],
            "global_seed": results[0]["global_seed"],
            "inference_official_protocol_compatible": all(
                result.get("official_protocol_compatible") is True for result in results
            ),
            "formal_complete_dataset": not args.allow_partial,
            "prepared_manifest_sha256": file_sha256(data_root / "manifest.json"),
            "prepared_inputs_sha256": file_sha256(data_root / "inputs.jsonl"),
            "inference_shard_count": len(results),
            "prediction_count": len(predictions),
        }
    )
    formal_example_count = int(manifest["protocol"]["formal_example_count"])
    formal_prepared_data = bool(
        int(manifest["example_count"]) == formal_example_count
        and manifest["protocol"].get("formal_data_validation_passed") is True
    )
    prompt_transport = str(manifest["protocol"].get("prompt_transport", ""))
    upstream_prompt_transport = (
        prompt_transport == "raw"
        if score["benchmark"] == RULER_BENCHMARK
        else prompt_transport == "chat_template"
    )
    score["formal_prepared_data"] = formal_prepared_data
    score["upstream_prompt_transport_compatible"] = upstream_prompt_transport
    score["official_protocol_compatible"] = bool(
        score["official_protocol_compatible"]
        and score["inference_official_protocol_compatible"]
        and score["formal_complete_dataset"]
        and formal_prepared_data
        and upstream_prompt_transport
    )
    assert_finite_scores(score)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "score.json", score)
    if score["benchmark"] == RULER_BENCHMARK:
        write_csv(output_root / "ruler-task-by-length.csv", score["task_rows"])
        write_csv(output_root / "ruler-summary.csv", score["summary_rows"])
    elif score["benchmark"] == LONGBENCH_V2_BENCHMARK:
        write_csv(
            output_root / "longbench-v2-official-summary.csv", _longbench_official_summary(score)
        )
        write_csv(output_root / "longbench-v2-breakdown.csv", score["breakdown_rows"])
        write_jsonl_atomic(output_root / "longbench-v2-scored.jsonl", score["scored_rows"])
    else:
        raise ValueError(f"unsupported benchmark: {score['benchmark']}")
    return score


def main() -> None:
    """Score one complete inference run."""

    result = score_run(parse_args())
    print(f"status={result['status']}")
    print(f"benchmark={result['benchmark']}")
    print(f"official_protocol_compatible={result['official_protocol_compatible']}")


if __name__ == "__main__":
    main()
