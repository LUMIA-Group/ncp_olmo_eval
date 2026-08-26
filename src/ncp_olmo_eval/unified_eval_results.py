#!/usr/bin/env python3
"""Rescore and materialize outputs for the unified evaluation workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .core_native_answer_eval import OFFICIAL_OLMO_EVAL_COMMIT, score_gsm_answer

GSM8K_EXAMPLE_COUNT = 1319
GSM8K_SCORE_STATUS = "UNIFIED_GSM8K_OFFICIAL_RESCORE_OK"
FINAL_STATUS = "UNIFIED_EVALUATION_RESULT_READY"
_GSM8K_RESULT_STATUSES = frozenset(
    {
        "GSM8K_EVAL_OK",
        "GSM8K_VLLM_GENERATION_OK",
        "GSM8K_LMDEPLOY_GENERATION_OK",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _prediction_files(root: Path) -> list[Path]:
    aggregate = root / "predictions.jsonl"
    if aggregate.is_file():
        return [aggregate]
    files = sorted(root.glob("shard-*/predictions.jsonl"))
    if not files:
        raise FileNotFoundError(f"no GSM8K predictions under {root}")
    return files


def _validate_gsm8k_shards(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("shard-*/result.json"))
    if len(paths) != 8:
        raise ValueError(f"GSM8K requires 8 result shards, found {len(paths)}")
    results = []
    for expected_index, path in enumerate(paths):
        result = _read_json(path)
        if result.get("status") not in _GSM8K_RESULT_STATUSES:
            raise ValueError(f"incomplete GSM8K shard: {path}")
        shard_index = result.get("shard_index", result.get("rank"))
        if int(shard_index if shard_index is not None else -1) != expected_index:
            raise ValueError(f"GSM8K shard identity changed: {path}")
        if int(result.get("dataset_sample_count", -1)) != GSM8K_EXAMPLE_COUNT:
            raise ValueError(f"GSM8K dataset count changed: {path}")
        if result.get("artifact_mutated") is not False:
            raise ValueError(f"model artifact mutation was not ruled out: {path}")
        if (
            "source_model_mutated" in result
            and result["source_model_mutated"] is not False
        ):
            raise ValueError(f"source model mutation was not ruled out: {path}")
        if (
            result.get("model_source") == "dcp"
            and result.get("checkpoint_mutated") is not False
        ):
            raise ValueError(f"checkpoint mutation was not ruled out: {path}")
        results.append(result)
    return results


def _gold_answer(row: dict[str, Any], path: Path, line_number: int) -> str:
    for field in ("gold_answer", "gold", "official_gold"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"prediction has no gold answer: {path}:{line_number}")


def score_gsm8k(inference_root: Path, output_root: Path) -> dict[str, Any]:
    """Official-rescore one complete 8-shard GSM8K run for every backend."""

    inference_root = inference_root.resolve()
    output_root = output_root.resolve()
    shard_results = _validate_gsm8k_shards(inference_root)
    prediction_files = _prediction_files(inference_root)
    seen: set[int] = set()
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    for path in prediction_files:
        for line_number, row in _read_jsonl(path):
            doc_index = int(row["doc_index"])
            if doc_index in seen:
                raise ValueError(f"duplicate GSM8K doc_index={doc_index}")
            sample_index = row.get("sample_index", 0)
            if int(sample_index) != 0:
                raise ValueError(
                    f"GSM8K sample_index must be zero: {path}:{line_number}"
                )
            output = str(row.get("output", row.get("generation", "")))
            result = score_gsm_answer(output, _gold_answer(row, path, line_number))
            is_correct = bool(result["primary_correct"])
            correct += int(is_correct)
            seen.add(doc_index)
            scored_rows.append(
                {
                    "doc_index": doc_index,
                    "correct": is_correct,
                    "normalized_prediction": result["normalized_prediction"],
                    "normalized_gold": result["normalized_gold"],
                }
            )
    if seen != set(range(GSM8K_EXAMPLE_COUNT)):
        missing = sorted(set(range(GSM8K_EXAMPLE_COUNT)) - seen)
        raise ValueError(
            f"GSM8K prediction coverage incomplete: {len(seen)}/{GSM8K_EXAMPLE_COUNT}; "
            f"first_missing={missing[:16]}"
        )
    score = correct / GSM8K_EXAMPLE_COUNT
    if not math.isfinite(score):
        raise ValueError("GSM8K score is not finite")
    payload = {
        "status": GSM8K_SCORE_STATUS,
        "created_at": _utc_now(),
        "inference_root": str(inference_root),
        "inference_result_shards": len(shard_results),
        "prediction_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in prediction_files
        ],
        "example_count": GSM8K_EXAMPLE_COUNT,
        "pass_at_1_correct": correct,
        "pass_at_1": score,
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "score.json", payload)
    with (output_root / "gsm8k-score.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "value", "correct", "example_count"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "pass@1",
                "value": f"{score:.12f}",
                "correct": correct,
                "example_count": GSM8K_EXAMPLE_COUNT,
            }
        )
    with (output_root / "gsm8k-scored.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(scored_rows, key=lambda item: int(item["doc_index"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_root / "_SUCCESS").touch()
    return payload


def finalize_result(
    benchmark: str, scoring_root: Path, output_root: Path
) -> dict[str, Any]:
    """Copy sealed score artifacts into one immutable display directory."""

    scoring_root = scoring_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite final output: {output_root}")
    score_path = scoring_root / "score.json"
    if not score_path.is_file():
        raise FileNotFoundError(f"score.json is missing: {score_path}")
    score = _read_json(score_path)
    expected_files = {
        "gsm8k": ("score.json", "gsm8k-score.csv", "gsm8k-scored.jsonl"),
        "sciq": ("score.json", "sciq-score.csv"),
        "ruler": ("score.json", "ruler-summary.csv", "ruler-task-by-length.csv"),
        "helmet": (
            "score.json",
            "helmet-length-summary.csv",
            "helmet-category-summary.csv",
            "helmet-dataset-summary.csv",
            "helmet-scored.jsonl",
        ),
    }
    if benchmark not in expected_files:
        raise ValueError(f"unsupported direct finalization benchmark: {benchmark}")
    sources = [scoring_root / name for name in expected_files[benchmark]]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"score artifacts are incomplete: {missing}")
    output_root.mkdir(parents=True)
    copied = []
    for source in sources:
        destination = output_root / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "name": source.name,
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    manifest = {
        "status": FINAL_STATUS,
        "created_at": _utc_now(),
        "benchmark": benchmark,
        "scoring_root": str(scoring_root),
        "scoring_score_sha256": _sha256(score_path),
        "score_status": score.get("status"),
        "files": copied,
    }
    _write_json(output_root / "manifest.json", manifest)
    (output_root / "_SUCCESS").touch()
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score-gsm8k")
    score.add_argument("--inference-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument(
        "--benchmark", choices=("gsm8k", "sciq", "ruler", "helmet"), required=True
    )
    final.add_argument("--scoring-root", type=Path, required=True)
    final.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "score-gsm8k":
        result = score_gsm8k(args.inference_root, args.output_root)
    else:
        result = finalize_result(args.benchmark, args.scoring_root, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
