#!/usr/bin/env python3
"""Prepare sealed, tokenizer-specific RULER or LongBench v2 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .long_context_protocol import (
    LONGBENCH_V2_BENCHMARK,
    LONGBENCH_V2_EXAMPLE_COUNT,
    LONGBENCH_V2_PROMPT_SHA256,
    RULER_BENCHMARK,
    RULER_DATA_ARCHIVE,
    RULER_DATA_ARCHIVE_SHA256,
    RULER_DATA_ARCHIVE_SIZE_BYTES,
    RULER_DATA_FILES_CONTRACT_SHA256,
    RULER_DATA_GENERATION_CONTRACT,
    RULER_DATASET,
    RULER_DATASET_REVISION,
    RULER_EXAMPLES_PER_TASK,
    RULER_QA_PROMPT_CONTRACT,
    RULER_SEQUENCE_LENGTHS,
    RULER_SOURCE_DATA_KIND,
    RULER_TASKS,
    benchmark_sampling_contract,
    build_prepared_manifest,
    canonical_json_sha256,
    file_sha256,
    prepare_longbench_v2_rows,
    prepare_ruler_rows,
    read_json_or_jsonl,
    ruler_source_path,
    tokenizer_artifact_contract,
    write_json_atomic,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    """Parse the long-context data-preparation command."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    ruler = subparsers.add_parser(RULER_BENCHMARK)
    ruler.add_argument(
        "--olmes-ruler-data-root",
        "--official-ruler-root",
        dest="olmes_ruler_data_root",
        type=Path,
        required=True,
        help=(
            "Pinned extracted allenai/ruler_data cache containing data_100_samples.tgz "
            "and data/ruler/<task>/validation_<length>.jsonl. The deprecated "
            "--official-ruler-root spelling is accepted as an alias."
        ),
    )
    ruler.add_argument(
        "--sequence-lengths",
        default=",".join(str(value) for value in RULER_SEQUENCE_LENGTHS),
        help="Comma-separated RULER lengths; the formal sweep is 4096 through 65536.",
    )
    ruler.add_argument("--tokenizer-model", type=Path, required=True)
    ruler.add_argument("--output-root", type=Path, required=True)
    ruler.add_argument("--prompt-transport", choices=("raw", "chat_template"), default="raw")
    ruler.add_argument(
        "--allow-nonstandard-count", action=argparse.BooleanOptionalAction, default=False
    )

    longbench = subparsers.add_parser(LONGBENCH_V2_BENCHMARK)
    longbench.add_argument("--source-data", type=Path, required=True)
    longbench.add_argument("--tokenizer-model", type=Path, required=True)
    longbench.add_argument("--max-input-tokens", type=int, required=True)
    longbench.add_argument("--output-root", type=Path, required=True)
    longbench.add_argument("--prompt-transport", choices=("raw", "chat_template"), default="raw")
    longbench.add_argument(
        "--allow-nonstandard-count", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def _load_tokenizer(path: Path) -> object:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    resolved = str(path.resolve())
    try:
        return AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=True, local_files_only=True, use_fast=True
        )
    except ValueError as error:
        if "Tokenizer class TokenizersBackend does not exist" not in str(error):
            raise
    return PreTrainedTokenizerFast.from_pretrained(resolved, local_files_only=True)


def _parse_sequence_lengths(raw: str) -> tuple[int, ...]:
    """Parse a unique, increasing list of positive RULER sequence lengths."""

    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"invalid RULER sequence lengths: {raw!r}") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("RULER sequence lengths must be positive")
    if len(values) != len(set(values)) or values != tuple(sorted(values)):
        raise ValueError("RULER sequence lengths must be unique and increasing")
    return values


def _validate_olmes_ruler_data_root(root: Path) -> tuple[Path, dict[str, object]]:
    """Validate the fixed AllenAI RULER archive and its 4K-through-64K files."""

    root = root.resolve()
    archive_path = root / RULER_DATA_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(f"missing cached OLMES RULER archive: {archive_path}")
    if archive_path.stat().st_size != RULER_DATA_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            "OLMES RULER archive size mismatch: "
            f"{archive_path.stat().st_size} != {RULER_DATA_ARCHIVE_SIZE_BYTES}"
        )
    archive_sha256 = file_sha256(archive_path)
    if archive_sha256 != RULER_DATA_ARCHIVE_SHA256:
        raise ValueError(
            f"OLMES RULER archive hash mismatch: {archive_sha256} "
            f"!= {RULER_DATA_ARCHIVE_SHA256}"
        )

    setup_root = root / "data/ruler"
    files_by_length: dict[str, dict[str, dict[str, object]]] = {}
    files_contract: dict[str, str] = {}
    for sequence_length in RULER_SEQUENCE_LENGTHS:
        length_files: dict[str, dict[str, object]] = {}
        for task in RULER_TASKS:
            path = ruler_source_path(setup_root, task, sequence_length)
            rows = read_json_or_jsonl(path)
            if len(rows) != RULER_EXAMPLES_PER_TASK:
                raise ValueError(
                    f"OLMES RULER source count mismatch for {path}: "
                    f"{len(rows)} != {RULER_EXAMPLES_PER_TASK}"
                )
            relative = str(path.relative_to(root))
            sha256 = file_sha256(path)
            files_contract[relative] = sha256
            length_files[task] = {
                "path": str(path),
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256,
            }
        files_by_length[str(sequence_length)] = length_files
    files_contract_sha256 = canonical_json_sha256(files_contract)
    if files_contract_sha256 != RULER_DATA_FILES_CONTRACT_SHA256:
        raise ValueError(
            f"OLMES RULER extracted-file contract mismatch: {files_contract_sha256} "
            f"!= {RULER_DATA_FILES_CONTRACT_SHA256}"
        )
    return setup_root, {
        "kind": RULER_SOURCE_DATA_KIND,
        "dataset": RULER_DATASET,
        "dataset_revision": RULER_DATASET_REVISION,
        "cache_root": str(root),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "files_contract_sha256": files_contract_sha256,
        "files_by_sequence_length": files_by_length,
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    """Prepare and seal the selected benchmark dataset."""

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_model.resolve()
    tokenizer = _load_tokenizer(tokenizer_path)
    tokenizer_contract = tokenizer_artifact_contract(tokenizer_path)

    if args.benchmark == RULER_BENCHMARK:
        sequence_lengths = _parse_sequence_lengths(str(args.sequence_lengths))
        if any(value not in RULER_SEQUENCE_LENGTHS for value in sequence_lengths):
            raise ValueError(
                f"RULER lengths must be selected from {RULER_SEQUENCE_LENGTHS}: "
                f"{sequence_lengths}"
            )
        setup_root, source_data = _validate_olmes_ruler_data_root(
            args.olmes_ruler_data_root.resolve()
        )
        rows = []
        for sequence_length in sequence_lengths:
            length_rows = prepare_ruler_rows(
                setup_root,
                tokenizer,
                sequence_length=sequence_length,
                prompt_transport=str(args.prompt_transport),
                data_generation_contract=RULER_DATA_GENERATION_CONTRACT,
            )
            rows.extend(length_rows)
        expected_count = len(sequence_lengths) * len(RULER_TASKS) * RULER_EXAMPLES_PER_TASK
        task_counts = Counter(
            (int(row["metadata"]["sequence_length"]), str(row["task"])) for row in rows
        )
        expected_task_counts = {
            (sequence_length, task): RULER_EXAMPLES_PER_TASK
            for sequence_length in sequence_lengths
            for task in RULER_TASKS
        }
        formal_data_validation_passed = (
            len(rows) == expected_count and dict(task_counts) == expected_task_counts
        )
        if not args.allow_nonstandard_count and not formal_data_validation_passed:
            raise ValueError(
                "formal RULER requires an official length and exactly 100 rows for "
                "each of 13 tasks at every selected length; "
                f"lengths={sequence_lengths} total={len(rows)}"
            )
        protocol = {
            **benchmark_sampling_contract(RULER_BENCHMARK),
            "data_generation_contract": RULER_DATA_GENERATION_CONTRACT,
            "sequence_lengths": list(sequence_lengths),
            "formal_sequence_lengths": list(RULER_SEQUENCE_LENGTHS),
            "sequence_length_count": len(sequence_lengths),
            "task_count": len(RULER_TASKS),
            "examples_per_task": RULER_EXAMPLES_PER_TASK,
            "prompt_transport": str(args.prompt_transport),
            "qa_prompt_contract": RULER_QA_PROMPT_CONTRACT,
            "formal_example_count": expected_count,
            "formal_data_validation_passed": formal_data_validation_passed,
            "full_sweep_validation_passed": (
                sequence_lengths == RULER_SEQUENCE_LENGTHS and formal_data_validation_passed
            ),
        }
    else:
        source_path = args.source_data.resolve()
        source_rows = read_json_or_jsonl(source_path)
        rows = prepare_longbench_v2_rows(
            source_rows,
            tokenizer,
            max_input_tokens=int(args.max_input_tokens),
            prompt_transport=str(args.prompt_transport),
        )
        difficulties = {str(row["metadata"]["difficulty"]) for row in rows}
        lengths = {str(row["metadata"]["length"]) for row in rows}
        formal_data_validation_passed = (
            len(rows) == LONGBENCH_V2_EXAMPLE_COUNT
            and difficulties == {"easy", "hard"}
            and lengths == {"short", "medium", "long"}
        )
        if not args.allow_nonstandard_count and not formal_data_validation_passed:
            raise ValueError(
                "formal LongBench v2 requires 503 rows with difficulty={easy,hard} "
                "and length={short,medium,long}; "
                f"total={len(rows)} difficulty={sorted(difficulties)} "
                f"length={sorted(lengths)}"
            )
        source_data = {
            "kind": "longbench_v2_local_export",
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        }
        protocol = {
            **benchmark_sampling_contract(LONGBENCH_V2_BENCHMARK),
            "max_input_tokens": int(args.max_input_tokens),
            "head_tail_truncation": True,
            "prompt_transport": str(args.prompt_transport),
            "prompt_sha256": LONGBENCH_V2_PROMPT_SHA256,
            "formal_example_count": LONGBENCH_V2_EXAMPLE_COUNT,
            "formal_data_validation_passed": formal_data_validation_passed,
            "missing_prediction_compensation": False,
        }

    manifest = build_prepared_manifest(
        benchmark=str(args.benchmark),
        rows=rows,
        tokenizer_contract=tokenizer_contract,
        source_data=source_data,
        protocol=protocol,
    )
    write_jsonl_atomic(output_root / "inputs.jsonl", rows)
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    """Prepare one benchmark and print the resulting manifest path."""

    args = parse_args()
    manifest = prepare(args)
    print(f"status={manifest['status']}")
    print(f"benchmark={manifest['benchmark']}")
    print(f"example_count={manifest['example_count']}")
    print(f"manifest={args.output_root.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
