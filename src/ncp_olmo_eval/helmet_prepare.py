#!/usr/bin/env python3
"""Prepare sealed official HELMET inputs for one tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .helmet_protocol import (
    HELMET_LLAMA2_TOKENIZER_MODEL_SHA256,
    HELMET_PINNED_FILE_SHA256,
    HELMET_PROFILES,
    prepare_helmet_rows,
    promote_checkpoint_rows,
)
from .long_context_protocol import (
    HELMET_BENCHMARK,
    HELMET_OFFICIAL_COMMIT,
    HELMET_SEED,
    benchmark_sampling_contract,
    build_prepared_manifest,
    file_sha256,
    tokenizer_artifact_contract,
    write_json_atomic,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--official-llama2-tokenizer", type=Path, required=True)
    parser.add_argument("--hub-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-under",
        type=Path,
        default=None,
        help="reuse an exact sealed HELMET dataset under this directory before rebuilding",
    )
    parser.add_argument(
        "--derived-cache-root",
        type=Path,
        default=None,
        help="shared content-addressed cache for official source-only map/filter transforms",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--helmet-profile", choices=("8k-64k", "short"), default="8k-64k")
    parser.add_argument(
        "--prepare-processes",
        type=int,
        default=8,
        help="cap upstream Dataset.map/filter workers without changing samples or prompts",
    )
    parser.add_argument(
        "--chat-template-policy",
        choices=("force_raw", "official"),
        default="force_raw",
        help=(
            "force_raw preserves the historical base-model contract; official "
            "honors each pinned HELMET task configuration"
        ),
    )
    parser.add_argument(
        "--allow-nonstandard-count", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--keep-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _load_tokenizer(path: Path) -> object:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    resolved = path.resolve()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved), trust_remote_code=True, local_files_only=True, use_fast=True
        )
    except ValueError as error:
        # Transformers 5 exports generic tokenizers as TokenizersBackend.  The
        # older 4.x preparation environment cannot resolve that class name,
        # although its Rust tokenizer.json is fully compatible.  Fall back
        # only for that exact forward-compatibility case.
        if "Tokenizer class TokenizersBackend" not in str(error):
            raise
        config = json.loads((resolved / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(resolved / "tokenizer.json"),
            bos_token=config.get("bos_token"),
            eos_token=config.get("eos_token"),
            pad_token=config.get("pad_token"),
            unk_token=config.get("unk_token"),
            model_max_length=int(config.get("model_max_length", 10**30)),
            clean_up_tokenization_spaces=bool(config.get("clean_up_tokenization_spaces", False)),
        )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("HELMET exact offset truncation requires a fast tokenizer")
    return tokenizer


def _manifest_chat_template_policy(manifest: dict[str, Any]) -> str | None:
    protocol = manifest.get("protocol")
    source_data = manifest.get("source_data")
    if isinstance(protocol, dict) and protocol.get("chat_template_policy"):
        return str(protocol["chat_template_policy"])
    if isinstance(source_data, dict) and source_data.get("chat_template_policy"):
        return str(source_data["chat_template_policy"])
    if (
        isinstance(protocol, dict)
        and protocol.get("prompt_transport") == "raw"
        and protocol.get("base_model_use_chat_template_override") is False
    ):
        return "force_raw"
    return None


def _is_reusable_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    tokenizer_contract_sha256: str,
    profile: str,
    chat_template_policy: str,
) -> bool:
    profile_contract = HELMET_PROFILES[profile]
    protocol = manifest.get("protocol")
    source_data = manifest.get("source_data")
    tokenizer = manifest.get("tokenizer")
    if manifest.get("status") != "LONG_CONTEXT_DATA_READY":
        return False
    if manifest.get("benchmark") != HELMET_BENCHMARK:
        return False
    if int(manifest.get("example_count", -1)) != int(profile_contract["example_count"]):
        return False
    if not isinstance(protocol, dict) or not isinstance(source_data, dict):
        return False
    if protocol.get("formal_data_validation_passed") is not True:
        return False
    if source_data.get("formal_data_validation_passed") is not True:
        return False
    actual_lengths = tuple(sorted(int(value) for value in protocol.get("input_lengths", ())))
    if actual_lengths != tuple(profile_contract["input_lengths"]):
        return False
    if int(protocol.get("seed", -1)) != HELMET_SEED:
        return False
    if _manifest_chat_template_policy(manifest) != chat_template_policy:
        return False
    if not isinstance(tokenizer, dict):
        return False
    if tokenizer.get("contract_sha256") != tokenizer_contract_sha256:
        return False
    checkout = source_data.get("checkout")
    if not isinstance(checkout, dict) or checkout.get("commit") != HELMET_OFFICIAL_COMMIT:
        return False
    checkout_files = checkout.get("files")
    if not isinstance(checkout_files, dict):
        return False
    for relative, expected_sha256 in HELMET_PINNED_FILE_SHA256.items():
        metadata = checkout_files.get(relative)
        if not isinstance(metadata, dict) or metadata.get("sha256") != expected_sha256:
            return False
    llama2_tokenizer = source_data.get("official_llama2_tokenizer")
    if (
        not isinstance(llama2_tokenizer, dict)
        or llama2_tokenizer.get("tokenizer_model_sha256") != HELMET_LLAMA2_TOKENIZER_MODEL_SHA256
    ):
        return False
    hub_source = source_data.get("hub_source_snapshots")
    if (
        not isinstance(hub_source, dict)
        or hub_source.get("validation") != "HELMET_HUB_SOURCE_SNAPSHOTS_OK"
        or not hub_source.get("contract_sha256")
    ):
        return False
    inputs_path = root / "inputs.jsonl"
    return (
        inputs_path.is_file()
        and inputs_path.stat().st_size > 0
        and isinstance(manifest.get("inputs_jsonl_sha256"), str)
        and len(str(manifest["inputs_jsonl_sha256"])) == 64
    )


def find_reusable_helmet_root(
    parent: Path,
    *,
    tokenizer_contract_sha256: str,
    profile: str,
    chat_template_policy: str,
    exclude: Path | None = None,
) -> Path | None:
    """Return the newest exact protocol/tokenizer-compatible sealed dataset."""

    resolved_parent = parent.resolve()
    if not resolved_parent.is_dir():
        return None
    excluded = exclude.resolve() if exclude is not None else None
    candidates: list[Path] = []
    for manifest_path in resolved_parent.glob("*/manifest.json"):
        root = manifest_path.parent.resolve()
        if excluded is not None and root == excluded:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _is_reusable_manifest(
                root,
                manifest,
                tokenizer_contract_sha256=tokenizer_contract_sha256,
                profile=profile,
                chat_template_policy=chat_template_policy,
            ):
                candidates.append(root)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _reuse_prepared_root(
    source_root: Path, output_root: Path, tokenizer_contract: dict[str, Any]
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    source_inputs = (source_root / "inputs.jsonl").resolve()
    output_inputs = output_root / "inputs.jsonl"
    try:
        os.link(source_inputs, output_inputs)
        link_type = "hardlink"
    except OSError:
        output_inputs.symlink_to(source_inputs)
        link_type = "symlink"
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["tokenizer"] = tokenizer_contract
    manifest["prepared_reuse"] = {
        "source_root": str(source_root.resolve()),
        "link_type": link_type,
        "inputs_jsonl_sha256": str(manifest["inputs_jsonl_sha256"]),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def prepare(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.resolve()
    tokenizer_path = args.tokenizer_model.resolve()
    tokenizer_contract = tokenizer_artifact_contract(tokenizer_path)
    if (output_root / "manifest.json").is_file():
        existing_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
        if _is_reusable_manifest(
            output_root,
            existing_manifest,
            tokenizer_contract_sha256=str(tokenizer_contract["contract_sha256"]),
            profile=str(args.helmet_profile),
            chat_template_policy=str(args.chat_template_policy),
        ):
            return existing_manifest
        raise FileExistsError(f"output root has an incompatible manifest: {output_root}")
    if (
        args.reuse_under is not None
        and int(args.limit_per_dataset) == 0
        and not bool(args.allow_nonstandard_count)
    ):
        reusable_root = find_reusable_helmet_root(
            args.reuse_under,
            tokenizer_contract_sha256=str(tokenizer_contract["contract_sha256"]),
            profile=str(args.helmet_profile),
            chat_template_policy=str(args.chat_template_policy),
            exclude=output_root,
        )
        if reusable_root is not None:
            if output_root.exists():
                if any(output_root.iterdir()):
                    raise FileExistsError(f"output root is not empty: {output_root}")
                output_root.rmdir()
            manifest = _reuse_prepared_root(reusable_root, output_root, tokenizer_contract)
            print(f"helmet_prepare_reused_from={reusable_root}", flush=True)
            return manifest
    allowed_resume_entries = {".helmet_prepare_shards", "inputs.jsonl"}
    if output_root.exists():
        unexpected = {
            path.name for path in output_root.iterdir() if path.name not in allowed_resume_entries
        }
        if unexpected:
            raise FileExistsError(
                f"output root contains non-resumable entries: {output_root}: {sorted(unexpected)}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(tokenizer_path)
    official_llama2_tokenizer_path = args.official_llama2_tokenizer.resolve()
    official_llama2_tokenizer = _load_tokenizer(official_llama2_tokenizer_path)
    rows, source_contract = prepare_helmet_rows(
        args.official_root.resolve(),
        tokenizer,
        official_llama2_tokenizer,
        official_llama2_tokenizer_path,
        args.hub_source_root.resolve(),
        limit_per_dataset=int(args.limit_per_dataset),
        prepare_processes=int(args.prepare_processes),
        profile=str(args.helmet_profile),
        chat_template_policy=str(args.chat_template_policy),
        derived_cache_root=(
            args.derived_cache_root.resolve() if args.derived_cache_root is not None else None
        ),
        checkpoint_root=output_root / ".helmet_prepare_shards",
        tokenizer_contract_sha256=str(tokenizer_contract["contract_sha256"]),
    )
    if not args.allow_nonstandard_count and not source_contract["formal_data_validation_passed"]:
        raise ValueError(
            f"formal HELMET profile={args.helmet_profile} requires "
            f"{source_contract['formal_example_count']} examples with exact "
            f"category counts; observed={len(rows)} {source_contract['category_counts']}"
        )
    sampling_contract = benchmark_sampling_contract(HELMET_BENCHMARK)
    sampling_contract["max_new_tokens_by_task"] = dict(source_contract["max_new_tokens_by_task"])
    sampling_contract["max_new_tokens"] = max(source_contract["max_new_tokens_by_task"].values())
    sampling_contract["stop_by_task"] = {
        task: ["\n", "\n\n"] for task in source_contract["newline_stop_tasks"]
    }
    protocol = {
        **sampling_contract,
        "helmet_profile": str(args.helmet_profile),
        "input_lengths": list(source_contract["input_lengths"]),
        "formal_example_count": int(source_contract["formal_example_count"]),
        "formal_data_validation_passed": bool(source_contract["formal_data_validation_passed"]),
        "prompt_transport": (
            "mixed_official_task_config" if args.chat_template_policy == "official" else "raw"
        ),
        "chat_template_policy": str(args.chat_template_policy),
        "base_model_use_chat_template_override": (
            False if args.chat_template_policy == "force_raw" else None
        ),
        "seed": source_contract["seed"],
        "category_counts": source_contract["category_counts"],
        "formal_category_counts": source_contract["formal_category_counts"],
        "judge_dependent_primary_metrics": [
            "narrativeqa gpt-4-score",
            "infbench_sum gpt-4-f1",
            "multi_lexsum gpt-4-f1",
        ],
    }
    manifest = build_prepared_manifest(
        benchmark=HELMET_BENCHMARK,
        rows=rows,
        tokenizer_contract=tokenizer_contract,
        source_data=source_contract,
        protocol=protocol,
    )
    if not promote_checkpoint_rows(
        output_root / ".helmet_prepare_shards",
        output_root / "inputs.jsonl",
        expected_example_count=len(rows),
    ):
        write_jsonl_atomic(output_root / "inputs.jsonl", rows)
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)
    if not bool(args.keep_checkpoints):
        shutil.rmtree(output_root / ".helmet_prepare_shards", ignore_errors=True)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare(args)
    print(f"status={manifest['status']}")
    print(f"benchmark={manifest['benchmark']}")
    print(f"example_count={manifest['example_count']}")
    print(f"manifest={args.output_root.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
