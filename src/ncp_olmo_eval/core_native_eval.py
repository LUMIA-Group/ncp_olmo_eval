#!/usr/bin/env python3
"""Evaluate pre-rendered OLMo Core requests with Hugging Face models.

The input contract is a profile from the unified OLMo Core export. Few-shot
examples are already rendered into every ``input`` string, so this evaluator
never adds demonstrations of its own. The same entry point supports custom
ConceptLM HF exports and standard Transformers causal language models.

Every model result is written before aggregation.  Loglikelihood tasks retain
per-candidate token IDs and token log-probabilities; generation tasks retain the
full completion and generated token IDs for every sample. Metrics can therefore
be recomputed without another model run when labels or graders change. LBPP
code is never executed in the GPU job; its saved samples require an external,
isolated code-execution grader.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from . import benchmark
from .core_native_answer_eval import (
    OFFICIAL_OLMO_EVAL_COMMIT,
    score_bbh_answer,
    score_deepmind_answer,
    score_gsm_answer,
    score_math_answer,
)
from .inference import SamplingParams
from .lmdeploy_inference import (
    LMDEPLOY_BACKEND,
    LMDeployInferencer,
    add_lmdeploy_args,
    lmdeploy_engine_policy,
    validate_lmdeploy_args,
)
from .native_megatron_inference import NATIVE_MEGATRON_BACKEND, build_native_megatron_inferencer
from .native_vllm_inference import (
    NATIVE_VLLM_BACKEND,
    NATIVE_VLLM_OVERLAY_MANIFEST,
    NativeVLLMInferencer,
    add_native_vllm_args,
    native_vllm_max_model_len,
    native_vllm_scheduler_queue_size,
    native_vllm_speculative_kwargs,
    native_vllm_speculative_manifest,
    prepare_native_vllm_model,
    validate_native_vllm_args,
)
from .transformers_inference import TransformersInferencer

EVALUATION_SEED = 42
LOG_2_OF_E = 1.44269504089
GREEDY_EXACT_CONTINUATION_SCORING_CONTRACT = (
    "greedy exact continuation accuracy"
)
SUPPORTED_METRICS = {
    "acc",
    "acc_raw",
    "acc_per_char",
    "acc_per_token",
    "bits_per_byte",
    "bits_per_byte_corr",
    "execution_accuracy",
    "exact_match",
    "logits_per_byte",
    "pass@1",
}
ACCURACY_METRICS = {"acc", "acc_raw", "acc_per_char", "acc_per_token"}
GENERATION_METRICS = {"exact_match", "execution_accuracy", "pass@1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default="core88",
        help=(
            "Manifest section to evaluate (default: core88; core79 and "
            "the immutable core_native/Core74 profile remain available)."
        ),
    )
    parser.add_argument(
        "--hf-model-path",
        default="",
        help="HF artifact path; unused when --hf-backend=native_megatron.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help="Parent of iter_* DCP directories for native Megatron inference.",
    )
    parser.add_argument("--ckpt-step", type=int, default=0)
    parser.add_argument("--train-wandb-config", default="")
    parser.add_argument(
        "--model-identity-path",
        default="",
        help=(
            "Canonical model artifact identity recorded in run manifests. "
            "Use the unchanged source artifact when --hf-model-path is a "
            "non-destructive compatibility overlay."
        ),
    )
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument(
        "--hf-backend",
        choices=(
            "auto",
            "from_pretrained",
            "transformers",
            NATIVE_VLLM_BACKEND,
            NATIVE_MEGATRON_BACKEND,
            LMDEPLOY_BACKEND,
        ),
        default="auto",
        help=(
            "auto selects the ConceptLM remote-code backend for converted "
            "ConceptLM artifacts and standard Transformers otherwise; "
            "native_vllm selects the parity-gated optimized vLLM plugin"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=1,
        help=(
            "Scoring batch size. Defaults to one because custom HF runtimes "
            "must pass an explicit batch-parity gate before batched scores "
            "can be treated as accuracy-equivalent."
        ),
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help=(
            "Generation batch size. Defaults to one to keep autoregressive "
            "Core88 predictions independent of batch composition."
        ),
    )
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument(
        "--task-orders",
        default="",
        help="Optional comma-separated task orders. Empty means the full profile.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Smoke-only cap applied before job/rank partitioning.",
    )
    parser.add_argument(
        "--generation-samples-cap",
        type=int,
        default=1,
        help=(
            "Cap samples per generation example. Defaults to one for fast model "
            "evaluation; 0 restores every task's declared sampling contract."
        ),
    )
    parser.add_argument(
        "--max-gen-tokens-cap",
        type=int,
        default=0,
        help="Smoke-only generation length cap; 0 preserves the task contract.",
    )
    parser.add_argument("--hf-compile-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hf-align-dcp-runtime-config", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify-data-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_native_vllm_args(parser)
    add_lmdeploy_args(parser)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_manifest_paths(args: argparse.Namespace) -> tuple[str, str]:
    if getattr(args, "hf_backend", "") == NATIVE_MEGATRON_BACKEND:
        checkpoint_root = Path(args.checkpoint_root).resolve()
        return str(checkpoint_root), str(checkpoint_root)
    runtime_path = Path(args.hf_model_path).resolve()
    identity_path = Path(args.model_identity_path or args.hf_model_path).resolve()
    if identity_path == runtime_path:
        return str(identity_path), str(runtime_path)

    hf_manifest_path = runtime_path / "from_pretrained_adapter_manifest.json"
    native_manifest_path = runtime_path / NATIVE_VLLM_OVERLAY_MANIFEST
    if native_manifest_path.is_file():
        manifest_path = native_manifest_path
        expected_status = "CONCEPTLM_NATIVE_VLLM_MODEL_READY"
    elif hf_manifest_path.is_file():
        manifest_path = hf_manifest_path
        expected_status = "CONCEPTLM_HF_REMOTE_CODE_READY"
    else:
        raise RuntimeError(
            "--model-identity-path differs from --hf-model-path, but the "
            "runtime model has no supported overlay manifest: "
            f"{hf_manifest_path} or {native_manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != expected_status
        or manifest.get("weight_files_mutated") is not False
    ):
        raise RuntimeError(f"invalid compatibility-overlay manifest: {manifest_path}")
    if Path(str(manifest.get("source_model", ""))).resolve() != identity_path:
        raise RuntimeError(
            "compatibility-overlay source does not match --model-identity-path: "
            f"{manifest.get('source_model')!r} != {str(identity_path)!r}"
        )
    if Path(str(manifest.get("destination_model", ""))).resolve() != runtime_path:
        raise RuntimeError(
            "compatibility-overlay destination does not match --hf-model-path: "
            f"{manifest.get('destination_model')!r} != {str(runtime_path)!r}"
        )

    manifest_mode = manifest.get("mode")
    if manifest_mode == "overlay_tokenizer_compatibility":
        source_config = identity_path / "config.json"
        runtime_config = runtime_path / "config.json"
        declared_config = Path(str(manifest.get("runtime_config", ""))).resolve()
        declared_sha256 = str(manifest.get("runtime_config_sha256", ""))
        if (
            manifest_path != native_manifest_path
            or manifest.get("tokenizer_config_compatibility_rewrite") is not True
            or declared_config != source_config.resolve()
            or not source_config.is_file()
            or not runtime_config.is_file()
            or not declared_sha256
            or _file_sha256(source_config) != declared_sha256
            or _file_sha256(runtime_config) != declared_sha256
        ):
            raise RuntimeError(f"invalid compatibility-overlay manifest: {manifest_path}")
    elif manifest_mode != "overlay":
        raise RuntimeError(f"invalid compatibility-overlay manifest: {manifest_path}")

    for filename, expected in manifest.get("weight_files", {}).items():
        runtime_weight = runtime_path / filename
        source_weight = identity_path / filename
        if not runtime_weight.is_symlink() or runtime_weight.resolve() != source_weight.resolve():
            raise RuntimeError(
                f"compatibility-overlay weight is not linked to its source: {runtime_weight}"
            )
        stat = runtime_weight.stat()
        if int(stat.st_size) != int(expected["size"]) or int(stat.st_mtime_ns) != int(
            expected["mtime_ns"]
        ):
            raise RuntimeError(
                f"compatibility-overlay source weight metadata changed: {source_weight}"
            )
    for filename, expected_sha256 in manifest.get("remote_code_sha256", {}).items():
        actual_sha256 = _file_sha256(runtime_path / filename)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"compatibility-overlay remote code hash mismatch: {runtime_path / filename}"
            )
    return str(identity_path), str(runtime_path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _distributed_args() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} is outside WORLD_SIZE={world_size}")
    return rank, local_rank, world_size


def _initialize_distributed_process_group(rank: int, world_size: int) -> bool:
    """Make the final aggregation barrier real for multi-process launches."""

    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise RuntimeError("multi-process Core evaluation requires torch.distributed")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo", init_method="env://", rank=rank, world_size=world_size
        )
    return True


def _rank_root(output_root: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return output_root
    return output_root / f"shard-{rank:02d}"


def _isolate_compile_caches(rank: int, world_size: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    if world_size <= 1:
        return paths
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        root = os.environ.get(name)
        if not root:
            continue
        rank_path = Path(root) / f"rank-{rank:02d}"
        rank_path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(rank_path)
        paths[name] = str(rank_path)
    return paths


def _primary_metric(value: Any) -> str:
    """Resolve OLMo's ordered metric list to this evaluator's primary metric."""

    if isinstance(value, str):
        metric_names = [value]
    elif isinstance(value, dict):
        metric_names = [str(value.get("metric", ""))]
    elif isinstance(value, list):
        metric_names = [
            str(item.get("metric", "")) if isinstance(item, dict) else str(item)
            for item in value
        ]
    else:
        raise ValueError(f"invalid metric contract: {value!r}")
    for metric in metric_names:
        if metric in SUPPORTED_METRICS:
            return metric
    raise ValueError(f"no supported primary metric in contract: {metric_names}")


def _load_manifest(
    data_root: Path, profile: str, task_orders: set[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = data_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    section = summary.get(profile)
    if not isinstance(section, dict) or section.get("failures"):
        raise RuntimeError(f"{profile} export is missing or reports failures")
    tasks = section.get("tasks")
    declared_task_count = int(section.get("task_count", len(tasks or [])))
    if not isinstance(tasks, list) or len(tasks) != declared_task_count:
        raise RuntimeError(
            f"expected {declared_task_count} {profile} tasks, got {len(tasks or [])}"
        )
    expected_orders = list(range(1, declared_task_count + 1))
    actual_orders = [int(task.get("task_order", -1)) for task in tasks]
    if actual_orders != expected_orders:
        if not task_orders or sorted(actual_orders) != expected_orders:
            raise RuntimeError(
                f"{profile} task order is not the fixed 1..{declared_task_count} order"
            )
    selected = sorted(
        (
            task
            for task in tasks
            if not task_orders or int(task["task_order"]) in task_orders
        ),
        key=lambda task: int(task["task_order"]),
    )
    if task_orders != {int(task["task_order"]) for task in selected} and task_orders:
        missing = sorted(task_orders - {int(task["task_order"]) for task in selected})
        raise ValueError(f"unknown task orders: {missing}")
    normalized: list[dict[str, Any]] = []
    for source_task in selected:
        task = dict(source_task)
        relative_path = str(task.get("file", ""))
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"task file must remain inside data-root: {task}")
        if task.get("metric") is None or task.get("request_type") is None:
            with gzip.open(data_root / relative_path, "rt", encoding="utf-8") as handle:
                first_record = json.loads(next(handle))
            if task.get("metric") is None:
                task["metric"] = first_record.get("metric")
            if task.get("request_type") is None:
                task["request_type"] = first_record.get("request_type")
        metric_contract = task.get("metric")
        metric = _primary_metric(metric_contract)
        if not isinstance(metric_contract, str):
            task["metric_definitions"] = metric_contract
        task["metric"] = metric
        normalized.append(task)
    return summary, normalized


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metric_contract = row.get("metric")
            if not isinstance(metric_contract, str):
                row["metric_definitions"] = metric_contract
            row["metric"] = _primary_metric(metric_contract)
            rows.append(row)
    return rows


def _parse_task_orders(value: str, task_count: int) -> set[int]:
    if not value.strip():
        return set()
    orders = {int(part.strip()) for part in value.split(",") if part.strip()}
    if any(order < 1 or order > task_count for order in orders):
        raise ValueError(f"task orders must be in [1, {task_count}], got {sorted(orders)}")
    return orders


def _task_is_generation(task: dict[str, Any]) -> bool:
    request_type = str(task.get("request_type") or "")
    if request_type.startswith("generate_until") and request_type != (
        "generate_until_and_loglikelihood"
    ):
        return True
    if request_type == "loglikelihood":
        return False
    return str(task.get("metric")) in GENERATION_METRICS


def _selected_row(
    row_index: int, partition_index: int, partition_count: int, rank: int, world_size: int
) -> bool:
    total_shards = partition_count * world_size
    shard_index = partition_index * world_size + rank
    return row_index % total_shards == shard_index


def _selected_count(
    row_count: int, limit_per_task: int, partition_index: int, partition_count: int, world_size: int
) -> int:
    capped = min(row_count, limit_per_task) if limit_per_task > 0 else row_count
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    total_shards = partition_count * world_size
    first_shard = partition_index * world_size
    full_cycles, remaining_rows = divmod(capped, total_shards)
    remaining_selected = max(0, min(remaining_rows, first_shard + world_size) - first_shard)
    return full_cycles * world_size + remaining_selected


def _completed_example_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid resume JSON at {path}:{line_number}") from error
            completed.add(_stable_id(record["example_id"]))
    return completed


def _completed_generation_samples(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid resume JSON at {path}:{line_number}") from error
            completed.add((_stable_id(record["example_id"]), int(record.get("sample_index", 0))))
    return completed


def _model_args(args: argparse.Namespace, hf_backend: str) -> argparse.Namespace:
    native_megatron = hf_backend == NATIVE_MEGATRON_BACKEND
    return argparse.Namespace(
        model_source=("dcp" if native_megatron else "hf"),
        checkpoint_root=(args.checkpoint_root if native_megatron else ""),
        ckpt_step=(args.ckpt_step if native_megatron else 0),
        tokenizer_model=args.tokenizer_model or args.hf_model_path,
        train_wandb_config=(args.train_wandb_config if native_megatron else ""),
        hf_model_path=args.hf_model_path,
        hf_backend=hf_backend,
        hf_compile_routes=bool(args.hf_compile_routes),
        hf_align_dcp_runtime_config=bool(args.hf_align_dcp_runtime_config),
        allow_context_extension=bool(getattr(args, "allow_context_extension", False)),
        flash_decode=False,
        cuda_graph=False,
        seq_length=args.seq_length,
        batch_size=_max_inference_batch_size(args),
        max_new_tokens=1024,
        hf_model_parallel_size=int(
            getattr(args, "hf_model_parallel_size", 1)
        ),
        dcp_checkpoint_native_runtime=native_megatron,
    )


def _max_inference_batch_size(args: argparse.Namespace) -> int:
    """Keep the engine capacity aligned with both evaluator batch routes."""

    return max(int(args.score_batch_size), int(args.generation_batch_size))


def _resolve_hf_backend(args: argparse.Namespace) -> str:
    """Resolve the model-neutral CLI backend without loading any weights."""

    if args.hf_backend != "auto":
        return str(args.hf_backend)
    if not args.hf_model_path:
        raise ValueError("--hf-model-path is required unless native_megatron is selected")
    config_path = Path(args.hf_model_path).resolve() / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = config.get("auto_map") or {}
    architecture = " ".join(str(value) for value in config.get("architectures", []))
    if (
        config.get("model_type") == "conceptlm_v22_vq"
        or "ConceptLMV22VQ" in architecture
        or "configuration_conceptlm_v22_vq" in str(auto_map)
    ):
        return "from_pretrained"
    return "transformers"


def _bos_token_ids(tokenizer: Any) -> list[int]:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    return [] if bos_id is None else [int(bos_id)]


def _encode_context_and_continuation(
    tokenizer: Any,
    context: str,
    continuation: str,
) -> tuple[list[int], list[int]]:
    """Match OLMo-Eval/lm-eval continuation boundaries and BOS handling."""

    if context == "":
        return (
            _bos_token_ids(tokenizer),
            list(tokenizer.encode(continuation, add_special_tokens=False)),
        )

    trailing_space_count = len(context) - len(context.rstrip())
    if trailing_space_count > 0:
        continuation = context[-trailing_space_count:] + continuation
        context = context[:-trailing_space_count]

    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    bos_ids = _bos_token_ids(tokenizer) if add_bos else []
    whole_ids = list(
        tokenizer.encode(context + continuation, add_special_tokens=False)
    )
    context_ids = list(tokenizer.encode(context, add_special_tokens=False))
    continuation_ids = whole_ids[len(context_ids) :]
    return bos_ids + context_ids, continuation_ids


def _prepare_scoring_requests(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    seq_length: int,
    *,
    enable_fast_mc: bool = True,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any] | None]]]:
    requests: list[dict[str, Any]] = []
    candidate_results: list[list[dict[str, Any] | None]] = []
    for row_index, row in enumerate(rows):
        choices = list(row.get("choices") or [row.get("output", "")])
        encoded = [
            _encode_context_and_continuation(
                tokenizer,
                str(row["input"]),
                str(choice),
            )
            for choice in choices
        ]
        context_ids = encoded[0][0]
        if any(candidate_context != context_ids for candidate_context, _ in encoded):
            raise RuntimeError(
                "continuation-dependent context tokenization changed the context IDs: "
                f"task={row['task']} example={row['example_id']}"
            )
        continuations = [continuation_ids for _, continuation_ids in encoded]
        candidate_results.append([None] * len(choices))
        use_fast_mc = (
            enable_fast_mc
            and len(choices) > 1
            and all(len(token_ids) == 1 for token_ids in continuations)
            and row.get("metric") in {"acc", "acc_raw"}
        )
        if use_fast_mc:
            query_ids = context_ids[-seq_length:]
            position = len(query_ids) - 1
            requests.append(
                {
                    "row_index": row_index,
                    "query_ids": query_ids,
                    "targets": [
                        {
                            "candidate_index": candidate_index,
                            "positions": [position],
                            "token_ids": token_ids,
                        }
                        for candidate_index, token_ids in enumerate(continuations)
                    ],
                    "fast_mc": True,
                }
            )
            continue
        for candidate_index, token_ids in enumerate(continuations):
            query_ids = context_ids + token_ids[:-1]
            query_ids = query_ids[-seq_length:]
            start = len(query_ids) - len(token_ids)
            if start < 0:
                raise ValueError(
                    "continuation alone exceeds seq_length: "
                    f"task={row['task']} example={row['example_id']} "
                    f"continuation_tokens={len(token_ids)} seq_length={seq_length}"
                )
            requests.append(
                {
                    "row_index": row_index,
                    "query_ids": query_ids,
                    "targets": [
                        {
                            "candidate_index": candidate_index,
                            "positions": list(range(start, start + len(token_ids))),
                            "token_ids": token_ids,
                        }
                    ],
                    "fast_mc": False,
                }
            )
    return requests, candidate_results


def _batches(
    requests: list[dict[str, Any]], batch_size: int, pad_multiple: int
) -> Iterable[list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        length = len(request["query_ids"])
        bucket = math.ceil(length / pad_multiple)
        buckets[bucket].append(request)
    for bucket in sorted(buckets):
        values = buckets[bucket]
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def _effective_scoring_pad_multiple(batch_size: int, pad_multiple: int) -> int:
    """Avoid unnecessary right padding for the batch-one correctness path."""

    if batch_size <= 0:
        raise ValueError("score batch size must be positive")
    if pad_multiple <= 0:
        raise ValueError("pad multiple must be positive")
    return 1 if batch_size == 1 else pad_multiple


@torch.inference_mode()
def _score_requests(
    model: Any,
    tokenizer: Any,
    requests: list[dict[str, Any]],
    candidate_results: list[list[dict[str, Any] | None]],
    batch_size: int,
    pad_multiple: int,
    seq_length: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a pad or EOS token")
    forward_calls = 0
    padded_token_slots = 0
    query_token_count = 0
    continuation_token_count = 0
    effective_pad_multiple = _effective_scoring_pad_multiple(
        batch_size,
        pad_multiple,
    )
    for batch in _batches(requests, batch_size, effective_pad_multiple):
        max_query = max(len(request["query_ids"]) for request in batch)
        padded_length = min(
            seq_length,
            math.ceil(max_query / effective_pad_multiple)
            * effective_pad_multiple,
        )
        input_ids = torch.full(
            (len(batch), padded_length), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for batch_index, request in enumerate(batch):
            query_ids = request["query_ids"]
            query_length = len(query_ids)
            input_ids[batch_index, :query_length] = torch.tensor(
                query_ids, dtype=torch.long, device=device
            )
            attention_mask[batch_index, :query_length] = 1
            query_token_count += query_length
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
        ).logits
        if logits.ndim != 3:
            raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")

        logprob_tensors: list[torch.Tensor] = []
        greedy_tensors: list[torch.Tensor] = []
        target_metadata: list[tuple[int, int, list[int]]] = []
        for batch_index, request in enumerate(batch):
            for target in request["targets"]:
                positions = torch.tensor(target["positions"], dtype=torch.long, device=device)
                token_ids = torch.tensor(target["token_ids"], dtype=torch.long, device=device)
                selected_logits = logits[batch_index].index_select(0, positions)
                token_logprobs = -F.cross_entropy(
                    selected_logits.float(), token_ids, reduction="none"
                )
                logprob_tensors.append(token_logprobs)
                greedy_tensors.append(selected_logits.argmax(dim=-1).eq(token_ids))
                target_metadata.append(
                    (
                        int(request["row_index"]),
                        int(target["candidate_index"]),
                        list(target["token_ids"]),
                    )
                )
                continuation_token_count += len(target["token_ids"])
        flat_logprobs = (
            torch.cat(logprob_tensors).detach().cpu().tolist() if logprob_tensors else []
        )
        flat_greedy = torch.cat(greedy_tensors).detach().cpu().tolist() if greedy_tensors else []
        cursor = 0
        for (row_index, candidate_index, token_ids), tensor in zip(
            target_metadata, logprob_tensors, strict=True
        ):
            length = tensor.numel()
            token_logprobs = [float(value) for value in flat_logprobs[cursor : cursor + length]]
            greedy_flags = [bool(value) for value in flat_greedy[cursor : cursor + length]]
            cursor += length
            candidate_results[row_index][candidate_index] = {
                "token_ids": token_ids,
                "token_logprobs": token_logprobs,
                "sum_logprob": float(sum(token_logprobs)),
                "num_tokens": len(token_ids),
                "is_greedy": all(greedy_flags),
            }
        del logits
        forward_calls += 1
        padded_token_slots += len(batch) * padded_length
    return {
        "forward_calls": forward_calls,
        "query_token_count": query_token_count,
        "padded_token_slots": padded_token_slots,
        "continuation_token_count": continuation_token_count,
    }


def _prediction_indices(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
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


def _finish_scoring_record(
    row: dict[str, Any],
    raw_candidates: list[dict[str, Any] | None],
    data_file: Path,
    rank: int,
    partition_index: int,
) -> dict[str, Any]:
    choices = list(row.get("choices") or [row.get("output", "")])
    candidates: list[dict[str, Any]] = []
    for choice, raw in zip(choices, raw_candidates, strict=True):
        if raw is None:
            raise RuntimeError(f"candidate score missing for {row['task']}:{row['example_id']}")
        text = str(choice)
        num_chars = max(1, len(text))
        num_bytes = max(1, len(text.encode("utf-8")))
        num_tokens = max(1, int(raw["num_tokens"]))
        sum_logprob = float(raw["sum_logprob"])
        candidates.append(
            {
                "text": text,
                **raw,
                "num_chars": num_chars,
                "num_bytes": num_bytes,
                "logprob_per_token": sum_logprob / num_tokens,
                "logprob_per_char": sum_logprob / num_chars,
                "bits_per_byte": -LOG_2_OF_E * sum_logprob / num_bytes,
            }
        )
    predicted = _prediction_indices(candidates)
    target_index = row.get("target_index")
    effective_target_index = (
        int(target_index)
        if isinstance(target_index, int) and 0 <= target_index < len(candidates)
        else 0
    )
    metric = str(row["metric"])
    if (
        row.get("scoring_contract")
        == GREEDY_EXACT_CONTINUATION_SCORING_CONTRACT
    ):
        primary_prediction_name = "greedy_exact_continuation"
        primary_correct = bool(candidates[effective_target_index]["is_greedy"])
    else:
        primary_prediction_name = {
            "acc": "raw",
            "acc_raw": "raw",
            "acc_per_token": "per_token",
            "acc_per_char": "per_char",
        }.get(metric)
        primary_correct = (
            predicted[primary_prediction_name] == effective_target_index
            if primary_prediction_name is not None
            else None
        )
    target_bits_per_byte = candidates[effective_target_index]["bits_per_byte"]
    return {
        "schema_version": "hf-core-native-prediction-v2",
        "task_order": int(row["task_order"]),
        "task": row["task"],
        "example_id": row["example_id"],
        "native_id": row.get("native_id"),
        "input_sha256": _text_sha256(str(row["input"])),
        "data_file": str(data_file),
        "metric": metric,
        "request_type": row["request_type"],
        "num_fewshot": row["num_fewshot"],
        "target_index": row.get("target_index"),
        "effective_target_index": effective_target_index,
        "output": row.get("output"),
        "subset": row.get("subset"),
        "source_task": row.get("source_task"),
        "candidates": candidates,
        "predicted_index": predicted,
        "primary_prediction_normalization": primary_prediction_name,
        "primary_correct": primary_correct,
        "target_bits_per_byte": target_bits_per_byte,
        "distributed_rank": rank,
        "job_partition_index": partition_index,
    }


def _generation_score(
    row: dict[str, Any],
    completion: str,
    *,
    require_official_math_runtime: bool = False,
) -> dict[str, Any]:
    """Score text tasks while deferring untrusted code to an external sandbox."""

    if row.get("execution_contract"):
        return {
            "normalized_prediction": None,
            "normalized_gold": None,
            "primary_correct": None,
            "score_status": "PENDING_EXTERNAL_CODE_EXECUTION",
        }
    task_order = int(row.get("task_order") or -1)
    task_name = str(row.get("task") or "")
    if task_order == 75 or task_name.startswith(("gsm8k", "gsm_symbolic")):
        return score_gsm_answer(completion, str(row["output"]))
    if (
        26 <= task_order <= 32
        or task_order == 80
        or task_name.startswith("minerva_math")
        or task_name.startswith("math_500")
    ):
        return score_math_answer(
            completion,
            str(row["output"]),
            require_official_runtime=require_official_math_runtime,
        )
    if task_order == 77 or task_name.startswith("deepmind_mathematics"):
        return score_deepmind_answer(completion, str(row["output"]))
    if task_order == 78 or task_name.startswith("bbh"):
        return score_bbh_answer(completion, str(row["output"]))
    raise RuntimeError(
        "generation task lacks a pinned official scorer contract: "
        f"task_order={task_order} task={task_name!r}"
    )


def _task_sample_count(task: dict[str, Any]) -> int:
    return int(task.get("evaluation_sample_count", task.get("sample_count") or 1))


def _apply_generation_samples_cap(tasks: list[dict[str, Any]], cap: int) -> None:
    """Apply an evaluation-only sample cap without mutating source metadata."""

    if cap <= 0:
        return
    for task in tasks:
        if _task_is_generation(task):
            task["evaluation_sample_count"] = min(int(task.get("sample_count") or 1), cap)


def _generation_contract(row: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(row.get("generation_kwargs") or {})
    source_path = Path(str(row["source"]))
    config_path = source_path.parent / "config.json"
    config_sha256 = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task_kwargs = config.get("task_config", {}).get("generation_kwargs") or {}
        kwargs = {**task_kwargs, **kwargs}
        config_sha256 = _file_sha256(config_path)
    stop_strings = kwargs.pop("stop_sequences", kwargs.pop("until", []))
    declared_num_samples = kwargs.pop("num_samples", None)
    if declared_num_samples is not None:
        declared_num_samples = int(declared_num_samples)
        row_sample_count = int(row.get("sample_count") or 1)
        if declared_num_samples != row_sample_count:
            raise ValueError(
                "generation num_samples does not match row sample_count: "
                f"{declared_num_samples} != {row_sample_count}"
            )
    return {
        "max_gen_toks": int(kwargs.pop("max_gen_toks", 1024)),
        "temperature": float(kwargs.pop("temperature", 0.0)),
        "top_p": float(kwargs.pop("top_p", 1.0)),
        "do_sample": bool(kwargs.pop("do_sample", False)),
        "declared_num_samples": declared_num_samples,
        "stop_strings": [str(value) for value in stop_strings],
        "remaining_generation_kwargs": kwargs,
        "source_config": str(config_path) if config_path.is_file() else None,
        "source_config_sha256": config_sha256,
    }


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _write_progress(
    path: Path,
    *,
    task: dict[str, Any],
    completed: int,
    total: int,
    task_started: float,
    status: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "status": status,
            "task_order": int(task["task_order"]),
            "task": task["task"],
            "completed": completed,
            "total": total,
            "elapsed_seconds": time.perf_counter() - task_started,
            "updated_at": _utc_now(),
        },
    )


def _score_work_item(
    *,
    model: Any | None,
    tokenizer: Any,
    inferencer: Any | None = None,
    task: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
    worker_id: int,
    machine_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one row and return its prediction without writing shared files."""

    records, stats = _score_work_items(
        model=model,
        tokenizer=tokenizer,
        inferencer=inferencer,
        tasks=[task],
        rows=[row],
        args=args,
        worker_id=worker_id,
        machine_index=machine_index,
    )
    return records[0], stats


def _score_work_items(
    *,
    model: Any | None,
    tokenizer: Any,
    inferencer: Any | None,
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    worker_id: int,
    machine_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score a bounded row batch and return predictions without shared writes."""

    if not rows or len(tasks) != len(rows):
        raise ValueError(
            f"score work batch mismatch: tasks={len(tasks)} rows={len(rows)}"
        )
    requests, candidate_results = _prepare_scoring_requests(
        rows,
        tokenizer,
        args.seq_length,
        enable_fast_mc=inferencer is None,
    )
    if inferencer is not None:
        stats = inferencer.score_requests(requests, candidate_results)
    else:
        if model is None:
            raise RuntimeError("non-vLLM scoring requires a loaded model")
        stats = _score_requests(
            model,
            tokenizer,
            requests,
            candidate_results,
            args.score_batch_size,
            args.pad_multiple,
            args.seq_length,
        )
    records = [
        _finish_scoring_record(
            row,
            raw_candidates,
            Path(task["absolute_file"]),
            worker_id,
            machine_index,
        )
        for task, row, raw_candidates in zip(
            tasks, rows, candidate_results, strict=True
        )
    ]
    for record in records:
        record["worker_id"] = int(worker_id)
        record["machine_index"] = int(machine_index)
    return records, stats


def _generate_work_item(
    *,
    inferencer: Any,
    task: dict[str, Any],
    row: dict[str, Any],
    sample_index: int,
    sample_seed: int,
    args: argparse.Namespace,
    worker_id: int,
    machine_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate one batch-one sample with a topology-independent seed."""

    records, stats = _generate_work_items(
        inferencer=inferencer,
        tasks=[task],
        rows=[row],
        sample_indices=[sample_index],
        sample_seeds=[sample_seed],
        args=args,
        worker_id=worker_id,
        machine_index=machine_index,
    )
    return records[0], stats


def _generate_work_items(
    *,
    inferencer: Any,
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sample_indices: list[int],
    sample_seeds: list[int],
    args: argparse.Namespace,
    worker_id: int,
    machine_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate a bounded request batch with topology-independent seeds."""

    batch_size = len(rows)
    if (
        batch_size <= 0
        or len(tasks) != batch_size
        or len(sample_indices) != batch_size
        or len(sample_seeds) != batch_size
    ):
        raise ValueError(
            "generation work batch mismatch: "
            f"tasks={len(tasks)} rows={len(rows)} "
            f"samples={len(sample_indices)} seeds={len(sample_seeds)}"
        )
    contracts: list[dict[str, Any]] = []
    samplings: list[SamplingParams] = []
    for row, sample_seed in zip(rows, sample_seeds, strict=True):
        contract = _generation_contract(row)
        if args.max_gen_tokens_cap > 0:
            contract["max_gen_toks"] = min(
                int(contract["max_gen_toks"]),
                int(args.max_gen_tokens_cap),
            )
        if contract["remaining_generation_kwargs"]:
            raise ValueError(f"unsupported generation contract: {contract}")
        contracts.append(contract)
        samplings.append(
            SamplingParams(
                max_tokens=int(contract["max_gen_toks"]),
                temperature=(
                    float(contract["temperature"])
                    if contract["do_sample"]
                    else 0.0
                ),
                top_p=float(contract["top_p"]),
                stop=list(contract["stop_strings"]),
                seed=int(sample_seed),
            )
        )
    if hasattr(inferencer, "generate_batch"):
        completions = inferencer.generate_batch(
            [str(row["input"]) for row in rows],
            samplings,
        )
    elif batch_size == 1:
        completions = inferencer.generate(
            [str(rows[0]["input"])],
            samplings[0],
        )
    else:
        raise ValueError("inferencer does not support request-local batched sampling")
    records: list[dict[str, Any]] = []
    returned_token_count = 0
    for (
        task,
        row,
        sample_index,
        sample_seed,
        contract,
        completion,
    ) in zip(
        tasks,
        rows,
        sample_indices,
        sample_seeds,
        contracts,
        completions,
        strict=True,
    ):
        returned_token_count += len(completion.token_ids)
        records.append(
            {
                "schema_version": "hf-core-native-prediction-v2",
                "task_order": int(row["task_order"]),
                "task": row["task"],
                "example_id": row["example_id"],
                "native_id": row.get("native_id"),
                "input_sha256": _text_sha256(str(row["input"])),
                "data_file": str(task["absolute_file"]),
                "metric": row["metric"],
                "request_type": row["request_type"],
                "num_fewshot": row["num_fewshot"],
                "target_index": row.get("target_index"),
                "output": row.get("output"),
                "subset": row.get("subset"),
                "source_task": row.get("source_task"),
                "sample_index": int(sample_index),
                "sample_count": _task_sample_count(task),
                "sample_seed": int(sample_seed),
                "completion": completion.text,
                "generated_token_ids": completion.token_ids,
                "finish_reason": completion.finish_reason,
                **_generation_score(row, completion.text),
                "requires_code_execution": bool(row.get("execution_contract")),
                "generation_contract": contract,
                "distributed_rank": int(worker_id),
                "job_partition_index": int(machine_index),
                "worker_id": int(worker_id),
                "machine_index": int(machine_index),
            }
        )
    return records, {
        "returned_token_count": returned_token_count,
        "generation_calls": 1,
        "request_count": batch_size,
    }


def _score_task(
    *,
    model: Any | None,
    tokenizer: Any,
    inferencer: Any | None,
    task: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
    progress_path: Path,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    task_started = time.perf_counter()
    completed_ids = _completed_example_ids(output_path) if args.resume else set()
    pending = [row for row in rows if _stable_id(row["example_id"]) not in completed_ids]
    stats = {
        "forward_calls": 0,
        "query_token_count": 0,
        "padded_token_slots": 0,
        "continuation_token_count": 0,
    }
    completed = len(rows) - len(pending)
    for chunk_start in range(0, len(pending), args.row_chunk_size):
        chunk = pending[chunk_start : chunk_start + args.row_chunk_size]
        requests, candidate_results = _prepare_scoring_requests(
            chunk,
            tokenizer,
            args.seq_length,
            enable_fast_mc=inferencer is None,
        )
        if inferencer is not None:
            chunk_stats = inferencer.score_requests(requests, candidate_results)
        else:
            if model is None:
                raise RuntimeError("non-vLLM scoring requires a loaded model")
            chunk_stats = _score_requests(
                model,
                tokenizer,
                requests,
                candidate_results,
                args.score_batch_size,
                args.pad_multiple,
                args.seq_length,
            )
        for name, value in chunk_stats.items():
            stats[name] += int(value)
        records = [
            _finish_scoring_record(
                row, raw_candidates, Path(task["absolute_file"]), rank, args.partition_index
            )
            for row, raw_candidates in zip(chunk, candidate_results, strict=True)
        ]
        _append_records(output_path, records)
        completed += len(records)
        if completed % max(1, args.progress_every) == 0 or completed == len(rows):
            _write_progress(
                progress_path,
                task=task,
                completed=completed,
                total=len(rows),
                task_started=task_started,
                status="RUNNING" if completed < len(rows) else "COMPLETE",
            )
    if not pending:
        _write_progress(
            progress_path,
            task=task,
            completed=completed,
            total=len(rows),
            task_started=task_started,
            status="COMPLETE",
        )
    return {
        "task_order": int(task["task_order"]),
        "task": task["task"],
        "mode": "loglikelihood",
        "sample_count": len(rows),
        "resumed_sample_count": len(rows) - len(pending),
        **stats,
        "elapsed_seconds": time.perf_counter() - task_started,
    }


def _generate_task(
    *,
    inferencer: Any,
    task: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
    progress_path: Path,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    task_started = time.perf_counter()
    completed_samples = _completed_generation_samples(output_path) if args.resume else set()
    returned_tokens = 0
    generation_batches = 0
    contract = _generation_contract(rows[0]) if rows else {}
    if contract and args.max_gen_tokens_cap > 0:
        contract["max_gen_toks"] = min(int(contract["max_gen_toks"]), int(args.max_gen_tokens_cap))
    if contract and contract["remaining_generation_kwargs"]:
        raise ValueError(f"unsupported generation contract: {contract}")
    samples_per_example = _task_sample_count(task)
    if samples_per_example <= 0:
        raise ValueError(f"sample_count must be positive: {task}")
    requests = [(row, sample_index) for row in rows for sample_index in range(samples_per_example)]
    expected_samples = {
        (_stable_id(row["example_id"]), sample_index) for row, sample_index in requests
    }
    completed = len(expected_samples & completed_samples)
    request_queue_size = (
        native_vllm_scheduler_queue_size(args, args.generation_batch_size)
        if isinstance(inferencer, NativeVLLMInferencer)
        else args.generation_batch_size
    )
    for start in range(0, len(requests), request_queue_size):
        request_batch = requests[start : start + request_queue_size]
        if all(
            (_stable_id(row["example_id"]), sample_index) in completed_samples
            for row, sample_index in request_batch
        ):
            continue
        request_sample_seeds = [
            (
                EVALUATION_SEED
                + int(task["task_order"]) * 1_000_003
                + (start + offset) // args.generation_batch_size
            )
            % (2**31 - 1)
            for offset in range(len(request_batch))
        ]
        request_samplings = [
            SamplingParams(
                max_tokens=int(contract["max_gen_toks"]),
                temperature=(float(contract["temperature"]) if contract["do_sample"] else 0.0),
                top_p=float(contract["top_p"]),
                stop=list(contract["stop_strings"]),
                seed=sample_seed,
            )
            for sample_seed in request_sample_seeds
        ]
        if isinstance(inferencer, NativeVLLMInferencer):
            completions = inferencer.generate_batch(
                [str(row["input"]) for row, _ in request_batch],
                request_samplings,
            )
        else:
            completions = inferencer.generate(
                [str(row["input"]) for row, _ in request_batch],
                request_samplings[0],
            )
        records: list[dict[str, Any]] = []
        for (row, sample_index), sample_seed, completion in zip(
            request_batch, request_sample_seeds, completions, strict=True
        ):
            sample_key = (_stable_id(row["example_id"]), sample_index)
            if sample_key in completed_samples:
                continue
            returned_tokens += len(completion.token_ids)
            records.append(
                {
                    "schema_version": "hf-core-native-prediction-v2",
                    "task_order": int(row["task_order"]),
                    "task": row["task"],
                    "example_id": row["example_id"],
                    "native_id": row.get("native_id"),
                    "input_sha256": _text_sha256(str(row["input"])),
                    "data_file": str(task["absolute_file"]),
                    "metric": row["metric"],
                    "request_type": row["request_type"],
                    "num_fewshot": row["num_fewshot"],
                    "target_index": row.get("target_index"),
                    "output": row.get("output"),
                    "subset": row.get("subset"),
                    "source_task": row.get("source_task"),
                    "sample_index": sample_index,
                    "sample_count": samples_per_example,
                    "sample_seed": sample_seed,
                    "completion": completion.text,
                    "generated_token_ids": completion.token_ids,
                    "finish_reason": completion.finish_reason,
                    **_generation_score(row, completion.text),
                    "requires_code_execution": bool(row.get("execution_contract")),
                    "generation_contract": contract,
                    "distributed_rank": rank,
                    "job_partition_index": args.partition_index,
                }
            )
        _append_records(output_path, records)
        completed += len(records)
        generation_batches += 1
        if completed % max(1, args.progress_every) == 0 or completed == len(requests):
            _write_progress(
                progress_path,
                task=task,
                completed=completed,
                total=len(requests),
                task_started=task_started,
                status=("RUNNING" if completed < len(requests) else "COMPLETE"),
            )
    if completed == len(requests):
        _write_progress(
            progress_path,
            task=task,
            completed=completed,
            total=len(requests),
            task_started=task_started,
            status="COMPLETE",
        )
    return {
        "task_order": int(task["task_order"]),
        "task": task["task"],
        "mode": "generation",
        "sample_count": len(rows),
        "samples_per_example": samples_per_example,
        "expected_prediction_count": len(requests),
        "resumed_prediction_count": len(expected_samples & completed_samples),
        "generation_batches": generation_batches,
        "returned_token_count": returned_tokens,
        "requires_external_code_execution": any(
            bool(row.get("execution_contract")) for row in rows
        ),
        "generation_contract": contract,
        "elapsed_seconds": time.perf_counter() - task_started,
    }


def _pass_at_k(correctness: list[bool], k: int) -> float | None:
    n = len(correctness)
    if n < k:
        return None
    c = sum(correctness)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _aggregate_prediction_files(
    output_root: Path, tasks: list[dict[str, Any]], expected_counts: dict[str, int]
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_name = str(task["task"])
        samples_per_example = _task_sample_count(task)
        accumulators[task_name] = {
            "manifest": task,
            "expected_examples": expected_counts[task_name],
            "expected_predictions": (expected_counts[task_name] * samples_per_example),
            "records_by_example": defaultdict(list),
            "prediction_keys": set(),
            "prediction_files": [],
        }
    prediction_paths = sorted(output_root.glob("**/predictions/*.jsonl"))
    for path in prediction_paths:
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        if not records:
            continue
        task_name = str(records[0]["task"])
        if task_name not in accumulators:
            continue
        accumulator = accumulators[task_name]
        accumulator["prediction_files"].append(str(path))
        for record in records:
            example_key = _stable_id(record["example_id"])
            sample_index = int(record.get("sample_index", 0))
            prediction_key = (example_key, sample_index)
            if prediction_key in accumulator["prediction_keys"]:
                raise RuntimeError(f"duplicate prediction for {task_name}:{prediction_key}")
            accumulator["prediction_keys"].add(prediction_key)
            accumulator["records_by_example"][example_key].append(record)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        accumulator = accumulators[str(task["task"])]
        metric = str(task["metric"])
        is_generation = _task_is_generation(task)
        example_values: list[float] = []
        subset_values: dict[str, list[float]] = defaultdict(list)
        pass_at_4_values: list[float] = []
        pending_external_scores = 0
        correct_samples = 0
        scored_samples = 0
        for records in accumulator["records_by_example"].values():
            subset = str(records[0].get("subset") or "__all__")
            if is_generation:
                correctness = [
                    record.get("primary_correct") for record in records
                ]
                pending_external_scores += sum(
                    value is None for value in correctness
                )
                scored = [bool(value) for value in correctness if value is not None]
                correct_samples += sum(scored)
                scored_samples += len(scored)
                if not scored:
                    continue
                value = sum(scored) / len(scored)
                if "pass@4" in (task.get("secondary_metrics") or []):
                    pass_at_4 = _pass_at_k(scored, 4)
                    if pass_at_4 is not None:
                        pass_at_4_values.append(pass_at_4)
            elif metric in ACCURACY_METRICS:
                value = float(bool(records[0]["primary_correct"]))
            else:
                value = float(records[0]["target_bits_per_byte"])
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
        observed_examples = len(accumulator["records_by_example"])
        observed_predictions = len(accumulator["prediction_keys"])
        prediction_complete = observed_examples == int(
            accumulator["expected_examples"]
        ) and observed_predictions == int(accumulator["expected_predictions"])
        score_complete = (
            prediction_complete
            and pending_external_scores == 0
            and len(example_values) == observed_examples
        )
        row = {
            "task_order": int(task["task_order"]),
            "task": task["task"],
            "name": task.get("name"),
            "spec": task.get("spec"),
            "metric": metric,
            "request_type": task.get("request_type"),
            "num_fewshot": task["num_fewshot"],
            "aggregation": task.get("aggregation", "mean_over_examples"),
            "samples_per_example": _task_sample_count(task),
            "expected_examples": int(accumulator["expected_examples"]),
            "observed_examples": observed_examples,
            "expected_predictions": int(accumulator["expected_predictions"]),
            "observed_predictions": observed_predictions,
            "prediction_complete": prediction_complete,
            "score_complete": score_complete,
            "pending_external_scores": pending_external_scores,
            "primary_score": primary_score,
            "subset_scores": subset_scores,
            "prediction_files": accumulator["prediction_files"],
        }
        row["score_status"] = (
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
            row["accuracy"] = primary_score
            row["correct_samples"] = correct_samples
            row["scored_samples"] = scored_samples
        else:
            row["mean_target_bits_per_byte"] = primary_score
        if pass_at_4_values:
            row["pass@4"] = sum(pass_at_4_values) / len(pass_at_4_values)
        rows.append(row)

    prediction_tasks_complete = sum(bool(row["prediction_complete"]) for row in rows)
    score_tasks_complete = sum(bool(row["score_complete"]) for row in rows)
    return {
        "schema_version": "hf-core-native-summary-v2",
        "task_count": len(rows),
        "prediction_task_count_complete": prediction_tasks_complete,
        "score_task_count_complete": score_tasks_complete,
        "task_count_complete": prediction_tasks_complete,
        "expected_examples": sum(int(row["expected_examples"]) for row in rows),
        "observed_examples": sum(int(row["observed_examples"]) for row in rows),
        "expected_predictions": sum(int(row["expected_predictions"]) for row in rows),
        "observed_predictions": sum(int(row["observed_predictions"]) for row in rows),
        "tasks": rows,
    }


def _write_scores_csv(path: Path, report: dict[str, Any]) -> None:
    """Write one deterministic, manifest-ordered score row per task."""

    fieldnames = [
        "model_label",
        "hf_model_path",
        "profile",
        "task_order",
        "task",
        "name",
        "spec",
        "num_fewshot",
        "metric",
        "request_type",
        "aggregation",
        "primary_score",
        "accuracy",
        "mean_target_bits_per_byte",
        "pass@4",
        "score_status",
        "prediction_complete",
        "score_complete",
        "expected_examples",
        "observed_examples",
        "samples_per_example",
        "expected_predictions",
        "observed_predictions",
        "pending_external_scores",
        "correct_samples",
        "scored_samples",
        "subset_scores_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in sorted(
            report.get("tasks") or [],
            key=lambda row: int(row["task_order"]),
        ):
            writer.writerow(
                {
                    "model_label": report.get("model_label", ""),
                    "hf_model_path": report.get("hf_model_path", ""),
                    "profile": report.get("profile", ""),
                    "task_order": int(task["task_order"]),
                    "task": task["task"],
                    "name": task.get("name") or "",
                    "spec": task.get("spec") or "",
                    "num_fewshot": task.get("num_fewshot", ""),
                    "metric": task.get("metric", ""),
                    "request_type": task.get("request_type") or "",
                    "aggregation": task.get("aggregation", ""),
                    "primary_score": (
                        ""
                        if task.get("primary_score") is None
                        else task["primary_score"]
                    ),
                    "accuracy": (
                        ""
                        if task.get("accuracy") is None
                        else task["accuracy"]
                    ),
                    "mean_target_bits_per_byte": (
                        ""
                        if task.get("mean_target_bits_per_byte") is None
                        else task["mean_target_bits_per_byte"]
                    ),
                    "pass@4": (
                        "" if task.get("pass@4") is None else task["pass@4"]
                    ),
                    "score_status": task.get("score_status", ""),
                    "prediction_complete": task.get("prediction_complete", ""),
                    "score_complete": task.get("score_complete", ""),
                    "expected_examples": task.get("expected_examples", ""),
                    "observed_examples": task.get("observed_examples", ""),
                    "samples_per_example": task.get(
                        "samples_per_example", ""
                    ),
                    "expected_predictions": task.get(
                        "expected_predictions", ""
                    ),
                    "observed_predictions": task.get(
                        "observed_predictions", ""
                    ),
                    "pending_external_scores": task.get(
                        "pending_external_scores", ""
                    ),
                    "correct_samples": task.get("correct_samples", ""),
                    "scored_samples": task.get("scored_samples", ""),
                    "subset_scores_json": json.dumps(
                        task.get("subset_scores") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError(
            "partition_index must be in [0, partition_count), got "
            f"{args.partition_index}/{args.partition_count}"
        )
    for name in (
        "score_batch_size",
        "generation_batch_size",
        "row_chunk_size",
        "pad_multiple",
        "seq_length",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("generation_samples_cap", "max_gen_tokens_cap"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.max_gen_tokens_cap > 0 and args.limit_per_task <= 0:
        raise ValueError("max-gen-tokens-cap is smoke-only and requires --limit-per-task")

    process_started = time.perf_counter()
    rank, local_rank, world_size = _distributed_args()
    _initialize_distributed_process_group(rank, world_size)
    compile_cache_paths = _isolate_compile_caches(rank, world_size)
    summary_payload = json.loads((args.data_root / "summary.json").read_text(encoding="utf-8"))
    profile_section = summary_payload.get(args.profile)
    if not isinstance(profile_section, dict):
        raise ValueError(f"unknown Core profile: {args.profile}")
    profile_task_count = int(
        profile_section.get("task_count", len(profile_section.get("tasks") or []))
    )
    task_orders = _parse_task_orders(args.task_orders, profile_task_count)
    summary, tasks = _load_manifest(
        args.data_root,
        args.profile,
        task_orders,
    )
    if args.generation_samples_cap > 0:
        for task in tasks:
            if _task_is_generation(task):
                task["evaluation_sample_count"] = min(
                    int(task.get("sample_count") or 1),
                    int(args.generation_samples_cap),
                )
    hf_backend = _resolve_hf_backend(args)
    if hf_backend == NATIVE_MEGATRON_BACKEND:
        if not args.checkpoint_root:
            raise ValueError("--checkpoint-root is required for native_megatron")
        if args.ckpt_step <= 0:
            raise ValueError("--ckpt-step must be positive for native_megatron")
        if args.score_batch_size != 1 or args.generation_batch_size != 1:
            raise ValueError("native_megatron Core88 requires batch size one")
    if hf_backend == LMDEPLOY_BACKEND:
        validate_lmdeploy_args(
            args,
            batch_size=_max_inference_batch_size(args),
            processes_per_gpu=int(
                os.environ.get("CONCEPTLM_PROCESSES_PER_GPU", "1")
            ),
        )
        if world_size != 1:
            raise ValueError(
                "core_native_eval lmdeploy mode supports one local engine; "
                "use core_native_pool for multi-GPU dynamic dispatch"
            )
    source_artifact_before = None
    if hf_backend == NATIVE_VLLM_BACKEND:
        validate_native_vllm_args(
            args,
            batch_size=_max_inference_batch_size(args),
            processes_per_gpu=int(
                os.environ.get("CONCEPTLM_PROCESSES_PER_GPU", "1")
            ),
        )
        if world_size != 1:
            raise ValueError(
                "core_native_eval native_vllm mode supports one local engine; "
                "use core_native_pool for multi-GPU dynamic dispatch"
            )
        model_identity = Path(
            args.model_identity_path or args.hf_model_path
        ).resolve()
        source_artifact_before = benchmark._hf_artifact_fingerprint(
            model_identity
        )
        overlay_dir = (
            Path(args.vllm_model_overlay_dir)
            if args.vllm_model_overlay_dir
            else args.output_root / "native-vllm-model"
        )
        runtime_model, _ = prepare_native_vllm_model(
            source_model=args.hf_model_path,
            runtime_config=args.vllm_runtime_config or None,
            overlay_dir=overlay_dir,
            model_family=args.vllm_model_family,
        )
        args.model_identity_path = str(model_identity)
        args.hf_model_path = str(runtime_model)
    model_args = _model_args(args, hf_backend)
    for task in tasks:
        task["absolute_file"] = str(args.data_root / str(task["file"]))
    if args.verify_data_sha256 and rank == 0:
        for task in tasks:
            actual = _file_sha256(Path(task["absolute_file"]))
            if actual != task["sha256"]:
                raise RuntimeError(
                    f"dataset SHA256 mismatch for {task['task']}: " f"{actual} != {task['sha256']}"
                )

    expected_counts = {
        str(task["task"]): _selected_count(
            int(task["num_examples"]),
            args.limit_per_task,
            args.partition_index,
            args.partition_count,
            world_size,
        )
        for task in tasks
    }
    model_identity_path, runtime_hf_model_path = _model_manifest_paths(args)
    run_manifest = {
        "schema_version": "hf-core-native-run-v2",
        "created_at": _utc_now(),
        "model_label": args.model_label,
        "hf_model_path": model_identity_path,
        "runtime_hf_model_path": runtime_hf_model_path,
        "hf_backend_requested": args.hf_backend,
        "hf_backend_resolved": hf_backend,
        "data_root": str(args.data_root.resolve()),
        "profile": args.profile,
        "profile_task_count": profile_task_count,
        "data_summary_sha256": _file_sha256(args.data_root / "summary.json"),
        "source_schema_version": summary.get("schema_version"),
        "task_orders": [int(task["task_order"]) for task in tasks],
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "distributed_world_size": world_size,
        "local_process_count": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "processes_per_gpu": int(os.environ.get("CONCEPTLM_PROCESSES_PER_GPU", "1")),
        "limit_per_task": args.limit_per_task,
        "generation_samples_cap": args.generation_samples_cap,
        "max_gen_tokens_cap": args.max_gen_tokens_cap,
        "seq_length": args.seq_length,
        "score_batch_size": args.score_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "row_chunk_size": args.row_chunk_size,
        "pad_multiple": args.pad_multiple,
        "effective_score_pad_multiple": _effective_scoring_pad_multiple(
            args.score_batch_size,
            args.pad_multiple,
        ),
        "hf_compile_routes": bool(args.hf_compile_routes),
        "hf_align_dcp_runtime_config": bool(args.hf_align_dcp_runtime_config),
        "native_vllm_requested": hf_backend == NATIVE_VLLM_BACKEND,
        "native_megatron_requested": hf_backend == NATIVE_MEGATRON_BACKEND,
        "lmdeploy_requested": hf_backend == LMDEPLOY_BACKEND,
        "checkpoint_root": (
            str(Path(args.checkpoint_root).resolve())
            if hf_backend == NATIVE_MEGATRON_BACKEND
            else None
        ),
        "ckpt_step": args.ckpt_step if hf_backend == NATIVE_MEGATRON_BACKEND else None,
        "native_megatron_generation": (
            "full_prefix_recompute_no_kv_cache"
            if hf_backend == NATIVE_MEGATRON_BACKEND
            else None
        ),
        "native_vllm_config": (
            {
                "experimental": True,
                "model_family": args.vllm_model_family,
                "max_model_len": native_vllm_max_model_len(args),
                "max_num_seqs": _max_inference_batch_size(args),
                "scheduler_queue_size": native_vllm_scheduler_queue_size(
                    args, _max_inference_batch_size(args)
                ),
                "execution_mode": args.vllm_execution_mode,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "attention_backend": args.vllm_attention_backend,
                "flash_attn_version": args.vllm_flash_attn_version,
                "hlm_attention_impl": args.vllm_hlm_attention_impl,
                "prefix_caching": False,
                **native_vllm_speculative_manifest(args),
            }
            if hf_backend == NATIVE_VLLM_BACKEND
            else None
        ),
        "lmdeploy_config": (
            lmdeploy_engine_policy(
                max_batch_size=max(
                    args.score_batch_size, args.generation_batch_size
                ),
                cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            )
            if hf_backend == LMDEPLOY_BACKEND
            else None
        ),
        "flash_decode": False,
        "fewshot_contract": "already_embedded_in_input_no_additional_shots",
        "tokenization_contract": (
            "olmo_eval_f881_encode_context_and_continuation;"
            "move_context_trailing_spaces_to_continuation;"
            "joint_context_continuation_encoding;"
            "BOS_only_when_tokenizer_add_bos_token;"
            "causal_shift;left_truncate_to_seq_length"
        ),
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "metric_contract": {
            "acc": (
                "request-aware: argmax summed continuation logprob for "
                "loglikelihood or normalized exact-answer correctness for generation"
            ),
            "acc_raw": "argmax summed continuation logprob",
            "acc_per_token": "argmax summed logprob divided by continuation tokens",
            "acc_per_char": "argmax summed logprob divided by Python character count",
            "bits_per_byte": "mean gold -log2 probability per UTF-8 byte",
            "bits_per_byte_corr": "mean gold -log2 probability per UTF-8 byte",
            "logits_per_byte": "mean gold -log2 probability per UTF-8 byte",
            "exact_match": "unified_exact_v1 over task-config generation",
            "pass@1": "mean sample correctness; code execution is externally graded",
            "execution_accuracy": (
                "mean sample correctness from an external isolated code grader"
            ),
            "pass@4": "unbiased pass@k estimator from saved per-example samples",
        },
        "expected_counts": expected_counts,
        "tasks": tasks,
    }
    if rank == 0:
        _write_json_atomic(args.output_root / "run_manifest.json", run_manifest)

    artifact_before = benchmark._artifact_fingerprint(model_args)
    print(
        f"[core-native] rank={rank}/{world_size} loading model={args.model_label} "
        f"backend={hf_backend} profile={args.profile} "
        f"partition={args.partition_index}/{args.partition_count}",
        flush=True,
    )
    model_load_started = time.perf_counter()
    native_inferencer: NativeVLLMInferencer | None = None
    lmdeploy_inferencer: LMDeployInferencer | None = None
    if hf_backend == NATIVE_VLLM_BACKEND:
        native_inferencer = NativeVLLMInferencer(
            model_path=args.hf_model_path,
            max_model_len=native_vllm_max_model_len(args),
            max_batch_size=_max_inference_batch_size(args),
            scheduler_queue_size=native_vllm_scheduler_queue_size(
                args, _max_inference_batch_size(args)
            ),
            seed=EVALUATION_SEED,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            execution_mode=args.vllm_execution_mode,
            attention_backend=args.vllm_attention_backend,
            flash_attn_version=args.vllm_flash_attn_version,
            hlm_attention_impl=args.vllm_hlm_attention_impl,
            model_family=args.vllm_model_family,
            **native_vllm_speculative_kwargs(args),
        )
        model = None
        tokenizer = native_inferencer.tokenizer
        inferencer: Any = native_inferencer
    elif hf_backend == LMDEPLOY_BACKEND:
        lmdeploy_inferencer = LMDeployInferencer(
            model_path=args.hf_model_path,
            seed=EVALUATION_SEED,
            max_batch_size=_max_inference_batch_size(args),
            cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            log_level=args.lmdeploy_log_level,
        )
        model = None
        tokenizer = lmdeploy_inferencer.tokenizer
        inferencer = lmdeploy_inferencer
    else:
        eval_model = benchmark.load_eval_model(model_args)
        model = getattr(eval_model, "hf_model", None)
        if model is None:
            model = eval_model.model
        tokenizer = eval_model.tokenizer
        if hf_backend == NATIVE_MEGATRON_BACKEND:
            inferencer = build_native_megatron_inferencer(eval_model)
        else:
            inferencer = TransformersInferencer(
                eval_model,
                max_batch_size=args.generation_batch_size,
            )
    model_load_seconds = time.perf_counter() - model_load_started
    rank_root = _rank_root(args.output_root, rank, world_size)
    predictions_root = rank_root / "predictions"
    progress_root = rank_root / "progress"
    task_results: list[dict[str, Any]] = []
    rank_expected = 0
    for task in tasks:
        data_path = Path(task["absolute_file"])
        all_rows = _read_rows(data_path)
        if args.limit_per_task > 0:
            all_rows = all_rows[: args.limit_per_task]
        rows = [
            row
            for row_index, row in enumerate(all_rows)
            if _selected_row(
                row_index, args.partition_index, args.partition_count, rank, world_size
            )
        ]
        rank_expected += len(rows)
        task_order = int(task["task_order"])
        task_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task["task"]))
        prediction_path = predictions_root / f"{task_order:03d}_{task_slug}.jsonl"
        progress_path = progress_root / f"{task_order:03d}_{task_slug}.json"
        print(
            f"[core-native] rank={rank} task={task['task']} rows={len(rows)} "
            f"metric={task['metric']}",
            flush=True,
        )
        if _task_is_generation(task):
            result = _generate_task(
                inferencer=inferencer,
                task=task,
                rows=rows,
                output_path=prediction_path,
                progress_path=progress_path,
                args=args,
                rank=rank,
            )
        else:
            result = _score_task(
                model=model,
                tokenizer=tokenizer,
                inferencer=(
                    inferencer
                    if hf_backend
                    in (NATIVE_VLLM_BACKEND, NATIVE_MEGATRON_BACKEND, LMDEPLOY_BACKEND)
                    else None
                ),
                task=task,
                rows=rows,
                output_path=prediction_path,
                progress_path=progress_path,
                args=args,
                rank=rank,
            )
        task_results.append(result)

    if lmdeploy_inferencer is not None:
        lmdeploy_inferencer.close()
    artifact_after = benchmark._artifact_fingerprint(model_args)
    artifact_mutated = artifact_before != artifact_after
    if artifact_mutated:
        raise RuntimeError("runtime HF model artifact changed during evaluation")
    source_artifact_after = (
        benchmark._hf_artifact_fingerprint(model_identity_path)
        if source_artifact_before is not None
        else None
    )
    source_artifact_mutated = (
        source_artifact_before != source_artifact_after
        if source_artifact_before is not None
        else artifact_mutated
    )
    if source_artifact_mutated:
        raise RuntimeError("source HF model artifact changed during evaluation")
    rank_result = {
        "status": "CORE_NATIVE_EVAL_OK",
        "model_label": args.model_label,
        "hf_model_path": str(Path(args.hf_model_path).resolve()),
        "model_identity_path": model_identity_path,
        "hf_backend": hf_backend,
        "profile": args.profile,
        "distributed_rank": rank,
        "distributed_local_rank": local_rank,
        "distributed_world_size": world_size,
        "cuda_device_index": int(torch.cuda.current_device()),
        "processes_per_gpu": int(os.environ.get("CONCEPTLM_PROCESSES_PER_GPU", "1")),
        "job_partition_index": args.partition_index,
        "job_partition_count": args.partition_count,
        "expected_examples": rank_expected,
        "task_results": task_results,
        "model_load_seconds": model_load_seconds,
        "process_wall_seconds": time.perf_counter() - process_started,
        "peak_gpu_memory_gib": benchmark._peak_memory_gib(),
        "compile_cache_paths": compile_cache_paths,
        "native_vllm_runtime": (
            native_inferencer.runtime_metadata
            if native_inferencer is not None
            else None
        ),
        "lmdeploy_runtime": (
            lmdeploy_inferencer.runtime_metadata
            if lmdeploy_inferencer is not None
            else None
        ),
        "artifact_mutated": artifact_mutated,
        "source_artifact_mutated": source_artifact_mutated,
        "artifact_metadata_before": artifact_before,
        "artifact_metadata_after": artifact_after,
        "source_artifact_metadata_before": source_artifact_before,
        "source_artifact_metadata_after": source_artifact_after,
        "predictions_root": str(predictions_root),
        "finished_at": _utc_now(),
    }
    _write_json_atomic(rank_root / "result.json", rank_result)
    print(json.dumps(rank_result, indent=2, ensure_ascii=False), flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        merged = _aggregate_prediction_files(args.output_root, tasks, expected_counts)
        merged.update(
            {
                "status": (
                    "CORE_NATIVE_JOB_OK"
                    if merged["score_task_count_complete"] == merged["task_count"]
                    else (
                        "CORE_NATIVE_JOB_PREDICTIONS_OK"
                        if merged["prediction_task_count_complete"] == merged["task_count"]
                        else "CORE_NATIVE_JOB_INCOMPLETE"
                    )
                ),
                "model_label": args.model_label,
                "hf_model_path": str(Path(args.hf_model_path).resolve()),
                "hf_backend": hf_backend,
                "profile": args.profile,
                "partition_index": args.partition_index,
                "partition_count": args.partition_count,
                "run_manifest": str(args.output_root / "run_manifest.json"),
                "scores_csv": str(args.output_root / "scores.csv"),
            }
        )
        _write_json_atomic(args.output_root / "summary.json", merged)
        _write_scores_csv(args.output_root / "scores.csv", merged)
        if merged["status"] == "CORE_NATIVE_JOB_INCOMPLETE":
            raise RuntimeError(
                "merged prediction coverage is incomplete: "
                f"{merged['observed_predictions']}/"
                f"{merged['expected_predictions']}"
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
