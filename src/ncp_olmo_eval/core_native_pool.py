#!/usr/bin/env python3
"""Run one machine of a hierarchical Core evaluation worker pool.

One CPU coordinator owns all output files.  It dynamically dispatches bounded
request batches to independent GPU workers and acknowledges work only after
the corresponding predictions have been appended.  No worker participates in
a distributed process group or a final barrier.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import queue
import random
import re
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from . import benchmark
from .core_native_eval import (
    _aggregate_prediction_files,
    _apply_generation_samples_cap,
    _effective_scoring_pad_multiple,
    _file_sha256,
    _generate_work_items,
    _load_manifest,
    _model_manifest_paths,
    _read_rows,
    _score_work_items,
    _stable_id,
    _task_is_generation,
    _write_json_atomic,
    _write_scores_csv,
)
from .core_native_work import (
    WorkItem,
    load_machine_manifest,
    stable_request_key,
    stable_request_seed,
)
from .lmdeploy_inference import (
    LMDEPLOY_BACKEND,
    add_lmdeploy_args,
    lmdeploy_engine_policy,
    validate_lmdeploy_args,
)
from .native_megatron_inference import NATIVE_MEGATRON_BACKEND
from .native_vllm_inference import (
    NATIVE_VLLM_BACKEND,
    add_native_vllm_args,
    native_vllm_scheduler_queue_size,
    native_vllm_speculative_manifest,
    prepare_native_vllm_model,
    validate_native_vllm_args,
)


def parse_args() -> argparse.Namespace:
    """Parse one machine coordinator command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--profile", default="core88")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--machine-index", type=int, required=True)
    parser.add_argument("--machine-count", type=int, default=4)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--workflow-id", default="")
    parser.add_argument("--repo-commit", default="")
    parser.add_argument("--hf-model-path", default="")
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--ckpt-step", type=int, default=0)
    parser.add_argument("--train-wandb-config", default="")
    parser.add_argument("--model-identity-path", default="")
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
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--processes-per-gpu", type=int, default=4)
    parser.add_argument("--worker-restarts", type=int, default=1)
    parser.add_argument("--worker-start-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--row-chunk-size", type=int, default=1)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--generation-samples-cap", type=int, default=1)
    parser.add_argument("--max-gen-tokens-cap", type=int, default=0)
    parser.add_argument("--hf-compile-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hf-align-dcp-runtime-config", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify-data-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_native_vllm_args(parser)
    add_lmdeploy_args(parser)
    return parser.parse_args()


class PredictionWriter:
    """Single-writer append handles for machine-local prediction files."""

    def __init__(self, predictions_root: Path) -> None:
        self.predictions_root = predictions_root
        self._handles: dict[tuple[int, str], Any] = {}

    def append(self, record: dict[str, Any]) -> Path:
        """Append one prediction and flush it before returning."""

        task_order = int(record["task_order"])
        task_name = str(record["task"])
        key = (task_order, task_name)
        handle = self._handles.get(key)
        if handle is None:
            task_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name)
            path = self.predictions_root / f"{task_order:03d}_{task_slug}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self._handles[key] = handle
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        return Path(handle.name)

    def close(self) -> None:
        """Close every task output handle."""

        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _worker_main(
    worker_id: int,
    worker_epoch: int,
    worker_count: int,
    processes_per_gpu: int,
    args_payload: dict[str, Any],
    input_queue: Any,
    result_queue: Any,
) -> None:
    native_vllm = args_payload.get("hf_backend") == NATIVE_VLLM_BACKEND
    native_megatron = args_payload.get("hf_backend") == NATIVE_MEGATRON_BACKEND
    lmdeploy_backend = args_payload.get("hf_backend") == LMDEPLOY_BACKEND
    if native_vllm or native_megatron or lmdeploy_backend:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id // processes_per_gpu)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    else:
        os.environ["LOCAL_RANK"] = str(worker_id)
        os.environ["LOCAL_WORLD_SIZE"] = str(worker_count)
    # These are independent model replicas, not a distributed process group.
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if native_megatron:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29600 + worker_id)
    os.environ["CONCEPTLM_PROCESSES_PER_GPU"] = str(processes_per_gpu)
    try:
        import torch

        from . import benchmark
        from .core_native_eval import _isolate_compile_caches, _model_args, _resolve_hf_backend
        from .device_layout import local_cuda_device_index
        from .lmdeploy_inference import LMDeployInferencer
        from .native_megatron_inference import build_native_megatron_inferencer
        from .native_vllm_inference import (
            NativeVLLMInferencer,
            native_vllm_max_model_len,
            native_vllm_scheduler_queue_size,
            native_vllm_speculative_kwargs,
        )
        from .transformers_inference import TransformersInferencer

        args = argparse.Namespace(**args_payload)
        device_index = (
            0
            if (native_vllm or native_megatron or lmdeploy_backend)
            else local_cuda_device_index()
        )
        torch.cuda.set_device(device_index)
        random.seed(int(args.global_seed))
        torch.manual_seed(int(args.global_seed))
        compile_cache_paths = _isolate_compile_caches(worker_id, worker_count)
        hf_backend = _resolve_hf_backend(args)
        model_args = _model_args(args, hf_backend)
        artifact_before = benchmark._artifact_fingerprint(model_args)
        load_started = time.perf_counter()
        if hf_backend == NATIVE_VLLM_BACKEND:
            inferencer = NativeVLLMInferencer(
                model_path=args.hf_model_path,
                max_model_len=native_vllm_max_model_len(args),
                seed=args.global_seed,
                max_batch_size=max(args.score_batch_size, args.generation_batch_size),
                scheduler_queue_size=native_vllm_scheduler_queue_size(
                    args, max(args.score_batch_size, args.generation_batch_size)
                ),
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                execution_mode=args.vllm_execution_mode,
                attention_backend=args.vllm_attention_backend,
                flash_attn_version=args.vllm_flash_attn_version,
                hlm_attention_impl=args.vllm_hlm_attention_impl,
                model_family=args.vllm_model_family,
                **native_vllm_speculative_kwargs(args),
            )
            model = None
            tokenizer = inferencer.tokenizer
        elif hf_backend == LMDEPLOY_BACKEND:
            inferencer = LMDeployInferencer(
                model_path=args.hf_model_path,
                seed=args.global_seed,
                max_batch_size=max(
                    args.score_batch_size, args.generation_batch_size
                ),
                cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
                log_level=args.lmdeploy_log_level,
            )
            model = None
            tokenizer = inferencer.tokenizer
        elif hf_backend == NATIVE_MEGATRON_BACKEND:
            eval_model = benchmark.load_eval_model(model_args)
            model = eval_model.model
            tokenizer = eval_model.tokenizer
            inferencer = build_native_megatron_inferencer(eval_model)
        else:
            eval_model = benchmark.load_eval_model(model_args)
            model = getattr(eval_model, "hf_model", None)
            if model is None:
                model = eval_model.model
            tokenizer = eval_model.tokenizer
            inferencer = TransformersInferencer(eval_model, max_batch_size=1)
        result_queue.put(
            {
                "event": "READY",
                "worker_id": worker_id,
                "worker_epoch": worker_epoch,
                "gpu_index": device_index,
                "model_load_seconds": time.perf_counter() - load_started,
                "compile_cache_paths": compile_cache_paths,
                "artifact_before": artifact_before,
                "backend_runtime": (
                    inferencer.runtime_metadata
                    if hf_backend in (NATIVE_VLLM_BACKEND, LMDEPLOY_BACKEND)
                    else None
                ),
            }
        )
        while True:
            payload = input_queue.get()
            if payload["command"] == "STOP":
                if hf_backend == LMDEPLOY_BACKEND:
                    inferencer.close()
                artifact_after = benchmark._artifact_fingerprint(model_args)
                result_queue.put(
                    {
                        "event": "STOPPED",
                        "worker_id": worker_id,
                        "worker_epoch": worker_epoch,
                        "gpu_index": device_index,
                        "artifact_mutated": artifact_before != artifact_after,
                        "artifact_before": artifact_before,
                        "artifact_after": artifact_after,
                        "peak_gpu_memory_gib": benchmark._peak_memory_gib(),
                    }
                )
                return
            if payload["command"] != "RUN_BATCH":
                raise RuntimeError(f"worker received an unsupported command: {payload['command']}")
            batch_payloads = list(payload["items"])
            works = [WorkItem.from_json(item["work"]) for item in batch_payloads]
            if not works:
                raise RuntimeError("worker received an empty request batch")
            modes = {work.mode for work in works}
            if len(modes) != 1:
                raise RuntimeError(f"worker received mixed request modes: {modes}")
            request_keys = [work.request_key for work in works]
            result_queue.put(
                {
                    "event": "STARTED",
                    "worker_id": worker_id,
                    "worker_epoch": worker_epoch,
                    "request_keys": request_keys,
                }
            )
            started = time.perf_counter()
            if works[0].mode == "generation":
                records, stats = _generate_work_items(
                    inferencer=inferencer,
                    tasks=[item["task"] for item in batch_payloads],
                    rows=[item["row"] for item in batch_payloads],
                    sample_indices=[int(work.sample_index) for work in works],
                    sample_seeds=[int(item["sample_seed"]) for item in batch_payloads],
                    args=args,
                    worker_id=worker_id,
                    machine_index=int(args.machine_index),
                )
            else:
                records, stats = _score_work_items(
                    model=model,
                    tokenizer=tokenizer,
                    inferencer=(
                        inferencer
                        if hf_backend
                        in (NATIVE_VLLM_BACKEND, NATIVE_MEGATRON_BACKEND, LMDEPLOY_BACKEND)
                        else None
                    ),
                    tasks=[item["task"] for item in batch_payloads],
                    rows=[item["row"] for item in batch_payloads],
                    args=args,
                    worker_id=worker_id,
                    machine_index=int(args.machine_index),
                )
            result_queue.put(
                {
                    "event": "RECORD",
                    "worker_id": worker_id,
                    "worker_epoch": worker_epoch,
                    "request_keys": request_keys,
                    "records": records,
                    "stats": stats,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    except Exception as error:
        result_queue.put(
            {
                "event": "ERROR",
                "worker_id": worker_id,
                "worker_epoch": worker_epoch,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _prediction_request_key(record: dict[str, Any], generation_tasks: set[str]) -> str:
    sample_index = (
        int(record.get("sample_index", 0)) if str(record["task"]) in generation_tasks else None
    )
    return stable_request_key(str(record["task"]), record["example_id"], sample_index)


def _completed_request_keys(output_root: Path, generation_tasks: set[str]) -> set[str]:
    completed: set[str] = set()
    for path in sorted(output_root.glob("machine-*/predictions/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid resume JSON at {path}:{line_number}") from error
                key = _prediction_request_key(record, generation_tasks)
                if key in completed:
                    raise RuntimeError(f"duplicate completed request: {key}")
                completed.add(key)
    return completed


def _materialize_work(
    *, data_root: Path, profile: str, plan: dict[str, Any], items: list[WorkItem]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    task_orders = {int(value) for value in plan["task_orders"]}
    _, tasks = _load_manifest(data_root, profile, task_orders)
    _apply_generation_samples_cap(tasks, int(plan["generation_samples_cap"]))
    items_by_task: dict[int, dict[int, list[WorkItem]]] = {}
    for item in items:
        items_by_task.setdefault(item.task_order, {}).setdefault(item.row_index, []).append(item)
    materialized: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_order = int(task["task_order"])
        task["absolute_file"] = str(data_root / str(task["file"]))
        requested_rows = items_by_task.get(task_order, {})
        if not requested_rows:
            continue
        rows = _read_rows(Path(task["absolute_file"]))
        limit = int(plan["limit_per_task"])
        if limit > 0:
            rows = rows[:limit]
        for row_index, work_items in requested_rows.items():
            if not 0 <= row_index < len(rows):
                raise RuntimeError(
                    f"planned row {task_order}:{row_index} is outside {len(rows)} rows"
                )
            row = rows[row_index]
            for work in work_items:
                if work.example_key != _stable_id(row["example_id"]):
                    raise RuntimeError(f"planned example identity changed: {work.request_key}")
                materialized[work.request_key] = {"work": work.to_json(), "task": task, "row": row}
    if len(materialized) != len(items):
        raise RuntimeError(f"materialized {len(materialized)}/{len(items)} work items")
    return tasks, materialized


def _validate_args(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if args.score_batch_size <= 0 or args.generation_batch_size <= 0:
        raise ValueError("hierarchical Core batch sizes must be positive")
    if args.pad_multiple <= 0:
        raise ValueError("hierarchical Core pad multiple must be positive")
    if args.hf_backend == NATIVE_MEGATRON_BACKEND:
        if not args.checkpoint_root or args.ckpt_step <= 0:
            raise ValueError("native_megatron requires checkpoint_root and a positive ckpt_step")
        if args.processes_per_gpu != 1:
            raise ValueError("native_megatron currently requires one process per GPU")
    if args.hf_backend not in (NATIVE_VLLM_BACKEND, LMDEPLOY_BACKEND) and (
        args.score_batch_size != 1 or args.generation_batch_size != 1
    ):
        raise ValueError(
            "multi-request hierarchical Core batches require native_vllm or lmdeploy"
        )
    if args.gpus <= 0 or args.processes_per_gpu <= 0:
        raise ValueError("gpus and processes_per_gpu must be positive")
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        validate_native_vllm_args(
            args,
            batch_size=max(args.score_batch_size, args.generation_batch_size),
            processes_per_gpu=args.processes_per_gpu,
        )
    elif args.hf_backend == LMDEPLOY_BACKEND:
        validate_lmdeploy_args(
            args,
            batch_size=max(args.score_batch_size, args.generation_batch_size),
            processes_per_gpu=args.processes_per_gpu,
        )
    if args.machine_count != int(plan["machine_count"]):
        raise ValueError("machine count does not match the dispatch plan")
    if args.global_seed != int(plan["global_seed"]):
        raise ValueError("all four machine jobs must use the dispatch plan global seed")
    if args.profile != str(plan["profile"]):
        raise ValueError("profile does not match the dispatch plan")
    if args.limit_per_task != int(plan["limit_per_task"]):
        raise ValueError("limit_per_task does not match the dispatch plan")
    if args.generation_samples_cap != int(plan["generation_samples_cap"]):
        raise ValueError("generation_samples_cap does not match the dispatch plan")
    if args.max_gen_tokens_cap != int(plan["max_gen_tokens_cap"]):
        raise ValueError("max_gen_tokens_cap does not match the dispatch plan")
    if str(args.data_root.resolve()) != str(plan["data_root"]):
        raise ValueError("data_root does not match the dispatch plan")
    if _file_sha256(args.data_root / "summary.json") != plan["data_summary_sha256"]:
        raise RuntimeError("Core data summary changed after dispatch planning")


def _worker_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    for name in ("data_root", "plan_root", "output_root"):
        payload[name] = str(payload[name])
    return payload


def _shared_run_manifest_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that must agree across all machine jobs."""

    contract = dict(payload)
    # Manifests written before the NCP batch-one padding fix did not record
    # these fields.  Their launcher default was 128, so normalize that exact
    # legacy contract while still rejecting a changed explicit value.
    contract.setdefault("pad_multiple", 128)
    contract.setdefault(
        "effective_score_pad_multiple",
        _effective_scoring_pad_multiple(
            int(contract["score_batch_size"]),
            int(contract["pad_multiple"]),
        ),
    )
    # Native vLLM overlays are intentionally machine-local.  Legacy manifests
    # recorded the first machine's path here, which made every other machine
    # fail the otherwise-identical shared contract.
    contract.pop("runtime_hf_model_path", None)
    return contract


def _write_or_validate_run_manifest(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    tasks: list[dict[str, Any]],
    model_identity_path: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "hf-core-native-pool-run-v1",
        "workflow_id": getattr(args, "workflow_id", "") or None,
        "repo_commit": getattr(args, "repo_commit", "") or None,
        "model_label": args.model_label,
        "hf_model_path": model_identity_path,
        "hf_backend_requested": args.hf_backend,
        "native_vllm_requested": args.hf_backend == NATIVE_VLLM_BACKEND,
        "native_megatron_requested": args.hf_backend == NATIVE_MEGATRON_BACKEND,
        "lmdeploy_requested": args.hf_backend == LMDEPLOY_BACKEND,
        "checkpoint_root": (
            str(Path(args.checkpoint_root).resolve())
            if args.hf_backend == NATIVE_MEGATRON_BACKEND
            else None
        ),
        "ckpt_step": args.ckpt_step if args.hf_backend == NATIVE_MEGATRON_BACKEND else None,
        "native_megatron_generation": (
            "full_prefix_recompute_no_kv_cache"
            if args.hf_backend == NATIVE_MEGATRON_BACKEND
            else None
        ),
        "native_vllm_config": (
            {
                "experimental": True,
                "model_family": args.vllm_model_family,
                "max_model_len": (
                    args.vllm_max_model_len if args.vllm_max_model_len > 0 else args.seq_length + 2
                ),
                "max_num_seqs": max(args.score_batch_size, args.generation_batch_size),
                "scheduler_queue_size": native_vllm_scheduler_queue_size(
                    args, max(args.score_batch_size, args.generation_batch_size)
                ),
                "execution_mode": args.vllm_execution_mode,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "attention_backend": args.vllm_attention_backend,
                "flash_attn_version": args.vllm_flash_attn_version,
                "hlm_attention_impl": args.vllm_hlm_attention_impl,
                "prefix_caching": False,
                **native_vllm_speculative_manifest(args),
            }
            if args.hf_backend == NATIVE_VLLM_BACKEND
            else None
        ),
        "lmdeploy_config": (
            lmdeploy_engine_policy(
                max_batch_size=max(
                    args.score_batch_size, args.generation_batch_size
                ),
                cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            )
            if args.hf_backend == LMDEPLOY_BACKEND
            else None
        ),
        "data_root": str(args.data_root.resolve()),
        "profile": args.profile,
        "data_summary_sha256": plan["data_summary_sha256"],
        "source_schema_version": plan["source_schema_version"],
        "task_orders": plan["task_orders"],
        "plan_id": plan["plan_id"],
        "dispatch_plan_schema": plan.get("schema_version"),
        "assignment_strategy": plan.get("assignment_strategy"),
        "cost_model": plan.get("cost_model"),
        "global_seed": args.global_seed,
        "machine_count": args.machine_count,
        "gpus_per_machine": args.gpus,
        "processes_per_gpu": args.processes_per_gpu,
        "global_worker_count": (args.machine_count * args.gpus * args.processes_per_gpu),
        "limit_per_task": args.limit_per_task,
        "generation_samples_cap": args.generation_samples_cap,
        "max_gen_tokens_cap": args.max_gen_tokens_cap,
        "score_batch_size": args.score_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "pad_multiple": args.pad_multiple,
        "effective_score_pad_multiple": _effective_scoring_pad_multiple(
            args.score_batch_size,
            args.pad_multiple,
        ),
        "local_dispatch": plan["local_dispatch"],
        "flash_decode": False,
        "expected_counts": plan["expected_counts"],
        "tasks": tasks,
    }
    path = args.output_root / "run_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _shared_run_manifest_contract(existing) != (_shared_run_manifest_contract(manifest)):
            raise RuntimeError(f"shared run manifest contract mismatch: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.machine-{args.machine_index:02d}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    written = json.loads(path.read_text(encoding="utf-8"))
    if _shared_run_manifest_contract(written) != (_shared_run_manifest_contract(manifest)):
        raise RuntimeError(f"shared run manifest race changed its contract: {path}")
    return written


def _maybe_aggregate(
    args: argparse.Namespace, plan: dict[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    lock_path = args.output_root / "aggregate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return _maybe_aggregate_locked(args, plan, tasks)


def _maybe_aggregate_locked(
    args: argparse.Namespace, plan: dict[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Aggregate once all machine markers exist while holding a file lock."""

    machine_results: list[dict[str, Any]] = []
    for machine_index in range(args.machine_count):
        path = args.output_root / f"machine-{machine_index:02d}" / "result.json"
        if not path.is_file():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "CORE_NATIVE_MACHINE_OK":
            return None
        expected_contract = {
            "machine_index": machine_index,
            "machine_count": args.machine_count,
            "model_label": args.model_label,
            "hf_model_path": _model_manifest_paths(args)[0],
            "profile": args.profile,
            "plan_id": plan["plan_id"],
            "dispatch_plan_schema": plan.get("schema_version"),
            "assignment_strategy": plan.get("assignment_strategy"),
            "cost_model": plan.get("cost_model"),
            "global_seed": args.global_seed,
            "processes_per_gpu": args.processes_per_gpu,
            "score_batch_size": args.score_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "local_dispatch": plan["local_dispatch"],
        }
        for name, expected in expected_contract.items():
            if result.get(name) != expected:
                raise RuntimeError(
                    f"machine result contract mismatch at {path}: "
                    f"{name}={result.get(name)!r} != {expected!r}"
                )
        machine_results.append(result)
    merged = _aggregate_prediction_files(
        args.output_root,
        tasks,
        {str(name): int(value) for name, value in plan["expected_counts"].items()},
    )
    merged.update(
        {
            "status": (
                "CORE_NATIVE_POOL_OK"
                if merged["score_task_count_complete"] == merged["task_count"]
                else (
                    "CORE_NATIVE_POOL_PREDICTIONS_OK"
                    if merged["prediction_task_count_complete"] == merged["task_count"]
                    else "CORE_NATIVE_POOL_INCOMPLETE"
                )
            ),
            "model_label": args.model_label,
            "hf_model_path": _model_manifest_paths(args)[0],
            "runtime_hf_model_path": _model_manifest_paths(args)[1],
            "profile": args.profile,
            "global_seed": args.global_seed,
            "machine_count": args.machine_count,
            "processes_per_gpu": args.processes_per_gpu,
            "score_batch_size": args.score_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "local_dispatch": plan["local_dispatch"],
            "worker_count": args.machine_count * args.gpus * args.processes_per_gpu,
            "plan_id": plan["plan_id"],
            "dispatch_plan_schema": plan.get("schema_version"),
            "assignment_strategy": plan.get("assignment_strategy"),
            "cost_model": plan.get("cost_model"),
            "machine_results": machine_results,
            "scores_csv": str(args.output_root / "scores.csv"),
        }
    )
    _write_json_atomic(args.output_root / "summary.json", merged)
    _write_scores_csv(args.output_root / "scores.csv", merged)
    if merged["status"] == "CORE_NATIVE_POOL_INCOMPLETE":
        raise RuntimeError(
            "pool prediction coverage is incomplete: "
            f"{merged['observed_predictions']}/{merged['expected_predictions']}"
        )
    return merged


def main() -> None:
    """Run the local coordinator and its independent GPU workers."""

    args = parse_args()
    process_started = time.perf_counter()
    plan, machine_items = load_machine_manifest(args.plan_root, args.machine_index)
    _validate_args(args, plan)
    tasks, materialized = _materialize_work(
        data_root=args.data_root, profile=args.profile, plan=plan, items=machine_items
    )
    if args.verify_data_sha256:
        for task in tasks:
            actual = _file_sha256(Path(task["absolute_file"]))
            if actual != task["sha256"]:
                raise RuntimeError(
                    f"dataset SHA256 mismatch for {task['task']}: {actual} != {task['sha256']}"
                )
    generation_tasks = {str(task["task"]) for task in tasks if _task_is_generation(task)}
    completed = (
        _completed_request_keys(args.output_root, generation_tasks) if args.resume else set()
    )
    pending_by_mode: dict[str, deque[WorkItem]] = {"generation": deque(), "loglikelihood": deque()}
    for item in machine_items:
        if item.request_key not in completed:
            pending_by_mode[item.mode].append(item)

    def pending_count() -> int:
        return sum(len(items) for items in pending_by_mode.values())

    machine_root = args.output_root / f"machine-{args.machine_index:02d}"
    machine_root.mkdir(parents=True, exist_ok=True)
    source_artifact_before = None
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        model_identity = Path(args.model_identity_path or args.hf_model_path).resolve()
        source_artifact_before = benchmark._hf_artifact_fingerprint(model_identity)
        overlay_dir = (
            Path(args.vllm_model_overlay_dir)
            if args.vllm_model_overlay_dir
            else machine_root / "native-vllm-model"
        )
        runtime_model, _ = prepare_native_vllm_model(
            source_model=args.hf_model_path,
            runtime_config=args.vllm_runtime_config or None,
            overlay_dir=overlay_dir,
            model_family=args.vllm_model_family,
        )
        args.model_identity_path = str(model_identity)
        args.hf_model_path = str(runtime_model)
    model_identity_path, runtime_hf_model_path = _model_manifest_paths(args)
    _write_or_validate_run_manifest(
        args=args, plan=plan, tasks=tasks, model_identity_path=model_identity_path
    )
    machine_manifest = {
        "schema_version": "core-native-machine-run-v1",
        "workflow_id": args.workflow_id or None,
        "repo_commit": args.repo_commit or None,
        "created_at": time.time(),
        "machine_index": args.machine_index,
        "machine_count": args.machine_count,
        "plan_id": plan["plan_id"],
        "dispatch_plan_schema": plan.get("schema_version"),
        "assignment_strategy": plan.get("assignment_strategy"),
        "cost_model": plan.get("cost_model"),
        "global_seed": args.global_seed,
        "model_label": args.model_label,
        "hf_model_path": model_identity_path,
        "runtime_hf_model_path": runtime_hf_model_path,
        "profile": args.profile,
        "gpus": args.gpus,
        "processes_per_gpu": args.processes_per_gpu,
        "local_worker_count": args.gpus * args.processes_per_gpu,
        "score_batch_size": args.score_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "local_dispatch": plan["local_dispatch"],
        "planned_work_items": len(machine_items),
        "resumed_work_items": len(machine_items) - pending_count(),
    }
    _write_json_atomic(machine_root / "run_manifest.json", machine_manifest)

    worker_count = args.gpus * args.processes_per_gpu
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=max(128, worker_count * 4))
    input_queues = [context.Queue(maxsize=1) for _ in range(worker_count)]
    processes: dict[int, Any] = {}
    restart_counts = [0] * worker_count
    worker_epochs = [0] * worker_count
    worker_metadata: dict[int, dict[str, Any]] = {}
    worker_results: dict[int, dict[str, Any]] = {}
    ready: set[int] = set()
    inflight: dict[int, dict[str, Any]] = {}
    stop_sent: set[int] = set()
    stopped: set[int] = set()
    errors: list[dict[str, Any]] = []
    completed_count = len(machine_items) - pending_count()
    writer = PredictionWriter(machine_root / "predictions")

    def start_worker(worker_id: int, *, replace_queue: bool = False) -> None:
        if replace_queue:
            input_queues[worker_id] = context.Queue(maxsize=1)
        process = context.Process(
            target=_worker_main,
            args=(
                worker_id,
                worker_epochs[worker_id],
                worker_count,
                args.processes_per_gpu,
                _worker_args_payload(args),
                input_queues[worker_id],
                result_queue,
            ),
            name=f"core-worker-{worker_id:02d}",
        )
        process.start()
        processes[worker_id] = process

    def assign_ready_workers() -> None:
        while ready and pending_count() and not errors:
            worker_id = min(ready)
            ready.remove(worker_id)
            available_modes = [mode for mode, items in pending_by_mode.items() if items]
            mode = max(available_modes, key=lambda name: pending_by_mode[name][0].estimated_cost)
            batch_limit = (
                native_vllm_scheduler_queue_size(args, args.generation_batch_size)
                if mode == "generation" and args.hf_backend == NATIVE_VLLM_BACKEND
                else (
                    args.generation_batch_size if mode == "generation" else args.score_batch_size
                )
            )
            works = [
                pending_by_mode[mode].popleft()
                for _ in range(min(batch_limit, len(pending_by_mode[mode])))
            ]
            items: list[dict[str, Any]] = []
            for work in works:
                item_payload = dict(materialized[work.request_key])
                if work.mode == "generation":
                    item_payload["sample_seed"] = stable_request_seed(
                        args.global_seed,
                        work.task,
                        item_payload["row"]["example_id"],
                        int(work.sample_index),
                    )
                items.append(item_payload)
            payload = {"command": "RUN_BATCH", "mode": mode, "items": items}
            inflight[worker_id] = payload
            input_queues[worker_id].put(payload)

    def stop_idle_workers() -> None:
        if pending_count() or inflight:
            return
        for worker_id in sorted(ready - stop_sent):
            input_queues[worker_id].put({"command": "STOP"})
            stop_sent.add(worker_id)
            ready.remove(worker_id)

    try:
        for worker_id in range(worker_count):
            start_worker(worker_id)
            if args.worker_start_stagger_seconds > 0:
                time.sleep(args.worker_start_stagger_seconds)
        while len(stopped) < worker_count:
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                event = None
            if event is not None:
                worker_id = int(event["worker_id"])
                event_epoch = int(event.get("worker_epoch", -1))
                if event_epoch != worker_epochs[worker_id]:
                    continue
                event_type = str(event["event"])
                if event_type == "READY":
                    worker_metadata[worker_id] = event
                    ready.add(worker_id)
                elif event_type == "STARTED":
                    expected = inflight.get(worker_id)
                    expected_keys = (
                        []
                        if expected is None
                        else [item["work"]["request_key"] for item in expected["items"]]
                    )
                    if expected_keys != event["request_keys"]:
                        errors.append(
                            {
                                "worker_id": worker_id,
                                "error": "worker started an unexpected request batch",
                                "event": event,
                            }
                        )
                elif event_type == "RECORD":
                    payload = inflight.pop(worker_id, None)
                    expected_keys = (
                        []
                        if payload is None
                        else [item["work"]["request_key"] for item in payload["items"]]
                    )
                    if expected_keys != event["request_keys"]:
                        errors.append(
                            {
                                "worker_id": worker_id,
                                "error": "worker returned an unexpected request batch",
                                "event": event,
                            }
                        )
                    else:
                        duplicate_keys = [key for key in expected_keys if key in completed]
                        if duplicate_keys:
                            errors.append(
                                {
                                    "worker_id": worker_id,
                                    "error": (
                                        "worker returned already completed requests: "
                                        f"{duplicate_keys[:4]}"
                                    ),
                                    "event": event,
                                }
                            )
                            continue
                        if len(event["records"]) != len(expected_keys):
                            errors.append(
                                {
                                    "worker_id": worker_id,
                                    "error": (
                                        "worker returned a record-count mismatch: "
                                        f"{len(event['records'])} != {len(expected_keys)}"
                                    ),
                                    "event": event,
                                }
                            )
                            continue
                        record_keys = [
                            _prediction_request_key(record, generation_tasks)
                            for record in event["records"]
                        ]
                        if record_keys != expected_keys:
                            errors.append(
                                {
                                    "worker_id": worker_id,
                                    "error": "worker record identities changed within a batch",
                                    "expected_keys": expected_keys,
                                    "record_keys": record_keys,
                                }
                            )
                            continue
                        for key, record in zip(expected_keys, event["records"], strict=True):
                            writer.append(record)
                            completed.add(str(key))
                        completed_count += len(expected_keys)
                        ready.add(worker_id)
                        if completed_count % max(
                            1, args.progress_every
                        ) == 0 or completed_count == len(machine_items):
                            _write_json_atomic(
                                machine_root / "progress.json",
                                {
                                    "status": (
                                        "RUNNING"
                                        if completed_count < len(machine_items)
                                        else "COMPLETE"
                                    ),
                                    "completed": completed_count,
                                    "total": len(machine_items),
                                    "pending": pending_count(),
                                    "inflight_batches": len(inflight),
                                    "inflight_requests": sum(
                                        len(payload["items"]) for payload in inflight.values()
                                    ),
                                    "updated_at": time.time(),
                                },
                            )
                elif event_type == "STOPPED":
                    worker_results[worker_id] = event
                    stopped.add(worker_id)
                    inflight.pop(worker_id, None)
                    ready.discard(worker_id)
                    if event["artifact_mutated"]:
                        errors.append(
                            {"worker_id": worker_id, "error": "runtime HF model artifact changed"}
                        )
                elif event_type == "ERROR":
                    errors.append(event)
                    inflight.pop(worker_id, None)
                    ready.discard(worker_id)
            assign_ready_workers()
            stop_idle_workers()

            for worker_id, process in list(processes.items()):
                if process.is_alive() or worker_id in stopped:
                    continue
                if worker_id in stop_sent and process.exitcode == 0:
                    continue
                payload = inflight.pop(worker_id, None)
                if payload is not None:
                    for item_payload in reversed(payload["items"]):
                        work = WorkItem.from_json(item_payload["work"])
                        pending_by_mode[work.mode].appendleft(work)
                if restart_counts[worker_id] < args.worker_restarts and not errors:
                    process.join(timeout=0)
                    restart_counts[worker_id] += 1
                    worker_epochs[worker_id] += 1
                    start_worker(worker_id, replace_queue=True)
                elif not any(error.get("worker_id") == worker_id for error in errors):
                    errors.append(
                        {
                            "worker_id": worker_id,
                            "error": f"worker exited with code {process.exitcode}",
                        }
                    )
                    stopped.add(worker_id)
            if errors:
                for worker_id in sorted(ready - stop_sent):
                    input_queues[worker_id].put({"command": "STOP"})
                    stop_sent.add(worker_id)
                    ready.remove(worker_id)
                if not inflight:
                    for worker_id, process in processes.items():
                        if process.is_alive() and worker_id not in stop_sent:
                            process.terminate()
                    break
    finally:
        writer.close()
        for process in processes.values():
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    source_artifact_after = (
        benchmark._hf_artifact_fingerprint(model_identity_path)
        if source_artifact_before is not None
        else None
    )
    source_artifact_mutated = (
        source_artifact_before != source_artifact_after
        if source_artifact_before is not None
        else False
    )
    if source_artifact_mutated:
        errors.append(
            {"worker_id": None, "error": "source HF model artifact changed during evaluation"}
        )
    if errors:
        _write_json_atomic(
            machine_root / "result.json",
            {
                "status": "CORE_NATIVE_MACHINE_FAILED",
                "machine_index": args.machine_index,
                "completed_work_items": completed_count,
                "planned_work_items": len(machine_items),
                "errors": errors,
                "finished_at": time.time(),
            },
        )
        raise RuntimeError(f"Core worker pool failed: {errors[0]}")
    if completed_count != len(machine_items):
        raise RuntimeError(f"machine coverage incomplete: {completed_count}/{len(machine_items)}")
    machine_result = {
        "status": "CORE_NATIVE_MACHINE_OK",
        "machine_index": args.machine_index,
        "machine_count": args.machine_count,
        "model_label": args.model_label,
        "hf_model_path": model_identity_path,
        "profile": args.profile,
        "plan_id": plan["plan_id"],
        "dispatch_plan_schema": plan.get("schema_version"),
        "assignment_strategy": plan.get("assignment_strategy"),
        "cost_model": plan.get("cost_model"),
        "global_seed": args.global_seed,
        "planned_work_items": len(machine_items),
        "completed_work_items": completed_count,
        "local_worker_count": worker_count,
        "processes_per_gpu": args.processes_per_gpu,
        "score_batch_size": args.score_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "local_dispatch": plan["local_dispatch"],
        "worker_restarts": restart_counts,
        "worker_metadata": worker_metadata,
        "worker_results": worker_results,
        "artifact_mutated": any(
            bool(result.get("artifact_mutated")) for result in worker_results.values()
        ),
        "source_artifact_mutated": source_artifact_mutated,
        "source_artifact_metadata_before": source_artifact_before,
        "source_artifact_metadata_after": source_artifact_after,
        "process_wall_seconds": time.perf_counter() - process_started,
        "finished_at": time.time(),
    }
    _write_json_atomic(machine_root / "result.json", machine_result)
    _maybe_aggregate(args, plan, tasks)


if __name__ == "__main__":
    main()
