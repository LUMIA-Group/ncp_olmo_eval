#!/usr/bin/env python3
"""Run sealed RULER or LongBench v2 prompts through one of four backends."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from . import benchmark
from .core_native_eval import (
    _initialize_distributed_process_group,
    _isolate_compile_caches,
    _model_args,
    _resolve_hf_backend,
)
from .inference import SamplingParams
from .lmdeploy_inference import (
    LMDEPLOY_BACKEND,
    LMDeployInferencer,
    add_lmdeploy_args,
    validate_lmdeploy_args,
)
from .long_context_protocol import (
    HELMET_BENCHMARK,
    HELMET_SEED,
    PREDICTION_STATUS,
    RULER_BENCHMARK,
    SCHEMA_VERSION,
    SHARD_STATUS,
    benchmark_sampling_contract,
    canonical_json_sha256,
    file_sha256,
    model_context_capability,
    prepared_official_sampling_contract,
    resolve_model_context_contract,
    stable_sample_seed,
    tokenizer_artifact_contract,
    validate_backend_shape,
    validate_prepared_dataset,
    write_json_atomic,
)
from .native_megatron_inference import NATIVE_MEGATRON_BACKEND, build_native_megatron_inferencer
from .native_vllm_inference import (
    NATIVE_VLLM_BACKEND,
    NativeVLLMInferencer,
    add_native_vllm_args,
    native_vllm_speculative_kwargs,
    prepare_native_vllm_model,
    validate_native_vllm_args,
)
from .transformers_inference import TransformersInferencer


def parse_args() -> argparse.Namespace:
    """Parse one long-context inference shard."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--backend",
        choices=("hf", NATIVE_MEGATRON_BACKEND, NATIVE_VLLM_BACKEND, LMDEPLOY_BACKEND),
        required=True,
    )
    parser.add_argument("--hf-model-path", default="")
    parser.add_argument("--model-identity-path", default="")
    parser.add_argument(
        "--model-config-path",
        default="",
        help=(
            "HF model directory or config.json used to verify the native context "
            "window. Required for every backend; defaults to the identity/HF model."
        ),
    )
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--ckpt-step", type=int, default=0)
    parser.add_argument("--train-wandb-config", default="")
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-example", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=HELMET_SEED)
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--shard-count", type=int, default=-1)
    parser.add_argument(
        "--ruler-sequence-length",
        type=int,
        default=0,
        help="Select one prepared RULER length; 0 keeps every prepared length.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Keep the first N sealed examples of every task; intended for smoke tests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--model-context-length", type=int, default=0)
    parser.add_argument(
        "--allow-context-extension",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly allow runtime context beyond the source model config. "
            "This is recorded as unvalidated extrapolation and never inferred."
        ),
    )
    parser.add_argument("--hf-compile-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hf-align-dcp-runtime-config", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_native_vllm_args(parser)
    parser.add_argument(
        "--hf-model-parallel-size",
        type=int,
        default=1,
        help=(
            "Visible GPUs used by one standard Transformers model. Values above "
            "one use a single process with explicit model sharding."
        ),
    )
    add_lmdeploy_args(parser)
    return parser.parse_args()


def _distributed_shard(args: argparse.Namespace) -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0")) if args.shard_index < 0 else args.shard_index
    world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if args.shard_count < 0 else args.shard_count
    )
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid shard index/count: {rank}/{world_size}")
    return rank, world_size


def _uses_torchrun_process_group(args: argparse.Namespace) -> bool:
    """Return whether this shard participates in a torchrun process group."""

    return args.backend == NATIVE_MEGATRON_BACKEND or (
        args.backend == "hf" and args.shard_index < 0
    )


def _row_max_new_tokens(
    row: dict[str, Any], max_new_tokens: int, max_new_tokens_by_task: dict[str, int]
) -> int:
    """Resolve a sealed row budget before falling back to task defaults."""

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        sealed = metadata.get("max_new_tokens")
        if isinstance(sealed, int) and not isinstance(sealed, bool) and sealed > 0:
            return sealed
    return int(max_new_tokens_by_task.get(str(row["task"]), max_new_tokens))


def _runtime_context_contract(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int,
    max_new_tokens_by_task: dict[str, int],
    benchmark_name: str = "",
) -> dict[str, Any]:
    config_source = args.model_config_path or args.model_identity_path or args.hf_model_path
    if not config_source:
        raise ValueError(
            "--model-config-path is required when neither --model-identity-path "
            "nor --hf-model-path identifies the model config"
        )
    config_path = Path(config_source).resolve()
    if config_path.is_dir():
        config_path /= "config.json"
    capability = model_context_capability(config_path)
    prompt_token_counts: list[int] = []
    count_sources: set[str] = set()
    for row in rows:
        truncation = row.get("metadata", {}).get("truncation", {})
        sealed_count = truncation.get("prepared_token_count")
        if isinstance(sealed_count, int) and not isinstance(sealed_count, bool):
            if sealed_count <= 0:
                raise ValueError(
                    f"invalid sealed prompt token count for {row.get('example_id')}: "
                    f"{sealed_count}"
                )
            prompt_token_counts.append(sealed_count)
            count_sources.add("sealed_prepared_metadata")
        else:
            prompt_token_counts.append(len(tokenizer.encode(str(row["prompt"]))))
            count_sources.add("runtime_tokenizer_fallback")
    required = max(
        prompt_token_count + _row_max_new_tokens(row, max_new_tokens, max_new_tokens_by_task)
        for row, prompt_token_count in zip(rows, prompt_token_counts, strict=True)
    )
    contract = resolve_model_context_contract(
        capability=capability,
        required_context_length=required,
        requested_context_length=int(args.model_context_length),
        allow_context_extension=bool(args.allow_context_extension),
        # OLMES treats the named RULER length as prompt length and gives each
        # task/length pair its own output budget. Permit only the
        # request-derived overhang, bounded by the protocol-wide maximum; this
        # must never become a general context extension.
        max_generation_overhang=(int(max_new_tokens) if benchmark_name == RULER_BENCHMARK else 0),
    )
    contract["prompt_token_count_sources"] = sorted(count_sources)
    if args.backend != NATIVE_MEGATRON_BACKEND:
        runtime_config = Path(args.hf_model_path).resolve() / "config.json"
        runtime_capability = model_context_capability(runtime_config)
        runtime_length = int(contract["runtime_context_length"])
        if (
            int(runtime_capability["native_context_length"]) < runtime_length
            and not contract["bounded_generation_overhang"]
        ):
            raise ValueError(
                "runtime HF model config does not declare the requested context: "
                f"{runtime_capability['native_context_length']} < {runtime_length}; "
                "create an explicit long-context model overlay"
            )
        contract["runtime_model_config"] = runtime_capability
    return contract


def _apply_native_vllm_context_policy(context_contract: dict[str, Any]) -> None:
    """Lift vLLM's config ceiling only for a validated context contract."""

    if not context_contract.get("uses_context_extrapolation"):
        return
    if not (
        context_contract.get("bounded_generation_overhang")
        or context_contract.get("allow_context_extension")
    ):
        raise ValueError("vLLM context extrapolation lacks an explicit validated policy")
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def _hf_resolution_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(hf_backend="auto", hf_model_path=args.hf_model_path)


def _backend_model_args(
    args: argparse.Namespace, resolved_backend: str, model_context_length: int
) -> argparse.Namespace:
    bridge = argparse.Namespace(
        checkpoint_root=args.checkpoint_root,
        ckpt_step=args.ckpt_step,
        train_wandb_config=args.train_wandb_config,
        tokenizer_model=str(args.tokenizer_model),
        hf_model_path=args.hf_model_path,
        hf_compile_routes=bool(args.hf_compile_routes),
        hf_align_dcp_runtime_config=bool(args.hf_align_dcp_runtime_config),
        allow_context_extension=bool(args.allow_context_extension),
        seq_length=model_context_length,
        score_batch_size=1,
        generation_batch_size=args.batch_size,
        hf_model_parallel_size=int(getattr(args, "hf_model_parallel_size", 1)),
    )
    return _model_args(bridge, resolved_backend)


def _model_identity_args(
    args: argparse.Namespace, resolved_backend: str, model_context_length: int
) -> argparse.Namespace:
    if resolved_backend == NATIVE_MEGATRON_BACKEND:
        return _backend_model_args(args, resolved_backend, model_context_length)
    identity = args.model_identity_path or args.hf_model_path
    return argparse.Namespace(model_source="hf", hf_model_path=identity)


def _load_contract_tokenizer(tokenizer_model: Path) -> Any:
    """Load exported tokenizers across the HF and vLLM Transformers versions.

    Recent Transformers versions serialize the generic fast tokenizer class as
    ``TokenizersBackend``.  The vLLM environment intentionally uses an older
    compatible Transformers release which does not expose that AutoTokenizer
    class name, although it can read the same tokenizer.json directly.
    """

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    kwargs = {"trust_remote_code": True, "local_files_only": True, "use_fast": True}
    resolved = str(tokenizer_model.resolve())
    try:
        return AutoTokenizer.from_pretrained(resolved, **kwargs)
    except ValueError as error:
        if "Tokenizer class TokenizersBackend does not exist" not in str(error):
            raise
    return PreTrainedTokenizerFast.from_pretrained(resolved, local_files_only=True)


def _build_inferencer(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    max_new_tokens_by_task: dict[str, int],
    benchmark_name: str = "",
) -> tuple[Any, Any, str, dict[str, Any], argparse.Namespace, dict[str, Any]]:
    if args.backend == "hf":
        if not args.hf_model_path:
            raise ValueError("--hf-model-path is required for the hf backend")
        resolved_backend = _resolve_hf_backend(_hf_resolution_args(args))
    else:
        resolved_backend = str(args.backend)

    if resolved_backend == NATIVE_MEGATRON_BACKEND:
        if not args.checkpoint_root or args.ckpt_step <= 0:
            raise ValueError("native_megatron requires --checkpoint-root and positive --ckpt-step")
    elif not args.hf_model_path:
        raise ValueError(f"--hf-model-path is required for {resolved_backend}")

    contract_tokenizer = _load_contract_tokenizer(args.tokenizer_model)
    context_contract = _runtime_context_contract(
        args, rows, contract_tokenizer, max_new_tokens, max_new_tokens_by_task, benchmark_name
    )
    model_context_length = int(context_contract["runtime_context_length"])
    identity_args = _model_identity_args(args, resolved_backend, model_context_length)
    artifact_before = benchmark._artifact_fingerprint(identity_args)

    if resolved_backend == NATIVE_VLLM_BACKEND:
        _apply_native_vllm_context_policy(context_contract)
        validation_args = argparse.Namespace(**vars(args), hf_backend=resolved_backend)
        validate_native_vllm_args(validation_args, batch_size=args.batch_size, processes_per_gpu=1)
        overlay_dir = (
            Path(args.vllm_model_overlay_dir)
            if args.vllm_model_overlay_dir
            else args.output_root / "native-vllm-model"
        )
        runtime_model, overlay_manifest = prepare_native_vllm_model(
            source_model=args.hf_model_path,
            runtime_config=args.vllm_runtime_config or None,
            overlay_dir=overlay_dir,
            model_family=args.vllm_model_family,
            **native_vllm_speculative_kwargs(args),
        )
        inferencer = NativeVLLMInferencer(
            model_path=runtime_model,
            max_model_len=model_context_length,
            seed=args.global_seed,
            max_batch_size=args.batch_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            execution_mode=args.vllm_execution_mode,
            attention_backend=args.vllm_attention_backend,
            flash_attn_version=args.vllm_flash_attn_version,
            hlm_attention_impl=args.vllm_hlm_attention_impl,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            model_family=args.vllm_model_family,
        )
        runtime = {
            **inferencer.runtime_metadata,
            "model_overlay": overlay_manifest,
            "model_context_contract": context_contract,
        }
    elif resolved_backend == LMDEPLOY_BACKEND:
        validation_args = argparse.Namespace(**vars(args), hf_backend=resolved_backend)
        validate_lmdeploy_args(validation_args, batch_size=args.batch_size, processes_per_gpu=1)
        inferencer = LMDeployInferencer(
            model_path=args.hf_model_path,
            seed=args.global_seed,
            max_batch_size=args.batch_size,
            cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            log_level=args.lmdeploy_log_level,
        )
        if inferencer.session_len < model_context_length:
            raise ValueError(
                "LMDeploy default session length is smaller than the required context: "
                f"{inferencer.session_len} < {model_context_length}"
            )
        runtime = {**inferencer.runtime_metadata, "model_context_contract": context_contract}
    else:
        model_args = _backend_model_args(args, resolved_backend, model_context_length)
        eval_model = benchmark.load_eval_model(model_args)
        if resolved_backend == NATIVE_MEGATRON_BACKEND:
            inferencer = build_native_megatron_inferencer(eval_model)
        else:
            inferencer = TransformersInferencer(eval_model, max_batch_size=1)
        runtime = {
            "backend": args.backend,
            "resolved_backend": resolved_backend,
            "batch_size": 1,
            "hf_model_parallel_size": (
                int(getattr(eval_model, "hf_model_parallel_size", 1))
                if resolved_backend == "transformers"
                else 1
            ),
            "hf_device_map": (
                dict(getattr(eval_model, "hf_device_map", {}) or {})
                if resolved_backend == "transformers"
                else {}
            ),
            "attention_implementation": str(
                getattr(eval_model, "attention_implementation", "unknown")
            ),
            "model_context_length": model_context_length,
            "model_context_contract": context_contract,
            "native_megatron_full_prefix_recompute": (resolved_backend == NATIVE_MEGATRON_BACKEND),
        }
    return (
        inferencer,
        inferencer.tokenizer,
        resolved_backend,
        runtime,
        identity_args,
        artifact_before,
    )


def _completed_keys(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid resume JSON at {path}:{line_number}") from error
            key = (str(row.get("example_id")), int(row.get("sample_index", -1)))
            if key in completed:
                raise RuntimeError(f"duplicate resume key at {path}:{line_number}: {key}")
            completed.add(key)
    return completed


def _select_rows(
    args: argparse.Namespace, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = rows
    sequence_length = int(getattr(args, "ruler_sequence_length", 0))
    if sequence_length < 0:
        raise ValueError("--ruler-sequence-length must be non-negative")
    if sequence_length:
        if manifest.get("benchmark") != "ruler":
            raise ValueError("--ruler-sequence-length is only valid for RULER")
        selected = [
            row for row in selected if int(row["metadata"]["sequence_length"]) == sequence_length
        ]
        if not selected:
            raise ValueError(f"prepared RULER data does not contain length {sequence_length}")
    if args.limit > 0:
        selected = selected[: args.limit]
    limit_per_task = int(getattr(args, "limit_per_task", 0))
    if limit_per_task < 0:
        raise ValueError("--limit-per-task must be non-negative")
    if limit_per_task:
        counts: dict[str, int] = {}
        stratified = []
        for row in selected:
            task = str(row["task"])
            if counts.get(task, 0) >= limit_per_task:
                continue
            stratified.append(row)
            counts[task] = counts.get(task, 0) + 1
        selected = stratified
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one deterministic shard and preserve every generated sample."""

    if int(args.global_seed) != HELMET_SEED:
        raise ValueError(
            f"long-context evaluation global seed is fixed to {HELMET_SEED}, "
            f"got {args.global_seed}"
        )
    validate_backend_shape(args.backend, args.batch_size, args.samples_per_example)
    manifest, rows = validate_prepared_dataset(args.data_root.resolve())
    rows = _select_rows(args, manifest, rows)
    if not rows:
        raise ValueError("selected prepared dataset is empty")
    rank, world_size = _distributed_shard(args)
    distributed_initialized = False
    # Native Megatron is launched by torchrun and must join one process group.
    # Multi-GPU HF instead launches several independent model-parallel engines
    # with explicit shard indices.  Those engines partition evaluation work but
    # do not form a distributed model, so making them join a shard-count-sized
    # process group deadlocks each engine on its private rendezvous port.
    if _uses_torchrun_process_group(args):
        distributed_initialized = _initialize_distributed_process_group(rank, world_size)
    if args.backend in {"hf", NATIVE_MEGATRON_BACKEND}:
        _isolate_compile_caches(rank, world_size)
    sampling_defaults = benchmark_sampling_contract(str(manifest["benchmark"]))
    manifest_protocol = manifest.get("protocol", {})
    for key in (
        "max_new_tokens",
        "max_new_tokens_by_task",
        "max_new_tokens_by_sequence_length",
        "temperature",
        "top_p",
        "stop",
        "stop_by_task",
    ):
        if key in manifest_protocol:
            sampling_defaults[key] = manifest_protocol[key]
    max_new_tokens_override = int(args.max_new_tokens) if args.max_new_tokens > 0 else None
    max_new_tokens = int(max_new_tokens_override or sampling_defaults["max_new_tokens"])
    max_new_tokens_by_task = {
        str(task): int(value)
        for task, value in sampling_defaults.get("max_new_tokens_by_task", {}).items()
    }
    stop_by_task = {
        str(task): [str(stop) for stop in stops]
        for task, stops in sampling_defaults.get("stop_by_task", {}).items()
    }
    temperature = (
        float(sampling_defaults["temperature"])
        if args.temperature is None
        else float(args.temperature)
    )
    top_p = float(sampling_defaults["top_p"]) if args.top_p is None else float(args.top_p)
    if max_new_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling override")

    shard_root = args.output_root.resolve() / f"shard-{rank:03d}-of-{world_size:03d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    prediction_path = shard_root / "predictions.jsonl"
    if prediction_path.exists() and not args.resume:
        raise FileExistsError(f"predictions already exist: {prediction_path}")
    completed = _completed_keys(prediction_path)
    work = [
        {
            "row": row,
            "sample_index": sample_index,
            "seed": stable_sample_seed(args.global_seed, str(row["example_id"]), sample_index),
        }
        for row in rows
        for sample_index in range(args.samples_per_example)
    ]
    selected = [item for index, item in enumerate(work) if index % world_size == rank]
    expected_keys = {
        (str(item["row"]["example_id"]), int(item["sample_index"])) for item in selected
    }
    if not completed <= expected_keys:
        raise RuntimeError("resume file contains predictions outside this shard")

    tokenizer_contract = tokenizer_artifact_contract(args.tokenizer_model.resolve())
    if tokenizer_contract["contract_sha256"] != manifest["tokenizer"]["contract_sha256"]:
        raise RuntimeError("runtime tokenizer does not match prepared tokenizer contract")

    load_started = time.perf_counter()
    (inferencer, tokenizer, resolved_backend, runtime, identity_args, artifact_before) = (
        _build_inferencer(
            args, rows, max_new_tokens, max_new_tokens_by_task, str(manifest["benchmark"])
        )
    )
    load_seconds = time.perf_counter() - load_started
    pending = [
        item
        for item in selected
        if (str(item["row"]["example_id"]), int(item["sample_index"])) not in completed
    ]
    generated_count = 0
    started = time.perf_counter()
    with prediction_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [str(item["row"]["prompt"]) for item in batch]
            samplings = [
                SamplingParams(
                    max_tokens=int(
                        max_new_tokens_override
                        or _row_max_new_tokens(item["row"], max_new_tokens, max_new_tokens_by_task)
                    ),
                    temperature=temperature,
                    top_p=top_p,
                    stop=list(
                        stop_by_task.get(str(item["row"]["task"]), sampling_defaults["stop"])
                    ),
                    seed=int(item["seed"]),
                )
                for item in batch
            ]
            if len(batch) > 1:
                generate_batch = getattr(inferencer, "generate_batch", None)
                if not callable(generate_batch):
                    raise RuntimeError(f"{args.backend} does not support batched sampling")
                completions = generate_batch(prompts, samplings)
            else:
                completions = inferencer.generate(prompts, samplings[0])
            if len(completions) != len(batch):
                raise RuntimeError("backend returned an invalid completion count")
            for item, completion in zip(batch, completions, strict=True):
                row = item["row"]
                row_max_new_tokens = int(
                    max_new_tokens_override
                    or _row_max_new_tokens(row, max_new_tokens, max_new_tokens_by_task)
                )
                row_stop = list(stop_by_task.get(str(row["task"]), sampling_defaults["stop"]))
                runtime_prompt_count = len(tokenizer.encode(str(row["prompt"])))
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": PREDICTION_STATUS,
                    "benchmark": manifest["benchmark"],
                    "example_id": row["example_id"],
                    "task": row["task"],
                    "sample_index": int(item["sample_index"]),
                    "samples_per_example": int(args.samples_per_example),
                    "sample_seed": int(item["seed"]),
                    "prompt_sha256": row["prompt_sha256"],
                    "runtime_prompt_token_count": runtime_prompt_count,
                    "expected_answer": row["expected_answer"],
                    "metadata": row["metadata"],
                    "generation": str(completion.text),
                    "generated_token_ids": [int(value) for value in completion.token_ids],
                    "finish_reason": str(completion.finish_reason),
                    "backend": args.backend,
                    "resolved_backend": resolved_backend,
                    "shard_index": rank,
                    "shard_count": world_size,
                    "sampling": {
                        "max_new_tokens": row_max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop": row_stop,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                generated_count += 1
            if (
                resolved_backend == "transformers"
                and os.environ.get("HF_EMPTY_CACHE_BETWEEN_EXAMPLES", "0") == "1"
            ):
                # Release the previous generate() graph/cache before the next
                # long prompt. This is intentionally opt-in because allocator
                # cache reuse is faster for ordinary short-context workloads.
                del completions
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()

    close = getattr(inferencer, "close", None)
    if callable(close):
        close()
    artifact_after = benchmark._artifact_fingerprint(identity_args)
    final_keys = _completed_keys(prediction_path)
    if final_keys != expected_keys:
        raise RuntimeError(
            "shard prediction coverage is incomplete: " f"{len(final_keys)}/{len(expected_keys)}"
        )
    official_defaults = prepared_official_sampling_contract(manifest)
    effective_sampling_matches_official = (
        max_new_tokens == int(official_defaults["max_new_tokens"])
        and max_new_tokens_by_task
        == {
            str(task): int(value)
            for task, value in official_defaults.get("max_new_tokens_by_task", {}).items()
        }
        and sampling_defaults.get("max_new_tokens_by_sequence_length", {})
        == official_defaults.get("max_new_tokens_by_sequence_length", {})
        and list(sampling_defaults["stop"]) == list(official_defaults["stop"])
        and stop_by_task
        == {
            str(task): [str(stop) for stop in stops]
            for task, stops in official_defaults.get("stop_by_task", {}).items()
        }
        and sampling_defaults.get("eos_stopping", True) is True
        and official_defaults.get("eos_stopping", True) is True
    )
    official_compatible = (
        args.samples_per_example == 1
        and max_new_tokens_override is None
        and effective_sampling_matches_official
        and temperature == float(official_defaults["temperature"])
        and top_p == float(official_defaults["top_p"])
        and args.limit == 0
        and int(getattr(args, "limit_per_task", 0)) == 0
        and int(getattr(args, "ruler_sequence_length", 0)) == 0
        and manifest.get("protocol", {}).get("formal_data_validation_passed") is True
        and (manifest["benchmark"] != HELMET_BENCHMARK or (int(args.global_seed) == HELMET_SEED))
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": SHARD_STATUS,
        "benchmark": manifest["benchmark"],
        "prepared_manifest_sha256": file_sha256(args.data_root / "manifest.json"),
        "prepared_inputs_sha256": file_sha256(args.data_root / "inputs.jsonl"),
        "model_label": args.model_label,
        "backend": args.backend,
        "resolved_backend": resolved_backend,
        "runtime": runtime,
        "batch_size": args.batch_size,
        "samples_per_example": args.samples_per_example,
        "global_seed": args.global_seed,
        "sampling": {
            "max_new_tokens": max_new_tokens,
            "max_new_tokens_override": max_new_tokens_override,
            "max_new_tokens_by_task": max_new_tokens_by_task,
            "max_new_tokens_by_sequence_length": sampling_defaults.get(
                "max_new_tokens_by_sequence_length", {}
            ),
            "eos_stopping": bool(sampling_defaults.get("eos_stopping", True)),
            "temperature": temperature,
            "top_p": top_p,
            "stop": list(sampling_defaults["stop"]),
            "stop_by_task": stop_by_task,
        },
        "official_protocol_compatible": official_compatible,
        "selection": {
            "ruler_sequence_length": int(getattr(args, "ruler_sequence_length", 0)),
            "limit": int(args.limit),
            "limit_per_task": int(getattr(args, "limit_per_task", 0)),
        },
        "shard_index": rank,
        "shard_count": world_size,
        "expected_prediction_count": len(expected_keys),
        "prediction_count": len(final_keys),
        "prediction_keys_sha256": canonical_json_sha256(sorted(final_keys)),
        "new_prediction_count": generated_count,
        "model_load_seconds": load_seconds,
        "generation_seconds": time.perf_counter() - started,
        "artifact_mutated": artifact_before != artifact_after,
        "artifact_before": artifact_before,
        "artifact_after": artifact_after,
        "tokenizer_contract": tokenizer_contract,
    }
    write_json_atomic(shard_root / "result.json", result)
    if distributed_initialized:
        import torch

        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


def main() -> None:
    """Execute one inference shard."""

    result = run(parse_args())
    print(f"status={result['status']}")
    print("predictions=" f"{result['prediction_count']}/{result['expected_prediction_count']}")


if __name__ == "__main__":
    main()
