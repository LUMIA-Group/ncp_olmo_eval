#!/usr/bin/env python3
"""Evaluate ConceptLM with the fixed OLMo Stage-1 paper GSM8K protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import benchmark
from .core_native_answer_eval import OFFICIAL_OLMO_EVAL_COMMIT, score_gsm_answer
from .inference import ConceptLMInferencer, SamplingParams
from .lm_eval_runtime import runtime_identity, task_manager_class
from .lmdeploy_inference import (
    LMDEPLOY_BACKEND,
    LMDeployInferencer,
    add_lmdeploy_args,
    validate_lmdeploy_args,
)
from .native_vllm_inference import (
    NATIVE_VLLM_BACKEND,
    NativeVLLMInferencer,
    add_native_vllm_args,
    native_vllm_max_model_len,
    native_vllm_speculative_kwargs,
    prepare_native_vllm_model,
    validate_native_vllm_args,
)
from .transformers_inference import TransformersInferencer

STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE_ANSWER_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]
EVALUATION_SEED = 42
STANDARD_TASK_GROUP = "olmo_eval_paper_math_gsm_8shot"
STANDARD_TASK_NAME = "olmo_eval_paper_gsm8k_main"
STANDARD_NUM_FEWSHOT = 8
STANDARD_TEMPERATURE = 0.6
STANDARD_TOP_P = 0.6
STANDARD_MAX_GEN_TOKS = 512
STANDARD_REPEATS = 1
PROMPT_TRANSPORTS = ("raw_completion", "chat_template")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-source",
        choices=("dcp", "hf"),
        default="dcp",
        help="Load either the original distributed checkpoint or HF SafeTensors.",
    )
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--ckpt-step", type=int, default=1382310)
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--train-wandb-config", default="")
    parser.add_argument(
        "--hf-model-path",
        default="",
        help="Read-only HF model directory used when --model-source=hf.",
    )
    parser.add_argument(
        "--hf-backend",
        choices=(
            "conceptlm",
            "transformers",
            "from_pretrained",
            NATIVE_VLLM_BACKEND,
            LMDEPLOY_BACKEND,
        ),
        default="conceptlm",
        help=(
            "Use the custom ConceptLM SafeTensors runtime, standard "
            "Transformers cached generation, or the ConceptLM remote-code "
            "from_pretrained compatibility path. native_vllm selects the "
            "parity-gated optimized vLLM plugin."
        ),
    )
    parser.add_argument(
        "--hf-compile-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the HF-loaded model with the same compiled routes as DCP.",
    )
    parser.add_argument(
        "--hf-align-dcp-runtime-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override stale HF export metadata with the measured DCP inference config.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--standard-input-config", required=True)
    parser.add_argument("--task-name", default=STANDARD_TASK_NAME)
    parser.add_argument("--task-include-path", default="")
    parser.add_argument("--dataset-test-file", default="")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=STANDARD_MAX_GEN_TOKS)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--prompt-transport",
        choices=PROMPT_TRANSPORTS,
        default="raw_completion",
        help=(
            "Pass the fixed OLMo-Eval few-shot prompt either directly or as "
            "one user message rendered with the model tokenizer's chat template."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip a shard only after validating its complete saved result and prediction coverage.",
    )
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--flash-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable static-batch Flash Decode; currently requires batch size one.",
    )
    parser.add_argument(
        "--active-compaction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check EOS/stop patterns on CUDA and compact finished request slots.",
    )
    parser.add_argument(
        "--compaction-interval",
        type=int,
        default=4,
        help="Copy one batched alive mask to the scheduler every N decode steps.",
    )
    parser.add_argument(
        "--continuous-batching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refill compacted slots from the shard's pending FIFO request queue.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    add_native_vllm_args(parser)
    add_lmdeploy_args(parser)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _distributed_shard_args(args: argparse.Namespace) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1:
        if args.shard_count not in (1, world_size):
            raise ValueError(
                f"--shard-count must be 1 or WORLD_SIZE={world_size}, "
                f"got {args.shard_count}"
            )
        args.shard_count = world_size
        args.shard_index = rank
    return rank, local_rank, world_size


def _rank_output_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size <= 1:
        return path
    return path.parent / f"shard-{rank:02d}" / path.name


def _isolate_compile_caches(rank: int, world_size: int) -> dict[str, str]:
    cache_paths: dict[str, str] = {}
    if world_size <= 1:
        return cache_paths
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        root = os.environ.get(key)
        if not root:
            continue
        rank_path = Path(root) / f"rank-{rank:02d}"
        rank_path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(rank_path)
        cache_paths[key] = str(rank_path)
    return cache_paths


def _append_extra_site_packages(value: str) -> list[str]:
    appended: list[str] = []
    for raw_path in value.split(os.pathsep):
        if not raw_path:
            continue
        path = str(Path(raw_path).resolve())
        if not Path(path).is_dir():
            raise FileNotFoundError(f"extra site-packages directory not found: {path}")
        if path not in sys.path:
            sys.path.append(path)
        appended.append(path)
    return appended


def _strict_extract(text: str) -> str:
    matches = STRICT_ANSWER_RE.findall(text)
    return matches[0].strip() if matches else "[invalid]"


def _flexible_extract(text: str) -> str:
    matches = FLEXIBLE_ANSWER_RE.findall(text)
    if not matches:
        return "[invalid]"
    selected = matches[-1]
    if isinstance(selected, tuple):
        return next((part.strip() for part in selected if part), "[invalid]")
    return str(selected).strip()


def _normalize_exact_match(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r",", "", normalized)
    normalized = re.sub(r"\$", "", normalized)
    normalized = re.sub(r"(?s).*#### ", "", normalized)
    normalized = re.sub(r"\.$", "", normalized)
    return normalized.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gsm8k(
    task_name: str,
    task_include_path: str,
) -> tuple[Any, list[dict[str, Any]]]:
    TaskManager = task_manager_class()
    loaded = TaskManager(
        include_path=task_include_path,
        include_defaults=False,
    ).load(task_name)
    task = loaded["tasks"][task_name]
    eval_docs = task.eval_docs
    docs = list(eval_docs() if callable(eval_docs) else eval_docs)
    return task, docs


def _load_standard_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"standard GSM8K input config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("standard GSM8K input config must be a JSON object")
    tasks = config.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list) or STANDARD_TASK_GROUP not in tasks:
        raise ValueError(
            "standard GSM8K input config must select "
            f"{STANDARD_TASK_GROUP!r}, got {tasks!r}"
        )
    task_include_path = config.get("task_include_path")
    if not isinstance(task_include_path, str) or not task_include_path:
        raise ValueError("standard GSM8K input config has no task_include_path")
    return config


def _task_test_file(task: Any) -> Path:
    dataset_kwargs = task.config.dataset_kwargs or {}
    data_files = dataset_kwargs.get("data_files") or {}
    test_file = data_files.get("test")
    if not isinstance(test_file, str) or not test_file:
        raise ValueError(f"task {task.config.task!r} has no test data file")
    return Path(test_file)


def _validate_standard_task(task: Any) -> dict[str, Any]:
    if task.config.task != STANDARD_TASK_NAME:
        raise ValueError(
            f"standard GSM8K leaf task must be {STANDARD_TASK_NAME!r}, "
            f"got {task.config.task!r}"
        )
    num_fewshot = int(task.config.num_fewshot)
    if num_fewshot != STANDARD_NUM_FEWSHOT:
        raise ValueError(
            f"standard GSM8K must use {STANDARD_NUM_FEWSHOT}-shot, got {num_fewshot}"
        )
    sampler = getattr(task.fewshot_cfg, "sampler", None)
    samples = getattr(task.fewshot_cfg, "samples", None)
    if sampler != "first_n" or samples is None:
        raise ValueError(
            "standard GSM8K must use first_n over fixed fewshot_config.samples"
        )
    generation_kwargs = dict(task.config.generation_kwargs or {})
    expected = {
        "do_sample": True,
        "temperature": STANDARD_TEMPERATURE,
        "top_p": STANDARD_TOP_P,
        "max_gen_toks": STANDARD_MAX_GEN_TOKS,
    }
    actual = {key: generation_kwargs.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"standard GSM8K generation contract changed: {actual!r} != {expected!r}"
        )
    until = list(generation_kwargs.get("until") or [])
    if until != STOP_STRINGS:
        raise ValueError(
            f"standard GSM8K stop strings changed: {until!r} != {STOP_STRINGS!r}"
        )
    return {
        "num_fewshot": num_fewshot,
        "fewshot_sampler": sampler,
        "fixed_fewshot_samples": True,
        "generation_kwargs": generation_kwargs,
        "runtime_repeats": STANDARD_REPEATS,
    }


def _prompt_sha256(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _apply_prompt_transport(
    prompts: list[str],
    tokenizer: Any,
    prompt_transport: str,
) -> tuple[list[str], str | None]:
    if prompt_transport == "raw_completion":
        return list(prompts), None
    if prompt_transport != "chat_template":
        raise ValueError(f"unsupported prompt transport: {prompt_transport!r}")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("tokenizer does not support apply_chat_template")
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    if not chat_template:
        raise ValueError(
            "chat_template prompt transport requested but tokenizer has no template"
        )
    rendered_prompts = [
        str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for prompt in prompts
    ]
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    return rendered_prompts, chat_template_sha256


def _ensure_chat_template(tokenizer: Any, tokenizer_path: str) -> str:
    if str(getattr(tokenizer, "chat_template", "") or ""):
        return "tokenizer.chat_template"
    template_path = Path(tokenizer_path) / "chat_template.jinja"
    if not template_path.is_file():
        raise FileNotFoundError(
            "chat_template prompt transport requested, but neither the loaded "
            f"tokenizer nor {template_path} provides a template"
        )
    chat_template = template_path.read_text(encoding="utf-8")
    if not chat_template:
        raise ValueError(f"chat template file is empty: {template_path}")
    tokenizer.chat_template = chat_template
    return str(template_path.resolve())


def _build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_source=args.model_source,
        checkpoint_root=args.checkpoint_root,
        ckpt_step=args.ckpt_step,
        tokenizer_model=args.tokenizer_model,
        train_wandb_config=args.train_wandb_config,
        hf_model_path=args.hf_model_path,
        hf_backend=args.hf_backend,
        hf_compile_routes=bool(args.hf_compile_routes),
        hf_align_dcp_runtime_config=bool(args.hf_align_dcp_runtime_config),
        flash_decode=bool(args.flash_decode),
        cuda_graph=False,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dcp_checkpoint_native_runtime=(args.model_source == "dcp"),
    )


def main() -> None:
    args = parse_args()
    if (
        args.fewshot_seed != EVALUATION_SEED
        or args.sampling_seed != EVALUATION_SEED
    ):
        raise ValueError(
            "GSM8K evaluation fixes few-shot and sampling seeds to "
            f"{EVALUATION_SEED}"
        )
    extra_site_packages = _append_extra_site_packages(
        os.environ.get("CONCEPTLM_EXTRA_SITE_PACKAGES", "")
    )
    rank, local_rank, world_size = _distributed_shard_args(args)
    compile_cache_paths = _isolate_compile_caches(rank, world_size)
    if args.model_source == "dcp":
        if not args.checkpoint_root:
            raise ValueError("--checkpoint-root is required for --model-source=dcp")
        if not args.train_wandb_config:
            raise ValueError("--train-wandb-config is required for --model-source=dcp")
        if args.batch_size != 1:
            raise ValueError("native Megatron DCP GSM8K requires batch size one")
    elif not args.hf_model_path:
        raise ValueError("--hf-model-path is required for --model-source=hf")
    transformers_backends = ("transformers", "from_pretrained")
    tokenizer_from_hf_backends = (
        *transformers_backends,
        NATIVE_VLLM_BACKEND,
        LMDEPLOY_BACKEND,
    )
    if not args.tokenizer_model and not (
        args.model_source == "hf"
        and args.hf_backend in tokenizer_from_hf_backends
    ):
        raise ValueError("--tokenizer-model is required")
    if args.shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {args.shard_count}")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f"shard_index must be in [0, {args.shard_count}), got {args.shard_index}"
        )
    if args.flash_decode and args.batch_size != 1:
        raise ValueError(
            "Flash Decode GSM8K currently requires batch_size=1 because normal "
            "GSM8K batches have variable prompt lengths"
        )
    if args.hf_backend in transformers_backends and args.flash_decode:
        raise ValueError(
            "Transformers generation backends do not support the ConceptLM "
            "Flash Decode switch"
        )
    if args.hf_backend in transformers_backends and (
        args.active_compaction or args.continuous_batching
    ):
        raise ValueError(
            "active compaction and continuous batching are ConceptLM-runtime "
            "features, not standard Transformers generate features"
        )
    if args.continuous_batching and not args.active_compaction:
        raise ValueError("continuous batching requires --active-compaction")
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        validate_native_vllm_args(
            args,
            batch_size=args.batch_size,
            model_source=args.model_source,
        )
    elif args.hf_backend == LMDEPLOY_BACKEND:
        validate_lmdeploy_args(
            args,
            batch_size=args.batch_size,
            model_source=args.model_source,
        )
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        if world_size != 1:
            raise ValueError(
                "gsm8k_eval native_vllm mode supports one local engine; "
                "use ncp_olmo_eval.portable_tasks gsm8k for multi-GPU sharding"
            )
        if args.flash_decode or args.active_compaction or args.continuous_batching:
            raise ValueError(
                "native_vllm owns PagedAttention scheduling and does not accept "
                "the custom-runtime Flash Decode or compaction switches"
            )
    if args.hf_backend == LMDEPLOY_BACKEND:
        if world_size != 1:
            raise ValueError(
                "gsm8k_eval lmdeploy mode supports one local engine; use a "
                "scheduler-neutral multi-GPU task adapter for sharding"
            )
        if args.flash_decode or args.active_compaction or args.continuous_batching:
            raise ValueError(
                "lmdeploy owns PagedAttention scheduling and does not accept "
                "the custom-runtime Flash Decode or compaction switches"
            )
    process_started = time.perf_counter()
    process_started_at = _utc_now()
    output_path = _rank_output_path(Path(args.output_json), rank, world_size)
    predictions_path = _rank_output_path(
        Path(args.predictions_jsonl),
        rank,
        world_size,
    )
    progress_path = _rank_output_path(Path(args.progress_json), rank, world_size)
    standard_input_path = Path(args.standard_input_config)
    standard_input = _load_standard_input(standard_input_path)
    task_include_path = str(standard_input["task_include_path"])
    if args.task_include_path and args.task_include_path != task_include_path:
        raise ValueError(
            "--task-include-path does not match standard input config: "
            f"{args.task_include_path!r} != {task_include_path!r}"
        )
    for path in (output_path, predictions_path, progress_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[gsm8k] loading standard task={args.task_name} "
        f"input_config={standard_input_path}",
        flush=True,
    )
    task_load_started = time.perf_counter()
    task, docs = _load_gsm8k(args.task_name, task_include_path)
    task_contract = _validate_standard_task(task)
    task_contract["lm_eval"] = runtime_identity()
    if args.max_new_tokens != task_contract["generation_kwargs"]["max_gen_toks"]:
        raise ValueError(
            "--max-new-tokens must match the standard task YAML: "
            f"{args.max_new_tokens} != "
            f"{task_contract['generation_kwargs']['max_gen_toks']}"
        )
    dataset_test_path = _task_test_file(task)
    if args.dataset_test_file and Path(args.dataset_test_file) != dataset_test_path:
        raise ValueError(
            "--dataset-test-file does not match the standard task YAML: "
            f"{args.dataset_test_file!r} != {str(dataset_test_path)!r}"
        )
    if not dataset_test_path.is_file():
        raise FileNotFoundError(f"standard GSM8K test file not found: {dataset_test_path}")
    dataset_test_sha256 = _file_sha256(dataset_test_path)
    if args.limit > 0:
        docs = docs[: args.limit]
    task.set_fewshot_seed(args.fewshot_seed)
    prompts = [
        task.fewshot_context(doc, num_fewshot=task_contract["num_fewshot"])
        for doc in docs
    ]
    task_load_seconds = time.perf_counter() - task_load_started

    schedule = list(range(len(docs)))[args.shard_index :: args.shard_count]
    batches = [
        schedule[start : start + args.batch_size]
        for start in range(0, len(schedule), args.batch_size)
    ]
    if args.resume and output_path.is_file() and predictions_path.is_file():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        checks = {
            "status": "GSM8K_EVAL_OK",
            "dataset_sample_count": len(docs),
            "sample_count": len(schedule),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "distributed_rank": rank,
            "distributed_world_size": world_size,
            "batch_size": args.batch_size,
            "fewshot_seed": args.fewshot_seed,
            "sampling_seed": args.sampling_seed,
            "prompt_transport": args.prompt_transport,
            "artifact_mutated": False,
        }
        for field, expected in checks.items():
            if saved.get(field) != expected:
                raise RuntimeError(
                    f"saved GSM8K shard cannot be resumed: {field}="
                    f"{saved.get(field)!r} != {expected!r}"
                )
        seen = []
        with predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.append(int(json.loads(line)["doc_index"]))
        if seen != schedule:
            raise RuntimeError("saved GSM8K prediction coverage cannot be resumed")
        print(
            f"[gsm8k] resume validated completed shard={args.shard_index}/"
            f"{args.shard_count}",
            flush=True,
        )
        return
    source_model_path = (
        str(Path(args.hf_model_path).resolve())
        if args.model_source == "hf"
        else str(Path(args.checkpoint_root).resolve())
    )
    native_model_manifest = None
    source_artifact_before = None
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        source_artifact_before = benchmark._hf_artifact_fingerprint(
            source_model_path
        )
        output_parent = args.output_json.resolve().parent
        overlay_dir = (
            Path(args.vllm_model_overlay_dir)
            if args.vllm_model_overlay_dir
            else output_parent / "native-vllm-model"
        )
        runtime_model, native_model_manifest = prepare_native_vllm_model(
            source_model=source_model_path,
            runtime_config=args.vllm_runtime_config or None,
            overlay_dir=overlay_dir,
        )
        args.hf_model_path = str(runtime_model)
    artifact_before = benchmark._artifact_fingerprint(_build_model_args(args))

    print(
        f"[gsm8k] loading model_source={args.model_source} step={args.ckpt_step}",
        flush=True,
    )
    model_load_started = time.perf_counter()
    eval_model: Any | None = None
    native_inferencer: NativeVLLMInferencer | None = None
    lmdeploy_inferencer: LMDeployInferencer | None = None
    if args.hf_backend == NATIVE_VLLM_BACKEND:
        native_inferencer = NativeVLLMInferencer(
            model_path=args.hf_model_path,
            max_model_len=native_vllm_max_model_len(args),
            seed=args.sampling_seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            execution_mode=args.vllm_execution_mode,
            attention_backend=args.vllm_attention_backend,
            flash_attn_version=args.vllm_flash_attn_version,
            hlm_attention_impl=args.vllm_hlm_attention_impl,
            model_family=args.vllm_model_family,
            max_batch_size=args.batch_size,
            **native_vllm_speculative_kwargs(args),
        )
        inferencer: Any = native_inferencer
    elif args.hf_backend == LMDEPLOY_BACKEND:
        lmdeploy_inferencer = LMDeployInferencer(
            model_path=args.hf_model_path,
            seed=args.sampling_seed,
            max_batch_size=args.batch_size,
            cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            log_level=args.lmdeploy_log_level,
        )
        inferencer = lmdeploy_inferencer
    else:
        eval_model = benchmark.load_eval_model(_build_model_args(args))
        if (
            args.model_source == "hf"
            and args.hf_backend in transformers_backends
        ):
            inferencer = TransformersInferencer(
                eval_model,
                max_batch_size=args.batch_size,
            )
        else:
            inferencer = ConceptLMInferencer(
                eval_model,
                max_batch_size=args.batch_size,
                use_kv_cache=(args.model_source != "dcp"),
                active_compaction=bool(args.active_compaction),
                compaction_interval=args.compaction_interval,
            )
    chat_template_source = None
    if args.prompt_transport == "chat_template":
        tokenizer_identity_path = (
            source_model_path
            if args.model_source == "hf"
            else str(Path(args.tokenizer_model).resolve())
        )
        chat_template_source = _ensure_chat_template(
            inferencer.tokenizer,
            tokenizer_identity_path,
        )
    runtime_prompts, chat_template_sha256 = _apply_prompt_transport(
        prompts,
        inferencer.tokenizer,
        args.prompt_transport,
    )
    prompt_token_counts = [
        len(inferencer.tokenizer.encode(prompt, add_special_tokens=True))
        for prompt in runtime_prompts
    ]
    effective_max_model_len = (
        native_vllm_max_model_len(args)
        if native_inferencer is not None
        else (
            lmdeploy_inferencer.session_len
            if lmdeploy_inferencer is not None
            else args.seq_length
        )
    )
    if (
        prompt_token_counts
        and max(prompt_token_counts) + args.max_new_tokens
        > effective_max_model_len
    ):
        raise ValueError(
            "GSM8K prompt plus generation exceeds configured model length: "
            f"max_prompt={max(prompt_token_counts)}, max_new={args.max_new_tokens}, "
            f"max_model_len={effective_max_model_len}"
        )
    model_load_seconds = time.perf_counter() - model_load_started
    print(
        f"[gsm8k] model loaded in {model_load_seconds:.3f}s; "
        f"prompt_tokens={min(prompt_token_counts)}..{max(prompt_token_counts)}",
        flush=True,
    )

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=STANDARD_TEMPERATURE,
        top_p=STANDARD_TOP_P,
        stop=list(STOP_STRINGS),
    )
    warmup_seconds = 0.0
    if batches and args.warmup_tokens > 0:
        warmup_ids = batches[0]
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
        inferencer.generate(
            [runtime_prompts[index] for index in warmup_ids],
            SamplingParams(max_tokens=args.warmup_tokens, temperature=0.0),
        )
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        print(f"[gsm8k] warmup completed in {warmup_seconds:.3f}s", flush=True)

    torch.manual_seed(args.sampling_seed)
    torch.cuda.manual_seed_all(args.sampling_seed)
    standard_correct = 0
    strict_correct = 0
    flexible_correct = 0
    returned_token_count = 0
    completed = 0
    actual_batch_histogram: dict[str, int] = {}
    observed_varlen_attention_backends: set[str] = set()
    actual_computed_token_slots = 0
    actual_model_decode_token_steps = 0
    avoided_computed_token_slots = 0
    compaction_sync_count = 0
    compaction_rebuild_count = 0
    predictions_path.write_text("", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    evaluation_started_at = _utc_now()
    print(
        f"[gsm8k] evaluating samples={len(schedule)}/{len(docs)} "
        f"shard={args.shard_index}/{args.shard_count} batch_size={args.batch_size} "
        f"length_bucketing=false flash_decode={args.flash_decode}",
        flush=True,
    )

    with predictions_path.open("a", encoding="utf-8") as prediction_file:
        evaluation_batches = [schedule] if args.continuous_batching else batches
        for batch_index, doc_indices in enumerate(evaluation_batches):
            batch_prompts = [runtime_prompts[index] for index in doc_indices]
            if args.continuous_batching:
                completions = inferencer.generate_continuous_with_kv_cache(
                    batch_prompts,
                    sampling,
                    max_num_seqs=args.batch_size,
                )
            else:
                completions = inferencer.generate(batch_prompts, sampling)
            observed_varlen_attention_backends.update(
                str(completion.kv_stats["varlen_attention_backend"])
                for completion in completions
                if completion.kv_stats.get("varlen_attention_backend") is not None
            )
            batch_kv_stats = completions[0].kv_stats if completions else {}
            if args.continuous_batching:
                actual_batch_histogram.update(
                    {
                        str(key): int(value)
                        for key, value in batch_kv_stats.get(
                            "active_batch_histogram",
                            {},
                        ).items()
                    }
                )
            else:
                actual_batch_histogram[str(len(doc_indices))] = (
                    actual_batch_histogram.get(str(len(doc_indices)), 0) + 1
                )
            actual_computed_token_slots += int(
                batch_kv_stats.get(
                    "computed_token_slots",
                    len(doc_indices) * args.max_new_tokens,
                )
            )
            actual_model_decode_token_steps += int(
                batch_kv_stats.get(
                    "model_decode_token_steps",
                    len(doc_indices) * max(0, args.max_new_tokens - 1),
                )
            )
            avoided_computed_token_slots += int(
                batch_kv_stats.get("avoided_computed_token_slots", 0)
            )
            compaction_sync_count += int(
                batch_kv_stats.get("compaction_sync_count", 0)
            )
            compaction_rebuild_count += int(
                batch_kv_stats.get("compaction_rebuild_count", 0)
            )
            for doc_index, completion in zip(doc_indices, completions, strict=True):
                doc = docs[doc_index]
                official_score = score_gsm_answer(
                    completion.text,
                    str(doc["answer"]),
                )
                standard_match = bool(official_score["primary_correct"])
                pass_at_1 = float(standard_match)
                strict_prediction = _strict_extract(completion.text)
                flexible_prediction = _flexible_extract(completion.text)
                gold = _normalize_exact_match(doc["answer"])
                strict_match = _normalize_exact_match(strict_prediction) == gold
                flexible_match = _normalize_exact_match(flexible_prediction) == gold
                standard_correct += int(standard_match)
                strict_correct += int(strict_match)
                flexible_correct += int(flexible_match)
                returned_token_count += len(completion.token_ids)
                completed += 1
                prediction_file.write(
                    json.dumps(
                        {
                            "doc_index": doc_index,
                            "question": doc["question"],
                            "gold": gold,
                            "output": completion.text,
                            "pass_at_1": pass_at_1,
                            "standard_correct": standard_match,
                            "official_prediction": official_score[
                                "normalized_prediction"
                            ],
                            "official_gold": official_score["normalized_gold"],
                            "answer_scorer": official_score["answer_scorer"],
                            "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
                            "strict_prediction": strict_prediction,
                            "flexible_prediction": flexible_prediction,
                            "strict_correct": strict_match,
                            "flexible_correct": flexible_match,
                            "returned_token_count": len(completion.token_ids),
                            "prompt_token_count": prompt_token_counts[doc_index],
                            "finish_reason": completion.finish_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            prediction_file.flush()

            if (batch_index + 1) % max(1, args.progress_every) == 0 or completed == len(schedule):
                elapsed = time.perf_counter() - evaluation_started
                _write_json(
                    progress_path,
                    {
                        "status": (
                            "RUNNING" if completed < len(schedule) else "EVALUATION_COMPLETE"
                        ),
                        "batch_size": args.batch_size,
                        "flash_decode": bool(args.flash_decode),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "completed": completed,
                        "total": len(schedule),
                        "pass_at_1_correct": standard_correct,
                        "pass_at_1": standard_correct / completed if completed else 0.0,
                        "strict_correct": strict_correct,
                        "strict_accuracy": strict_correct / completed if completed else 0.0,
                        "flexible_correct": flexible_correct,
                        "flexible_accuracy": flexible_correct / completed if completed else 0.0,
                        "evaluation_seconds": elapsed,
                        "updated_at": _utc_now(),
                    },
                )

    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_finished_at = _utc_now()
    if lmdeploy_inferencer is not None:
        lmdeploy_inferencer.close()
    artifact_after = benchmark._artifact_fingerprint(_build_model_args(args))
    artifact_mutated = artifact_before != artifact_after
    if artifact_mutated:
        raise RuntimeError("runtime model artifact changed during GSM8K evaluation")
    source_artifact_after = (
        benchmark._hf_artifact_fingerprint(source_model_path)
        if source_artifact_before is not None
        else None
    )
    source_model_mutated = (
        source_artifact_before != source_artifact_after
        if source_artifact_before is not None
        else artifact_mutated
        if args.model_source == "hf"
        else None
    )
    if source_model_mutated:
        raise RuntimeError("source model artifact changed during GSM8K evaluation")

    fixed_decode_steps = len(schedule) * args.max_new_tokens
    result = {
        "status": "GSM8K_EVAL_OK",
        "model_source": args.model_source,
        "hf_backend": args.hf_backend if args.model_source == "hf" else None,
        "model_identifier": (
            Path(source_model_path).name
            if args.model_source == "hf"
            else f"iter_{args.ckpt_step}"
        ),
        "task": args.task_name,
        "task_group": STANDARD_TASK_GROUP,
        "task_include_path": task_include_path,
        "standard_input_config": str(standard_input_path),
        "standard_input_config_sha256": _file_sha256(standard_input_path),
        "task_contract": task_contract,
        "dataset_path": "parquet",
        "dataset_test_file": str(dataset_test_path),
        "dataset_test_sha256": dataset_test_sha256,
        "split": "test",
        "num_fewshot": task_contract["num_fewshot"],
        "fewshot_sampler": task_contract["fewshot_sampler"],
        "fixed_fewshot_samples": task_contract["fixed_fewshot_samples"],
        "fewshot_seed": args.fewshot_seed,
        "sampling_seed": args.sampling_seed,
        "sample_count": len(schedule),
        "dataset_sample_count": len(docs),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "distributed_rank": rank,
        "distributed_local_rank": local_rank,
        "distributed_world_size": world_size,
        "compile_cache_paths": compile_cache_paths,
        "extra_site_packages": extra_site_packages,
        "schedule_position_rule": "doc_indices[shard_index::shard_count]",
        "batch_size": args.batch_size,
        "normal_varlen_batching": args.batch_size > 1,
        "active_compaction": bool(args.active_compaction),
        "compaction_interval": args.compaction_interval,
        "continuous_batching": bool(args.continuous_batching),
        "length_bucketing": False,
        "varlen_attention_backends": sorted(observed_varlen_attention_backends),
        "attention_implementation": getattr(
            eval_model,
            "attention_implementation",
            None,
        ),
        "cache_backend": getattr(eval_model, "cache_backend", None),
        "transformers_use_cache": getattr(
            eval_model,
            "transformers_use_cache",
            None,
        ),
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
        "actual_batch_histogram": actual_batch_histogram,
        "v21_route_torch_compile_dynamic": os.getenv(
            "CONCEPTLM_V21_ROUTE_TORCH_COMPILE_DYNAMIC",
            "1",
        ).lower()
        in ("1", "true", "yes", "on"),
        "v21_route_torch_compile_mode": os.getenv(
            "CONCEPTLM_V21_ROUTE_TORCH_COMPILE_MODE",
            "default",
        ),
        "flash_decode": bool(args.flash_decode),
        "cuda_graph": False,
        "max_new_tokens": args.max_new_tokens,
        "seq_length": args.seq_length,
        "do_sample": True,
        "temperature": STANDARD_TEMPERATURE,
        "top_p": STANDARD_TOP_P,
        "repeats": STANDARD_REPEATS,
        "stop_strings": STOP_STRINGS,
        "primary_metric": "pass@1",
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "pass_at_1_correct": standard_correct,
        "pass_at_1": standard_correct / len(schedule) if schedule else 0.0,
        "legacy_extractors_are_diagnostic_only": True,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / len(schedule) if schedule else 0.0,
        "flexible_correct": flexible_correct,
        "flexible_accuracy": flexible_correct / len(schedule) if schedule else 0.0,
        "prompt_transport": args.prompt_transport,
        "chat_template_source": chat_template_source,
        "chat_template_sha256": chat_template_sha256,
        "prompt_sha256": _prompt_sha256(prompts),
        "rendered_prompt_sha256": _prompt_sha256(runtime_prompts),
        "prompt_token_count": {
            "min": min(prompt_token_counts) if prompt_token_counts else 0,
            "max": max(prompt_token_counts) if prompt_token_counts else 0,
            "sum": sum(prompt_token_counts),
        },
        "returned_token_count": returned_token_count,
        "computed_decode_steps": fixed_decode_steps,
        "fixed_baseline_computed_token_slots": fixed_decode_steps,
        "actual_computed_token_slots": actual_computed_token_slots,
        "avoided_computed_token_slots": avoided_computed_token_slots,
        "avoided_computed_token_fraction": (
            avoided_computed_token_slots / fixed_decode_steps
            if fixed_decode_steps > 0
            else 0.0
        ),
        "actual_model_decode_token_steps": actual_model_decode_token_steps,
        "compaction_sync_count": compaction_sync_count,
        "compaction_rebuild_count": compaction_rebuild_count,
        "computed_decode_steps_per_second": (
            fixed_decode_steps / evaluation_seconds if evaluation_seconds > 0 else 0.0
        ),
        "actual_computed_token_slots_per_second": (
            actual_computed_token_slots / evaluation_seconds
            if evaluation_seconds > 0
            else 0.0
        ),
        "returned_tokens_per_second": (
            returned_token_count / evaluation_seconds if evaluation_seconds > 0 else 0.0
        ),
        "task_and_prompt_load_seconds": task_load_seconds,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "evaluation_seconds": evaluation_seconds,
        "process_wall_seconds": time.perf_counter() - process_started,
        "process_started_at": process_started_at,
        "evaluation_started_at": evaluation_started_at,
        "evaluation_finished_at": evaluation_finished_at,
        "peak_gpu_memory_gib": benchmark._peak_memory_gib(),
        "checkpoint_root": args.checkpoint_root,
        "ckpt_step": args.ckpt_step,
        "hf_model_path": args.hf_model_path,
        "model_identity_path": source_model_path,
        "native_vllm_model_manifest": native_model_manifest,
        "hf_compile_routes": bool(args.hf_compile_routes),
        "hf_align_dcp_runtime_config": bool(args.hf_align_dcp_runtime_config),
        "artifact_mutated": artifact_mutated,
        "checkpoint_mutated": (
            artifact_mutated if args.model_source == "dcp" else None
        ),
        "source_model_mutated": (
            source_model_mutated if args.model_source == "hf" else None
        ),
        "artifact_metadata_before": artifact_before,
        "artifact_metadata_after": artifact_after,
        "source_artifact_metadata_before": source_artifact_before,
        "source_artifact_metadata_after": source_artifact_after,
        "predictions_jsonl": str(predictions_path),
        "progress_json": str(progress_path),
    }
    _write_json(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
