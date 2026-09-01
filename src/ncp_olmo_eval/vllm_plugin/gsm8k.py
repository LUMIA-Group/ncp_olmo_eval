"""Rank-sharded GSM8K evaluation for the native vLLM backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ncp_olmo_eval.core_native_answer_eval import (
    OFFICIAL_OLMO_EVAL_COMMIT,
    score_gsm_answer,
)
from ncp_olmo_eval.lm_eval_runtime import runtime_identity, task_manager_class

STANDARD_TASK_GROUP = "olmo_eval_paper_math_gsm_8shot"
STANDARD_TASK_NAME = "olmo_eval_paper_gsm8k_main"
STANDARD_NUM_FEWSHOT = 8
STANDARD_TEMPERATURE = 0.6
STANDARD_TOP_P = 0.6
STANDARD_MAX_GEN_TOKS = 512
STANDARD_REPEATS = 1
STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]
MODEL_FINGERPRINT_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
MODEL_FINGERPRINT_OPTIONAL_FILES = ("model.safetensors.index.json",)
EVALUATION_SEED = 42


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--standard-input-config", type=Path, required=True)
    parser.add_argument("--task-name", default=STANDARD_TASK_NAME)
    parser.add_argument("--task-include-path", default="")
    parser.add_argument("--dataset-test-file", default="")
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    _add_protocol_args(prepare)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--input-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--model-dir", type=Path, required=True)
    generate.add_argument("--model-family", choices=("conceptlm", "auto"), default="conceptlm")
    generate.add_argument("--rank", type=int, required=True)
    generate.add_argument("--world-size", type=int, required=True)
    generate.add_argument("--sampling-seed", type=int, default=42)
    generate.add_argument("--samples-per-doc", type=int, default=1)
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument(
        "--scheduler-queue-size",
        type=int,
        default=0,
        help="requests submitted per generate call; zero uses batch-size",
    )
    generate.add_argument("--max-model-len", type=int, default=2048)
    generate.add_argument("--warmup-tokens", type=int, default=2)
    generate.add_argument("--progress-every", type=int, default=10)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    generate.add_argument("--execution-mode", choices=("eager", "piecewise"), default="eager")
    generate.add_argument("--attention-backend", default="FLASH_ATTN")
    generate.add_argument("--flash-attn-version", type=int, choices=(2, 3), default=3)
    generate.add_argument(
        "--hlm-attention-impl", choices=("legacy_mixed", "uniform_flash"), default="legacy_mixed"
    )
    generate.add_argument("--speculative-draft-model", type=Path)
    generate.add_argument("--speculative-num-tokens", type=int, default=16)
    generate.add_argument("--speculative-telemetry-path", type=Path)
    generate.add_argument(
        "--speculative-verification-mode",
        choices=(
            "sequential_exact",
            "intra_chunk_exact",
            "segmented_kv_approx",
            "transactional_exact",
            "chunk_parallel",
        ),
        default="sequential_exact",
        help=(
            "segmented_kv_approx rolls target state back after rejection and "
            "may change target tokens; transactional_exact and chunk_parallel "
            "are legacy input aliases"
        ),
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--world-size", type=int, required=True)
    aggregate.add_argument("--samples-per-doc", type=int, default=1)
    aggregate.add_argument("--batch-size", type=int, default=1)
    aggregate.add_argument("--model-family", choices=("conceptlm", "auto"), default="conceptlm")
    _add_protocol_args(aggregate)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_prediction(doc: dict[str, Any], output: str) -> dict[str, Any]:
    """Score GSM8K exclusively with the pinned OLMo-Eval answer contract."""

    score = score_gsm_answer(output, str(doc["answer"]))
    return {
        "pass_at_1": float(bool(score["primary_correct"])),
        "standard_correct": bool(score["primary_correct"]),
        "official_prediction": score["normalized_prediction"],
        "official_gold": score["normalized_gold"],
        "answer_scorer": score["answer_scorer"],
        "olmo_eval_commit": score["olmo_eval_commit"],
        "official_source_sha256s": score["official_source_sha256s"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_sha256(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_and_validate_prepared_inputs(
    input_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load prepared inputs and reject any post-prepare artifact drift."""

    manifest_path = input_dir / "manifest.json"
    inputs_path = input_dir / "inputs.jsonl"
    if not (input_dir / "_SUCCESS").is_file():
        raise RuntimeError(f"prepared GSM8K inputs have no success marker: {input_dir}")
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(inputs_path)
    if str(inputs_path.resolve()) != str(Path(manifest["inputs_jsonl"]).resolve()):
        raise RuntimeError("prepared GSM8K inputs path differs from its manifest")
    inputs_sha256 = _file_sha256(inputs_path)
    if inputs_sha256 != manifest.get("inputs_jsonl_sha256"):
        raise RuntimeError("prepared GSM8K inputs JSONL changed after preparation")
    expected_count = int(manifest["dataset_sample_count"])
    if len(rows) != expected_count:
        raise RuntimeError("GSM8K input row count does not match its manifest")
    if [row.get("doc_index") for row in rows] != list(range(expected_count)):
        raise RuntimeError("prepared GSM8K doc_index sequence changed")
    prompts = [row.get("prompt") for row in rows]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise RuntimeError("prepared GSM8K inputs contain a non-string prompt")
    if _prompt_sha256(prompts) != manifest.get("prompt_sha256"):
        raise RuntimeError("prepared GSM8K rendered prompts changed")
    for path_field, sha_field, label in (
        ("standard_input_config", "standard_input_config_sha256", "standard input config"),
        ("dataset_test_file", "dataset_test_sha256", "dataset test file"),
    ):
        artifact = Path(str(manifest[path_field]))
        if not artifact.is_file():
            raise RuntimeError(f"prepared GSM8K {label} is missing: {artifact}")
        if _file_sha256(artifact) != manifest.get(sha_field):
            raise RuntimeError(f"prepared GSM8K {label} changed after preparation")
    return manifest, rows


def _load_standard_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"standard GSM8K input config not found: {path}")
    config = _read_json(path)
    tasks = config.get("tasks")
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list) or STANDARD_TASK_GROUP not in tasks:
        raise ValueError(
            "standard GSM8K input config must select " f"{STANDARD_TASK_GROUP!r}, got {tasks!r}"
        )
    task_include_path = config.get("task_include_path")
    if not isinstance(task_include_path, str) or not task_include_path:
        raise ValueError("standard GSM8K input config has no task_include_path")
    return config


def _load_gsm8k(task_name: str, task_include_path: str) -> tuple[Any, list[dict[str, Any]]]:
    TaskManager = task_manager_class()
    loaded = TaskManager(include_path=task_include_path, include_defaults=False).load(task_name)
    task = loaded["tasks"][task_name]
    eval_docs = task.eval_docs
    docs = list(eval_docs() if callable(eval_docs) else eval_docs)
    return task, docs


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
            f"standard GSM8K leaf task must be {STANDARD_TASK_NAME!r}, " f"got {task.config.task!r}"
        )
    num_fewshot = int(task.config.num_fewshot)
    if num_fewshot != STANDARD_NUM_FEWSHOT:
        raise ValueError(f"standard GSM8K must use {STANDARD_NUM_FEWSHOT}-shot, got {num_fewshot}")
    sampler = getattr(task.fewshot_cfg, "sampler", None)
    samples = getattr(task.fewshot_cfg, "samples", None)
    if sampler != "first_n" or samples is None:
        raise ValueError("standard GSM8K must use first_n over fixed fewshot_config.samples")
    generation_kwargs = dict(task.config.generation_kwargs or {})
    expected = {
        "do_sample": True,
        "temperature": STANDARD_TEMPERATURE,
        "top_p": STANDARD_TOP_P,
        "max_gen_toks": STANDARD_MAX_GEN_TOKS,
    }
    actual = {key: generation_kwargs.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"standard GSM8K generation contract changed: {actual!r} != {expected!r}")
    until = list(generation_kwargs.get("until") or [])
    if until != STOP_STRINGS:
        raise ValueError(f"standard GSM8K stop strings changed: {until!r} != {STOP_STRINGS!r}")
    return {
        "num_fewshot": num_fewshot,
        "fewshot_sampler": sampler,
        "fixed_fewshot_samples": True,
        "generation_kwargs": generation_kwargs,
        "runtime_repeats": STANDARD_REPEATS,
    }


def _resolve_protocol(
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], Path]:
    standard_input = _load_standard_input(args.standard_input_config)
    task_include_path = str(standard_input["task_include_path"])
    if args.task_include_path and args.task_include_path != task_include_path:
        raise ValueError(
            "--task-include-path does not match standard input config: "
            f"{args.task_include_path!r} != {task_include_path!r}"
        )
    task, docs = _load_gsm8k(args.task_name, task_include_path)
    task_contract = _validate_standard_task(task)
    task_contract["lm_eval"] = runtime_identity()
    dataset_test_path = _task_test_file(task)
    if args.dataset_test_file and Path(args.dataset_test_file) != dataset_test_path:
        raise ValueError(
            "--dataset-test-file does not match the standard task YAML: "
            f"{args.dataset_test_file!r} != {str(dataset_test_path)!r}"
        )
    if not dataset_test_path.is_file():
        raise FileNotFoundError(f"standard GSM8K test file not found: {dataset_test_path}")
    if args.limit > 0:
        docs = docs[: args.limit]
    task.set_fewshot_seed(args.fewshot_seed)
    return task, docs, task_contract, dataset_test_path


def _model_fingerprint(model_dir: Path) -> dict[str, Any]:
    weight_paths = sorted(
        (path for path in model_dir.glob("*.safetensors") if path.is_file()),
        key=lambda path: path.name,
    )
    if not weight_paths:
        raise RuntimeError(f"model artifact has no safetensor weights: {model_dir}")

    paths = []
    for name in MODEL_FINGERPRINT_REQUIRED_FILES:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"model artifact file not found: {path}")
        paths.append(path)
    paths.extend(
        path for name in MODEL_FINGERPRINT_OPTIONAL_FILES if (path := model_dir / name).is_file()
    )
    paths.extend(weight_paths)

    files = []
    for path in sorted(paths, key=lambda candidate: candidate.name):
        name = path.name
        resolved = path.resolve()
        stat = resolved.stat()
        row: dict[str, Any] = {
            "name": name,
            "resolved_path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            row["sha256"] = _file_sha256(resolved)
        files.append(row)
    return {"model_dir": str(model_dir.resolve()), "files": files}


def prepare(args: argparse.Namespace) -> None:
    if args.fewshot_seed != EVALUATION_SEED:
        raise ValueError(f"GSM8K few-shot seed is fixed to {EVALUATION_SEED}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    task, docs, task_contract, dataset_test_path = _resolve_protocol(args)
    prompts = [task.fewshot_context(doc, num_fewshot=task_contract["num_fewshot"]) for doc in docs]
    rows = [
        {
            "doc_index": doc_index,
            "prompt": prompt,
            "question": str(doc.get("question", "")),
            "gold_answer": str(doc.get("answer", "")),
        }
        for doc_index, (doc, prompt) in enumerate(zip(docs, prompts, strict=True))
    ]
    _write_jsonl(args.output_dir / "inputs.jsonl", rows)
    manifest = {
        "status": "GSM8K_INPUTS_READY",
        "created_at": _utc_now(),
        "task": args.task_name,
        "task_group": STANDARD_TASK_GROUP,
        "task_include_path": str(
            _load_standard_input(args.standard_input_config)["task_include_path"]
        ),
        "standard_input_config": str(args.standard_input_config.resolve()),
        "standard_input_config_sha256": _file_sha256(args.standard_input_config),
        "task_contract": task_contract,
        "dataset_test_file": str(dataset_test_path),
        "dataset_test_sha256": _file_sha256(dataset_test_path),
        "dataset_sample_count": len(docs),
        "fewshot_seed": args.fewshot_seed,
        "prompt_sha256": _prompt_sha256(prompts),
        "inputs_jsonl": str((args.output_dir / "inputs.jsonl").resolve()),
        "inputs_jsonl_sha256": _file_sha256(args.output_dir / "inputs.jsonl"),
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def _request_timing(output: Any) -> tuple[float | None, float | None]:
    metrics = output.metrics
    ttft = getattr(metrics, "first_token_latency", None)
    if ttft is None or float(ttft) <= 0:
        first = getattr(metrics, "first_token_time", None)
        arrival = getattr(metrics, "arrival_time", None)
        ttft = None if first is None or arrival is None else float(first - arrival)
    finished = getattr(metrics, "finished_time", None)
    first = getattr(metrics, "first_token_time", None)
    tpot = None
    output_length = len(output.outputs[0].token_ids)
    if output_length > 1 and finished is not None and first is not None:
        tpot = float(finished - first) / (output_length - 1)
    return (None if ttft is None else float(ttft), tpot)


def _request_seed(base_seed: int, doc_index: int, sample_index: int) -> int:
    """Return a stable per-request seed independent of rank and batch shape."""

    payload = f"gsm8k\0{base_seed}\0{doc_index}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _rank_schedule(
    *, doc_count: int, samples_per_doc: int, rank: int, world_size: int
) -> list[tuple[int, int]]:
    requests = [
        (doc_index, sample_index)
        for doc_index in range(doc_count)
        for sample_index in range(samples_per_doc)
    ]
    return requests[rank::world_size]


def generate(args: argparse.Namespace) -> None:
    if args.sampling_seed != EVALUATION_SEED:
        raise ValueError(f"GSM8K sampling seed is fixed to {EVALUATION_SEED}")
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank {args.rank} is outside world size {args.world_size}")
    if args.samples_per_doc <= 0:
        raise ValueError("samples per doc must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    scheduler_queue_size = args.scheduler_queue_size or args.batch_size
    if scheduler_queue_size < args.batch_size:
        raise ValueError("scheduler queue size must be at least the active batch size")
    if not 0 < args.speculative_num_tokens <= 16:
        raise ValueError("speculative_num_tokens must be in [1, 16]")
    verification_mode = {
        "chunk_parallel": "intra_chunk_exact",
        "transactional_exact": "segmented_kv_approx",
    }.get(args.speculative_verification_mode, args.speculative_verification_mode)
    if args.speculative_draft_model and args.model_family != "conceptlm":
        raise ValueError("NCP DFlash requires model_family=conceptlm")
    input_manifest, input_rows = _load_and_validate_prepared_inputs(args.input_dir)
    if int(input_manifest.get("fewshot_seed", -1)) != EVALUATION_SEED:
        raise ValueError(
            "prepared GSM8K inputs use a noncanonical few-shot seed; "
            "prepare a fresh seed=42 input artifact instead of resuming"
        )
    schedule = _rank_schedule(
        doc_count=len(input_rows),
        samples_per_doc=args.samples_per_doc,
        rank=args.rank,
        world_size=args.world_size,
    )
    shard_dir = args.output_dir / f"shard-{args.rank:02d}"
    shard_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from vllm import LLM, SamplingParams
    from vllm import __version__ as vllm_version

    if args.speculative_draft_model and vllm_version != "0.13.0":
        raise RuntimeError(f"NCP DFlash requires vLLM 0.13.0, got {vllm_version}")

    if args.model_family == "conceptlm":
        from .plugin import register

        os.environ["CONCEPTLM_VLLM_ENABLE_UNVERIFIED"] = "1"
        os.environ["CONCEPTLM_HLM_ATTENTION_IMPL"] = args.hlm_attention_impl
        if args.speculative_draft_model:
            draft_root = args.speculative_draft_model.resolve()
            draft_config = _read_json(draft_root / "config.json")
            if not (draft_root / "model.safetensors").is_file():
                raise FileNotFoundError(f"NCP DFlash model.safetensors is missing: {draft_root}")
            os.environ["CONCEPTLM_VLLM_ENABLE_NCP_DFLASH"] = "1"
            os.environ["CONCEPTLM_DFLASH_CHECKPOINT"] = str(draft_root)
            os.environ["CONCEPTLM_DFLASH_TARGET_LAYERS"] = ",".join(
                str(value) for value in draft_config["target_layer_ids"]
            )
            os.environ["CONCEPTLM_DFLASH_CHUNK_SIZE"] = str(draft_config["concept_chunk_size"])
            os.environ["CONCEPTLM_DFLASH_MAX_MODEL_LEN"] = str(args.max_model_len)
            os.environ["CONCEPTLM_DFLASH_VERIFICATION_MODE"] = verification_mode
            if args.speculative_telemetry_path:
                os.environ["CONCEPTLM_DFLASH_TELEMETRY_PATH"] = str(
                    args.speculative_telemetry_path.resolve()
                )
        register()
    enforce_eager = args.execution_mode == "eager"
    compilation_config = (
        None if enforce_eager else {"mode": 3, "cudagraph_mode": "PIECEWISE", "custom_ops": ["all"]}
    )
    artifact_before = _model_fingerprint(args.model_dir)
    load_started = time.perf_counter()
    engine_kwargs = {
        "model": str(args.model_dir),
        "tensor_parallel_size": 1,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": False,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": False,
        "skip_tokenizer_init": False,
        "seed": args.sampling_seed,
        "compilation_config": compilation_config,
        "attention_config": {
            "backend": args.attention_backend,
            "flash_attn_version": args.flash_attn_version,
        },
    }
    if args.model_family == "conceptlm":
        engine_kwargs.update(
            trust_remote_code=True, worker_cls="ncp_olmo_eval.vllm_plugin.worker.ConceptLMGPUWorker"
        )
    if args.speculative_draft_model:
        engine_kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.speculative_num_tokens,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 1,
        }
    llm = LLM(**engine_kwargs)
    model_load_seconds = time.perf_counter() - load_started
    tokenizer = llm.get_tokenizer()
    prompt_token_counts = [
        len(tokenizer.encode(str(row["prompt"]), add_special_tokens=True)) for row in input_rows
    ]
    if prompt_token_counts and (
        max(prompt_token_counts) + STANDARD_MAX_GEN_TOKS > args.max_model_len
    ):
        raise ValueError(
            "GSM8K prompt plus generation exceeds max model length: "
            f"{max(prompt_token_counts)} + {STANDARD_MAX_GEN_TOKS} > "
            f"{args.max_model_len}"
        )

    warmup_seconds = 0.0
    if schedule and args.warmup_tokens > 0:
        warmup_started = time.perf_counter()
        warmup_requests = schedule[: args.batch_size]
        warmup = SamplingParams(
            temperature=0.0, max_tokens=args.warmup_tokens, ignore_eos=True, detokenize=False
        )
        llm.generate(
            [str(input_rows[doc_index]["prompt"]) for doc_index, _ in warmup_requests],
            warmup,
            use_tqdm=False,
        )
        warmup_seconds = time.perf_counter() - warmup_started

    predictions_path = shard_dir / "predictions.jsonl"
    progress_path = shard_dir / "progress.json"
    predictions_path.write_text("", encoding="utf-8")
    returned_token_count = 0
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    finish_reasons: dict[str, int] = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    evaluation_started_epoch = time.time()
    evaluation_started_at = _utc_now()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for batch_start in range(0, len(schedule), scheduler_queue_size):
            request_batch = schedule[batch_start : batch_start + scheduler_queue_size]
            rows = [input_rows[doc_index] for doc_index, _ in request_batch]
            request_seeds = [
                _request_seed(args.sampling_seed, doc_index, sample_index)
                for doc_index, sample_index in request_batch
            ]
            sampling = [
                SamplingParams(
                    temperature=STANDARD_TEMPERATURE,
                    top_p=STANDARD_TOP_P,
                    max_tokens=STANDARD_MAX_GEN_TOKS,
                    stop=list(STOP_STRINGS),
                    detokenize=True,
                    seed=request_seed,
                )
                for request_seed in request_seeds
            ]
            outputs = llm.generate([str(row["prompt"]) for row in rows], sampling, use_tqdm=False)
            if len(outputs) != len(request_batch) or any(
                len(output.outputs) != 1 for output in outputs
            ):
                raise RuntimeError(
                    "GSM8K generation returned an invalid batch shape: "
                    f"requests={len(request_batch)} outputs={len(outputs)}"
                )
            for (doc_index, sample_index), row, request_seed, request_output in zip(
                request_batch, rows, request_seeds, outputs, strict=True
            ):
                completion = request_output.outputs[0]
                token_ids = [int(token_id) for token_id in completion.token_ids]
                text = str(completion.text)
                finish_reason = str(completion.finish_reason)
                finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
                returned_token_count += len(token_ids)
                ttft, tpot = _request_timing(request_output)
                if ttft is not None:
                    ttft_values.append(ttft)
                if tpot is not None:
                    tpot_values.append(tpot)
                handle.write(
                    json.dumps(
                        {
                            "doc_index": doc_index,
                            "sample_index": sample_index,
                            "sample_seed": request_seed,
                            "question": row["question"],
                            "gold_answer": row["gold_answer"],
                            "output": text,
                            "returned_token_count": len(token_ids),
                            "prompt_token_count": prompt_token_counts[doc_index],
                            "finish_reason": finish_reason,
                            "request_id": request_output.request_id,
                            "ttft_seconds": ttft,
                            "tpot_seconds": tpot,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            completed = batch_start + len(request_batch)
            if completed % max(1, args.progress_every) == 0 or completed == len(schedule):
                elapsed = time.perf_counter() - evaluation_started
                _write_json(
                    progress_path,
                    {
                        "status": (
                            "RUNNING" if completed < len(schedule) else "GENERATION_COMPLETE"
                        ),
                        "rank": args.rank,
                        "world_size": args.world_size,
                        "completed": completed,
                        "total": len(schedule),
                        "returned_token_count": returned_token_count,
                        "returned_tokens_per_second": (
                            returned_token_count / elapsed if elapsed > 0 else 0.0
                        ),
                        "updated_at": _utc_now(),
                    },
                )
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_finished_epoch = time.time()
    evaluation_finished_at = _utc_now()
    artifact_after = _model_fingerprint(args.model_dir)
    artifact_mutated = artifact_before != artifact_after
    if artifact_mutated:
        raise RuntimeError("source model artifact changed during GSM8K generation")
    result = {
        "status": "GSM8K_VLLM_GENERATION_OK",
        "artifact_mutated": artifact_mutated,
        "model_family": args.model_family,
        "rank": args.rank,
        "world_size": args.world_size,
        "schedule_position_rule": ("doc_sample_pairs_doc_major[rank::world_size]"),
        "sample_count": len(schedule),
        "dataset_sample_count": len(input_rows),
        "samples_per_doc": args.samples_per_doc,
        "batch_size": args.batch_size,
        "scheduler_queue_size": scheduler_queue_size,
        "max_num_seqs": args.batch_size,
        "max_model_len": args.max_model_len,
        "sampling_seed": args.sampling_seed,
        "sampling_seed_semantics": (
            "sha256(base_seed,doc_index,sample_index) uint32; "
            "independent of rank and batch shape"
        ),
        "temperature": STANDARD_TEMPERATURE,
        "top_p": STANDARD_TOP_P,
        "max_new_tokens": STANDARD_MAX_GEN_TOKS,
        "stop_strings": STOP_STRINGS,
        "execution_mode": args.execution_mode,
        "enforce_eager": enforce_eager,
        "attention_backend": args.attention_backend,
        "flash_attn_version": args.flash_attn_version,
        "hlm_attention_impl": args.hlm_attention_impl,
        "prefix_caching": False,
        "speculative_decoding": bool(args.speculative_draft_model),
        "speculative_method": (
            "ncp_dflash_vllm_0_13" if args.speculative_draft_model else "disabled"
        ),
        "speculative_verification_mode": (
            verification_mode if args.speculative_draft_model else "not_applicable"
        ),
        "speculative_output_contract": (
            ("approximate" if verification_mode == "segmented_kv_approx" else "target_exact")
            if args.speculative_draft_model
            else "not_applicable"
        ),
        "speculative_num_tokens": (
            args.speculative_num_tokens if args.speculative_draft_model else 0
        ),
        "speculative_draft_attention_backend": (
            os.environ.get("CONCEPTLM_DFLASH_ATTENTION_BACKEND", "sdpa")
            if args.speculative_draft_model
            else "disabled"
        ),
        "speculative_context_kv_cache": (
            os.environ.get("CONCEPTLM_DFLASH_CONTEXT_KV_CACHE", "0") == "1"
            if args.speculative_draft_model
            else False
        ),
        "speculative_sparse_context_projection": (
            os.environ.get("CONCEPTLM_DFLASH_SPARSE_CONTEXT_PROJECTION", "0") == "1"
            if args.speculative_draft_model
            else False
        ),
        "speculative_min_eligible_batch": (
            int(os.environ.get("CONCEPTLM_DFLASH_MIN_ELIGIBLE_BATCH", "1"))
            if args.speculative_draft_model
            else 0
        ),
        "speculative_min_proposal_tokens_per_row": (
            int(os.environ.get("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW", "1"))
            if args.speculative_draft_model
            else 0
        ),
        "speculative_min_proposal_tokens_per_batch": (
            int(os.environ.get("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH", "1"))
            if args.speculative_draft_model
            else 0
        ),
        "speculative_runtime_block_size": (
            int(os.environ.get("CONCEPTLM_DFLASH_RUNTIME_BLOCK_SIZE", "0"))
            if args.speculative_draft_model
            else 0
        ),
        "speculative_active_batch_widths": (
            os.environ.get("CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS", "")
            if args.speculative_draft_model
            else ""
        ),
        "speculative_dynamic_runtime_block_size": (
            os.environ.get("CONCEPTLM_DFLASH_DYNAMIC_RUNTIME_BLOCK_SIZE", "0") == "1"
            if args.speculative_draft_model
            else False
        ),
        "speculative_runtime_layer_count": (
            int(os.environ.get("CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT", "0"))
            if args.speculative_draft_model
            else 0
        ),
        "speculative_runtime_local_mixer": (
            os.environ.get("CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER", "full")
            if args.speculative_draft_model
            else "disabled"
        ),
        "speculative_mixer_compile_mode": (
            os.environ.get("CONCEPTLM_DFLASH_MIXER_COMPILE_MODE", "default")
            if args.speculative_draft_model
            else "disabled"
        ),
        "cuda_graph_mode": "NONE" if enforce_eager else "PIECEWISE",
        "model_dir": str(args.model_dir.resolve()),
        "model_artifact": artifact_after,
        "input_manifest": input_manifest,
        "input_manifest_sha256": _file_sha256(args.input_dir / "manifest.json"),
        "prompt_token_count": {
            "min": min(prompt_token_counts) if prompt_token_counts else 0,
            "max": max(prompt_token_counts) if prompt_token_counts else 0,
            "sum": sum(prompt_token_counts[doc_index] for doc_index, _ in schedule),
        },
        "returned_token_count": returned_token_count,
        "finish_reasons": finish_reasons,
        "mean_ttft_seconds": (sum(ttft_values) / len(ttft_values) if ttft_values else None),
        "mean_tpot_seconds": (sum(tpot_values) / len(tpot_values) if tpot_values else None),
        "returned_tokens_per_second": (
            returned_token_count / evaluation_seconds if evaluation_seconds > 0 else 0.0
        ),
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_started_epoch": evaluation_started_epoch,
        "evaluation_finished_epoch": evaluation_finished_epoch,
        "evaluation_started_at": evaluation_started_at,
        "evaluation_finished_at": evaluation_finished_at,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "vllm_version": vllm_version,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "predictions_jsonl": str(predictions_path.resolve()),
        "progress_json": str(progress_path.resolve()),
    }
    _write_json(shard_dir / "result.json", result)
    (shard_dir / "_SUCCESS").touch()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def _require_consistent(results: list[dict[str, Any]], field: str) -> Any:
    values = [result[field] for result in results]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"inconsistent shard field {field}: {values!r}")
    return values[0]


def _model_artifact_identity(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the stable identity used when aggregating resumed shards.

    Generated model overlays can be materialized again between an initial run
    and a rank-level resume. Their mtimes then differ even though the resolved
    source files and content are unchanged. Keep mtimes in the recorded
    fingerprint for auditing, but exclude them from cross-shard identity.
    """

    return {
        "model_dir": artifact["model_dir"],
        "files": [
            {
                key: row[key]
                for key in ("name", "resolved_path", "size", "sha256")
                if key in row
            }
            for row in artifact["files"]
        ],
    }


def _require_consistent_model_artifact(results: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts = [result["model_artifact"] for result in results]
    identities = [_model_artifact_identity(artifact) for artifact in artifacts]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(f"inconsistent shard field model_artifact: {identities!r}")
    return artifacts[0]


def aggregate(args: argparse.Namespace) -> None:
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    if args.samples_per_doc <= 0:
        raise ValueError("samples per doc must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    task, docs, task_contract, dataset_test_path = _resolve_protocol(args)
    input_manifest, input_rows = _load_and_validate_prepared_inputs(args.input_dir)
    if len(docs) != int(input_manifest["dataset_sample_count"]):
        raise ValueError("aggregate dataset count differs from prepared inputs")
    if len(input_rows) != len(docs):
        raise ValueError("aggregate input row count differs from loaded dataset")
    if str(args.standard_input_config.resolve()) != str(
        Path(str(input_manifest["standard_input_config"])).resolve()
    ):
        raise ValueError("aggregate standard input config differs from preparation")
    if _file_sha256(args.standard_input_config) != input_manifest.get(
        "standard_input_config_sha256"
    ):
        raise ValueError("aggregate standard input config hash differs from preparation")
    if str(dataset_test_path.resolve()) != str(
        Path(str(input_manifest["dataset_test_file"])).resolve()
    ):
        raise ValueError("aggregate dataset path differs from preparation")
    if _file_sha256(dataset_test_path) != input_manifest.get("dataset_test_sha256"):
        raise ValueError("aggregate dataset hash differs from preparation")
    results = []
    predictions = []
    for rank in range(args.world_size):
        shard_dir = args.output_dir / f"shard-{rank:02d}"
        if not (shard_dir / "_SUCCESS").is_file():
            raise FileNotFoundError(f"missing shard success marker: {shard_dir}")
        result = _read_json(shard_dir / "result.json")
        if int(result["rank"]) != rank:
            raise ValueError(f"shard result rank mismatch in {shard_dir}")
        results.append(result)
        predictions.extend(_read_jsonl(shard_dir / "predictions.jsonl"))

    for field in (
        "world_size",
        "dataset_sample_count",
        "samples_per_doc",
        "batch_size",
        "scheduler_queue_size",
        "max_num_seqs",
        "max_model_len",
        "sampling_seed",
        "sampling_seed_semantics",
        "temperature",
        "top_p",
        "max_new_tokens",
        "stop_strings",
        "execution_mode",
        "attention_backend",
        "flash_attn_version",
        "hlm_attention_impl",
        "speculative_decoding",
        "speculative_method",
        "speculative_num_tokens",
        "speculative_draft_attention_backend",
        "speculative_context_kv_cache",
        "speculative_sparse_context_projection",
        "speculative_min_eligible_batch",
        "speculative_min_proposal_tokens_per_row",
        "speculative_min_proposal_tokens_per_batch",
        "speculative_runtime_block_size",
        "speculative_active_batch_widths",
        "speculative_dynamic_runtime_block_size",
        "speculative_runtime_layer_count",
        "model_family",
        "input_manifest_sha256",
    ):
        _require_consistent(results, field)
    model_artifact = _require_consistent_model_artifact(results)
    predictions.sort(key=lambda row: (int(row["doc_index"]), int(row["sample_index"])))
    request_keys = [(int(row["doc_index"]), int(row["sample_index"])) for row in predictions]
    expected_keys = [
        (doc_index, sample_index)
        for doc_index in range(len(docs))
        for sample_index in range(args.samples_per_doc)
    ]
    if request_keys != expected_keys:
        raise ValueError(
            "rank-sharded prediction coverage is not exact: "
            f"found={len(request_keys)} expected={len(expected_keys)}"
        )

    scored_rows = []
    pass_at_1_correct = 0
    correct_by_doc: dict[int, list[bool]] = {doc_index: [] for doc_index in range(len(docs))}
    for row in predictions:
        doc_index = int(row["doc_index"])
        score = _score_prediction(docs[doc_index], str(row["output"]))
        standard_correct = bool(score["standard_correct"])
        pass_at_1_correct += int(standard_correct)
        correct_by_doc[doc_index].append(standard_correct)
        scored_rows.append({**row, **score})
    _write_jsonl(args.output_dir / "predictions.jsonl", scored_rows)

    evaluation_started_epoch = min(float(result["evaluation_started_epoch"]) for result in results)
    evaluation_finished_epoch = max(
        float(result["evaluation_finished_epoch"]) for result in results
    )
    parallel_evaluation_seconds = evaluation_finished_epoch - evaluation_started_epoch
    returned_token_count = sum(int(result["returned_token_count"]) for result in results)
    pass_at_samples_correct = sum(any(correctness) for correctness in correct_by_doc.values())
    prompt_sha = str(input_manifest["prompt_sha256"])
    if prompt_sha != _prompt_sha256(
        [task.fewshot_context(doc, num_fewshot=task_contract["num_fewshot"]) for doc in docs]
    ):
        raise ValueError("prepared GSM8K prompt hash changed during aggregation")
    model_family = _require_consistent(results, "model_family")
    if model_family != args.model_family:
        raise ValueError(
            f"aggregate model family {args.model_family!r} does not match "
            f"generation shards {model_family!r}"
        )
    aggregate_result = {
        "status": "GSM8K_VLLM_EVAL_OK",
        "model_identifier": str(model_artifact["model_dir"]),
        "model_family": model_family,
        "backend": "native_vllm",
        "task": args.task_name,
        "task_group": STANDARD_TASK_GROUP,
        "task_contract": task_contract,
        "standard_input_config": str(args.standard_input_config.resolve()),
        "standard_input_config_sha256": _file_sha256(args.standard_input_config),
        "dataset_test_file": str(dataset_test_path),
        "dataset_test_sha256": _file_sha256(dataset_test_path),
        "dataset_sample_count": len(docs),
        "sample_count": len(scored_rows),
        "samples_per_doc": args.samples_per_doc,
        "num_fewshot": task_contract["num_fewshot"],
        "fewshot_sampler": task_contract["fewshot_sampler"],
        "fixed_fewshot_samples": task_contract["fixed_fewshot_samples"],
        "fewshot_seed": args.fewshot_seed,
        "sampling_seed": _require_consistent(results, "sampling_seed"),
        "sampling_seed_semantics": _require_consistent(results, "sampling_seed_semantics"),
        "prompt_sha256": prompt_sha,
        "batch_size": _require_consistent(results, "batch_size"),
        "scheduler_queue_size": _require_consistent(results, "scheduler_queue_size"),
        "max_num_seqs": _require_consistent(results, "max_num_seqs"),
        "max_model_len": _require_consistent(results, "max_model_len"),
        "gpu_count": args.world_size,
        "shard_count": args.world_size,
        "schedule_position_rule": ("doc_sample_pairs_doc_major[rank::world_size]"),
        "temperature": STANDARD_TEMPERATURE,
        "top_p": STANDARD_TOP_P,
        "max_new_tokens": STANDARD_MAX_GEN_TOKS,
        "stop_strings": STOP_STRINGS,
        "primary_metric": "pass@1",
        "answer_scorer": "olmo_eval_gsm_last_number_exact_match",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "official_scorer_source_sha256s": sorted(
            {source_sha for row in scored_rows for source_sha in row["official_source_sha256s"]}
        ),
        "pass_at_1_correct": pass_at_1_correct,
        "pass_at_1": (pass_at_1_correct / len(scored_rows) if scored_rows else 0.0),
        "pass_at_samples_per_doc_k": args.samples_per_doc,
        "pass_at_samples_per_doc_correct": pass_at_samples_correct,
        "pass_at_samples_per_doc": (pass_at_samples_correct / len(docs) if docs else 0.0),
        "returned_token_count": returned_token_count,
        "parallel_evaluation_seconds": parallel_evaluation_seconds,
        "aggregate_returned_tokens_per_second": (
            returned_token_count / parallel_evaluation_seconds
            if parallel_evaluation_seconds > 0
            else 0.0
        ),
        "execution_mode": _require_consistent(results, "execution_mode"),
        "attention_backend": _require_consistent(results, "attention_backend"),
        "flash_attn_version": _require_consistent(results, "flash_attn_version"),
        "hlm_attention_impl": _require_consistent(results, "hlm_attention_impl"),
        "prefix_caching": False,
        "speculative_decoding": _require_consistent(results, "speculative_decoding"),
        "speculative_method": _require_consistent(results, "speculative_method"),
        "speculative_num_tokens": _require_consistent(results, "speculative_num_tokens"),
        "speculative_draft_attention_backend": _require_consistent(
            results, "speculative_draft_attention_backend"
        ),
        "speculative_context_kv_cache": _require_consistent(
            results, "speculative_context_kv_cache"
        ),
        "speculative_sparse_context_projection": _require_consistent(
            results, "speculative_sparse_context_projection"
        ),
        "speculative_min_eligible_batch": _require_consistent(
            results, "speculative_min_eligible_batch"
        ),
        "speculative_min_proposal_tokens_per_row": _require_consistent(
            results, "speculative_min_proposal_tokens_per_row"
        ),
        "speculative_min_proposal_tokens_per_batch": _require_consistent(
            results, "speculative_min_proposal_tokens_per_batch"
        ),
        "speculative_runtime_block_size": _require_consistent(
            results, "speculative_runtime_block_size"
        ),
        "speculative_active_batch_widths": _require_consistent(
            results, "speculative_active_batch_widths"
        ),
        "speculative_dynamic_runtime_block_size": _require_consistent(
            results, "speculative_dynamic_runtime_block_size"
        ),
        "speculative_runtime_layer_count": _require_consistent(
            results, "speculative_runtime_layer_count"
        ),
        "model_artifact": model_artifact,
        "vllm_version": _require_consistent(results, "vllm_version"),
        "torch_version": _require_consistent(results, "torch_version"),
        "devices": [result["device"] for result in results],
        "shards": [
            {
                "rank": result["rank"],
                "sample_count": result["sample_count"],
                "returned_token_count": result["returned_token_count"],
                "returned_tokens_per_second": result["returned_tokens_per_second"],
                "evaluation_seconds": result["evaluation_seconds"],
                "peak_gpu_memory_gib": result["peak_gpu_memory_gib"],
            }
            for result in results
        ],
        "predictions_jsonl": str((args.output_dir / "predictions.jsonl").resolve()),
        "completed_at": _utc_now(),
    }
    if args.samples_per_doc > 1:
        aggregate_result[f"pass_at_{args.samples_per_doc}"] = aggregate_result[
            "pass_at_samples_per_doc"
        ]
    _write_json(args.output_dir / "aggregate.json", aggregate_result)
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(aggregate_result, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "generate":
        generate(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
