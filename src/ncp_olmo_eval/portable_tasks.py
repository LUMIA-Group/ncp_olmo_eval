"""Portable multi-GPU task payloads used by scheduler-neutral task specs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _run(
    argv: Sequence[str], *, env: dict[str, str] | None = None, log: Path | None = None
) -> None:
    if log is None:
        subprocess.run(list(argv), check=True, env=env)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(list(argv), check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)


def _spawn_all(commands: Sequence[tuple[list[str], dict[str, str], Path]]) -> None:
    processes: list[tuple[subprocess.Popen[bytes], Path]] = []
    handles = []
    try:
        for argv, env, log in commands:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("ab")
            handles.append(handle)
            processes.append(
                (subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT), log)
            )
        failures = []
        for process, log in processes:
            returncode = process.wait()
            if returncode:
                failures.append(f"{log}: exit {returncode}")
        if failures:
            raise RuntimeError("one or more workers failed: " + "; ".join(failures))
    finally:
        for handle in handles:
            handle.close()


def _module(name: str, *args: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(value) for value in args)]


def run_gsm8k(args: argparse.Namespace) -> None:
    if args.fewshot_seed != 42 or args.sampling_seed != 42:
        raise ValueError("GSM8K fixes both seeds to 42")
    if args.batch_size <= 0:
        raise ValueError("GSM8K batch size must be positive")
    if args.speculative_draft_model and args.model_family != "conceptlm":
        raise ValueError("NCP DFlash requires model-family=conceptlm")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    overlay = output / "model"
    prepare_manifest = output / "model-preparation.json"
    prepare = _module(
        "ncp_olmo_eval.prepare_native_vllm_model",
        "--source-model",
        args.model,
        "--overlay-dir",
        overlay,
        "--model-family",
        args.model_family,
        "--output-manifest",
        prepare_manifest,
    )
    if args.runtime_config:
        prepare.extend(["--runtime-config", str(args.runtime_config)])
    model_dir = subprocess.run(prepare, check=True, capture_output=True, text=True).stdout.strip()
    if not model_dir:
        raise RuntimeError("model overlay preparation returned an empty path")

    inputs = output / "inputs"
    _run(
        _module(
            "ncp_olmo_eval.vllm_plugin.gsm8k",
            "prepare",
            "--output-dir",
            inputs,
            "--standard-input-config",
            args.standard_input_config,
            "--task-name",
            "olmo_eval_paper_gsm8k_main",
            "--task-include-path",
            args.task_include_path,
            "--dataset-test-file",
            args.dataset_test_file,
            "--fewshot-seed",
            args.fewshot_seed,
            "--limit",
            0,
        ),
        log=output / "prepare.log",
    )
    if args.workflow_manifest:
        _run(
            _module(
                "ncp_olmo_eval.core88_workflow",
                "verify-inputs",
                "--workflow-manifest",
                args.workflow_manifest,
                "--gsm8k-root",
                output,
            ),
            log=output / "verify-inputs.log",
        )

    commands = []
    for rank in range(args.gpus):
        shard = output / f"shard-{rank:02d}"
        if args.resume and (shard / "_SUCCESS").is_file() and (shard / "result.json").is_file():
            continue
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(rank),
                "RANK": str(rank),
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(args.gpus),
                "CONCEPTLM_VLLM_ENABLE_UNVERIFIED": "1",
                "CONCEPTLM_HLM_ATTENTION_IMPL": args.hlm_attention_impl,
                "PYTHONHASHSEED": "42",
            }
        )
        generate_argv = _module(
            "ncp_olmo_eval.vllm_plugin.gsm8k",
            "generate",
                    "--input-dir",
                    inputs,
                    "--output-dir",
                    output,
                    "--model-dir",
                    model_dir,
                    "--model-family",
                    args.model_family,
                    "--rank",
                    rank,
                    "--world-size",
                    args.gpus,
                    "--sampling-seed",
                    args.sampling_seed,
                    "--samples-per-doc",
                    1,
                    "--batch-size",
                    args.batch_size,
                    "--scheduler-queue-size",
                    args.scheduler_queue_size,
                    "--max-model-len",
                    args.max_model_len,
                    "--warmup-tokens",
                    2,
                    "--progress-every",
                    10,
                    "--gpu-memory-utilization",
                    args.gpu_memory_utilization,
                    "--execution-mode",
                    args.execution_mode,
                    "--attention-backend",
                    args.attention_backend,
                    "--flash-attn-version",
                    args.flash_attn_version,
            "--hlm-attention-impl",
            args.hlm_attention_impl,
        )
        if args.speculative_draft_model:
            generate_argv.extend(
                [
                    "--speculative-draft-model",
                    str(args.speculative_draft_model),
                    "--speculative-num-tokens",
                    str(args.speculative_num_tokens),
                    "--speculative-verification-mode",
                    args.speculative_verification_mode,
                    "--speculative-telemetry-path",
                    str(shard / "ncp-dflash-telemetry.jsonl"),
                ]
            )
        commands.append(
            (
                generate_argv,
                env,
                output / f"rank-{rank}.log",
            )
        )
    _spawn_all(commands)
    _run(
        _module(
            "ncp_olmo_eval.vllm_plugin.gsm8k",
            "aggregate",
            "--input-dir",
            inputs,
            "--output-dir",
            output,
            "--world-size",
            args.gpus,
            "--samples-per-doc",
            1,
            "--batch-size",
            args.batch_size,
            "--model-family",
            args.model_family,
            "--standard-input-config",
            args.standard_input_config,
            "--task-name",
            "olmo_eval_paper_gsm8k_main",
            "--task-include-path",
            args.task_include_path,
            "--dataset-test-file",
            args.dataset_test_file,
            "--fewshot-seed",
            args.fewshot_seed,
            "--limit",
            0,
        ),
        log=output / "aggregate.log",
    )
    if args.workflow_manifest:
        _run(
            _module(
                "ncp_olmo_eval.core88_workflow",
                "seal",
                "--workflow-manifest",
                args.workflow_manifest,
                "--gsm8k-root",
                output,
                "--repo-commit",
                args.repo_commit,
                "--model-identity-path",
                args.model,
                "--runtime-model-path",
                model_dir,
                "--model-family",
                args.model_family,
                "--model-preparation-manifest",
                prepare_manifest,
            ),
            log=output / "core88-companion.log",
        )


def run_long_context(args: argparse.Namespace) -> None:
    if args.global_seed != 42:
        raise ValueError("long-context evaluation fixes seed=42")
    if args.gpus % args.tensor_parallel_size:
        raise ValueError("tensor parallel size must divide the GPU count")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine_count = args.gpus // args.tensor_parallel_size
    common = _module(
        "ncp_olmo_eval.long_context_inference",
        "--data-root",
        args.data_root,
        "--output-root",
        output,
        "--model-label",
        args.model_label,
        "--backend",
        "native_vllm",
        "--hf-model-path",
        args.model,
        "--model-identity-path",
        args.model,
        "--model-config-path",
        args.model_config,
        "--tokenizer-model",
        args.tokenizer,
        "--batch-size",
        4,
        "--samples-per-example",
        1,
        "--global-seed",
        42,
        "--limit",
        0,
        "--limit-per-task",
        0,
        "--ruler-sequence-length",
        0,
        "--max-new-tokens",
        0,
        "--model-context-length",
        65536,
        "--vllm-runtime-config",
        args.runtime_config or "",
        "--vllm-model-family",
        args.model_family,
        "--vllm-gpu-memory-utilization",
        args.gpu_memory_utilization,
        "--vllm-execution-mode",
        args.execution_mode,
        "--vllm-attention-backend",
        args.attention_backend,
        "--vllm-flash-attn-version",
        args.flash_attn_version,
        "--vllm-hlm-attention-impl",
        "uniform_flash",
        "--vllm-tensor-parallel-size",
        args.tensor_parallel_size,
        "--allow-unverified-native-vllm",
        "--no-hf-compile-routes",
        "--no-hf-align-dcp-runtime-config",
        "--no-allow-context-extension",
        "--resume" if args.resume else "--no-resume",
    )
    commands = []
    for engine in range(engine_count):
        visible = ",".join(
            str(engine * args.tensor_parallel_size + offset)
            for offset in range(args.tensor_parallel_size)
        )
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": visible,
                "PYTHONHASHSEED": "42",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "0",
                "CONCEPTLM_BENCH_FLASH_DECODE": "0",
                "CONCEPTLM_BENCH_CUDA_GRAPH": "0",
                "CONCEPTLM_PROCESSES_PER_GPU": "1",
            }
        )
        argv = [
            *common,
            "--shard-index",
            str(engine),
            "--shard-count",
            str(engine_count),
            "--vllm-model-overlay-dir",
            str(output / f"native-vllm-model-{engine}"),
        ]
        commands.append((argv, env, output / f"shard-{engine:03d}.log"))
    _spawn_all(commands)


def run_core_final(args: argparse.Namespace) -> None:
    """Fail closed on stale cached evidence, then rescore raw artifacts."""

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite final output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output_json = output / "core88-results.json"
    output_csv = output / "core88-scores-88.csv"
    summary_csv = output / "core88-summary-30cols.csv"
    summary_json = output / "core88-summary-30cols.json"
    cached = _module(
        "ncp_olmo_eval.core_native_cached_finalize",
        "--base-report",
        args.base_report,
        "--output-json",
        output_json,
        "--output-csv",
        output_csv,
        "--output-summary-csv",
        summary_csv,
        "--output-summary-metadata-json",
        summary_json,
        "--gsm8k-results-root",
        args.gsm8k_results_root,
        "--workflow-manifest",
        args.workflow_manifest,
        "--verify-code-result-hashes",
    )
    for directory in args.code_results_dir:
        cached.extend(["--code-summary-dir", str(directory)])
    cached_result = subprocess.run(
        cached, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    (output / "cached-finalize.log").write_text(cached_result.stdout, encoding="utf-8")
    if cached_result.returncode:
        for path in (output_json, output_csv, summary_csv, summary_json):
            path.unlink(missing_ok=True)
        raw = _module(
            "ncp_olmo_eval.core_native_aggregate",
            "--data-root",
            args.data_root,
            "--profile",
            "core88",
            "--input-root",
            args.inference_root,
            "--output-json",
            output_json,
            "--output-csv",
            output_csv,
            "--output-summary-csv",
            summary_csv,
            "--output-summary-metadata-json",
            summary_json,
            "--workflow-manifest",
            args.workflow_manifest,
            "--gsm8k-results-root",
            args.gsm8k_results_root,
        )
        for directory in args.code_results_dir:
            raw.extend(["--code-results-dir", str(directory)])
        _run(raw, log=output / "raw-rescore.log")
    for path in (output_json, output_csv, summary_csv, summary_json):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"finalization did not produce {path}")
    (output / "_SUCCESS").touch()


def run_sciq_score(args: argparse.Namespace) -> None:
    """Rescore one sealed SciQ run and materialize its raw-accuracy result."""

    from .evaluation_cli import (
        SCIQ_EXAMPLE_COUNT,
        SCIQ_SOURCE_PROFILE,
        SCIQ_TASK_NAME,
        SCIQ_TASK_ORDER,
    )

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_json = output / "score.json"
    score_csv = output / "sciq-score.csv"
    _run(
        _module(
            "ncp_olmo_eval.core_native_aggregate",
            "--data-root",
            args.data_root,
            "--profile",
            SCIQ_SOURCE_PROFILE,
            "--task-orders",
            SCIQ_TASK_ORDER,
            "--input-root",
            args.inference_root,
            "--output-json",
            score_json,
            "--output-csv",
            score_csv,
        ),
        log=output / "score.log",
    )
    payload = json.loads(score_json.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    task = tasks[0] if isinstance(tasks, list) and len(tasks) == 1 else None
    if (
        payload.get("status") != "CORE_NATIVE_FULL_OK"
        or payload.get("profile") != SCIQ_SOURCE_PROFILE
        or payload.get("task_orders") != [SCIQ_TASK_ORDER]
        or int(payload.get("expected_predictions", -1)) != SCIQ_EXAMPLE_COUNT
        or int(payload.get("observed_predictions", -1)) != SCIQ_EXAMPLE_COUNT
        or not isinstance(task, dict)
        or task.get("task_order") != SCIQ_TASK_ORDER
        or task.get("task") != SCIQ_TASK_NAME
        or task.get("metric") != "acc"
        or task.get("request_type") != "loglikelihood"
        or task.get("score_status") != "SCORED"
    ):
        raise RuntimeError(f"SciQ score contract is incomplete: {payload}")
    if not score_csv.is_file() or score_csv.stat().st_size == 0:
        raise RuntimeError("SciQ scorer did not produce sciq-score.csv")
    (output / "_SUCCESS").touch()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    gsm = subparsers.add_parser("gsm8k")
    gsm.add_argument("--model", type=Path, required=True)
    gsm.add_argument("--output-root", type=Path, required=True)
    gsm.add_argument("--standard-input-config", type=Path, required=True)
    gsm.add_argument("--task-include-path", type=Path, required=True)
    gsm.add_argument("--dataset-test-file", type=Path, required=True)
    gsm.add_argument("--model-family", choices=("auto", "conceptlm"), required=True)
    gsm.add_argument("--runtime-config", type=Path)
    gsm.add_argument("--gpus", type=int, default=8)
    gsm.add_argument("--fewshot-seed", type=int, default=42)
    gsm.add_argument("--sampling-seed", type=int, default=42)
    gsm.add_argument("--batch-size", type=int, default=8)
    gsm.add_argument(
        "--scheduler-queue-size",
        type=int,
        default=0,
        help="requests submitted per vLLM call; zero uses batch-size",
    )
    gsm.add_argument("--max-model-len", type=int, default=2048)
    gsm.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    gsm.add_argument("--execution-mode", choices=("eager", "piecewise"), default="eager")
    gsm.add_argument("--attention-backend", default="FLASH_ATTN")
    gsm.add_argument("--flash-attn-version", type=int, choices=(2, 3), default=3)
    gsm.add_argument("--hlm-attention-impl", default="legacy_mixed")
    gsm.add_argument("--speculative-draft-model", type=Path)
    gsm.add_argument("--speculative-num-tokens", type=int, default=16)
    gsm.add_argument(
        "--speculative-verification-mode",
        choices=(
            "sequential_exact",
            "intra_chunk_exact",
            "segmented_kv_approx",
            "transactional_exact",
            "chunk_parallel",
        ),
        default="sequential_exact",
    )
    gsm.add_argument("--workflow-manifest", type=Path)
    gsm.add_argument("--repo-commit", default="")
    gsm.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    gsm.set_defaults(handler=run_gsm8k)

    long_context = subparsers.add_parser("long-context")
    long_context.add_argument("--data-root", type=Path, required=True)
    long_context.add_argument("--output-root", type=Path, required=True)
    long_context.add_argument("--model", type=Path, required=True)
    long_context.add_argument("--model-config", type=Path, required=True)
    long_context.add_argument("--tokenizer", type=Path, required=True)
    long_context.add_argument("--model-label", required=True)
    long_context.add_argument("--model-family", choices=("auto", "conceptlm"), required=True)
    long_context.add_argument("--runtime-config", type=Path)
    long_context.add_argument("--gpus", type=int, default=8)
    long_context.add_argument("--tensor-parallel-size", type=int, default=1)
    long_context.add_argument("--global-seed", type=int, default=42)
    long_context.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    long_context.add_argument("--execution-mode", choices=("eager", "piecewise"), default="eager")
    long_context.add_argument("--attention-backend", default="FLASH_ATTN")
    long_context.add_argument("--flash-attn-version", type=int, choices=(2, 3), default=3)
    long_context.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    long_context.set_defaults(handler=run_long_context)

    sciq_score = subparsers.add_parser("score-sciq")
    sciq_score.add_argument("--data-root", type=Path, required=True)
    sciq_score.add_argument("--inference-root", type=Path, required=True)
    sciq_score.add_argument("--output-root", type=Path, required=True)
    sciq_score.set_defaults(handler=run_sciq_score)

    final_core = subparsers.add_parser("final-core")
    final_core.add_argument("--data-root", type=Path, required=True)
    final_core.add_argument("--inference-root", type=Path, required=True)
    final_core.add_argument("--base-report", type=Path, required=True)
    final_core.add_argument("--code-results-dir", type=Path, action="append", required=True)
    final_core.add_argument("--gsm8k-results-root", type=Path, required=True)
    final_core.add_argument("--workflow-manifest", type=Path, required=True)
    final_core.add_argument("--output-root", type=Path, required=True)
    final_core.set_defaults(handler=run_core_final)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.handler(args)
    print(json.dumps({"status": "PORTABLE_TASK_OK", "mode": args.mode}))


if __name__ == "__main__":
    main()
