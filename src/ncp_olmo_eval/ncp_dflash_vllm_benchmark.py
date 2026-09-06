"""Correctness-first target-only versus NCP DFlash vLLM benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .dflash_checkpoint import dflash_checkpoint_identity
from .dflash_contract import (
    dflash_operating_point_from_runtime,
    validate_dflash_operating_point,
)
from .inference import SamplingParams
from .native_vllm_inference import (
    NCP_DFLASH_APPROXIMATE_VERIFICATION_MODES,
    NativeVLLMInferencer,
    normalize_ncp_dflash_verification_mode,
    prepare_native_vllm_model,
)


class _NvidiaSmiMemorySampler:
    """Sample whole-GPU memory without creating a parent CUDA context."""

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.samples_mib: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if values:
                self.samples_mib.append(max(values))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if len(self.errors) < 3:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        self._sample()
        peak_mib = max(self.samples_mib) if self.samples_mib else None
        return {
            "source": "nvidia-smi memory.used",
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples_mib),
            "peak_mib": peak_mib,
            "peak_gib": peak_mib / 1024 if peak_mib is not None else None,
            "errors": self.errors,
        }


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prompt_hash(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _draft_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return dflash_checkpoint_identity(path)


def _target_identity(overlay: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_model": overlay["source_model"],
        "runtime_config": overlay["runtime_config"],
        "runtime_config_sha256": overlay["runtime_config_sha256"],
        "weight_files": overlay["weight_files"],
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "event_count": 0,
            "proposed_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate_lower_bound": None,
        }
    events = _read_jsonl(path)
    proposed = sum(
        int(event.get("proposed_tokens", 0))
        for event in events
        if event.get("event") == "proposal_batch"
    )
    legacy_rejected = sum(
        int(event.get("rejected_tokens", 0))
        for event in events
        if event.get("event") == "target_state_rollback"
    )
    transaction_commits = [
        event for event in events if event.get("event") == "target_state_transaction_commit"
    ]
    if transaction_commits:
        accepted = min(
            proposed,
            sum(max(0, int(event.get("committed_tokens", 0)) - 1) for event in transaction_commits),
        )
        # Proposals issued after a request's last target step are deliberately
        # included as not-observed acceptance. This keeps the metric a lower
        # bound instead of reporting them as accepted by default.
        rejected = proposed - accepted
    else:
        rejected = legacy_rejected
        accepted = max(0, proposed - rejected)
    proposal_seconds = sum(
        float(event.get("elapsed_seconds", 0.0))
        for event in events
        if event.get("event") == "proposal_batch"
    )
    return {
        "path": str(path),
        "event_count": len(events),
        "proposed_tokens": proposed,
        "rejected_tokens": rejected,
        "accepted_tokens_observed": accepted,
        "acceptance_rate_lower_bound": (accepted / proposed if proposed else None),
        "proposal_seconds": proposal_seconds,
        "events_by_type": {
            name: sum(event.get("event") == name for event in events)
            for name in sorted({str(event.get("event")) for event in events})
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.speculative_verification_mode = normalize_ncp_dflash_verification_mode(
        args.speculative_verification_mode
    )
    import torch
    import vllm

    if args.seed != 42:
        raise ValueError("the benchmark fixes seed=42")
    if args.offset < 0 or args.limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")
    scheduler_queue_size = args.scheduler_queue_size or args.batch_size
    if scheduler_queue_size < args.batch_size:
        raise ValueError("scheduler queue size must be at least the active batch size")
    rows = _read_jsonl(args.prompt_jsonl)
    prompts = []
    for row in rows:
        prompt = row.get(args.prompt_field)
        if not isinstance(prompt, str):
            raise ValueError(f"prompt field {args.prompt_field!r} is missing")
        prompts.append(prompt)
    prompts = prompts[args.offset : args.offset + args.limit]
    if not prompts:
        raise ValueError("the benchmark has no prompts")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_model, overlay = prepare_native_vllm_model(
        source_model=args.model,
        runtime_config=args.runtime_config,
        overlay_dir=output_dir / "model-overlay",
        model_family="conceptlm",
    )
    telemetry_path = output_dir / "ncp-dflash-telemetry.jsonl"
    memory_sampler = _NvidiaSmiMemorySampler()
    memory_sampler.start()
    try:
        load_started = time.perf_counter()
        inferencer = NativeVLLMInferencer(
            model_path=runtime_model,
            max_model_len=args.max_model_len,
            seed=args.seed,
            max_batch_size=args.batch_size,
            scheduler_queue_size=scheduler_queue_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            execution_mode=args.execution_mode,
            attention_backend="FLASH_ATTN",
            flash_attn_version=3,
            hlm_attention_impl="legacy_mixed",
            model_family="conceptlm",
            speculative_draft_model=(args.draft_model if args.mode == "speculative" else None),
            speculative_num_tokens=args.speculative_num_tokens,
            speculative_verification_mode=args.speculative_verification_mode,
            speculative_telemetry_path=(telemetry_path if args.mode == "speculative" else None),
        )
        model_load_seconds = time.perf_counter() - load_started
        warmup = SamplingParams(
            max_tokens=args.warmup_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=[],
            seed=args.seed,
            ignore_eos=args.ignore_eos,
        )
        inferencer.generate_batch(
            prompts[: args.batch_size], [warmup] * min(args.batch_size, len(prompts))
        )

        completions = []
        latencies = []
        total_tokens = 0
        started = time.perf_counter()
        for batch_start in range(0, len(prompts), scheduler_queue_size):
            prompt_batch = prompts[batch_start : batch_start + scheduler_queue_size]
            samplings = [
                SamplingParams(
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    stop=[],
                    seed=args.seed + batch_start + offset,
                    ignore_eos=args.ignore_eos,
                )
                for offset in range(len(prompt_batch))
            ]
            batch_started = time.perf_counter()
            batch_outputs = inferencer.generate_batch(prompt_batch, samplings)
            batch_elapsed = time.perf_counter() - batch_started
            per_request_latency = batch_elapsed / len(prompt_batch)
            for prompt_index, completion in enumerate(batch_outputs, start=batch_start):
                token_count = len(completion.token_ids)
                total_tokens += token_count
                latencies.append(per_request_latency)
                completions.append(
                    {
                        "prompt_index": prompt_index,
                        "token_ids": completion.token_ids,
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                        "output_token_count": token_count,
                        "latency_seconds": per_request_latency,
                    }
                )
        elapsed = time.perf_counter() - started
    finally:
        gpu_memory = memory_sampler.stop()
    result = {
        "status": "NCP_DFLASH_VLLM_BENCHMARK_OK",
        "mode": args.mode,
        "correctness_status": (
            "AWAITING_CROSS_RUN_COMPARISON" if args.mode == "speculative" else "TARGET_REFERENCE"
        ),
        "seed": args.seed,
        "prompt_count": len(prompts),
        "prompt_offset": args.offset,
        "prompt_sha256": _prompt_hash(prompts),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "scheduler_queue_size": scheduler_queue_size,
        "ignore_eos": args.ignore_eos,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "execution_mode": args.execution_mode,
        "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "auto"),
        "speculative_verification_mode": (
            args.speculative_verification_mode if args.mode == "speculative" else "not_applicable"
        ),
        "model_load_seconds": model_load_seconds,
        "evaluation_seconds": elapsed,
        "output_token_count": total_tokens,
        "output_tokens_per_second": total_tokens / elapsed if elapsed else 0.0,
        "request_latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "peak_gpu_memory_gib": gpu_memory["peak_gib"],
        "gpu_memory": gpu_memory,
        "runtime": inferencer.runtime_metadata,
        "model_overlay": overlay,
        "target_model_identity": _target_identity(overlay),
        "draft_model_identity": _draft_identity(
            args.draft_model if args.mode == "speculative" else None
        ),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "telemetry": _telemetry_summary(telemetry_path),
        "completions": completions,
    }
    _write_json(output_dir / "benchmark.json", result)
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    target = json.loads(args.target_result.read_text(encoding="utf-8"))
    speculative = json.loads(args.speculative_result.read_text(encoding="utf-8"))
    for field in (
        "seed",
        "prompt_count",
        "prompt_offset",
        "prompt_sha256",
        "max_new_tokens",
        "batch_size",
        "scheduler_queue_size",
        "gpu_memory_utilization",
        "execution_mode",
        "vllm_use_v2_model_runner",
        "vllm_version",
        "ignore_eos",
    ):
        if target.get(field) != speculative.get(field):
            raise ValueError(f"benchmark contract differs for {field}")
    if target.get("target_model_identity") != speculative.get("target_model_identity"):
        raise ValueError("target model identity differs between benchmark runs")
    target_runtime = target.get("runtime")
    speculative_runtime = speculative.get("runtime")
    if not isinstance(target_runtime, dict) or not isinstance(speculative_runtime, dict):
        raise ValueError("benchmark runs are missing native-vLLM runtime metadata")
    for field in (
        "max_model_len",
        "max_num_seqs",
        "scheduler_queue_size",
        "tensor_parallel_size",
        "execution_mode",
        "gpu_memory_utilization",
        "attention_backend",
        "flash_attn_version",
        "hlm_attention_impl",
        "vllm_use_v2_model_runner",
    ):
        if target_runtime.get(field) != speculative_runtime.get(field):
            raise ValueError(f"target/speculative runtime differs for {field}")
    draft_identity = speculative.get("draft_model_identity")
    if not isinstance(draft_identity, dict):
        raise ValueError("speculative benchmark is missing its draft identity")
    verification_mode = normalize_ncp_dflash_verification_mode(
        str(speculative.get("speculative_verification_mode", ""))
    )
    approximate_mode = verification_mode in NCP_DFLASH_APPROXIMATE_VERIFICATION_MODES
    allow_output_divergence = bool(getattr(args, "allow_output_divergence", False))
    if allow_output_divergence and not approximate_mode:
        raise ValueError("--allow-output-divergence is valid only for segmented_kv_approx")
    target_outputs = target["completions"]
    speculative_outputs = speculative["completions"]
    exact = []
    first_mismatch: dict[str, Any] | None = None
    for reference, candidate in zip(target_outputs, speculative_outputs, strict=True):
        match = (
            reference["token_ids"] == candidate["token_ids"]
            and reference["finish_reason"] == candidate["finish_reason"]
        )
        exact.append(match)
        if not match and first_mismatch is None:
            token_index = next(
                (
                    index
                    for index, (expected, actual) in enumerate(
                        zip(reference["token_ids"], candidate["token_ids"])
                    )
                    if expected != actual
                ),
                min(len(reference["token_ids"]), len(candidate["token_ids"])),
            )
            first_mismatch = {
                "prompt_index": reference["prompt_index"],
                "generated_token_index": token_index,
                "target_token_ids": reference["token_ids"][token_index : token_index + 16],
                "speculative_token_ids": candidate["token_ids"][token_index : token_index + 16],
                "target_finish_reason": reference["finish_reason"],
                "speculative_finish_reason": candidate["finish_reason"],
            }
    target_tps = float(target["output_tokens_per_second"])
    speculative_tps = float(speculative["output_tokens_per_second"])
    exact_match = all(exact)
    approximate_ab_accepted = approximate_mode and allow_output_divergence
    if approximate_ab_accepted:
        status = "NCP_DFLASH_VLLM_APPROXIMATE_AB_OK"
    elif exact_match:
        status = "NCP_DFLASH_VLLM_EXACT_MATCH_OK"
    else:
        status = "NCP_DFLASH_VLLM_EXACT_MATCH_FAILED"
    benchmark_contract = {
        field: target[field]
        for field in (
            "seed",
            "prompt_count",
            "prompt_offset",
            "prompt_sha256",
            "max_new_tokens",
            "batch_size",
            "scheduler_queue_size",
            "gpu_memory_utilization",
            "vllm_use_v2_model_runner",
            "vllm_version",
            "ignore_eos",
        )
    }
    speculative_operating_point = validate_dflash_operating_point(
        dflash_operating_point_from_runtime(speculative_runtime),
        benchmark_contract=benchmark_contract,
    )
    comparison = {
        "status": status,
        "speculative_output_contract": ("approximate" if approximate_mode else "target_exact"),
        "downstream_score_required": approximate_mode,
        "exact_token_match_count": sum(exact),
        "comparison_count": len(exact),
        "exact_prompt_match_rate": sum(exact) / len(exact) if exact else None,
        "generated_token_count": int(target["output_token_count"]),
        "benchmark_contract": benchmark_contract,
        "speculative_operating_point": speculative_operating_point,
        "target_model_identity": target["target_model_identity"],
        "draft_model_identity": draft_identity,
        "vllm_version": target["vllm_version"],
        "speculative_verification_mode": verification_mode,
        "first_mismatch": first_mismatch,
        "target_output_tokens_per_second": target_tps,
        "speculative_output_tokens_per_second": speculative_tps,
        "throughput_speedup": speculative_tps / target_tps if target_tps else None,
        "target_mean_latency_seconds": target["request_latency_seconds"]["mean"],
        "speculative_mean_latency_seconds": speculative["request_latency_seconds"]["mean"],
        "latency_speedup": (
            float(target["request_latency_seconds"]["mean"])
            / float(speculative["request_latency_seconds"]["mean"])
        ),
        "target_peak_gpu_memory_gib": target["peak_gpu_memory_gib"],
        "speculative_peak_gpu_memory_gib": speculative["peak_gpu_memory_gib"],
        "speculative_telemetry": speculative["telemetry"],
    }
    _write_json(args.output, comparison)
    if not exact_match and not approximate_ab_accepted:
        raise AssertionError(
            "speculative output diverged from target-only vLLM at prompt "
            f"{first_mismatch['prompt_index']}; see {args.output}"
        )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("target", "speculative"), required=True)
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--runtime-config", type=Path)
    run_parser.add_argument("--draft-model", type=Path)
    run_parser.add_argument("--prompt-jsonl", type=Path, required=True)
    run_parser.add_argument("--prompt-field", default="input")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--limit", type=int, default=8)
    run_parser.add_argument("--offset", type=int, default=0)
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument(
        "--scheduler-queue-size",
        type=int,
        default=0,
        help=(
            "requests submitted to one vLLM generate call; values above batch-size "
            "exercise continuous scheduler refill"
        ),
    )
    run_parser.add_argument("--max-new-tokens", type=int, default=64)
    run_parser.add_argument("--warmup-tokens", type=int, default=4)
    run_parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="force the requested token count for a non-trivial correctness gate",
    )
    run_parser.add_argument("--max-model-len", type=int, default=8192)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--speculative-num-tokens", type=int, default=16)
    run_parser.add_argument(
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
            "segmented_kv_approx is the isolated approximate cache/state path; "
            "transactional_exact and chunk_parallel are legacy input aliases"
        ),
    )
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run_parser.add_argument(
        "--execution-mode",
        choices=("eager", "piecewise"),
        default="eager",
        help="target execution mode; piecewise enables attention CUDA graphs",
    )
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--target-result", type=Path, required=True)
    compare_parser.add_argument("--speculative-result", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--allow-output-divergence",
        action="store_true",
        help=(
            "accept and persist divergent output only for segmented_kv_approx; "
            "the artifact remains approximate and requires downstream score A/B"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args) if args.action == "run" else compare(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
