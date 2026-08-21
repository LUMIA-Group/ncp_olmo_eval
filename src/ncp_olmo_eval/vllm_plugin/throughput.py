"""Experimental greedy-throughput benchmark for the parity-gated backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--warmup-prompt-length", type=int, default=32)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "piecewise"),
        default="eager",
    )
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument(
        "--flash-attn-version",
        type=int,
        choices=(2, 3),
        default=3,
    )
    parser.add_argument(
        "--hlm-attention-impl",
        choices=("legacy_mixed", "uniform_flash"),
        default="legacy_mixed",
    )
    parser.add_argument("--accuracy-temperature", type=float, default=0.8)
    parser.add_argument("--accuracy-top-p", type=float, default=0.95)
    parser.add_argument("--accuracy-seed", type=int, default=42)
    parser.add_argument("--accuracy-logprobs", type=int, default=10)
    return parser.parse_args()


def _parse_batch_sizes(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid batch sizes: {raw!r}")
    return values


def _repeat_prompt(seed: list[int], length: int, offset: int) -> list[int]:
    if not seed:
        raise ValueError("prompt seed is empty")
    rotated = seed[offset % len(seed) :] + seed[: offset % len(seed)]
    return (rotated * ((length + len(rotated) - 1) // len(rotated)))[:length]


def _metric_delta(metrics: Any, end: str, start: str) -> float | None:
    end_value = getattr(metrics, end, None)
    start_value = getattr(metrics, start, None)
    if end_value is None or start_value is None:
        return None
    return float(end_value - start_value)


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(statistics.mean(present)) if present else None


def _median_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _ttft(metrics: Any) -> float | None:
    value = getattr(metrics, "first_token_latency", None)
    if value is not None and value > 0:
        return float(value)
    return _metric_delta(metrics, "first_token_time", "arrival_time")


def _tpot(metrics: Any, max_new_tokens: int) -> float | None:
    if max_new_tokens <= 1:
        return None
    decode_time = _metric_delta(metrics, "last_token_ts", "first_token_ts")
    if decode_time is None:
        decode_time = _metric_delta(
            metrics,
            "finished_time",
            "first_token_time",
        )
    if decode_time is None:
        return None
    return decode_time / (max_new_tokens - 1)


def _run_case(
    llm: Any,
    *,
    prompt_seed: list[int],
    batch_size: int,
    prompt_length: int,
    prompt_offset: int,
    max_new_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompt_variants = [
        prompt_offset + index
        for index in range(batch_size)
    ]
    prompts = [
        {
            "prompt_token_ids": _repeat_prompt(
                prompt_seed,
                prompt_length - (variant % 4),
                variant,
            )
        }
        for variant in prompt_variants
    ]
    prompt_indices = {
        tuple(prompt["prompt_token_ids"]): index
        for index, prompt in enumerate(prompts)
    }
    if len(prompt_indices) != len(prompts):
        raise RuntimeError("benchmark prompts must be unique")
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    measurements = []
    for repeat in range(repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        output_lengths = [
            len(output.outputs[0].token_ids)
            for output in outputs
        ]
        if output_lengths != [max_new_tokens] * batch_size:
            raise RuntimeError(
                f"short generation for batch {batch_size}: {output_lengths}"
            )
        ttft = [_ttft(output.metrics) for output in outputs]
        tpot = [_tpot(output.metrics, max_new_tokens) for output in outputs]
        request_rows = []
        for output in outputs:
            prompt_token_ids = tuple(
                int(token_id)
                for token_id in output.prompt_token_ids
            )
            if prompt_token_ids not in prompt_indices:
                raise RuntimeError("vLLM returned an unknown benchmark prompt")
            token_ids = [
                int(token_id)
                for token_id in output.outputs[0].token_ids
            ]
            request_rows.append(
                {
                    "prompt_index": prompt_indices[prompt_token_ids],
                    "request_id": output.request_id,
                    "prompt_length": len(prompt_token_ids),
                    "token_ids": token_ids,
                    "token_sha256": hashlib.sha256(
                        json.dumps(
                            token_ids,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        request_rows.sort(key=lambda row: row["prompt_index"])
        token_digest = hashlib.sha256()
        for row in request_rows:
            token_digest.update(
                json.dumps(
                    row["token_ids"],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        measurements.append(
            {
                "repeat": repeat,
                "wall_seconds": elapsed,
                "output_tokens": batch_size * max_new_tokens,
                "output_tokens_per_second": (
                    batch_size * max_new_tokens / elapsed
                ),
                "requests_per_second": batch_size / elapsed,
                "mean_ttft_seconds": _mean_optional(ttft),
                "mean_tpot_seconds": _mean_optional(tpot),
                "mean_per_request_decode_tokens_per_second": _mean_optional(
                    [
                        None if value is None or value <= 0 else 1.0 / value
                        for value in tpot
                    ]
                ),
                "token_sha256": token_digest.hexdigest(),
                "requests": request_rows,
            }
        )
    repeat_hashes = [row["token_sha256"] for row in measurements]
    return {
        "batch_size": batch_size,
        "prompt_lengths": [
            prompt_length - (variant % 4)
            for variant in prompt_variants
        ],
        "prompt_variants": prompt_variants,
        "max_new_tokens": max_new_tokens,
        "measurements": measurements,
        "repeat_token_hashes_match": len(set(repeat_hashes)) == 1,
        "median_output_tokens_per_second": statistics.median(
            row["output_tokens_per_second"] for row in measurements
        ),
        "median_ttft_seconds": _median_optional(
            [row["mean_ttft_seconds"] for row in measurements]
        ),
        "median_tpot_seconds": _median_optional(
            [row["mean_tpot_seconds"] for row in measurements]
        ),
        "median_per_request_decode_tokens_per_second": _median_optional(
            [
                row["mean_per_request_decode_tokens_per_second"]
                for row in measurements
            ]
        ),
    }


def _run_sampled_accuracy(
    llm: Any,
    *,
    prompt_seed: list[int],
    prompt_length: int,
    prompt_offset: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    logprobs: int,
) -> dict[str, Any]:
    """Run one untimed, fixed-seed sampling trajectory for FA2/FA3 comparison."""

    from vllm import SamplingParams

    prompt = _repeat_prompt(prompt_seed, prompt_length, prompt_offset)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        logprobs=logprobs,
        max_tokens=max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(
        {"prompt_token_ids": prompt},
        sampling,
        use_tqdm=False,
    )
    sample = outputs[0].outputs[0]
    token_ids = [int(token_id) for token_id in sample.token_ids]
    if len(token_ids) != max_new_tokens:
        raise RuntimeError(
            "short sampled accuracy generation: "
            f"{len(token_ids)} != {max_new_tokens}"
        )
    step_logprobs = []
    for step in sample.logprobs or []:
        ranked = sorted(
            (
                {
                    "token_id": int(token_id),
                    "logprob": float(value.logprob),
                    "rank": (
                        None
                        if value.rank is None
                        else int(value.rank)
                    ),
                }
                for token_id, value in step.items()
                if value.rank is not None and int(value.rank) <= logprobs
            ),
            key=lambda row: (
                row["rank"],
                row["token_id"],
            ),
        )
        step_logprobs.append(ranked)
    if len(step_logprobs) != max_new_tokens:
        raise RuntimeError(
            "sampled accuracy logprob length mismatch: "
            f"{len(step_logprobs)} != {max_new_tokens}"
        )
    return {
        "prompt_length": prompt_length,
        "prompt_offset": prompt_offset,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "requested_logprobs": logprobs,
        "token_ids": token_ids,
        "token_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "step_logprobs": step_logprobs,
    }


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm import __version__ as vllm_version

    from .plugin import register

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    if args.prompt_length + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            "prompt length plus generated length exceeds max model length: "
            f"{args.prompt_length} + {args.max_new_tokens} > "
            f"{args.max_model_len}"
        )
    if (
        args.warmup_prompt_length + args.warmup_max_new_tokens
        > args.max_model_len
    ):
        raise ValueError(
            "warmup prompt length plus generated length exceeds max model "
            f"length: {args.warmup_prompt_length} + "
            f"{args.warmup_max_new_tokens} > {args.max_model_len}"
        )
    inputs = json.loads(args.inputs_json.read_text())
    prompt_seed = [int(token_id) for token_id in inputs["greedy_prompt_ids"]]
    os.environ["CONCEPTLM_VLLM_ENABLE_UNVERIFIED"] = "1"
    os.environ["CONCEPTLM_HLM_ATTENTION_IMPL"] = args.hlm_attention_impl
    register()
    enforce_eager = args.execution_mode == "eager"
    compilation_config = (
        None
        if enforce_eager
        else {
            "mode": 3,
            "cudagraph_mode": "PIECEWISE",
            # Keep vLLM's fused linear/norm kernels. The top-level ConceptLM
            # graph owns mutable request state and is intentionally not a
            # torch.compile target; PIECEWISE capture is limited to attention.
            "custom_ops": ["all"],
        }
    )
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        worker_cls="ncp_olmo_eval.vllm_plugin.worker.ConceptLMGPUWorker",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        disable_log_stats=False,
        skip_tokenizer_init=True,
        compilation_config=compilation_config,
        attention_config={
            "backend": args.attention_backend,
            "flash_attn_version": args.flash_attn_version,
        },
    )

    warmup = SamplingParams(
        temperature=0.0,
        max_tokens=args.warmup_max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    warmup_started = time.perf_counter()
    warmup_outputs = llm.generate(
        {
            "prompt_token_ids": _repeat_prompt(
                prompt_seed,
                args.warmup_prompt_length,
                0,
            )
        },
        warmup,
        use_tqdm=False,
    )
    warmup_seconds = time.perf_counter() - warmup_started
    warmup_output_lengths = [
        len(output.outputs[0].token_ids) for output in warmup_outputs
    ]
    if warmup_output_lengths != [args.warmup_max_new_tokens]:
        raise RuntimeError(
            "short warmup generation: "
            f"{warmup_output_lengths}, expected "
            f"{args.warmup_max_new_tokens}"
        )

    cases = [
        _run_case(
            llm,
            prompt_seed=prompt_seed,
            batch_size=batch_size,
            prompt_length=args.prompt_length,
            prompt_offset=args.prompt_offset,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
        )
        for batch_size in batch_sizes
    ]
    sampled_accuracy = _run_sampled_accuracy(
        llm,
        prompt_seed=prompt_seed,
        prompt_length=args.prompt_length,
        prompt_offset=args.prompt_offset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.accuracy_temperature,
        top_p=args.accuracy_top_p,
        seed=args.accuracy_seed,
        logprobs=args.accuracy_logprobs,
    )
    payload = {
        "status": "EXPERIMENTAL_GREEDY_AND_FIXED_SEED_SAMPLING",
        "backend_constraints": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "execution_mode": args.execution_mode,
            "enforce_eager": enforce_eager,
            "attention_backend": args.attention_backend,
            "flash_attn_version": args.flash_attn_version,
            "hlm_attention_impl": args.hlm_attention_impl,
            "hlm_flash_attn_version": (
                args.flash_attn_version
                if args.hlm_attention_impl == "uniform_flash"
                else None
            ),
            "cudagraph_mode": (
                "NONE" if enforce_eager else "PIECEWISE"
            ),
            "cudagraph_scope": (
                "NONE" if enforce_eager else "ATTENTION_CUSTOM_OPS"
            ),
            "top_level_torch_compile": False,
            "enable_prefix_caching": False,
            "speculative_decoding": False,
        },
        "warmup_excluded": True,
        "warmup": {
            "prompt_length": args.warmup_prompt_length,
            "max_new_tokens": args.warmup_max_new_tokens,
            "wall_seconds": warmup_seconds,
        },
        "max_model_len": args.max_model_len,
        "model_dir": args.model_dir,
        "vllm_version": vllm_version,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "cases": cases,
        "sampled_accuracy": sampled_accuracy,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
