"""Diagnose greedy batch-shape sensitivity for one fixed prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .throughput import _repeat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=62)
    parser.add_argument("--prompt-offset", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=13)
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
    return parser.parse_args()


def _trace_summary(path: Path) -> dict[str, Any]:
    logits = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()
    top2 = torch.topk(logits, 2, dim=-1)
    return {
        "file": path.name,
        "rows": int(logits.shape[0]),
        "top1": [
            int(token_id)
            for token_id in torch.argmax(logits, dim=-1)
        ],
        "top2_candidates": [
            [int(token_id) for token_id in row]
            for row in top2.indices
        ],
        "top1_margin": [
            float(value)
            for value in (top2.values[:, 0] - top2.values[:, 1])
        ],
    }


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    from .plugin import register

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    trace_dir = args.output_dir / "traces"
    trace_dir.mkdir()
    inputs = json.loads(args.inputs_json.read_text())
    prompt_seed = [int(token_id) for token_id in inputs["greedy_prompt_ids"]]
    prompt = _repeat_prompt(
        prompt_seed,
        args.prompt_length,
        args.prompt_offset,
    )
    os.environ["CONCEPTLM_VLLM_ENABLE_UNVERIFIED"] = "1"
    os.environ["CONCEPTLM_HLM_ATTENTION_IMPL"] = args.hlm_attention_impl
    os.environ["CONCEPTLM_VLLM_LOGITS_TRACE_DIR"] = str(trace_dir)
    register()
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        worker_cls="ncp_olmo_eval.vllm_plugin.worker.ConceptLMGPUWorker",
        max_model_len=128,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        skip_tokenizer_init=True,
        attention_config={
            "backend": "FLASH_ATTN",
            "flash_attn_version": args.flash_attn_version,
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    single = llm.generate(
        {"prompt_token_ids": prompt},
        sampling,
        use_tqdm=False,
    )
    single_trace_count = len(tuple(trace_dir.glob("logits-*.pt")))
    batch = llm.generate(
        [{"prompt_token_ids": prompt} for _ in range(args.batch_size)],
        sampling,
        use_tqdm=False,
    )
    trace_paths = sorted(trace_dir.glob("logits-*.pt"))
    payload = {
        "flash_attn_version": args.flash_attn_version,
        "hlm_attention_impl": args.hlm_attention_impl,
        "prompt_length": len(prompt),
        "prompt_offset": args.prompt_offset,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "single_tokens": [
            int(token_id)
            for token_id in single[0].outputs[0].token_ids
        ],
        "batch_tokens": [
            [
                int(token_id)
                for token_id in output.outputs[0].token_ids
            ]
            for output in batch
        ],
        "single_traces": [
            _trace_summary(path)
            for path in trace_paths[:single_trace_count]
        ],
        "batch_traces": [
            _trace_summary(path)
            for path in trace_paths[single_trace_count:]
        ],
    }
    (args.output_dir / "batch_invariance.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.output_dir / "_SUCCESS").touch()


if __name__ == "__main__":
    main()
