"""Batch-1 Megatron versus native-vLLM parity harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_BASE_STAGE_NAMES = (
    "encoder_hidden",
    "final_concepts",
    "decoder_final",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reference", "vllm", "compare"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--ckpt-step", type=int, default=11920)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--train-wandb-config")
    parser.add_argument("--model-dir")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--prompt-jsonl", default="")
    parser.add_argument("--prompt-indices", default="")
    parser.add_argument("--prompt-lengths", default="12,13,14,15")
    parser.add_argument("--max-abs-tolerance", type=float, default=0.5)
    parser.add_argument("--mean-abs-tolerance", type=float, default=0.02)
    parser.add_argument("--report-only", action="store_true")
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


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.mode} mode requires: {', '.join(missing)}")


def _prompt_ids(tokenizer: Any, target_length: int) -> list[int]:
    text = (
        "Question: A careful engineer checks every cache boundary before "
        "shipping an inference backend. What should be verified next?\nAnswer:"
    )
    seed = list(tokenizer.encode(text))
    if not seed:
        raise RuntimeError("tokenizer returned an empty prompt")
    repeated = (seed * ((target_length + len(seed) - 1) // len(seed)))[:target_length]
    if len(repeated) != target_length:
        raise RuntimeError("failed to construct the requested prompt length")
    return [int(token_id) for token_id in repeated]


def _comma_separated_ints(value: str, *, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values or min(values) < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def _trace_cases(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    if args.prompt_jsonl:
        prompt_path = Path(args.prompt_jsonl)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"trace prompt JSONL not found: {prompt_path}")
        prompt_rows = []
        with prompt_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                    raise ValueError(
                        f"{prompt_path}:{line_number} must contain a prompt string"
                    )
                prompt_rows.append(row)
        prompt_indices = _comma_separated_ints(
            args.prompt_indices,
            name="--prompt-indices",
        )
        cases = []
        for prompt_index in prompt_indices:
            if prompt_index >= len(prompt_rows):
                raise ValueError(
                    f"prompt index {prompt_index} exceeds {len(prompt_rows)} rows"
                )
            token_ids = [
                int(token_id)
                for token_id in tokenizer.encode(
                    prompt_rows[prompt_index]["prompt"],
                    add_special_tokens=True,
                )
            ]
            cases.append(
                {
                    "source": "jsonl",
                    "prompt_index": prompt_index,
                    "prompt_length": len(token_ids),
                    "phase": len(token_ids) % 4,
                    "token_ids": token_ids,
                }
            )
        return cases

    prompt_lengths = _comma_separated_ints(
        args.prompt_lengths,
        name="--prompt-lengths",
    )
    return [
        {
            "source": "synthetic",
            "phase": prompt_length % 4,
            "prompt_length": prompt_length,
            "token_ids": _prompt_ids(tokenizer, prompt_length),
        }
        for prompt_length in prompt_lengths
    ]


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("cannot trace an empty module output")
        return _first_tensor(value[0])
    return value


def _last_reference_vector(value: Any) -> torch.Tensor:
    value = _first_tensor(value)
    if hasattr(value, "unwrap") and type(value).__name__ == "WrappedTensor":
        value = value.unwrap()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected stage tensor, got {type(value).__name__}")
    if value.ndim == 3:
        value = value[-1, 0]
    elif value.ndim == 2:
        value = value[-1]
    elif value.ndim != 1:
        raise ValueError(f"unexpected stage tensor shape: {tuple(value.shape)}")
    return value.detach().float().cpu()


class _ReferenceStageCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self._latest: dict[str, torch.Tensor] = {}
        self._tower_lengths = {
            "encoder": len(model.encoder.layers),
            "hlm": len(model.concept_predictor.hlm_block.layers),
            "decoder": len(model.decoder.layers),
        }
        self._handles = [
            model.fusion_tok_norm.register_forward_hook(
                self._capture_input("encoder_hidden")
            ),
            model.fusion_hl_norm.register_forward_hook(
                self._capture_input("final_concepts")
            ),
            model.decoder.final_layernorm.register_forward_hook(
                self._capture_output("decoder_final")
            ),
            model.concept_vq_input_norm.register_forward_hook(
                self._capture_output("hlm.input.0")
            ),
            model.concept_predictor.hlm_block.final_layernorm.register_forward_hook(
                self._capture_input(
                    f"hlm.routed.{self._tower_lengths['hlm'] - 1}"
                )
            ),
            model.concept_predictor.hlm_block.final_layernorm.register_forward_hook(
                self._capture_output("hlm.final_norm")
            ),
            model.decoder.final_layernorm.register_forward_hook(
                self._capture_input("decoder.pre_final_norm")
            ),
        ]
        for tower_name, layers in (
            ("encoder", model.encoder.layers),
            ("hlm", model.concept_predictor.hlm_block.layers),
            ("decoder", model.decoder.layers),
        ):
            for layer_index, layer in enumerate(layers):
                input_name = f"{tower_name}.input.{layer_index}"
                if not (tower_name == "hlm" and layer_index == 0):
                    self._handles.append(
                        layer.register_forward_pre_hook(
                            self._capture_hidden_input(input_name),
                            with_kwargs=True,
                        )
                    )
                self._handles.append(
                    layer.register_forward_hook(
                        self._capture_output(
                            f"{tower_name}.raw.{layer_index}"
                        )
                    )
                )
                if tower_name == "hlm":
                    self._handles.extend(
                        (
                            layer.self_attention.register_forward_hook(
                                self._capture_output(
                                    f"hlm.attention.raw.{layer_index}"
                                )
                            ),
                            layer.post_attention_layernorm.register_forward_hook(
                                self._capture_output(
                                    f"hlm.attention.norm.{layer_index}"
                                )
                            ),
                            layer.mlp.register_forward_pre_hook(
                                self._capture_hidden_input(
                                    f"hlm.attention.residual.{layer_index}"
                                ),
                                with_kwargs=True,
                            ),
                            layer.mlp.register_forward_hook(
                                self._capture_output(
                                    f"hlm.mlp.raw.{layer_index}"
                                )
                            ),
                            layer.mlp.linear_fc1.register_forward_hook(
                                self._capture_output(
                                    f"hlm.mlp.fc1.{layer_index}"
                                )
                            ),
                            layer.mlp.linear_fc2.register_forward_pre_hook(
                                self._capture_hidden_input(
                                    f"hlm.mlp.activation.{layer_index}"
                                ),
                                with_kwargs=True,
                            ),
                            layer.mlp.linear_fc2.register_forward_hook(
                                self._capture_output(
                                    f"hlm.mlp.fc2.{layer_index}"
                                )
                            ),
                            layer.post_feedforward_layernorm.register_forward_hook(
                                self._capture_output(
                                    f"hlm.mlp.norm.{layer_index}"
                                )
                            ),
                        )
                    )
        names = list(_BASE_STAGE_NAMES)
        for tower_name in ("encoder", "hlm", "decoder"):
            for layer_index in range(self._tower_lengths[tower_name]):
                names.extend(
                    (
                        f"{tower_name}.input.{layer_index}",
                        f"{tower_name}.raw.{layer_index}",
                        f"{tower_name}.routed.{layer_index}",
                    )
                )
        names.extend(("hlm.final_norm", "fusion_output", "decoder.pre_final_norm"))
        for layer_index in range(self._tower_lengths["hlm"]):
            names.extend(
                (
                    f"hlm.attention.raw.{layer_index}",
                    f"hlm.attention.norm.{layer_index}",
                    f"hlm.attention.residual.{layer_index}",
                    f"hlm.mlp.raw.{layer_index}",
                    f"hlm.mlp.fc1.{layer_index}",
                    f"hlm.mlp.activation.{layer_index}",
                    f"hlm.mlp.fc2.{layer_index}",
                    f"hlm.mlp.norm.{layer_index}",
                )
            )
        self.stage_names = tuple(dict.fromkeys(names))

    def _capture_input(self, name: str) -> Any:
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            self._latest[name] = _last_reference_vector(inputs[0])

        return hook

    def _capture_output(self, name: str) -> Any:
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            self._latest[name] = _last_reference_vector(output)

        return hook

    def _capture_hidden_input(self, name: str) -> Any:
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            value = kwargs.get("hidden_states")
            if value is None:
                if not inputs:
                    raise RuntimeError(f"missing hidden-state input for {name}")
                value = inputs[-1]
            self._latest[name] = _last_reference_vector(value)

        return hook

    def reset(self) -> None:
        self._latest.clear()

    def consume(self) -> dict[str, torch.Tensor]:
        for tower_name in ("encoder", "hlm", "decoder"):
            tower_length = self._tower_lengths[tower_name]
            for layer_index in range(tower_length - 1):
                self._latest[f"{tower_name}.routed.{layer_index}"] = self._latest[
                    f"{tower_name}.input.{layer_index + 1}"
                ]
        self._latest[
            f"encoder.routed.{self._tower_lengths['encoder'] - 1}"
        ] = self._latest["encoder_hidden"]
        self._latest[
            f"decoder.routed.{self._tower_lengths['decoder'] - 1}"
        ] = self._latest["decoder.pre_final_norm"]
        self._latest["fusion_output"] = self._latest["decoder.input.0"]
        missing = sorted(set(self.stage_names) - set(self._latest))
        unexpected = sorted(set(self._latest) - set(self.stage_names))
        if missing or unexpected:
            raise RuntimeError(
                f"invalid reference stage trace: missing={missing}, "
                f"unexpected={unexpected}"
            )
        return dict(self._latest)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _stack_stage_rows(
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot stack an empty stage trace")
    stage_names = tuple(rows[0])
    for row in rows[1:]:
        if tuple(row) != stage_names:
            raise ValueError("reference stage trace keys changed between rows")
    return {
        name: torch.stack([row[name] for row in rows])
        for name in stage_names
    }


def run_reference(args: argparse.Namespace) -> None:
    _require(
        args,
        "checkpoint_root",
        "tokenizer_model",
        "train_wandb_config",
    )
    from ncp_olmo_eval import benchmark
    from ncp_olmo_eval.common import forward_last_token_logits

    args.output_dir.mkdir(parents=True, exist_ok=False)
    lm_eval_model = benchmark.load_model(
        SimpleNamespace(
            checkpoint_root=args.checkpoint_root,
            ckpt_step=args.ckpt_step,
            tokenizer_model=args.tokenizer_model,
            train_wandb_config=args.train_wandb_config,
            flash_decode=False,
            cuda_graph=False,
            batch_size=1,
            seq_length=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
        )
    )
    eval_model = benchmark.EvalModelAdapter(lm_eval_model)
    capture = _ReferenceStageCapture(benchmark.unwrap_model(eval_model))
    phase_cases = _trace_cases(args, eval_model.tokenizer)
    if max(case["prompt_length"] for case in phase_cases) + args.max_new_tokens > (
        args.max_model_len
    ):
        raise ValueError(
            "trace prompt plus continuation exceeds max model length: "
            f"{max(case['prompt_length'] for case in phase_cases)} + "
            f"{args.max_new_tokens} > {args.max_model_len}"
        )
    phase_logits = []
    phase_stages = []
    greedy_stages = []
    try:
        for case in phase_cases:
            token_ids = case["token_ids"]
            capture.reset()
            phase_logits.append(
                forward_last_token_logits(eval_model, [token_ids])[0].float().cpu()
            )
            phase_stages.append(capture.consume())

        greedy_prompt_ids = list(phase_cases[-1]["token_ids"])
        greedy_ids = list(greedy_prompt_ids)
        greedy_tokens = []
        greedy_logits = []
        for _ in range(args.max_new_tokens):
            capture.reset()
            logits = forward_last_token_logits(eval_model, [greedy_ids])[0].float()
            greedy_stages.append(capture.consume())
            next_token = int(torch.argmax(logits).item())
            greedy_logits.append(logits.cpu())
            greedy_tokens.append(next_token)
            greedy_ids.append(next_token)
    finally:
        capture.close()

    inputs = {
        "phase_cases": phase_cases,
        "greedy_prompt_ids": greedy_prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "prompt_jsonl": args.prompt_jsonl,
        "prompt_indices": args.prompt_indices,
    }
    (args.output_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "phase_logits": torch.stack(phase_logits),
            "greedy_logits": torch.stack(greedy_logits),
            "greedy_tokens": greedy_tokens,
            "phase_stages": _stack_stage_rows(phase_stages),
            "greedy_stages": _stack_stage_rows(greedy_stages),
        },
        args.output_dir / "reference.pt",
    )
    (args.output_dir / "_SUCCESS").touch()


def run_vllm(args: argparse.Namespace) -> None:
    _require(args, "model_dir")
    from vllm import LLM, SamplingParams

    from .plugin import register

    inputs = json.loads((args.output_dir / "inputs.json").read_text())
    trace_dir = args.output_dir / "vllm_logits"
    if trace_dir.exists():
        raise FileExistsError(f"refusing to reuse trace directory: {trace_dir}")
    trace_dir.mkdir()
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
        max_model_len=int(inputs["max_model_len"]),
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        skip_tokenizer_init=True,
        attention_config={
            "backend": "FLASH_ATTN",
            "flash_attn_version": args.flash_attn_version,
        },
    )
    one_token = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )
    phase_tokens = []
    for case in inputs["phase_cases"]:
        outputs = llm.generate(
            {"prompt_token_ids": case["token_ids"]},
            one_token,
            use_tqdm=False,
        )
        phase_tokens.append(int(outputs[0].outputs[0].token_ids[0]))

    batch_outputs = llm.generate(
        [
            {"prompt_token_ids": case["token_ids"]}
            for case in inputs["phase_cases"]
        ],
        one_token,
        use_tqdm=False,
    )
    batched_phase_tokens = [
        int(output.outputs[0].token_ids[0])
        for output in batch_outputs
    ]

    full_prefix_ids = list(inputs["greedy_prompt_ids"])
    full_prefix_tokens = []
    for _ in range(int(inputs["max_new_tokens"])):
        outputs = llm.generate(
            {"prompt_token_ids": full_prefix_ids},
            one_token,
            use_tqdm=False,
        )
        next_token = int(outputs[0].outputs[0].token_ids[0])
        full_prefix_tokens.append(next_token)
        full_prefix_ids.append(next_token)

    greedy = SamplingParams(
        temperature=0.0,
        max_tokens=int(inputs["max_new_tokens"]),
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(
        {"prompt_token_ids": inputs["greedy_prompt_ids"]},
        greedy,
        use_tqdm=False,
    )
    greedy_tokens = [int(token_id) for token_id in outputs[0].outputs[0].token_ids]
    (args.output_dir / "vllm_output.json").write_text(
        json.dumps(
            {
                "flash_attn_version": args.flash_attn_version,
                "hlm_attention_impl": args.hlm_attention_impl,
                "phase_tokens": phase_tokens,
                "batched_phase_tokens": batched_phase_tokens,
                "full_prefix_tokens": full_prefix_tokens,
                "greedy_tokens": greedy_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "_VLLM_SUCCESS").touch()


def _compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"logit shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    difference = (reference - candidate).abs()
    centered_reference = reference - reference.mean()
    centered_candidate = candidate - candidate.mean()
    centered_difference = (centered_reference - centered_candidate).abs()
    reference_log_probability = torch.log_softmax(reference, dim=-1)
    candidate_log_probability = torch.log_softmax(candidate, dim=-1)
    reference_probability = reference_log_probability.exp()
    candidate_probability = candidate_log_probability.exp()
    reference_kl = torch.sum(
        reference_probability
        * (reference_log_probability - candidate_log_probability)
    )
    candidate_kl = torch.sum(
        candidate_probability
        * (candidate_log_probability - reference_log_probability)
    )
    topk = min(10, int(reference.numel()))
    reference_topk = set(torch.topk(reference, topk).indices.tolist())
    candidate_topk = set(torch.topk(candidate, topk).indices.tolist())
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "mean_shift": float((candidate - reference).mean().item()),
        "centered_max_abs": float(centered_difference.max().item()),
        "centered_mean_abs": float(centered_difference.mean().item()),
        "symmetric_kl": float(((reference_kl + candidate_kl) / 2).item()),
        "total_variation": float(
            ((reference_probability - candidate_probability).abs().sum() / 2).item()
        ),
        "top10_overlap": len(reference_topk & candidate_topk),
        "reference_top1": int(torch.argmax(reference).item()),
        "candidate_top1": int(torch.argmax(candidate).item()),
        "top1_match": bool(torch.argmax(reference) == torch.argmax(candidate)),
    }


def _compare_stage_vectors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"stage shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(candidate.shape)}"
        )
    difference = reference - candidate
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    norm_product = torch.clamp(reference_norm * candidate_norm, min=1e-12)
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.clamp(reference_norm, min=1e-12)
            ).item()
        ),
        "cosine_similarity": float(
            (torch.dot(reference, candidate) / norm_product).item()
        ),
    }


def _compare_stage_groups(
    reference: dict[str, torch.Tensor],
    candidates: list[dict[str, torch.Tensor]],
) -> dict[str, list[dict[str, float]]]:
    reference_names = tuple(reference)
    for candidate in candidates:
        if set(candidate) != set(reference_names):
            missing = sorted(set(reference_names) - set(candidate))
            unexpected = sorted(set(candidate) - set(reference_names))
            raise ValueError(
                f"candidate stage keys changed: missing={missing}, "
                f"unexpected={unexpected}"
            )
    return {
        name: [
            _compare_stage_vectors(reference[name][index], candidate[name])
            for index, candidate in enumerate(candidates)
        ]
        for name in reference_names
    }


def run_compare(args: argparse.Namespace) -> None:
    reference = torch.load(
        args.output_dir / "reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    output = json.loads((args.output_dir / "vllm_output.json").read_text())
    inputs = json.loads((args.output_dir / "inputs.json").read_text())
    max_new_tokens = int(inputs["max_new_tokens"])
    phase_count = len(inputs["phase_cases"])
    trace_paths = sorted((args.output_dir / "vllm_logits").glob("logits-*.pt"))
    fixed_trace_count = phase_count + 2 * max_new_tokens
    batch_trace_count = len(trace_paths) - fixed_trace_count
    if batch_trace_count < 1:
        raise RuntimeError(
            "expected at least one mixed-batch vLLM logit trace, got "
            f"{batch_trace_count}"
        )
    full_prefix_start = phase_count + batch_trace_count
    cached_start = full_prefix_start + max_new_tokens
    traces = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in trace_paths
    ]
    stage_paths = sorted((args.output_dir / "vllm_logits").glob("stages-*.pt"))
    if len(stage_paths) != len(trace_paths):
        raise RuntimeError(
            f"expected {len(trace_paths)} vLLM stage traces, got "
            f"{len(stage_paths)}"
        )
    stage_traces = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in stage_paths
    ]
    phase_rows = [
        _compare_logits(reference["phase_logits"][index], traces[index][0])
        for index in range(phase_count)
    ]
    full_prefix_rows = [
        _compare_logits(
            reference["greedy_logits"][index],
            traces[index + full_prefix_start][0],
        )
        for index in range(max_new_tokens)
    ]
    cached_rows = [
        _compare_logits(
            reference["greedy_logits"][index],
            traces[index + cached_start][0],
        )
        for index in range(max_new_tokens)
    ]
    cache_vs_full_prefix_rows = [
        _compare_logits(
            traces[index + full_prefix_start][0],
            traces[index + cached_start][0],
        )
        for index in range(max_new_tokens)
    ]
    stage_report = {
        "phase": _compare_stage_groups(
            reference["phase_stages"],
            stage_traces[:phase_count],
        ),
        "full_prefix": _compare_stage_groups(
            reference["greedy_stages"],
            stage_traces[
                full_prefix_start : full_prefix_start + max_new_tokens
            ],
        ),
        "cached": _compare_stage_groups(
            reference["greedy_stages"],
            stage_traces[cached_start:],
        ),
        "cache_vs_full_prefix": {
            name: [
                _compare_stage_vectors(
                    stage_traces[full_prefix_start + index][name],
                    stage_traces[cached_start + index][name],
                )
                for index in range(max_new_tokens)
            ]
            for name in reference["greedy_stages"]
        },
    }
    reference_phase_tokens = [row["reference_top1"] for row in phase_rows]
    reference_greedy_tokens = [
        int(token_id) for token_id in reference["greedy_tokens"]
    ]
    reference_rows = phase_rows + full_prefix_rows
    cache_rows = cache_vs_full_prefix_rows
    report = {
        "flash_attn_version": int(output["flash_attn_version"]),
        "hlm_attention_impl": output["hlm_attention_impl"],
        "phase_logits": phase_rows,
        "batched_phase_scheduler_steps": batch_trace_count,
        "full_prefix_logits": full_prefix_rows,
        "cached_logits": cached_rows,
        "cache_vs_full_prefix_logits": cache_vs_full_prefix_rows,
        "stage_vectors": stage_report,
        "phase_tokens_reference": reference_phase_tokens,
        "phase_tokens_vllm": output["phase_tokens"],
        "batched_phase_tokens_vllm": output["batched_phase_tokens"],
        "greedy_tokens_reference": reference_greedy_tokens,
        "full_prefix_tokens_vllm": output["full_prefix_tokens"],
        "greedy_tokens_vllm": output["greedy_tokens"],
        "reference_max_abs": max(row["max_abs"] for row in reference_rows),
        "reference_mean_abs_max": max(
            row["mean_abs"] for row in reference_rows
        ),
        "cache_max_abs": max(row["max_abs"] for row in cache_rows),
        "cache_mean_abs_max": max(row["mean_abs"] for row in cache_rows),
        "reference_centered_max_abs": max(
            row["centered_max_abs"] for row in reference_rows
        ),
        "reference_symmetric_kl_max": max(
            row["symmetric_kl"] for row in reference_rows
        ),
        "reference_total_variation_max": max(
            row["total_variation"] for row in reference_rows
        ),
        "reference_top10_overlap_min": min(
            row["top10_overlap"] for row in reference_rows
        ),
        "cache_centered_max_abs": max(
            row["centered_max_abs"] for row in cache_rows
        ),
        "cache_symmetric_kl_max": max(
            row["symmetric_kl"] for row in cache_rows
        ),
        "cache_total_variation_max": max(
            row["total_variation"] for row in cache_rows
        ),
        "cache_top10_overlap_min": min(
            row["top10_overlap"] for row in cache_rows
        ),
    }
    report["greedy_ok"] = bool(
        all(row["top1_match"] for row in reference_rows)
        and all(row["top1_match"] for row in cache_rows)
        and reference_phase_tokens == output["phase_tokens"]
        and reference_phase_tokens == output["batched_phase_tokens"]
        and reference_greedy_tokens == output["full_prefix_tokens"]
        and reference_greedy_tokens == output["greedy_tokens"]
    )
    report["raw_logits_ok"] = bool(
        report["reference_max_abs"] <= args.max_abs_tolerance
        and report["reference_mean_abs_max"] <= args.mean_abs_tolerance
    )
    report["cache_raw_logits_ok"] = bool(
        report["cache_max_abs"] <= args.max_abs_tolerance
        and report["cache_mean_abs_max"] <= args.mean_abs_tolerance
    )
    report["ok"] = bool(
        report["greedy_ok"]
        and report["raw_logits_ok"]
        and report["cache_raw_logits_ok"]
    )
    (args.output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["ok"] and not args.report_only:
        raise RuntimeError(f"ConceptLM parity failed: {json.dumps(report)}")
    if report["ok"]:
        (args.output_dir / "_PARITY_SUCCESS").touch()
    else:
        (args.output_dir / "_PARITY_REPORTED").touch()


def main() -> None:
    args = parse_args()
    if args.mode == "reference":
        run_reference(args)
    elif args.mode == "vllm":
        run_vllm(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
