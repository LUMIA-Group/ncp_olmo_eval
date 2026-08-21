"""Adapter from the local evaluation interface to native ConceptLM vLLM.

The native backend remains an explicitly selected experimental path.  It owns
one vLLM engine per visible GPU and keeps prefix caching and speculative
decoding disabled.  Evaluation batches retain a stable seed per request so
their sampling contract does not depend on the coordinator's batch shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from .inference import SamplingParams, TextCompletion

NATIVE_VLLM_BACKEND = "native_vllm"
NATIVE_VLLM_OVERLAY_MANIFEST = "native_vllm_overlay_manifest.json"


def add_native_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared native-vLLM runtime contract to an evaluation parser."""

    parser.add_argument(
        "--allow-unverified-native-vllm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Acknowledge that native ConceptLM vLLM is still parity-gated. "
            "Required when --hf-backend=native_vllm."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="Native vLLM maximum sequence length; 0 uses seq_length + 2.",
    )
    parser.add_argument(
        "--vllm-runtime-config",
        default="",
        help=(
            "Complete native-vLLM config.json. When set, the evaluator creates "
            "a result-local read-only-weight overlay and never mutates the "
            "source HF artifact."
        ),
    )
    parser.add_argument(
        "--vllm-model-family",
        choices=("conceptlm", "auto"),
        default="conceptlm",
        help=(
            "conceptlm uses the custom ConceptLM plugin and worker; auto uses "
            "vLLM's built-in model registry and worker for standard HF models."
        ),
    )
    parser.add_argument(
        "--vllm-model-overlay-dir",
        default="",
        help=(
            "Optional native-vLLM model overlay destination. Evaluators choose "
            "a result-local directory when this is empty."
        ),
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-execution-mode", choices=("eager", "piecewise"), default="eager")
    parser.add_argument("--vllm-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--vllm-flash-attn-version", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--vllm-hlm-attention-impl",
        choices=("legacy_mixed", "uniform_flash"),
        default="legacy_mixed",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs owned by each native-vLLM engine.",
    )


def validate_native_vllm_args(
    args: argparse.Namespace,
    *,
    batch_size: int,
    processes_per_gpu: int = 1,
    model_source: str = "hf",
) -> None:
    """Fail closed when an evaluation violates the validated engine shape."""

    if getattr(args, "hf_backend", None) != NATIVE_VLLM_BACKEND:
        return
    if not bool(getattr(args, "allow_unverified_native_vllm", False)):
        raise ValueError("--hf-backend=native_vllm requires " "--allow-unverified-native-vllm")
    if model_source != "hf":
        raise ValueError("native_vllm requires --model-source=hf")
    if not 1 <= int(batch_size) <= 8:
        raise ValueError("native_vllm evaluation batch size must be in [1, 8]")
    if int(processes_per_gpu) != 1:
        raise ValueError("native_vllm requires exactly one engine process per GPU")
    if int(getattr(args, "vllm_max_model_len", 0)) < 0:
        raise ValueError("vllm_max_model_len must be non-negative")
    if int(getattr(args, "vllm_tensor_parallel_size", 1)) <= 0:
        raise ValueError("vllm_tensor_parallel_size must be positive")
    runtime_config = str(getattr(args, "vllm_runtime_config", ""))
    if runtime_config and not Path(runtime_config).is_file():
        raise ValueError(f"vLLM runtime config does not exist: {runtime_config}")
    model_family = str(getattr(args, "vllm_model_family", "conceptlm"))
    if model_family == "auto" and runtime_config:
        raise ValueError("stock vLLM models must use their own config.json")
    utilization = float(getattr(args, "vllm_gpu_memory_utilization", 0.85))
    if not 0.0 < utilization < 1.0:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1)")


def native_vllm_max_model_len(args: argparse.Namespace) -> int:
    """Reserve target and generated-token slots for continuation scoring."""

    configured = int(getattr(args, "vllm_max_model_len", 0))
    return configured if configured > 0 else int(args.seq_length) + 2


def _make_backend_importable() -> None:
    backend_src = Path(__file__).resolve().parent / "vllm_backend" / "src"
    backend_src_string = str(backend_src)
    if backend_src_string not in sys.path:
        sys.path.insert(0, backend_src_string)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_native_config(config_path: Path) -> dict[str, Any]:
    _make_backend_importable()
    from ncp_olmo_eval.vllm_plugin.contract import ConceptLMBackendConfig

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"native vLLM config is not a JSON object: {config_path}")
    normalized = ConceptLMBackendConfig.from_mapping(payload)
    return normalized.to_dict()


def prepare_native_vllm_model(
    *,
    source_model: str | Path,
    runtime_config: str | Path | None,
    overlay_dir: str | Path | None,
    model_family: str = "conceptlm",
) -> tuple[Path, dict[str, Any]]:
    """Validate or create an immutable-weight native-vLLM model directory."""

    source = Path(source_model).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source HF model directory does not exist: {source}")
    source_config = source / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"source HF config does not exist: {source_config}")

    if model_family == "auto":
        if runtime_config:
            raise ValueError("stock vLLM models must use their own config.json")
        config_payload = json.loads(source_config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise ValueError(f"stock vLLM config is not a JSON object: {source_config}")
        weight_files = sorted(source.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"stock vLLM model has no safetensor weights: {source}")
        return source, {
            "status": "STOCK_VLLM_MODEL_READY",
            "mode": "source",
            "source_model": str(source),
            "destination_model": str(source),
            "runtime_config": str(source_config),
            "runtime_config_sha256": _file_sha256(source_config),
            "weight_files_mutated": False,
            "weight_file_count": len(weight_files),
            "model_type": config_payload.get("model_type"),
            "architectures": config_payload.get("architectures"),
        }
    if model_family != "conceptlm":
        raise ValueError(f"unsupported vLLM model family: {model_family}")

    tokenizer_config_source = source / "tokenizer_config.json"
    tokenizer_config_payload: dict[str, Any] | None = None
    tokenizer_compatibility_rewrite = False
    if tokenizer_config_source.is_file():
        tokenizer_config_payload = json.loads(tokenizer_config_source.read_text(encoding="utf-8"))
        if not isinstance(tokenizer_config_payload, dict):
            raise ValueError(
                "native vLLM tokenizer config is not a JSON object: " f"{tokenizer_config_source}"
            )
        tokenizer_compatibility_rewrite = (
            tokenizer_config_payload.get("tokenizer_class") == "TokenizersBackend"
        )

    if not runtime_config and not tokenizer_compatibility_rewrite:
        normalized = _validate_native_config(source_config)
        return source, {
            "status": "CONCEPTLM_NATIVE_VLLM_MODEL_READY",
            "mode": "source",
            "source_model": str(source),
            "destination_model": str(source),
            "runtime_config": str(source_config),
            "runtime_config_sha256": _file_sha256(source_config),
            "weight_files_mutated": False,
            "tokenizer_config_compatibility_rewrite": False,
            "normalized_contract": normalized,
        }

    config_source = Path(runtime_config).resolve() if runtime_config else source_config
    if not config_source.is_file():
        raise FileNotFoundError(f"native vLLM runtime config does not exist: {config_source}")
    if not overlay_dir:
        reason = (
            "a separate runtime config"
            if runtime_config
            else "the TokenizersBackend compatibility rewrite"
        )
        raise ValueError(f"overlay_dir is required with {reason}")
    destination = Path(overlay_dir).resolve()
    if destination == source:
        raise ValueError("native vLLM overlay must differ from the source model")
    destination.mkdir(parents=True, exist_ok=True)

    normalized = _validate_native_config(config_source)
    weight_files: dict[str, dict[str, Any]] = {}
    for source_path in sorted(source.iterdir(), key=lambda path: path.name):
        if not source_path.is_file() or source_path.name in {
            "config.json",
            "tokenizer_config.json",
            NATIVE_VLLM_OVERLAY_MANIFEST,
        }:
            continue
        destination_path = destination / source_path.name
        if destination_path.exists() or destination_path.is_symlink():
            if (
                not destination_path.is_symlink()
                or destination_path.resolve() != source_path.resolve()
            ):
                raise RuntimeError(
                    "native vLLM overlay contains an unexpected file: " f"{destination_path}"
                )
        else:
            destination_path.symlink_to(source_path)
        if source_path.name.endswith(".safetensors"):
            stat = source_path.stat()
            weight_files[source_path.name] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    if not weight_files:
        raise RuntimeError(f"source HF model has no safetensor weights: {source}")

    shutil.copyfile(config_source, destination / "config.json")
    if tokenizer_config_payload is not None:
        tokenizer_config = dict(tokenizer_config_payload)
        if tokenizer_compatibility_rewrite:
            tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        _atomic_write_json(destination / "tokenizer_config.json", tokenizer_config)

    manifest = {
        "status": "CONCEPTLM_NATIVE_VLLM_MODEL_READY",
        "mode": ("overlay" if runtime_config else "overlay_tokenizer_compatibility"),
        "source_model": str(source),
        "destination_model": str(destination),
        "runtime_config": str(config_source),
        "runtime_config_sha256": _file_sha256(config_source),
        "weight_files_mutated": False,
        "tokenizer_config_compatibility_rewrite": tokenizer_compatibility_rewrite,
        "weight_files": weight_files,
        "normalized_contract": normalized,
    }
    _atomic_write_json(destination / NATIVE_VLLM_OVERLAY_MANIFEST, manifest)
    return destination, manifest


def _logprob_value(entry: Any) -> float:
    value = getattr(entry, "logprob", None)
    if value is None and isinstance(entry, dict):
        value = entry.get("logprob")
    if value is None:
        value = entry
    return float(value)


def _logprob_rank(entry: Any) -> int | None:
    value = getattr(entry, "rank", None)
    if value is None and isinstance(entry, dict):
        value = entry.get("rank")
    return None if value is None else int(value)


def _token_logprob_entry(entries: Any, token_id: int) -> Any:
    if entries is None:
        raise RuntimeError(f"vLLM omitted prompt logprobs for token {token_id}")
    if token_id in entries:
        return entries[token_id]
    token_key = str(token_id)
    if token_key in entries:
        return entries[token_key]
    raise RuntimeError(
        "vLLM prompt_logprobs did not retain the observed prompt token: "
        f"token_id={token_id} available={list(entries)[:8]}"
    )


class NativeVLLMInferencer:
    """Expose native vLLM generation and continuation scoring to eval code."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        max_model_len: int,
        seed: int,
        max_batch_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        execution_mode: str = "eager",
        attention_backend: str = "FLASH_ATTN",
        flash_attn_version: int = 3,
        hlm_attention_impl: str = "legacy_mixed",
        tensor_parallel_size: int = 1,
        model_family: str = "conceptlm",
        engine: Any | None = None,
        sampling_params_factory: Callable[..., Any] | None = None,
    ) -> None:
        if int(max_model_len) <= 0:
            raise ValueError("max_model_len must be positive")
        if not 1 <= int(max_batch_size) <= 8:
            raise ValueError("max_batch_size must be in [1, 8]")
        if execution_mode not in ("eager", "piecewise"):
            raise ValueError(f"unsupported vLLM execution mode: {execution_mode}")
        if int(tensor_parallel_size) <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if model_family not in ("conceptlm", "auto"):
            raise ValueError(f"unsupported vLLM model family: {model_family}")
        self.model_path = str(Path(model_path).resolve())
        self.max_model_len = int(max_model_len)
        self.max_batch_size = int(max_batch_size)
        self.seed = int(seed)
        self.execution_mode = execution_mode
        self.attention_backend = str(attention_backend)
        self.flash_attn_version = int(flash_attn_version)
        self.hlm_attention_impl = str(hlm_attention_impl)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.model_family = str(model_family)
        self.gpu_memory_utilization = float(gpu_memory_utilization)

        if engine is None:
            from vllm import LLM
            from vllm import SamplingParams as VLLMSamplingParams

            if self.model_family == "conceptlm":
                _make_backend_importable()
                from ncp_olmo_eval.vllm_plugin.plugin import register

                os.environ["CONCEPTLM_VLLM_ENABLE_UNVERIFIED"] = "1"
                os.environ["CONCEPTLM_HLM_ATTENTION_IMPL"] = self.hlm_attention_impl
                register()
            enforce_eager = execution_mode == "eager"
            compilation_config = (
                None
                if enforce_eager
                else {"mode": 3, "cudagraph_mode": "PIECEWISE", "custom_ops": ["all"]}
            )
            engine_kwargs: dict[str, Any] = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "enforce_eager": enforce_eager,
                "enable_prefix_caching": False,
                "max_model_len": self.max_model_len,
                "max_num_seqs": self.max_batch_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "disable_log_stats": False,
                "skip_tokenizer_init": False,
                "seed": self.seed,
                "compilation_config": compilation_config,
                "attention_config": {
                    "backend": self.attention_backend,
                    "flash_attn_version": self.flash_attn_version,
                },
            }
            if self.model_family == "conceptlm":
                engine_kwargs.update(
                    trust_remote_code=True,
                    worker_cls="ncp_olmo_eval.vllm_plugin.worker.ConceptLMGPUWorker",
                )
            engine = LLM(**engine_kwargs)
            sampling_params_factory = VLLMSamplingParams
        elif sampling_params_factory is None:
            raise ValueError("sampling_params_factory is required with an injected engine")
        self.engine = engine
        self._sampling_params_factory = sampling_params_factory
        self.tokenizer = engine.get_tokenizer()

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backend": NATIVE_VLLM_BACKEND,
            "model_family": self.model_family,
            "experimental": True,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "execution_mode": self.execution_mode,
            "prefix_caching": False,
            "speculative_decoding": False,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "attention_backend": self.attention_backend,
            "flash_attn_version": self.flash_attn_version,
            "hlm_attention_impl": self.hlm_attention_impl,
        }

    def generate(self, prompts: list[str], sampling: SamplingParams) -> list[TextCompletion]:
        """Generate one homogeneous batch using a shared sampling contract."""

        return self.generate_batch(prompts, [sampling for _ in prompts])

    def generate_batch(
        self, prompts: list[str], samplings: list[SamplingParams]
    ) -> list[TextCompletion]:
        """Generate a bounded batch with request-local sampling parameters."""

        if not prompts:
            raise ValueError("prompt must not be empty")
        if len(prompts) != len(samplings):
            raise ValueError(
                "native_vllm prompt/sampling batch mismatch: " f"{len(prompts)} != {len(samplings)}"
            )
        if len(prompts) > self.max_batch_size:
            raise ValueError(
                "native_vllm batch exceeds max_batch_size: "
                f"{len(prompts)} > {self.max_batch_size}"
            )
        parameters: list[dict[str, Any]] = []
        for sampling in samplings:
            request_parameters: dict[str, Any] = {
                "temperature": float(sampling.temperature),
                "top_p": float(sampling.top_p),
                "max_tokens": int(sampling.max_tokens),
                "stop": list(sampling.stop),
                "ignore_eos": False,
                "detokenize": True,
            }
            if sampling.seed is not None:
                request_parameters["seed"] = int(sampling.seed)
            parameters.append(request_parameters)
        outputs = self.engine.generate(
            prompts,
            [
                self._sampling_params_factory(**request_parameters)
                for request_parameters in parameters
            ],
            use_tqdm=False,
        )
        if len(outputs) != len(prompts) or any(len(output.outputs) != 1 for output in outputs):
            raise RuntimeError(
                "native vLLM returned an invalid batch shape: "
                f"requests={len(prompts)} outputs={len(outputs)}"
            )
        completions: list[TextCompletion] = []
        for request_output, sampling in zip(outputs, samplings, strict=True):
            completion = request_output.outputs[0]
            token_ids = [int(token_id) for token_id in completion.token_ids]
            completions.append(
                TextCompletion(
                    text=str(completion.text),
                    token_ids=token_ids,
                    finish_reason=str(completion.finish_reason),
                    kv_stats={
                        "cache_backend": "native_vllm_paged_attention",
                        "computed_token_slots": len(token_ids),
                        "model_decode_token_steps": max(0, len(token_ids) - 1),
                        "avoided_computed_token_slots": max(
                            0, int(sampling.max_tokens) - len(token_ids)
                        ),
                        **self.runtime_metadata,
                    },
                )
            )
        return completions

    def score_requests(
        self, requests: list[dict[str, Any]], candidate_results: list[list[dict[str, Any] | None]]
    ) -> dict[str, int]:
        """Score exact continuation tokens through vLLM prompt logprobs."""

        prepared: list[tuple[dict[str, Any], list[int]]] = []
        for request in requests:
            targets = list(request["targets"])
            if len(targets) != 1:
                raise ValueError(
                    "native_vllm scoring requires one continuation candidate " "per request"
                )
            target = targets[0]
            token_ids = [int(value) for value in target["token_ids"]]
            positions = [int(value) for value in target["positions"]]
            if not token_ids or len(token_ids) != len(positions):
                raise ValueError("invalid native_vllm continuation target")
            query_ids = [int(value) for value in request["query_ids"]]
            full_prompt_ids = [*query_ids, token_ids[-1]]
            required_model_len = len(full_prompt_ids) + 1
            if required_model_len > self.max_model_len:
                raise ValueError(
                    "native_vllm scoring prompt plus its required output token "
                    "exceeds max_model_len: "
                    f"{len(full_prompt_ids)} + 1 = {required_model_len} > "
                    f"{self.max_model_len}"
                )
            for position, token_id in zip(positions, token_ids, strict=True):
                prompt_index = position + 1
                if (
                    prompt_index >= len(full_prompt_ids)
                    or full_prompt_ids[prompt_index] != token_id
                ):
                    raise RuntimeError("continuation position no longer matches the vLLM prompt")
            prepared.append((request, full_prompt_ids))

        query_token_count = 0
        continuation_token_count = 0
        prompt_token_slots = 0
        forward_calls = 0
        sampling = self._sampling_params_factory(
            temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False, prompt_logprobs=1
        )
        for start in range(0, len(prepared), self.max_batch_size):
            batch = prepared[start : start + self.max_batch_size]
            outputs = self.engine.generate(
                [{"prompt_token_ids": full_prompt_ids} for _, full_prompt_ids in batch],
                sampling,
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    "native vLLM scoring returned an invalid batch shape: "
                    f"{len(outputs)} != {len(batch)}"
                )
            forward_calls += 1
            for (request, full_prompt_ids), output in zip(batch, outputs, strict=True):
                target = request["targets"][0]
                token_ids = [int(value) for value in target["token_ids"]]
                positions = [int(value) for value in target["positions"]]
                prompt_logprobs = output.prompt_logprobs
                if prompt_logprobs is None or len(prompt_logprobs) != len(full_prompt_ids):
                    raise RuntimeError(
                        "native vLLM returned incomplete prompt_logprobs: "
                        f"{0 if prompt_logprobs is None else len(prompt_logprobs)} "
                        f"!= {len(full_prompt_ids)}"
                    )
                token_logprobs: list[float] = []
                greedy_flags: list[bool] = []
                for position, token_id in zip(positions, token_ids, strict=True):
                    entry = _token_logprob_entry(prompt_logprobs[position + 1], token_id)
                    token_logprobs.append(_logprob_value(entry))
                    greedy_flags.append(_logprob_rank(entry) == 1)
                row_index = int(request["row_index"])
                candidate_index = int(target["candidate_index"])
                candidate_results[row_index][candidate_index] = {
                    "token_ids": token_ids,
                    "token_logprobs": token_logprobs,
                    "sum_logprob": float(sum(token_logprobs)),
                    "num_tokens": len(token_ids),
                    "is_greedy": all(greedy_flags),
                }
                query_token_count += len(request["query_ids"])
                continuation_token_count += len(token_ids)
                prompt_token_slots += len(full_prompt_ids)
        return {
            "forward_calls": forward_calls,
            "query_token_count": query_token_count,
            "padded_token_slots": prompt_token_slots,
            "continuation_token_count": continuation_token_count,
        }
