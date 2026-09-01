"""Pure-Python contracts for reproducible NCP DFlash operating points."""

from __future__ import annotations

from typing import Any, Mapping


DFLASH_OPERATING_POINT_SCHEMA = "ncp-dflash-operating-point-v1"


def canonical_active_batch_widths(raw_policy: str, maximum_width: int) -> str:
    """Validate and canonicalize upper-bound active-batch width buckets."""

    raw_policy = str(raw_policy).strip()
    if not raw_policy:
        return ""
    buckets: dict[int, int] = {}
    for raw_bucket in raw_policy.split(","):
        upper_text, separator, width_text = raw_bucket.strip().partition(":")
        if not separator:
            raise ValueError(
                "DFlash active-batch widths must use comma-separated "
                "upper_bound:width entries"
            )
        upper_bound = int(upper_text)
        width = int(width_text)
        if upper_bound < 1:
            raise ValueError("DFlash active-batch upper bounds must be positive")
        if not 0 <= width <= int(maximum_width):
            raise ValueError(
                "DFlash active-batch width must be between zero and the "
                f"speculative width: width={width} maximum={maximum_width}"
            )
        if upper_bound in buckets:
            raise ValueError(f"duplicate DFlash active-batch upper bound: {upper_bound}")
        buckets[upper_bound] = width
    return ",".join(f"{upper_bound}:{buckets[upper_bound]}" for upper_bound in sorted(buckets))


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"DFlash operating point requires boolean {key}")
    return value


def _require_int(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"DFlash operating point requires integer {key}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"DFlash operating point requires integer {key}") from error
    if result < minimum:
        raise ValueError(f"DFlash operating point requires {key} >= {minimum}")
    return result


def _require_float(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"DFlash operating point requires numeric {key}") from error


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"DFlash operating point requires non-empty string {key}")
    return value


def dflash_operating_point_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the complete target-plus-draft operating point from runtime metadata."""

    if runtime.get("speculative_decoding") is not True:
        raise ValueError("speculative benchmark runtime did not enable DFlash")
    point = {
        "schema_version": DFLASH_OPERATING_POINT_SCHEMA,
        "max_model_len": _require_int(runtime, "max_model_len", minimum=1),
        "max_num_seqs": _require_int(runtime, "max_num_seqs", minimum=1),
        "scheduler_queue_size": _require_int(runtime, "scheduler_queue_size", minimum=1),
        "continuous_batching_enabled": _require_bool(runtime, "continuous_batching_enabled"),
        "tensor_parallel_size": _require_int(runtime, "tensor_parallel_size", minimum=1),
        "execution_mode": _require_string(runtime, "execution_mode"),
        "gpu_memory_utilization": _require_float(runtime, "gpu_memory_utilization"),
        "attention_backend": _require_string(runtime, "attention_backend"),
        "flash_attn_version": _require_int(runtime, "flash_attn_version", minimum=1),
        "hlm_attention_impl": _require_string(runtime, "hlm_attention_impl"),
        "vllm_use_v2_model_runner": _require_string(runtime, "vllm_use_v2_model_runner"),
        "speculative_num_tokens": _require_int(
            runtime, "speculative_num_tokens", minimum=1
        ),
        "speculative_verification_mode": _require_string(
            runtime, "speculative_verification_mode"
        ),
        "draft_attention_backend": _require_string(
            runtime, "speculative_draft_attention_backend"
        ),
        "context_kv_cache": _require_bool(runtime, "speculative_context_kv_cache"),
        "sparse_context_projection": _require_bool(
            runtime, "speculative_sparse_context_projection"
        ),
        "min_eligible_batch": _require_int(
            runtime, "speculative_min_eligible_batch", minimum=1
        ),
        "min_proposal_tokens_per_row": _require_int(
            runtime, "speculative_min_proposal_tokens_per_row", minimum=1
        ),
        "min_proposal_tokens_per_batch": _require_int(
            runtime, "speculative_min_proposal_tokens_per_batch", minimum=1
        ),
        "runtime_block_size": _require_int(runtime, "speculative_runtime_block_size"),
        "active_batch_widths": str(runtime.get("speculative_active_batch_widths", "")),
        "dynamic_runtime_block_size": _require_bool(
            runtime, "speculative_dynamic_runtime_block_size"
        ),
        "runtime_layer_count": _require_int(runtime, "speculative_runtime_layer_count"),
        "runtime_local_mixer": _require_string(
            runtime, "speculative_runtime_local_mixer"
        ),
        "mixer_compile_mode": _require_string(runtime, "speculative_mixer_compile_mode"),
        "chunk_size": _require_int(runtime, "speculative_chunk_size", minimum=1),
        "target_layers": _require_string(runtime, "speculative_target_layers"),
        "telemetry_flush_interval": _require_int(
            runtime, "speculative_telemetry_flush_interval", minimum=1
        ),
    }
    point["active_batch_widths"] = canonical_active_batch_widths(
        point["active_batch_widths"], point["speculative_num_tokens"]
    )
    validate_dflash_operating_point(point)
    return point


def validate_dflash_operating_point(
    point: Mapping[str, Any], *, benchmark_contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a canonical operating point or fail closed on an incomplete one."""

    if point.get("schema_version") != DFLASH_OPERATING_POINT_SCHEMA:
        raise ValueError("unsupported or missing DFlash operating-point schema")
    if "speculative_decoding" in point:
        return dflash_operating_point_from_runtime(point)

    max_model_len = _require_int(point, "max_model_len", minimum=1)
    max_num_seqs = _require_int(point, "max_num_seqs", minimum=1)
    queue_size = _require_int(point, "scheduler_queue_size", minimum=1)
    if queue_size < max_num_seqs:
        raise ValueError("DFlash scheduler queue must be at least max_num_seqs")
    continuous = _require_bool(point, "continuous_batching_enabled")
    if continuous != (queue_size > max_num_seqs):
        raise ValueError("DFlash continuous-batching flag disagrees with batch/queue")
    if max_num_seqs > 8:
        raise ValueError("DFlash max_num_seqs exceeds the validated limit of 8")
    utilization = _require_float(point, "gpu_memory_utilization")
    if not 0.0 < utilization < 1.0:
        raise ValueError("DFlash gpu_memory_utilization must be in (0, 1)")
    speculative_width = _require_int(point, "speculative_num_tokens", minimum=1)
    if speculative_width > 16:
        raise ValueError("DFlash speculative_num_tokens exceeds 16")
    active_widths = canonical_active_batch_widths(
        str(point.get("active_batch_widths", "")), speculative_width
    )
    if _require_bool(point, "sparse_context_projection") and not _require_bool(
        point, "context_kv_cache"
    ):
        raise ValueError("DFlash sparse context projection requires context KV cache")
    for key in (
        "tensor_parallel_size",
        "flash_attn_version",
        "min_eligible_batch",
        "min_proposal_tokens_per_row",
        "min_proposal_tokens_per_batch",
        "chunk_size",
        "telemetry_flush_interval",
    ):
        _require_int(point, key, minimum=1)
    for key in ("runtime_block_size", "runtime_layer_count"):
        _require_int(point, key)
    for key in (
        "execution_mode",
        "attention_backend",
        "hlm_attention_impl",
        "vllm_use_v2_model_runner",
        "speculative_verification_mode",
        "draft_attention_backend",
        "runtime_local_mixer",
        "mixer_compile_mode",
        "target_layers",
    ):
        _require_string(point, key)
    _require_bool(point, "dynamic_runtime_block_size")
    if benchmark_contract is not None:
        batch_size = _require_int(benchmark_contract, "batch_size", minimum=1)
        contract_queue = _require_int(
            benchmark_contract, "scheduler_queue_size", minimum=1
        )
        prompt_count = _require_int(benchmark_contract, "prompt_count", minimum=1)
        if batch_size != max_num_seqs or contract_queue != queue_size:
            raise ValueError("DFlash benchmark batch/queue differs from its operating point")
        if queue_size > max_num_seqs and prompt_count < queue_size:
            raise ValueError(
                "continuous-batch A/B must include at least one full scheduler queue"
            )
        if max_model_len < _require_int(benchmark_contract, "max_new_tokens", minimum=1):
            raise ValueError("DFlash max_model_len is smaller than benchmark generation length")
    result = dict(point)
    result["active_batch_widths"] = active_widths
    result["gpu_memory_utilization"] = utilization
    return result


def dflash_operating_point_env(point: Mapping[str, Any]) -> dict[str, str]:
    """Render every draft runtime knob from one validated immutable point."""

    point = validate_dflash_operating_point(point)
    env = {
        "CONCEPTLM_DFLASH_ATTENTION_BACKEND": str(point["draft_attention_backend"]),
        "CONCEPTLM_DFLASH_CONTEXT_KV_CACHE": "1" if point["context_kv_cache"] else "0",
        "CONCEPTLM_DFLASH_SPARSE_CONTEXT_PROJECTION": (
            "1" if point["sparse_context_projection"] else "0"
        ),
        "CONCEPTLM_DFLASH_MIN_ELIGIBLE_BATCH": str(point["min_eligible_batch"]),
        "CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW": str(
            point["min_proposal_tokens_per_row"]
        ),
        "CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH": str(
            point["min_proposal_tokens_per_batch"]
        ),
        "CONCEPTLM_DFLASH_RUNTIME_BLOCK_SIZE": str(point["runtime_block_size"]),
        "CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS": str(point["active_batch_widths"]),
        "CONCEPTLM_DFLASH_DYNAMIC_RUNTIME_BLOCK_SIZE": (
            "1" if point["dynamic_runtime_block_size"] else "0"
        ),
        "CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT": str(point["runtime_layer_count"]),
        "CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER": str(point["runtime_local_mixer"]),
        "CONCEPTLM_DFLASH_MIXER_COMPILE_MODE": str(point["mixer_compile_mode"]),
        "CONCEPTLM_DFLASH_CHUNK_SIZE": str(point["chunk_size"]),
        "CONCEPTLM_DFLASH_TARGET_LAYERS": str(point["target_layers"]),
        "CONCEPTLM_DFLASH_MAX_MODEL_LEN": str(point["max_model_len"]),
        "CONCEPTLM_DFLASH_TELEMETRY_FLUSH_INTERVAL": str(
            point["telemetry_flush_interval"]
        ),
    }
    model_runner = str(point["vllm_use_v2_model_runner"])
    if model_runner != "auto":
        env["VLLM_USE_V2_MODEL_RUNNER"] = model_runner
    return env
