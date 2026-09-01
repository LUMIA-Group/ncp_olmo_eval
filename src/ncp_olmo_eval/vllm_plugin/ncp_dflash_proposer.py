"""Correctness-gated NCP DFlash proposer for vLLM 0.13."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from .ncp_dflash_state import append_telemetry, target_model


def _parse_active_batch_widths(raw_policy: str, maximum_width: int) -> tuple[tuple[int, int], ...]:
    """Parse upper-bound active-batch buckets into proposal-width caps."""

    raw_policy = raw_policy.strip()
    if not raw_policy:
        return ()
    buckets: dict[int, int] = {}
    for raw_bucket in raw_policy.split(","):
        upper_text, separator, width_text = raw_bucket.strip().partition(":")
        if not separator:
            raise ValueError(
                "CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS must use "
                "comma-separated upper_bound:width entries"
            )
        upper_bound = int(upper_text)
        width = int(width_text)
        if upper_bound < 1:
            raise ValueError("DFlash active-batch upper bounds must be positive")
        if not 0 <= width <= maximum_width:
            raise ValueError(
                "DFlash active-batch width must be between zero and the "
                f"configured speculative width: width={width} maximum={maximum_width}"
            )
        if upper_bound in buckets:
            raise ValueError(f"duplicate DFlash active-batch upper bound: {upper_bound}")
        buckets[upper_bound] = width
    return tuple(sorted(buckets.items()))


def _dflash_sdpa_mask(
    anchor_positions: torch.Tensor, *, context_length: int, block_size: int
) -> torch.Tensor:
    """Materialize the small-query DFlash visibility mask for SDPA."""

    batch_size, anchor_count = anchor_positions.shape
    query_length = anchor_count * block_size
    query_blocks = torch.arange(anchor_count, device=anchor_positions.device).repeat_interleave(
        block_size
    )
    valid_queries = (anchor_positions >= 0).repeat_interleave(block_size, dim=1)
    anchor_by_query = anchor_positions.repeat_interleave(block_size, dim=1)
    context_indices = torch.arange(context_length, device=anchor_positions.device)
    context_allowed = (
        context_indices.view(1, 1, context_length) < anchor_by_query.unsqueeze(-1)
    ) & valid_queries.unsqueeze(-1)
    draft_blocks = torch.arange(query_length, device=anchor_positions.device) // block_size
    draft_allowed = (
        query_blocks.view(1, query_length, 1) == draft_blocks.view(1, 1, query_length)
    ) & valid_queries.unsqueeze(-1)
    return torch.cat((context_allowed, draft_allowed), dim=-1).view(
        batch_size, 1, query_length, context_length + query_length
    )


def _dflash_cached_context_kv_rows(
    layer: torch.nn.Module, context: torch.Tensor, anchor_positions: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return request-local context KV rows with one packed suffix projection.

    The final context row is a zero placeholder for the current anchor.  Only
    rows strictly before the anchor are stable across proposal steps and may be
    cached.  New suffixes from every active request are concatenated before the
    linear projections so batch execution never degenerates into one GEMM per
    request.
    """

    batch_size, sequence_length, _ = context.shape
    request_ids = getattr(layer, "_ncp_dflash_request_ids", None)
    if request_ids is None:
        raise ValueError("DFlash cached context KV requires request IDs")

    if len(request_ids) != batch_size:
        raise ValueError("DFlash request IDs do not match the context batch")
    if int(anchor_positions.shape[0]) != batch_size:
        raise ValueError("DFlash anchor positions do not match the context batch")
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    if causal_lengths is not None and len(causal_lengths) != batch_size:
        raise ValueError("DFlash causal lengths do not match the context batch")
    context_offsets = getattr(layer, "_ncp_dflash_context_offsets", None)
    if context_offsets is not None and len(context_offsets) != batch_size:
        raise ValueError("DFlash context offsets do not match the context batch")

    cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
    if cache is None:
        cache = {}
        layer._ncp_dflash_context_kv_cache = cache
    projection_dtype = getattr(getattr(layer.k_proj, "weight", None), "dtype", context.dtype)
    records: list[tuple[str, int, int, torch.Tensor, torch.Tensor]] = []
    suffixes: list[torch.Tensor] = []
    suffix_positions: list[torch.Tensor] = []
    suffix_lengths: list[int] = []
    projected_tokens = 0
    reused_tokens = 0
    for row_index, raw_request_id in enumerate(request_ids):
        request_id = str(raw_request_id)
        if causal_lengths is None:
            valid_anchors = anchor_positions[row_index]
            valid_anchors = valid_anchors[valid_anchors >= 0]
            causal_length = int(valid_anchors.max().item()) if int(valid_anchors.numel()) else 0
        else:
            causal_length = int(causal_lengths[row_index])
        context_offset = 0 if context_offsets is None else int(context_offsets[row_index])
        if not (
            0 <= context_offset <= causal_length
            and causal_length - context_offset <= sequence_length
        ):
            raise ValueError(
                "DFlash causal context is outside the supplied context slice: "
                f"request={request_id!r} offset={context_offset} "
                f"causal={causal_length} supplied={sequence_length}"
            )

        cached = cache.get(request_id)
        if cached is None:
            cached_length = 0
            cached_key = torch.empty(
                (1, layer.num_attention_heads, 0, layer.head_size),
                device=context.device,
                dtype=projection_dtype,
            )
            cached_value = cached_key.clone()
        else:
            cached_length, cached_key, cached_value = cached
            if (
                cached_length > causal_length
                or cached_key.device != context.device
                or cached_key.dtype != projection_dtype
            ):
                cached_length = 0
                cached_key = torch.empty(
                    (1, layer.num_attention_heads, 0, layer.head_size),
                    device=context.device,
                    dtype=projection_dtype,
                )
                cached_value = cached_key.clone()

        suffix_length = causal_length - cached_length
        if suffix_length:
            if cached_length < context_offset:
                raise RuntimeError(
                    "DFlash compact context starts after the valid KV cache: "
                    f"request={request_id!r} cached={cached_length} "
                    f"offset={context_offset}"
                )
            relative_start = cached_length - context_offset
            relative_end = causal_length - context_offset
            suffixes.append(context[row_index, relative_start:relative_end])
            suffix_positions.append(
                torch.arange(cached_length, causal_length, device=context.device)
            )
            suffix_lengths.append(suffix_length)
            projected_tokens += suffix_length
        reused_tokens += min(cached_length, causal_length)
        records.append((request_id, causal_length, suffix_length, cached_key, cached_value))

    projected_keys: list[torch.Tensor] = []
    projected_values: list[torch.Tensor] = []
    if suffixes:
        packed_suffix = torch.cat(suffixes, dim=0).unsqueeze(0)
        packed_positions = torch.cat(suffix_positions, dim=0).unsqueeze(0)
        packed_key = layer._split_heads(layer.k_norm(layer.k_proj(packed_suffix)))
        packed_value = layer._split_heads(layer.v_proj(packed_suffix))
        packed_key = layer.rotary(packed_key, packed_positions).transpose(1, 2)
        packed_value = packed_value.transpose(1, 2)
        projected_keys = list(torch.split(packed_key, suffix_lengths, dim=2))
        projected_values = list(torch.split(packed_value, suffix_lengths, dim=2))

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    projected_index = 0
    for request_id, causal_length, suffix_length, cached_key, cached_value in records:
        if suffix_length:
            required_length = int(causal_length)
            capacity = int(cached_key.shape[2])
            if required_length > capacity:
                target_capacity = max(16, required_length, capacity * 2)
                new_capacity = 1 << (target_capacity - 1).bit_length()
                key_storage = cached_key.new_empty(
                    (1, layer.num_attention_heads, new_capacity, layer.head_size)
                )
                value_storage = cached_value.new_empty(key_storage.shape)
                cached_length = required_length - int(suffix_length)
                if cached_length:
                    key_storage[:, :, :cached_length].copy_(cached_key[:, :, :cached_length])
                    value_storage[:, :, :cached_length].copy_(cached_value[:, :, :cached_length])
                cached_key = key_storage
                cached_value = value_storage
            suffix_start = required_length - int(suffix_length)
            cached_key[:, :, suffix_start:required_length].copy_(
                projected_keys[projected_index].to(dtype=cached_key.dtype)
            )
            cached_value[:, :, suffix_start:required_length].copy_(
                projected_values[projected_index].to(dtype=cached_value.dtype)
            )
            projected_index += 1
        cache[request_id] = (causal_length, cached_key.detach(), cached_value.detach())
        keys.append(cached_key[:, :, :causal_length])
        values.append(cached_value[:, :, :causal_length])

    if projected_index != len(projected_keys):
        raise RuntimeError("DFlash packed context KV suffix accounting drifted")

    layer._ncp_dflash_context_kv_projected_tokens = (
        int(getattr(layer, "_ncp_dflash_context_kv_projected_tokens", 0)) + projected_tokens
    )
    layer._ncp_dflash_context_kv_reused_tokens = (
        int(getattr(layer, "_ncp_dflash_context_kv_reused_tokens", 0)) + reused_tokens
    )
    return keys, values


def _dflash_context_kv(
    layer: torch.nn.Module, context: torch.Tensor, anchor_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded context KV for the compatibility SDPA backend."""

    batch_size, supplied_length, _ = context.shape
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    sequence_length = (
        max((int(length) for length in causal_lengths), default=0)
        if causal_lengths is not None
        else supplied_length
    )
    request_ids = getattr(layer, "_ncp_dflash_request_ids", None)
    if request_ids is None:
        context_positions = torch.arange(sequence_length, device=context.device)
        context_key = layer._split_heads(layer.k_norm(layer.k_proj(context)))
        context_value = layer._split_heads(layer.v_proj(context))
        context_key = layer.rotary(
            context_key, context_positions.view(1, sequence_length).expand(batch_size, -1)
        ).transpose(1, 2)
        return context_key, context_value.transpose(1, 2)

    key_rows, value_rows = _dflash_cached_context_kv_rows(layer, context, anchor_positions)
    keys = []
    values = []
    for key, value in zip(key_rows, value_rows, strict=True):
        padding = sequence_length - int(key.shape[2])
        keys.append(key if padding == 0 else F.pad(key, (0, 0, 0, padding)))
        values.append(value if padding == 0 else F.pad(value, (0, 0, 0, padding)))
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def _sdpa_dflash_attention(
    layer: torch.nn.Module,
    slots: torch.Tensor,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    """Inference-only equivalent of the checkpoint's FlexAttention layer."""

    del block_mask
    batch_size, anchor_count, block_size, hidden_size = slots.shape
    causal_lengths = getattr(layer, "_ncp_dflash_causal_lengths", None)
    sequence_length = (
        max((int(length) for length in causal_lengths), default=0)
        if causal_lengths is not None
        else int(context.shape[1])
    )
    slot_positions = anchor_positions.clamp(min=0).unsqueeze(-1) + torch.arange(
        block_size, device=slots.device
    )

    query = layer._split_heads(layer.q_norm(layer.q_proj(slots)))
    slot_key = layer._split_heads(layer.k_norm(layer.k_proj(slots)))
    slot_value = layer._split_heads(layer.v_proj(slots))
    query = layer.rotary(query, slot_positions)
    slot_key = layer.rotary(slot_key, slot_positions)

    context_key, context_value = _dflash_context_kv(layer, context, anchor_positions)

    query_length = anchor_count * block_size
    query = (
        query.reshape(batch_size, query_length, layer.num_attention_heads, layer.head_size)
        .transpose(1, 2)
        .contiguous()
    )
    slot_key = slot_key.reshape(
        batch_size, query_length, layer.num_attention_heads, layer.head_size
    ).transpose(1, 2)
    slot_value = slot_value.reshape(
        batch_size, query_length, layer.num_attention_heads, layer.head_size
    ).transpose(1, 2)
    key = torch.cat((context_key, slot_key), dim=2)
    value = torch.cat((context_value, slot_value), dim=2)
    attention_mask = _dflash_sdpa_mask(
        anchor_positions, context_length=sequence_length, block_size=block_size
    )
    attended = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=layer.head_size**-0.5
    )
    return attended.transpose(1, 2).reshape(batch_size, anchor_count, block_size, hidden_size)


def _flash_varlen_dflash_attention(
    layer: torch.nn.Module,
    slots: torch.Tensor,
    context: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    """Packed FlashAttention for one DFlash anchor block per request."""

    del block_mask
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch_size, anchor_count, block_size, hidden_size = slots.shape
    if anchor_count != 1:
        raise ValueError("packed DFlash attention currently requires one anchor per request")
    slot_positions = anchor_positions.clamp(min=0).unsqueeze(-1) + torch.arange(
        block_size, device=slots.device
    )
    query = layer._split_heads(layer.q_norm(layer.q_proj(slots)))
    slot_key = layer._split_heads(layer.k_norm(layer.k_proj(slots)))
    slot_value = layer._split_heads(layer.v_proj(slots))
    query = layer.rotary(query, slot_positions)
    slot_key = layer.rotary(slot_key, slot_positions)

    context_keys, context_values = _dflash_cached_context_kv_rows(layer, context, anchor_positions)
    query_length = anchor_count * block_size
    query = query.reshape(
        batch_size * query_length, layer.num_attention_heads, layer.head_size
    ).contiguous()
    slot_key = slot_key.reshape(
        batch_size, query_length, layer.num_attention_heads, layer.head_size
    )
    slot_value = slot_value.reshape(
        batch_size, query_length, layer.num_attention_heads, layer.head_size
    )
    key_rows = [
        torch.cat(
            (
                context_key.squeeze(0).transpose(0, 1).to(dtype=query.dtype),
                slot_key[row_index].to(dtype=query.dtype),
            ),
            dim=0,
        )
        for row_index, context_key in enumerate(context_keys)
    ]
    value_rows = [
        torch.cat(
            (
                context_value.squeeze(0).transpose(0, 1).to(dtype=query.dtype),
                slot_value[row_index].to(dtype=query.dtype),
            ),
            dim=0,
        )
        for row_index, context_value in enumerate(context_values)
    ]
    key_lengths = [int(key.shape[0]) for key in key_rows]
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * query_length, query_length, device=slots.device, dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor([0, *key_lengths], device=slots.device, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    attended = flash_attn_varlen_func(
        q=query,
        k=torch.cat(key_rows, dim=0),
        v=torch.cat(value_rows, dim=0),
        max_seqlen_q=query_length,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max(key_lengths),
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=0.0,
        softmax_scale=layer.head_size**-0.5,
        causal=False,
        fa_version=3,
    )
    return attended.reshape(batch_size, anchor_count, block_size, hidden_size)


def _install_draft_attention_backend(model: torch.nn.Module) -> str:
    backend = os.environ.get("CONCEPTLM_DFLASH_ATTENTION_BACKEND", "sdpa")
    if backend not in {"flash_varlen", "sdpa", "flex_attention"}:
        raise ValueError(
            "CONCEPTLM_DFLASH_ATTENTION_BACKEND must be flash_varlen, sdpa, " "or flex_attention"
        )
    model.config.flex_attention_compile = False
    if backend != "flex_attention":
        # The checkpoint's remote-code ``forward`` always materializes a
        # FlexAttention BlockMask before entering its draft layers.  Both of
        # our inference backends implement the same visibility rule directly
        # and ignore that argument, so constructing the BlockMask on every
        # decode step is pure overhead.  Patch only this process-local remote
        # module; the checkpoint artifact remains untouched.
        forward_function = getattr(model.forward, "__func__", model.forward)
        forward_globals = getattr(forward_function, "__globals__", None)
        if not isinstance(forward_globals, dict) or (
            "_create_dflash_block_mask" not in forward_globals
        ):
            raise RuntimeError("DFlash remote model no longer exposes its block-mask helper")
        forward_globals["_create_dflash_block_mask"] = lambda *_args, **_kwargs: None
        model._ncp_dflash_unused_block_mask_disabled = True
    layers = list(model.layers)
    for layer in layers:
        layer.flex_attention_compile = False
        if backend == "sdpa":
            layer._attention = MethodType(_sdpa_dflash_attention, layer)
        elif backend == "flash_varlen":
            layer._attention = MethodType(_flash_varlen_dflash_attention, layer)
    return backend


def _static_two_tap_conv_forward(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Use the learned two-tap kernel without its small dynamic correction.

    The full mixer launches two narrow GEMMs plus several elementwise kernels
    at each of four sites in every draft layer.  Its correction is bounded by
    ``0.1 * tanh`` while the learned base kernel carries the dominant identity
    and previous-slot paths.  This opt-in approximation preserves that trained
    base path and is substantially less destructive than deleting the mixer.
    Target verification remains authoritative.
    """

    previous = torch.cat((hidden_states[:, :, :1], hidden_states[:, :, :-1]), dim=2)
    coefficients = module.base_kernel.to(dtype=hidden_states.dtype)
    return coefficients[0] * hidden_states + coefficients[1] * previous


_COMPILED_DYNAMIC_MIXERS: dict[str, Any] = {}


def _dynamic_two_tap_functional(
    hidden_states: torch.Tensor,
    base_kernel: torch.Tensor,
    condition_down_weight: torch.Tensor,
    condition_up_weight: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    previous = torch.cat((hidden_states[:, :, :1], hidden_states[:, :, :-1]), dim=2)
    correction = F.linear(
        F.silu(F.linear(hidden_states, condition_down_weight)), condition_up_weight
    )
    num_groups = int(hidden_states.shape[-1]) // group_size
    correction = correction.unflatten(-1, (num_groups, 2)).transpose(-1, -2)
    correction = correction.repeat_interleave(group_size, dim=-1)
    coefficients = base_kernel.to(dtype=hidden_states.dtype) + 0.1 * torch.tanh(correction)
    return coefficients[..., 0, :] * hidden_states + coefficients[..., 1, :] * previous


def _compiled_two_tap_conv_forward(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    mode = str(getattr(module, "_ncp_dflash_compile_mode", "default"))
    compiled = _COMPILED_DYNAMIC_MIXERS.get(mode)
    if compiled is None:
        compiled = torch.compile(
            _dynamic_two_tap_functional, dynamic=True, fullgraph=True, mode=mode
        )
        _COMPILED_DYNAMIC_MIXERS[mode] = compiled
    return compiled(
        hidden_states,
        module.base_kernel,
        module.condition_down.weight,
        module.condition_up.weight,
        int(module.group_size),
    )


def _install_runtime_local_mixer(model: torch.nn.Module, mode: str) -> str:
    if mode not in {
        "full",
        "compile",
        "hybrid_first_pre_attention",
        "hybrid_pre_attention",
        "static",
        "none",
    }:
        raise ValueError(
            "CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER must be full, compile, "
            "hybrid_first_pre_attention, hybrid_pre_attention, static, or none"
        )
    mixer_names = (
        "pre_attention_conv",
        "post_attention_conv",
        "pre_feedforward_conv",
        "post_feedforward_conv",
    )
    for layer_index, layer in enumerate(model.layers):
        for name in mixer_names:
            mixer = getattr(layer, name, None)
            if mixer is None or mode == "full":
                continue
            keep_dynamic = (mode == "hybrid_pre_attention" and name == "pre_attention_conv") or (
                mode == "hybrid_first_pre_attention"
                and layer_index == 0
                and name == "pre_attention_conv"
            )
            if keep_dynamic:
                continue
            if mode == "compile":
                mixer._ncp_dflash_compile_mode = os.environ.get(
                    "CONCEPTLM_DFLASH_MIXER_COMPILE_MODE", "default"
                )
                mixer.forward = MethodType(_compiled_two_tap_conv_forward, mixer)
            elif mode in {"hybrid_first_pre_attention", "hybrid_pre_attention", "static"}:
                mixer.forward = MethodType(_static_two_tap_conv_forward, mixer)
            else:
                setattr(layer, name, None)
    return mode


class ConceptLMDFlashProposer:
    """Use native target feature histories to propose a draft suffix.

    ``segmented_kv_approx`` may cross HLM chunk boundaries because the target
    state is committed or rolled back immediately after rejection sampling.
    This mirrors the original NCPDFlash segmented-cache experiment's state
    reuse boundary, but it is explicitly approximate: strict target-only A/B
    has shown token divergence for the causal-HLM checkpoint. Exact modes are
    kept separate and the approximate mode is never selected implicitly.
    """

    def __init__(self, vllm_config: Any) -> None:
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "ngram":
            raise ValueError("ConceptLMDFlashProposer requires vLLM 0.13 ngram plumbing")
        raw_checkpoint = os.environ.get("CONCEPTLM_DFLASH_CHECKPOINT", "")
        if not raw_checkpoint:
            raise ValueError("CONCEPTLM_DFLASH_CHECKPOINT is required")
        self.checkpoint = Path(raw_checkpoint).resolve()
        self.num_speculative_tokens = int(speculative_config.num_speculative_tokens)
        self.max_model_len = int(
            os.environ.get(
                "CONCEPTLM_DFLASH_MAX_MODEL_LEN", str(vllm_config.model_config.max_model_len)
            )
        )
        self.max_batch_size = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        self.active_batch_widths = _parse_active_batch_widths(
            os.environ.get("CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS", ""),
            self.num_speculative_tokens,
        )
        self.dynamic_runtime_block_size = (
            os.environ.get("CONCEPTLM_DFLASH_DYNAMIC_RUNTIME_BLOCK_SIZE", "0") == "1"
        )
        self.chunk_size = int(os.environ.get("CONCEPTLM_DFLASH_CHUNK_SIZE", "4"))
        self.context_kv_cache = os.environ.get("CONCEPTLM_DFLASH_CONTEXT_KV_CACHE", "0") == "1"
        self.sparse_context_projection = (
            os.environ.get("CONCEPTLM_DFLASH_SPARSE_CONTEXT_PROJECTION", "0") == "1"
        )
        if self.sparse_context_projection and not self.context_kv_cache:
            raise ValueError("DFlash sparse context projection requires context KV cache")
        self.min_eligible_batch = int(os.environ.get("CONCEPTLM_DFLASH_MIN_ELIGIBLE_BATCH", "1"))
        if self.min_eligible_batch < 1:
            raise ValueError("CONCEPTLM_DFLASH_MIN_ELIGIBLE_BATCH must be at least 1")
        self.min_proposal_tokens_per_row = int(
            os.environ.get("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW", "1")
        )
        if self.min_proposal_tokens_per_row < 1:
            raise ValueError("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW must be at least 1")
        self.min_proposal_tokens_per_batch = int(
            os.environ.get("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH", "1")
        )
        if self.min_proposal_tokens_per_batch < 1:
            raise ValueError("CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH must be at least 1")
        self.runtime_block_size = int(os.environ.get("CONCEPTLM_DFLASH_RUNTIME_BLOCK_SIZE", "0"))
        if self.runtime_block_size < 0:
            raise ValueError("CONCEPTLM_DFLASH_RUNTIME_BLOCK_SIZE must be non-negative")
        self.runtime_layer_count = int(os.environ.get("CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT", "0"))
        if self.runtime_layer_count < 0:
            raise ValueError("CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT must be non-negative")
        self.runtime_local_mixer = os.environ.get("CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER", "full")
        if self.runtime_local_mixer not in {
            "full",
            "compile",
            "hybrid_first_pre_attention",
            "hybrid_pre_attention",
            "static",
            "none",
        }:
            raise ValueError(
                "CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER must be full, compile, "
                "hybrid_first_pre_attention, hybrid_pre_attention, static, "
                "or none"
            )
        raw_verification_mode = os.environ.get(
            "CONCEPTLM_DFLASH_VERIFICATION_MODE", "sequential_exact"
        )
        self.verification_mode = {
            "chunk_parallel": "intra_chunk_exact",
            "transactional_exact": "segmented_kv_approx",
        }.get(raw_verification_mode, raw_verification_mode)
        if self.verification_mode not in {
            "sequential_exact",
            "intra_chunk_exact",
            "segmented_kv_approx",
        }:
            raise ValueError(
                "CONCEPTLM_DFLASH_VERIFICATION_MODE must be sequential_exact "
                "intra_chunk_exact, or segmented_kv_approx"
            )
        self._draft_model: torch.nn.Module | None = None
        self._draft_device: torch.device | None = None
        append_telemetry(
            "proposer_initialized",
            checkpoint=str(self.checkpoint),
            num_speculative_tokens=self.num_speculative_tokens,
            max_model_len=self.max_model_len,
            max_batch_size=self.max_batch_size,
            active_batch_widths=self.active_batch_widths,
            dynamic_runtime_block_size=self.dynamic_runtime_block_size,
            chunk_size=self.chunk_size,
            proposal_window=(
                "segmented_kv_approx_cross_chunk"
                if self.verification_mode == "segmented_kv_approx"
                else "never_complete_hlm_chunk"
            ),
            verification_mode=self.verification_mode,
            output_contract=(
                "approximate" if self.verification_mode == "segmented_kv_approx" else "target_exact"
            ),
            context_kv_cache=self.context_kv_cache,
            sparse_context_projection=self.sparse_context_projection,
            min_eligible_batch=self.min_eligible_batch,
            min_proposal_tokens_per_row=self.min_proposal_tokens_per_row,
            min_proposal_tokens_per_batch=self.min_proposal_tokens_per_batch,
            requested_runtime_block_size=self.runtime_block_size,
            requested_runtime_layer_count=self.runtime_layer_count,
            runtime_local_mixer=self.runtime_local_mixer,
        )

    def _resolve_runtime_block_size(self, checkpoint_block_size: int) -> int:
        """Resolve a safe inference-time draft block width.

        The tuned checkpoint was trained with a 16-slot block, but exact
        ConceptLM verification can consume at most two slots before the next
        HLM boundary.  A smaller runtime block is therefore a valid draft-only
        speed experiment: it changes proposals, never target verification.
        Keep the checkpoint width by default and require an explicitly large
        enough override for the selected verification mode.
        """

        checkpoint_block_size = int(checkpoint_block_size)
        if checkpoint_block_size <= 0:
            raise ValueError("DFlash checkpoint block_size must be positive")
        if self.verification_mode == "sequential_exact":
            required = min(self.num_speculative_tokens, 1)
        elif self.verification_mode == "intra_chunk_exact":
            required = min(self.num_speculative_tokens, max(1, self.chunk_size - 2))
        else:
            required = self.num_speculative_tokens
        requested = int(getattr(self, "runtime_block_size", 0))
        runtime_block_size = checkpoint_block_size if requested == 0 else requested
        if runtime_block_size < required:
            raise ValueError(
                "DFlash runtime block is smaller than the maximum proposal "
                f"window: runtime={runtime_block_size} required={required}"
            )
        if runtime_block_size > checkpoint_block_size:
            raise ValueError(
                "DFlash runtime block exceeds the trained checkpoint block: "
                f"runtime={runtime_block_size} checkpoint={checkpoint_block_size}"
            )
        return runtime_block_size

    def _resolve_runtime_layer_count(self, checkpoint_layer_count: int) -> int:
        checkpoint_layer_count = int(checkpoint_layer_count)
        if checkpoint_layer_count <= 0:
            raise ValueError("DFlash checkpoint must contain draft layers")
        requested = int(getattr(self, "runtime_layer_count", 0))
        runtime_layer_count = checkpoint_layer_count if requested == 0 else requested
        if not 1 <= runtime_layer_count <= checkpoint_layer_count:
            raise ValueError(
                "DFlash runtime layer count must be within the checkpoint: "
                f"runtime={runtime_layer_count} checkpoint={checkpoint_layer_count}"
            )
        return runtime_layer_count

    def _active_batch_width(self, active_batch_size: int) -> int:
        """Return the proposal cap for the current continuous-batch step."""

        active_batch_size = int(active_batch_size)
        if active_batch_size <= 0:
            return 0
        policy = getattr(self, "active_batch_widths", ())
        if not policy:
            return self.num_speculative_tokens
        for upper_bound, width in policy:
            if active_batch_size <= upper_bound:
                return width
        return policy[-1][1]

    def _load_draft(self, device: torch.device) -> torch.nn.Module:
        if self._draft_model is not None:
            if self._draft_device != device:
                raise RuntimeError(
                    f"DFlash target device changed: {self._draft_device} -> {device}"
                )
            return self._draft_model
        from transformers import AutoModel

        started = time.perf_counter()
        model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        model.gradient_checkpointing = False
        model.config.gradient_checkpointing = False
        checkpoint_layer_count = len(model.layers)
        runtime_layer_count = self._resolve_runtime_layer_count(checkpoint_layer_count)
        if runtime_layer_count != checkpoint_layer_count:
            model.layers = torch.nn.ModuleList(list(model.layers)[:runtime_layer_count])
        checkpoint_block_size = int(model.config.block_size)
        runtime_block_size = self._resolve_runtime_block_size(checkpoint_block_size)
        model.config.block_size = runtime_block_size
        self._draft_runtime_max_block_size = runtime_block_size
        attention_backend = _install_draft_attention_backend(model)
        runtime_local_mixer = _install_runtime_local_mixer(model, self.runtime_local_mixer)
        if runtime_local_mixer == "compile":
            first_layer = model.layers[0]
            first_mixer = first_layer.pre_attention_conv
            compile_batches = sorted({1, self.max_batch_size})
            for batch_size in compile_batches:
                dummy = torch.zeros(
                    (batch_size, 1, runtime_block_size, int(model.config.draft_hidden_size)),
                    device=device,
                    dtype=torch.bfloat16,
                )
                first_mixer(dummy)
            torch.cuda.synchronize(device)
            del dummy
        if attention_backend == "flash_varlen" and not self.context_kv_cache:
            raise ValueError(
                "flash_varlen DFlash attention requires request-local context KV cache"
            )
        if str(model.config.proposal_method) != "path_selector":
            raise ValueError("the tuned NCP drafter must use path_selector")
        if str(model.config.hlm_conditioning) != "causal_residual":
            raise ValueError("the tuned NCP drafter must use causal_residual HLM state")
        if int(model.config.concept_chunk_size) != self.chunk_size:
            raise ValueError("target and DFlash concept chunk sizes do not match")
        self._draft_model = model
        self._draft_device = device
        self._draft_attention_backend = attention_backend
        append_telemetry(
            "draft_loaded",
            elapsed_seconds=time.perf_counter() - started,
            device=str(device),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            attention_backend=attention_backend,
            unused_flex_block_mask_disabled=bool(
                getattr(model, "_ncp_dflash_unused_block_mask_disabled", False)
            ),
            checkpoint_block_size=checkpoint_block_size,
            runtime_block_size=runtime_block_size,
            checkpoint_layer_count=checkpoint_layer_count,
            runtime_layer_count=runtime_layer_count,
            runtime_local_mixer=runtime_local_mixer,
            mixer_compile_prewarm_batches=(
                compile_batches if runtime_local_mixer == "compile" else []
            ),
        )
        return model

    @staticmethod
    def _path_selector_tokens(
        model: torch.nn.Module,
        hidden: torch.Tensor,
        output_weight: torch.Tensor,
        embedding_weight: torch.Tensor,
        previous_token_id: int,
        proposal_count: int,
    ) -> list[int]:
        base_logits = torch.matmul(hidden[:proposal_count], output_weight.transpose(0, 1))
        proposals: list[int] = []
        previous = int(previous_token_id)
        for position in range(proposal_count):
            topk_values, topk_ids = torch.topk(
                base_logits[position].float(), k=int(model.config.selector_top_k)
            )
            previous_embedding = embedding_weight[previous]
            candidate_embeddings = embedding_weight[topk_ids]
            previous_feature = model.selector_previous_projection(previous_embedding).float()
            candidate_features = model.selector_candidate_projection(candidate_embeddings).float()
            context_gate = torch.sigmoid(
                model.selector_context_projection(hidden[position]).float()
            )
            compatibility = torch.sum(
                candidate_features * (previous_feature * context_gate).unsqueeze(0), dim=-1
            ) / math.sqrt(float(previous_feature.shape[-1]))
            selected = int(topk_ids[(topk_values + compatibility).argmax()].item())
            proposals.append(selected)
            previous = selected
        return proposals

    @staticmethod
    def _path_selector_batch(
        model: torch.nn.Module,
        hidden: torch.Tensor,
        output_weight: torch.Tensor,
        embedding_weight: torch.Tensor,
        previous_token_ids: list[int] | torch.Tensor,
        proposal_counts: list[int],
    ) -> list[list[int]]:
        """Batch the expensive vocabulary projection across request rows."""

        batch_size = int(hidden.shape[0])
        if len(previous_token_ids) != batch_size or len(proposal_counts) != batch_size:
            raise ValueError("path-selector batch metadata has the wrong length")
        max_count = max(proposal_counts, default=0)
        if max_count <= 0:
            return [[] for _ in range(batch_size)]
        base_logits = torch.matmul(hidden[:, :max_count], output_weight.transpose(0, 1))
        topk_values, topk_ids = torch.topk(
            base_logits.float(), k=int(model.config.selector_top_k), dim=-1
        )
        del base_logits
        if isinstance(previous_token_ids, torch.Tensor):
            if tuple(previous_token_ids.shape) != (batch_size,):
                raise ValueError("path-selector previous-token tensor has the wrong shape")
            previous = previous_token_ids.to(device=hidden.device, dtype=torch.long).clone()
        else:
            previous = torch.tensor(previous_token_ids, dtype=torch.long, device=hidden.device)
        proposal_tokens = torch.full(
            (batch_size, max_count), -1, dtype=torch.long, device=hidden.device
        )
        selector_scale = math.sqrt(float(model.selector_previous_projection.out_features))
        uniform_count = all(count == max_count for count in proposal_counts)
        proposal_counts_tensor = (
            None
            if uniform_count
            else torch.tensor(proposal_counts, dtype=torch.long, device=hidden.device)
        )
        for position in range(max_count):
            active = (
                slice(None)
                if proposal_counts_tensor is None
                else proposal_counts_tensor > position
            )
            position_topk_values = topk_values[active, position]
            position_topk_ids = topk_ids[active, position]
            previous_embeddings = embedding_weight[previous[active]]
            candidate_embeddings = embedding_weight[position_topk_ids]
            previous_features = model.selector_previous_projection(previous_embeddings).float()
            candidate_features = model.selector_candidate_projection(candidate_embeddings).float()
            context_gates = torch.sigmoid(
                model.selector_context_projection(hidden[active, position]).float()
            )
            compatibility = (
                torch.sum(
                    candidate_features * (previous_features * context_gates).unsqueeze(1), dim=-1
                )
                / selector_scale
            )
            selected_indices = (position_topk_values + compatibility).argmax(dim=-1)
            selected = position_topk_ids.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
            previous[active] = selected
            proposal_tokens[active, position] = selected
        proposal_rows = proposal_tokens.tolist()
        return [
            [int(token_id) for token_id in row[:count]]
            for row, count in zip(proposal_rows, proposal_counts, strict=True)
        ]

    @staticmethod
    def _pad_context_batch(contexts: list[torch.Tensor]) -> torch.Tensor:
        """Pad variable request histories for one batched draft forward."""

        if not contexts:
            raise ValueError("cannot pad an empty DFlash context batch")
        max_length = max(int(context.shape[1]) for context in contexts)
        padded = []
        for context in contexts:
            if context.ndim != 4 or int(context.shape[0]) != 1:
                raise ValueError("DFlash target context must be [1, sequence, features, hidden]")
            padding = max_length - int(context.shape[1])
            padded.append(F.pad(context, (0, 0, 0, 0, 0, padding)))
        return torch.cat(padded, dim=0)

    @staticmethod
    def _valid_cached_context_length(
        layer: torch.nn.Module, request_id: str, *, causal_length: int, device: torch.device
    ) -> int:
        cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
        cached = None if cache is None else cache.get(str(request_id))
        if cached is None:
            return 0
        cached_length, cached_key, _cached_value = cached
        projection_dtype = getattr(getattr(layer.k_proj, "weight", None), "dtype", cached_key.dtype)
        if (
            int(cached_length) > int(causal_length)
            or cached_key.device != device
            or cached_key.dtype != projection_dtype
        ):
            return 0
        return int(cached_length)

    def _forward_sparse_context(
        self,
        draft: torch.nn.Module,
        contexts: list[torch.Tensor],
        *,
        request_ids: list[str],
        prefix_lengths: list[int],
        anchor_embeddings: torch.Tensor,
        mask_embedding: torch.Tensor,
        anchor_positions: torch.Tensor,
        hlm_hidden_states: torch.Tensor,
    ) -> Any:
        """Run the drafter while projecting only uncached context suffixes.

        Attention KV rows strictly before the current anchor are already
        request-local and immutable.  The checkpoint's stock ``forward`` still
        reprojects the complete target-feature history at every decode step
        before those KV rows are reused.  Build identical per-layer context
        rows only from the suffix that at least one layer has not cached; the
        attention backend ignores all earlier tensor rows and reads its cached
        KV instead.
        """

        batch_size = len(contexts)
        if not (len(request_ids) == batch_size == len(prefix_lengths)):
            raise ValueError("DFlash sparse context metadata is misaligned")
        suffix_starts: list[int] = []
        suffixes: list[torch.Tensor] = []
        suffix_lengths: list[int] = []
        for row_index, (request_id, prefix_length) in enumerate(
            zip(request_ids, prefix_lengths, strict=True)
        ):
            causal_length = int(prefix_length) - 1
            cached_lengths = [
                self._valid_cached_context_length(
                    layer,
                    request_id,
                    causal_length=causal_length,
                    device=contexts[row_index].device,
                )
                for layer in draft.layers
            ]
            suffix_start = min(cached_lengths, default=0)
            suffix_length = causal_length - suffix_start
            suffix_starts.append(suffix_start)
            suffix_lengths.append(suffix_length)
            if suffix_length:
                suffixes.append(contexts[row_index][0, suffix_start:causal_length])

        hidden_size = int(draft.config.draft_hidden_size)
        layer_suffixes: list[torch.Tensor]
        if suffixes:
            packed_features = torch.cat(suffixes, dim=0)
            shared_suffix = draft.feature_norm(
                draft.feature_projection(packed_features.flatten(start_dim=1))
            )
            layer_suffixes = [
                draft._context_for_layer(
                    packed_features.unsqueeze(0), shared_suffix.unsqueeze(0), layer_index
                ).squeeze(0)
                for layer_index in range(len(draft.layers))
            ]
        else:
            layer_suffixes = [anchor_embeddings.new_empty((0, hidden_size)) for _ in draft.layers]

        compact_context_length = max(suffix_lengths, default=0)

        block_size = int(draft.config.block_size)
        mask_slots = mask_embedding.view(1, 1, 1, -1).expand(
            batch_size, int(anchor_embeddings.shape[1]), block_size - 1, -1
        )
        target_slots = torch.cat((anchor_embeddings.unsqueeze(2), mask_slots), dim=2)
        slots = draft.input_projection(target_slots)
        uniform_suffix_length = (
            suffix_lengths[0]
            if suffix_lengths and all(length == suffix_lengths[0] for length in suffix_lengths)
            else None
        )
        packed_offset = 0
        for layer_index, layer in enumerate(draft.layers):
            layer._ncp_dflash_context_offsets = suffix_starts
            if uniform_suffix_length is not None:
                layer_context = layer_suffixes[layer_index].reshape(
                    batch_size, uniform_suffix_length, hidden_size
                )
            else:
                layer_context = slots.new_zeros(
                    (batch_size, compact_context_length, hidden_size)
                )
                for row_index, (suffix_start, suffix_length) in enumerate(
                    zip(suffix_starts, suffix_lengths, strict=True)
                ):
                    if suffix_length:
                        layer_context[row_index, :suffix_length] = layer_suffixes[layer_index][
                            packed_offset : packed_offset + suffix_length
                        ]
                    packed_offset += suffix_length
            packed_offset = 0
            layer_hlm = draft._hlm_for_layer(hlm_hidden_states, slots, layer_index)
            slots = layer(slots, layer_context, layer_hlm, anchor_positions, None)
        hidden = draft.output_projection(draft.final_norm(slots))
        self._last_sparse_context_projection = {
            "enabled": True,
            "projected_target_tokens": sum(suffix_lengths),
            "total_causal_target_tokens": sum(
                int(prefix_length) - 1 for prefix_length in prefix_lengths
            ),
        }
        return SimpleNamespace(last_hidden_state=hidden)

    def _propose_batch(
        self, entries: list[tuple[int, str, list[int], int]]
    ) -> dict[int, list[int]]:
        """Run all eligible request rows through one native batched drafter."""

        if not entries:
            return {}
        target = target_model()
        contexts: list[torch.Tensor] = []
        hlm_states: list[torch.Tensor] = []
        embedding_weight: torch.Tensor | None = None
        output_weight: torch.Tensor | None = None
        for _, request_id, prefix, _ in entries:
            context, hlm_state, row_embedding_weight, row_output_weight = (
                target.ncp_dflash_proposal_context(request_id, len(prefix))
            )
            if embedding_weight is None:
                embedding_weight = row_embedding_weight
                output_weight = row_output_weight
            elif (
                embedding_weight.data_ptr() != row_embedding_weight.data_ptr()
                or output_weight is None
                or output_weight.data_ptr() != row_output_weight.data_ptr()
            ):
                raise RuntimeError("DFlash request rows do not share target weights")
            contexts.append(context)
            hlm_states.append(hlm_state)
        assert embedding_weight is not None and output_weight is not None
        draft = self._load_draft(embedding_weight.device)
        prefixes = [prefix for _, _, prefix, _ in entries]
        prefix_lengths = [len(prefix) for prefix in prefixes]
        proposal_counts = [proposal_count for _, _, _, proposal_count in entries]
        proposal_block_size = max(proposal_counts)
        runtime_max_block_size = int(
            getattr(self, "_draft_runtime_max_block_size", draft.config.block_size)
        )
        if proposal_block_size > runtime_max_block_size:
            raise RuntimeError(
                "DFlash proposal block exceeds the loaded runtime block: "
                f"proposal={proposal_block_size} runtime={runtime_max_block_size}"
            )
        if getattr(self, "dynamic_runtime_block_size", False):
            draft.config.block_size = proposal_block_size
        else:
            draft.config.block_size = runtime_max_block_size
        anchor_ids = torch.tensor(
            [prefix[-1] for prefix in prefixes], dtype=torch.long, device=embedding_weight.device
        )
        anchor_embeddings = embedding_weight[anchor_ids].unsqueeze(1)
        mask_embedding = embedding_weight[int(draft.config.mask_token_id)]
        anchor_positions = torch.tensor(
            [[prefix_length - 1] for prefix_length in prefix_lengths],
            dtype=torch.long,
            device=embedding_weight.device,
        )
        sequence_lengths = torch.tensor(
            prefix_lengths, dtype=torch.long, device=embedding_weight.device
        )
        request_ids = [request_id for _, request_id, _, _ in entries]
        cache_counters_before = []
        for layer in draft.layers:
            cache_counters_before.append(
                (
                    int(getattr(layer, "_ncp_dflash_context_kv_projected_tokens", 0)),
                    int(getattr(layer, "_ncp_dflash_context_kv_reused_tokens", 0)),
                )
            )
            if self.context_kv_cache:
                layer._ncp_dflash_request_ids = request_ids
                layer._ncp_dflash_causal_lengths = [
                    prefix_length - 1 for prefix_length in prefix_lengths
                ]
                if self.sparse_context_projection:
                    layer._ncp_dflash_context_offsets = None
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            try:
                if self.sparse_context_projection:
                    output = self._forward_sparse_context(
                        draft,
                        contexts,
                        request_ids=request_ids,
                        prefix_lengths=prefix_lengths,
                        anchor_embeddings=anchor_embeddings,
                        mask_embedding=mask_embedding,
                        anchor_positions=anchor_positions,
                        hlm_hidden_states=torch.cat(hlm_states, dim=0),
                    )
                else:
                    output = draft(
                        aux_hidden_states=self._pad_context_batch(contexts),
                        anchor_embeddings=anchor_embeddings,
                        mask_embedding=mask_embedding,
                        anchor_positions=anchor_positions,
                        sequence_lengths=sequence_lengths,
                        hlm_hidden_states=torch.cat(hlm_states, dim=0),
                        return_dict=True,
                    )
            finally:
                for layer in draft.layers:
                    if hasattr(layer, "_ncp_dflash_request_ids"):
                        del layer._ncp_dflash_request_ids
                    if hasattr(layer, "_ncp_dflash_causal_lengths"):
                        del layer._ncp_dflash_causal_lengths
                    if hasattr(layer, "_ncp_dflash_context_offsets"):
                        del layer._ncp_dflash_context_offsets
            batched_tokens = self._path_selector_batch(
                draft,
                output.last_hidden_state[:, 0],
                output_weight,
                embedding_weight,
                anchor_ids,
                proposal_counts,
            )
            proposals = {
                row_index: batched_tokens[batch_index]
                for batch_index, (row_index, _, _, _) in enumerate(entries)
            }
        projected_tokens = 0
        reused_tokens = 0
        for layer, (projected_before, reused_before) in zip(
            draft.layers, cache_counters_before, strict=True
        ):
            projected_tokens += (
                int(getattr(layer, "_ncp_dflash_context_kv_projected_tokens", 0)) - projected_before
            )
            reused_tokens += (
                int(getattr(layer, "_ncp_dflash_context_kv_reused_tokens", 0)) - reused_before
            )
        self._last_context_kv_cache_stats = {
            "enabled": self.context_kv_cache,
            "projected_tokens": projected_tokens,
            "reused_tokens": reused_tokens,
            "sparse_context_projection": getattr(
                self,
                "_last_sparse_context_projection",
                {"enabled": False, "projected_target_tokens": 0, "total_causal_target_tokens": 0},
            ),
            "proposal_block_size": proposal_block_size,
            "runtime_block_size": int(draft.config.block_size),
        }
        return proposals

    def _prune_context_kv_cache(self, request_ids: list[str]) -> None:
        if (
            not getattr(self, "context_kv_cache", False)
            or getattr(self, "_draft_model", None) is None
        ):
            return
        active_request_ids = {str(request_id) for request_id in request_ids}
        for layer in self._draft_model.layers:
            cache = getattr(layer, "_ncp_dflash_context_kv_cache", None)
            if cache is None:
                continue
            for request_id in tuple(cache):
                if request_id not in active_request_ids:
                    del cache[request_id]

    def safe_proposal_count(
        self, prefix_length: int, *, proposal_cap: int | None = None
    ) -> int:
        """Return the verified proposal width for this target prefix."""

        anchor_position = int(prefix_length) - 1
        maximum_proposals = self.num_speculative_tokens
        if proposal_cap is not None:
            maximum_proposals = min(maximum_proposals, max(0, int(proposal_cap)))
        context_limited_count = max(
            0, min(maximum_proposals, self.max_model_len - int(prefix_length))
        )
        if self.verification_mode == "segmented_kv_approx":
            return context_limited_count
        safe_before_chunk = self.chunk_size - 2 - (anchor_position % self.chunk_size)
        proposal_count = min(context_limited_count, max(0, safe_before_chunk))
        if self.verification_mode == "sequential_exact":
            # Correctness baseline: verify at most one draft token per target
            # step.  The exact-match benchmark remains the enabling gate; the
            # wider chunk-parallel path is never selected implicitly.
            proposal_count = min(1, proposal_count)
        if proposal_count < int(getattr(self, "min_proposal_tokens_per_row", 1)):
            return 0
        return proposal_count

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        request_ids: list[str] | None = None,
        slot_mappings: Any | None = None,
    ) -> list[list[int]]:
        """Return one proposal list per vLLM request row."""

        del slot_mappings
        if request_ids is None:
            raise ValueError("DFlash request IDs are required for request-local state")
        if len(request_ids) != len(sampled_token_ids):
            raise ValueError("DFlash request IDs and sampled rows have different lengths")
        self._prune_context_kv_cache(request_ids)
        self._last_context_kv_cache_stats = {
            "enabled": getattr(self, "context_kv_cache", False),
            "projected_tokens": 0,
            "reused_tokens": 0,
        }
        started = time.perf_counter()
        results: list[list[int]] = [[] for _ in sampled_token_ids]
        reasons: dict[str, int] = {}
        proposal_details: list[dict[str, Any]] = []
        eligible: list[tuple[int, str, list[int], int]] = []
        active_decode_batch_size = sum(bool(sampled_ids) for sampled_ids in sampled_token_ids)
        active_batch_width = self._active_batch_width(active_decode_batch_size)
        for row_index, sampled_ids in enumerate(sampled_token_ids):
            request_id = str(request_ids[row_index])
            detail: dict[str, Any] = {"row_index": row_index, "request_id": request_id}
            if not sampled_ids:
                reasons["no_sampled_token"] = reasons.get("no_sampled_token", 0) + 1
                detail.update(prefix_length=None, proposals=[], reason="no_sampled_token")
                proposal_details.append(detail)
                continue
            prefix_length = int(num_tokens_no_spec[row_index])
            prefix = [int(value) for value in token_ids_cpu[row_index, :prefix_length]]
            if prefix_length >= self.max_model_len:
                reason = "max_model_len"
            elif active_batch_width <= 0:
                reason = "adaptive_width_disabled"
            else:
                proposal_count = self.safe_proposal_count(
                    prefix_length, proposal_cap=active_batch_width
                )
                reason = "proposed" if proposal_count > 0 else "hlm_chunk_boundary"
                if proposal_count > 0:
                    eligible.append((row_index, request_id, prefix, proposal_count))
            reasons[reason] = reasons.get(reason, 0) + 1
            detail.update(prefix_length=prefix_length, proposals=[], reason=reason)
            proposal_details.append(detail)

        min_eligible_batch = int(getattr(self, "min_eligible_batch", 1))
        configured_multi_request = int(getattr(self, "max_batch_size", len(results))) > 1
        if configured_multi_request and 0 < len(eligible) < min_eligible_batch:
            skipped_rows = {row_index for row_index, *_ in eligible}
            skipped_count = len(skipped_rows)
            reasons["proposed"] -= skipped_count
            if reasons["proposed"] == 0:
                del reasons["proposed"]
            reasons["adaptive_batch_too_small"] = skipped_count
            for detail in proposal_details:
                if int(detail["row_index"]) in skipped_rows:
                    detail["reason"] = "adaptive_batch_too_small"
            eligible = []

        minimum_batch_tokens = int(getattr(self, "min_proposal_tokens_per_batch", 1))
        batch_proposal_tokens = sum(entry[3] for entry in eligible)
        if eligible and batch_proposal_tokens < minimum_batch_tokens:
            skipped_rows = {row_index for row_index, *_ in eligible}
            skipped_count = len(skipped_rows)
            reasons["proposed"] -= skipped_count
            if reasons["proposed"] == 0:
                del reasons["proposed"]
            reasons["adaptive_proposal_budget_too_small"] = skipped_count
            for detail in proposal_details:
                if int(detail["row_index"]) in skipped_rows:
                    detail["reason"] = "adaptive_proposal_budget_too_small"
            eligible = []

        batched_proposals = self._propose_batch(eligible)
        for detail in proposal_details:
            row_index = int(detail["row_index"])
            proposals = batched_proposals.get(row_index, [])
            results[row_index] = proposals
            detail["proposals"] = proposals
        append_telemetry(
            "proposal_batch",
            elapsed_seconds=time.perf_counter() - started,
            request_count=len(results),
            eligible_request_count=len(eligible),
            proposed_tokens=sum(len(values) for values in results),
            active_decode_batch_size=active_decode_batch_size,
            active_batch_width=active_batch_width,
            context_kv_cache=getattr(
                self,
                "_last_context_kv_cache_stats",
                {
                    "enabled": getattr(self, "context_kv_cache", False),
                    "projected_tokens": 0,
                    "reused_tokens": 0,
                },
            ),
            reasons=reasons,
            details=proposal_details,
        )
        return results

    def load_model(self, *_: Any, **__: Any) -> None:
        """Match proposer protocols that optionally call a load hook."""

        return None

