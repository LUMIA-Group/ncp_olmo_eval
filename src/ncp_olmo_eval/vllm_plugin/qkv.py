"""Tensor transforms for native Megatron QKV checkpoints."""

from __future__ import annotations

import torch


def deinterleave_megatron_qkv(
    loaded_weight: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Megatron grouped QKV rows into vLLM's contiguous Q/K/V shards."""

    if num_attention_heads % num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    output_size, hidden_size = loaded_weight.shape
    head_dim = hidden_size // num_attention_heads
    query_heads_per_group = num_attention_heads // num_key_value_heads
    expected_output_size = (
        num_attention_heads + 2 * num_key_value_heads
    ) * head_dim
    if hidden_size % num_attention_heads or output_size != expected_output_size:
        raise ValueError(
            "invalid Megatron QKV shape: "
            f"found {tuple(loaded_weight.shape)}, expected "
            f"({expected_output_size}, {hidden_size})"
        )
    grouped = loaded_weight.reshape(
        num_key_value_heads,
        query_heads_per_group + 2,
        head_dim,
        hidden_size,
    )
    query = grouped[:, :query_heads_per_group].reshape(
        num_attention_heads * head_dim, hidden_size
    )
    key = grouped[:, query_heads_per_group].reshape(
        num_key_value_heads * head_dim, hidden_size
    )
    value = grouped[:, query_heads_per_group + 1].reshape(
        num_key_value_heads * head_dim, hidden_size
    )
    return query, key, value
