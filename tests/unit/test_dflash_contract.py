from __future__ import annotations

import pytest

from ncp_olmo_eval.dflash_contract import (
    canonical_active_batch_widths,
    dflash_operating_point_env,
    validate_dflash_operating_point,
)


def _point() -> dict[str, object]:
    return {
        "schema_version": "ncp-dflash-operating-point-v1",
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "scheduler_queue_size": 32,
        "continuous_batching_enabled": True,
        "tensor_parallel_size": 1,
        "execution_mode": "eager",
        "gpu_memory_utilization": 0.8,
        "attention_backend": "FLASH_ATTN",
        "flash_attn_version": 3,
        "hlm_attention_impl": "legacy_mixed",
        "vllm_use_v2_model_runner": "0",
        "speculative_num_tokens": 8,
        "speculative_verification_mode": "segmented_kv_approx",
        "draft_attention_backend": "flash_varlen",
        "context_kv_cache": True,
        "sparse_context_projection": True,
        "min_eligible_batch": 1,
        "min_proposal_tokens_per_row": 1,
        "min_proposal_tokens_per_batch": 1,
        "runtime_block_size": 0,
        "active_batch_widths": "8:2,1:8,4:4,2:8",
        "dynamic_runtime_block_size": True,
        "runtime_layer_count": 5,
        "runtime_local_mixer": "full",
        "mixer_compile_mode": "default",
        "chunk_size": 4,
        "target_layers": "1,4,7,10,13",
        "telemetry_flush_interval": 8,
    }


def test_operating_point_preserves_original_batch_width_policy() -> None:
    point = validate_dflash_operating_point(
        _point(),
        benchmark_contract={
            "batch_size": 8,
            "scheduler_queue_size": 32,
            "prompt_count": 32,
            "max_new_tokens": 128,
        },
    )

    assert point["active_batch_widths"] == "1:8,2:8,4:4,8:2"
    env = dflash_operating_point_env(point)
    assert env["CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS"] == "1:8,2:8,4:4,8:2"
    assert env["CONCEPTLM_DFLASH_DYNAMIC_RUNTIME_BLOCK_SIZE"] == "1"
    assert env["CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT"] == "5"


def test_operating_point_rejects_batch_queue_drift() -> None:
    with pytest.raises(ValueError, match="batch/queue"):
        validate_dflash_operating_point(
            _point(),
            benchmark_contract={
                "batch_size": 8,
                "scheduler_queue_size": 16,
                "prompt_count": 32,
                "max_new_tokens": 128,
            },
        )


def test_active_widths_reject_width_above_speculative_window() -> None:
    with pytest.raises(ValueError, match="between zero"):
        canonical_active_batch_widths("8:9", 8)
