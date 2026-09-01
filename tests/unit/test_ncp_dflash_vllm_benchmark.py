from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ncp_olmo_eval import ncp_dflash_vllm_benchmark as benchmark
from ncp_olmo_eval.inference import SamplingParams


def _result(
    token_ids: list[int], *, speculative: bool, verification_mode: str = "sequential_exact"
) -> dict:
    runtime = {
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "scheduler_queue_size": 1,
        "continuous_batching_enabled": False,
        "tensor_parallel_size": 1,
        "execution_mode": "eager",
        "gpu_memory_utilization": 0.8,
        "attention_backend": "FLASH_ATTN",
        "flash_attn_version": 3,
        "hlm_attention_impl": "legacy_mixed",
        "vllm_use_v2_model_runner": "0",
        "speculative_decoding": speculative,
    }
    if speculative:
        runtime.update(
            {
                "speculative_num_tokens": 8,
                "speculative_verification_mode": {
                    "transactional_exact": "segmented_kv_approx",
                    "chunk_parallel": "intra_chunk_exact",
                }.get(verification_mode, verification_mode),
                "speculative_draft_attention_backend": "flash_varlen",
                "speculative_context_kv_cache": True,
                "speculative_sparse_context_projection": True,
                "speculative_min_eligible_batch": 1,
                "speculative_min_proposal_tokens_per_row": 1,
                "speculative_min_proposal_tokens_per_batch": 1,
                "speculative_runtime_block_size": 0,
                "speculative_active_batch_widths": "1:8,2:8,4:4,8:2",
                "speculative_dynamic_runtime_block_size": True,
                "speculative_runtime_layer_count": 5,
                "speculative_runtime_local_mixer": "full",
                "speculative_mixer_compile_mode": "default",
                "speculative_chunk_size": 4,
                "speculative_target_layers": "1,4,7,10,13",
                "speculative_telemetry_flush_interval": 8,
            }
        )
    return {
        "seed": 42,
        "prompt_count": 1,
        "prompt_offset": 0,
        "prompt_sha256": "prompt",
        "max_new_tokens": len(token_ids),
        "batch_size": 1,
        "scheduler_queue_size": 1,
        "ignore_eos": True,
        "gpu_memory_utilization": 0.8,
        "execution_mode": "eager",
        "vllm_use_v2_model_runner": "0",
        "runtime": runtime,
        "output_token_count": len(token_ids),
        "vllm_version": "0.13.0",
        "speculative_verification_mode": (verification_mode if speculative else "not_applicable"),
        "output_tokens_per_second": 2.0 if speculative else 1.0,
        "request_latency_seconds": {"mean": 0.5 if speculative else 1.0},
        "peak_gpu_memory_gib": 2.0 if speculative else 1.0,
        "target_model_identity": {"source_model": "/target"},
        "draft_model_identity": (
            {"path": "/draft", "config_sha256": "config"} if speculative else None
        ),
        "telemetry": {"proposed_tokens": 1} if speculative else {},
        "completions": [{"prompt_index": 0, "token_ids": token_ids, "finish_reason": "length"}],
    }


def test_sampling_params_supports_forced_token_verification() -> None:
    sampling = SamplingParams(max_tokens=128, ignore_eos=True, seed=42)

    assert sampling.ignore_eos is True
    assert sampling.max_tokens == 128


def test_transaction_telemetry_reports_observed_acceptance_lower_bound(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"event": "proposal_batch", "proposed_tokens": 4},
                {
                    "event": "target_state_transaction_commit",
                    "verifier_tokens": 5,
                    "committed_tokens": 3,
                    "rolled_back_tokens": 2,
                },
                # This final proposal was never scheduled before completion.
                {"event": "proposal_batch", "proposed_tokens": 4},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = benchmark._telemetry_summary(telemetry)

    assert summary["proposed_tokens"] == 8
    assert summary["accepted_tokens_observed"] == 2
    assert summary["rejected_tokens"] == 6
    assert summary["acceptance_rate_lower_bound"] == pytest.approx(0.25)


def test_compare_persists_exact_mismatch_before_failing(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    speculative_path = tmp_path / "speculative.json"
    output_path = tmp_path / "comparison.json"
    target_path.write_text(json.dumps(_result([1, 2, 3], speculative=False)))
    speculative_path.write_text(json.dumps(_result([1, 4, 3], speculative=True)))

    with pytest.raises(AssertionError, match="see"):
        benchmark.compare(
            argparse.Namespace(
                target_result=target_path, speculative_result=speculative_path, output=output_path
            )
        )

    comparison = json.loads(output_path.read_text())
    assert comparison["status"] == "NCP_DFLASH_VLLM_EXACT_MATCH_FAILED"
    assert comparison["exact_token_match_count"] == 0
    assert comparison["benchmark_contract"]["ignore_eos"] is True
    assert comparison["speculative_operating_point"]["speculative_num_tokens"] == 8
    assert (
        comparison["speculative_operating_point"]["active_batch_widths"]
        == "1:8,2:8,4:4,8:2"
    )
    assert comparison["first_mismatch"] == {
        "prompt_index": 0,
        "generated_token_index": 1,
        "target_token_ids": [2, 3],
        "speculative_token_ids": [4, 3],
        "target_finish_reason": "length",
        "speculative_finish_reason": "length",
    }


def test_compare_accepts_only_explicit_approximate_divergence(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    speculative_path = tmp_path / "speculative.json"
    output_path = tmp_path / "comparison.json"
    target_path.write_text(json.dumps(_result([1, 2, 3], speculative=False)))
    speculative_path.write_text(
        json.dumps(_result([1, 4, 3], speculative=True, verification_mode="transactional_exact"))
    )

    comparison = benchmark.compare(
        argparse.Namespace(
            target_result=target_path,
            speculative_result=speculative_path,
            output=output_path,
            allow_output_divergence=True,
        )
    )

    assert comparison["status"] == "NCP_DFLASH_VLLM_APPROXIMATE_AB_OK"
    assert comparison["speculative_verification_mode"] == "segmented_kv_approx"
    assert comparison["speculative_output_contract"] == "approximate"
    assert comparison["downstream_score_required"] is True
    assert comparison["exact_prompt_match_rate"] == 0.0


def test_compare_rejects_divergence_opt_in_for_exact_mode(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    speculative_path = tmp_path / "speculative.json"
    output_path = tmp_path / "comparison.json"
    target_path.write_text(json.dumps(_result([1, 2, 3], speculative=False)))
    speculative_path.write_text(json.dumps(_result([1, 2, 3], speculative=True)))

    with pytest.raises(ValueError, match="segmented_kv_approx"):
        benchmark.compare(
            argparse.Namespace(
                target_result=target_path,
                speculative_result=speculative_path,
                output=output_path,
                allow_output_divergence=True,
            )
        )
