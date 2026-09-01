from __future__ import annotations

from pathlib import Path

import pytest

from ncp_olmo_eval import core88_workflow
from ncp_olmo_eval.source_identity import source_state


def _speculative_point() -> dict[str, object]:
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
        "active_batch_widths": "1:8,2:8,4:4,8:2",
        "dynamic_runtime_block_size": True,
        "runtime_layer_count": 5,
        "runtime_local_mixer": "full",
        "mixer_compile_mode": "default",
        "chunk_size": 4,
        "target_layers": "1,4,7,10,13",
        "telemetry_flush_interval": 8,
    }


def test_core88_gsm8k_contract_is_path_portable_and_hash_pinned() -> None:
    assert not core88_workflow.CANONICAL_STANDARD_INPUT_CONFIG.is_absolute()
    assert not core88_workflow.CANONICAL_DATASET_TEST_FILE.is_absolute()
    assert core88_workflow.CANONICAL_STANDARD_INPUT_CONFIG_SHA256 == (
        "295395763cbe551cbe41481c1a9ad16b491768c8d50911bd6ab1cd4f69e2b265"
    )
    assert core88_workflow.CANONICAL_DATASET_TEST_SHA256 == (
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
    )
    assert core88_workflow.GSM8K_SEED == 42


def test_scoring_descendant_paths_match_standalone_layout() -> None:
    paths = core88_workflow._SCORING_DESCENDANT_RELATIVE_PATHS
    assert "src/ncp_olmo_eval/core88_workflow.py" in paths
    assert "src/ncp_olmo_eval/core_native_aggregate.py" in paths
    assert all(not path.startswith("experiments/") for path in paths)
    assert all(not Path(path).is_absolute() for path in paths)


def test_core88_workflow_accepts_installed_package_source_identity(
    tmp_path: Path, monkeypatch
) -> None:
    revision = "b" * 40
    monkeypatch.setenv("NCP_OLMO_SOURCE_REVISION", revision)
    state = source_state(tmp_path / "no-git-checkout")
    workflow = core88_workflow.create_workflow(
        output_root=tmp_path / "workflow",
        repo_root=tmp_path / "no-git-checkout",
        repo_commit=revision,
        source_kind="installed-package",
        source_tree_sha256=state["source_tree_sha256"],
        hf_model_path=tmp_path / "model",
        model_identity_path=tmp_path / "model",
        model_label="fixture",
        run_tag="portable",
        core_job_names=[f"core-{index}" for index in range(4)],
        gsm8k_job_name="gsm8k",
        global_seed=42,
        diagnostic_planless=True,
    )
    assert workflow["source_identity"] == {
        "kind": "installed-package",
        "revision": revision,
        "tree_sha256": state["source_tree_sha256"],
    }
    core88_workflow._validate_repo_state(workflow)


def test_core88_workflow_seals_one_dflash_operating_point(tmp_path: Path) -> None:
    point = _speculative_point()
    kwargs = {
        "output_root": tmp_path / "workflow",
        "repo_root": tmp_path / "repo",
        "repo_commit": "c" * 40,
        "source_kind": "installed-package",
        "source_tree_sha256": "d" * 64,
        "hf_model_path": tmp_path / "model",
        "model_identity_path": tmp_path / "model",
        "model_label": "fixture",
        "run_tag": "speculative",
        "core_job_names": [f"core-{index}" for index in range(4)],
        "gsm8k_job_name": "gsm8k",
        "global_seed": 42,
        "hf_backend": "native_vllm",
        "score_batch_size": 8,
        "generation_batch_size": 8,
        "row_chunk_size": 8,
        "vllm_max_model_len": 8192,
        "vllm_scheduler_queue_size": 32,
        "vllm_speculative_operating_point": point,
        "gsm8k_batch_size": 8,
        "gsm8k_scheduler_queue_size": 32,
        "gsm8k_max_model_len": 8192,
        "diagnostic_planless": True,
    }
    workflow = core88_workflow.create_workflow(**kwargs)

    assert workflow["core_protocol"]["vllm_speculative_operating_point"] == point
    assert workflow["gsm8k_protocol"]["vllm_speculative_operating_point"] == point

    mismatched = {**kwargs, "output_root": tmp_path / "invalid", "gsm8k_batch_size": 1}
    with pytest.raises(ValueError, match="GSM8K batch differs"):
        core88_workflow.create_workflow(**mismatched)
