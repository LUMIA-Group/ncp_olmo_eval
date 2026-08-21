from __future__ import annotations

from pathlib import Path

from ncp_olmo_eval import core88_workflow
from ncp_olmo_eval.source_identity import source_state


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
