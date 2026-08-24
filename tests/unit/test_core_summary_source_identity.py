from __future__ import annotations

from ncp_olmo_eval.core88_workflow import validate_scoring_evaluator_repo
from ncp_olmo_eval.core_native_summary import _reported_evaluator_state


def test_summary_preserves_matching_installed_package_tree_across_paths(tmp_path) -> None:
    revision = "a" * 40
    tree_sha256 = "b" * 64
    workflow = {
        "repo_root": str(tmp_path / "workflow-checkout"),
        "repo_commit": revision,
        "source_identity": {
            "kind": "installed-package",
            "revision": revision,
            "tree_sha256": tree_sha256,
        },
    }
    report = {
        "evaluator_repo_root": str(
            tmp_path / "runtime" / "site-packages" / "ncp_olmo_eval"
        ),
        "evaluator_repo_commit": revision,
        "evaluator_repo_dirty": False,
        "evaluator_source_kind": "installed-package",
        "evaluator_source_tree_sha256": tree_sha256,
        "evaluator_package_version": "0.1.0a9",
    }

    state = _reported_evaluator_state(report)
    proof = validate_scoring_evaluator_repo(workflow, state)

    assert state == report
    assert proof["status"] == "CORE88_EVALUATOR_EXACT_WORKFLOW_COMMIT"
    assert proof["source_tree_sha256"] == tree_sha256
    assert proof["workflow_repo_root"] != proof["evaluator_repo_root"]
