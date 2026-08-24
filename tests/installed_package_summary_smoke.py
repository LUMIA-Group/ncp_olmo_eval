"""Exercise Core88 summary identity validation from an installed wheel."""

from __future__ import annotations

import json
from pathlib import Path

from ncp_olmo_eval.core88_workflow import validate_scoring_evaluator_repo
from ncp_olmo_eval.core_native_summary import (
    _reported_evaluator_state,
    _validate_reported_evaluator,
)
from ncp_olmo_eval.source_identity import evaluator_state


def main() -> None:
    state = evaluator_state(Path("/nonexistent-workflow-checkout"))
    if state["evaluator_source_kind"] != "installed-package":
        raise RuntimeError(f"expected installed-package identity, got {state!r}")
    workflow = {
        "repo_root": "/sealed/workflow/checkout",
        "repo_commit": state["evaluator_repo_commit"],
        "source_identity": {
            "kind": "installed-package",
            "revision": state["evaluator_repo_commit"],
            "tree_sha256": state["evaluator_source_tree_sha256"],
        },
    }
    proof = validate_scoring_evaluator_repo(
        workflow, _reported_evaluator_state(state)
    )
    if proof["status"] != "CORE88_EVALUATOR_EXACT_WORKFLOW_COMMIT":
        raise RuntimeError(f"installed-package summary proof failed: {proof!r}")
    if proof["workflow_repo_root"] == proof["evaluator_repo_root"]:
        raise RuntimeError("smoke did not exercise different absolute paths")
    cached_report = {
        "evaluator_repo_root": "/sealed/scoring/checkout",
        "evaluator_repo_commit": state["evaluator_repo_commit"],
        "evaluator_repo_dirty": False,
        "evaluator_source_kind": "installed-package",
        "evaluator_source_tree_sha256": state["evaluator_source_tree_sha256"],
        "evaluator_package_version": state["evaluator_package_version"],
    }
    cached_proof = _validate_reported_evaluator(workflow, cached_report)
    if cached_proof["status"] != "CORE88_EVALUATOR_EXACT_WORKFLOW_COMMIT":
        raise RuntimeError(f"cached-report identity proof failed: {cached_proof!r}")
    if cached_proof["workflow_repo_root"] == cached_proof["evaluator_repo_root"]:
        raise RuntimeError("cached-report smoke did not exercise different absolute paths")
    print(json.dumps({"summary": proof, "cached_report": cached_proof}, sort_keys=True))


if __name__ == "__main__":
    main()
