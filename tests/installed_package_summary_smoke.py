"""Exercise Core88 summary identity validation from an installed wheel."""

from __future__ import annotations

import json
from pathlib import Path

from ncp_olmo_eval.core88_workflow import validate_scoring_evaluator_repo
from ncp_olmo_eval.core_native_summary import _reported_evaluator_state
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
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
