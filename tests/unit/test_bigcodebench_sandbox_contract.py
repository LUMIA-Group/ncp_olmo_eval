from __future__ import annotations

from pathlib import Path


def test_formal_bigcodebench_image_exposes_locked_parser_wheels() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "docker/core88_bigcodebench_sandbox.Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "CORE88_OLMO_EVAL_DEPS=/opt/core88/olmo-eval-deps" in dockerfile
    assert "PYTHONPATH=/opt/core88/olmo-eval-deps" in dockerfile
    assert "import tree_sitter, tree_sitter_python" in dockerfile
    assert dockerfile.index("PYTHONPATH=/opt/core88/olmo-eval-deps") < dockerfile.index(
        "import tree_sitter, tree_sitter_python"
    )
    assert "COPY ${TREE_SITTER_WHEEL} /tmp/core88-wheels/tree_sitter.whl" in dockerfile
    assert (
        "COPY ${TREE_SITTER_PYTHON_WHEEL} /tmp/core88-wheels/tree_sitter_python.whl"
        in dockerfile
    )


def test_ci_executes_the_formal_bigcodebench_image() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "docker/core88_bigcodebench_sandbox.Dockerfile" in workflow
    assert "docker/ci_scorer_smoke.Dockerfile" not in workflow
