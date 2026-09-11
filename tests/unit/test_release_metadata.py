from __future__ import annotations

import ast
import re
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib
from pathlib import Path


def _fallback_version(init_path: Path) -> str:
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "__version__"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(values) == 1
    return values[0]


def test_release_version_is_consistent_across_metadata() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    assert version == "0.1.1"
    assert project["project"]["urls"] == {
        "Repository": "https://github.com/LUMIA-Group/ncp_olmo_eval",
        "Issues": "https://github.com/LUMIA-Group/ncp_olmo_eval/issues",
        "Technical Report": "https://arxiv.org/abs/2609.10715",
        "Hugging Face Papers": "https://huggingface.co/papers/2609.10715",
    }
    assert _fallback_version(root / "src/ncp_olmo_eval/__init__.py") == version

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    releases = re.findall(r"^## ([^\n]+)$", changelog, flags=re.MULTILINE)
    assert releases
    released_versions = [release for release in releases if release != "Unreleased"]
    assert released_versions
    assert released_versions[0] == version

    validation = (root / "docs/validation/dflash_a14_synthetic.json").read_text(
        encoding="utf-8"
    )
    assert '"release": "0.1.0a14"' in validation
