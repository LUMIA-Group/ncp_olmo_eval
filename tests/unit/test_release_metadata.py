from __future__ import annotations

import ast
import re
import tomllib
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
    assert version == "0.1.0a14"
    assert _fallback_version(root / "src/ncp_olmo_eval/__init__.py") == version

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    first_release = re.search(r"^## ([^\n]+)$", changelog, flags=re.MULTILINE)
    assert first_release is not None
    assert first_release.group(1) == version

    validation = (root / "docs/validation/dflash_a14_synthetic.json").read_text(
        encoding="utf-8"
    )
    assert f'"release": "{version}"' in validation
