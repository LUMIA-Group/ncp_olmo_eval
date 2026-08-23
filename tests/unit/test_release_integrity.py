from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from ncp_olmo_eval import device_layout
from ncp_olmo_eval.lmdeploy_inference import LMDEPLOY_BACKEND, validate_lmdeploy_args


def _published_modules(package_root: Path) -> set[str]:
    modules = {"ncp_olmo_eval"}
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.add(".".join(("ncp_olmo_eval", *parts)).rstrip("."))
    return modules


def test_every_internal_relative_import_is_published() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "ncp_olmo_eval"
    published = _published_modules(package_root)
    missing: list[str] = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        parts = list(relative.with_suffix("").parts)
        module_name = ".".join(("ncp_olmo_eval", *parts))
        package_name = (
            module_name.removesuffix(".__init__")
            if parts[-1] == "__init__"
            else module_name.rpartition(".")[0]
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                target = importlib.util.resolve_name(relative_name, package_name)
                targets = (
                    [target]
                    if node.module
                    else [f"{target}.{alias.name}" for alias in node.names]
                )
            elif node.module and node.module.startswith("ncp_olmo_eval"):
                targets = [node.module]
            else:
                continue
            missing.extend(
                f"{path.relative_to(package_root)}:{node.lineno}:{target}"
                for target in targets
                if target not in published
            )
    assert not missing, "missing internal modules:\n" + "\n".join(sorted(missing))


def test_published_commands_do_not_reference_internal_lmdeploy_flags() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "ncp_olmo_eval"
    leaked = [
        str(path.relative_to(package_root))
        for path in package_root.rglob("*.py")
        if "--no-allow-unverified-lmdeploy" in path.read_text(encoding="utf-8")
    ]
    assert not leaked, "internal LMDeploy flag leaked into published commands: " + ", ".join(leaked)


def test_omitted_lmdeploy_validation_is_backend_gated() -> None:
    validate_lmdeploy_args(SimpleNamespace(hf_backend="native_vllm"), batch_size=8)
    with pytest.raises(RuntimeError, match="LMDeploy is not included"):
        validate_lmdeploy_args(SimpleNamespace(hf_backend=LMDEPLOY_BACKEND), batch_size=8)


def test_device_layout_maps_independent_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_layout, "_visible_cuda_device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CONCEPTLM_PROCESSES_PER_GPU", "2")
    assert device_layout.local_cuda_device_index() == 1


def test_device_layout_rejects_inconsistent_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_layout, "_visible_cuda_device_count", lambda: 4)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("CONCEPTLM_PROCESSES_PER_GPU", "2")
    with pytest.raises(RuntimeError, match="expected=8"):
        device_layout.local_cuda_device_index()


def test_device_layout_rejects_missing_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_layout, "_visible_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="at least one visible CUDA device"):
        device_layout.local_cuda_device_index()
