from __future__ import annotations

import ast
import importlib.util
import json
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib
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


def test_release_pins_official_math_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    expected = {"sympy==1.14.0", "antlr4-python3-runtime==4.11"}
    assert expected <= set(optional["scoring"])
    assert expected <= set(optional["vllm"])
    assert expected <= set(optional["gsm8k"])
    dockerfile = (root / "docker/runtime.Dockerfile").read_text(encoding="utf-8")
    assert "INSTALL_EXTRAS=vllm,helmet,scoring" in dockerfile
    assert "ncp_olmo_eval.runtime_smoke math" in dockerfile


def test_public_package_metadata_has_no_direct_url_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project["dependencies"])
    requirements.extend(
        requirement
        for extra in project["optional-dependencies"].values()
        for requirement in extra
    )
    assert not [requirement for requirement in requirements if " @ " in requirement]
    assert "lm-eval==0.4.13" in project["optional-dependencies"]["vllm"]
    assert "lm-eval==0.4.13" in project["optional-dependencies"]["gsm8k"]


def test_release_pins_the_validated_vllm_dependency_pair() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["dependencies"])
    assert "transformers==4.57.6" in dependencies
    assert "huggingface-hub==0.36.2" in dependencies


def test_release_metadata_declares_apache_and_packages_notices() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) == {
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
    }
    license_lines = (root / "LICENSE").read_text(encoding="utf-8").splitlines()
    assert [line.strip() for line in license_lines[:2]] == [
        "Apache License",
        "Version 2.0, January 2004",
    ]
    assert "NCP-ArchPreview contributors" in (root / "NOTICE").read_text(encoding="utf-8")


def test_public_assets_and_image_bases_are_exact_and_placeholder_free() -> None:
    root = Path(__file__).resolve().parents[2]
    asset_payload = json.loads((root / "configs/assets.example.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in asset_payload["assets"]} == {
        "helmet-autoais",
        "ruler-data",
        "helmet-classic-data",
    }
    assert all(len(item["revision"]) == 40 for item in asset_payload["assets"])

    images = json.loads(
        (root / "configs/public-image-bases.json").read_text(encoding="utf-8")
    )
    assert images["release_version"] == "0.1.0"
    assert len(images["upstream_bases"]) == 5
    assert all(
        len(value.rsplit("@sha256:", 1)[-1]) == 64
        for value in images["upstream_bases"].values()
    )
    assert all(
        value.startswith("ghcr.io/luckysjtu/ncp-olmo-eval-")
        and value.endswith(":0.1.0")
        for value in images["release_tags"].values()
    )

    public_paths = [
        root / "README.md",
        root / "pyproject.toml",
        root / "configs/runtime.env.example",
        root / "configs/apptainer-images.example.json",
        root / "configs/public-image-bases.json",
        root / "docs/IMAGES.md",
        *sorted((root / "docker").glob("*.Dockerfile")),
    ]
    forbidden = ("REPLACE_ME", "your-org", "ghcr.io/ORG")
    leaked = [
        str(path.relative_to(root))
        for path in public_paths
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]
    assert not leaked, "public release placeholders remain in: " + ", ".join(leaked)


def test_release_workflows_use_oidc_and_pinned_actions() -> None:
    root = Path(__file__).resolve().parents[2]
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    images = (root / ".github/workflows/publish-images.yml").read_text(encoding="utf-8")
    assert "id-token: write" in release
    assert "environment:\n      name: pypi" in release
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in release
    assert "password:" not in release
    assert "packages: write" in images
    assert "scripts/build-public-images.sh" in images
    for workflow in (release, images):
        assert 'gh release upload "$RELEASE_TAG"' in workflow
        assert '--repo "$GITHUB_REPOSITORY"' in workflow
    for workflow in (release, images):
        action_refs = [
            line.strip().split("uses:", 1)[1].strip()
            for line in workflow.splitlines()
            if line.strip().startswith("- uses:")
        ]
        assert action_refs
        assert all("@" in ref and not ref.endswith(("@main", "@master", "@v1", "@v3")) for ref in action_refs)


def test_public_dockerfiles_do_not_require_private_archives() -> None:
    root = Path(__file__).resolve().parents[2]
    ds1000 = (root / "docker/core88_ds1000.Dockerfile").read_text(encoding="utf-8")
    assert "ds1000-runtime.tar.gz" not in ds1000
    assert "python:3.10.13-slim-bookworm@sha256:" in ds1000
    assert "tensorflow-cpu==2.16.1" in ds1000
    bigcodebench = (root / "docker/core88_bigcodebench_sandbox.Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "bigcodebench/bigcodebench-gradio@sha256:" in bigcodebench
    assert "PYTHONPATH=/opt/core88/olmo-eval-deps" in bigcodebench
    multiple = (root / "docker/core88_thin_sandbox.Dockerfile").read_text(encoding="utf-8")
    assert "'setuptools==80.9.0'" in multiple
    assert "--no-build-isolation ." in multiple


def test_cuda_image_publish_job_reclaims_hosted_runner_disk() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/publish-images.yml").read_text(encoding="utf-8")
    assert "if: matrix.image_key == 'NCP_OLMO_EVAL_IMAGE'" in workflow
    assert "/usr/local/lib/android" in workflow
    assert "docker system prune --all --force" in workflow


def test_public_prose_uses_ncp_archpreview_brand() -> None:
    root = Path(__file__).resolve().parents[2]
    public_paths = [
        root / "README.md",
        root / "CHANGELOG.md",
        root / "pyproject.toml",
        root / "src/ncp_olmo_eval/__init__.py",
        *sorted((root / "docs").glob("*.md")),
        *sorted((root / "docker").glob("*.Dockerfile")),
    ]
    legacy = [
        str(path.relative_to(root))
        for path in public_paths
        if "NCP OLMo" in path.read_text(encoding="utf-8")
        or "NCP-OLMo" in path.read_text(encoding="utf-8")
    ]
    assert not legacy, "legacy user-facing NCP OLMo brand remains in: " + ", ".join(legacy)

    readme = (root / "README.md").read_text(encoding="utf-8")
    public_models = (
        "NCP_ArchPreview_dolma3_8.9B_Stage1",
        "NCP_ArchPreview_dolma3_8.9B_Stage2_v1",
        "NCP_ArchPreview_dolma3_8.9B_Stage2_v2",
        "NCP_ArchPreview_dolma3_8.9B_Stage2_v3",
        "NCP_ArchPreview_dolma3_8.9B_Stage2_DFlash2_NCPFlash",
    )
    assert all(
        f"https://huggingface.co/ArchSpace-Collection/{model_id}" in readme
        for model_id in public_models
    )


def test_bigcodebench_does_not_import_olmo_eval_task_registry() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/ncp_olmo_eval/core_native_code_eval.py").read_text(encoding="utf-8")
    assert "from olmo_eval.evals.tasks" not in source
    assert "importlib.util.find_spec" not in source


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
