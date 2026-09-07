from __future__ import annotations

from types import SimpleNamespace

import pytest

from ncp_olmo_eval import lm_eval_runtime


def test_missing_lm_eval_has_actionable_install_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> str:
        raise lm_eval_runtime.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(lm_eval_runtime.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r"ncp-olmo-eval\[vllm\]"):
        lm_eval_runtime.task_manager_class()


def test_version_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm_eval_runtime.importlib.metadata, "version", lambda _: "0.4.9")
    with pytest.raises(RuntimeError, match="expected='0.4.13'"):
        lm_eval_runtime.task_manager_class()


def test_pinned_task_manager_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    task_manager = type("TaskManager", (), {})
    monkeypatch.setattr(
        lm_eval_runtime.importlib.metadata,
        "version",
        lambda _: lm_eval_runtime.LM_EVAL_VERSION,
    )
    monkeypatch.setattr(
        lm_eval_runtime.importlib,
        "import_module",
        lambda _: SimpleNamespace(TaskManager=task_manager),
    )

    assert lm_eval_runtime.task_manager_class() is task_manager
    assert lm_eval_runtime.runtime_identity()["commit"] == lm_eval_runtime.LM_EVAL_COMMIT
    assert lm_eval_runtime.LM_EVAL_REQUIREMENT == "lm-eval==0.4.13"
