from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ncp_olmo_eval import core_native_code_eval as code_eval


def _write_pinned_sources(root: Path) -> dict[str, tuple[Path, str]]:
    sanitize = root / "src/olmo_eval/evals/extract/sanitize.py"
    bigcodebench = root / "src/olmo_eval/evals/tasks/bigcodebench.py"
    tasks_init = root / "src/olmo_eval/evals/tasks/__init__.py"
    sanitize.parent.mkdir(parents=True)
    bigcodebench.parent.mkdir(parents=True)
    tasks_init.write_text("raise AssertionError('task registry must not be imported')\n", encoding="utf-8")
    sanitize.write_text(
        "def sanitize_code(code: str, entrypoint: str | None = None) -> str:\n"
        "    return code.strip()\n",
        encoding="utf-8",
    )
    bigcodebench.write_text(
        "raise AssertionError('module top level must not execute')\n\n"
        "def _build_bcb_execution_script(solution: str, test_code: str) -> str:\n"
        "    return solution + '\\n' + test_code\n",
        encoding="utf-8",
    )
    return {
        "olmo_eval.evals.extract.sanitize": (
            sanitize.relative_to(root),
            hashlib.sha256(sanitize.read_bytes()).hexdigest(),
        ),
        "olmo_eval.evals.tasks.bigcodebench": (
            bigcodebench.relative_to(root),
            hashlib.sha256(bigcodebench.read_bytes()).hexdigest(),
        ),
    }


def test_bigcodebench_helpers_do_not_import_task_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_pinned_sources(tmp_path)
    monkeypatch.setattr(code_eval, "OFFICIAL_OLMO_EVAL_SOURCE", sources)
    monkeypatch.setenv("OLMO_EVAL_COMMIT", code_eval.OFFICIAL_OLMO_EVAL_COMMIT)
    monkeypatch.setenv("OLMO_EVAL_ROOT", str(tmp_path))
    monkeypatch.setattr(
        code_eval.importlib.util,
        "find_spec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("module discovery must not be used")
        ),
    )

    sanitize_code, build_script = code_eval._load_official_bigcodebench_helpers()

    assert sanitize_code("  def solve(): return 1  ", entrypoint="solve") == (
        "def solve(): return 1"
    )
    assert build_script("solution", "tests") == "solution\ntests"
    assert code_eval._verify_official_olmo_eval_source() == {
        name: expected_sha for name, (_path, expected_sha) in sources.items()
    }


def test_bigcodebench_source_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_pinned_sources(tmp_path)
    monkeypatch.setattr(code_eval, "OFFICIAL_OLMO_EVAL_SOURCE", sources)
    monkeypatch.setenv("OLMO_EVAL_COMMIT", code_eval.OFFICIAL_OLMO_EVAL_COMMIT)
    monkeypatch.setenv("OLMO_EVAL_ROOT", str(tmp_path))
    path = tmp_path / sources["olmo_eval.evals.extract.sanitize"][0]
    path.write_text("tampered = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source SHA256 mismatch"):
        code_eval._load_official_bigcodebench_helpers()
