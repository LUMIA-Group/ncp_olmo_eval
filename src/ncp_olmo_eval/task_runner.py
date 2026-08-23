"""Execute scheduler-neutral task specs inside an existing allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

from .task_spec import read_status, read_task, write_status


def _task_environment(spec_env: dict[str, str]) -> dict[str, str]:
    """Merge a task environment without hiding image-local scorer modules."""

    env = os.environ.copy()
    env.update(spec_env)
    scorer_dependencies = env.get("CORE88_OLMO_EVAL_DEPS", "").strip()
    if not scorer_dependencies:
        return env
    dependency_root = Path(scorer_dependencies)
    if not dependency_root.is_absolute() or not dependency_root.is_dir():
        raise RuntimeError(
            "CORE88_OLMO_EVAL_DEPS must name an existing absolute directory: "
            f"{scorer_dependencies!r}"
        )
    python_path = [value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value]
    if scorer_dependencies not in python_path:
        python_path.insert(0, scorer_dependencies)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def _resolved_argv(argv: Sequence[str], env: dict[str, str]) -> list[str]:
    resolved = list(argv)
    replacement = env.get("NCP_OLMO_TASK_PYTHON", "").strip()
    if not replacement:
        return resolved
    if not resolved:
        raise RuntimeError("task argv is empty")
    executable = Path(resolved[0]).name
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable) is None:
        raise RuntimeError(
            "NCP_OLMO_TASK_PYTHON may only replace a Python task executable, "
            f"got {resolved[0]!r}"
        )
    selected = Path(replacement)
    if not selected.is_absolute() or not selected.is_file() or not os.access(selected, os.X_OK):
        raise RuntimeError(f"NCP_OLMO_TASK_PYTHON is not an executable absolute path: {selected}")
    resolved[0] = str(selected)
    return resolved


def run_task(spec_path: Path) -> dict[str, object]:
    spec = read_task(spec_path.resolve())
    cwd = Path(spec.cwd)
    if not cwd.is_dir():
        raise RuntimeError(f"task cwd does not exist: {cwd}")
    Path(spec.output_root).mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _task_environment(spec.env)
    argv = _resolved_argv(spec.argv, env)
    write_status(Path(spec.status_path), spec=spec, state="Running")
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"task_id={spec.task_id}\n"
                f"argv={json.dumps(spec.argv)}\n"
                f"resolved_argv={json.dumps(argv)}\n"
            )
            log.flush()
            result = subprocess.run(
                argv, cwd=spec.cwd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
    except BaseException as error:
        write_status(Path(spec.status_path), spec=spec, state="Failed", detail=repr(error))
        raise
    state = "Succeeded" if result.returncode == 0 else "Failed"
    write_status(Path(spec.status_path), spec=spec, state=state, returncode=result.returncode)
    return read_status(Path(spec.status_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = run_task(args.spec)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["state"] != "Succeeded":
        raise SystemExit(int(result.get("returncode", 1)))


if __name__ == "__main__":
    main()
