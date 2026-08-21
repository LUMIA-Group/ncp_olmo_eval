"""Execute scheduler-neutral task specs inside an existing allocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from .task_spec import read_status, read_task, write_status


def run_task(spec_path: Path) -> dict[str, object]:
    spec = read_task(spec_path.resolve())
    cwd = Path(spec.cwd)
    if not cwd.is_dir():
        raise RuntimeError(f"task cwd does not exist: {cwd}")
    Path(spec.output_root).mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(spec.env)
    write_status(Path(spec.status_path), spec=spec, state="Running")
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"task_id={spec.task_id}\nargv={json.dumps(spec.argv)}\n")
            log.flush()
            result = subprocess.run(
                list(spec.argv), cwd=spec.cwd, env=env, stdout=log, stderr=subprocess.STDOUT
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
