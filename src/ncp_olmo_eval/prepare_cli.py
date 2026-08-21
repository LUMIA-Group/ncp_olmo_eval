"""Unified preparation entry point for sealed RULER and HELMET inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("ruler", "helmet"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    if parsed.benchmark == "ruler":
        command = [sys.executable, "-m", "ncp_olmo_eval.long_context_prepare", "ruler"]
    else:
        command = [sys.executable, "-m", "ncp_olmo_eval.helmet_prepare"]
    command.extend(parsed.args)
    return subprocess.run(command, check=False).returncode


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
