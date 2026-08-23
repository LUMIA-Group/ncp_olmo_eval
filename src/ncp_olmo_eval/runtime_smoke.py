"""CPU-only runtime preflights used by release Docker CI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from typing import Sequence

from .core_native_answer_eval import require_official_math_runtime, score_math_answer
from .core_native_code_eval import (
    OFFICIAL_OLMO_EVAL_COMMIT,
    _load_official_bigcodebench_helpers,
    _verify_official_olmo_eval_source,
)


def smoke_math() -> dict[str, object]:
    runtime = require_official_math_runtime()
    expected = {
        "sympy": "1.14.0",
        "antlr4-python3-runtime": "4.11.0",
    }
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"math scorer dependency drift: {observed} != {expected}")
    scored = score_math_answer(
        r"The final answer is $\boxed{\frac{1}{2}}$.",
        r"$\boxed{0.5}$",
        require_official_runtime=True,
    )
    if not scored["primary_correct"]:
        raise RuntimeError(f"official Minerva/MATH scorer smoke failed: {scored}")
    return {
        "status": "MATH_SCORER_RUNTIME_OK",
        "runtime": runtime,
        "score_status": scored["score_status"],
    }


def smoke_bigcodebench() -> dict[str, object]:
    if "olmo_eval.evals.tasks" in sys.modules:
        raise RuntimeError("OLMo-Eval task registry was imported before the smoke")
    source_fidelity = _verify_official_olmo_eval_source()
    sanitize_code, build_script = _load_official_bigcodebench_helpers()
    sanitized = sanitize_code("def solve():\n    return 1\n", entrypoint="solve")
    program = build_script(sanitized, "class TestCases: pass")
    if "def solve" not in sanitized or "class TestCases" not in program:
        raise RuntimeError("BigCodeBench helper smoke produced invalid source")
    imported_registry = "olmo_eval.evals.tasks" in sys.modules
    if imported_registry:
        raise RuntimeError("BigCodeBench helper imported the OLMo-Eval task registry")
    return {
        "status": "BIGCODEBENCH_SCORER_PREFLIGHT_OK",
        "olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "source_fidelity": source_fidelity,
        "task_registry_imported": imported_registry,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", choices=("math", "bigcodebench", "all"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    results: list[dict[str, object]] = []
    if args.contract in {"math", "all"}:
        results.append(smoke_math())
    if args.contract in {"bigcodebench", "all"}:
        results.append(smoke_bigcodebench())
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
