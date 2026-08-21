#!/usr/bin/env python3
"""Execute saved Core code predictions in a nested, fail-closed sandbox.

GPU inference deliberately never executes model output. This module is meant
to run in a separate CPU allocation. The parent process may read shared
storage, but every candidate is staged under /tmp and executed by bubblewrap in fresh user,
mount, PID, IPC, UTS, cgroup, and network namespaces.  The candidate sees only
read-only runtime files plus its disposable /work directory; shared storage is
not mounted in the nested sandbox.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

PYTHON_CODE_TASKS = {79, 81, 82, 83, 84, 85}
MULTIPLE_TASKS = {86, 87}
DEFAULT_CODE_TASKS = PYTHON_CODE_TASKS | MULTIPLE_TASKS
VERIFIED_SCORER_TASKS = {79, 82, 83, 85}
RESULT_SCHEMA_VERSION = "core-native-code-result-v1"
SUMMARY_SCHEMA_VERSION = "core-native-code-sandbox-v2"
OFFICIAL_OLMO_EVAL_COMMIT = "f8816eea36563f27b4a9dd2533d68d34f3c67d3f"
OFFICIAL_OLMO_EVAL_SOURCE_SHA256 = {
    "olmo_eval.evals.extract.sanitize": (
        "ca2b52216ac8b6ef27af87d55b3d8c5b8a6a2b85b65b9e3b8227e176d2cf3039"
    ),
    "olmo_eval.evals.tasks.bigcodebench": (
        "4bc9a1c9bdb2f710cca49e5d254a9474c3d1614f36c27d8df100feae0d0be910"
    ),
}
DS1000_PYTHON_VERSION = "3.10.13"
DS1000_DISTRIBUTIONS = (
    ("numpy", "1.26.4"),
    ("pandas", "1.5.3"),
    ("matplotlib", "3.8.4"),
    ("scipy", "1.12.0"),
    ("scikit-learn", "1.4.0"),
    ("seaborn", "0.13.2"),
    ("statsmodels", "0.14.1"),
    ("xgboost", "2.0.3"),
    ("gensim", "4.3.2"),
    ("torch", "2.2.0+cpu"),
    ("tensorflow-cpu", "2.16.1"),
)
DS1000_IMPORT_MODULES = (
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "seaborn",
    "statsmodels",
    "xgboost",
    "gensim",
    "torch",
    "tensorflow",
)
MULTIPLE_CORE88_TASK_SHA256 = {
    86: "43f8ed93f35569a25a4745d1104b304c565e22640e3b1654c84e583d33f83122",
    87: "8e25a99cef51f1962b1cd92363ee77fdaa76b78e6b233a142e3d5f2b3f5d3823",
}
MULTIPLE_CORE88_LANGUAGES = frozenset({"cpp", "cs", "go_test.go", "java", "js", "php"})
JAVATUPLES_1_2_SHA256 = "2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a"
MULTIPLE_LANGUAGE_CONFIG = {
    "cpp": {
        "filename": "code.cpp",
        "command": ("timeout 15 g++ -std=c++17 -o /work/a.out code.cpp && timeout 15 /work/a.out"),
        "timeout": 35.0,
        "required_commands": ("timeout", "g++"),
        "command_probes": (("g++", ("--version",)),),
        "required_paths": (),
        "required_file_sha256": (),
        "runtime_name": "multiple-cpp",
    },
    "cs": {
        "filename": "code.cs",
        "command": (
            "timeout 15 csc /d:DEBUG -r:System.Numerics.dll code.cs "
            "/out:/work/code.exe && "
            "{ set +e; timeout 5 env MONO_TRACE_LISTENER=Console.Error "
            "mono /work/code.exe 2>/work/mono.stderr; rc=$?; "
            "cat /work/mono.stderr >&2; "
            "if grep -Eq 'System.Diagnostics.DefaultTraceListener.Fail|"
            "Unhandled Exception' /work/mono.stderr; then exit 1; fi; "
            "exit \"$rc\"; }"
        ),
        "timeout": 30.0,
        "required_commands": ("timeout", "csc", "mono", "grep", "env"),
        "command_probes": (("csc", ("-version",)), ("mono", ("--version",))),
        "required_paths": (),
        "required_file_sha256": (),
        "runtime_name": "multiple-cs",
    },
    "go_test.go": {
        "filename": "code_test.go",
        "command": "timeout 30 go test code_test.go",
        "timeout": 35.0,
        "required_commands": ("timeout", "go"),
        "command_probes": (("go", ("version",)),),
        "required_paths": (),
        "required_file_sha256": (),
        "runtime_name": "multiple-go",
    },
    "java": {
        "filename": "Problem.java",
        "command": (
            "timeout 15 javac -J-Xms32m -J-Xmx512m -J-XX:+UseSerialGC "
            "-J-XX:ActiveProcessorCount=2 -encoding UTF8 -cp "
            "'/usr/multiple/javatuples-1.2.jar' Problem.java "
            "&& timeout 15 java -ea -Xms32m -Xmx512m -XX:+UseSerialGC "
            "-XX:ActiveProcessorCount=2 -cp "
            "'/usr/multiple/javatuples-1.2.jar:.' Problem"
        ),
        "timeout": 35.0,
        "required_commands": ("timeout", "javac", "java"),
        "command_probes": (("javac", ("-version",)), ("java", ("-version",))),
        "required_paths": (),
        "required_file_sha256": (("/usr/multiple/javatuples-1.2.jar", JAVATUPLES_1_2_SHA256),),
        "runtime_name": "multiple-java",
    },
    "js": {
        "filename": "code.js",
        "command": "timeout 5 node code.js",
        "timeout": 10.0,
        "required_commands": ("timeout", "node"),
        "command_probes": (("node", ("--version",)),),
        "required_paths": (),
        "required_file_sha256": (),
        "runtime_name": "multiple-js",
    },
    "php": {
        "filename": "code.php",
        "command": "timeout 15 php code.php",
        "timeout": 20.0,
        "required_commands": ("timeout", "php"),
        "command_probes": (("php", ("--version",)),),
        "required_paths": (),
        "required_file_sha256": (),
        "runtime_name": "multiple-php",
    },
}


@dataclasses.dataclass(frozen=True)
class RuntimeRequirement:
    name: str
    python_version: str | None = None
    distributions: tuple[tuple[str, str], ...] = ()
    import_modules: tuple[str, ...] = ()
    command_probes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_paths: tuple[str, ...] = ()
    required_file_sha256: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class ExecutionSpec:
    files: dict[str, str]
    argv: tuple[str, ...]
    timeout_seconds: float
    scorer: str
    required_commands: tuple[str, ...] = ()
    runtime_requirement: RuntimeRequirement = RuntimeRequirement("core88-python")
    captured_files: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SandboxLimits:
    address_space_bytes: int
    file_size_bytes: int
    process_count: int
    open_files: int


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stable_id(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_task_orders(value: str, *, allow_experimental_scorers: bool = False) -> set[int]:
    if not value.strip():
        return set(VERIFIED_SCORER_TASKS)
    orders = {int(part.strip()) for part in value.split(",") if part.strip()}
    unknown = orders - DEFAULT_CODE_TASKS
    if unknown:
        raise ValueError(f"not a Core88 code-execution task: {sorted(unknown)}")
    experimental = orders - VERIFIED_SCORER_TASKS
    if experimental and not allow_experimental_scorers:
        raise ValueError(
            "task orders require an official dependency/toolchain image or "
            "--allow-experimental-scorers: "
            f"{sorted(experimental)}"
        )
    return orders


def _prediction_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("**/predictions/*.jsonl"))


def _load_gold(
    data_root: Path, profile: str, task_orders: set[int]
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    summary_path = data_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    section = summary.get(profile)
    if not isinstance(section, dict) or section.get("failures"):
        raise RuntimeError(f"{profile} export is missing or reports failures")
    tasks = [task for task in section.get("tasks", []) if int(task["task_order"]) in task_orders]
    if {int(task["task_order"]) for task in tasks} != task_orders:
        present = {int(task["task_order"]) for task in tasks}
        raise RuntimeError(f"missing task orders in {profile}: {sorted(task_orders - present)}")
    gold: dict[tuple[int, str], dict[str, Any]] = {}
    multiple_languages: dict[int, set[str]] = {}
    for task in tasks:
        task_order = int(task["task_order"])
        expected_multiple_sha256 = MULTIPLE_CORE88_TASK_SHA256.get(task_order)
        if expected_multiple_sha256 and str(task["sha256"]) != expected_multiple_sha256:
            raise RuntimeError(
                f"unsupported Core88 MultiPL-E snapshot for task {task_order}: "
                f"{task['sha256']} != {expected_multiple_sha256}"
            )
        path = data_root / str(task["file"])
        if _file_sha256(path) != str(task["sha256"]):
            raise RuntimeError(f"dataset SHA256 mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("execution_contract", {}).get("requires_isolated_sandbox"):
                    raise RuntimeError(f"task {task['task_order']} lacks the sandbox contract")
                source_commit = row.get("official_olmo_eval_commit")
                if source_commit is not None and source_commit != OFFICIAL_OLMO_EVAL_COMMIT:
                    raise RuntimeError(f"unsupported OLMo-Eval scorer commit: {source_commit}")
                key = (int(row["task_order"]), _stable_id(row["example_id"]))
                if key in gold:
                    raise RuntimeError(f"duplicate gold key: {key}")
                gold[key] = row
                if key[0] in MULTIPLE_TASKS:
                    multiple_languages.setdefault(key[0], set()).add(str(row["language"]))
    for task_order, languages in multiple_languages.items():
        if languages != MULTIPLE_CORE88_LANGUAGES:
            raise RuntimeError(
                f"unsupported Core88 MultiPL-E languages for task {task_order}: "
                f"{sorted(languages)} != {sorted(MULTIPLE_CORE88_LANGUAGES)}"
            )
    return gold, tasks


def _load_predictions(
    roots: list[Path], gold: dict[tuple[int, str], dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    files: list[str] = []
    seen: set[tuple[int, str, int]] = set()
    selected_task_orders = {task_order for task_order, _ in gold}
    for root in roots:
        for path in _prediction_files(root):
            match = re.match(r"^(\d{3})_", path.name)
            # Core88 writes one task per prediction file using a stable
            # three-digit task-order prefix.  Avoid parsing the other ~80
            # files in every sandbox shard.  Unknown legacy filenames still
            # take the conservative full-scan path below.
            if match is not None and int(match.group(1)) not in selected_task_orders:
                continue
            files.append(str(path.resolve()))
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    prediction = json.loads(line)
                    key = (int(prediction["task_order"]), _stable_id(prediction["example_id"]))
                    if key not in gold:
                        continue
                    sample_key = (*key, int(prediction.get("sample_index", 0)))
                    if sample_key in seen:
                        raise RuntimeError(
                            f"duplicate prediction at {path}:{line_number}: {sample_key}"
                        )
                    seen.add(sample_key)
                    rows.append((prediction, gold[key]))
    rows.sort(
        key=lambda pair: (
            int(pair[0]["task_order"]),
            _stable_id(pair[0]["example_id"]),
            int(pair[0].get("sample_index", 0)),
        )
    )
    return rows, files


_FENCE = re.compile(
    r"```(?:python|py|cpp|c\+\+|csharp|cs|go|java|javascript|js|php)?\s*\n" r"(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _first_fenced_code(text: str) -> str | None:
    match = _FENCE.search(text)
    return match.group(1).strip("\n") if match else None


def _strip_markdown_fence(text: str) -> str:
    fenced = _first_fenced_code(text)
    if fenced is not None:
        return fenced
    return text.split("```", maxsplit=1)[0].strip("\n")


def _strip_thinking_tags(text: str) -> str:
    pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
    result = pattern.sub("", text)
    return result.strip() if result != text else result


def _extract_humaneval_code(text: str) -> str:
    text = _strip_thinking_tags(text)
    language_fence = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = language_fence.findall(text)
    if matches:
        return matches[0]
    generic_fence = re.compile(r"```\n?(.*?)```", re.DOTALL)
    matches = generic_fence.findall(text)
    if matches:
        return matches[0]
    return re.sub(r"\n?```\s*$", "", text)


def _indent_function_body(code: str) -> str:
    if not code:
        return code
    first_content = next((line for line in code.split("\n") if line.strip()), None)
    if first_content is None or first_content.startswith((" ", "\t")):
        return code
    return "\n".join("    " + line if line.strip() else line for line in code.split("\n"))


def _extract_code_before_fence(text: str) -> str:
    index = text.find("```")
    return text[:index] if index >= 0 else text


def _extract_deepseek_continuation(text: str) -> str:
    code_blocks = re.findall(r"```python\n?(.*?)\n?```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    candidates = re.split(r"\ndef|\nclass|\nif|\n#|\nprint", text)
    return candidates[0] if candidates else text


def _verify_official_olmo_eval_source() -> dict[str, str]:
    if os.environ.get("OLMO_EVAL_COMMIT") != OFFICIAL_OLMO_EVAL_COMMIT:
        raise RuntimeError("BigCodeBench requires OLMO_EVAL_COMMIT=" f"{OFFICIAL_OLMO_EVAL_COMMIT}")
    observed: dict[str, str] = {}
    for module_name, expected_sha256 in OFFICIAL_OLMO_EVAL_SOURCE_SHA256.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin:
            raise RuntimeError(f"official OLMo-Eval module is unavailable: {module_name}")
        path = Path(spec.origin).resolve()
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"OLMo-Eval source SHA256 mismatch for {module_name}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        observed[module_name] = actual_sha256
    return observed


def _python_import_modules(*sources: str) -> tuple[str, ...]:
    modules: set[str] = set()
    for source in sources:
        if not source.strip():
            continue
        try:
            trees = (ast.parse(source),)
        except SyntaxError:
            # BigCodeBench prompts commonly end at an incomplete function
            # signature. Parse standalone import lines instead of dropping
            # imports that precede the incomplete body.
            parsed_lines: list[ast.AST] = []
            for line in source.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                try:
                    parsed_lines.append(ast.parse(stripped))
                except SyntaxError:
                    continue
            trees = tuple(parsed_lines)
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".", maxsplit=1)[0])
    return tuple(sorted(module for module in modules if module not in sys.stdlib_module_names))


_DS1000_EXEC_WRAPPER = """\
import subprocess, sys, tempfile, os
_inner = '''
import contextlib, io, os, sys, tempfile

class WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **k): raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False

class RedirectStdin(contextlib._RedirectStream):
    _stream = "stdin"

_td = tempfile.mkdtemp()
os.chdir(_td)
_stream = WriteOnlyStringIO()
with contextlib.redirect_stdout(_stream):
    with contextlib.redirect_stderr(_stream):
        with RedirectStdin(_stream):
            exec(compile(_TEST_CODE, "<test>", "exec"), {})
'''
_f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
_f.write(f"_TEST_CODE = {repr(_TEST_CODE)}\\n")
_f.write(_inner)
_f.close()
try:
    _r = subprocess.run([sys.executable, _f.name], timeout=115)
    sys.exit(_r.returncode)
finally:
    os.unlink(_f.name)
"""


def _build_ds1000_execution_program(code_context: str, completion: str) -> str:
    inner_code = (
        code_context
        + "\n"
        + f"code = {completion!r}\n"
        + "test_execution(code)\n"
        + ("test_string(code)\n" if "test_string(" in code_context else "\n")
    )
    return f"_TEST_CODE = {inner_code!r}\n" + _DS1000_EXEC_WRAPPER


def _runtime_requirement_for_gold(gold: dict[str, Any]) -> RuntimeRequirement:
    order = int(gold["task_order"])
    if order == 81:
        return RuntimeRequirement(
            "bigcodebench-f8816ee",
            import_modules=_python_import_modules(
                str(gold.get("complete_prompt") or ""),
                str(gold.get("code_prompt") or ""),
                str(gold.get("test") or ""),
            ),
        )
    if order == 84:
        return RuntimeRequirement(
            "ds1000-python310",
            python_version=DS1000_PYTHON_VERSION,
            distributions=DS1000_DISTRIBUTIONS,
            import_modules=DS1000_IMPORT_MODULES,
        )
    if order in MULTIPLE_TASKS:
        language = str(gold["language"])
        config = MULTIPLE_LANGUAGE_CONFIG.get(language)
        if config is None:
            raise RuntimeError(f"unsupported MultiPL-E language: {language}")
        return RuntimeRequirement(
            str(config["runtime_name"]),
            command_probes=config["command_probes"],
            required_paths=config["required_paths"],
            required_file_sha256=config["required_file_sha256"],
        )
    return RuntimeRequirement("core88-python")


def _python_spec(gold: dict[str, Any], completion: str) -> ExecutionSpec:
    order = int(gold["task_order"])
    contract = dict(gold["execution_contract"])
    scorer = str(contract.get("scorer") or "LBPPCodeExecution")
    timeout = float(contract.get("timeout_seconds", 10))
    if order == 79:
        return ExecutionSpec(
            files={"code.py": _strip_markdown_fence(completion), "test.py": str(gold["test_file"])},
            argv=("$PYTHON", "-B", "-s", "-E", "test.py"),
            timeout_seconds=timeout,
            scorer=scorer,
        )
    if order == 81:
        _verify_official_olmo_eval_source()
        try:
            from olmo_eval.evals.extract import sanitize_code
            from olmo_eval.evals.tasks.bigcodebench import _build_bcb_execution_script
        except ImportError as error:
            raise RuntimeError(
                "BigCodeBench requires official olmo-eval commit " f"{OFFICIAL_OLMO_EVAL_COMMIT}"
            ) from error
        candidate = str(gold["complete_prompt"]) + completion
        candidate = sanitize_code(candidate, entrypoint=str(gold["entry_point"]))
        code_prompt = str(gold.get("code_prompt") or "")
        if code_prompt:
            candidate = code_prompt + "\n    pass\n" + candidate
        program = _build_bcb_execution_script(candidate, str(gold["test"]))
        return ExecutionSpec(
            files={"program.py": program},
            argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
            timeout_seconds=3,
            scorer=scorer,
            runtime_requirement=_runtime_requirement_for_gold(gold),
        )
    if order == 82:
        code = _indent_function_body(_extract_humaneval_code(completion))
        program = str(gold["original_prompt"]) + code + "\n\n" + str(gold["test"])
        return ExecutionSpec(
            files={"program.py": program},
            argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
            timeout_seconds=timeout,
            scorer=scorer,
        )
    if order == 83:
        program = (
            str(gold["input"])
            + _extract_deepseek_continuation(completion)
            + "\n\n"
            + str(gold["test"])
        )
        return ExecutionSpec(
            files={"program.py": program},
            argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
            timeout_seconds=timeout,
            scorer=scorer,
        )
    if order == 84:
        code_context = str(gold["code_context"])
        program = _build_ds1000_execution_program(code_context, completion)
        return ExecutionSpec(
            files={"program.py": program},
            argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
            timeout_seconds=120,
            scorer=scorer,
            runtime_requirement=_runtime_requirement_for_gold(gold),
        )
    if order == 85:
        program = (
            _extract_code_before_fence(completion).rstrip() + "\n\n" + str(gold["test"]) + "\n"
        )
        return ExecutionSpec(
            files={"program.py": program},
            argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
            timeout_seconds=timeout,
            scorer=scorer,
        )
    raise RuntimeError(f"unsupported Python code task: {order}")


def _multiple_spec(gold: dict[str, Any], completion: str) -> ExecutionSpec:
    language = str(gold["language"])
    program = str(gold["input"]) + completion + "\n" + str(gold["test"])
    config = MULTIPLE_LANGUAGE_CONFIG.get(language)
    if config is None:
        raise RuntimeError(f"unsupported MultiPL-E language: {language}")
    return ExecutionSpec(
        files={str(config["filename"]): program},
        argv=("/bin/bash", "-c", str(config["command"])),
        timeout_seconds=float(config["timeout"]),
        scorer="MultiplEScorer",
        required_commands=tuple(config["required_commands"]),
        runtime_requirement=_runtime_requirement_for_gold(gold),
    )


def _execution_spec(gold: dict[str, Any], completion: str) -> ExecutionSpec:
    order = int(gold["task_order"])
    if order in PYTHON_CODE_TASKS:
        return _python_spec(gold, completion)
    if order in MULTIPLE_TASKS:
        return _multiple_spec(gold, completion)
    raise RuntimeError(f"unsupported Core code task: {order}")


def _merged_runtime_requirements(
    gold_rows: Iterable[dict[str, Any]]
) -> tuple[RuntimeRequirement, ...]:
    by_name: dict[str, RuntimeRequirement] = {}
    for gold in gold_rows:
        requirement = _runtime_requirement_for_gold(gold)
        previous = by_name.get(requirement.name)
        if previous is None:
            by_name[requirement.name] = requirement
            continue
        if (
            previous.python_version != requirement.python_version
            or previous.distributions != requirement.distributions
            or previous.command_probes != requirement.command_probes
            or previous.required_paths != requirement.required_paths
            or previous.required_file_sha256 != requirement.required_file_sha256
        ):
            raise RuntimeError(f"inconsistent runtime requirement: {requirement.name}")
        by_name[requirement.name] = dataclasses.replace(
            previous,
            import_modules=tuple(
                sorted(set(previous.import_modules) | set(requirement.import_modules))
            ),
        )
    return tuple(by_name[name] for name in sorted(by_name))


def _canonical_control_specs(
    task_orders: set[int], multiple_languages: set[str]
) -> dict[str, tuple[ExecutionSpec, ExecutionSpec]]:
    controls: dict[str, tuple[ExecutionSpec, ExecutionSpec]] = {}
    common_contract = {
        "requires_isolated_sandbox": True,
        "scorer": "canonical_control",
        "timeout_seconds": 10,
    }
    if 79 in task_orders:
        gold = {
            "task_order": 79,
            "test_file": "from code import answer\nassert answer() == 42\n",
            "execution_contract": common_contract,
        }
        controls["task-79"] = (
            _execution_spec(gold, "def answer():\n    return 42\n"),
            _execution_spec(gold, "def answer():\n    return 0\n"),
        )
    if 81 in task_orders:
        gold = {
            "task_order": 81,
            "complete_prompt": "def add(a, b):\n",
            "entry_point": "add",
            "code_prompt": "",
            "test": (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(19, 23), 42)\n"
            ),
            "execution_contract": common_contract,
        }
        controls["task-81"] = (
            _execution_spec(gold, "    return a + b\n"),
            _execution_spec(gold, "    return a - b\n"),
        )
    if 82 in task_orders:
        gold = {
            "task_order": 82,
            "original_prompt": "def answer():\n",
            "test": "assert answer() == 42\n",
            "execution_contract": common_contract,
        }
        controls["task-82"] = (
            _execution_spec(gold, "return 42\n"),
            _execution_spec(gold, "return 0\n"),
        )
    if 83 in task_orders:
        gold = {
            "task_order": 83,
            "input": "def answer():\n    ",
            "test": "assert answer() == 42\n",
            "execution_contract": common_contract,
        }
        controls["task-83"] = (
            _execution_spec(gold, "return 42"),
            _execution_spec(gold, "return 0"),
        )
    if 84 in task_orders:
        gold = {
            "task_order": 84,
            "code_context": (
                "def test_execution(code):\n"
                "    namespace = {}\n"
                "    exec(code, namespace)\n"
                "    assert namespace['answer'] == 42\n"
            ),
            "execution_contract": common_contract,
        }
        controls["task-84"] = (
            _execution_spec(gold, "answer = 42"),
            _execution_spec(gold, "answer = 0"),
        )
    if 85 in task_orders:
        gold = {
            "task_order": 85,
            "test": "assert answer() == 42\n",
            "execution_contract": common_contract,
        }
        controls["task-85"] = (
            _execution_spec(gold, "def answer():\n    return 42\n"),
            _execution_spec(gold, "def answer():\n    return 0\n"),
        )
    multiple_programs = {
        "cpp": (
            "#include <cassert>\nint answer() {\n",
            "return 42;\n}\n",
            "int main() { assert(answer() == 42); }\n",
            "return 0;\n}\n",
        ),
        "cs": (
            (
                "using System;\nusing System.Diagnostics;\n"
                "public class Program {\npublic static int Answer() {\n"
            ),
            "return 42;\n}\n",
            ("public static void Main(string[] args) { " "Debug.Assert(Answer() == 42); }\n}\n"),
            "return 0;\n}\n",
        ),
        "go_test.go": (
            "package answer_test\nimport \"testing\"\nfunc answer() int {\n",
            "return 42\n}\n",
            ("func TestAnswer(t *testing.T) { " "if answer() != 42 { t.Fail() } }\n"),
            "return 0\n}\n",
        ),
        "java": (
            "public class Problem {\nstatic int answer() {\n",
            "return 42;\n}\n",
            (
                "public static void main(String[] args) { "
                "if (answer() != 42) throw new AssertionError(); }\n}\n"
            ),
            "return 0;\n}\n",
        ),
        "js": (
            "function answer() {\n",
            "return 42;\n}\n",
            "if (answer() !== 42) process.exit(1);\n",
            "return 0;\n}\n",
        ),
        "php": (
            "<?php\nfunction answer() {\n",
            "return 42;\n}\n",
            "if (answer() !== 42) { exit(1); }\n",
            "return 0;\n}\n",
        ),
    }
    for language in sorted(multiple_languages):
        if language not in multiple_programs:
            raise RuntimeError(f"unsupported MultiPL-E language: {language}")
        prompt, passing, test, failing = multiple_programs[language]
        gold = {
            "task_order": 86,
            "language": language,
            "input": prompt,
            "test": test,
            "execution_contract": common_contract,
        }
        controls[f"multiple-{language}"] = (
            _execution_spec(gold, passing),
            _execution_spec(gold, failing),
        )
    return controls


def _run_canonical_controls(
    sandbox: "BubblewrapSandbox", *, task_orders: set[int], multiple_languages: set[str]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    failed: list[str] = []
    for name, (passing_spec, failing_spec) in _canonical_control_specs(
        task_orders, multiple_languages
    ).items():
        passing = sandbox.run(passing_spec)
        expected_failure = sandbox.run(failing_spec)
        ok = passing["passed"] and not expected_failure["passed"]
        evidence[name] = {
            "status": "CORE_NATIVE_CANONICAL_CONTROL_OK" if ok else "FAILED",
            "passing": passing,
            "expected_failure": expected_failure,
        }
        if not ok:
            failed.append(name)
    if failed:
        failure_evidence = {name: evidence[name] for name in failed}
        raise RuntimeError(
            "canonical scorer controls failed: "
            + ", ".join(failed)
            + "; evidence="
            + json.dumps(failure_evidence, sort_keys=True)
        )
    return evidence


def _partition_key(prediction: dict[str, Any]) -> int:
    value = (
        f"{int(prediction['task_order'])}:"
        f"{_stable_id(prediction['example_id'])}:"
        f"{int(prediction.get('sample_index', 0))}"
    )
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


class BubblewrapSandbox:
    def __init__(
        self, *, bwrap_path: Path, runtime_prefix: Path, limits: SandboxLimits, stderr_limit: int
    ) -> None:
        self.bwrap_path = bwrap_path.resolve()
        self.runtime_prefix = runtime_prefix.resolve()
        self.limits = limits
        self.stderr_limit = stderr_limit
        if not self.bwrap_path.is_file():
            raise RuntimeError(f"bubblewrap is unavailable: {self.bwrap_path}")
        runtime_python_candidates = (
            self.runtime_prefix / "bin/python",
            self.runtime_prefix / "bin/python3",
        )
        self.runtime_python = next(
            (path for path in runtime_python_candidates if path.is_file()), None
        )
        if self.runtime_python is None:
            raise RuntimeError(
                "runtime Python is unavailable: "
                + ", ".join(str(path) for path in runtime_python_candidates)
            )
        self.runtime_python_in_sandbox = "/runtime/" + str(
            self.runtime_python.relative_to(self.runtime_prefix)
        )
        self.runtime_libpython_in_sandbox: str | None = None
        resolved_python_match = re.search(r"python(\d+\.\d+)$", self.runtime_python.resolve().name)
        if resolved_python_match is not None:
            libpython = (
                self.runtime_prefix / "lib" / f"libpython{resolved_python_match.group(1)}.so.1.0"
            )
            if libpython.is_file():
                self.runtime_libpython_in_sandbox = "/runtime/" + str(
                    libpython.relative_to(self.runtime_prefix)
                )
        self._runtime_evidence: dict[RuntimeRequirement, dict[str, Any]] = {}

    def _base_command(self, workdir: Path) -> list[str]:
        mono_runtime = Path("/etc/mono").is_dir()
        command = [
            str(self.bwrap_path),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(path).exists():
                command.extend(("--ro-bind", path, path))
        command.extend(("--dir", "/etc"))
        for path in ("/etc/alternatives", "/etc/java-11-openjdk", "/etc/mono", "/etc/php"):
            if Path(path).exists():
                command.extend(("--ro-bind", path, path))
        proc_mount = ("--ro-bind", "/proc", "/proc") if mono_runtime else ("--dir", "/proc")
        command.extend(
            (
                "--ro-bind",
                str(self.runtime_prefix),
                "/runtime",
                "--dev",
                "/dev",
                *proc_mount,
                "--tmpfs",
                "/tmp",
                "--dir",
                "/home",
                "--dir",
                "/run",
                "--bind",
                str(workdir),
                "/work",
                "--chdir",
                "/work",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PATH",
                "/runtime/bin:/usr/bin:/bin",
                "--setenv",
                "PYTHONHOME",
                "/runtime",
                "--setenv",
                "LD_LIBRARY_PATH",
                "/runtime/lib:/runtime/lib64",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "--setenv",
                "MPLCONFIGDIR",
                "/tmp/matplotlib",
                "--setenv",
                "MPLBACKEND",
                "Agg",
                "--setenv",
                "OMP_NUM_THREADS",
                "1",
                "--setenv",
                "OPENBLAS_NUM_THREADS",
                "1",
                "--setenv",
                "MKL_NUM_THREADS",
                "1",
                "--uid",
                "65534",
                "--gid",
                "65534",
            )
        )
        if self.runtime_libpython_in_sandbox is not None:
            command.extend(("--setenv", "LD_PRELOAD", self.runtime_libpython_in_sandbox))
        return command

    def _set_limits(self, timeout_seconds: float) -> None:
        resource.setrlimit(
            resource.RLIMIT_AS, (self.limits.address_space_bytes, self.limits.address_space_bytes)
        )
        cpu_seconds = max(1, int(timeout_seconds) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (self.limits.file_size_bytes, self.limits.file_size_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_NPROC, (self.limits.process_count, self.limits.process_count)
        )
        resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits.open_files, self.limits.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(self, spec: ExecutionSpec) -> dict[str, Any]:
        missing = [command for command in spec.required_commands if shutil.which(command) is None]
        if missing:
            raise RuntimeError("sandbox image lacks required commands: " + ", ".join(missing))
        captured_files: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="core88-code-") as temporary:
            workdir = Path(temporary)
            workdir.chmod(0o777)
            for relative, text in spec.files.items():
                path = workdir / relative
                path.write_text(text, encoding="utf-8")
                path.chmod(0o644)
            argv = [
                self.runtime_python_in_sandbox if value == "$PYTHON" else value
                for value in spec.argv
            ]
            command = self._base_command(workdir) + ["--", *argv]
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=lambda: self._set_limits(spec.timeout_seconds),
            )
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            duration = time.monotonic() - started
            for relative in spec.captured_files:
                capture_path = (workdir / relative).resolve()
                if workdir.resolve() not in capture_path.parents:
                    raise RuntimeError(f"captured file escapes workdir: {relative}")
                if not capture_path.is_file():
                    continue
                if capture_path.stat().st_size > 1024 * 1024:
                    raise RuntimeError(f"captured file is too large: {relative}")
                captured_files[relative] = capture_path.read_text(
                    encoding="utf-8", errors="replace"
                )
        stderr_text = stderr.decode("utf-8", errors="replace")
        stdout_text = stdout.decode("utf-8", errors="replace")
        return {
            "passed": not timed_out and process.returncode == 0,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "stdout_tail": stdout_text[-self.stderr_limit :],
            "stderr_tail": stderr_text[-self.stderr_limit :],
            "captured_files": captured_files,
        }

    def verify_runtime(self, requirement: RuntimeRequirement) -> dict[str, Any]:
        cached = self._runtime_evidence.get(requirement)
        if cached is not None:
            return cached
        requirement_payload = dataclasses.asdict(requirement)
        probe_program = f"""
import importlib
import importlib.metadata
import hashlib
import json
import os
import platform
import subprocess

requirement = {requirement_payload!r}
evidence = {{
    "status": "CORE_NATIVE_SCORER_RUNTIME_OK",
    "requirement": requirement,
    "python_version": platform.python_version(),
    "distributions": {{}},
    "imports": {{}},
    "commands": {{}},
    "paths": {{}},
    "files": {{}},
    "errors": [],
}}
expected_python = requirement.get("python_version")
if expected_python and evidence["python_version"] != expected_python:
    evidence["errors"].append(
        f"python version {{evidence['python_version']}} != {{expected_python}}"
    )
for distribution, expected_version in requirement.get("distributions", ()):
    try:
        actual_version = importlib.metadata.version(distribution)
    except Exception as error:
        evidence["errors"].append(f"distribution {{distribution}}: {{type(error).__name__}}: {{error}}")
    else:
        evidence["distributions"][distribution] = actual_version
        if actual_version != expected_version:
            evidence["errors"].append(
                f"distribution {{distribution}} {{actual_version}} != {{expected_version}}"
            )
for module_name in requirement.get("import_modules", ()):
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        evidence["errors"].append(f"import {{module_name}}: {{type(error).__name__}}: {{error}}")
    else:
        evidence["imports"][module_name] = getattr(module, "__file__", None)
for command, arguments in requirement.get("command_probes", ()):
    try:
        result = subprocess.run(
            [command, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception as error:
        evidence["errors"].append(f"command {{command}}: {{type(error).__name__}}: {{error}}")
    else:
        output = result.stdout.strip()
        evidence["commands"][command] = {{
            "exit_code": result.returncode,
            "output": output[:1024],
        }}
        if result.returncode != 0:
            evidence["errors"].append(f"command {{command}} exited {{result.returncode}}")
for path in requirement.get("required_paths", ()):
    exists = os.path.exists(path)
    evidence["paths"][path] = exists
    if not exists:
        evidence["errors"].append(f"required path is absent: {{path}}")
for path, expected_sha256 in requirement.get("required_file_sha256", ()):
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
    except Exception as error:
        evidence["errors"].append(f"file {{path}}: {{type(error).__name__}}: {{error}}")
    else:
        evidence["files"][path] = actual_sha256
        if actual_sha256 != expected_sha256:
            evidence["errors"].append(
                f"file {{path}} sha256 {{actual_sha256}} != {{expected_sha256}}"
            )
if evidence["errors"]:
    evidence["status"] = "CORE_NATIVE_SCORER_RUNTIME_FAILED"
with open("runtime_evidence.json", "w", encoding="utf-8") as handle:
    json.dump(evidence, handle, sort_keys=True)
print("CORE_NATIVE_RUNTIME_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
raise SystemExit(0 if not evidence["errors"] else 2)
"""
        result = self.run(
            ExecutionSpec(
                files={"runtime_probe.py": probe_program},
                argv=("$PYTHON", "-B", "-s", "-E", "runtime_probe.py"),
                timeout_seconds=30,
                scorer=f"runtime_probe:{requirement.name}",
                captured_files=("runtime_evidence.json",),
            )
        )
        captured = result["captured_files"].get("runtime_evidence.json")
        if captured is None:
            raise RuntimeError(
                f"runtime probe emitted no evidence for {requirement.name}: "
                f"{result['stderr_tail']}"
            )
        evidence = json.loads(captured)
        if not result["passed"] or evidence.get("status") != "CORE_NATIVE_SCORER_RUNTIME_OK":
            raise RuntimeError(
                f"runtime requirement failed for {requirement.name}: "
                + "; ".join(evidence.get("errors") or ["probe failed"])
                + f"; stdout={result['stdout_tail']!r}; stderr={result['stderr_tail']!r}"
            )
        self._runtime_evidence[requirement] = evidence
        return evidence

    def self_test(self) -> dict[str, Any]:
        isolation_program = """
import os
import socket

assert os.getuid() == 65534
assert not os.path.exists("/host")
sock = socket.socket()
sock.settimeout(0.25)
assert sock.connect_ex(("1.1.1.1", 53)) != 0
print("CORE_NATIVE_SANDBOX_ISOLATION_OK")
"""
        pass_result = self.run(
            ExecutionSpec(
                files={"program.py": isolation_program},
                argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
                timeout_seconds=3,
                scorer="self_test_isolation",
            )
        )
        fail_result = self.run(
            ExecutionSpec(
                files={"program.py": "raise AssertionError('expected failure')\n"},
                argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
                timeout_seconds=3,
                scorer="self_test_failure",
            )
        )
        timeout_result = self.run(
            ExecutionSpec(
                files={"program.py": "while True:\n    pass\n"},
                argv=("$PYTHON", "-B", "-s", "-E", "program.py"),
                timeout_seconds=0.5,
                scorer="self_test_timeout",
            )
        )
        ok = (
            pass_result["passed"]
            and "CORE_NATIVE_SANDBOX_ISOLATION_OK" in pass_result["stdout_tail"]
            and not fail_result["passed"]
            and timeout_result["timed_out"]
        )
        return {
            "status": "CORE_NATIVE_SANDBOX_SELF_TEST_OK" if ok else "FAILED",
            "isolation": pass_result,
            "expected_failure": fail_result,
            "timeout": timeout_result,
        }


def _load_existing_results(path: Path) -> set[tuple[int, str, int]]:
    seen: set[tuple[int, str, int]] = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            key = (int(row["task_order"]), _stable_id(row["example_id"]), int(row["sample_index"]))
            if key in seen:
                raise RuntimeError(f"duplicate existing result at {path}:{line_number}: {key}")
            if row.get("schema_version") != RESULT_SCHEMA_VERSION:
                raise RuntimeError(f"unexpected result schema at {path}:{line_number}")
            seen.add(key)
    return seen


def _result_aggregate(
    path: Path, selected: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    """Seal the complete resumed shard into a small mergeable score summary."""

    expected = {
        (
            int(prediction["task_order"]),
            _stable_id(prediction["example_id"]),
            int(prediction.get("sample_index", 0)),
        ): gold_row
        for prediction, gold_row in selected
    }
    seen: set[tuple[int, str, int]] = set()
    task_counts: dict[int, dict[str, int]] = {}
    subset_counts: dict[int, dict[str, dict[str, int]]] = {}
    example_counts: dict[int, dict[str, dict[str, Any]]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("schema_version") != RESULT_SCHEMA_VERSION:
                    raise RuntimeError(f"unexpected result schema at {path}:{line_number}")
                if row.get("status") not in {"PASS", "FAIL"}:
                    raise RuntimeError(f"unexpected result status at {path}:{line_number}")
                if row.get("sandbox_self_test") != "CORE_NATIVE_SANDBOX_SELF_TEST_OK":
                    raise RuntimeError(f"unverified sandbox result at {path}:{line_number}")
                key = (
                    int(row["task_order"]),
                    _stable_id(row["example_id"]),
                    int(row["sample_index"]),
                )
                if key in seen:
                    raise RuntimeError(f"duplicate existing result at {path}:{line_number}: {key}")
                gold_row = expected.get(key)
                if gold_row is None:
                    raise RuntimeError(
                        f"result is outside this scorer shard at {path}:{line_number}"
                    )
                seen.add(key)
                passed = bool(row["passed"])
                counts = task_counts.setdefault(key[0], {"selected": 0, "passed": 0, "failed": 0})
                counts["selected"] += 1
                counts["passed" if passed else "failed"] += 1
                subset = str(
                    gold_row.get("language")
                    if key[0] in MULTIPLE_TASKS
                    else (gold_row.get("subset") or "__all__")
                )
                subset_task_counts = subset_counts.setdefault(key[0], {})
                subset_row = subset_task_counts.setdefault(
                    subset, {"selected": 0, "passed": 0, "failed": 0}
                )
                subset_row["selected"] += 1
                subset_row["passed" if passed else "failed"] += 1
                task_example_counts = example_counts.setdefault(key[0], {})
                example_row = task_example_counts.setdefault(
                    key[1], {"subset": subset, "selected": 0, "passed": 0, "failed": 0}
                )
                if example_row["subset"] != subset:
                    raise RuntimeError(f"example subset changed at {path}:{line_number}: {key}")
                example_row["selected"] += 1
                example_row["passed" if passed else "failed"] += 1
    missing = sorted(set(expected) - seen)
    return {
        "status": (
            "CORE_NATIVE_CODE_RESULT_AGGREGATE_OK"
            if not missing and len(seen) == len(expected)
            else "CORE_NATIVE_CODE_RESULT_AGGREGATE_INCOMPLETE"
        ),
        "expected_result_count": len(expected),
        "observed_result_count": len(seen),
        "missing_result_count": len(missing),
        "output_jsonl_sha256": _file_sha256(path) if path.exists() else None,
        "output_jsonl_bytes": path.stat().st_size if path.exists() else 0,
        "task_counts": {str(key): value for key, value in sorted(task_counts.items())},
        "subset_counts": {
            str(task_order): dict(sorted(counts.items()))
            for task_order, counts in sorted(subset_counts.items())
        },
        "example_counts": {
            str(task_order): dict(sorted(counts.items()))
            for task_order, counts in sorted(example_counts.items())
        },
    }


def _append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sync_jsonl(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _prepare_scorer_runtime(
    sandbox: BubblewrapSandbox,
    *,
    gold: dict[tuple[int, str], dict[str, Any]],
    task_orders: set[int],
) -> dict[str, Any]:
    source_fidelity = _verify_official_olmo_eval_source() if 81 in task_orders else {}
    requirements = _merged_runtime_requirements(gold.values())
    runtime_evidence = {
        requirement.name: sandbox.verify_runtime(requirement) for requirement in requirements
    }
    multiple_languages = {
        str(row["language"]) for row in gold.values() if int(row["task_order"]) in MULTIPLE_TASKS
    }
    controls = _run_canonical_controls(
        sandbox, task_orders=task_orders, multiple_languages=multiple_languages
    )
    return {
        "status": "CORE_NATIVE_SCORER_PREFLIGHT_OK",
        "official_olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
        "official_source_sha256": source_fidelity,
        "runtime_evidence": runtime_evidence,
        "canonical_controls": controls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--profile", default="core88")
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--error-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--task-orders", default="")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bwrap-path", type=Path, default=None)
    parser.add_argument("--runtime-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--runtime-image", default="")
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--file-size-mib", type=int, default=32)
    parser.add_argument("--process-limit", type=int, default=64)
    parser.add_argument("--open-file-limit", type=int, default=128)
    parser.add_argument("--stderr-limit", type=int, default=4096)
    parser.add_argument("--fsync-every", type=int, default=16)
    parser.add_argument("--allow-experimental-scorers", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.partition_count <= 0 or not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index must be in [0, partition count)")
    bwrap = args.bwrap_path or Path(shutil.which("bwrap") or "")
    sandbox = BubblewrapSandbox(
        bwrap_path=bwrap,
        runtime_prefix=args.runtime_prefix,
        limits=SandboxLimits(
            address_space_bytes=args.memory_mib * 1024 * 1024,
            file_size_bytes=args.file_size_mib * 1024 * 1024,
            process_count=args.process_limit,
            open_files=args.open_file_limit,
        ),
        stderr_limit=args.stderr_limit,
    )
    self_test = sandbox.self_test()
    if self_test["status"] != "CORE_NATIVE_SANDBOX_SELF_TEST_OK":
        print(json.dumps(self_test, indent=2, ensure_ascii=False))
        raise SystemExit(2)
    if args.self_test:
        payload: dict[str, Any] = {
            "sandbox_self_test": self_test,
            "runtime_image": args.runtime_image,
        }
        if args.task_orders.strip():
            if not args.data_root:
                raise ValueError("--data-root is required for task-specific scorer self-test")
            task_orders = _parse_task_orders(
                args.task_orders, allow_experimental_scorers=args.allow_experimental_scorers
            )
            data_root = args.data_root.resolve()
            gold, _ = _load_gold(data_root, args.profile, task_orders)
            payload["scorer_preflight"] = _prepare_scorer_runtime(
                sandbox, gold=gold, task_orders=task_orders
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if not args.data_root or not args.input_root or not args.output_jsonl:
        raise ValueError("--data-root, at least one --input-root, and --output-jsonl are required")
    task_orders = _parse_task_orders(
        args.task_orders, allow_experimental_scorers=args.allow_experimental_scorers
    )
    data_root = args.data_root.resolve()
    gold, tasks = _load_gold(data_root, args.profile, task_orders)
    scorer_preflight = _prepare_scorer_runtime(sandbox, gold=gold, task_orders=task_orders)
    predictions, prediction_files = _load_predictions(args.input_root, gold)
    selected = [
        pair
        for pair in predictions
        if _partition_key(pair[0]) % args.partition_count == args.partition_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not predictions:
        raise RuntimeError("no predictions matched the selected code-execution task orders")
    if not selected:
        raise RuntimeError(f"partition {args.partition_index}/{args.partition_count} is empty")
    if args.fsync_every <= 0:
        raise ValueError("--fsync-every must be positive")
    output_jsonl = args.output_jsonl.resolve()
    error_jsonl = (
        args.error_jsonl.resolve()
        if args.error_jsonl
        else output_jsonl.with_name(output_jsonl.stem + "-errors.jsonl")
    )
    summary_json = (
        args.summary_json.resolve()
        if args.summary_json
        else output_jsonl.with_name(output_jsonl.stem + "-summary.json")
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output_jsonl.unlink(missing_ok=True)
        error_jsonl.unlink(missing_ok=True)
    completed = _load_existing_results(output_jsonl)
    task_counts: dict[int, dict[str, int]] = {}
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    new_results_since_sync = 0
    mode = "a"
    with (
        output_jsonl.open(mode, encoding="utf-8") as output_handle,
        error_jsonl.open(mode, encoding="utf-8") as error_handle,
    ):
        for prediction, gold_row in selected:
            key = (
                int(prediction["task_order"]),
                _stable_id(prediction["example_id"]),
                int(prediction.get("sample_index", 0)),
            )
            counts = task_counts.setdefault(
                key[0], {"selected": 0, "passed": 0, "failed": 0, "errors": 0}
            )
            counts["selected"] += 1
            if key in completed:
                skipped += 1
                continue
            try:
                spec = _execution_spec(gold_row, str(prediction["completion"]))
                result = sandbox.run(spec)
            except Exception as exc:
                errors += 1
                counts["errors"] += 1
                _append_jsonl(
                    error_handle,
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "status": "SANDBOX_INFRASTRUCTURE_ERROR",
                        "task_order": key[0],
                        "example_id": prediction["example_id"],
                        "sample_index": key[2],
                        "error": f"{type(exc).__name__}: {exc}",
                        "recorded_at": _utc_now(),
                    },
                )
                _sync_jsonl(error_handle)
                continue
            payload = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "PASS" if result["passed"] else "FAIL",
                "task_order": key[0],
                "example_id": prediction["example_id"],
                "sample_index": key[2],
                "passed": bool(result["passed"]),
                "scorer": spec.scorer,
                "sandbox_backend": "bubblewrap",
                "sandbox_self_test": self_test["status"],
                "recorded_at": _utc_now(),
                **result,
            }
            _append_jsonl(output_handle, payload)
            new_results_since_sync += 1
            if new_results_since_sync >= args.fsync_every:
                _sync_jsonl(output_handle)
                new_results_since_sync = 0
            if result["passed"]:
                passed += 1
                counts["passed"] += 1
            else:
                failed += 1
                counts["failed"] += 1
        _sync_jsonl(output_handle)
        _sync_jsonl(error_handle)
    result_aggregate = _result_aggregate(output_jsonl, selected)
    aggregate_complete = result_aggregate["status"] == "CORE_NATIVE_CODE_RESULT_AGGREGATE_OK"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": (
            "CORE_NATIVE_CODE_RESULTS_OK"
            if errors == 0 and aggregate_complete
            else "CORE_NATIVE_CODE_RESULTS_INFRASTRUCTURE_ERROR"
        ),
        "profile": args.profile,
        "data_root": str(data_root),
        "data_summary_sha256": _file_sha256(data_root / "summary.json"),
        "input_roots": [str(path.resolve()) for path in args.input_root],
        "prediction_files": prediction_files,
        "task_orders": sorted(task_orders),
        "scorer_fidelity": {
            "official_olmo_eval_commit": OFFICIAL_OLMO_EVAL_COMMIT,
            "verified_task_orders": sorted(task_orders & VERIFIED_SCORER_TASKS),
            "experimental_task_orders": sorted(task_orders - VERIFIED_SCORER_TASKS),
        },
        "scorer_preflight": scorer_preflight,
        "runtime_image": args.runtime_image,
        "tasks": tasks,
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "selected_predictions": len(selected),
        "new_passed": passed,
        "new_failed": failed,
        "infrastructure_errors": errors,
        "resume_skipped": skipped,
        "output_jsonl": str(output_jsonl),
        "error_jsonl": str(error_jsonl),
        "sandbox_self_test": self_test,
        "task_counts": {str(key): value for key, value in sorted(task_counts.items())},
        "result_aggregate": result_aggregate,
        "finished_at": _utc_now(),
    }
    temporary = summary_json.with_suffix(summary_json.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors or not aggregate_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
