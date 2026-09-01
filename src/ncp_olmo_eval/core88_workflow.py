#!/usr/bin/env python3
"""Bind one Core88 inference run to one fresh standalone GSM8K run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dflash_contract import validate_dflash_operating_point
from .lmdeploy_inference import lmdeploy_engine_policy
from .source_identity import source_state

SCHEMA_VERSION = "core88-standalone-gsm8k-workflow-v2"
WORKFLOW_STATUS = "CORE88_STANDALONE_GSM8K_WORKFLOW_READY"
COMPANION_STATUS = "CORE88_GSM8K_COMPANION_OK"
FORMAL_WORKFLOW_MODE = "formal_dispatch_plan"
DIAGNOSTIC_WORKFLOW_MODE = "diagnostic_planless"
CANONICAL_STANDARD_INPUT_CONFIG = Path(
    os.environ.get("NCP_OLMO_GSM8K_PROMPT", "standard_input_ours_gsm8k.json")
)
CANONICAL_STANDARD_INPUT_CONFIG_SHA256 = (
    "295395763cbe551cbe41481c1a9ad16b491768c8d50911bd6ab1cd4f69e2b265"
)
CANONICAL_PROMPT_SHA256 = "03423a681bb3c3571df5c22d7cb5338adaa25fa14c17a149e96fee3f11d88d5e"
CANONICAL_DATASET_TEST_FILE = Path(
    os.environ.get("NCP_OLMO_GSM8K_TEST_FILE", "test-00000-of-00001.parquet")
)
CANONICAL_DATASET_TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
CANONICAL_INPUTS_JSONL_SHA256 = "0bc37f497eb3542ca26dbadf7379727fb6a1b5c47a9d0631efed8216cf45f126"
GSM8K_TASK_GROUP = "olmo_eval_paper_math_gsm_8shot"
GSM8K_TASK = "olmo_eval_paper_gsm8k_main"
GSM8K_EXAMPLE_COUNT = 1319
EVALUATION_SEED = 42
GSM8K_SEED = EVALUATION_SEED
SCORING_DESCENDANT_STATUS = "CORE88_SCORING_DESCENDANT_OK"
_SCORING_DESCENDANT_RELATIVE_PATHS = frozenset(
    {
        "src/ncp_olmo_eval/core88_workflow.py",
        "src/ncp_olmo_eval/core_native_aggregate.py",
        "src/ncp_olmo_eval/core_native_summary.py",
        "src/ncp_olmo_eval/portable_tasks.py",
        "tests/unit/test_core88_workflow.py",
        "tests/unit/test_core_native_summary.py",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _same_path(first: str | Path, second: str | Path) -> bool:
    return Path(first).resolve() == Path(second).resolve()


def _validate_repo_state(workflow: dict[str, Any]) -> None:
    """Require the evaluator source to retain the workflow identity."""

    expected_commit = str(workflow["repo_commit"])
    identity = workflow.get("source_identity")
    current = source_state(Path(str(workflow["repo_root"])))
    if current["repo_dirty"] or current["repo_commit"] != expected_commit:
        raise RuntimeError(
            "Core88 evaluator source changed after submission: "
            f"revision={current['repo_commit']!r}, dirty={current['repo_dirty']}"
        )
    if isinstance(identity, dict):
        expected_tree = str(identity.get("tree_sha256", ""))
        if len(expected_tree) != 64 or current["source_tree_sha256"] != expected_tree:
            raise RuntimeError("Core88 evaluator source-tree digest changed after submission")


def validate_scoring_evaluator_repo(
    workflow: dict[str, Any],
    state: dict[str, Any],
    *,
    allow_descendant: bool = False,
) -> dict[str, Any]:
    """Validate a clean final scorer against the inference source identity.

    Exact source revision and tree equality remains the default. An explicit scoring
    upgrade may use a clean descendant commit when every changed file belongs
    to the small scoring/sealing allowlist. This permits corrected graders and
    artifact validators without weakening the immutable inference contract.
    """

    evaluator_root = Path(str(state["evaluator_repo_root"])).resolve()
    evaluator_commit = str(state.get("evaluator_repo_commit", ""))
    workflow_root = Path(str(workflow["repo_root"])).resolve()
    workflow_commit = str(workflow["repo_commit"])
    if state.get("evaluator_repo_dirty") is not False:
        raise RuntimeError("Core88 evaluator repository must be clean")
    workflow_identity = workflow.get("source_identity")
    evaluator_tree = str(state.get("evaluator_source_tree_sha256", ""))
    workflow_tree = (
        str(workflow_identity.get("tree_sha256", ""))
        if isinstance(workflow_identity, dict)
        else ""
    )
    exact_tree = bool(workflow_tree) and evaluator_tree == workflow_tree
    if evaluator_commit == workflow_commit and (exact_tree or evaluator_root == workflow_root):
        return {
            "status": "CORE88_EVALUATOR_EXACT_WORKFLOW_COMMIT",
            "workflow_repo_root": str(workflow_root),
            "workflow_repo_commit": workflow_commit,
            "evaluator_repo_root": str(evaluator_root),
            "evaluator_repo_commit": evaluator_commit,
            "source_tree_sha256": evaluator_tree or None,
            "changed_files": [],
        }
    if not allow_descendant:
        raise RuntimeError("Core88 evaluator repository differs from workflow")

    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(evaluator_root),
                "merge-base",
                "--is-ancestor",
                workflow_commit,
                evaluator_commit,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        prefix = subprocess.run(
            ["git", "-C", str(evaluator_root), "rev-parse", "--show-prefix"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        changed_output = subprocess.run(
            [
                "git",
                "-C",
                str(evaluator_root),
                "diff",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                f"{workflow_commit}..{evaluator_commit}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Core88 evaluator is not a descendant of the workflow commit"
        ) from error

    allowed_paths = {f"{prefix}{path}" for path in _SCORING_DESCENDANT_RELATIVE_PATHS}
    changed_files = sorted(line.strip() for line in changed_output.splitlines() if line.strip())
    unexpected = sorted(set(changed_files) - allowed_paths)
    if not changed_files or unexpected:
        raise RuntimeError(
            "Core88 scoring descendant contains non-scoring changes: "
            f"{unexpected or changed_files}"
        )
    return {
        "status": SCORING_DESCENDANT_STATUS,
        "workflow_repo_root": str(workflow_root),
        "workflow_repo_commit": workflow_commit,
        "evaluator_repo_root": str(evaluator_root),
        "evaluator_repo_commit": evaluator_commit,
        "changed_files": changed_files,
    }


def _prompt_sha256(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _gsm8k_protocol(
    backend: str = "native_vllm",
    *,
    lmdeploy_cache_max_entry_count: float = 0.8,
    batch_size: int = 8,
    scheduler_queue_size: int = 0,
    max_model_len: int = 2048,
    speculative_operating_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend not in {"native_vllm", "lmdeploy"}:
        raise ValueError(f"unsupported Core88 GSM8K backend: {backend}")
    protocol = {
        "backend": backend,
        "standard_input_config": str(CANONICAL_STANDARD_INPUT_CONFIG),
        "standard_input_config_sha256": CANONICAL_STANDARD_INPUT_CONFIG_SHA256,
        "dataset_test_file": str(CANONICAL_DATASET_TEST_FILE),
        "dataset_test_sha256": CANONICAL_DATASET_TEST_SHA256,
        "prompt_sha256": CANONICAL_PROMPT_SHA256,
        "inputs_jsonl_sha256": CANONICAL_INPUTS_JSONL_SHA256,
        "task_group": GSM8K_TASK_GROUP,
        "task": GSM8K_TASK,
        "dataset_sample_count": GSM8K_EXAMPLE_COUNT,
        "num_fewshot": 8,
        "fewshot_seed": GSM8K_SEED,
        "sampling_seed": GSM8K_SEED,
        "gpu_count": 8,
        "samples_per_doc": 1,
        "batch_size": batch_size,
        "scheduler_queue_size": scheduler_queue_size or batch_size,
        "limit": 0,
        "max_model_len": max_model_len,
        "vllm_speculative_operating_point": speculative_operating_point,
    }
    if backend == "lmdeploy":
        # This legacy schema field records the model's configured context
        # length. LMDeploy is deliberately not given a session_len override;
        # sealing validates the positive value resolved by the runtime instead.
        protocol["max_model_len"] = 8192
        protocol["lmdeploy_config"] = lmdeploy_engine_policy(
            max_batch_size=8,
            cache_max_entry_count=lmdeploy_cache_max_entry_count,
        )
    return protocol


def validate_gsm8k_max_model_len(
    protocol: dict[str, Any], aggregate: dict[str, Any]
) -> int:
    """Validate and return the effective standalone GSM8K context length.

    Native vLLM receives a fixed ``max_model_len`` and must reproduce the
    workflow value exactly. LMDeploy intentionally leaves ``session_len`` at
    its runtime default, so its aggregate must instead agree with the resolved
    ``lmdeploy_runtime.session_len`` reported by that same inference run.
    """

    backend = str(protocol.get("backend"))
    actual = aggregate.get("max_model_len")
    if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 1:
        raise RuntimeError(f"GSM8K aggregate max_model_len is invalid: {actual!r}")
    if backend != "lmdeploy":
        expected = protocol.get("max_model_len")
        if actual != expected:
            raise RuntimeError(
                f"GSM8K aggregate max_model_len changed: {actual!r} != {expected!r}"
            )
        return actual

    expected_runtime = protocol.get("lmdeploy_config")
    runtime = aggregate.get("lmdeploy_runtime")
    if not isinstance(expected_runtime, dict) or not isinstance(runtime, dict):
        raise RuntimeError("GSM8K LMDeploy runtime metadata is missing")
    defaulted_fields = expected_runtime.get("defaulted_engine_fields")
    if not isinstance(defaulted_fields, list) or "session_len" not in defaulted_fields:
        raise RuntimeError("GSM8K LMDeploy session_len is not runtime-defaulted")
    resolved = runtime.get("session_len")
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved <= 1:
        raise RuntimeError(f"GSM8K LMDeploy resolved session_len is invalid: {resolved!r}")
    if actual != resolved:
        raise RuntimeError(
            "GSM8K LMDeploy aggregate max_model_len differs from its resolved "
            f"session_len: {actual!r} != {resolved!r}"
        )
    return actual


def create_workflow(
    *,
    output_root: Path,
    repo_root: Path,
    repo_commit: str,
    source_kind: str = "git",
    source_tree_sha256: str = "",
    hf_model_path: Path,
    model_identity_path: Path,
    model_label: str,
    run_tag: str,
    core_job_names: list[str],
    gsm8k_job_name: str,
    global_seed: int,
    core_plan_json: Path | None = None,
    hf_backend: str = "auto",
    vllm_model_family: str = "conceptlm",
    vllm_runtime_config: Path | None = None,
    processes_per_gpu: int = 4,
    score_batch_size: int = 1,
    generation_batch_size: int = 1,
    row_chunk_size: int = 1,
    seq_length: int = 2048,
    vllm_max_model_len: int = 0,
    vllm_scheduler_queue_size: int = 0,
    vllm_speculative_operating_point: dict[str, Any] | None = None,
    gsm8k_batch_size: int = 8,
    gsm8k_scheduler_queue_size: int = 0,
    gsm8k_max_model_len: int = 2048,
    lmdeploy_cache_max_entry_count: float = 0.8,
    diagnostic_planless: bool = False,
) -> dict[str, Any]:
    """Create a fresh workflow root; existing output is never reused."""

    if len(core_job_names) != 4 or len(set(core_job_names)) != 4:
        raise ValueError("Core88 workflow requires four unique Core job names")
    if global_seed != EVALUATION_SEED:
        raise ValueError(
            f"Core88 global seed is fixed to {EVALUATION_SEED}, got {global_seed}"
        )
    if len(repo_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repo_commit.lower()
    ):
        raise ValueError("Core88 workflow requires a full 40-character source revision")
    if not source_tree_sha256:
        source_tree_sha256 = str(source_state(repo_root)["source_tree_sha256"])
    if source_kind not in {"git", "installed-package"}:
        raise ValueError(f"unsupported evaluator source kind: {source_kind}")
    if len(source_tree_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_tree_sha256.lower()
    ):
        raise ValueError("Core88 workflow requires a full evaluator source-tree SHA-256")
    if hf_backend not in {
        "auto",
        "from_pretrained",
        "transformers",
        "native_vllm",
        "lmdeploy",
    }:
        raise ValueError(f"unsupported Core88 HF backend: {hf_backend}")
    if vllm_model_family not in {"conceptlm", "auto"}:
        raise ValueError(f"unsupported native-vLLM model family: {vllm_model_family}")
    for name, value in {
        "processes_per_gpu": processes_per_gpu,
        "score_batch_size": score_batch_size,
        "generation_batch_size": generation_batch_size,
        "row_chunk_size": row_chunk_size,
        "seq_length": seq_length,
    }.items():
        if value <= 0:
            raise ValueError(f"Core88 {name} must be positive")
    if vllm_max_model_len < 0:
        raise ValueError("Core88 vllm_max_model_len cannot be negative")
    if vllm_scheduler_queue_size < 0:
        raise ValueError("Core88 vllm_scheduler_queue_size cannot be negative")
    if vllm_scheduler_queue_size and vllm_scheduler_queue_size < generation_batch_size:
        raise ValueError("Core88 scheduler queue must be at least generation batch size")
    if gsm8k_batch_size <= 0:
        raise ValueError("Core88 GSM8K batch size must be positive")
    if gsm8k_scheduler_queue_size and gsm8k_scheduler_queue_size < gsm8k_batch_size:
        raise ValueError("Core88 GSM8K scheduler queue must be at least batch size")
    if gsm8k_max_model_len <= 0:
        raise ValueError("Core88 GSM8K max_model_len must be positive")
    if vllm_speculative_operating_point is not None:
        vllm_speculative_operating_point = validate_dflash_operating_point(
            vllm_speculative_operating_point
        )
        expected_batch = int(vllm_speculative_operating_point["max_num_seqs"])
        expected_queue = int(vllm_speculative_operating_point["scheduler_queue_size"])
        expected_max_model_len = int(vllm_speculative_operating_point["max_model_len"])
        if score_batch_size != expected_batch or generation_batch_size != expected_batch:
            raise ValueError("Core88 batch sizes differ from sealed DFlash operating point")
        if (vllm_scheduler_queue_size or generation_batch_size) != expected_queue:
            raise ValueError("Core88 scheduler queue differs from sealed DFlash operating point")
        if vllm_max_model_len != expected_max_model_len:
            raise ValueError("Core88 max_model_len differs from sealed DFlash operating point")
        if gsm8k_batch_size != expected_batch:
            raise ValueError("Core88 GSM8K batch differs from sealed DFlash operating point")
        if (gsm8k_scheduler_queue_size or gsm8k_batch_size) != expected_queue:
            raise ValueError("Core88 GSM8K queue differs from sealed DFlash operating point")
        if gsm8k_max_model_len != expected_max_model_len:
            raise ValueError("Core88 GSM8K max_model_len differs from sealed DFlash operating point")
    if not 0.0 < lmdeploy_cache_max_entry_count < 1.0:
        raise ValueError("Core88 LMDeploy cache fraction must be in (0, 1)")
    if vllm_model_family == "auto" and vllm_runtime_config is not None:
        raise ValueError("stock vLLM models cannot use a ConceptLM runtime config")
    if core_plan_json is None and not diagnostic_planless:
        raise ValueError(
            "formal Core88 workflow requires core_plan_json; use "
            "diagnostic_planless=True only for non-final diagnostics"
        )
    if core_plan_json is not None and diagnostic_planless:
        raise ValueError("diagnostic_planless cannot be combined with a Core88 dispatch plan")

    output_root = output_root.resolve()
    plan: dict[str, Any] | None = None
    plan_path: Path | None = None
    if core_plan_json is not None:
        plan_path = core_plan_json.resolve()
        if plan_path != output_root / "dispatch-plan" / "plan.json":
            raise ValueError("Core88 dispatch plan must be OUTPUT_ROOT/dispatch-plan/plan.json")
        plan = _read_json(plan_path)
        if int(plan.get("machine_count", -1)) != 4:
            raise ValueError("Core88 dispatch plan must contain four machines")
        if int(plan.get("global_seed", -1)) != global_seed:
            raise ValueError("Core88 dispatch plan seed differs from the workflow seed")
        if output_root.is_dir():
            children = sorted(path.name for path in output_root.iterdir())
            if children != ["dispatch-plan"]:
                raise FileExistsError(
                    "preplanned Core88 workflow root contains unexpected entries: " f"{children}"
                )
        else:
            raise FileNotFoundError(
                "Core88 workflow output root must contain its dispatch plan first"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    runtime_config = None
    if vllm_runtime_config is not None:
        runtime_path = vllm_runtime_config.resolve()
        if not runtime_path.is_file():
            raise FileNotFoundError(f"Core88 runtime config is missing: {runtime_path}")
        runtime_config = {"path": str(runtime_path), "sha256": file_sha256(runtime_path)}
    core_protocol: dict[str, Any] = {
        "machine_count": 4,
        "gpus_per_machine": 8,
        "global_seed": global_seed,
        "hf_backend": hf_backend,
        "vllm_model_family": vllm_model_family,
        "vllm_runtime_config": runtime_config,
        "processes_per_gpu": processes_per_gpu,
        "score_batch_size": score_batch_size,
        "generation_batch_size": generation_batch_size,
        "row_chunk_size": row_chunk_size,
        "seq_length": seq_length,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_scheduler_queue_size": vllm_scheduler_queue_size,
        "vllm_speculative_operating_point": vllm_speculative_operating_point,
    }
    if hf_backend == "lmdeploy":
        core_protocol.update(
            {
                "lmdeploy_cache_max_entry_count": lmdeploy_cache_max_entry_count,
                "lmdeploy_engine_policy": lmdeploy_engine_policy(
                    max_batch_size=max(score_batch_size, generation_batch_size),
                    cache_max_entry_count=lmdeploy_cache_max_entry_count,
                ),
            }
        )
    if plan is not None and plan_path is not None:
        core_protocol.update(
            {
                "plan_json": str(plan_path),
                "plan_json_sha256": file_sha256(plan_path),
                "plan_id": plan["plan_id"],
                "profile": plan["profile"],
                "data_root": plan["data_root"],
                "data_summary_sha256": plan["data_summary_sha256"],
                "task_orders": plan["task_orders"],
                "limit_per_task": plan["limit_per_task"],
                "generation_samples_cap": plan["generation_samples_cap"],
                "max_gen_tokens_cap": plan["max_gen_tokens_cap"],
                "local_dispatch": plan["local_dispatch"],
                "dispatch_plan_schema": plan.get("schema_version"),
                "assignment_strategy": plan.get("assignment_strategy"),
                "cost_model": plan.get("cost_model"),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": WORKFLOW_STATUS,
        "workflow_mode": (FORMAL_WORKFLOW_MODE if plan is not None else DIAGNOSTIC_WORKFLOW_MODE),
        "workflow_id": uuid.uuid4().hex,
        "created_at": _utc_now(),
        "run_tag": run_tag,
        "repo_root": str(repo_root.resolve()),
        "repo_commit": repo_commit,
        "source_identity": {
            "kind": source_kind,
            "revision": repo_commit,
            "tree_sha256": source_tree_sha256,
        },
        "model_label": model_label,
        "hf_model_path": str(hf_model_path.resolve()),
        "model_identity_path": str(model_identity_path.resolve()),
        "output_root": str(output_root),
        "core_results_root": str(output_root),
        "gsm8k_results_root": str(output_root / "standalone-gsm8k"),
        "finalize_root": str(output_root / "finalize"),
        "core_job_names": list(core_job_names),
        "gsm8k_job_name": gsm8k_job_name,
        "core_protocol": core_protocol,
        "gsm8k_protocol": _gsm8k_protocol(
            "lmdeploy" if hf_backend == "lmdeploy" else "native_vllm",
            lmdeploy_cache_max_entry_count=lmdeploy_cache_max_entry_count,
            batch_size=gsm8k_batch_size,
            scheduler_queue_size=gsm8k_scheduler_queue_size,
            max_model_len=gsm8k_max_model_len,
            speculative_operating_point=vllm_speculative_operating_point,
        ),
    }
    _write_json_atomic(output_root / "workflow.json", payload)
    return payload


def load_workflow(path: Path) -> dict[str, Any]:
    """Load and validate the immutable parts of a workflow manifest."""

    path = path.resolve()
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported Core88 workflow schema in {path}")
    if payload.get("status") != WORKFLOW_STATUS:
        raise RuntimeError(f"Core88 workflow is not ready: {path}")
    workflow_id = payload.get("workflow_id")
    if (
        not isinstance(workflow_id, str)
        or len(workflow_id) != 32
        or any(character not in "0123456789abcdef" for character in workflow_id)
    ):
        raise RuntimeError(f"invalid Core88 workflow id in {path}")
    core_job_names = payload.get("core_job_names")
    if (
        not isinstance(core_job_names, list)
        or len(core_job_names) != 4
        or any(not isinstance(name, str) for name in core_job_names)
        or len(set(core_job_names)) != 4
    ):
        raise RuntimeError("workflow does not contain four unique Core jobs")
    core_protocol = payload.get("core_protocol")
    if not isinstance(core_protocol, dict) or {
        "machine_count": core_protocol.get("machine_count"),
        "gpus_per_machine": core_protocol.get("gpus_per_machine"),
    } != {"machine_count": 4, "gpus_per_machine": 8}:
        raise RuntimeError("workflow Core topology differs from 4x8 GPUs")
    output_root = Path(str(payload["output_root"])).resolve()
    expected_paths = {
        "core_results_root": output_root,
        "gsm8k_results_root": output_root / "standalone-gsm8k",
        "finalize_root": output_root / "finalize",
    }
    for field, expected in expected_paths.items():
        if not _same_path(payload[field], expected):
            raise RuntimeError(f"workflow {field} escapes its output root")
    if path != output_root / "workflow.json":
        raise RuntimeError("workflow manifest is not at OUTPUT_ROOT/workflow.json")
    expected_gsm_backend = (
        "lmdeploy" if core_protocol.get("hf_backend") == "lmdeploy" else "native_vllm"
    )
    if payload.get("gsm8k_protocol") != _gsm8k_protocol(
        expected_gsm_backend,
        lmdeploy_cache_max_entry_count=float(
            core_protocol.get("lmdeploy_cache_max_entry_count", 0.8)
        ),
    ):
        raise RuntimeError("workflow GSM8K protocol differs from the fixed contract")
    plan_json = core_protocol.get("plan_json")
    expected_mode = FORMAL_WORKFLOW_MODE if plan_json is not None else DIAGNOSTIC_WORKFLOW_MODE
    if payload.get("workflow_mode") != expected_mode:
        raise RuntimeError("workflow mode does not match its Core dispatch-plan binding")
    if plan_json is not None:
        expected_plan_path = output_root / "dispatch-plan" / "plan.json"
        if not _same_path(str(plan_json), expected_plan_path):
            raise RuntimeError("workflow Core dispatch plan escapes its output root")
        if file_sha256(expected_plan_path) != core_protocol.get("plan_json_sha256"):
            raise RuntimeError("workflow Core dispatch plan changed")
        plan = _read_json(expected_plan_path)
        for field in (
            "plan_id",
            "profile",
            "data_root",
            "data_summary_sha256",
            "task_orders",
            "limit_per_task",
            "generation_samples_cap",
            "max_gen_tokens_cap",
            "local_dispatch",
        ):
            if plan.get(field) != core_protocol.get(field):
                raise RuntimeError(f"workflow Core plan field changed: {field}")
        optional_plan_fields = {
            "schema_version": "dispatch_plan_schema",
            "assignment_strategy": "assignment_strategy",
            "cost_model": "cost_model",
        }
        for plan_field, protocol_field in optional_plan_fields.items():
            expected = core_protocol.get(protocol_field)
            if expected is not None and plan.get(plan_field) != expected:
                raise RuntimeError(f"workflow Core plan field changed: {plan_field}")
    runtime_config = core_protocol.get("vllm_runtime_config")
    if runtime_config is not None:
        if not isinstance(runtime_config, dict):
            raise RuntimeError("workflow native-vLLM runtime config is invalid")
        runtime_path = Path(str(runtime_config.get("path")))
        if not runtime_path.is_file() or file_sha256(runtime_path) != runtime_config.get("sha256"):
            raise RuntimeError("workflow native-vLLM runtime config changed")
    return payload


def require_formal_workflow(workflow: dict[str, Any]) -> None:
    """Reject a planless diagnostic workflow at every formal-output gate."""

    protocol = workflow.get("core_protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("formal Core88 workflow has no Core protocol")
    required_plan_fields = (
        "plan_json",
        "plan_json_sha256",
        "plan_id",
        "profile",
        "data_root",
        "data_summary_sha256",
        "task_orders",
        "limit_per_task",
        "generation_samples_cap",
        "max_gen_tokens_cap",
        "local_dispatch",
    )
    if workflow.get("workflow_mode") != FORMAL_WORKFLOW_MODE or any(
        protocol.get(field) is None for field in required_plan_fields
    ):
        raise RuntimeError(
            "final Core88 output requires a workflow bound to a dispatch plan; "
            "planless workflows are diagnostic only"
        )


def validate_core_run_manifest(workflow: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Validate one shared Core run manifest against its workflow contract."""

    require_formal_workflow(workflow)
    protocol = workflow["core_protocol"]
    direct_checks = {
        "workflow_id": workflow["workflow_id"],
        "repo_commit": workflow["repo_commit"],
        "profile": protocol.get("profile"),
        "data_root": protocol.get("data_root"),
        "data_summary_sha256": protocol.get("data_summary_sha256"),
        "task_orders": protocol.get("task_orders"),
        "plan_id": protocol.get("plan_id"),
        "global_seed": protocol["global_seed"],
        "machine_count": protocol["machine_count"],
        "gpus_per_machine": protocol["gpus_per_machine"],
        "processes_per_gpu": protocol.get("processes_per_gpu"),
        "limit_per_task": protocol.get("limit_per_task"),
        "generation_samples_cap": protocol.get("generation_samples_cap"),
        "max_gen_tokens_cap": protocol.get("max_gen_tokens_cap"),
        "score_batch_size": protocol.get("score_batch_size"),
        "generation_batch_size": protocol.get("generation_batch_size"),
        "hf_backend_requested": protocol.get("hf_backend"),
        "dispatch_plan_schema": protocol.get("dispatch_plan_schema"),
        "assignment_strategy": protocol.get("assignment_strategy"),
        "cost_model": protocol.get("cost_model"),
    }
    for field, expected in direct_checks.items():
        if expected is not None and manifest.get(field) != expected:
            raise RuntimeError(
                f"Core run manifest {field} differs from workflow: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    if protocol.get("hf_backend") == "native_vllm":
        native = manifest.get("native_vllm_config")
        if not isinstance(native, dict):
            raise RuntimeError("Core run manifest lost native-vLLM configuration")
        expected_max_model_len = protocol.get("vllm_max_model_len") or (
            int(protocol["seq_length"]) + 2
        )
        native_checks = {
            "model_family": protocol.get("vllm_model_family"),
            "max_model_len": expected_max_model_len,
            "max_num_seqs": max(
                int(protocol["score_batch_size"]), int(protocol["generation_batch_size"])
            ),
            "scheduler_queue_size": protocol.get("vllm_scheduler_queue_size")
            or max(int(protocol["score_batch_size"]), int(protocol["generation_batch_size"])),
        }
        for field, expected in native_checks.items():
            if native.get(field) != expected:
                raise RuntimeError(
                    f"Core native-vLLM {field} differs from workflow: "
                    f"{native.get(field)!r} != {expected!r}"
                )
        speculative_point = protocol.get("vllm_speculative_operating_point")
        if isinstance(speculative_point, dict):
            point_checks = {
                "vllm_use_v2_model_runner": speculative_point[
                    "vllm_use_v2_model_runner"
                ],
                "execution_mode": speculative_point["execution_mode"],
                "gpu_memory_utilization": speculative_point["gpu_memory_utilization"],
                "attention_backend": speculative_point["attention_backend"],
                "flash_attn_version": speculative_point["flash_attn_version"],
                "hlm_attention_impl": speculative_point["hlm_attention_impl"],
                "speculative_verification_mode": speculative_point[
                    "speculative_verification_mode"
                ],
                "speculative_num_tokens": speculative_point["speculative_num_tokens"],
                "speculative_draft_attention_backend": speculative_point[
                    "draft_attention_backend"
                ],
                "speculative_context_kv_cache": speculative_point["context_kv_cache"],
                "speculative_sparse_context_projection": speculative_point[
                    "sparse_context_projection"
                ],
                "speculative_min_eligible_batch": speculative_point[
                    "min_eligible_batch"
                ],
                "speculative_min_proposal_tokens_per_row": speculative_point[
                    "min_proposal_tokens_per_row"
                ],
                "speculative_min_proposal_tokens_per_batch": speculative_point[
                    "min_proposal_tokens_per_batch"
                ],
                "speculative_runtime_block_size": speculative_point["runtime_block_size"],
                "speculative_active_batch_widths": speculative_point[
                    "active_batch_widths"
                ],
                "speculative_dynamic_runtime_block_size": speculative_point[
                    "dynamic_runtime_block_size"
                ],
                "speculative_runtime_layer_count": speculative_point[
                    "runtime_layer_count"
                ],
                "speculative_runtime_local_mixer": speculative_point[
                    "runtime_local_mixer"
                ],
                "speculative_mixer_compile_mode": speculative_point[
                    "mixer_compile_mode"
                ],
                "speculative_chunk_size": speculative_point["chunk_size"],
                "speculative_target_layers": speculative_point["target_layers"],
                "speculative_telemetry_flush_interval": speculative_point[
                    "telemetry_flush_interval"
                ],
            }
            for field, expected in point_checks.items():
                if native.get(field) != expected:
                    raise RuntimeError(
                        f"Core sealed DFlash {field} differs from workflow: "
                        f"{native.get(field)!r} != {expected!r}"
                    )
    if protocol.get("hf_backend") == "lmdeploy":
        lmdeploy = manifest.get("lmdeploy_config")
        if not isinstance(lmdeploy, dict):
            raise RuntimeError("Core run manifest lost LMDeploy configuration")
        lmdeploy_checks = protocol["lmdeploy_engine_policy"]
        for field, expected in lmdeploy_checks.items():
            if lmdeploy.get(field) != expected:
                raise RuntimeError(
                    f"Core LMDeploy {field} differs from workflow: "
                    f"{lmdeploy.get(field)!r} != {expected!r}"
                )


def verify_inputs(workflow_path: Path, gsm8k_root: Path) -> dict[str, Any]:
    """Fail before GPU generation if any fixed GSM8K input artifact drifted."""

    workflow = load_workflow(workflow_path)
    root = gsm8k_root.resolve()
    if not _same_path(root, workflow["gsm8k_results_root"]):
        raise RuntimeError("GSM8K output root does not belong to this workflow")
    manifest_path = root / "inputs" / "manifest.json"
    manifest = _read_json(manifest_path)
    expected = workflow["gsm8k_protocol"]
    inputs_jsonl = root / "inputs" / "inputs.jsonl"
    checks = {
        "status": "GSM8K_INPUTS_READY",
        "task": expected["task"],
        "task_group": expected["task_group"],
        "standard_input_config": expected["standard_input_config"],
        "standard_input_config_sha256": expected["standard_input_config_sha256"],
        "dataset_test_file": expected["dataset_test_file"],
        "dataset_test_sha256": expected["dataset_test_sha256"],
        "dataset_sample_count": expected["dataset_sample_count"],
        "fewshot_seed": expected["fewshot_seed"],
        "prompt_sha256": expected["prompt_sha256"],
        "inputs_jsonl": str(inputs_jsonl.resolve()),
        "inputs_jsonl_sha256": expected["inputs_jsonl_sha256"],
    }
    for field, value in checks.items():
        actual = manifest.get(field)
        if field in {"standard_input_config", "dataset_test_file", "inputs_jsonl"}:
            matches = _same_path(str(actual), str(value))
        else:
            matches = actual == value
        if not matches:
            raise RuntimeError(f"prepared GSM8K {field} changed: {actual!r} != {value!r}")
    artifact_checks = (
        (
            Path(expected["standard_input_config"]),
            expected["standard_input_config_sha256"],
            "standard input config",
        ),
        (Path(expected["dataset_test_file"]), expected["dataset_test_sha256"], "dataset test file"),
        (inputs_jsonl, expected["inputs_jsonl_sha256"], "inputs JSONL"),
    )
    for path, expected_sha256, label in artifact_checks:
        if not path.is_file():
            raise RuntimeError(f"GSM8K {label} is missing: {path}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"GSM8K {label} SHA256 changed: " f"{actual_sha256} != {expected_sha256}"
            )
    rows = _read_jsonl(inputs_jsonl)
    if len(rows) != expected["dataset_sample_count"]:
        raise RuntimeError(
            "prepared GSM8K input row count changed: "
            f"{len(rows)} != {expected['dataset_sample_count']}"
        )
    doc_indices = [row.get("doc_index") for row in rows]
    if doc_indices != list(range(expected["dataset_sample_count"])):
        raise RuntimeError("prepared GSM8K doc_index sequence changed")
    prompts = [row.get("prompt") for row in rows]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise RuntimeError("prepared GSM8K inputs contain a non-string prompt")
    actual_prompt_sha256 = _prompt_sha256(prompts)
    if actual_prompt_sha256 != expected["prompt_sha256"]:
        raise RuntimeError(
            "prepared GSM8K rendered prompts changed: "
            f"{actual_prompt_sha256} != {expected['prompt_sha256']}"
        )
    if not (root / "inputs" / "_SUCCESS").is_file():
        raise RuntimeError("prepared GSM8K inputs have no success marker")
    return manifest


def seal_companion(
    *,
    workflow_path: Path,
    gsm8k_root: Path,
    repo_commit: str,
    model_identity_path: Path,
    runtime_model_path: Path,
    model_family: str,
    model_preparation_manifest: Path,
) -> dict[str, Any]:
    """Bind a completed GSM8K aggregate to this exact Core88 workflow."""

    workflow = load_workflow(workflow_path)
    require_formal_workflow(workflow)
    root = gsm8k_root.resolve()
    verify_inputs(workflow_path, root)
    if not _same_path(root, workflow["gsm8k_results_root"]):
        raise RuntimeError("GSM8K output root does not belong to this workflow")
    if repo_commit != workflow["repo_commit"]:
        raise RuntimeError("GSM8K repo commit differs from the Core88 workflow")
    _validate_repo_state(workflow)
    if not _same_path(model_identity_path, workflow["model_identity_path"]):
        raise RuntimeError("GSM8K model identity differs from the Core88 workflow")
    preparation = _read_json(model_preparation_manifest)
    if preparation.get("weight_files_mutated") is not False:
        raise RuntimeError("GSM8K model preparation did not preserve source weights")
    if not _same_path(preparation["source_model"], workflow["hf_model_path"]):
        raise RuntimeError("GSM8K prepared the wrong HF model source")
    if not _same_path(preparation["destination_model"], runtime_model_path):
        raise RuntimeError("GSM8K runtime model differs from its preparation manifest")
    aggregate_path = root / "aggregate.json"
    aggregate = _read_json(aggregate_path)
    backend = str(workflow["gsm8k_protocol"]["backend"])
    expected_aggregate_status = (
        "GSM8K_LMDEPLOY_EVAL_OK"
        if backend == "lmdeploy"
        else "GSM8K_VLLM_EVAL_OK"
    )
    if aggregate.get("status") != expected_aggregate_status:
        raise RuntimeError("GSM8K aggregate is not successful")
    if aggregate.get("backend") != backend:
        raise RuntimeError("GSM8K aggregate backend differs from the workflow")
    if aggregate.get("model_family") != model_family:
        raise RuntimeError("GSM8K aggregate model family changed")
    protocol = workflow["gsm8k_protocol"]
    if backend == "lmdeploy":
        lmdeploy_runtime = aggregate.get("lmdeploy_runtime")
        expected_runtime = protocol.get("lmdeploy_config")
        if not isinstance(lmdeploy_runtime, dict) or not isinstance(
            expected_runtime, dict
        ):
            raise RuntimeError("GSM8K LMDeploy runtime metadata is missing")
        for field, expected in expected_runtime.items():
            if lmdeploy_runtime.get(field) != expected:
                raise RuntimeError(
                    f"GSM8K LMDeploy {field} changed: "
                    f"{lmdeploy_runtime.get(field)!r} != {expected!r}"
                )
    speculative_point = protocol.get("vllm_speculative_operating_point")
    if isinstance(speculative_point, dict):
        speculative_checks = {
            "batch_size": speculative_point["max_num_seqs"],
            "max_num_seqs": speculative_point["max_num_seqs"],
            "scheduler_queue_size": speculative_point["scheduler_queue_size"],
            "max_model_len": speculative_point["max_model_len"],
            "gpu_memory_utilization": speculative_point["gpu_memory_utilization"],
            "execution_mode": speculative_point["execution_mode"],
            "attention_backend": speculative_point["attention_backend"],
            "flash_attn_version": speculative_point["flash_attn_version"],
            "hlm_attention_impl": speculative_point["hlm_attention_impl"],
            "vllm_use_v2_model_runner": speculative_point["vllm_use_v2_model_runner"],
            "speculative_verification_mode": speculative_point[
                "speculative_verification_mode"
            ],
            "speculative_num_tokens": speculative_point["speculative_num_tokens"],
            "speculative_draft_attention_backend": speculative_point[
                "draft_attention_backend"
            ],
            "speculative_context_kv_cache": speculative_point["context_kv_cache"],
            "speculative_sparse_context_projection": speculative_point[
                "sparse_context_projection"
            ],
            "speculative_min_eligible_batch": speculative_point["min_eligible_batch"],
            "speculative_min_proposal_tokens_per_row": speculative_point[
                "min_proposal_tokens_per_row"
            ],
            "speculative_min_proposal_tokens_per_batch": speculative_point[
                "min_proposal_tokens_per_batch"
            ],
            "speculative_runtime_block_size": speculative_point["runtime_block_size"],
            "speculative_active_batch_widths": speculative_point[
                "active_batch_widths"
            ],
            "speculative_dynamic_runtime_block_size": speculative_point[
                "dynamic_runtime_block_size"
            ],
            "speculative_runtime_layer_count": speculative_point["runtime_layer_count"],
            "speculative_runtime_local_mixer": speculative_point["runtime_local_mixer"],
            "speculative_mixer_compile_mode": speculative_point["mixer_compile_mode"],
            "speculative_chunk_size": speculative_point["chunk_size"],
            "speculative_target_layers": speculative_point["target_layers"],
            "speculative_telemetry_flush_interval": speculative_point[
                "telemetry_flush_interval"
            ],
        }
        for field, expected in speculative_checks.items():
            if aggregate.get(field) != expected:
                raise RuntimeError(
                    f"GSM8K sealed DFlash {field} changed: "
                    f"{aggregate.get(field)!r} != {expected!r}"
                )
    effective_max_model_len = validate_gsm8k_max_model_len(protocol, aggregate)
    aggregate_checks = {
        "task": protocol["task"],
        "task_group": protocol["task_group"],
        "standard_input_config_sha256": protocol["standard_input_config_sha256"],
        "dataset_test_sha256": protocol["dataset_test_sha256"],
        "dataset_sample_count": protocol["dataset_sample_count"],
        "sample_count": protocol["dataset_sample_count"],
        "samples_per_doc": protocol["samples_per_doc"],
        "fewshot_seed": protocol["fewshot_seed"],
        "sampling_seed": protocol["sampling_seed"],
        "prompt_sha256": protocol["prompt_sha256"],
        "batch_size": protocol["batch_size"],
        "max_num_seqs": protocol["batch_size"],
        "scheduler_queue_size": protocol.get("scheduler_queue_size", protocol["batch_size"]),
        "gpu_count": protocol["gpu_count"],
    }
    if backend == "lmdeploy":
        aggregate_checks["top_k"] = 0
    for field, expected in aggregate_checks.items():
        actual = aggregate.get(field)
        if actual != expected:
            raise RuntimeError(f"GSM8K aggregate {field} changed: {actual!r} != {expected!r}")
    predictions_path = (root / "predictions.jsonl").resolve()
    if not predictions_path.is_file():
        raise RuntimeError("GSM8K aggregate has no root predictions.jsonl")
    if not _same_path(str(aggregate.get("predictions_jsonl", "")), predictions_path):
        raise RuntimeError("GSM8K aggregate predictions escape its result root")
    inputs_manifest_path = root / "inputs" / "manifest.json"
    inputs_jsonl_path = root / "inputs" / "inputs.jsonl"
    companion = {
        "schema_version": SCHEMA_VERSION,
        "status": COMPANION_STATUS,
        "created_at": _utc_now(),
        "workflow_id": workflow["workflow_id"],
        "workflow_manifest": str(workflow_path.resolve()),
        "workflow_manifest_sha256": file_sha256(workflow_path),
        "repo_commit": repo_commit,
        "hf_model_path": workflow["hf_model_path"],
        "model_identity_path": str(model_identity_path.resolve()),
        "runtime_model_path": str(runtime_model_path.resolve()),
        "model_family": model_family,
        "backend": backend,
        "max_model_len": effective_max_model_len,
        "max_model_len_policy": (
            "lmdeploy_runtime_default" if backend == "lmdeploy" else "fixed_workflow_value"
        ),
        "model_preparation_manifest": str(model_preparation_manifest.resolve()),
        "model_preparation_manifest_sha256": file_sha256(model_preparation_manifest),
        "gsm8k_results_root": str(root),
        "aggregate_json": str(aggregate_path.resolve()),
        "aggregate_json_sha256": file_sha256(aggregate_path),
        "predictions_jsonl": str(predictions_path),
        "predictions_jsonl_sha256": file_sha256(predictions_path),
        "inputs_manifest": str(inputs_manifest_path.resolve()),
        "inputs_manifest_sha256": file_sha256(inputs_manifest_path),
        "inputs_jsonl": str(inputs_jsonl_path.resolve()),
        "inputs_jsonl_sha256": file_sha256(inputs_jsonl_path),
    }
    _write_json_atomic(root / "core88-companion.json", companion)
    (root / "_CORE88_SUCCESS").touch()
    return companion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--repo-root", type=Path, required=True)
    create.add_argument("--repo-commit", required=True)
    create.add_argument("--hf-model-path", type=Path, required=True)
    create.add_argument("--model-identity-path", type=Path, required=True)
    create.add_argument("--model-label", required=True)
    create.add_argument("--run-tag", required=True)
    create.add_argument("--core-job-name", action="append", required=True)
    create.add_argument("--gsm8k-job-name", required=True)
    create.add_argument("--global-seed", type=int, required=True)
    create.add_argument("--core-plan-json", type=Path)
    create.add_argument(
        "--hf-backend",
        choices=("auto", "from_pretrained", "transformers", "native_vllm", "lmdeploy"),
        default="auto",
    )
    create.add_argument("--vllm-model-family", choices=("conceptlm", "auto"), default="conceptlm")
    create.add_argument("--vllm-runtime-config", type=Path)
    create.add_argument("--processes-per-gpu", type=int, default=4)
    create.add_argument("--score-batch-size", type=int, default=1)
    create.add_argument("--generation-batch-size", type=int, default=1)
    create.add_argument("--row-chunk-size", type=int, default=1)
    create.add_argument("--seq-length", type=int, default=2048)
    create.add_argument("--vllm-max-model-len", type=int, default=0)
    create.add_argument("--vllm-scheduler-queue-size", type=int, default=0)
    create.add_argument("--lmdeploy-cache-max-entry-count", type=float, default=0.8)
    create.add_argument(
        "--diagnostic-planless",
        action="store_true",
        help=(
            "Create a planless diagnostic workflow. Such a workflow cannot be "
            "sealed or used for final Core88 output."
        ),
    )

    show = subparsers.add_parser("show")
    show.add_argument("--workflow-manifest", type=Path, required=True)
    show.add_argument("--field", required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--workflow-manifest", type=Path, required=True)
    verify.add_argument("--gsm8k-root", type=Path, required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--workflow-manifest", type=Path, required=True)
    seal.add_argument("--gsm8k-root", type=Path, required=True)
    seal.add_argument("--repo-commit", required=True)
    seal.add_argument("--model-identity-path", type=Path, required=True)
    seal.add_argument("--runtime-model-path", type=Path, required=True)
    seal.add_argument("--model-family", choices=("conceptlm", "auto"), required=True)
    seal.add_argument("--model-preparation-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        result = create_workflow(
            output_root=args.output_root,
            repo_root=args.repo_root,
            repo_commit=args.repo_commit,
            hf_model_path=args.hf_model_path,
            model_identity_path=args.model_identity_path,
            model_label=args.model_label,
            run_tag=args.run_tag,
            core_job_names=args.core_job_name,
            gsm8k_job_name=args.gsm8k_job_name,
            global_seed=args.global_seed,
            core_plan_json=args.core_plan_json,
            hf_backend=args.hf_backend,
            vllm_model_family=args.vllm_model_family,
            vllm_runtime_config=args.vllm_runtime_config,
            processes_per_gpu=args.processes_per_gpu,
            score_batch_size=args.score_batch_size,
            generation_batch_size=args.generation_batch_size,
            row_chunk_size=args.row_chunk_size,
            seq_length=args.seq_length,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_scheduler_queue_size=args.vllm_scheduler_queue_size,
            lmdeploy_cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
            diagnostic_planless=args.diagnostic_planless,
        )
    elif args.command == "show":
        result = load_workflow(args.workflow_manifest)
        if args.field not in result:
            raise KeyError(f"workflow has no field {args.field!r}")
        print(result[args.field])
        return
    elif args.command == "verify-inputs":
        result = verify_inputs(args.workflow_manifest, args.gsm8k_root)
    else:
        result = seal_companion(
            workflow_path=args.workflow_manifest,
            gsm8k_root=args.gsm8k_root,
            repo_commit=args.repo_commit,
            model_identity_path=args.model_identity_path,
            runtime_model_path=args.runtime_model_path,
            model_family=args.model_family,
            model_preparation_manifest=args.model_preparation_manifest,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
