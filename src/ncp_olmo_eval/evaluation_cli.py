#!/usr/bin/env python3
"""Unified, scheduler-neutral model registration and evaluation workflow.

The CLI records immutable model identities, emits benchmark task specs, and
validates artifacts.  It deliberately has no cluster API dependency: execute
the generated specs locally or adapt them to Slurm, Kubernetes, or another
scheduler without changing the benchmark protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .core_native_contract import (
    CORE88_ASSIGNMENT_STRATEGY,
    CORE88_COST_MODEL,
    CORE88_LOCAL_DISPATCH,
    CORE88_PLAN_SCHEMA,
)
from .dflash_checkpoint import dflash_checkpoint_identity, dflash_identity_matches
from .dflash_contract import (
    dflash_operating_point_env,
    validate_dflash_operating_point,
)
from .long_context_protocol import HELMET_SWEEP_INPUT_LENGTHS, RULER_SEQUENCE_LENGTHS
from .native_vllm_inference import (
    NCP_DFLASH_APPROXIMATE_VERIFICATION_MODES,
    NCP_DFLASH_EXACT_VERIFICATION_MODES,
    NCP_DFLASH_VERIFICATION_MODES,
    normalize_ncp_dflash_verification_mode,
)
from .source_identity import source_state
from .task_spec import Resources, TaskSpec, read_status, write_plan, write_status, write_task

_CACHE_ROOT = Path(
    os.environ.get("NCP_OLMO_EVAL_CACHE", "~/.cache/ncp_olmo_eval")
).expanduser()
EVALUATION_ROOT = Path(
    os.environ.get("NCP_OLMO_EVAL_ROOT", str(_CACHE_ROOT / "evaluations"))
).expanduser()
JOB_PREFIX = os.environ.get("NCP_OLMO_JOB_PREFIX", "ncp-eval")
DEFAULT_EXECUTOR = os.environ.get("NCP_OLMO_EVAL_EXECUTOR", "emit")
RUNTIME_IMAGE = os.environ.get("NCP_OLMO_EVAL_IMAGE", "")
PYTHON_BIN = os.environ.get("NCP_OLMO_PYTHON", "python")
DEFAULT_GLOBAL_SEED = 42
CORE88_VLLM_MAX_MODEL_LEN = 8192
MODEL_SCHEMA_VERSION = "conceptlm-unified-evaluation-model-v1"
BENCHMARK_SCHEMA_VERSION = "conceptlm-unified-evaluation-benchmark-v1"
BACKENDS = ("vllm",)
BENCHMARKS = ("gsm8k", "sciq", "core88", "ruler", "helmet")
LONG_CONTEXT_LENGTHS = {"ruler": RULER_SEQUENCE_LENGTHS, "helmet": HELMET_SWEEP_INPUT_LENGTHS}
BACKEND_NAMES = {"vllm": "native_vllm"}
ACTIVE_STATES = frozenset({"Planned", "Running"})
SUCCESS_STATES = frozenset({"Succeeded"})
RETRYABLE_STATES = frozenset({"Failed"})
KNOWN_STATES = ACTIVE_STATES | SUCCESS_STATES | RETRYABLE_STATES

CORE_DATA_ROOT = Path(
    os.environ.get("NCP_OLMO_CORE88_DATA_ROOT", str(_CACHE_ROOT / "core88"))
).expanduser()
SCIQ_SOURCE_PROFILE = "all_supported_local"
SCIQ_TASK_ORDER = 348
SCIQ_TASK_NAME = "olmo_eval_sciq"
SCIQ_EXAMPLE_COUNT = 1000
SCIQ_SOURCE_FILE = "all_supported_local/348_olmo_eval_sciq.jsonl.gz"
SCIQ_SOURCE_SHA256 = "b0bf31832d352e29b846f0e05893b1ffcef9d7ba2d4c350fdb36fad1f9fa0db3"
PREPARED_DATA_ROOT = Path(
    os.environ.get("NCP_OLMO_PREPARED_DATA_ROOT", str(_CACHE_ROOT / "prepared"))
).expanduser()
HELMET_OFFICIAL_ROOT = Path(
    os.environ.get("NCP_OLMO_HELMET_SOURCE_ROOT", str(_CACHE_ROOT / "sources/HELMET-af609c4"))
).expanduser()
ANSWER_SCORER_ROOT = Path(
    os.environ.get("NCP_OLMO_CORE88_ANSWER_SCORER_ROOT", str(_CACHE_ROOT / "core88-answer"))
).expanduser()
OLMO_EVAL_ROOT = Path(
    os.environ.get("NCP_OLMO_OLMO_EVAL_ROOT", str(_CACHE_ROOT / "sources/olmo-eval-f8816ee"))
).expanduser()
OLMO_EVAL_COMMIT = "f8816eea36563f27b4a9dd2533d68d34f3c67d3f"
STANDARD_GSM8K_INPUT = Path(
    os.environ.get("NCP_OLMO_GSM8K_PROMPT", str(_CACHE_ROOT / "gsm8k/standard_input.json"))
).expanduser()
GSM8K_TASK_INCLUDE = Path(
    os.environ.get("NCP_OLMO_GSM8K_TASK_INCLUDE", str(_CACHE_ROOT / "gsm8k/tasks"))
).expanduser()
GSM8K_TEST_FILE = Path(
    os.environ.get("NCP_OLMO_GSM8K_TEST_FILE", str(_CACHE_ROOT / "gsm8k/test.parquet"))
).expanduser()

CORE_SCORER_GROUPS = (
    {
        "index": 0,
        "task_orders": "79,82,83,85",
        "partition_count": 8,
        "image_env": "CORE88_PYTHON_IMAGE",
        "default_image": (
            os.environ.get("NCP_OLMO_CORE88_PYTHON_IMAGE", "")
        ),
        "runtime_prefix": "/usr",
        "allow_experimental": "0",
    },
    {
        "index": 1,
        "task_orders": "81",
        "partition_count": 8,
        "image_env": "CORE88_BIGCODEBENCH_IMAGE",
        "default_image": (
            os.environ.get("NCP_OLMO_CORE88_BIGCODEBENCH_IMAGE", "")
        ),
        "runtime_prefix": "/usr/local",
        "allow_experimental": "1",
    },
    {
        "index": 2,
        "task_orders": "84",
        "partition_count": 8,
        "image_env": "CORE88_DS1000_IMAGE",
        "default_image": (
            os.environ.get("NCP_OLMO_CORE88_DS1000_IMAGE", "")
        ),
        "runtime_prefix": "/opt/core88/ds1000/python",
        "allow_experimental": "1",
    },
    {
        "index": 3,
        "task_orders": "86,87",
        # Passing MultiPL-E candidates execute substantially more of each
        # language test suite than early-failing candidates.  Keep the other
        # scorer families at eight shards, but fan this long-tail family out
        # fourfold so stronger models do not serialize the final Core88 score.
        "partition_count": 32,
        "image_env": "CORE88_MULTIPLE_IMAGE",
        "default_image": (
            os.environ.get("NCP_OLMO_CORE88_MULTIPLE_IMAGE", "")
        ),
        "runtime_prefix": "/usr",
        "allow_experimental": "1",
    },
)
CORE88_LEGACY_SCORING_JOB_COUNT = 32
CORE88_SCORING_CONTRACT_VERSION = "core88-score-snapshot-sharding-v3"


class EvaluationError(RuntimeError):
    """A user-facing evaluation workflow failure."""


@dataclasses.dataclass(frozen=True)
class CommandSpec:
    """One reproducible command before it is materialized as a task spec."""

    argv: tuple[str, ...]
    env: dict[str, str]
    resources: Resources
    output_root: Path
    container_image: str = ""

    def as_json(self) -> dict[str, Any]:
        """Return a stable JSON representation."""

        return {
            "argv": list(self.argv),
            "env": dict(sorted(self.env.items())),
            "resources": self.resources.as_json(),
            "output_root": str(self.output_root),
            "container_image": self.container_image,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"无法读取 JSON：{path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON 顶层必须是对象：{path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def _locked(root: Path) -> Iterable[None]:
    lock_path = root / ".evaluation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_evaluation_root(root: Path) -> Path:
    """Create or validate a writable evaluation root."""

    try:
        root.expanduser().mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise EvaluationError(f"无法创建测评根目录：{root}: {error}") from error
    if not os.access(root, os.W_OK):
        raise EvaluationError(f"测评根目录不可写：{root}")
    return root.resolve()


def _normal_checkpoint_path(checkpoint: Path) -> Path:
    path = checkpoint.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise EvaluationError(f"checkpoint 目录不存在：{path}")
    return path


def _weight_files_from_index(checkpoint: Path, index: Path) -> list[Path]:
    payload = _read_json(index)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise EvaluationError(f"权重索引缺少 weight_map：{index}")
    names = sorted({str(value) for value in weight_map.values()})
    files = [checkpoint / name for name in names]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise EvaluationError(f"权重索引引用了缺失分片：{missing[:8]}")
    return files


def _infer_megatron_checkpoint(checkpoint: Path) -> tuple[Path, int, Path]:
    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    if match:
        step = int(match.group(1))
        return checkpoint.parent, step, checkpoint
    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text(encoding="utf-8").strip()
        if not value.isdigit() or int(value) <= 0:
            raise EvaluationError(f"Megatron tracker 非法：{tracker}")
        step = int(value)
        candidates = (checkpoint / f"iter_{step:07d}", checkpoint / f"iter_{step}")
        iteration = next((path for path in candidates if path.is_dir()), None)
        if iteration is None:
            raise EvaluationError(f"tracker 指向的迭代目录不存在：step={step}")
        return checkpoint, step, iteration
    raise EvaluationError(
        "Megatron checkpoint 需传入 iter_<step> 目录，或包含 "
        "latest_checkpointed_iteration.txt 的 checkpoint 根目录"
    )


def validate_checkpoint(checkpoint: Path, backend: str) -> dict[str, Any]:
    """Validate model files without loading or modifying model weights."""

    path = _normal_checkpoint_path(checkpoint)
    if backend == "megatron":
        root, step, iteration = _infer_megatron_checkpoint(path)
        evidence: list[str] = []
        inspected_files = 0
        for directory, subdirectories, filenames in os.walk(iteration):
            subdirectories.sort()
            filenames.sort()
            relative_depth = len(Path(directory).relative_to(iteration).parts)
            if relative_depth >= 2:
                subdirectories.clear()
            for filename in filenames:
                inspected_files += 1
                candidate = Path(directory) / filename
                if candidate.suffix in {".pt", ".distcp"} or filename in {
                    "metadata.json",
                    ".metadata",
                }:
                    evidence.append(str(candidate.relative_to(iteration)))
                if inspected_files >= 4096:
                    subdirectories.clear()
                    break
            if inspected_files >= 4096:
                break
        if inspected_files == 0:
            raise EvaluationError(f"Megatron 迭代目录为空：{iteration}")
        if not evidence:
            raise EvaluationError(
                f"Megatron checkpoint 未发现 .metadata/.distcp/.pt 权重证据：{iteration}"
            )
        return {
            "status": "CHECKPOINT_FILES_OK",
            "checkpoint_path": str(path),
            "checkpoint_root": str(root.resolve()),
            "ckpt_step": step,
            "iteration_path": str(iteration.resolve()),
            "evidence_files": sorted(evidence)[:64],
            "bounded_files_inspected": inspected_files,
        }

    config_path = path / "config.json"
    if not config_path.is_file():
        raise EvaluationError(f"HF 兼容模型缺少 config.json：{path}")
    config = _read_json(config_path)
    index_candidates = (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    )
    index = next((candidate for candidate in index_candidates if candidate.is_file()), None)
    if index is not None:
        weight_files = _weight_files_from_index(path, index)
    else:
        weight_files = [
            candidate
            for candidate in (path / "model.safetensors", path / "pytorch_model.bin")
            if candidate.is_file()
        ]
    if not weight_files:
        raise EvaluationError(f"HF 兼容模型没有完整的 safetensors/bin 权重：{path}")
    tokenizer_files = [
        path / name
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")
        if (path / name).is_file()
    ]
    if not tokenizer_files:
        raise EvaluationError(f"HF 兼容模型缺少 tokenizer 文件：{path}")
    return {
        "status": "CHECKPOINT_FILES_OK",
        "checkpoint_path": str(path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "model_type": str(config.get("model_type", "")),
        "architectures": config.get("architectures", []),
        "declared_transformers_version": str(config.get("transformers_version", "")),
        "auto_map": config.get("auto_map") or {},
        "weight_index": str(index) if index else None,
        "weight_files": [str(candidate) for candidate in weight_files],
        "weight_sizes": {str(candidate): candidate.stat().st_size for candidate in weight_files},
        "tokenizer_files": [str(candidate) for candidate in tokenizer_files],
    }


def _model_key(checkpoint: Path, backend: str) -> str:
    digest = hashlib.sha256(str(checkpoint.resolve()).encode("utf-8")).hexdigest()[:6]
    return f"{backend}-{digest}"


def _existing_versions(root: Path, model_key: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(model_key)}-v([1-9][0-9]*)$")
    matches: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            matches.append((int(match.group(1)), path))
    return sorted(matches)


def _validate_dflash_verification(
    verification_path: Path,
    *,
    checkpoint_path: Path,
    draft_model: Path,
    draft_validation: dict[str, Any],
    allow_approximate: bool = False,
) -> dict[str, Any]:
    """Validate a target/draft-bound DFlash A/B artifact."""

    path = verification_path.expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"NCP DFlash correctness artifact 不存在：{path}")
    verification = _read_json(path)
    exact = int(verification.get("exact_token_match_count", -1))
    comparisons = int(verification.get("comparison_count", -1))
    generated = int(verification.get("generated_token_count", -1))
    vllm_version = str(verification.get("vllm_version", ""))
    verification_mode = normalize_ncp_dflash_verification_mode(
        str(verification.get("speculative_verification_mode", ""))
    )
    approximate_mode = verification_mode in NCP_DFLASH_APPROXIMATE_VERIFICATION_MODES
    if comparisons < 8 or generated < 1024:
        raise EvaluationError(
            "NCP DFlash A/B artifact 未覆盖至少 8 prompts / 1024 tokens"
        )
    if vllm_version != "0.13.0":
        raise EvaluationError(
            "NCP DFlash correctness artifact 必须来自 vLLM 0.13.0，"
            f"当前为 {vllm_version or 'missing'}"
        )
    if verification_mode not in NCP_DFLASH_VERIFICATION_MODES:
        raise EvaluationError(
            "NCP DFlash correctness artifact 缺少受支持的 verification mode"
        )
    try:
        throughput_speedup = float(
            verification.get("throughput_speedup", float("nan"))
        )
    except (TypeError, ValueError):
        throughput_speedup = float("nan")
    if verification_mode != "sequential_exact" and (
        not math.isfinite(throughput_speedup) or throughput_speedup <= 1.0
    ):
        raise EvaluationError(
            "NCP DFlash parallel artifact 必须同时证明吞吐快于 target-only"
        )
    if approximate_mode:
        if not allow_approximate:
            raise EvaluationError(
                "segmented_kv_approx 可能改变生成 token；注册时必须显式设置 "
                "--vllm-speculative-allow-approximate"
            )
        if (
            verification.get("status") != "NCP_DFLASH_VLLM_APPROXIMATE_AB_OK"
            or verification.get("speculative_output_contract") != "approximate"
            or verification.get("downstream_score_required") is not True
        ):
            raise EvaluationError(
                "segmented_kv_approx artifact 缺少 approximate A/B 合同"
            )
        validation_status = "APPROXIMATE_AB_VERIFIED"
        output_contract = "approximate"
    else:
        if (
            verification_mode not in NCP_DFLASH_EXACT_VERIFICATION_MODES
            or verification.get("status") != "NCP_DFLASH_VLLM_EXACT_MATCH_OK"
            or exact != comparisons
        ):
            raise EvaluationError(
                "NCP DFlash correctness artifact 未通过逐 token exact-match 门槛"
            )
        validation_status = "EXACT_MATCH_VERIFIED"
        output_contract = "target_exact"
    contract = verification.get("benchmark_contract")
    if not isinstance(contract, dict) or (
        int(contract.get("seed", -1)) != DEFAULT_GLOBAL_SEED
        or int(contract.get("prompt_count", -1)) < 8
        or int(contract.get("max_new_tokens", -1)) < 128
        or not 1 <= int(contract.get("batch_size", -1)) <= 8
        or int(contract.get("scheduler_queue_size", -1))
        < int(contract.get("batch_size", -1))
        or contract.get("ignore_eos") is not True
    ):
        raise EvaluationError("NCP DFlash correctness artifact 的 benchmark 合同不完整")
    raw_operating_point = verification.get("speculative_operating_point")
    if not isinstance(raw_operating_point, dict):
        raise EvaluationError(
            "NCP DFlash correctness artifact 缺少完整 speculative operating point；"
            "请使用当前版本重新生成 target/spec comparison"
        )
    try:
        operating_point = validate_dflash_operating_point(
            raw_operating_point, benchmark_contract=contract
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"NCP DFlash operating point 无效：{error}") from error
    if operating_point["speculative_verification_mode"] != verification_mode:
        raise EvaluationError("NCP DFlash operating point 的 verification mode 不匹配")
    if float(operating_point["gpu_memory_utilization"]) != float(
        contract.get("gpu_memory_utilization", -1)
    ):
        raise EvaluationError("NCP DFlash operating point 的 GPU memory contract 不匹配")
    if str(operating_point["vllm_use_v2_model_runner"]) != str(
        contract.get("vllm_use_v2_model_runner", "")
    ):
        raise EvaluationError("NCP DFlash operating point 的 vLLM model-runner contract 不匹配")
    target_identity = verification.get("target_model_identity")
    if not isinstance(target_identity, dict) or Path(
        str(target_identity.get("source_model", ""))
    ).resolve() != checkpoint_path.resolve():
        raise EvaluationError("NCP DFlash correctness artifact 与当前 target 不匹配")
    draft_identity = verification.get("draft_model_identity")
    expected_draft = {"path": str(draft_model.resolve()), **draft_validation}
    if not isinstance(draft_identity, dict) or not dflash_identity_matches(
        draft_identity, expected_draft
    ):
        raise EvaluationError("NCP DFlash correctness artifact 与当前 draft 不匹配")
    return {
        "status": validation_status,
        "path": str(path),
        "sha256": _sha256(path),
        "exact_token_match_count": exact,
        "comparison_count": comparisons,
        "generated_token_count": generated,
        "vllm_version": vllm_version,
        "speculative_verification_mode": verification_mode,
        "speculative_output_contract": output_contract,
        "downstream_score_required": approximate_mode,
        "exact_prompt_match_rate": exact / comparisons if comparisons else None,
        "throughput_speedup": (
            throughput_speedup if math.isfinite(throughput_speedup) else None
        ),
        "benchmark_contract": contract,
        "speculative_operating_point": operating_point,
    }


def register_model(
    *,
    root: Path,
    checkpoint: Path,
    backend: str,
    new_version: bool,
    tokenizer_model: Path | None = None,
    train_wandb_config: Path | None = None,
    model_config_path: Path | None = None,
    vllm_runtime_config: Path | None = None,
    vllm_speculative_draft_model: Path | None = None,
    vllm_speculative_verification: Path | None = None,
    vllm_speculative_allow_approximate: bool = False,
) -> dict[str, Any]:
    """Register one immutable checkpoint/backend pair in a versioned directory."""

    if backend not in BACKENDS:
        raise EvaluationError(f"不支持的 backend：{backend}")
    root = require_evaluation_root(root)
    checkpoint_path = _normal_checkpoint_path(checkpoint)
    validation = validate_checkpoint(checkpoint_path, backend)
    if backend == "megatron" and tokenizer_model is None:
        raise EvaluationError("Megatron 注册必须显式提供 --tokenizer-model")
    if backend == "megatron" and train_wandb_config is None:
        raise EvaluationError("Megatron 注册必须显式提供 --train-wandb-config")
    resolved_tokenizer = (tokenizer_model or checkpoint_path).expanduser().resolve()
    if not resolved_tokenizer.is_dir():
        raise EvaluationError(f"tokenizer 目录不存在：{resolved_tokenizer}")
    resolved_train_config = None
    if train_wandb_config is not None:
        resolved_train_config = train_wandb_config.expanduser().resolve()
        if not resolved_train_config.is_file():
            raise EvaluationError(f"训练配置不存在：{resolved_train_config}")
    if model_config_path is None:
        resolved_model_config = (
            resolved_tokenizer / "config.json"
            if backend == "megatron"
            else checkpoint_path / "config.json"
        )
    else:
        resolved_model_config = model_config_path.expanduser().resolve()
        if resolved_model_config.is_dir():
            resolved_model_config /= "config.json"
    if not resolved_model_config.is_file():
        raise EvaluationError(
            "模型上下文配置不存在；长上下文评测需要可读取的 config.json："
            f"{resolved_model_config}"
        )
    resolved_runtime_config = None
    if vllm_runtime_config is not None:
        resolved_runtime_config = vllm_runtime_config.expanduser().resolve()
        if not resolved_runtime_config.is_file():
            raise EvaluationError(f"vLLM runtime config 不存在：{resolved_runtime_config}")
    resolved_draft_model = None
    draft_validation: dict[str, Any] | None = None
    speculative_verification: dict[str, Any] | None = None
    if vllm_speculative_draft_model is not None:
        if backend != "vllm":
            raise EvaluationError("NCP DFlash 只支持 vllm backend 注册")
        resolved_draft_model = vllm_speculative_draft_model.expanduser().resolve()
        draft_config_path = resolved_draft_model / "config.json"
        try:
            draft_identity = dflash_checkpoint_identity(resolved_draft_model)
        except (OSError, ValueError) as error:
            raise EvaluationError(f"NCP DFlash checkpoint 不完整：{error}") from error
        draft_config = _read_json(draft_config_path)
        expected = {
            "model_type": "conceptlm_dflash",
            "proposal_method": "path_selector",
            "hlm_conditioning": "causal_residual",
            "concept_chunk_size": 4,
            "target_layer_ids": [1, 4, 7, 10, 13],
        }
        mismatched = {
            key: (draft_config.get(key), value)
            for key, value in expected.items()
            if draft_config.get(key) != value
        }
        if mismatched:
            raise EvaluationError(f"NCP DFlash 配置不符合已验证合同：{mismatched}")
        draft_validation = {
            **{key: value for key, value in draft_identity.items() if key != "path"},
            "block_size": int(draft_config["block_size"]),
            **expected,
        }
        if vllm_speculative_verification is not None:
            speculative_verification = _validate_dflash_verification(
                vllm_speculative_verification,
                checkpoint_path=checkpoint_path,
                draft_model=resolved_draft_model,
                draft_validation=draft_validation,
                allow_approximate=vllm_speculative_allow_approximate,
            )
    elif vllm_speculative_verification is not None:
        raise EvaluationError(
            "--vllm-speculative-verification 必须与 draft checkpoint 一起提供"
        )
    elif vllm_speculative_allow_approximate:
        raise EvaluationError(
            "--vllm-speculative-allow-approximate 必须与 draft checkpoint 一起提供"
        )
    key = _model_key(checkpoint_path, backend)
    with _locked(root):
        versions = _existing_versions(root, key)
        if versions and not new_version:
            names = ", ".join(path.name for _, path in versions)
            raise EvaluationError(
                f"模型已经注册：{names}。复用已有目录，或显式传 --new-version 创建新版本。"
            )
        version = versions[-1][0] + 1 if versions else 1
        name = f"{key}-v{version}"
        model_root = root / name
        model_root.mkdir()
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "MODEL_REGISTERED",
            "registration_id": uuid.uuid4().hex,
            "registration_name": name,
            "model_key": key,
            "version": version,
            "created_at": _utc_now(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_path_sha256": hashlib.sha256(
                str(checkpoint_path).encode("utf-8")
            ).hexdigest(),
            "backend": backend,
            "backend_runtime": BACKEND_NAMES[backend],
            "checkpoint_validation": validation,
            "tokenizer_model": str(resolved_tokenizer),
            "train_wandb_config": (str(resolved_train_config) if resolved_train_config else ""),
            "model_config_path": str(resolved_model_config),
            "vllm_runtime_config": (
                str(resolved_runtime_config) if resolved_runtime_config else ""
            ),
            "vllm_speculative_draft_model": (
                str(resolved_draft_model) if resolved_draft_model else ""
            ),
            "vllm_speculative_draft_validation": draft_validation,
            "vllm_speculative_verification": speculative_verification,
            "vllm_speculative_allow_approximate": bool(
                vllm_speculative_allow_approximate
            ),
            "vllm_speculative_correctness_status": (
                str(speculative_verification["status"])
                if speculative_verification is not None
                else ("UNVERIFIED" if resolved_draft_model else "NOT_APPLICABLE")
            ),
        }
        _write_json(model_root / "metadata.json", metadata)
    return metadata


def load_registration(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """Load one registered model and reject path traversal or corrupt metadata."""

    root = require_evaluation_root(root)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise EvaluationError(f"非法测评目录名称：{name}")
    model_root = (root / name).resolve()
    if model_root.parent != root or not model_root.is_dir():
        raise EvaluationError(f"测评目录不存在：{root / name}；请先注册模型。")
    metadata = _read_json(model_root / "metadata.json")
    if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise EvaluationError(f"不支持的模型 metadata：{model_root / 'metadata.json'}")
    if metadata.get("registration_name") != name:
        raise EvaluationError("模型 metadata 的目录名称不匹配")
    checkpoint_path = Path(str(metadata["checkpoint_path"]))
    if _model_key(checkpoint_path, str(metadata["backend"])) != metadata.get("model_key"):
        raise EvaluationError("模型 metadata 的 checkpoint/backend 标识不匹配")
    current_validation = validate_checkpoint(checkpoint_path, str(metadata["backend"]))
    if _canonical_hash(current_validation) != _canonical_hash(
        metadata.get("checkpoint_validation")
    ):
        raise EvaluationError("checkpoint 文件合同自注册后发生变化，请新注册一个版本")
    for field in ("tokenizer_model", "model_config_path"):
        path = Path(str(metadata.get(field, "")))
        if not path.exists():
            raise EvaluationError(f"注册 metadata 引用的路径已不存在：{field}={path}")
    for field in ("train_wandb_config", "vllm_runtime_config"):
        value = str(metadata.get(field, ""))
        if value and not Path(value).is_file():
            raise EvaluationError(f"注册 metadata 引用的文件已不存在：{field}={value}")
    draft_value = str(metadata.get("vllm_speculative_draft_model", ""))
    if draft_value:
        draft_root = Path(draft_value)
        validation = metadata.get("vllm_speculative_draft_validation")
        try:
            current = dflash_checkpoint_identity(draft_root)
        except (OSError, ValueError) as error:
            raise EvaluationError(f"注册的 NCP DFlash checkpoint 已不完整：{error}") from error
        stored_identity = (
            {"path": str(draft_root.resolve()), **validation}
            if isinstance(validation, dict)
            else {}
        )
        if not dflash_identity_matches(stored_identity, current):
            raise EvaluationError("NCP DFlash checkpoint 自注册后发生变化，请新注册版本")
        verification = metadata.get("vllm_speculative_verification")
        if verification is not None:
            if not isinstance(verification, dict):
                raise EvaluationError("NCP DFlash correctness metadata 已损坏")
            current_verification = _validate_dflash_verification(
                Path(str(verification.get("path", ""))),
                checkpoint_path=checkpoint_path,
                draft_model=draft_root,
                draft_validation=validation,
                allow_approximate=bool(
                    metadata.get("vllm_speculative_allow_approximate", False)
                ),
            )
            if _canonical_hash(current_verification) != _canonical_hash(verification):
                raise EvaluationError("NCP DFlash correctness artifact 自注册后发生变化")
    return model_root, metadata


def _git_state(repo_root: Path) -> dict[str, Any]:
    return source_state(repo_root)


def _job_name(phase: str, registration: str, benchmark: str, suffix: str) -> str:
    raw = f"{JOB_PREFIX}-{phase}-{registration}-{benchmark}-{suffix}".lower()
    raw = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    if len(raw) < 50:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
    fixed = f"{JOB_PREFIX}-{phase}-"
    tail = f"-{benchmark}-{suffix}-{digest}"
    room = 49 - len(fixed) - len(tail)
    compact_registration = registration[: max(1, room)].rstrip("-")
    return f"{fixed}{compact_registration}{tail}"


def _final_job_name(registration: str, benchmark: str, suffix: str) -> str:
    """Keep the user-requested autofinal-<benchmark>-<registration> ordering."""

    raw = f"{JOB_PREFIX}-final-{benchmark}-{registration}-{suffix}".lower()
    raw = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    if len(raw) < 50:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
    fixed = f"{JOB_PREFIX}-final-{benchmark}-"
    tail = f"-{suffix}-{digest}"
    room = 49 - len(fixed) - len(tail)
    return f"{fixed}{registration[: max(1, room)].rstrip('-')}{tail}"


def protocol_for(backend: str, benchmark: str) -> dict[str, Any]:
    """Return the unified batching/sampling contract for one combination."""

    if backend not in BACKENDS or benchmark not in BENCHMARKS:
        raise EvaluationError(f"不支持的组合：{backend}/{benchmark}")
    if benchmark in {"ruler", "helmet"}:
        batch_size = 4
        samples = 1
    else:
        batch_size = 8
        samples = 0
    protocol: dict[str, Any] = {
        "backend": backend,
        "runtime_backend": BACKEND_NAMES[backend],
        "batch_size": batch_size,
        "sample_setting": samples,
        "resume": True,
        "gpus_per_job": 8,
    }
    if benchmark == "gsm8k":
        protocol.update(
            {
                "samples_per_doc": 1,
                "generation_mode": "greedy",
                "fewshot_seed": DEFAULT_GLOBAL_SEED,
                "sampling_seed": DEFAULT_GLOBAL_SEED,
            }
        )
    elif benchmark == "sciq":
        protocol.update(
            {
                "request_type": "loglikelihood",
                "metric": "acc",
                "num_fewshot": 0,
                "samples_per_example": 1,
                "sampling": False,
                "global_seed": DEFAULT_GLOBAL_SEED,
                "machine_count": 1,
                "source_profile": SCIQ_SOURCE_PROFILE,
                "task_order": SCIQ_TASK_ORDER,
                "task_name": SCIQ_TASK_NAME,
                "example_count": SCIQ_EXAMPLE_COUNT,
                "source_file": SCIQ_SOURCE_FILE,
                "source_sha256": SCIQ_SOURCE_SHA256,
            }
        )
    elif benchmark == "core88":
        protocol.update(
            {
                "generation_samples_cap": samples,
                "sample_zero_semantics": (
                    "use_official_task_sample_count" if samples == 0 else "cap_each_task_at_one"
                ),
                "global_seed": DEFAULT_GLOBAL_SEED,
                "machine_count": 4,
                "dispatch_plan_schema": CORE88_PLAN_SCHEMA,
                "dispatch_assignment_strategy": CORE88_ASSIGNMENT_STRATEGY,
                "dispatch_cost_model": CORE88_COST_MODEL,
                "local_dispatch": CORE88_LOCAL_DISPATCH,
            }
        )
        if backend == "vllm":
            protocol["vllm_max_model_len"] = CORE88_VLLM_MAX_MODEL_LEN
    else:
        protocol.update(
            {
                "samples_per_example": 1,
                "lengths": list(LONG_CONTEXT_LENGTHS[benchmark]),
                "official_generation_defaults": True,
                "global_seed": DEFAULT_GLOBAL_SEED,
            }
        )
        if benchmark == "ruler":
            from .long_context_protocol import (
                RULER_DATA_GENERATION_CONTRACT,
                RULER_MAX_NEW_TOKENS,
                RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH,
                RULER_MAX_NEW_TOKENS_BY_TASK,
                RULER_QA_PROMPT_CONTRACT,
            )

            protocol["qa_prompt_contract"] = RULER_QA_PROMPT_CONTRACT
            protocol["data_generation_contract"] = RULER_DATA_GENERATION_CONTRACT
            protocol["tokens_to_generate"] = RULER_MAX_NEW_TOKENS
            protocol["tokens_to_generate_by_task"] = dict(RULER_MAX_NEW_TOKENS_BY_TASK)
            protocol["tokens_to_generate_by_sequence_length"] = {
                length: dict(task_budgets)
                for length, task_budgets in RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH.items()
            }
            protocol["eos_stopping"] = True
    return protocol


def _protocol_for_registration(
    registration: dict[str, Any], benchmark: str
) -> dict[str, Any]:
    """Bind speculative generation to the batch shape proven by its artifact."""

    backend = str(registration["backend"])
    protocol = protocol_for(backend, benchmark)
    draft_model = str(registration.get("vllm_speculative_draft_model", ""))
    if not draft_model:
        return protocol
    if backend != "vllm":
        raise EvaluationError("NCP DFlash speculative decoding 只支持 vLLM backend")
    if benchmark not in {"gsm8k", "core88"}:
        raise EvaluationError(
            "NCP DFlash 当前只完成 GSM8K/Core88 生成路径验证；"
            f"拒绝将该 speculative 注册用于 {benchmark}"
        )
    verification = registration.get("vllm_speculative_verification")
    contract = verification.get("benchmark_contract") if isinstance(verification, dict) else None
    if not isinstance(contract, dict):
        raise EvaluationError("NCP DFlash 注册缺少 correctness artifact benchmark 合同")
    raw_operating_point = verification.get("speculative_operating_point")
    if not isinstance(raw_operating_point, dict):
        raise EvaluationError("NCP DFlash 注册缺少封存的 speculative operating point")
    try:
        operating_point = validate_dflash_operating_point(
            raw_operating_point, benchmark_contract=contract
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"NCP DFlash 注册的 operating point 无效：{error}") from error
    verified_batch_size = int(operating_point["max_num_seqs"])
    scheduler_queue_size = int(operating_point["scheduler_queue_size"])
    if verified_batch_size <= 0:
        raise EvaluationError("NCP DFlash correctness artifact 缺少有效 batch size")
    required_model_len = int(protocol.get("vllm_max_model_len", 0))
    if int(operating_point["max_model_len"]) < required_model_len:
        raise EvaluationError(
            "NCP DFlash A/B 的 max_model_len 小于 benchmark 协议："
            f"verified={operating_point['max_model_len']} required={required_model_len}"
        )
    protocol["speculative_decoding"] = {
        "enabled": True,
        "draft_model": draft_model,
        "verification_mode": str(
            verification.get("speculative_verification_mode", "")
        ),
        "vllm_version": str(verification.get("vllm_version", "")),
        "verified_generation_batch_size": verified_batch_size,
        "verified_scheduler_queue_size": scheduler_queue_size,
        "continuous_batching_enabled": bool(
            operating_point["continuous_batching_enabled"]
        ),
        "operating_point": operating_point,
        "output_contract": str(
            verification.get("speculative_output_contract", "target_exact")
        ),
        "downstream_score_required": bool(
            verification.get("downstream_score_required", False)
        ),
        "baseline_requirement": (
            "matched_target_only_registration"
            if verification.get("downstream_score_required") is True
            else "none"
        ),
    }
    if benchmark in {"gsm8k", "core88"}:
        protocol["batch_size"] = verified_batch_size
        protocol["generation_batch_size"] = verified_batch_size
        protocol["vllm_scheduler_queue_size"] = scheduler_queue_size
        protocol["vllm_max_model_len"] = int(operating_point["max_model_len"])
        protocol["vllm_gpu_memory_utilization"] = float(
            operating_point["gpu_memory_utilization"]
        )
        protocol["vllm_execution_mode"] = str(operating_point["execution_mode"])
        protocol["vllm_attention_backend"] = str(operating_point["attention_backend"])
        protocol["vllm_flash_attn_version"] = int(
            operating_point["flash_attn_version"]
        )
        protocol["vllm_hlm_attention_impl"] = str(
            operating_point["hlm_attention_impl"]
        )
    return protocol


def _benchmark_path(model_root: Path, benchmark: str) -> Path:
    return model_root / benchmark


def _load_benchmark_state(
    path: Path, benchmark: str, registration: dict[str, Any]
) -> dict[str, Any]:
    state_path = path / "metadata.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise EvaluationError(f"不支持的 benchmark metadata：{state_path}")
        return state
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "BENCHMARK_REGISTERED",
        "benchmark": benchmark,
        "registration_id": registration["registration_id"],
        "registration_name": registration["registration_name"],
        "protocol": _protocol_for_registration(registration, benchmark),
        "created_at": _utc_now(),
        "inference_attempts": [],
        "scoring_attempts": [],
        "final_attempts": [],
    }


def _save_benchmark_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json(path / "metadata.json", state)


def _require_current_inference_protocol(
    state: dict[str, Any], registration: dict[str, Any], benchmark: str
) -> None:
    """Prevent resume from mixing predictions produced under another seed."""

    expected = _protocol_for_registration(registration, benchmark)
    actual = state.get("protocol")
    if actual == expected:
        return
    if not state.get("inference_attempts"):
        state["protocol"] = expected
        return
    raise EvaluationError(
        f"{benchmark} 已有推理使用旧协议，不能按当前 seed=42/prompt/"
        "fixed-OLMES-data/sampling 协议继续 resume；"
        "请为该模型注册新版本后重新推理。历史结果仍可按原 metadata 打分和查看。"
    )


def _prepared_lengths(manifest: dict[str, Any]) -> tuple[int, ...]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        return ()
    values = protocol.get("input_lengths") or protocol.get("sequence_lengths") or ()
    return tuple(sorted(int(value) for value in values))


def validate_prepared_data(
    data_root: Path, benchmark: str, tokenizer_model: Path
) -> dict[str, Any]:
    """Validate a sealed RULER/HELMET root and its tokenizer/length contract."""

    manifest_path = data_root.resolve() / "manifest.json"
    inputs_path = data_root.resolve() / "inputs.jsonl"
    if not manifest_path.is_file() or not inputs_path.is_file():
        raise EvaluationError(f"预处理数据缺少 manifest.json/inputs.jsonl：{data_root}")
    manifest = _read_json(manifest_path)
    if manifest.get("benchmark") != benchmark:
        raise EvaluationError(
            f"预处理数据 benchmark 不匹配：{manifest.get('benchmark')} != {benchmark}"
        )
    if benchmark == "ruler":
        from .long_context_protocol import validate_prepared_dataset

        try:
            manifest, _ = validate_prepared_dataset(data_root.resolve())
        except (OSError, ValueError) as error:
            raise EvaluationError(
                "RULER 预处理数据不符合 OLMES 协议：必须来自固定的 "
                "allenai/ruler_data data_100_samples.tgz 缓存，并使用 OLMES "
                "逐任务、逐长度生成上限；"
                f"详情：{error}"
            ) from error
        protocol = manifest.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("formal_data_validation_passed") is not True
        ):
            raise EvaluationError(
                f"{benchmark.upper()} 统一评测只接受完整官方 profile；"
                "非完整数据只能用于单独 smoke"
            )
    elif benchmark == "helmet":
        from .long_context_protocol import validate_prepared_helmet_manifest_contract

        try:
            validate_prepared_helmet_manifest_contract(manifest)
        except (OSError, ValueError, TypeError) as error:
            raise EvaluationError(
                "HELMET 预处理数据不符合已选官方 profile：必须使用该 profile "
                "封存的逐长度任务名、生成预算和停止条件；"
                f"详情：{error}"
            ) from error
        protocol = manifest.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("formal_data_validation_passed") is not True
        ):
            raise EvaluationError(
                "HELMET 统一评测只接受完整官方 profile；非完整数据只能用于单独 smoke"
            )
    actual_lengths = _prepared_lengths(manifest)
    expected_lengths = LONG_CONTEXT_LENGTHS[benchmark]
    if actual_lengths != expected_lengths:
        raise EvaluationError(
            f"{benchmark} 需要精确长度 {expected_lengths}，预处理数据为 {actual_lengths}"
        )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not tokenizer.get("contract_sha256"):
        raise EvaluationError("预处理 manifest 缺少 tokenizer contract")
    from .long_context_protocol import tokenizer_artifact_contract

    runtime_contract = tokenizer_artifact_contract(tokenizer_model.resolve())
    if runtime_contract["contract_sha256"] != tokenizer["contract_sha256"]:
        raise EvaluationError("预处理数据 tokenizer 与注册模型不匹配")
    return {
        "path": str(data_root.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "inputs_sha256": str(manifest.get("inputs_jsonl_sha256", "")),
        "example_count": int(manifest.get("example_count", -1)),
        "tokenizer_contract_sha256": tokenizer["contract_sha256"],
        "lengths": list(actual_lengths),
    }


def discover_prepared_data(benchmark: str, tokenizer_model: Path) -> Path | None:
    """Find the newest exact tokenizer/length-compatible prepared root."""

    if not PREPARED_DATA_ROOT.is_dir():
        return None
    candidates: list[Path] = []
    for manifest_path in PREPARED_DATA_ROOT.glob("*/manifest.json"):
        try:
            validate_prepared_data(manifest_path.parent, benchmark, tokenizer_model)
        except (EvaluationError, OSError, ValueError):
            continue
        candidates.append(manifest_path.parent)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _base_launch_env(registration: dict[str, Any]) -> dict[str, str]:
    source = source_state(_repo_root())
    env = {
        "PYTHONHASHSEED": str(DEFAULT_GLOBAL_SEED),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "NCP_OLMO_SOURCE_REVISION": str(source["repo_commit"]),
    }
    configured = {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
        "HELMET_AUTOAIS_MODEL": os.environ.get("HELMET_AUTOAIS_MODEL", ""),
        "NLTK_DATA": os.environ.get("NLTK_DATA", ""),
    }
    env.update({key: value for key, value in configured.items() if value})
    draft_model = str(registration.get("vllm_speculative_draft_model", ""))
    if draft_model:
        verification_status = str(
            registration.get("vllm_speculative_correctness_status", "")
        )
        if verification_status not in {
            "EXACT_MATCH_VERIFIED",
            "APPROXIMATE_AB_VERIFIED",
        }:
            raise EvaluationError(
                "NCP DFlash 已注册但尚未通过当前 target/draft 绑定的 "
                "8 prompts / 1024 tokens A/B 门槛；统一入口拒绝启用 speculative"
            )
        verification = registration.get("vllm_speculative_verification")
        if not isinstance(verification, dict):
            raise EvaluationError("NCP DFlash 注册缺少 correctness artifact 元数据")
        verification_mode = str(
            verification.get("speculative_verification_mode", "")
        )
        if verification_mode not in NCP_DFLASH_VERIFICATION_MODES:
            raise EvaluationError("NCP DFlash 注册缺少受支持的 verification mode")
        if (
            verification_mode in NCP_DFLASH_APPROXIMATE_VERIFICATION_MODES
            and verification_status != "APPROXIMATE_AB_VERIFIED"
        ):
            raise EvaluationError("近似 DFlash mode 与注册 A/B 状态不一致")
        env.update(
            {
                "VLLM_SPECULATIVE_OUTPUT_CONTRACT": str(
                    verification.get(
                        "speculative_output_contract", "target_exact"
                    )
                ),
            }
        )
        raw_operating_point = verification.get("speculative_operating_point")
        if not isinstance(raw_operating_point, dict):
            raise EvaluationError("NCP DFlash 注册缺少封存的 speculative operating point")
        try:
            env.update(dflash_operating_point_env(raw_operating_point))
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"NCP DFlash operating point 无法生成环境：{error}") from error
    return env


def _speculative_cli_args(
    registration: dict[str, Any],
    *,
    telemetry_path: Path,
    option_prefix: str,
) -> list[str]:
    """Return the sealed draft arguments for one emitted inference task."""

    draft_model = str(registration.get("vllm_speculative_draft_model", ""))
    if not draft_model:
        return []
    verification = registration.get("vllm_speculative_verification")
    if not isinstance(verification, dict):
        raise EvaluationError("NCP DFlash 注册缺少 correctness artifact 元数据")
    raw_operating_point = verification.get("speculative_operating_point")
    if not isinstance(raw_operating_point, dict):
        raise EvaluationError("NCP DFlash 注册缺少封存的 speculative operating point")
    try:
        operating_point = validate_dflash_operating_point(raw_operating_point)
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"NCP DFlash operating point 无效：{error}") from error
    return [
        f"--{option_prefix}speculative-draft-model",
        draft_model,
        f"--{option_prefix}speculative-num-tokens",
        str(operating_point["speculative_num_tokens"]),
        f"--{option_prefix}speculative-telemetry-path",
        str(telemetry_path),
        f"--{option_prefix}speculative-verification-mode",
        str(verification["speculative_verification_mode"]),
    ]


def _model_family(registration: dict[str, Any]) -> str:
    model_type = str(registration["checkpoint_validation"].get("model_type", ""))
    return "auto" if model_type in {"olmo", "olmo2", "olmo3"} else "conceptlm"


def _hf_runtime_backend(registration: dict[str, Any]) -> str:
    """Select the legacy bridge only for Megatron-backed HF artifacts.

    Pure-HF remote-code artifacts such as ``ncp_olmo3`` must use the ordinary
    Transformers loader.  The legacy ConceptLM artifact instead requires the
    repository/Megatron bridge and its ConceptLM-specific load kwargs.  Keep
    this decision aligned with ``core_native_eval._resolve_hf_backend`` while
    avoiding heavyweight evaluator imports in the orchestration CLI.
    """

    validation = registration["checkpoint_validation"]
    model_type = str(validation.get("model_type", ""))
    architecture = " ".join(str(value) for value in validation.get("architectures", []))
    if model_type == "conceptlm_v22_vq" or "ConceptLMV22VQ" in architecture:
        return "from_pretrained"
    return "transformers"


def _gsm8k_command(
    registration: dict[str, Any], output_dir: Path, job_name: str
) -> CommandSpec:
    protocol = _protocol_for_registration(registration, "gsm8k")
    argv = [
        PYTHON_BIN,
        "-m",
        "ncp_olmo_eval.portable_tasks",
        "gsm8k",
        "--model",
        str(registration["checkpoint_path"]),
        "--output-root",
        str(output_dir),
        "--standard-input-config",
        str(STANDARD_GSM8K_INPUT),
        "--task-include-path",
        str(GSM8K_TASK_INCLUDE),
        "--dataset-test-file",
        str(GSM8K_TEST_FILE),
        "--model-family",
        _model_family(registration),
        "--gpus",
        "8",
        "--fewshot-seed",
        str(protocol["fewshot_seed"]),
        "--sampling-seed",
        str(protocol["sampling_seed"]),
        "--batch-size",
        str(protocol.get("generation_batch_size", protocol["batch_size"])),
        "--scheduler-queue-size",
        str(
            protocol.get(
                "vllm_scheduler_queue_size",
                protocol.get("generation_batch_size", protocol["batch_size"]),
            )
        ),
        "--max-model-len",
        str(protocol.get("vllm_max_model_len", 2048)),
        "--gpu-memory-utilization",
        str(protocol.get("vllm_gpu_memory_utilization", 0.85)),
        "--execution-mode",
        str(protocol.get("vllm_execution_mode", "eager")),
        "--attention-backend",
        str(protocol.get("vllm_attention_backend", "FLASH_ATTN")),
        "--flash-attn-version",
        str(protocol.get("vllm_flash_attn_version", 3)),
        "--hlm-attention-impl",
        str(protocol.get("vllm_hlm_attention_impl", "legacy_mixed")),
        "--resume",
    ]
    if registration.get("vllm_runtime_config"):
        argv.extend(["--runtime-config", str(registration["vllm_runtime_config"])])
    if registration.get("vllm_speculative_draft_model"):
        verification = registration["vllm_speculative_verification"]
        operating_point = verification["speculative_operating_point"]
        argv.extend(
            [
                "--speculative-draft-model",
                str(registration["vllm_speculative_draft_model"]),
                "--speculative-num-tokens",
                str(operating_point["speculative_num_tokens"]),
                "--speculative-verification-mode",
                str(verification["speculative_verification_mode"]),
            ]
        )
    return CommandSpec(
        tuple(argv),
        _base_launch_env(registration),
        Resources(gpus=8, cpus=64, memory_gib=256),
        output_dir,
        RUNTIME_IMAGE,
    )


def _validate_sciq_source_contract() -> dict[str, Any]:
    """Validate the exact OLMo SciQ export before materializing a formal plan."""

    from .core_native_eval import _load_manifest

    _, selected = _load_manifest(CORE_DATA_ROOT, SCIQ_SOURCE_PROFILE, {SCIQ_TASK_ORDER})
    if len(selected) != 1:
        raise EvaluationError(f"SciQ source selection returned {len(selected)} tasks")
    task = selected[0]
    expected = {
        "task_order": SCIQ_TASK_ORDER,
        "task": SCIQ_TASK_NAME,
        "file": SCIQ_SOURCE_FILE,
        "sha256": SCIQ_SOURCE_SHA256,
        "num_examples": SCIQ_EXAMPLE_COUNT,
        "metric": "acc",
        "request_type": "loglikelihood",
    }
    for field, value in expected.items():
        if task.get(field) != value:
            raise EvaluationError(
                f"SciQ source contract mismatch: {field}={task.get(field)!r} != {value!r}"
            )
    source = CORE_DATA_ROOT / SCIQ_SOURCE_FILE
    if _sha256(source) != SCIQ_SOURCE_SHA256:
        raise EvaluationError(f"SciQ source SHA256 mismatch: {source}")
    return task


def _sciq_command(
    registration: dict[str, Any], attempt_root: Path, job_name: str
) -> tuple[CommandSpec, list[str]]:
    """Build one scheduler-neutral eight-GPU SciQ likelihood task."""

    protocol = protocol_for(str(registration["backend"]), "sciq")
    plan_root = attempt_root / "dispatch-plan"
    repo_commit = str(_git_state(_repo_root())["repo_commit"])
    argv = [
        PYTHON_BIN,
        "-m",
        "ncp_olmo_eval.core_native_pool",
        "--data-root",
        str(CORE_DATA_ROOT),
        "--profile",
        SCIQ_SOURCE_PROFILE,
        "--plan-root",
        str(plan_root),
        "--machine-index",
        "0",
        "--machine-count",
        "1",
        "--global-seed",
        str(DEFAULT_GLOBAL_SEED),
        "--workflow-id",
        str(registration["registration_id"]),
        "--repo-commit",
        repo_commit,
        "--hf-model-path",
        str(registration["checkpoint_path"]),
        "--model-identity-path",
        str(registration["checkpoint_path"]),
        "--tokenizer-model",
        str(registration["tokenizer_model"]),
        "--hf-backend",
        "native_vllm",
        "--output-root",
        str(attempt_root),
        "--model-label",
        str(registration["registration_name"]),
        "--gpus",
        "8",
        "--processes-per-gpu",
        "1",
        "--worker-restarts",
        "1",
        "--worker-start-stagger-seconds",
        "2",
        "--seq-length",
        "2048",
        "--score-batch-size",
        str(protocol["batch_size"]),
        "--generation-batch-size",
        str(protocol["batch_size"]),
        "--row-chunk-size",
        str(protocol["batch_size"]),
        "--pad-multiple",
        "128",
        "--progress-every",
        "100",
        "--limit-per-task",
        "0",
        "--generation-samples-cap",
        "1",
        "--max-gen-tokens-cap",
        "0",
        "--vllm-max-model-len",
        str(CORE88_VLLM_MAX_MODEL_LEN),
        "--vllm-model-family",
        _model_family(registration),
        "--vllm-model-overlay-dir",
        str(attempt_root / "model-overlay-m0"),
        "--vllm-gpu-memory-utilization",
        "0.85",
        "--vllm-execution-mode",
        "eager",
        "--vllm-attention-backend",
        "FLASH_ATTN",
        "--vllm-flash-attn-version",
        "3",
        "--vllm-hlm-attention-impl",
        "legacy_mixed",
        "--allow-unverified-native-vllm",
        "--no-hf-align-dcp-runtime-config",
        "--verify-data-sha256",
        "--resume",
    ]
    if registration.get("vllm_runtime_config"):
        argv.extend(["--vllm-runtime-config", str(registration["vllm_runtime_config"])])
    plan_command = [
        (
            str(Path(os.environ.get("PLAN_ENV_PREFIX", "")) / "bin/python")
            if os.environ.get("PLAN_ENV_PREFIX")
            else sys.executable
        ),
        "-m",
        "ncp_olmo_eval.core_native_plan",
        "--data-root",
        str(CORE_DATA_ROOT),
        "--profile",
        SCIQ_SOURCE_PROFILE,
        "--plan-root",
        str(plan_root),
        "--task-orders",
        str(SCIQ_TASK_ORDER),
        "--limit-per-task",
        "0",
        "--generation-samples-cap",
        "1",
        "--max-gen-tokens-cap",
        "0",
        "--machine-count",
        "1",
        "--global-seed",
        str(DEFAULT_GLOBAL_SEED),
        "--decode-weight",
        "8",
    ]
    return (
        CommandSpec(
            tuple(argv),
            _base_launch_env(registration),
            Resources(gpus=8, cpus=96, memory_gib=512),
            attempt_root / "machine-00",
            RUNTIME_IMAGE,
        ),
        plan_command,
    )


def _core_commands(
    registration: dict[str, Any],
    attempt_root: Path,
    job_names: Sequence[str],
    workflow: dict[str, Any] | None = None,
) -> tuple[list[CommandSpec], list[str]]:
    if len(job_names) != 5:
        raise EvaluationError("Core88 需要四个主任务和一个 GSM8K 任务")
    backend = str(registration["backend"])
    protocol = _protocol_for_registration(registration, "core88")
    core_root = attempt_root
    plan_root = core_root / "dispatch-plan"
    if backend != "vllm":
        raise EvaluationError("发布版统一入口只支持 vLLM")
    repo_commit = str(_git_state(_repo_root())["repo_commit"])
    workflow_id = str(workflow["workflow_id"] if workflow else registration["registration_id"])
    commands: list[CommandSpec] = []
    for machine_index in range(4):
        argv = [
            PYTHON_BIN, "-m", "ncp_olmo_eval.core_native_pool",
            "--data-root", str(CORE_DATA_ROOT),
            "--profile", "core88",
            "--plan-root", str(plan_root),
            "--machine-index", str(machine_index),
            "--machine-count", "4",
            "--global-seed", str(DEFAULT_GLOBAL_SEED),
            "--workflow-id", workflow_id,
            "--repo-commit", repo_commit,
            "--hf-model-path", str(registration["checkpoint_path"]),
            "--model-identity-path", str(registration["checkpoint_path"]),
            "--tokenizer-model", str(registration["tokenizer_model"]),
            "--hf-backend", "native_vllm",
            "--output-root", str(core_root),
            "--model-label", str(registration["registration_name"]),
            "--gpus", "8",
            "--processes-per-gpu", "1",
            "--worker-restarts", "1",
            "--worker-start-stagger-seconds", "2",
            "--seq-length", "2048",
            "--score-batch-size", str(protocol["batch_size"]),
            "--generation-batch-size",
            str(protocol.get("generation_batch_size", protocol["batch_size"])),
            "--row-chunk-size", str(protocol["batch_size"]),
            "--pad-multiple", "128",
            "--progress-every", "100",
            "--limit-per-task", "0",
            "--generation-samples-cap", str(protocol["generation_samples_cap"]),
            "--max-gen-tokens-cap", "0",
            "--vllm-max-model-len", str(protocol["vllm_max_model_len"]),
            "--vllm-scheduler-queue-size",
            str(
                protocol.get(
                    "vllm_scheduler_queue_size",
                    protocol.get("generation_batch_size", protocol["batch_size"]),
                )
            ),
            "--vllm-model-family", _model_family(registration),
            "--vllm-model-overlay-dir", str(core_root / f"model-overlay-m{machine_index}"),
            "--vllm-gpu-memory-utilization",
            str(protocol.get("vllm_gpu_memory_utilization", 0.85)),
            "--vllm-execution-mode",
            str(protocol.get("vllm_execution_mode", "eager")),
            "--vllm-attention-backend",
            str(protocol.get("vllm_attention_backend", "FLASH_ATTN")),
            "--vllm-flash-attn-version",
            str(protocol.get("vllm_flash_attn_version", 3)),
            "--vllm-hlm-attention-impl",
            str(protocol.get("vllm_hlm_attention_impl", "legacy_mixed")),
            "--allow-unverified-native-vllm",
            "--no-hf-align-dcp-runtime-config",
            "--verify-data-sha256",
            "--resume",
        ]
        if registration.get("vllm_runtime_config"):
            argv.extend(["--vllm-runtime-config", str(registration["vllm_runtime_config"])])
        argv.extend(
            _speculative_cli_args(
                registration,
                telemetry_path=core_root
                / f"machine-{machine_index:02d}"
                / "ncp-dflash-telemetry.jsonl",
                option_prefix="vllm-",
            )
        )
        commands.append(
            CommandSpec(
                tuple(argv),
                _base_launch_env(registration),
                Resources(gpus=8, cpus=96, memory_gib=512),
                core_root / f"machine-{machine_index:02d}",
                RUNTIME_IMAGE,
            )
        )
    gsm8k_root = attempt_root / "standalone-gsm8k"
    gsm8k_command = _gsm8k_command(registration, gsm8k_root, job_names[4])
    if workflow is not None:
        gsm8k_command = dataclasses.replace(
            gsm8k_command,
            argv=(*gsm8k_command.argv, "--workflow-manifest", str(attempt_root / "workflow.json"), "--repo-commit", str(workflow["repo_commit"])),
        )
    commands.append(gsm8k_command)
    plan_command = [
        (
            str(Path(os.environ.get("PLAN_ENV_PREFIX", "")) / "bin/python")
            if os.environ.get("PLAN_ENV_PREFIX")
            else sys.executable
        ),
        "-m",
        "ncp_olmo_eval.core_native_plan",
        "--data-root",
        str(CORE_DATA_ROOT),
        "--profile",
        "core88",
        "--plan-root",
        str(plan_root),
        "--task-orders",
        "",
        "--limit-per-task",
        "0",
        "--generation-samples-cap",
        str(protocol["generation_samples_cap"]),
        "--max-gen-tokens-cap",
        "0",
        "--machine-count",
        "4",
        "--global-seed",
        str(DEFAULT_GLOBAL_SEED),
        "--decode-weight",
        "8",
    ]
    return commands, plan_command


def _create_core_workflow(
    registration: dict[str, Any],
    attempt_root: Path,
    job_names: Sequence[str],
    repo_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Create the strict same-run workflow when its companion format supports it."""

    backend = str(registration["backend"])
    if backend not in {"vllm", "lmdeploy"}:
        return None
    from .core88_workflow import create_workflow

    protocol = _protocol_for_registration(registration, "core88")
    gsm8k_protocol = _protocol_for_registration(registration, "gsm8k")
    speculative = protocol.get("speculative_decoding")
    operating_point = (
        speculative.get("operating_point") if isinstance(speculative, dict) else None
    )
    return create_workflow(
        output_root=attempt_root,
        repo_root=_repo_root(),
        repo_commit=str(repo_state["repo_commit"]),
        source_kind=str(repo_state["source_kind"]),
        source_tree_sha256=str(repo_state["source_tree_sha256"]),
        hf_model_path=Path(str(registration["checkpoint_path"])),
        model_identity_path=Path(str(registration["checkpoint_path"])),
        model_label=str(registration["registration_name"]),
        run_tag=f"unified-a{attempt_root.name.split('-')[-1]}",
        core_job_names=list(job_names[:4]),
        gsm8k_job_name=str(job_names[4]),
        global_seed=DEFAULT_GLOBAL_SEED,
        core_plan_json=attempt_root / "dispatch-plan" / "plan.json",
        hf_backend="native_vllm" if backend == "vllm" else "lmdeploy",
        vllm_model_family=_model_family(registration),
        vllm_runtime_config=(
            Path(str(registration["vllm_runtime_config"]))
            if registration.get("vllm_runtime_config")
            else None
        ),
        processes_per_gpu=1,
        score_batch_size=int(protocol["batch_size"]),
        generation_batch_size=int(
            protocol.get("generation_batch_size", protocol["batch_size"])
        ),
        row_chunk_size=int(protocol["batch_size"]),
        seq_length=2048,
        vllm_max_model_len=int(protocol.get("vllm_max_model_len", 0)),
        vllm_scheduler_queue_size=int(
            protocol.get(
                "vllm_scheduler_queue_size",
                protocol.get("generation_batch_size", protocol["batch_size"]),
            )
        ),
        vllm_speculative_operating_point=(
            dict(operating_point) if isinstance(operating_point, dict) else None
        ),
        gsm8k_batch_size=int(
            gsm8k_protocol.get("generation_batch_size", gsm8k_protocol["batch_size"])
        ),
        gsm8k_scheduler_queue_size=int(
            gsm8k_protocol.get(
                "vllm_scheduler_queue_size",
                gsm8k_protocol.get(
                    "generation_batch_size", gsm8k_protocol["batch_size"]
                ),
            )
        ),
        gsm8k_max_model_len=int(gsm8k_protocol.get("vllm_max_model_len", 2048)),
        lmdeploy_cache_max_entry_count=0.8,
    )


def _long_context_command(
    registration: dict[str, Any],
    benchmark: str,
    data_root: Path,
    output_dir: Path,
    job_name: str,
) -> CommandSpec:
    if registration.get("vllm_speculative_draft_model"):
        raise EvaluationError(
            "NCP DFlash 当前只完成 Core88/GSM8K 短上下文正确性验证；"
            f"拒绝将该 speculative 注册用于 {benchmark}"
        )
    argv = [
        PYTHON_BIN, "-m", "ncp_olmo_eval.portable_tasks", "long-context",
        "--data-root", str(data_root),
        "--output-root", str(output_dir),
        "--model", str(registration["checkpoint_path"]),
        "--model-config", str(registration["model_config_path"]),
        "--tokenizer", str(registration["tokenizer_model"]),
        "--model-label", str(registration["registration_name"]),
        "--model-family", _model_family(registration),
        "--gpus", "8",
        "--global-seed", str(DEFAULT_GLOBAL_SEED),
        "--resume",
    ]
    if registration.get("vllm_runtime_config"):
        argv.extend(["--runtime-config", str(registration["vllm_runtime_config"])])
    return CommandSpec(
        tuple(argv),
        _base_launch_env(registration),
        Resources(gpus=8, cpus=96, memory_gib=512),
        output_dir,
        RUNTIME_IMAGE,
    )


def _run_command(
    command: CommandSpec,
    *,
    task_id: str,
    phase: str,
    benchmark: str,
    role: str,
    task_root: Path,
    executor: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Materialize a task and optionally execute it in the current allocation."""

    if executor not in {"emit", "local"}:
        raise EvaluationError(f"不支持的 executor：{executor}")
    task_path = task_root / "task.json"
    status_path = task_root / "status.json"
    log_path = task_root / "task.log"
    spec = TaskSpec(
        task_id=task_id,
        phase=phase,
        benchmark=benchmark,
        role=role,
        argv=command.argv,
        cwd=str(_repo_root().resolve()),
        env=command.env,
        resources=command.resources,
        output_root=str(command.output_root.resolve()),
        status_path=str(status_path.resolve()),
        log_path=str(log_path.resolve()),
        container_image=command.container_image,
    )
    if dry_run:
        return {"status": "DryRun", "task": spec.as_json(), "task_path": str(task_path)}
    task_root.mkdir(parents=True, exist_ok=True)
    write_task(task_path, spec)
    write_status(status_path, spec=spec, state="Planned")
    if executor == "local":
        from .task_runner import run_task

        status = run_task(task_path)
    else:
        status = read_status(status_path)
    return {
        "status": status["state"],
        "task_path": str(task_path),
        "status_path": str(status_path),
        "task": spec.as_json(),
    }


def _latest_attempt(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    attempts = state.get(key)
    return attempts[-1] if isinstance(attempts, list) and attempts else None


def _refresh_jobs(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    refreshed = []
    for job in attempt.get("jobs", []):
        item = dict(job)
        status_path = Path(str(item.get("status_path", "")))
        if status_path.is_file():
            try:
                status = read_status(status_path)
                item["status_record"] = status
                item["status"] = status["state"]
            except (OSError, ValueError, json.JSONDecodeError) as error:
                item["status_record"] = {"state": "Unreadable", "detail": str(error)}
        refreshed.append(item)
    attempt["jobs"] = refreshed
    attempt["refreshed_at"] = _utc_now()
    return refreshed


def _latest_jobs_by_role(attempt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the newest emitted task for every logical role."""

    latest: dict[str, dict[str, Any]] = {}
    for job in attempt.get("jobs", []):
        latest[str(job["role"])] = job
    return latest


def _expected_roles(benchmark: str) -> list[str]:
    if benchmark == "core88":
        return ["core88-m0", "core88-m1", "core88-m2", "core88-m3", "gsm8k-m4"]
    return [benchmark]


def _submission_candidates(
    latest: dict[str, Any] | None, expected_roles: Sequence[str], force: bool
) -> set[str] | None:
    if latest is None:
        return None
    _refresh_jobs(latest)
    jobs = _latest_jobs_by_role(latest)
    if force:
        return set(expected_roles)
    return {
        role
        for role in expected_roles
        if role not in jobs or str(jobs[role].get("status")) in RETRYABLE_STATES
    }


def submit_inference(
    *,
    root: Path,
    registration_name: str,
    benchmark: str,
    data_root: Path | None,
    executor: str,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Emit or locally run only missing/retryable inference tasks."""

    if benchmark not in BENCHMARKS:
        raise EvaluationError(f"不支持的 benchmark：{benchmark}")
    model_root, registration = load_registration(root, registration_name)
    repo_state = _git_state(_repo_root())
    if repo_state["repo_dirty"] and not dry_run:
        raise EvaluationError("统一评测提交要求当前 Git worktree 完全干净")
    benchmark_root = _benchmark_path(model_root, benchmark)
    if not dry_run:
        benchmark_root.mkdir(exist_ok=True)
    with _locked(model_root):
        state = _load_benchmark_state(benchmark_root, benchmark, registration)
        _require_current_inference_protocol(state, registration, benchmark)
        if benchmark in LONG_CONTEXT_LENGTHS:
            selected_root = data_root
            if selected_root is None and state.get("prepared_data"):
                selected_root = Path(str(state["prepared_data"]["path"]))
            if selected_root is None:
                selected_root = discover_prepared_data(
                    benchmark, Path(str(registration["tokenizer_model"]))
                )
            if selected_root is None:
                raise EvaluationError(
                    f"找不到与 tokenizer 匹配的 {benchmark} 预处理数据；请传 --data-root"
                )
            state["prepared_data"] = validate_prepared_data(
                selected_root, benchmark, Path(str(registration["tokenizer_model"]))
            )
        latest = _latest_attempt(state, "inference_attempts")
        roles = _expected_roles(benchmark)
        candidates = _submission_candidates(latest, roles, force)
        if latest is not None and candidates == set():
            if not dry_run:
                _save_benchmark_state(benchmark_root, state)
            return {
                "submitted": [],
                "jobs": list(_latest_jobs_by_role(latest).values()),
                "attempt": latest["attempt"],
                "task_plan": latest.get("task_plan"),
            }
        if latest is None:
            attempt_number = len(state["inference_attempts"]) + 1
            attempt_root = benchmark_root / "inference" / f"attempt-{attempt_number:04d}"
            attempt = {
                "attempt": attempt_number,
                "attempt_root": str(attempt_root),
                "created_at": _utc_now(),
                "repo_state": repo_state,
                "jobs": [],
                "submission_rounds": [],
            }
            submission_round = 1
        else:
            attempt = latest
            attempt_number = int(attempt["attempt"])
            attempt_root = Path(str(attempt["attempt_root"]))
            rounds = attempt.setdefault("submission_rounds", [])
            submission_round = len(rounds) + 1
        if not dry_run:
            attempt_root.mkdir(parents=True, exist_ok=True)
        suffixes = ["m0"] if benchmark != "core88" else ["m0", "m1", "m2", "m3", "m4"]
        names = [
            _job_name(
                "pred",
                registration_name,
                benchmark,
                f"{suffix}-a{attempt_number}r{submission_round}",
            )
            for suffix in suffixes
        ]
        workflow: dict[str, Any] | None = None
        if benchmark == "core88":
            commands, plan_command = _core_commands(registration, attempt_root, names)
            plan_root = attempt_root / "dispatch-plan"
            workflow_path = attempt_root / "workflow.json"
            if workflow_path.is_file():
                from .core88_workflow import load_workflow

                workflow = load_workflow(workflow_path)
            elif not dry_run:
                plan_root.mkdir(parents=True, exist_ok=True)
                assert plan_command is not None
                env = os.environ.copy()
                env["PYTHONPATH"] = str(_repo_root())
                result = subprocess.run(
                    plan_command, check=False, capture_output=True, text=True, env=env
                )
                if result.returncode:
                    raise EvaluationError(
                        "Core88 分配计划生成失败：\n"
                        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
                    )
                workflow = _create_core_workflow(registration, attempt_root, names, repo_state)
            commands, _ = _core_commands(registration, attempt_root, names, workflow=workflow)
        elif benchmark == "gsm8k":
            commands = [_gsm8k_command(registration, attempt_root, names[0])]
        elif benchmark == "sciq":
            sciq_command, plan_command = _sciq_command(
                registration, attempt_root, names[0]
            )
            plan_root = attempt_root / "dispatch-plan"
            if not dry_run and not (plan_root / "plan.json").is_file():
                _validate_sciq_source_contract()
                plan_root.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(_repo_root())
                result = subprocess.run(
                    plan_command, check=False, capture_output=True, text=True, env=env
                )
                if result.returncode:
                    raise EvaluationError(
                        "SciQ 分配计划生成失败：\n"
                        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
                    )
                plan = _read_json(plan_root / "plan.json")
                if (
                    plan.get("profile") != SCIQ_SOURCE_PROFILE
                    or plan.get("task_orders") != [SCIQ_TASK_ORDER]
                    or int(plan.get("machine_count", -1)) != 1
                    or int(plan.get("global_seed", -1)) != DEFAULT_GLOBAL_SEED
                    or int((plan.get("expected_counts") or {}).get(SCIQ_TASK_NAME, -1))
                    != SCIQ_EXAMPLE_COUNT
                ):
                    raise EvaluationError(f"SciQ 分配计划协议不匹配：{plan}")
            commands = [sciq_command]
        else:
            commands = [
                _long_context_command(
                    registration,
                    benchmark,
                    Path(str(state["prepared_data"]["path"])),
                    attempt_root,
                    names[0],
                )
            ]
        selected_indices = [
            index for index, role in enumerate(roles) if candidates is None or role in candidates
        ]
        if not selected_indices:
            if not dry_run:
                _save_benchmark_state(benchmark_root, state)
            return {
                "submitted": [],
                "jobs": list(_latest_jobs_by_role(attempt).values()),
                "attempt": attempt_number,
            }
        if dry_run:
            new_jobs = []
            submitted = []
            for index in selected_indices:
                role = roles[index]
                name = names[index]
                command = commands[index]
                outcome = _run_command(
                    command,
                    task_id=name,
                    phase="inference",
                    benchmark=benchmark,
                    role=role,
                    task_root=attempt_root / "tasks" / name,
                    executor=executor,
                    dry_run=True,
                )
                submitted.append(name)
                new_jobs.append(
                    {
                        "role": role,
                        "name": name,
                        "status": outcome["status"],
                        "output_root": str(command.output_root),
                        "command": command.as_json(),
                        "submission_round": submission_round,
                    }
                )
            return {
                "submitted": submitted,
                "jobs": new_jobs,
                "attempt": attempt_number,
                "dry_run": True,
            }
        if latest is None:
            state["inference_attempts"].append(attempt)
        round_state = {
            "round": submission_round,
            "created_at": _utc_now(),
            "roles": [roles[index] for index in selected_indices],
            "planned_job_names": [names[index] for index in selected_indices],
            "submitted_job_names": [],
            "status": "Submitting",
        }
        attempt["submission_rounds"].append(round_state)
        _save_benchmark_state(benchmark_root, state)
        submitted = []
        task_paths: list[Path] = []
        try:
            for index in selected_indices:
                role = roles[index]
                name = names[index]
                command = commands[index]
                task_root = attempt_root / "tasks" / name
                outcome = _run_command(
                    command,
                    task_id=name,
                    phase="inference",
                    benchmark=benchmark,
                    role=role,
                    task_root=task_root,
                    executor=executor,
                    dry_run=False,
                )
                submitted.append(name)
                task_paths.append(Path(str(outcome["task_path"])))
                attempt["jobs"].append(
                    {
                        "role": role,
                        "name": name,
                        "status": outcome["status"],
                        "task_path": outcome["task_path"],
                        "status_path": outcome["status_path"],
                        "output_root": str(command.output_root),
                        "command": command.as_json(),
                        "submission_round": submission_round,
                        "materialized_at": _utc_now(),
                    }
                )
                round_state["submitted_job_names"] = list(submitted)
                _save_benchmark_state(benchmark_root, state)
        except Exception as error:
            round_state.update(
                {"status": "SubmissionFailed", "completed_at": _utc_now(), "error": str(error)}
            )
            _save_benchmark_state(benchmark_root, state)
            raise
        round_state.update({"status": "Submitted", "completed_at": _utc_now()})
        write_plan(attempt_root / "tasks" / "plan.json", task_paths)
        round_state["task_plan"] = str(attempt_root / "tasks" / "plan.json")
        attempt["task_plan"] = round_state["task_plan"]
        _save_benchmark_state(benchmark_root, state)
        return {
            "submitted": submitted,
            "jobs": list(_latest_jobs_by_role(attempt).values()),
            "attempt": attempt_number,
            "task_plan": round_state["task_plan"],
        }


def _gsm8k_shard_valid(path: Path, expected_index: int) -> bool:
    payload = _read_json(path)
    if payload.get("status") not in {
        "GSM8K_EVAL_OK",
        "GSM8K_VLLM_GENERATION_OK",
        "GSM8K_LMDEPLOY_GENERATION_OK",
    }:
        return False
    shard_index = payload.get("shard_index", payload.get("rank"))
    if int(shard_index if shard_index is not None else -1) != expected_index:
        return False
    if int(payload.get("dataset_sample_count", -1)) != 1319:
        return False
    if payload.get("artifact_mutated") is not False:
        return False
    if "source_model_mutated" in payload and payload["source_model_mutated"] is not False:
        return False
    if payload.get("model_source") == "dcp" and payload.get("checkpoint_mutated") is not False:
        return False
    predictions = path.parent / "predictions.jsonl"
    return predictions.is_file() and predictions.stat().st_size > 0


def _gsm8k_prediction_coverage(root: Path) -> dict[str, Any]:
    """Validate the complete, unique one-sample GSM8K prediction key space."""

    aggregate = root / "predictions.jsonl"
    paths = [aggregate] if aggregate.is_file() else sorted(root.glob("shard-*/predictions.jsonl"))
    seen: set[int] = set()
    error = ""
    try:
        if not paths:
            raise ValueError("no prediction files")
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    doc_index = int(row["doc_index"])
                    if doc_index in seen:
                        raise ValueError(f"duplicate doc_index={doc_index}")
                    if int(row.get("sample_index", 0)) != 0:
                        raise ValueError(f"nonzero sample_index at {path}:{line_number}")
                    seen.add(doc_index)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        error = str(exc)
    expected = set(range(1319))
    complete = not error and seen == expected
    return {
        "complete": complete,
        "prediction_count": len(seen),
        "expected_prediction_count": len(expected),
        "prediction_files": [str(path) for path in paths],
        "first_missing_doc_indices": sorted(expected - seen)[:16],
        "error": error,
    }


def _long_context_artifacts(root: Path) -> dict[str, Any]:
    """Validate all eight official long-context inference shards."""

    results = sorted(root.glob("shard-*/result.json"))
    complete = len(results) == 8
    error = ""
    total_expected = 0
    total_predictions = 0
    manifest_hashes: set[str] = set()
    inputs_hashes: set[str] = set()
    try:
        for expected_index, path in enumerate(results):
            payload = _read_json(path)
            if payload.get("status") != "LONG_CONTEXT_INFERENCE_SHARD_OK":
                raise ValueError(f"incomplete shard: {path}")
            if payload.get("artifact_mutated") is not False:
                raise ValueError(f"artifact mutation was not ruled out: {path}")
            if payload.get("official_protocol_compatible") is not True:
                raise ValueError(f"non-official inference protocol: {path}")
            if int(payload.get("shard_index", -1)) != expected_index:
                raise ValueError(f"shard index mismatch: {path}")
            if int(payload.get("shard_count", -1)) != 8:
                raise ValueError(f"shard count mismatch: {path}")
            expected_count = int(payload.get("expected_prediction_count", -1))
            prediction_count = int(payload.get("prediction_count", -1))
            if expected_count < 0 or prediction_count != expected_count:
                raise ValueError(f"prediction count mismatch: {path}")
            predictions = path.parent / "predictions.jsonl"
            if expected_count > 0 and (
                not predictions.is_file() or predictions.stat().st_size == 0
            ):
                raise ValueError(f"prediction file is missing or empty: {predictions}")
            total_expected += expected_count
            total_predictions += prediction_count
            manifest_hashes.add(str(payload.get("prepared_manifest_sha256", "")))
            inputs_hashes.add(str(payload.get("prepared_inputs_sha256", "")))
        if len(results) != 8:
            raise ValueError(f"expected 8 shards, found {len(results)}")
        if total_expected <= 0:
            raise ValueError("long-context prediction set is empty")
        if len(manifest_hashes) != 1 or "" in manifest_hashes:
            raise ValueError("prepared manifest identity differs across shards")
        if len(inputs_hashes) != 1 or "" in inputs_hashes:
            raise ValueError("prepared input identity differs across shards")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        complete = False
        error = str(exc)
    return {
        "complete": complete,
        "shards": [str(path) for path in results],
        "prediction_count": total_predictions,
        "expected_prediction_count": total_expected,
        "prepared_manifest_sha256": (
            next(iter(manifest_hashes)) if len(manifest_hashes) == 1 else None
        ),
        "prepared_inputs_sha256": (next(iter(inputs_hashes)) if len(inputs_hashes) == 1 else None),
        "error": error,
    }


def _sciq_artifacts(root: Path) -> dict[str, Any]:
    """Validate the sealed one-host official SciQ inference contract."""

    manifest_path = root / "run_manifest.json"
    summary_path = root / "summary.json"
    machine_result_path = root / "machine-00" / "result.json"
    prediction_paths = sorted(root.glob("machine-00/predictions/*.jsonl"))
    error = ""
    prediction_count = 0
    score: float | None = None
    try:
        if not manifest_path.is_file() or not summary_path.is_file():
            raise ValueError("SciQ run_manifest.json/summary.json is missing")
        if not machine_result_path.is_file():
            raise ValueError("SciQ machine-00/result.json is missing")
        manifest = _read_json(manifest_path)
        if manifest.get("profile") != SCIQ_SOURCE_PROFILE:
            raise ValueError(f"unexpected SciQ profile: {manifest.get('profile')}")
        if manifest.get("task_orders") != [SCIQ_TASK_ORDER]:
            raise ValueError(f"unexpected SciQ task orders: {manifest.get('task_orders')}")
        if int(manifest.get("global_seed", -1)) != DEFAULT_GLOBAL_SEED:
            raise ValueError("SciQ global seed is not 42")
        if int(manifest.get("machine_count", -1)) != 1:
            raise ValueError("SciQ machine_count is not 1")
        if int(manifest.get("gpus_per_machine", -1)) != 8:
            raise ValueError("SciQ gpus_per_machine is not 8")
        if manifest.get("hf_backend_requested") != "native_vllm":
            raise ValueError("SciQ release inference must use native_vllm")
        if int(manifest.get("score_batch_size", -1)) != 8:
            raise ValueError("SciQ score batch is not 8")
        if int(manifest.get("generation_batch_size", -1)) != 8:
            raise ValueError("SciQ generation batch is not 8")
        if int(manifest.get("processes_per_gpu", -1)) != 1:
            raise ValueError("SciQ processes_per_gpu is not 1")
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("SciQ run manifest must contain exactly one task")
        source_task = tasks[0]
        expected_task_contract = {
            "task_order": SCIQ_TASK_ORDER,
            "task": SCIQ_TASK_NAME,
            "file": SCIQ_SOURCE_FILE,
            "sha256": SCIQ_SOURCE_SHA256,
            "num_examples": SCIQ_EXAMPLE_COUNT,
            "metric": "acc",
            "request_type": "loglikelihood",
        }
        for field, expected in expected_task_contract.items():
            if source_task.get(field) != expected:
                raise ValueError(
                    f"SciQ source contract mismatch: {field}={source_task.get(field)!r} "
                    f"!= {expected!r}"
                )

        machine = _read_json(machine_result_path)
        if machine.get("status") != "CORE_NATIVE_MACHINE_OK":
            raise ValueError(f"SciQ machine status is {machine.get('status')}")
        if machine.get("artifact_mutated") is not False:
            raise ValueError("SciQ model artifact mutation was not ruled out")
        if machine.get("source_artifact_mutated") is not False:
            raise ValueError("SciQ source artifact mutation was not ruled out")
        if machine.get("completed_work_items") != machine.get("planned_work_items"):
            raise ValueError("SciQ machine work coverage is incomplete")

        summary = _read_json(summary_path)
        if summary.get("status") != "CORE_NATIVE_POOL_OK":
            raise ValueError(f"SciQ pool status is {summary.get('status')}")
        if int(summary.get("task_count", -1)) != 1:
            raise ValueError("SciQ summary task_count is not 1")
        if int(summary.get("prediction_task_count_complete", -1)) != 1:
            raise ValueError("SciQ prediction task is incomplete")
        if int(summary.get("score_task_count_complete", -1)) != 1:
            raise ValueError("SciQ score task is incomplete")
        if int(summary.get("expected_predictions", -1)) != SCIQ_EXAMPLE_COUNT:
            raise ValueError("SciQ expected prediction count is not 1000")
        if int(summary.get("observed_predictions", -1)) != SCIQ_EXAMPLE_COUNT:
            raise ValueError("SciQ observed prediction count is not 1000")
        summary_tasks = summary.get("tasks")
        if not isinstance(summary_tasks, list) or len(summary_tasks) != 1:
            raise ValueError("SciQ summary must contain exactly one task")
        task = summary_tasks[0]
        if (
            task.get("task_order") != SCIQ_TASK_ORDER
            or task.get("task") != SCIQ_TASK_NAME
            or task.get("metric") != "acc"
            or task.get("request_type") != "loglikelihood"
            or task.get("score_status") != "SCORED"
        ):
            raise ValueError(f"SciQ scored task contract is invalid: {task}")
        score = float(task["primary_score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"SciQ score is invalid: {score}")

        seen: set[str] = set()
        for path in prediction_paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("task_order") != SCIQ_TASK_ORDER:
                        raise ValueError(f"wrong task order at {path}:{line_number}")
                    if row.get("task") != SCIQ_TASK_NAME:
                        raise ValueError(f"wrong task name at {path}:{line_number}")
                    key = json.dumps(
                        row["example_id"],
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if key in seen:
                        raise ValueError(f"duplicate SciQ example at {path}:{line_number}")
                    seen.add(key)
        prediction_count = len(seen)
        if prediction_count != SCIQ_EXAMPLE_COUNT:
            raise ValueError(f"SciQ prediction coverage is {prediction_count}/1000")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        error = str(exc)
    return {
        "complete": not error,
        "run_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "pool_summary": str(summary_path) if summary_path.is_file() else None,
        "machine_result": str(machine_result_path) if machine_result_path.is_file() else None,
        "prediction_files": [str(path) for path in prediction_paths],
        "prediction_count": prediction_count,
        "expected_prediction_count": SCIQ_EXAMPLE_COUNT,
        "primary_score": score,
        "error": error,
    }


def _artifact_status(benchmark: str, attempt: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(attempt["attempt_root"]))
    if benchmark == "core88":
        summary_path = root / "summary.json"
        machine_results = sorted(root.glob("machine-*/result.json"))
        gsm_root = root / "standalone-gsm8k"
        summary_ok = summary_path.is_file() and len(machine_results) == 4
        if summary_path.is_file():
            summary = _read_json(summary_path)
            if summary.get("status") not in {
                "CORE_NATIVE_POOL_OK",
                "CORE_NATIVE_POOL_PREDICTIONS_OK",
            }:
                summary_ok = False
            if summary.get("observed_predictions") != summary.get("expected_predictions"):
                summary_ok = False
        for path in machine_results:
            payload = _read_json(path)
            if (
                payload.get("status") != "CORE_NATIVE_MACHINE_OK"
                or payload.get("artifact_mutated") is not False
                or payload.get("completed_work_items") != payload.get("planned_work_items")
            ):
                summary_ok = False
        gsm_results = sorted(gsm_root.glob("shard-*/result.json"))
        gsm_shards_ok = len(gsm_results) == 8 and all(
            _gsm8k_shard_valid(path, index) for index, path in enumerate(gsm_results)
        )
        gsm_coverage = _gsm8k_prediction_coverage(gsm_root)
        gsm_success = gsm_shards_ok and gsm_coverage["complete"]
        okay = summary_ok and gsm_success
        return {
            "complete": okay,
            "pool_summary": str(summary_path) if summary_path.is_file() else None,
            "machine_results": [str(path) for path in machine_results],
            "gsm8k_present": gsm_success,
            "gsm8k_coverage": gsm_coverage,
        }
    if benchmark == "gsm8k":
        aggregate = root / "aggregate.json"
        shards = sorted(root.glob("shard-*/result.json"))
        shards_ok = len(shards) == 8 and all(
            _gsm8k_shard_valid(path, index) for index, path in enumerate(shards)
        )
        coverage = _gsm8k_prediction_coverage(root)
        complete = shards_ok and coverage["complete"]
        return {
            "complete": complete,
            "aggregate": str(aggregate) if aggregate.is_file() else None,
            "shards": [str(path) for path in shards],
            "coverage": coverage,
        }
    if benchmark == "sciq":
        return _sciq_artifacts(root)
    return _long_context_artifacts(root)


def refresh_status(root: Path, registration_name: str, benchmark: str) -> dict[str, Any]:
    """Refresh inference, scoring, and final task/artifact state."""

    model_root, registration = load_registration(root, registration_name)
    benchmark_root = _benchmark_path(model_root, benchmark)
    state = _load_benchmark_state(benchmark_root, benchmark, registration)
    latest = _latest_attempt(state, "inference_attempts")
    if latest is None:
        return {
            "benchmark": benchmark,
            "inference": "NotSubmitted",
            "scoring": "NotSubmitted",
            "final": "NotSubmitted",
            "jobs": [],
        }
    jobs = _refresh_jobs(latest)
    artifacts = _artifact_status(benchmark, latest)
    effective_jobs = list(_latest_jobs_by_role(latest).values())
    expected_roles = set(_expected_roles(benchmark))
    tasks_complete = set(_latest_jobs_by_role(latest)) == expected_roles and all(
        job.get("status") == "Succeeded" for job in effective_jobs
    )
    if artifacts["complete"] and tasks_complete:
        latest["validated_status"] = "INFERENCE_COMPLETE"
    else:
        latest["validated_status"] = "INFERENCE_INCOMPLETE"
    latest["artifact_validation"] = artifacts

    scoring_status: dict[str, Any] = {"status": "NotSubmitted", "jobs": []}
    scoring = _latest_attempt(state, "scoring_attempts")
    if scoring is not None:
        scoring_jobs = _refresh_jobs(scoring)
        scoring_artifacts = _score_artifacts(benchmark, scoring)
        expected_score_jobs = _expected_scoring_job_count(benchmark, scoring)
        scoring_complete = (
            len(scoring_jobs) == expected_score_jobs
            and all(job.get("status") == "Succeeded" for job in scoring_jobs)
            and scoring_artifacts["complete"]
        )
        scoring["validated_status"] = (
            "SCORING_COMPLETE" if scoring_complete else "SCORING_INCOMPLETE"
        )
        scoring["artifact_validation"] = scoring_artifacts
        scoring_status = {
            "status": scoring["validated_status"],
            "jobs": scoring_jobs,
            "artifacts": scoring_artifacts,
        }

    final_status: dict[str, Any] = {"status": "NotSubmitted", "job": None}
    final = _latest_attempt(state, "final_attempts")
    if final is not None:
        final_job = dict(final["job"])
        status_path = Path(str(final_job.get("status_path", "")))
        if status_path.is_file():
            status = read_status(status_path)
            final_job["status_record"] = status
            final_job["status"] = status["state"]
        final["job"] = final_job
        output_root = Path(str(final["output_root"]))
        final_complete = (
            final_job.get("status") == "Succeeded" and (output_root / "_SUCCESS").is_file()
        )
        final["validated_status"] = "FINAL_COMPLETE" if final_complete else "FINAL_INCOMPLETE"
        final_status = {
            "status": final["validated_status"],
            "job": final_job,
            "output_root": str(output_root),
            "files": _result_files(output_root) if final_complete else [],
        }
    _save_benchmark_state(benchmark_root, state)
    return {
        "benchmark": benchmark,
        "inference": latest["validated_status"],
        "jobs": effective_jobs,
        "job_history": jobs,
        "artifacts": artifacts,
        "scoring": scoring_status,
        "final": final_status,
    }


def _core_scorer_specs() -> list[tuple[dict[str, Any], int, int]]:
    """Expand the fixed scorer-family partition contract."""

    specs: list[tuple[dict[str, Any], int, int]] = []
    for group in CORE_SCORER_GROUPS:
        partition_count = int(group["partition_count"])
        if partition_count <= 0:
            raise EvaluationError(
                f"Core88 scorer group {group['index']} has invalid partition count "
                f"{partition_count}"
            )
        specs.extend(
            (group, partition_index, partition_count) for partition_index in range(partition_count)
        )
    return specs


def _core_scoring_contract(specs: Sequence[tuple[dict[str, Any], int, int]]) -> dict[str, Any]:
    """Record enough sharding metadata to validate this scoring attempt later."""

    groups: list[dict[str, Any]] = []
    for group in CORE_SCORER_GROUPS:
        matching = [spec for spec in specs if int(spec[0]["index"]) == int(group["index"])]
        groups.append(
            {
                "index": int(group["index"]),
                "task_orders": str(group["task_orders"]),
                "partition_count": int(group["partition_count"]),
                "job_count": len(matching),
            }
        )
    return {
        "version": CORE88_SCORING_CONTRACT_VERSION,
        "expected_job_count": len(specs) + 1,
        "expected_code_summary_count": len(specs),
        "prediction_score_snapshot": "base/core88-prediction-scores.json",
        "groups": groups,
    }


def _expected_scoring_job_count(benchmark: str, attempt: dict[str, Any] | None = None) -> int:
    """Return the attempt-specific scorer count without invalidating old runs."""

    if benchmark != "core88":
        return 1
    if attempt is not None:
        contract = attempt.get("scoring_contract")
        if isinstance(contract, dict):
            value = contract.get("expected_job_count")
            if isinstance(value, int) and value > 0:
                return value
        planned_jobs = attempt.get("planned_jobs")
        if isinstance(planned_jobs, list) and planned_jobs:
            return len(planned_jobs)
        # Attempts created before the sharding contract always used four
        # scorer families times eight partitions.
        return CORE88_LEGACY_SCORING_JOB_COUNT
    return len(_core_scorer_specs()) + 1


def _score_command(
    registration: dict[str, Any],
    benchmark: str,
    inference_attempt: dict[str, Any],
    score_root: Path,
    job_name: str,
    group: dict[str, Any] | None = None,
    data_root: Path | None = None,
    partition_index: int = 0,
    partition_count: int = 1,
) -> CommandSpec:
    env = _base_launch_env(registration)
    inference_root = Path(str(inference_attempt["attempt_root"]))
    if benchmark == "core88":
        if group is None:
            output = score_root / "base"
            argv = (
                PYTHON_BIN, "-m", "ncp_olmo_eval.core_native_aggregate",
                "--data-root", str(CORE_DATA_ROOT),
                "--profile", "core88",
                "--input-root", str(inference_root),
                "--output-json", str(output / "core88-prediction-scores.json"),
                "--output-csv", str(output / "core88-prediction-scores.csv"),
            )
            return CommandSpec(
                argv, env, Resources(gpus=0, cpus=8, memory_gib=32), output, RUNTIME_IMAGE
            )
        output_dir = score_root / (
            f"g{int(group['index']):02d}-p{partition_index:02d}of{partition_count:02d}"
        )
        image = os.environ.get(str(group["image_env"]), str(group["default_image"]))
        output_jsonl = output_dir / f"code-results-p{partition_index}of{partition_count}.jsonl"
        error_jsonl = output_dir / f"code-results-p{partition_index}of{partition_count}-errors.jsonl"
        summary_json = output_dir / f"code-results-p{partition_index}of{partition_count}-summary.json"
        argv = [
            PYTHON_BIN, "-m", "ncp_olmo_eval.core_native_code_eval",
            "--data-root", str(CORE_DATA_ROOT),
            "--profile", "core88",
            "--input-root", str(inference_root),
            "--output-jsonl", str(output_jsonl),
            "--error-jsonl", str(error_jsonl),
            "--summary-json", str(summary_json),
            "--task-orders", str(group["task_orders"]),
            "--partition-index", str(partition_index),
            "--partition-count", str(partition_count),
            "--limit", "0",
            "--runtime-prefix", str(group["runtime_prefix"]),
            "--runtime-image", image,
            "--memory-mib", "30720" if int(group["index"]) == 1 else "4096",
            "--resume",
        ]
        if str(group["allow_experimental"]) == "1":
            argv.append("--allow-experimental-scorers")
        if int(group["index"]) == 1:
            env["OLMO_EVAL_COMMIT"] = OLMO_EVAL_COMMIT
            env["OLMO_EVAL_ROOT"] = str(OLMO_EVAL_ROOT)
            env["PYTHONPATH"] = f"{OLMO_EVAL_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"
        return CommandSpec(
            tuple(argv), env, Resources(gpus=0, cpus=8, memory_gib=32), output_dir, image
        )
    if benchmark == "sciq":
        return CommandSpec(
            (
                PYTHON_BIN,
                "-m",
                "ncp_olmo_eval.portable_tasks",
                "score-sciq",
                "--data-root",
                str(CORE_DATA_ROOT),
                "--inference-root",
                str(inference_root),
                "--output-root",
                str(score_root),
            ),
            env,
            Resources(gpus=0, cpus=8, memory_gib=32),
            score_root,
            RUNTIME_IMAGE,
        )
    if benchmark == "ruler":
        assert data_root is not None
        return CommandSpec(
            (
                PYTHON_BIN, "-m", "ncp_olmo_eval.long_context_score",
                "--data-root", str(data_root),
                "--inference-root", str(inference_root),
                "--output-root", str(score_root),
                "--no-allow-partial",
            ),
            env,
            Resources(gpus=0, cpus=8, memory_gib=32),
            score_root,
            RUNTIME_IMAGE,
        )
    if benchmark == "helmet":
        assert data_root is not None
        argv = [
            PYTHON_BIN, "-m", "ncp_olmo_eval.helmet_score",
            "--official-root", str(HELMET_OFFICIAL_ROOT),
            "--data-root", str(data_root),
            "--inference-root", str(inference_root),
            "--output-root", str(score_root),
            "--no-allow-partial",
        ]
        if os.environ.get("HELMET_RUN_CITATION_NLI", "1") == "1":
            argv.append("--run-citation-nli")
        else:
            argv.append("--no-run-citation-nli")
        judge_results = os.environ.get("HELMET_JUDGE_RESULTS", "")
        if judge_results:
            argv.extend(["--judge-results", judge_results])
        return CommandSpec(
            tuple(argv), env, Resources(gpus=0, cpus=16, memory_gib=128), score_root, RUNTIME_IMAGE
        )
    return CommandSpec(
        (
            PYTHON_BIN, "-m", "ncp_olmo_eval.unified_eval_results", "score-gsm8k",
            "--inference-root", str(inference_root),
            "--output-root", str(score_root),
        ),
        env,
        Resources(gpus=0, cpus=8, memory_gib=32),
        score_root,
        RUNTIME_IMAGE,
    )


def submit_scoring(
    *, root: Path, registration_name: str, benchmark: str, executor: str, dry_run: bool
) -> dict[str, Any]:
    """Always create a fresh scoring task plan after validated inference."""

    model_root, registration = load_registration(root, registration_name)
    repo_state = _git_state(_repo_root())
    if repo_state["repo_dirty"] and not dry_run:
        raise EvaluationError("统一评测提交要求当前 Git worktree 完全干净")
    benchmark_root = _benchmark_path(model_root, benchmark)
    with _locked(model_root):
        state = _load_benchmark_state(benchmark_root, benchmark, registration)
        inference = _latest_attempt(state, "inference_attempts")
        if inference is None:
            raise EvaluationError(f"{benchmark} 尚未提交推理")
        _refresh_jobs(inference)
        artifacts = _artifact_status(benchmark, inference)
        effective_inference_jobs = _latest_jobs_by_role(inference)
        tasks_complete = set(effective_inference_jobs) == set(
            _expected_roles(benchmark)
        ) and all(job.get("status") == "Succeeded" for job in effective_inference_jobs.values())
        if not dry_run and (not tasks_complete or not artifacts["complete"]):
            raise EvaluationError(
                f"{benchmark} 推理尚未完整："
                f"tasks={tasks_complete}, artifacts={artifacts}"
            )
        attempt_number = len(state["scoring_attempts"]) + 1
        score_root = benchmark_root / "scoring" / f"attempt-{attempt_number:04d}"
        commands = []
        names = []
        roles = []
        scorer_specs: list[tuple[dict[str, Any] | None, int, int]]
        if benchmark == "core88":
            scorer_specs = [(None, 0, 1), *_core_scorer_specs()]
        else:
            scorer_specs = [(None, 0, 1)]
        scoring_contract = (
            _core_scoring_contract(
                [
                    (group, partition_index, partition_count)
                    for group, partition_index, partition_count in scorer_specs
                    if group is not None
                ]
            )
            if benchmark == "core88"
            else {"version": "single-score-job-v1", "expected_job_count": 1}
        )
        for index, (group, partition_index, partition_count) in enumerate(scorer_specs):
            suffix = f"p{index:02d}-a{attempt_number}"
            name = _job_name("eval", registration_name, benchmark, suffix)
            names.append(name)
            roles.append(
                f"score-g{int(group['index']):02d}-p{partition_index:02d}"
                if group is not None
                else ("score-base" if benchmark == "core88" else "score")
            )
            prepared = (
                Path(str(state["prepared_data"]["path"])) if state.get("prepared_data") else None
            )
            commands.append(
                _score_command(
                    registration,
                    benchmark,
                    inference,
                    score_root,
                    name,
                    group=group,
                    data_root=prepared,
                    partition_index=partition_index,
                    partition_count=partition_count,
                )
            )
        if dry_run:
            jobs = []
            for role, name, command in zip(roles, names, commands, strict=True):
                outcome = _run_command(
                    command,
                    task_id=name,
                    phase="scoring",
                    benchmark=benchmark,
                    role=role,
                    task_root=score_root / "tasks" / name,
                    executor=executor,
                    dry_run=True,
                )
                jobs.append(
                    {
                        "role": role,
                        "name": name,
                        "status": outcome["status"],
                        "command": command.as_json(),
                    }
                )
            return {
                "submitted": names,
                "jobs": jobs,
                "attempt": attempt_number,
                "scoring_contract": scoring_contract,
                "dry_run": True,
            }

        score_root.mkdir(parents=True, exist_ok=True)
        attempt = {
            "attempt": attempt_number,
            "score_root": str(score_root),
            "inference_attempt": inference["attempt"],
            "created_at": _utc_now(),
            "repo_state": repo_state,
            "scoring_contract": scoring_contract,
            "planned_jobs": [
                {"role": role, "name": name} for role, name in zip(roles, names, strict=True)
            ],
            "jobs": [],
            "submission_status": "Submitting",
        }
        state["scoring_attempts"].append(attempt)
        _save_benchmark_state(benchmark_root, state)
        task_paths: list[Path] = []
        try:
            for role, name, command in zip(roles, names, commands, strict=True):
                outcome = _run_command(
                    command,
                    task_id=name,
                    phase="scoring",
                    benchmark=benchmark,
                    role=role,
                    task_root=score_root / "tasks" / name,
                    executor=executor,
                    dry_run=False,
                )
                task_paths.append(Path(str(outcome["task_path"])))
                attempt["jobs"].append(
                    {
                        "role": role,
                        "name": name,
                        "status": outcome["status"],
                        "task_path": outcome["task_path"],
                        "status_path": outcome["status_path"],
                        "command": command.as_json(),
                        "materialized_at": _utc_now(),
                    }
                )
                _save_benchmark_state(benchmark_root, state)
        except Exception as error:
            attempt.update(
                {
                    "submission_status": "SubmissionFailed",
                    "submission_completed_at": _utc_now(),
                    "submission_error": str(error),
                }
            )
            _save_benchmark_state(benchmark_root, state)
            raise
        attempt.update({"submission_status": "Submitted", "submission_completed_at": _utc_now()})
        write_plan(score_root / "tasks" / "plan.json", task_paths)
        attempt["task_plan"] = str(score_root / "tasks" / "plan.json")
        _save_benchmark_state(benchmark_root, state)
        return {
            "submitted": names,
            "jobs": attempt["jobs"],
            "attempt": attempt_number,
            "task_plan": attempt["task_plan"],
        }


def _score_artifacts(benchmark: str, attempt: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(attempt["score_root"]))
    if benchmark == "core88":
        summaries = sorted(root.glob("g*-p*of*/code-results-*-summary.json"))
        contract = attempt.get("scoring_contract")
        fast_contract = (
            isinstance(contract, dict)
            and contract.get("version") == CORE88_SCORING_CONTRACT_VERSION
        )
        expected_summary_count = (
            int(contract["expected_code_summary_count"])
            if fast_contract
            else _expected_scoring_job_count(benchmark, attempt)
        )
        base_report = root / "base" / "core88-prediction-scores.json"
        base_complete = not fast_contract
        if fast_contract and base_report.is_file():
            payload = _read_json(base_report)
            base_complete = payload.get(
                "status"
            ) == "CORE_NATIVE_FULL_PREDICTIONS_OK" and payload.get(
                "prediction_task_count_complete"
            ) == payload.get(
                "task_count"
            )
        complete = len(summaries) == expected_summary_count and base_complete
        for path in summaries:
            payload = _read_json(path)
            if payload.get("status") != "CORE_NATIVE_CODE_RESULTS_OK":
                complete = False
            if fast_contract and (
                payload.get("schema_version") != "core-native-code-sandbox-v2"
                or (payload.get("result_aggregate") or {}).get("status")
                != "CORE_NATIVE_CODE_RESULT_AGGREGATE_OK"
            ):
                complete = False
        return {
            "complete": complete,
            "expected_summary_count": expected_summary_count,
            "observed_summary_count": len(summaries),
            "summaries": [str(path) for path in summaries],
            "prediction_score_snapshot": str(base_report) if fast_contract else None,
            "prediction_score_snapshot_complete": base_complete,
        }
    score = root / "score.json"
    complete = False
    error = ""
    if score.is_file():
        payload = _read_json(score)
        expected_status = {
            "gsm8k": "UNIFIED_GSM8K_OFFICIAL_RESCORE_OK",
            "sciq": "CORE_NATIVE_FULL_OK",
            "ruler": "LONG_CONTEXT_SCORE_OK",
            "helmet": "LONG_CONTEXT_SCORE_OK",
        }.get(benchmark)
        complete = payload.get("status") == expected_status
        if benchmark == "sciq":
            tasks = payload.get("tasks")
            task = tasks[0] if isinstance(tasks, list) and len(tasks) == 1 else None
            try:
                if payload.get("profile") != SCIQ_SOURCE_PROFILE:
                    raise ValueError("SciQ score profile mismatch")
                if payload.get("task_orders") != [SCIQ_TASK_ORDER]:
                    raise ValueError("SciQ score task order mismatch")
                if int(payload.get("expected_predictions", -1)) != SCIQ_EXAMPLE_COUNT:
                    raise ValueError("SciQ score expected count mismatch")
                if int(payload.get("observed_predictions", -1)) != SCIQ_EXAMPLE_COUNT:
                    raise ValueError("SciQ score observed count mismatch")
                if not isinstance(task, dict):
                    raise ValueError("SciQ score must contain exactly one task")
                if (
                    task.get("task_order") != SCIQ_TASK_ORDER
                    or task.get("task") != SCIQ_TASK_NAME
                    or task.get("metric") != "acc"
                    or task.get("request_type") != "loglikelihood"
                    or task.get("score_status") != "SCORED"
                ):
                    raise ValueError("SciQ scored task contract mismatch")
                primary_score = float(task["primary_score"])
                if not math.isfinite(primary_score) or not 0.0 <= primary_score <= 1.0:
                    raise ValueError("SciQ score is not a finite accuracy")
                if not (root / "sciq-score.csv").is_file() or not (
                    root / "_SUCCESS"
                ).is_file():
                    raise ValueError("SciQ score CSV/_SUCCESS is missing")
            except (KeyError, TypeError, ValueError) as exc:
                complete = False
                error = str(exc)
        elif benchmark == "ruler":
            complete = complete and payload.get("official_protocol_compatible") is True
        elif benchmark == "helmet":
            complete = complete and payload.get("all_length_primary_scores_complete") is True
    return {"complete": complete, "score_json": str(score), "error": error}


def submit_final(
    *, root: Path, registration_name: str, benchmark: str, executor: str, dry_run: bool
) -> dict[str, Any]:
    """Show an existing final result or create one fresh materialization task."""

    model_root, registration = load_registration(root, registration_name)
    repo_state = _git_state(_repo_root())
    if repo_state["repo_dirty"] and not dry_run:
        raise EvaluationError("统一评测提交要求当前 Git worktree 完全干净")
    benchmark_root = _benchmark_path(model_root, benchmark)
    with _locked(model_root):
        state = _load_benchmark_state(benchmark_root, benchmark, registration)
        for existing in reversed(state.get("final_attempts", [])):
            result_root = Path(str(existing["output_root"]))
            success = result_root / "_SUCCESS"
            if success.is_file():
                return {
                    "existing": True,
                    "output_root": str(result_root),
                    "files": _result_files(result_root),
                }
        scoring = _latest_attempt(state, "scoring_attempts")
        inference = _latest_attempt(state, "inference_attempts")
        if scoring is None or inference is None:
            raise EvaluationError(f"{benchmark} 尚未完成推理和打分")
        _refresh_jobs(scoring)
        score_artifacts = _score_artifacts(benchmark, scoring)
        expected_score_jobs = _expected_scoring_job_count(benchmark, scoring)
        tasks_complete = len(scoring["jobs"]) == expected_score_jobs and all(
            job.get("status") == "Succeeded" for job in scoring["jobs"]
        )
        if not dry_run and (not tasks_complete or not score_artifacts["complete"]):
            raise EvaluationError(
                f"{benchmark} 打分尚未完整："
                f"tasks={tasks_complete}, artifacts={score_artifacts}"
            )
        attempt_number = len(state["final_attempts"]) + 1
        inference_root = Path(str(inference["attempt_root"]))
        strict_core_workflow = (
            benchmark == "core88" and (inference_root / "workflow.json").is_file()
        )
        output_root = (
            inference_root / "finalize"
            if strict_core_workflow
            else benchmark_root / "final" / f"attempt-{attempt_number:04d}"
        )
        if strict_core_workflow and output_root.exists():
            raise EvaluationError(
                "Core88 严格 workflow 的固定 finalize 目录已存在但没有 _SUCCESS；"
                f"为保留失败证据不会自动覆盖：{output_root}"
            )
        name = _final_job_name(registration_name, benchmark, f"a{attempt_number}")
        env = _base_launch_env(registration)
        if benchmark == "core88":
            if not strict_core_workflow:
                raise EvaluationError("Core88 正式汇总要求同轮 workflow.json")
            code_result_dirs = sorted(
                path for path in Path(str(scoring["score_root"])).glob("g*-p*of*") if path.is_dir()
            )
            argv = [
                PYTHON_BIN, "-m", "ncp_olmo_eval.portable_tasks", "final-core",
                "--data-root", str(CORE_DATA_ROOT),
                "--inference-root", str(inference_root),
                "--base-report", str(Path(str(scoring["score_root"])) / "base" / "core88-prediction-scores.json"),
                "--gsm8k-results-root", str(inference_root / "standalone-gsm8k"),
                "--workflow-manifest", str(inference_root / "workflow.json"),
                "--output-root", str(output_root),
            ]
            for directory in code_result_dirs:
                argv.extend(["--code-results-dir", str(directory)])
            command = CommandSpec(
                tuple(argv),
                env,
                Resources(gpus=0, cpus=8, memory_gib=32),
                output_root,
                RUNTIME_IMAGE,
            )
        else:
            command = CommandSpec(
                (
                    PYTHON_BIN, "-m", "ncp_olmo_eval.unified_eval_results", "finalize",
                    "--benchmark", benchmark,
                    "--scoring-root", str(scoring["score_root"]),
                    "--output-root", str(output_root),
                ),
                env,
                Resources(gpus=0, cpus=4, memory_gib=16),
                output_root,
                RUNTIME_IMAGE,
            )
        task_root = benchmark_root / "final" / "tasks" / name
        if dry_run:
            outcome = _run_command(
                command,
                task_id=name,
                phase="final",
                benchmark=benchmark,
                role="final",
                task_root=task_root,
                executor=executor,
                dry_run=True,
            )
            return {
                "existing": False,
                "submitted": name,
                "output_root": str(output_root),
                "job": {
                    "role": "final",
                    "name": name,
                    "status": outcome["status"],
                    "command": command.as_json(),
                },
                "dry_run": True,
            }

        attempt = {
            "attempt": attempt_number,
            "output_root": str(output_root),
            "created_at": _utc_now(),
            "repo_state": repo_state,
            "submission_status": "Submitting",
            "job": {
                "role": "final",
                "name": name,
                "status": "Submitting",
                "command": command.as_json(),
            },
        }
        state["final_attempts"].append(attempt)
        _save_benchmark_state(benchmark_root, state)
        try:
            outcome = _run_command(
                command,
                task_id=name,
                phase="final",
                benchmark=benchmark,
                role="final",
                task_root=task_root,
                executor=executor,
                dry_run=False,
            )
        except Exception as error:
            attempt.update(
                {
                    "submission_status": "SubmissionFailed",
                    "submission_completed_at": _utc_now(),
                    "submission_error": str(error),
                }
            )
            attempt["job"]["status"] = "SubmissionFailed"
            _save_benchmark_state(benchmark_root, state)
            raise
        attempt.update({"submission_status": "Submitted", "submission_completed_at": _utc_now()})
        attempt["job"].update(
            {
                "status": outcome["status"],
                "task_path": outcome["task_path"],
                "status_path": outcome["status_path"],
                "materialized_at": _utc_now(),
            }
        )
        write_plan(task_root.parent / "plan.json", [Path(str(outcome["task_path"]))])
        attempt["task_plan"] = str(task_root.parent / "plan.json")
        _save_benchmark_state(benchmark_root, state)
        return {
            "existing": False,
            "submitted": name,
            "output_root": str(output_root),
            "task_plan": attempt["task_plan"],
        }


def _result_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.glob("*")):
        if not path.is_file() or path.name == "job.log":
            continue
        item: dict[str, Any] = {"path": str(path), "size": path.stat().st_size}
        if path.suffix == ".json" and path.stat().st_size <= 2 * 1024 * 1024:
            try:
                item["content"] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        elif path.suffix == ".csv" and path.stat().st_size <= 512 * 1024:
            item["content"] = path.read_text(encoding="utf-8")
        files.append(item)
    return files


def _interactive_choice(prompt: str, choices: Sequence[str]) -> str:
    print(prompt)
    for index, choice in enumerate(choices, start=1):
        print(f"  {index}. {choice}")
    while True:
        value = input("请选择：").strip()
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1]
        if value in choices:
            return value
        print("输入无效，请重试。")


def interactive(root: Path) -> dict[str, Any]:
    """Run the menu-driven interface requested for human reuse."""

    action = _interactive_choice("请选择操作", ("模型注册", "推理", "打分", "结果汇总/显示"))
    if action == "模型注册":
        checkpoint = Path(input("checkpoint 路径：").strip())
        backend = _interactive_choice("backend", BACKENDS)
        key = _model_key(_normal_checkpoint_path(checkpoint), backend)
        versions = _existing_versions(require_evaluation_root(root), key)
        new_version = True
        if versions:
            print("已有版本：" + ", ".join(path.name for _, path in versions))
            new_version = input("仍要注册新版本？[y/N]：").strip().lower() in {"y", "yes"}
            if not new_version:
                _, existing = load_registration(root, versions[-1][1].name)
                return {"registration_name": existing["registration_name"], "existing": True}
        tokenizer_model = None
        train_wandb_config = None
        model_config_path = None
        vllm_speculative_draft_model = None
        vllm_speculative_verification = None
        vllm_speculative_allow_approximate = False
        if backend == "megatron":
            tokenizer_model = Path(input("tokenizer 目录：").strip())
            train_wandb_config = Path(input("训练 wandb/config JSON：").strip())
            value = input("模型上下文 config.json（回车表示 tokenizer/config.json）：").strip()
            model_config_path = Path(value) if value else None
        elif backend == "vllm":
            value = input("NCP DFlash checkpoint（回车禁用）：").strip()
            vllm_speculative_draft_model = Path(value) if value else None
            if vllm_speculative_draft_model is not None:
                value = input(
                    "target/spec comparison.json（回车仅注册实验，不允许正式推理）："
                ).strip()
                vllm_speculative_verification = Path(value) if value else None
                if vllm_speculative_verification is not None:
                    vllm_speculative_allow_approximate = input(
                        "允许注册会改变 token 的近似加速路径？[y/N]："
                    ).strip().lower() in {"y", "yes"}
        return register_model(
            root=root,
            checkpoint=checkpoint,
            backend=backend,
            new_version=new_version,
            tokenizer_model=tokenizer_model,
            train_wandb_config=train_wandb_config,
            model_config_path=model_config_path,
            vllm_speculative_draft_model=vllm_speculative_draft_model,
            vllm_speculative_verification=vllm_speculative_verification,
            vllm_speculative_allow_approximate=(
                vllm_speculative_allow_approximate
            ),
        )
    registration = input("测评目录名称（如 vllm-abc123-v1）：").strip()
    model_root, metadata = load_registration(root, registration)
    if action == "推理":
        benchmark = _interactive_choice("benchmark", BENCHMARKS)
        data_root = None
        if benchmark in LONG_CONTEXT_LENGTHS:
            discovered = discover_prepared_data(benchmark, Path(str(metadata["tokenizer_model"])))
            if discovered:
                print(f"自动匹配预处理数据：{discovered}")
                data_root = discovered
            else:
                data_root = Path(input("预处理数据根目录：").strip())
        return submit_inference(
            root=root,
            registration_name=registration,
            benchmark=benchmark,
            data_root=data_root,
            executor=DEFAULT_EXECUTOR,
            force=False,
            dry_run=False,
        )
    available = []
    for benchmark in BENCHMARKS:
        state_path = _benchmark_path(model_root, benchmark) / "metadata.json"
        if not state_path.is_file():
            continue
        state = _read_json(state_path)
        if action == "打分":
            attempt = _latest_attempt(state, "inference_attempts")
            status = refresh_status(root, registration, benchmark) if attempt else {}
            if status.get("inference") == "INFERENCE_COMPLETE":
                available.append(benchmark)
        else:
            attempt = _latest_attempt(state, "scoring_attempts")
            if attempt:
                _refresh_jobs(attempt)
            if (
                attempt
                and all(job.get("status") == "Succeeded" for job in attempt["jobs"])
                and _score_artifacts(benchmark, attempt)["complete"]
            ):
                available.append(benchmark)
    if not available:
        raise EvaluationError("当前没有满足前置完整性门槛的 benchmark")
    benchmark = _interactive_choice("可选 benchmark", available)
    if action == "打分":
        return submit_scoring(
            root=root,
            registration_name=registration,
            benchmark=benchmark,
            executor=DEFAULT_EXECUTOR,
            dry_run=False,
        )
    return submit_final(
        root=root,
        registration_name=registration,
        benchmark=benchmark,
        executor=DEFAULT_EXECUTOR,
        dry_run=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EVALUATION_ROOT)
    subparsers = parser.add_subparsers(dest="action")

    register = subparsers.add_parser("register", help="注册 checkpoint/backend")
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--backend", choices=BACKENDS, required=True)
    register.add_argument("--new-version", action="store_true")
    register.add_argument("--tokenizer-model", type=Path)
    register.add_argument("--train-wandb-config", type=Path)
    register.add_argument("--model-config-path", type=Path)
    register.add_argument("--vllm-runtime-config", type=Path)
    register.add_argument("--vllm-speculative-draft-model", type=Path)
    register.add_argument("--vllm-speculative-verification", type=Path)
    register.add_argument(
        "--vllm-speculative-allow-approximate",
        action="store_true",
        help=(
            "explicitly register segmented_kv_approx despite token divergence; "
            "the evaluation version then requires matched downstream A/B"
        ),
    )

    for action in ("infer", "score", "final", "status"):
        command = subparsers.add_parser(action)
        command.add_argument("--evaluation", required=True)
        command.add_argument("--benchmark", choices=BENCHMARKS, required=True)
        if action == "infer":
            command.add_argument("--data-root", type=Path)
            command.add_argument("--force", action="store_true")
        if action in {"infer", "score", "final"}:
            command.add_argument(
                "--executor",
                choices=("emit", "local"),
                default=DEFAULT_EXECUTOR,
                help="emit writes portable task specs; local runs them in the current allocation",
            )
            command.add_argument("--dry-run", action="store_true")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute one interactive or non-interactive CLI action."""

    args = _parser().parse_args(argv)
    require_evaluation_root(args.root)
    if args.action is None:
        return interactive(args.root)
    if args.action == "register":
        return register_model(
            root=args.root,
            checkpoint=args.checkpoint,
            backend=args.backend,
            new_version=args.new_version,
            tokenizer_model=args.tokenizer_model,
            train_wandb_config=args.train_wandb_config,
            model_config_path=args.model_config_path,
            vllm_runtime_config=args.vllm_runtime_config,
            vllm_speculative_draft_model=args.vllm_speculative_draft_model,
            vllm_speculative_verification=args.vllm_speculative_verification,
            vllm_speculative_allow_approximate=(
                args.vllm_speculative_allow_approximate
            ),
        )
    if args.action == "infer":
        return submit_inference(
            root=args.root,
            registration_name=args.evaluation,
            benchmark=args.benchmark,
            data_root=args.data_root,
            executor=args.executor,
            force=args.force,
            dry_run=args.dry_run,
        )
    if args.action == "score":
        return submit_scoring(
            root=args.root,
            registration_name=args.evaluation,
            benchmark=args.benchmark,
            executor=args.executor,
            dry_run=args.dry_run,
        )
    if args.action == "final":
        return submit_final(
            root=args.root,
            registration_name=args.evaluation,
            benchmark=args.benchmark,
            executor=args.executor,
            dry_run=args.dry_run,
        )
    return refresh_status(args.root, args.evaluation, args.benchmark)


def main() -> None:
    """CLI entry point."""

    try:
        result = run_cli()
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
