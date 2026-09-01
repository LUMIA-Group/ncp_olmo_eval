"""Scheduler-neutral task specifications for reproducible evaluation.

The evaluator never submits to a cluster API.  It emits this small, stable
contract; a local process, Slurm allocation, Kubernetes Job, or another site
adapter can execute the exact same ``argv`` without translating shell text.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

TASK_SCHEMA_VERSION = "ncp-olmo-eval-task-v1"
PLAN_SCHEMA_VERSION = "ncp-olmo-eval-plan-v1"
TASK_STATES = frozenset({"Planned", "Running", "Succeeded", "Failed"})
_SECRET_NAME = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY)", re.I)
_NON_SECRET_TOKEN_ENV_NAMES = frozenset(
    {
        "CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW",
        "CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH",
    }
)
_IMMUTABLE_OCI_IMAGE = re.compile(r".+@sha256:[0-9a-fA-F]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclasses.dataclass(frozen=True)
class Resources:
    """Portable resource request interpreted by the selected executor."""

    gpus: int = 0
    cpus: int = 1
    memory_gib: int = 4
    nodes: int = 1

    def validate(self) -> None:
        if self.gpus < 0 or self.cpus <= 0 or self.memory_gib <= 0 or self.nodes <= 0:
            raise ValueError(f"invalid resources: {self}")

    def as_json(self) -> dict[str, int]:
        self.validate()
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """One command that must run inside a resource allocation."""

    task_id: str
    phase: str
    benchmark: str
    role: str
    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str]
    resources: Resources
    output_root: str
    status_path: str
    log_path: str
    container_image: str = ""

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", self.task_id):
            raise ValueError(f"invalid task_id: {self.task_id!r}")
        if not self.argv or any(not isinstance(value, str) or not value for value in self.argv):
            raise ValueError("argv must contain non-empty strings")
        if not Path(self.cwd).is_absolute():
            raise ValueError("cwd must be absolute")
        for name, value in self.env.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError("env must map strings to strings")
            if _SECRET_NAME.search(name) and name not in _NON_SECRET_TOKEN_ENV_NAMES:
                raise ValueError(
                    f"task specs must not embed credentials ({name}); inject them at execution time"
                )
        for value in (self.output_root, self.status_path, self.log_path):
            if not Path(value).is_absolute():
                raise ValueError("output_root/status_path/log_path must be absolute")
        if self.container_image and not _IMMUTABLE_OCI_IMAGE.fullmatch(self.container_image):
            raise ValueError("container_image must be an immutable OCI reference with @sha256")
        self.resources.validate()

    def as_json(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": self.task_id,
            "phase": self.phase,
            "benchmark": self.benchmark,
            "role": self.role,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(sorted(self.env.items())),
            "resources": self.resources.as_json(),
            "output_root": self.output_root,
            "status_path": self.status_path,
            "log_path": self.log_path,
            "container_image": self.container_image,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        if payload.get("schema_version") != TASK_SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema: {payload.get('schema_version')}")
        resources = payload.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("task resources must be an object")
        spec = cls(
            task_id=str(payload["task_id"]),
            phase=str(payload["phase"]),
            benchmark=str(payload["benchmark"]),
            role=str(payload["role"]),
            argv=tuple(str(value) for value in payload["argv"]),
            cwd=str(payload["cwd"]),
            env={str(key): str(value) for key, value in dict(payload.get("env", {})).items()},
            resources=Resources(
                gpus=int(resources.get("gpus", 0)),
                cpus=int(resources.get("cpus", 1)),
                memory_gib=int(resources.get("memory_gib", 4)),
                nodes=int(resources.get("nodes", 1)),
            ),
            output_root=str(payload["output_root"]),
            status_path=str(payload["status_path"]),
            log_path=str(payload["log_path"]),
            container_image=str(payload.get("container_image", "")),
        )
        spec.validate()
        return spec


def read_task(path: Path) -> TaskSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task spec must be an object: {path}")
    return TaskSpec.from_json(payload)


def write_task(path: Path, spec: TaskSpec) -> None:
    _atomic_json(path, spec.as_json())


def write_status(
    path: Path,
    *,
    spec: TaskSpec,
    state: str,
    returncode: int | None = None,
    detail: str = "",
) -> None:
    if state not in TASK_STATES:
        raise ValueError(f"unsupported task state: {state}")
    payload: dict[str, Any] = {
        "schema_version": "ncp-olmo-eval-task-status-v1",
        "task_id": spec.task_id,
        "state": state,
        "updated_at": _utc_now(),
        "status_path": str(path),
        "log_path": spec.log_path,
    }
    if returncode is not None:
        payload["returncode"] = returncode
    if detail:
        payload["detail"] = detail[-4000:]
    _atomic_json(path, payload)


def read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "Planned", "status_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("state") not in TASK_STATES:
        raise ValueError(f"invalid task status: {path}")
    return payload


def write_plan(path: Path, task_paths: Sequence[Path]) -> None:
    _atomic_json(
        path,
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "tasks": [str(task.resolve()) for task in task_paths],
        },
    )
