"""Deterministic request-level work planning for hierarchical Core evaluation.

The four machine-level manifests are static and reproducible.  Generation
requests from one Core task may span machines so long generations cannot pin
the global critical path to one host.  Loglikelihood tasks remain whole to
avoid making every host read every scoring dataset.  Within each machine, a
coordinator dynamically hands bounded request batches to the local GPU
workers.  Generation requests are split at ``(example_id, sample_index)``
granularity so correlated samples from one example do not remain pinned to one
worker.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .core_native_contract import (
    CORE88_ASSIGNMENT_STRATEGY,
    CORE88_COST_MODEL,
    CORE88_LOCAL_DISPATCH,
    CORE88_PLAN_SCHEMA,
)
from .core_native_eval import (
    _apply_generation_samples_cap,
    _file_sha256,
    _generation_contract,
    _load_manifest,
    _read_rows,
    _stable_id,
    _task_is_generation,
    _task_sample_count,
    _write_json_atomic,
)

PLAN_SCHEMA = CORE88_PLAN_SCHEMA
WORK_SCHEMA = "core-native-dispatch-work-v1"
COST_MODEL = CORE88_COST_MODEL
ASSIGNMENT_STRATEGY = CORE88_ASSIGNMENT_STRATEGY
SEED_DERIVATION = "sha256-global-task-example-sample-v1"


@dataclass(frozen=True)
class WorkItem:
    """A single independently seeded inference request."""

    request_key: str
    task_order: int
    task: str
    row_index: int
    example_key: str
    sample_index: int | None
    mode: str
    estimated_cost: int

    def to_json(self) -> dict[str, Any]:
        """Return the stable JSON representation stored in machine manifests."""

        return {"schema_version": WORK_SCHEMA, **asdict(self)}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> WorkItem:
        """Load and validate one machine-manifest entry."""

        if payload.get("schema_version") != WORK_SCHEMA:
            raise ValueError(f"unsupported work schema: {payload.get('schema_version')!r}")
        values = dict(payload)
        values.pop("schema_version")
        return cls(**values)


def stable_request_key(
    task: str,
    example_id: Any,
    sample_index: int | None,
) -> str:
    """Return a process- and machine-independent request identity."""

    payload = {
        "task": str(task),
        "example_id": _stable_id(example_id),
        "sample_index": sample_index,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_request_seed(
    global_seed: int,
    task: str,
    example_id: Any,
    sample_index: int,
) -> int:
    """Derive a request seed that is independent of scheduling and topology."""

    payload = {
        "global_seed": int(global_seed),
        "task": str(task),
        "example_id": _stable_id(example_id),
        "sample_index": int(sample_index),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return value % (2**31 - 1)


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _estimated_scoring_cost(row: dict[str, Any]) -> int:
    context_cost = _estimated_tokens(str(row["input"]))
    choices = list(row.get("choices") or [row.get("output", "")])
    return sum(context_cost + _estimated_tokens(str(choice)) for choice in choices)


def _estimated_generation_cost(
    row: dict[str, Any],
    *,
    max_gen_tokens_cap: int,
    decode_weight: int,
) -> int:
    contract = _generation_contract(row)
    if contract["remaining_generation_kwargs"]:
        raise ValueError(f"unsupported generation contract: {contract}")
    max_tokens = int(contract["max_gen_toks"])
    if max_gen_tokens_cap > 0:
        max_tokens = min(max_tokens, int(max_gen_tokens_cap))
    return _estimated_tokens(str(row["input"])) + int(decode_weight) * max_tokens


def build_work_items(
    *,
    data_root: Path,
    profile: str,
    task_orders: set[int],
    limit_per_task: int,
    generation_samples_cap: int,
    max_gen_tokens_cap: int,
    decode_weight: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[WorkItem], dict[str, int]]:
    """Expand a Core profile into deterministic request-level work items."""

    summary, tasks = _load_manifest(data_root, profile, task_orders)
    _apply_generation_samples_cap(tasks, generation_samples_cap)
    items: list[WorkItem] = []
    expected_counts: dict[str, int] = {}
    for task in tasks:
        task["absolute_file"] = str(data_root / str(task["file"]))
        rows = _read_rows(Path(task["absolute_file"]))
        if limit_per_task > 0:
            rows = rows[:limit_per_task]
        task_name = str(task["task"])
        expected_counts[task_name] = len(rows)
        is_generation = _task_is_generation(task)
        for row_index, row in enumerate(rows):
            row_cost = (
                _estimated_generation_cost(
                    row,
                    max_gen_tokens_cap=max_gen_tokens_cap,
                    decode_weight=decode_weight,
                )
                if is_generation
                else _estimated_scoring_cost(row)
            )
            sample_indices: Iterable[int | None] = (
                range(_task_sample_count(task))
                if is_generation
                else (None,)
            )
            for sample_index in sample_indices:
                items.append(
                    WorkItem(
                        request_key=stable_request_key(
                            task_name,
                            row["example_id"],
                            sample_index,
                        ),
                        task_order=int(task["task_order"]),
                        task=task_name,
                        row_index=row_index,
                        example_key=_stable_id(row["example_id"]),
                        sample_index=sample_index,
                        mode="generation" if is_generation else "loglikelihood",
                        estimated_cost=row_cost,
                    )
                )
    return summary, tasks, items, expected_counts


def assign_work_items(
    items: Iterable[WorkItem],
    machine_count: int,
) -> tuple[list[list[WorkItem]], list[int]]:
    """Assign generation requests and whole scoring tasks with global LPT.

    A generation task may span machines.  Request identity, sampling seed,
    resume, and aggregation are all machine-independent, so this removes
    generation stragglers without changing the evaluation contract.  Scoring
    tasks remain whole for data locality.  Stable tie-breakers make the four
    manifests reproducible across processes and retries.
    """

    if machine_count <= 0:
        raise ValueError(f"machine_count must be positive, got {machine_count}")
    assignments: list[list[WorkItem]] = [[] for _ in range(machine_count)]
    loads = [0] * machine_count
    heap = [(0, machine_index) for machine_index in range(machine_count)]
    heapq.heapify(heap)
    scoring_items_by_task: dict[tuple[int, str], list[WorkItem]] = {}
    units: list[tuple[int, int, str, str, WorkItem | list[WorkItem]]] = []
    for item in items:
        if item.mode == "generation":
            units.append(
                (
                    item.estimated_cost,
                    item.task_order,
                    item.task,
                    item.request_key,
                    item,
                )
            )
        elif item.mode == "loglikelihood":
            scoring_items_by_task.setdefault((item.task_order, item.task), []).append(
                item
            )
        else:
            raise ValueError(f"unsupported Core work mode: {item.mode!r}")
    for (task_order, task), task_items in scoring_items_by_task.items():
        ordered_task_items = sorted(task_items, key=lambda item: item.request_key)
        units.append(
            (
                sum(item.estimated_cost for item in ordered_task_items),
                task_order,
                task,
                "",
                ordered_task_items,
            )
        )
    ordered_units = sorted(
        units,
        key=lambda unit: (-unit[0], unit[1], unit[2], unit[3]),
    )
    for unit_cost, _, _, _, payload in ordered_units:
        load, machine_index = heapq.heappop(heap)
        unit_items = [payload] if isinstance(payload, WorkItem) else payload
        assignments[machine_index].extend(unit_items)
        loads[machine_index] += unit_cost
        heapq.heappush(heap, (load + unit_cost, machine_index))
    return assignments, loads


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_dispatch_plan(
    *,
    output_root: Path,
    data_root: Path,
    profile: str,
    task_orders: set[int],
    limit_per_task: int,
    generation_samples_cap: int,
    max_gen_tokens_cap: int,
    machine_count: int,
    global_seed: int,
    decode_weight: int,
) -> dict[str, Any]:
    """Create or validate four deterministic machine-level manifests."""

    summary, tasks, items, expected_counts = build_work_items(
        data_root=data_root,
        profile=profile,
        task_orders=task_orders,
        limit_per_task=limit_per_task,
        generation_samples_cap=generation_samples_cap,
        max_gen_tokens_cap=max_gen_tokens_cap,
        decode_weight=decode_weight,
    )
    assignments, loads = assign_work_items(items, machine_count)
    planning_contract = {
        "schema_version": PLAN_SCHEMA,
        "data_root": str(data_root.resolve()),
        "data_summary_sha256": _file_sha256(data_root / "summary.json"),
        "source_schema_version": summary.get("schema_version"),
        "profile": profile,
        "task_orders": [int(task["task_order"]) for task in tasks],
        "limit_per_task": int(limit_per_task),
        "generation_samples_cap": int(generation_samples_cap),
        "max_gen_tokens_cap": int(max_gen_tokens_cap),
        "machine_count": int(machine_count),
        "global_seed": int(global_seed),
        "cost_model": COST_MODEL,
        "assignment_strategy": ASSIGNMENT_STRATEGY,
        "local_dispatch": CORE88_LOCAL_DISPATCH,
        "seed_derivation": SEED_DERIVATION,
        "decode_weight": int(decode_weight),
        "expected_counts": expected_counts,
        "work_item_count": len(items),
    }
    plan_id = _canonical_json_sha256(planning_contract)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_id") != plan_id:
            raise RuntimeError(
                f"dispatch plan contract changed at {plan_path}: "
                f"{existing.get('plan_id')} != {plan_id}"
            )
        for metadata in existing["machine_manifests"]:
            path = output_root / Path(str(metadata["path"])).name
            if _file_sha256(path) != metadata["sha256"]:
                raise RuntimeError(f"existing machine manifest hash mismatch: {path}")
        return existing

    machine_manifests: list[dict[str, Any]] = []
    for machine_index, machine_items in enumerate(assignments):
        path = output_root / f"machine-{machine_index:02d}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in machine_items:
                handle.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")
        temporary.replace(path)
        machine_manifests.append(
            {
                "machine_index": machine_index,
                "path": str(path),
                "sha256": _file_sha256(path),
                "task_orders": sorted(
                    {int(item.task_order) for item in machine_items}
                ),
                "task_count": len({item.task for item in machine_items}),
                "work_item_count": len(machine_items),
                "estimated_cost": loads[machine_index],
            }
        )
    plan = {
        **planning_contract,
        "plan_id": plan_id,
        "machine_manifests": machine_manifests,
    }
    _write_json_atomic(plan_path, plan)
    return plan


def load_machine_manifest(
    plan_root: Path,
    machine_index: int,
) -> tuple[dict[str, Any], list[WorkItem]]:
    """Load one machine manifest and verify it against the global plan."""

    plan_path = plan_root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"unsupported dispatch plan: {plan.get('schema_version')!r}")
    machine_count = int(plan["machine_count"])
    if not 0 <= machine_index < machine_count:
        raise ValueError(
            f"machine_index={machine_index} is outside machine_count={machine_count}"
        )
    metadata = plan["machine_manifests"][machine_index]
    path = plan_root / Path(str(metadata["path"])).name
    if _file_sha256(path) != metadata["sha256"]:
        raise RuntimeError(f"machine manifest hash mismatch: {path}")
    items = [
        WorkItem.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(items) != int(metadata["work_item_count"]):
        raise RuntimeError(f"machine manifest count mismatch: {path}")
    return plan, items
