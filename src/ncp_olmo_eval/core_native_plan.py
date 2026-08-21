#!/usr/bin/env python3
"""Build deterministic machine manifests for hierarchical Core evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core_native_eval import _parse_task_orders
from .core_native_work import write_dispatch_plan


def parse_args() -> argparse.Namespace:
    """Parse the machine-plan command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--profile", default="core88")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--task-orders", default="")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--generation-samples-cap", type=int, default=1)
    parser.add_argument("--max-gen-tokens-cap", type=int, default=0)
    parser.add_argument("--machine-count", type=int, default=4)
    parser.add_argument("--global-seed", type=int, required=True)
    parser.add_argument("--decode-weight", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Build the plan and print its validation metadata."""

    args = parse_args()
    summary = json.loads((args.data_root / "summary.json").read_text(encoding="utf-8"))
    profile = summary.get(args.profile)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown Core profile: {args.profile}")
    task_count = int(profile.get("task_count", len(profile.get("tasks") or [])))
    task_orders = _parse_task_orders(args.task_orders, task_count)
    plan = write_dispatch_plan(
        output_root=args.plan_root,
        data_root=args.data_root,
        profile=args.profile,
        task_orders=task_orders,
        limit_per_task=args.limit_per_task,
        generation_samples_cap=args.generation_samples_cap,
        max_gen_tokens_cap=args.max_gen_tokens_cap,
        machine_count=args.machine_count,
        global_seed=args.global_seed,
        decode_weight=args.decode_weight,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
