#!/usr/bin/env python3
"""Render task specs as Kubernetes Job JSON without site-specific defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--pvc", required=True)
    parser.add_argument("--mount-path", default="/shared")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_path in plan["tasks"]:
        spec = json.loads(Path(task_path).read_text(encoding="utf-8"))
        if not spec.get("container_image"):
            raise SystemExit(f"task has no immutable container image: {task_path}")
        resources = spec["resources"]
        limits = {
            "cpu": str(resources["cpus"]),
            "memory": f"{resources['memory_gib']}Gi",
        }
        if resources["gpus"]:
            limits["nvidia.com/gpu"] = str(resources["gpus"])
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": spec["task_id"]},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "evaluation",
                                "image": spec["container_image"],
                                "command": ["python", "-m", "ncp_olmo_eval.task_runner", task_path],
                                "env": [
                                    {"name": key, "value": value}
                                    for key, value in spec["env"].items()
                                ],
                                "resources": {"requests": limits, "limits": limits},
                                "volumeMounts": [{"name": "shared", "mountPath": args.mount_path}],
                            }
                        ],
                        "volumes": [
                            {"name": "shared", "persistentVolumeClaim": {"claimName": args.pvc}}
                        ],
                    }
                },
            },
        }
        (args.output_dir / f"{spec['task_id']}.json").write_text(
            json.dumps(job, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
