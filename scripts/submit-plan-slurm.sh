#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 /absolute/path/to/plan.json" >&2
  exit 2
fi
: "${NCP_OLMO_SLURM_PARTITION:?set NCP_OLMO_SLURM_PARTITION}"

PLAN="$(realpath "$1")"
python - "$PLAN" <<'PY' | while IFS=$'\t' read -r task name gpus cpus memory image; do
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
for path in plan["tasks"]:
    spec = json.load(open(path, encoding="utf-8"))
    r = spec["resources"]
    print(
        path, spec["task_id"], r["gpus"], r["cpus"], r["memory_gib"],
        spec.get("container_image", ""), sep="\t"
    )
PY
  args=(
    --parsable
    --job-name="$name"
    --partition="$NCP_OLMO_SLURM_PARTITION"
    --nodes=1
    --cpus-per-task="$cpus"
    --mem="${memory}G"
    --output="${task%/task.json}/slurm-%j.log"
  )
  if (( gpus > 0 )); then
    args+=(--gpus="$gpus")
  fi
  if [[ -n "${NCP_OLMO_SLURM_ACCOUNT:-}" ]]; then
    args+=(--account="$NCP_OLMO_SLURM_ACCOUNT")
  fi
  command=(python -m ncp_olmo_eval.task_runner "$task")
  if [[ -n "$image" ]]; then
    : "${NCP_OLMO_SLURM_CONTAINER_WRAPPER:?set it to a program accepting IMAGE_REF TASK_JSON}"
    command=("$NCP_OLMO_SLURM_CONTAINER_WRAPPER" "$image" "$task")
  fi
  printf -v wrapped '%q ' "${command[@]}"
  sbatch "${args[@]}" --wrap="$wrapped"
done
