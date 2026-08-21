#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 1 ]]; then
  echo "usage: $0 /absolute/path/to/plan.json" >&2
  exit 2
fi

PLAN="$(realpath "$1")"
python - "$PLAN" <<'PY' | while IFS=$'\t' read -r task image; do
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("schema_version") != "ncp-olmo-eval-plan-v1":
    raise SystemExit("unsupported task plan")
for path in payload["tasks"]:
    spec = json.load(open(path, encoding="utf-8"))
    print(path, spec.get("container_image", ""), sep="\t")
PY
  if [[ -n "$image" && "$image" != "${NCP_OLMO_ACTIVE_IMAGE:-}" ]]; then
    : "${NCP_OLMO_CONTAINER_WRAPPER:?set it to a program accepting IMAGE_REF TASK_JSON}"
    "$NCP_OLMO_CONTAINER_WRAPPER" "$image" "$task"
  else
    python -m ncp_olmo_eval.task_runner "$task"
  fi
done
