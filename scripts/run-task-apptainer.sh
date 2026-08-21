#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 IMAGE_REF /absolute/path/to/task.json" >&2
  exit 2
fi
: "${NCP_OLMO_APPTAINER_IMAGE_MAP:?set path to immutable OCI-ref to SIF JSON map}"

IMAGE_REF="$1"
TASK="$(realpath "$2")"
SIF="$({
  python - "$NCP_OLMO_APPTAINER_IMAGE_MAP" "$IMAGE_REF" <<'PY'
import json, sys
mapping = json.load(open(sys.argv[1], encoding="utf-8"))
path = mapping.get(sys.argv[2])
if not path:
    raise SystemExit(f"image is not sealed in the Apptainer map: {sys.argv[2]}")
print(path)
PY
})"
if [[ ! -f "$SIF" ]]; then
  echo "sealed Apptainer image is missing: $SIF" >&2
  exit 1
fi

args=(exec --cleanenv --nv)
if [[ -n "${NCP_OLMO_APPTAINER_BINDS:-}" ]]; then
  args+=(--bind "$NCP_OLMO_APPTAINER_BINDS")
fi
export APPTAINERENV_NCP_OLMO_ACTIVE_IMAGE="$IMAGE_REF"
exec apptainer "${args[@]}" "$SIF" python -m ncp_olmo_eval.task_runner "$TASK"
