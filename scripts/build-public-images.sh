#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-0.1.0a15}"
REGISTRY="${REGISTRY:-ghcr.io/luckysjtu}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH="${PUSH:-0}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git rev-parse HEAD)}"
BUILD_DIR="${BUILD_DIR:-.image-build}"
OUTPUT_DIR="${OUTPUT_DIR:-dist}"
IMAGE_KEY="${IMAGE_KEY:-}"

IMAGE_KEYS=(
  NCP_OLMO_EVAL_IMAGE
  CORE88_PYTHON_IMAGE
  CORE88_BIGCODEBENCH_IMAGE
  CORE88_DS1000_IMAGE
  CORE88_MULTIPLE_IMAGE
)

if [[ -n "$IMAGE_KEY" ]]; then
  supported=0
  for candidate in "${IMAGE_KEYS[@]}"; do
    if [[ "$IMAGE_KEY" == "$candidate" ]]; then
      supported=1
      break
    fi
  done
  if [[ "$supported" != "1" ]]; then
    echo "Unsupported IMAGE_KEY: $IMAGE_KEY" >&2
    exit 2
  fi
fi

if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a full Git commit SHA" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

check_sha256() {
  local path="$1"
  local expected="$2"
  local observed
  if command -v sha256sum >/dev/null 2>&1; then
    observed="$(sha256sum "$path" | awk '{print $1}')"
  else
    observed="$(shasum -a 256 "$path" | awk '{print $1}')"
  fi
  [[ "$observed" == "$expected" ]]
}

fetch_locked() {
  local url="$1"
  local output="$2"
  local sha256="$3"
  if [[ ! -f "$output" ]] || ! check_sha256 "$output" "$sha256"; then
    curl --fail --location --retry 5 --retry-all-errors "$url" --output "$output"
  fi
  check_sha256 "$output" "$sha256"
}

fetch_locked \
  'https://files.pythonhosted.org/packages/c5/a4/68ae301626f2393a62119481cb660eb93504a524fc741a6f1528a4568cf6/tree_sitter-0.25.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl' \
  "$BUILD_DIR/tree_sitter-0.25.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl" \
  '20b570690f87f1da424cd690e51cc56728d21d63f4abd4b326d382a30353acc7'
fetch_locked \
  'https://files.pythonhosted.org/packages/aa/cb/d9b0b67d037922d60cbe0359e0c86457c2da721bc714381a63e2c8e35eba/tree_sitter_python-0.25.0-cp310-abi3-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl' \
  "$BUILD_DIR/tree_sitter_python-0.25.0-cp310-abi3-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl" \
  '86f118e5eecad616ecdb81d171a36dde9bef5a0b21ed71ea9c3e390813c3baf5'

if [[ "$PUSH" == "1" ]]; then
  output_args=(--push)
else
  output_args=(--load)
fi

build_image() {
  local key="$1"
  local suffix="$2"
  local dockerfile="$3"
  shift 3
  local tag="${REGISTRY}/ncp-olmo-eval-${suffix}:${VERSION}"
  docker buildx build \
    --platform "$PLATFORM" \
    --file "$dockerfile" \
    --tag "$tag" \
    --metadata-file "$OUTPUT_DIR/${key}.metadata.json" \
    --build-arg "RELEASE_VERSION=$VERSION" \
    "${output_args[@]}" \
    "$@" \
    .
}

if [[ -z "$IMAGE_KEY" || "$IMAGE_KEY" == "NCP_OLMO_EVAL_IMAGE" ]]; then
  build_image NCP_OLMO_EVAL_IMAGE runtime docker/runtime.Dockerfile \
    --build-arg "SOURCE_REVISION=$SOURCE_REVISION"
fi
if [[ -z "$IMAGE_KEY" || "$IMAGE_KEY" == "CORE88_PYTHON_IMAGE" ]]; then
  build_image CORE88_PYTHON_IMAGE core88-python docker/core88_sandbox.Dockerfile \
    --build-arg "CORE88_SCORER_COMMIT=$SOURCE_REVISION"
fi
if [[ -z "$IMAGE_KEY" || "$IMAGE_KEY" == "CORE88_BIGCODEBENCH_IMAGE" ]]; then
  build_image CORE88_BIGCODEBENCH_IMAGE core88-bigcodebench \
    docker/core88_bigcodebench_sandbox.Dockerfile \
    --build-arg "CORE88_SCORER_COMMIT=$SOURCE_REVISION"
fi
if [[ -z "$IMAGE_KEY" || "$IMAGE_KEY" == "CORE88_DS1000_IMAGE" ]]; then
  build_image CORE88_DS1000_IMAGE core88-ds1000 docker/core88_ds1000.Dockerfile \
    --build-arg "CORE88_SCORER_COMMIT=$SOURCE_REVISION"
fi
if [[ -z "$IMAGE_KEY" || "$IMAGE_KEY" == "CORE88_MULTIPLE_IMAGE" ]]; then
  build_image CORE88_MULTIPLE_IMAGE core88-multiple docker/core88_thin_sandbox.Dockerfile \
    --build-arg "CORE88_SCORER_COMMIT=$SOURCE_REVISION"
fi

SOURCE_REVISION="$SOURCE_REVISION" VERSION="$VERSION" REGISTRY="$REGISTRY" \
OUTPUT_DIR="$OUTPUT_DIR" PUSH="$PUSH" IMAGE_KEY="$IMAGE_KEY" python3 - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
version = os.environ["VERSION"]
registry = os.environ["REGISTRY"]
source_revision = os.environ["SOURCE_REVISION"]
suffixes = {
    "NCP_OLMO_EVAL_IMAGE": "runtime",
    "CORE88_PYTHON_IMAGE": "core88-python",
    "CORE88_BIGCODEBENCH_IMAGE": "core88-bigcodebench",
    "CORE88_DS1000_IMAGE": "core88-ds1000",
    "CORE88_MULTIPLE_IMAGE": "core88-multiple",
}
images = {}
selected = os.environ["IMAGE_KEY"]
keys = [selected] if selected else list(suffixes)
for key in keys:
    suffix = suffixes[key]
    metadata = json.loads((output / f"{key}.metadata.json").read_text(encoding="utf-8"))
    digest = metadata.get("containerimage.digest")
    tag = f"{registry}/ncp-olmo-eval-{suffix}:{version}"
    if os.environ["PUSH"] == "1" and not digest:
        raise RuntimeError(f"pushed image metadata has no digest: {key}")
    images[key] = f"{tag}@{digest}" if digest and os.environ["PUSH"] == "1" else tag

payload = {
    "schema_version": "ncp-olmo-eval-published-images-v1",
    "release_version": version,
    "source_revision": source_revision,
    "immutable": os.environ["PUSH"] == "1",
    "images": images,
}
suffix = f"-{selected}" if selected else ""
(output / f"public-images{suffix}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(output / f"public-images{suffix}.env").write_text(
    "\n".join(f"{key}={value}" for key, value in images.items()) + "\n",
    encoding="utf-8",
)
PY

if [[ "$PUSH" == "1" ]]; then
  manifest_suffix="${IMAGE_KEY:+-$IMAGE_KEY}"
  expected_count=5
  if [[ -n "$IMAGE_KEY" ]]; then
    expected_count=1
  fi
  test "$(grep -Ec '@sha256:[0-9a-f]{64}$' "$OUTPUT_DIR/public-images${manifest_suffix}.env")" = "$expected_count"
fi

rm -rf "$BUILD_DIR"
printf 'Image manifest: %s\n' "$OUTPUT_DIR/public-images${IMAGE_KEY:+-$IMAGE_KEY}.json"
