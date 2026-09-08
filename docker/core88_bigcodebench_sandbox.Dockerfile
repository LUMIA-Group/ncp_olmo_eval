# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=bigcodebench/bigcodebench-gradio@sha256:3ca66b54f218649aff5f4e1f54e4d0f43fdb9632d65a08010264b2608e3faec8
FROM ${BASE_IMAGE}

ARG BASE_REFERENCE=bigcodebench/bigcodebench-gradio@sha256:3ca66b54f218649aff5f4e1f54e4d0f43fdb9632d65a08010264b2608e3faec8
ARG CORE88_UPSTREAM_REFERENCE=allenai/OLMo-Eval@f8816eea36563f27b4a9dd2533d68d34f3c67d3f
ARG CORE88_SCORER_COMMIT=unknown
ARG RELEASE_VERSION=0.1.0
ARG TREE_SITTER_WHEEL=.image-build/tree_sitter-0.25.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
ARG TREE_SITTER_SHA256=20b570690f87f1da424cd690e51cc56728d21d63f4abd4b326d382a30353acc7
ARG TREE_SITTER_PYTHON_WHEEL=.image-build/tree_sitter_python-0.25.0-cp310-abi3-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl
ARG TREE_SITTER_PYTHON_SHA256=86f118e5eecad616ecdb81d171a36dde9bef5a0b21ed71ea9c3e390813c3baf5

USER root

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq; \
    apt-get install -y -qq --no-install-recommends bubblewrap ca-certificates coreutils; \
    rm -rf /var/lib/apt/lists/*; \
    command -v bwrap

COPY ${TREE_SITTER_WHEEL} /tmp/core88-wheels/tree_sitter.whl
COPY ${TREE_SITTER_PYTHON_WHEEL} /tmp/core88-wheels/tree_sitter_python.whl

RUN set -eux; \
    echo "${TREE_SITTER_SHA256}  /tmp/core88-wheels/tree_sitter.whl" | sha256sum -c -; \
    echo "${TREE_SITTER_PYTHON_SHA256}  /tmp/core88-wheels/tree_sitter_python.whl" | sha256sum -c -; \
    mkdir -p /opt/core88/olmo-eval-deps /opt/core88/image; \
    python3 -m zipfile -e /tmp/core88-wheels/tree_sitter.whl /opt/core88/olmo-eval-deps; \
    python3 -m zipfile -e /tmp/core88-wheels/tree_sitter_python.whl /opt/core88/olmo-eval-deps; \
    test -f /opt/core88/olmo-eval-deps/tree_sitter/__init__.py; \
    test -f /opt/core88/olmo-eval-deps/tree_sitter_python/_binding.abi3.so; \
    rm -rf /tmp/core88-wheels; \
    printf '%s\n' \
      '{' \
      '  "status": "CORE88_SANDBOX_IMAGE_OK",' \
      '  "runtime_name": "bigcodebench-v0.2.4-f8816ee",' \
      '  "runtime_prefix": "/usr/local",' \
      "  \"upstream_reference\": \"${CORE88_UPSTREAM_REFERENCE}\"," \
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"," \
      '  "sandbox_layer": "bubblewrap-plus-locked-parser-wheels-no-task-registry-import",' \
      "  \"tree_sitter_sha256\": \"${TREE_SITTER_SHA256}\"," \
      "  \"tree_sitter_python_sha256\": \"${TREE_SITTER_PYTHON_SHA256}\"" \
      '}' \
      > /opt/core88/image/RUNTIME.json

ENV CORE88_RUNTIME_PREFIX=/usr/local \
    CORE88_OLMO_EVAL_DEPS=/opt/core88/olmo-eval-deps \
    PYTHONPATH=/opt/core88/olmo-eval-deps \
    NCP_OLMO_SOURCE_REVISION=${CORE88_SCORER_COMMIT}

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN python3 -m pip install --no-cache-dir --no-deps . \
 && python3 -c 'import tree_sitter, tree_sitter_python' \
 && python3 -c 'import ncp_olmo_eval; print(ncp_olmo_eval.__version__)'

LABEL org.opencontainers.image.source="https://github.com/LuckySJTU/ncp_olmo_eval" \
      org.opencontainers.image.description="Pinned BigCodeBench scorer over the official public runtime" \
      org.opencontainers.image.base.name="${BASE_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
