# syntax=docker/dockerfile:1

ARG TOOLS_IMAGE
ARG BASE_IMAGE
FROM ${TOOLS_IMAGE} AS sandbox_tools
FROM ${BASE_IMAGE}

ARG CORE88_UPSTREAM_REFERENCE
ARG CORE88_SCORER_COMMIT
ARG TREE_SITTER_WHEEL
ARG TREE_SITTER_SHA256
ARG TREE_SITTER_PYTHON_WHEEL
ARG TREE_SITTER_PYTHON_SHA256

USER root

COPY --from=sandbox_tools /usr/bin/bwrap /usr/bin/bwrap
COPY ${TREE_SITTER_WHEEL} /tmp/core88-wheels/
COPY ${TREE_SITTER_PYTHON_WHEEL} /tmp/core88-wheels/

RUN set -eux; \
    echo "${TREE_SITTER_SHA256}  /tmp/core88-wheels/${TREE_SITTER_WHEEL}" | sha256sum -c -; \
    echo "${TREE_SITTER_PYTHON_SHA256}  /tmp/core88-wheels/${TREE_SITTER_PYTHON_WHEEL}" | sha256sum -c -; \
    mkdir -p /opt/core88/olmo-eval-deps /opt/core88/image; \
    /usr/local/bin/python3 -m zipfile -e \
      "/tmp/core88-wheels/${TREE_SITTER_WHEEL}" /opt/core88/olmo-eval-deps; \
    /usr/local/bin/python3 -m zipfile -e \
      "/tmp/core88-wheels/${TREE_SITTER_PYTHON_WHEEL}" /opt/core88/olmo-eval-deps; \
    test -f /opt/core88/olmo-eval-deps/tree_sitter/__init__.py; \
    test -f /opt/core88/olmo-eval-deps/tree_sitter_python/_binding.abi3.so; \
    rm -rf /tmp/core88-wheels; \
    /usr/bin/bwrap --version; \
    test -x /usr/local/bin/python3; \
    printf '%s\n' \
      '{' \
      '  "status": "CORE88_SANDBOX_IMAGE_OK",' \
      '  "runtime_name": "bigcodebench-f8816ee",' \
      '  "runtime_prefix": "/usr/local",' \
      "  \"upstream_reference\": \"${CORE88_UPSTREAM_REFERENCE}\"," \
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"," \
      '  "sandbox_layer": "bubblewrap-plus-locked-parser-wheels-no-task-registry-import",' \
      "  \"tree_sitter_sha256\": \"${TREE_SITTER_SHA256}\"," \
      "  \"tree_sitter_python_sha256\": \"${TREE_SITTER_PYTHON_SHA256}\"" \
      '}' \
      > /opt/core88/image/RUNTIME.json

ENV CORE88_RUNTIME_PREFIX=/usr/local
ENV CORE88_OLMO_EVAL_DEPS=/opt/core88/olmo-eval-deps

LABEL org.opencontainers.image.source="https://github.com/allenai/OLMo-core" \
      org.opencontainers.image.description="Pinned BigCodeBench scorer without OLMo-Eval registry imports" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.vendor="NCP OLMo contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
