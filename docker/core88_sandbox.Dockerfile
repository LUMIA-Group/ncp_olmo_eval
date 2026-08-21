# syntax=docker/dockerfile:1

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CORE88_RUNTIME_NAME
ARG CORE88_RUNTIME_PREFIX
ARG CORE88_UPSTREAM_REFERENCE
ARG CORE88_SCORER_COMMIT
ARG http_proxy
ARG https_proxy
ARG no_proxy

USER root

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq; \
    apt-get install -y -qq --no-install-recommends bubblewrap ca-certificates coreutils; \
    rm -rf /var/lib/apt/lists/*; \
    command -v bwrap; \
    test -x "${CORE88_RUNTIME_PREFIX}/bin/python3"; \
    mkdir -p /opt/core88/image; \
    printf '%s\n' \
      '{' \
      '  "status": "CORE88_SANDBOX_IMAGE_OK",' \
      "  \"runtime_name\": \"${CORE88_RUNTIME_NAME}\"," \
      "  \"runtime_prefix\": \"${CORE88_RUNTIME_PREFIX}\"," \
      "  \"upstream_reference\": \"${CORE88_UPSTREAM_REFERENCE}\"," \
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"" \
      '}' \
      > /opt/core88/image/RUNTIME.json

ENV CORE88_RUNTIME_PREFIX=${CORE88_RUNTIME_PREFIX}

LABEL org.opencontainers.image.source="https://github.com/allenai/OLMo-core" \
      org.opencontainers.image.description="Pinned Core88 bubblewrap scorer runtime" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.vendor="NCP OLMo contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
