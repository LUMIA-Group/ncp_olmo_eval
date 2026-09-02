# syntax=docker/dockerfile:1

ARG TOOLS_IMAGE
ARG BASE_IMAGE
FROM ${TOOLS_IMAGE} AS sandbox_tools
FROM ${BASE_IMAGE}

ARG CORE88_RUNTIME_NAME
ARG CORE88_RUNTIME_PREFIX
ARG CORE88_UPSTREAM_REFERENCE
ARG CORE88_SCORER_COMMIT

USER root

COPY --from=sandbox_tools /usr/bin/bwrap /usr/bin/bwrap

RUN set -eux; \
    /usr/bin/bwrap --version; \
    test -x "${CORE88_RUNTIME_PREFIX}/bin/python3"; \
    mkdir -p /opt/core88/image; \
    printf '%s\n' \
      '{' \
      '  "status": "CORE88_SANDBOX_IMAGE_OK",' \
      "  \"runtime_name\": \"${CORE88_RUNTIME_NAME}\"," \
      "  \"runtime_prefix\": \"${CORE88_RUNTIME_PREFIX}\"," \
      "  \"upstream_reference\": \"${CORE88_UPSTREAM_REFERENCE}\"," \
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"," \
      '  "sandbox_layer": "bubblewrap-binary-copy"' \
      '}' \
      > /opt/core88/image/RUNTIME.json

ENV CORE88_RUNTIME_PREFIX=${CORE88_RUNTIME_PREFIX}

LABEL org.opencontainers.image.source="https://github.com/allenai/OLMo-core" \
      org.opencontainers.image.description="Pinned thin Core88 bubblewrap scorer layer" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
