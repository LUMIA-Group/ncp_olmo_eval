# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
FROM ${BASE_IMAGE}

ARG CORE88_RUNTIME_NAME=core88-python
ARG CORE88_RUNTIME_PREFIX=/usr
ARG CORE88_UPSTREAM_REFERENCE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
ARG CORE88_SCORER_COMMIT=unknown
ARG RELEASE_VERSION=0.1.0
ARG PACKAGE_PYTHON=/usr/local/bin/python3
ARG http_proxy
ARG https_proxy
ARG no_proxy

USER root

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq; \
    apt-get install -y -qq --no-install-recommends bubblewrap ca-certificates coreutils python3-minimal; \
    rm -rf /var/lib/apt/lists/*; \
    command -v bwrap; \
    test -x "${PACKAGE_PYTHON}"; \
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

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN "${PACKAGE_PYTHON}" -m pip install --no-cache-dir --no-deps . \
 && "${PACKAGE_PYTHON}" -c 'import ncp_olmo_eval; print(ncp_olmo_eval.__version__)'

ENV CORE88_RUNTIME_PREFIX=${CORE88_RUNTIME_PREFIX}

LABEL org.opencontainers.image.source="https://github.com/LuckySJTU/ncp_olmo_eval" \
      org.opencontainers.image.description="Pinned Core88 bubblewrap scorer runtime" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
