# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ghcr.io/nuprl/multipl-e-evaluation@sha256:8e8aed978fc7817fa51afb2e35b240de734fbbca2df41102f196bc54f7e8218c
FROM ${BASE_IMAGE}

ARG CORE88_RUNTIME_NAME=core88-multiple
ARG CORE88_RUNTIME_PREFIX=/usr
ARG CORE88_UPSTREAM_REFERENCE=ghcr.io/nuprl/multipl-e-evaluation@sha256:8e8aed978fc7817fa51afb2e35b240de734fbbca2df41102f196bc54f7e8218c
ARG CORE88_SCORER_COMMIT=unknown
ARG RELEASE_VERSION=0.1.0a16

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
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"," \
      '  "sandbox_layer": "bubblewrap-over-public-multiple-v1"' \
      '}' \
      > /opt/core88/image/RUNTIME.json

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN python3 -m pip install --no-cache-dir \
      'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1' \
 && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation . \
 && python3 -c 'import ncp_olmo_eval; print(ncp_olmo_eval.__version__)'

ENV CORE88_RUNTIME_PREFIX=${CORE88_RUNTIME_PREFIX} \
    NCP_OLMO_SOURCE_REVISION=${CORE88_SCORER_COMMIT}

# The base image is public and directly reusable, but its upstream license is
# BSD-3-Clause with an additional ML-training restriction. See
# THIRD_PARTY_NOTICES.md; the Apache-2.0 label applies only to this repository's
# added layer.
LABEL org.opencontainers.image.source="https://github.com/LuckySJTU/ncp_olmo_eval" \
      org.opencontainers.image.description="Pinned MultiPL-E scorer over the public v1 runtime" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-MultiPL-E-ML-Restriction" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
