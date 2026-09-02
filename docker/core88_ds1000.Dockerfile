# syntax=docker/dockerfile:1

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DS1000_ARCHIVE_SHA256
ARG DS1000_MANIFEST_SHA256
ARG CORE88_SCORER_COMMIT
ARG CORE88_UPSTREAM_REFERENCE
ARG http_proxy
ARG https_proxy
ARG no_proxy

USER root

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq; \
    apt-get install -y -qq --no-install-recommends bubblewrap ca-certificates coreutils; \
    rm -rf /var/lib/apt/lists/*; \
    command -v bwrap

COPY ds1000-runtime.tar.gz /tmp/ds1000-runtime.tar.gz

RUN set -eux; \
    printf '%s  %s\n' "${DS1000_ARCHIVE_SHA256}" /tmp/ds1000-runtime.tar.gz \
      | sha256sum -c -; \
    mkdir -p /opt/core88/ds1000; \
    tar -xzf /tmp/ds1000-runtime.tar.gz -C /opt/core88/ds1000; \
    rm /tmp/ds1000-runtime.tar.gz; \
    test -x /opt/core88/ds1000/python/bin/python3; \
    printf '%s  %s\n' \
      "${DS1000_MANIFEST_SHA256}" \
      /opt/core88/ds1000/CORE88_RUNTIME_MANIFEST.json \
      | sha256sum -c -; \
    grep -q '"status": "CORE88_DS1000_RUNTIME_OK"' \
      /opt/core88/ds1000/CORE88_RUNTIME_MANIFEST.json; \
    test "$(/opt/core88/ds1000/python/bin/python3 -c 'import platform; print(platform.python_version())')" = 3.10.13; \
    /opt/core88/ds1000/python/bin/python3 -c \
      'import gensim, matplotlib, numpy, pandas, scipy, seaborn, sklearn, statsmodels, tensorflow, torch, xgboost'; \
    mkdir -p /opt/core88/image; \
    printf '%s\n' \
      '{' \
      '  "status": "CORE88_SANDBOX_IMAGE_OK",' \
      '  "runtime_name": "ds1000-python-3.10.13",' \
      '  "runtime_prefix": "/opt/core88/ds1000/python",' \
      "  \"runtime_manifest_sha256\": \"${DS1000_MANIFEST_SHA256}\"," \
      "  \"upstream_reference\": \"${CORE88_UPSTREAM_REFERENCE}\"," \
      "  \"scorer_commit\": \"${CORE88_SCORER_COMMIT}\"" \
      '}' \
      > /opt/core88/image/RUNTIME.json

ENV CORE88_RUNTIME_PREFIX=/opt/core88/ds1000/python

LABEL org.opencontainers.image.source="https://github.com/allenai/OLMo-core" \
      org.opencontainers.image.description="Pinned Core88 DS-1000 bubblewrap scorer runtime" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
