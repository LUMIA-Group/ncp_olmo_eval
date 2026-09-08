# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=python:3.10.13-slim-bookworm@sha256:1326d0fd281d283b077fd249e618339a44c9ca5aae6e05cb4f069a087e827922
FROM ${BASE_IMAGE}

ARG CORE88_UPSTREAM_REFERENCE=python:3.10.13-slim-bookworm@sha256:1326d0fd281d283b077fd249e618339a44c9ca5aae6e05cb4f069a087e827922
ARG CORE88_SCORER_COMMIT=unknown
ARG RELEASE_VERSION=0.1.1

USER root

RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq; \
    apt-get install -y -qq --no-install-recommends \
      bubblewrap ca-certificates coreutils g++ gcc gfortran git libgomp1; \
    rm -rf /var/lib/apt/lists/*; \
    command -v bwrap

RUN python3 -m venv /opt/core88/ds1000/python \
 && /opt/core88/ds1000/python/bin/python -m pip install --no-cache-dir --upgrade \
      'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1' \
 && /opt/core88/ds1000/python/bin/python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.2.0+cpu' \
 && /opt/core88/ds1000/python/bin/python -m pip install --no-cache-dir \
      'numpy==1.26.4' \
      'pandas==1.5.3' \
      'matplotlib==3.8.4' \
      'scipy==1.12.0' \
      'scikit-learn==1.4.0' \
      'seaborn==0.13.2' \
      'statsmodels==0.14.1' \
      'xgboost==2.0.3' \
      'gensim==4.3.2' \
      'tensorflow-cpu==2.16.1'

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN export CORE88_UPSTREAM_REFERENCE="${CORE88_UPSTREAM_REFERENCE}"; \
    export CORE88_SCORER_COMMIT="${CORE88_SCORER_COMMIT}"; \
    /opt/core88/ds1000/python/bin/python -m pip install --no-cache-dir --no-deps .; \
    /opt/core88/ds1000/python/bin/python - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

expected = {
    "numpy": "1.26.4",
    "pandas": "1.5.3",
    "matplotlib": "3.8.4",
    "scipy": "1.12.0",
    "scikit-learn": "1.4.0",
    "seaborn": "0.13.2",
    "statsmodels": "0.14.1",
    "xgboost": "2.0.3",
    "gensim": "4.3.2",
    "torch": "2.2.0+cpu",
    "tensorflow-cpu": "2.16.1",
}
assert platform.python_version() == "3.10.13", platform.python_version()
assert {name: importlib.metadata.version(name) for name in expected} == expected
import gensim, matplotlib, numpy, pandas, scipy, seaborn, sklearn, statsmodels, tensorflow, torch, xgboost  # noqa: F401,E501

root = Path("/opt/core88/image")
root.mkdir(parents=True, exist_ok=True)
(root / "RUNTIME.json").write_text(
    json.dumps(
        {
            "status": "CORE88_SANDBOX_IMAGE_OK",
            "runtime_name": "ds1000-python-3.10.13",
            "runtime_prefix": "/opt/core88/ds1000/python",
            "upstream_reference": os.environ["CORE88_UPSTREAM_REFERENCE"],
            "scorer_commit": os.environ["CORE88_SCORER_COMMIT"],
            "python_version": platform.python_version(),
            "distributions": expected,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

ENV CORE88_RUNTIME_PREFIX=/opt/core88/ds1000/python \
    NCP_OLMO_SOURCE_REVISION=${CORE88_SCORER_COMMIT}

LABEL org.opencontainers.image.source="https://github.com/LUMIA-Group/ncp_olmo_eval" \
      org.opencontainers.image.description="Reproducible DS-1000 scorer runtime built from public packages" \
      org.opencontainers.image.base.name="${CORE88_UPSTREAM_REFERENCE}" \
      org.opencontainers.image.revision="${CORE88_SCORER_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="NCP-ArchPreview contributors"

ENTRYPOINT []
CMD ["/bin/bash"]
