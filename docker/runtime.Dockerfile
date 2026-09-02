# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_REVISION=unknown
ARG PYTHON_BIN=python3
ARG INSTALL_EXTRAS=vllm,helmet,scoring

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN "${PYTHON_BIN}" -c \
      'import sys; assert sys.version_info >= (3, 12), sys.version' \
 && "${PYTHON_BIN}" -m pip install --no-cache-dir ".[$INSTALL_EXTRAS]" \
 && "${PYTHON_BIN}" -m ncp_olmo_eval.runtime_smoke math

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NCP_OLMO_SOURCE_REVISION=${SOURCE_REVISION} \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

LABEL org.opencontainers.image.source="https://github.com/LuckySJTU/ncp_olmo_eval" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.description="Portable vLLM runtime for NCP-ArchPreview evaluation"

ENTRYPOINT []
CMD ["ncp-olmo-eval", "--help"]
