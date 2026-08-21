# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_REVISION=unknown

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates git python3.12 python3.12-dev python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ncp-olmo-eval
COPY . /opt/ncp-olmo-eval
RUN python3.12 -m pip install --no-cache-dir --break-system-packages '.[vllm,helmet]'

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NCP_OLMO_SOURCE_REVISION=${SOURCE_REVISION} \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

LABEL org.opencontainers.image.source="https://github.com/LuckySJTU/ncp_olmo_eval" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.description="Portable vLLM runtime for NCP OLMo evaluation"

ENTRYPOINT []
CMD ["ncp-olmo-eval", "--help"]
