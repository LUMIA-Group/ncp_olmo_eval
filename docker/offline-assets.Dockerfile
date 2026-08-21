# syntax=docker/dockerfile:1.7

ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}

# Build context must contain a previously prepared and verified `offline-assets`
# directory. Model/checkpoint redistribution remains the publisher's responsibility.
COPY offline-assets /opt/ncp-olmo-eval-assets
RUN python -m ncp_olmo_eval.assets verify \
      --lock /opt/ncp-olmo-eval-assets/assets.lock.json

ENV HF_HOME=/opt/ncp-olmo-eval-assets/hf-home \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
