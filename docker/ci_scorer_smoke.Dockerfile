# syntax=docker/dockerfile:1.7

ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}

ARG OLMO_EVAL_COMMIT

RUN python3 -m pip install --no-cache-dir \
      tree-sitter==0.25.2 tree-sitter-python==0.25.0

COPY .ci/olmo-eval /opt/olmo-eval

ENV OLMO_EVAL_ROOT=/opt/olmo-eval \
    OLMO_EVAL_COMMIT=${OLMO_EVAL_COMMIT}

RUN python3 -m ncp_olmo_eval.runtime_smoke bigcodebench

ENTRYPOINT []
CMD ["python3", "-m", "ncp_olmo_eval.runtime_smoke", "all"]
