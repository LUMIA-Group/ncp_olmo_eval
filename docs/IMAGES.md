# OCI images

## Inference/scoring runtime

Build from a clean source revision:

```bash
docker build -f docker/runtime.Dockerfile \
  --build-arg BASE_IMAGE='nvidia/cuda@sha256:REPLACE_WITH_PINNED_DIGEST' \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t ghcr.io/ORG/ncp-olmo-eval:0.1.0a3 .
docker push ghcr.io/ORG/ncp-olmo-eval:0.1.0a3
docker inspect --format '{{index .RepoDigests 0}}' \
  ghcr.io/ORG/ncp-olmo-eval:0.1.0a3
```

`SOURCE_REVISION` must be the full 40-character source revision used for the
build. Workers also record and verify a path-independent SHA-256 of the
installed evaluator source tree, so the runtime image does not need `.git`.

Record the resulting digest in `NCP_OLMO_EVAL_IMAGE`; formal plans should not
use mutable tags. The host NVIDIA driver must support the image CUDA runtime.

## Core88 sandbox images

The four recipes under `docker/core88_*.Dockerfile` are layers over pinned
upstream runtime images. Build arguments intentionally have no private default.
Supply base images, wheel/archive files, upstream references, scorer commit,
and SHA-256 values explicitly. Record final digests in the four `CORE88_*_IMAGE`
variables from `configs/runtime.env.example`.

Task specs only declare the expected image. Local execution assumes the caller
already entered that image and exported its exact reference as
`NCP_OLMO_ACTIVE_IMAGE`, or uses a container wrapper. Slurm can use
`scripts/run-task-apptainer.sh` plus a private copy of
`configs/apptainer-images.example.json`; Kubernetes Jobs use the OCI reference
directly. A mismatch fails rather than falling back to host Python.

## Reproducibility notes

- Do not place proxy URLs, credentials, private registries, or local caches in
  Dockerfiles.
- Resolve Python packages during a controlled image build and archive the image
  digest/SBOM. For air-gapped builds, use a pre-populated wheelhouse.
- Keep runtime images and sealed data/model assets separate unless their
  redistribution terms explicitly permit bundling.
