# Public OCI images

Release `v0.1.0` defines five Linux/amd64 images:

| Task | Public release tag |
|---|---|
| vLLM inference and non-code scoring | `ghcr.io/luckysjtu/ncp-olmo-eval-runtime:0.1.0` |
| Core88 Python execution | `ghcr.io/luckysjtu/ncp-olmo-eval-core88-python:0.1.0` |
| BigCodeBench execution | `ghcr.io/luckysjtu/ncp-olmo-eval-core88-bigcodebench:0.1.0` |
| DS-1000 execution | `ghcr.io/luckysjtu/ncp-olmo-eval-core88-ds1000:0.1.0` |
| MultiPL-E execution | `ghcr.io/luckysjtu/ncp-olmo-eval-core88-multiple:0.1.0` |

Tags make images discoverable; formal evaluations must use the immutable
`tag@sha256:...` values in the matching release's `public-images.env` or
`public-images.json` asset. A missing digest fails closed.

The initial GitHub Container Registry publication has one manual owner step:
after the workflow first creates each package, set its visibility to **Public**
in GitHub Packages. Package visibility is independent of repository visibility.

## Rebuild from public inputs

Every Dockerfile now has a directly pullable public base pinned by digest.
`configs/public-image-bases.json` records the bases and release tags. The
BigCodeBench parser wheels are fetched from PyPI and checked by SHA-256; the
DS-1000 environment is recreated from a public Python image and exact package
versions. No private registry, local runtime tarball, or cluster cache is a
build input.

Build locally from a clean release checkout:

```bash
SOURCE_REVISION="$(git rev-parse HEAD)" \
  scripts/build-public-images.sh
```

This loads Linux/amd64 images into the local Docker daemon and writes a
non-immutable discovery manifest under `dist/`. To publish them:

```bash
docker login ghcr.io
PUSH=1 SOURCE_REVISION="$(git rev-parse HEAD)" \
  scripts/build-public-images.sh
```

The release workflow performs the push with `GITHUB_TOKEN`, then attaches the
digest manifest to the GitHub release. The runtime build executes the
Minerva/MATH smoke; the BigCodeBench image imports both locked parser modules;
the DS-1000 image imports every pinned scientific runtime package.

## Apptainer

Convert the immutable references in `public-images.env`, not the release tags:

```bash
set -a
source public-images.env
set +a

apptainer pull /shared/images/ncp-olmo-eval-runtime.sif \
  "docker://$NCP_OLMO_EVAL_IMAGE"
```

Populate a private copy of `configs/apptainer-images.example.json` with each
exact OCI digest as the key and the corresponding absolute SIF path as the
value. The checked-in example is intentionally empty because a digest cannot
be known before publication.

## Runtime boundary

Task specs declare an expected image. Local execution assumes the caller has
already entered that image and exported the exact reference as
`NCP_OLMO_ACTIVE_IMAGE`, or uses a container wrapper. Slurm can use
`scripts/run-task-apptainer.sh`; Kubernetes Jobs use the OCI reference
directly. A mismatch fails instead of falling back to host Python.

The host NVIDIA driver must support the CUDA runtime in the pinned vLLM base.
Keep images and sealed data/model assets separate unless the assets' terms
explicitly permit bundling. The MultiPL-E image inherits its upstream
machine-learning-training restriction; see `THIRD_PARTY_NOTICES.md`.
