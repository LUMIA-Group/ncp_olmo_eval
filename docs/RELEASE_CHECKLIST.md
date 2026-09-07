# Release checklist

Use this checklist from a clean checkout of the exact source revision intended
for publication. Do not create a tag or publish an artifact until every
applicable check below passes and the release revision is approved.

## 1. Source and protocol review

- Confirm `git status --short` is empty before generating formal artifacts.
- Review the diff from the previous public tag and classify every change as a
  protocol, runtime, scorer, portability, documentation, or test change.
- For any protocol change, update `docs/PROTOCOLS.md`, tests, result metadata or
  schema as needed, and `CHANGELOG.md`.
- Confirm the public backend list, benchmark list, seeds, batching, sample
  semantics, sequence lengths, prompt hashes, and pinned upstream revisions
  agree between code and documentation.
- Confirm the package version is still the intended release version. Version
  bumping and tagging are explicit release actions, not preflight cleanup.

## 2. Public-tree hygiene

The published source must not contain:

- private mounts, registries, cluster endpoints, accounts, queues, or proxies;
- model weights, benchmark data, mutable cache contents, or generated results;
- credentials, API keys, judge secrets, registry credentials, or signed URLs;
- task plans or runtime defaults tied to one scheduler.

Run the automated release-integrity search in addition to reviewing all new
configuration defaults and examples:

```bash
pytest -q tests/unit/test_evaluation_cli.py \
  -k published_tree_has_no_site_specific_submission_or_storage_defaults
git status --short
git diff --check
```

## 3. Host-side validation

Use both Python 3.10 (the package floor) and Python 3.12 (the GPU runtime) with
the dependency constraints declared by the package:

```bash
python3.12 -m venv /tmp/ncp-olmo-eval-release-venv
source /tmp/ncp-olmo-eval-release-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,scoring]'

ruff check src tests scripts
python -m compileall -q src tests
pytest -q
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m py_compile scripts/*.py
git diff --check
python -m build
python -m twine check dist/*
```

## 4. Distribution inspection

Install the built wheel into another clean environment so source-path mistakes
cannot be hidden by the checkout:

```bash
python3.12 -m venv /tmp/ncp-olmo-eval-release-installed
/tmp/ncp-olmo-eval-release-installed/bin/python -m pip install --no-deps dist/*.whl
NCP_OLMO_SOURCE_REVISION="$(git rev-parse HEAD)" \
  /tmp/ncp-olmo-eval-release-installed/bin/python \
  tests/installed_package_summary_smoke.py
```

Inspect the wheel and source distribution before publication:

```bash
python -m zipfile -l dist/*.whl
tar -tzf dist/*.tar.gz
```

Confirm that the wheel contains every runtime module and the typed marker, and
that the source distribution contains the documentation, Dockerfiles,
examples, tests, and notices. Neither artifact may contain weights, datasets,
caches, results, virtual environments, or credentials.

## 5. OCI and scorer validation

- Build `docker/runtime.Dockerfile` from an immutable CUDA/vLLM base digest and
  record the full source revision in `SOURCE_REVISION`.
- Require the Math runtime smoke during the runtime build.
- Build the published Core88 sandbox Dockerfiles from pinned upstream artifacts
  and immutable bases; record the final image digests.
- Run the BigCodeBench smoke in the final published sandbox image, not a
  synthetic approximation.
- Verify the selected executor actually enters each image declared by a task
  spec. An image hint without container isolation is not a valid code score.

The checked-in GitHub Actions workflow performs CPU-buildable slices of these
checks. The image publication workflow builds the published recipes themselves
from public pinned bases. A release owner must still verify CUDA/driver
compatibility.

## 6. Model and benchmark smoke

For each newly supported model family or export format:

1. register an immutable local checkpoint;
2. load it with the pinned vLLM/Transformers runtime;
3. produce a finite, non-empty greedy continuation;
4. verify that the source checkpoint is unchanged;
5. remove only the temporary registration/output root.

Treat this as load-route validation, not numerical parity or benchmark
accuracy. Before claiming benchmark compatibility, also complete the relevant
inference, scoring, artifact-validation, and finalization lifecycle.

For NCP-DFlash speculative decoding, use the exact vLLM version declared by
the package and run both a target-only baseline and a speculative job against
the same prompts. The release smoke must additionally confirm:

- at least 8 prompts and 1,024 forced output tokens are compared;
- the target and draft checkpoint identities are recorded in the artifacts;
- an exact-labelled run either has full generated-token parity or persists a
  failing artifact that registration rejects; do not publish an exact-parity
  claim unless the former is true for the tested target/draft pair;
- the result contains finite acceptance, proposal, verification, timing, and
  throughput telemetry;
- when the scheduler queue exceeds the active batch, at least one complete
  queue is submitted and the result proves that dynamic batch sizes exercised
  the sealed batch-to-width policy;
- steady-state throughput excludes model loading and warmup, reports the draft
  cold-start increment separately, and does not turn a pair-specific synthetic
  speedup into a universal serving claim;
- `segmented_kv_approx` executes end to end, remains opt-in, cannot satisfy an
  exact registration gate, and is followed by matched downstream score A/B
  before benchmark results are compared; and
- `ncp-olmo-eval-dflash-benchmark --help` works from the installed wheel.

Do not publish private checkpoint paths or scheduler configuration as release
defaults. Site-specific mounts belong in an external compatibility layer.

## 7. One-time publication setup

- Create a PyPI pending Trusted Publisher for owner `LuckySJTU`, repository
  `ncp_olmo_eval`, workflow `release.yml`, and environment `pypi`. Protect that
  GitHub environment with required reviewers if desired.
- Verify that the package name `ncp-olmo-eval` is available or owned by the
  project maintainers.
- After the first image workflow push, change all five GHCR packages to Public.
  A public source repository does not make package visibility public
  automatically.
- Keep PyPI API tokens out of repository secrets. The release workflow uses
  short-lived OIDC credentials through Trusted Publishing.

## 8. Publication gate

Only after all applicable checks pass:

- record the approved source commit and source-tree SHA-256;
- update the version and changelog if the approved release requires it;
- build artifacts again from that exact clean revision;
- create an immutable `v<version>` tag and publish the GitHub release;
- let `release.yml` build and publish the wheel/sdist to PyPI through Trusted
  Publishing, and let `publish-images.yml` publish the five OCI images;
- verify the published wheel/image digest and the installed CLI;
- verify `public-images.env` contains five `tag@sha256:...` references and is
  attached to the GitHub release;
- preserve the release test evidence without publishing private paths or data.
