# ncp_olmo_eval

`ncp_olmo_eval` is a reproducible, scheduler-neutral evaluation toolkit for
stock OLMo and NCP-ArchPreview checkpoints on vLLM. One fail-closed CLI covers
immutable model registration, inference planning, sandboxed scoring, artifact
validation, and final result materialization for GSM8K, SciQ, Core88, RULER,
and HELMET. Public pinned assets, OCI build recipes, and scheduler adapters let
the same protocol run on a workstation, Slurm, Kubernetes, or another cluster.

The distribution name `ncp-olmo-eval`, Python package `ncp_olmo_eval`, CLI
commands, and `NCP_OLMO_*` environment variables are retained as stable
compatibility identifiers. User-facing model-family terminology is
**NCP-ArchPreview**.

The package does **not** submit to a cluster API. It emits ordinary JSON task
specifications that can run in an existing allocation or through a thin Slurm,
Kubernetes, or site-specific adapter. The published tree contains no private
mount, registry, proxy, account, credential, checkpoint, or scheduler default.

> **Status:** alpha. The public runtime supports vLLM only. Benchmark protocols
> and artifact checks are fail-closed; changing a seed, prompt, prepared-data
> identity, model identity, or source revision requires a new evaluation.

## What is supported

| Benchmark | Frozen inference contract | Inference plan | Final result |
|---|---|---:|---|
| GSM8K | fixed OLMo 8-shot prompt, seed 42, batch 8, greedy one sample | 1 x 8 GPU | aligned pass@1 |
| SciQ | official zero-shot four-choice likelihood, seed 42, batch 8 | 1 x 8 GPU | official raw accuracy |
| Core88 | seed 42, batch 8, official per-task sample counts, context 8,192 | 4 x 8 GPU + fresh GSM8K | 88-column and fixed 30-column CSVs |
| RULER | fixed OLMES data, 4K/8K/16K/32K/64K, seed 42, batch 4 | 1 x 8 GPU | task-by-length scores |
| HELMET | pinned official profile, 8K/16K/32K/64K, seed 42, batch 4 | 1 x 8 GPU | family-by-length scores |

NCP DFlash speculative decoding is available as a separately registered,
experimental NCP-ArchPreview path for GSM8K and Core88. Exact-labelled modes fail
closed unless token parity is proven; the currently evidenced tuned path is
explicitly approximate and requires matched downstream quality A/B. It never
changes the ordinary target-only protocol. See
[SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md).

Core88 scoring is a separate CPU/sandbox phase: one prediction-score snapshot,
eight Python shards, eight BigCodeBench shards, eight DS-1000 shards, and 32
MultiPL-E shards. Generated programs must execute in the matching sealed
sandbox image, not in the inference process.

SciQ is a standalone benchmark. It does not alter the Core88 88-task contract
or silently fill an optional SciQ column in the Core88 30-column table.

See [the exact protocol pins](docs/PROTOCOLS.md) before comparing results.

## Model compatibility

| Model family | Runtime path | Required checkpoint form |
|---|---|---|
| Stock OLMo | vLLM built-in implementation | local Hugging Face-compatible directory |
| NCP-ArchPreview | installed `vllm.general_plugins` entry point | pure-HF config/tokenizer plus complete safetensors or bin shards |
| NCP DFlash draft | packaged vLLM 0.13 proposer adapter | remote-code draft config plus one `model.safetensors` |

Registration checks, without modifying the checkpoint:

- `config.json` exists and is readable;
- either a complete indexed shard set or one unsharded weight file exists;
- at least one supported tokenizer file exists;
- indexed weight files exist and their sizes are recorded;
- later workflow steps see the same registered file contract.

For NCP-ArchPreview, install the package rather than only adding its source directory
to `PYTHONPATH`; vLLM discovers the model through the installed plugin entry
point. NCP-ArchPreview remains behind an explicit experimental opt-in gate.

## Public NCP-ArchPreview checkpoints

The following public Hugging Face checkpoints are the reference model set for
this workflow:

- [NCP_ArchPreview_dolma3_8.9B_Stage1](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage1)
- [NCP_ArchPreview_dolma3_8.9B_Stage2_v1](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1)
- [NCP_ArchPreview_dolma3_8.9B_Stage2_v2](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v2)
- [NCP_ArchPreview_dolma3_8.9B_Stage2_v3](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v3)

The paired speculative-decoding draft is
[NCP_ArchPreview_dolma3_8.9B_Stage2_DFlash2_NCPFlash](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_DFlash2_NCPFlash).

The current pinned runtime is Python 3.12, vLLM 0.13.0,
Transformers 4.57.6, and `huggingface-hub` 0.36.2. Pre-release load smoke has
also covered 17 local NCP-ArchPreview HF exports (14 Stage1 and three Stage2): every
checkpoint loaded and produced a non-empty greedy continuation with the pinned
runtime. That evidence is a load/route smoke, not a benchmark score or native
backend parity claim.

## Install

After the alpha is published to PyPI:

```bash
python -m pip install 'ncp-olmo-eval[vllm,helmet,scoring]==0.1.0a15'
```

Until then, or when validating a source revision, install from a clean checkout:

For the complete GPU runtime:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vllm,helmet,scoring]'
```

For CPU-only development and contract tests:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,scoring]'
```

Use `.[gsm8k,dev]` for CPU-only GSM8K preparation. The `vllm` extra already
contains the same pinned `lm-evaluation-harness` commit. A different harness
revision is rejected before the formal GSM8K/Core88 task data is read.

The `scoring` extra pins the exact SymPy/ANTLR runtime used by the Minerva,
MATH, and MATH-500 scorer. The production runtime image installs that slice and
runs its CPU smoke during image construction. See [IMAGES.md](docs/IMAGES.md)
for reproducible OCI and Core88 sandbox builds.

## Prepare a portable environment

1. Copy [`configs/runtime.env.example`](configs/runtime.env.example) to a
   private runtime file and replace shared paths for your site.
2. Download `public-images.env` from the matching GitHub release and source it
   to select the published immutable OCI digests. Public bases and release tags
   are listed in [`configs/public-image-bases.json`](configs/public-image-bases.json).
3. Use the already pinned public Hugging Face assets in
   [`configs/assets.example.json`](configs/assets.example.json). The gated
   Llama 2 tokenizer is deliberately separate in
   [`configs/assets.gated.example.json`](configs/assets.gated.example.json).
4. Download assets once on a connected host, seal them, then verify the sealed
   bundle before formal work:

```bash
ncp-olmo-eval-assets prepare \
  --manifest configs/assets.json \
  --output-root /shared/ncp-olmo-eval/assets
ncp-olmo-eval-assets verify \
  --lock /shared/ncp-olmo-eval/assets/assets.lock.json
```

Inference defaults to `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`. Missing assets fail instead of downloading during a
measurement. See [OFFLINE_ASSETS.md](docs/OFFLINE_ASSETS.md).

## Quick start

The same lifecycle applies to every benchmark:

```text
register -> infer -> run inference plan -> status
         -> score -> run scoring plan   -> status
         -> final -> run final plan     -> status/results
```

### 1. Register an immutable model identity

```bash
export EVAL_ROOT=/shared/ncp-olmo-eval/results

ncp-olmo-eval --root "$EVAL_ROOT" register \
  --checkpoint /models/olmo-or-ncp-archpreview \
  --backend vllm
```

The JSON response includes a versioned `registration_name`, for example
`vllm-abc123-v1`. Reuse that name for all later commands. Registering the same
checkpoint/backend path again fails unless `--new-version` is explicit.

For speculative decoding, first produce a target/draft-bound comparison
artifact, then register both identities:

```bash
ncp-olmo-eval --root "$EVAL_ROOT" register \
  --checkpoint /models/ncp-archpreview-target \
  --backend vllm \
  --vllm-speculative-draft-model /models/ncp-dflash-draft \
  --vllm-speculative-verification /results/dflash/comparison.json \
  --vllm-speculative-allow-approximate
```

The unified entry refuses speculative inference without the comparison gate.
The flag above is required for an approximate artifact; omit it only for a
genuinely passing exact artifact. Exact-labelled modes are diagnostic in this
release because the tested stateful NCP target did not pass full parity. The
standalone A/B command and tuning controls are documented in
[SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md).
The comparison must be generated by the current release: it seals the complete
batch, continuous scheduler queue, adaptive draft-width map, and target/draft
runtime settings. Legacy a12 artifacts that only prove output correctness are
rejected rather than silently run at a different operating point.

The retained a14 matched-H200 validation, including the `1.43x`
mixed-length continuous-queue result, fixed batch/width sweep, cold-start cost,
and downstream-quality caveats, is recorded in
[VALIDATION.md](docs/VALIDATION.md). These measurements describe one tested
target/draft pair and are not a general speed or quality guarantee.

### 2. Plan and run inference

```bash
export EVALUATION=vllm-abc123-v1

# Read-only validation: no state or task files are written.
ncp-olmo-eval --root "$EVAL_ROOT" infer \
  --evaluation "$EVALUATION" --benchmark core88 --dry-run

# Materialize task.json/status.json files and a portable plan.json.
ncp-olmo-eval --root "$EVAL_ROOT" infer \
  --evaluation "$EVALUATION" --benchmark core88 --executor emit
```

The response prints `task_plan`. Archive that exact file, then execute it with
one of the adapters below. `--executor local` is available when the CLI already
runs inside an allocation with the requested resources, but multi-task plans
run sequentially; use a scheduler adapter for Core88 parallelism.

```bash
scripts/run-plan-local.sh /absolute/path/to/plan.json

NCP_OLMO_SLURM_PARTITION=gpu \
NCP_OLMO_SLURM_CONTAINER_WRAPPER=scripts/run-task-apptainer.sh \
scripts/submit-plan-slurm.sh /absolute/path/to/plan.json

scripts/render-kubernetes-jobs.py /absolute/path/to/plan.json \
  --pvc shared-storage --mount-path /shared --output-dir /tmp/jobs
```

Inspect both scheduler state and artifact completeness:

```bash
ncp-olmo-eval --root "$EVAL_ROOT" status \
  --evaluation "$EVALUATION" --benchmark core88
```

A scheduler `Succeeded` state alone is insufficient. `status` marks inference
complete only after benchmark-specific manifests, hashes, shard counts, source
identity, model immutability, and prediction coverage pass validation.

### 3. Score and finalize

After inference reports `INFERENCE_COMPLETE`:

```bash
ncp-olmo-eval --root "$EVAL_ROOT" score \
  --evaluation "$EVALUATION" --benchmark core88 --executor emit
# Run the returned scoring task_plan, then refresh status.

ncp-olmo-eval --root "$EVAL_ROOT" final \
  --evaluation "$EVALUATION" --benchmark core88 --executor emit
# Run the returned final task_plan, then refresh status again.
```

Scoring always creates a fresh attempt. Core88 finalization fails closed when
the evaluator revision/tree digest, workflow identity, schema, coverage,
hashes, or aggregates disagree. Eligible cached scorer evidence may be reused;
otherwise the finalizer reopens raw predictions and results for a complete
rescore. A clean installed wheel or OCI image works without `.git` in the
worker because the release records a path-independent evaluator tree digest.

## Long-context preparation

RULER and HELMET consume sealed tokenizer/profile-specific inputs. Prepare each
dataset with the tokenizer of the model being evaluated:

```bash
ncp-olmo-eval-prepare ruler --help
ncp-olmo-eval-prepare helmet --help

ncp-olmo-eval --root "$EVAL_ROOT" infer \
  --evaluation "$EVALUATION" --benchmark ruler \
  --data-root /shared/prepared/ruler/tokenizer-hash --executor emit
```

Prepared data is content-addressed. A tokenizer or profile mismatch is rejected
rather than silently reusing an incompatible cache.

## Portability and trust boundaries

- Task specs contain an argument vector, non-secret environment, resource
  request, paths, and an optional immutable image digest—never interpolated
  shell or scheduler configuration.
- All paths in one plan must resolve identically in every worker/container.
- Site adapters provide accounts, queues, PVCs, registry credentials, proxies,
  secrets, and driver/runtime compatibility.
- Core88 program execution requires the matching no-network sandbox image.
- HELMET citation/judge steps require sealed local assets and, when applicable,
  credentials injected only at execution time. Secrets are rejected in task
  JSON.

See [PORTABILITY.md](docs/PORTABILITY.md) for status semantics and executor
requirements.

## Release preflight

Run the checked-in [release checklist](docs/RELEASE_CHECKLIST.md) from a clean
checkout. At minimum:

```bash
ruff check src tests scripts
python -m compileall -q src tests
pytest -q
find scripts -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m py_compile scripts/*.py
python -m build
git diff --check
```

The CI workflow additionally installs the built wheel at a different path,
validates source-identity-aware Core88 finalization, builds the published
runtime scorer slice, and executes the formal BigCodeBench sandbox smoke.
Release workflows publish the wheel through PyPI Trusted Publishing and build
the five public OCI images from pinned public inputs.

## Alpha limitations

- Only vLLM is exposed by the public unified workflow.
- NCP DFlash is pinned to vLLM 0.13.0, is opt-in, and is formally routed only
  to GSM8K/Core88. Approximate output requires a separate registration and
  matched downstream score A/B. The tested stateful target does not establish
  exact speculative parity, so exact-labelled modes remain diagnostic gates.
- CUDA driver compatibility, image publication, and scheduler integration are
  site responsibilities.
- Core88 sandbox images depend on pinned upstream runtime artifacts that may
  have separate redistribution terms.
- Strict numerical parity with native Megatron is not claimed. Optional parity
  tools may consume externally produced reference artifacts, but the private
  training stack is not bundled.
- The compatibility smoke above does not replace task-level accuracy and
  long-context validation for a newly exported checkpoint.

## Documentation

- [Evaluation protocols](docs/PROTOCOLS.md)
- [Portable execution contract](docs/PORTABILITY.md)
- [Offline assets](docs/OFFLINE_ASSETS.md)
- [OCI and sandbox images](docs/IMAGES.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

Original repository code is licensed under
[Apache License 2.0](LICENSE). Third-party code, benchmark data, model weights,
and container bases retain their own terms; in particular, the MultiPL-E
runtime has an additional machine-learning-training restriction. See
[`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The repository license does not license NCP-ArchPreview weights; each Hugging
Face model card must declare its independently reviewed weight license.
