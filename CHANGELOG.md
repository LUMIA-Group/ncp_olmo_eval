# Changelog

## Unreleased

## 0.1.0a15

- Adopt **NCP-ArchPreview** as the public model-family name throughout the
  documentation and package metadata while retaining existing distribution,
  import, CLI, environment-variable, schema, and model-config identifiers for
  backward compatibility.
- Document the public Stage1, Stage2 v1/v2/v3, and Stage2 DFlash2 NCPFlash
  checkpoints published by the ArchSpace Collection on Hugging Face.
- Re-license original repository code under Apache-2.0 and separate upstream
  attribution and asset terms into `NOTICE` and `THIRD_PARTY_NOTICES.md`.
- Pin the public upstream OCI bases used by the validated Core88 sandboxes and
  add release automation that publishes runtime and scorer images to GHCR with
  an immutable digest manifest.
- Add a PyPI Trusted Publishing workflow that builds once, validates the wheel
  and source distribution, and publishes through GitHub OIDC without a stored
  API token.
- Replace asset and image placeholders with pinned public references or an
  explicit gated-asset boundary, and document the one-time publisher setup.

## 0.1.0a14

- Publish the a14 validation record for the sealed NCP DFlash
  continuous-batching operating point introduced in `0.1.0a13`, without
  changing the inference or scoring protocols.
- Record both the fixed active-batch/draft-width sweep and a mixed-length
  32-request continuous queue. The tested pair reached `1.43x` steady-state
  continuous throughput and preserved the adaptive `1:8,2:8,4:4,8:2` policy.
- Document cold-start cost, pair-specific acceptance, and matched downstream
  quality A/B results so the synthetic speed result cannot be presented as a
  universal or target-exact guarantee.
- Add release-metadata consistency coverage and package the sanitized
  validation evidence in the source distribution.

## 0.1.0a13

- Seal the complete target-plus-DFlash operating point into every comparison
  artifact and registration, including active batch size, scheduler queue,
  draft width policy, model-runner selection, cache/mixer settings, and target
  runtime controls. Legacy speculative comparison artifacts must be regenerated.
- Preserve the verified batch-to-draft-width policy when emitting Core88 and
  GSM8K tasks instead of replacing it with a hard-coded width of 16 or planner
  environment variables.
- Require continuous-batching A/B runs to submit at least one full scheduler
  queue, and bind the same batch, queue, and runtime contract into formal
  inference manifests and GSM8K aggregates.

## 0.1.0a12

- Add the correctness-gated NCP DFlash proposer and vLLM 0.13 model-runner
  adapter without patching installed vLLM files.
- Add target/draft A/B benchmarking with exact and explicitly approximate
  verification modes, immutable artifact binding, speed gates, GPU-memory
  evidence, and proposal/rejection telemetry.
- Extend model registration with a sealed draft identity and comparison
  artifact. Formal speculative inference fails closed until the artifact
  covers at least eight prompts and 1,024 forced tokens at seed 42.
- Route verified speculative registrations through GSM8K and Core88 task specs,
  propagate their proven generation batch size, and reject SciQ/RULER/HELMET
  speculative use.
- Add continuous scheduler-queue support and request-local target state
  transactions needed for batched DFlash proposal and rejection.
- Document the source integration's known exact-parity limitation: exact modes
  remain fail-closed diagnostic gates, while the evidenced tuned path is an
  explicitly approximate registration requiring matched downstream quality
  A/B.

## 0.1.0a11

- Add standalone SciQ inference, scoring, status validation, and final result
  materialization to the scheduler-neutral unified entry.
- Seal SciQ to the OLMo `all_supported_local` task 348 export: 1,000 test
  examples, four continuation likelihoods, zero-shot `Question:/Answer:`
  prompts, seed 42, one eight-GPU vLLM job, and official raw `acc`.
- Normalize OLMo's ordered metric definitions without losing their source
  evidence, selecting raw `acc` ahead of `acc_norm` for SciQ.
- Add subset-aware Core-native planning and aggregation while retaining strict
  full-profile order checks, exact source SHA-256 validation, complete unique
  prediction coverage, and CPU-only rescoring.

## 0.1.0a10

- Separate the immutable evaluator identity recorded in a Core88 prediction-score
  snapshot from the installed release that only materializes the final tables.
- Preserve and validate the cached scorer commit/tree against the workflow even
  when the clean finalizer is installed at another path or comes from a newer
  release; record the finalizer identity independently in the output report.
- Extend the clean-wheel regression smoke to cover cached Core88 finalization
  across distinct workflow, scorer, and installed-package paths.

## 0.1.0a9

- Preserve the installed evaluator package tree SHA-256 when the fixed Core88
  30-column summary revalidates a workflow-bound GSM8K companion.
- Add a regression contract for matching evaluator packages installed at a
  different absolute path from the workflow checkout, including a clean-wheel
  installation smoke in CI.

## 0.1.0a8

- Expose the locked BigCodeBench parser wheels at
  `/opt/core88/olmo-eval-deps` through the formal sandbox image's
  `PYTHONPATH`, and verify both `tree_sitter` modules during the image build.
- Give copied parser wheels fixed container-side names, allowing formal builds
  to use either root-level files or paths nested inside the build context.
- Preserve that image-local dependency directory after task-spec environment
  merging, so an explicit OLMo-Eval source `PYTHONPATH` cannot hide the
  sandbox's parser modules.
- Replace the synthetic BigCodeBench CI layer with a build and execution of
  the published `core88_bigcodebench_sandbox.Dockerfile`, using the same
  locked-wheel extraction layout as the release image.

## 0.1.0a7

- Pin the mutually compatible and previously GPU-validated dependency pair
  `transformers==4.57.6` and `huggingface-hub==0.36.2`; the old public
  `transformers>=5` declaration conflicted with vLLM 0.13.0's `<5` bound.
- Pin the release Math scorer runtime to `sympy==1.14.0` and
  `antlr4-python3-runtime==4.11`, install it in the default runtime image, and
  execute a real Minerva/MATH equivalence smoke during the Docker build.
- Load BigCodeBench's two required helpers from SHA-verified files at the
  pinned OLMo-Eval commit instead of importing its eager task registry. This
  removes the accidental dependency on the deleted
  `huggingface_hub.utils.silent_tqdm` API without weakening source fidelity.
- Add Docker CI that builds the runtime scorer slice, runs the official Math
  runtime smoke, and runs a BigCodeBench helper preflight against the pinned
  upstream source.
- Allow site executors to replace only a Python task's interpreter through the
  explicit `NCP_OLMO_TASK_PYTHON` boundary, so heterogeneous sandbox images do
  not inherit an unrelated inference virtualenv.

## 0.1.0a6

- Gate LMDeploy argument validation on the selected backend in the Core88
  worker pool and standalone GSM8K evaluator. Public vLLM jobs no longer enter
  the intentionally unavailable LMDeploy compatibility boundary.
- Make the omitted-LMDeploy validator itself a no-op for non-LMDeploy
  backends, while continuing to fail closed when LMDeploy is explicitly
  selected.
- Add a regression test for both sides of that backend boundary.

## 0.1.0a5

- Remove the unpublished LMDeploy verification flag from portable vLLM
  Core88, RULER, and HELMET commands. The public package intentionally omits
  LMDeploy, so passing its internal-only flag made the public parsers reject
  otherwise valid jobs before model loading.
- Add release-integrity coverage that rejects future command-generation leaks
  of the removed flag; release validation also parses the generated Core88
  worker command in the pinned GPU runtime.

## 0.1.0a4

- Pin the official `lm-evaluation-harness` commit required by GSM8K/Core88.
- Include that runtime in the `vllm` and CPU-only `gsm8k` installation extras.
- Fail early on a missing or version-drifted harness and record its immutable
  identity in prepared GSM8K manifests.

## 0.1.0a3

- Restored the CUDA device-layout helper required when Core88 starts its GPU
  worker pool from an installed package.
- Made the unpublished native-Megatron reference side of the optional parity
  harness fail early with an actionable boundary instead of importing modules
  that are not part of the public distribution.
- Added release-integrity tests for internal package imports and worker device
  mapping so source extraction omissions fail in CPU-only CI.
- Kept the GSM8K, Core88, RULER, and HELMET vLLM protocol contracts unchanged.

## 0.1.0a2

- Replaced the site-specific submission layer with portable JSON task specs,
  atomic task status files, and a local task runner.
- Added example local, Slurm, and Kubernetes adapters with no site defaults.
- Added a reproducible CUDA runtime image, explicit Core88 sandbox image
  contracts, and sealed/verified Hugging Face asset preparation.
- Made formal execution offline by default and rejected serialized secrets.
- Preserved strict artifact validation and raw-rescore fallback for Core88.
- Replaced Core88's worker-side `.git` requirement with an exact evaluator
  source-tree SHA-256 while retaining Git revision evidence when available.

## 0.1.0a1

- First standalone alpha release.
- Added a unified interactive and non-interactive CLI for model registration,
  inference, scoring, status, and finalization.
- Added vLLM support for stock OLMo and the experimental NCP-ArchPreview plugin.
- Added Core88 plus standard GSM8K, RULER 4K-64K, and HELMET 8K-64K.
- Added tokenizer-bound sealed long-context preparation, fail-closed resume and
  finalization contracts and explicit seed 42.
- Removed private paths, registry names, mounts, accounts, proxies, and model
  defaults from the published code.
