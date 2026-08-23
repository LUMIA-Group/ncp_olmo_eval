# Changelog

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
- Added vLLM support for stock OLMo and the experimental NCP OLMo plugin.
- Added Core88 plus standard GSM8K, RULER 4K-64K, and HELMET 8K-64K.
- Added tokenizer-bound sealed long-context preparation, fail-closed resume and
  finalization contracts and explicit seed 42.
- Removed private paths, registry names, mounts, accounts, proxies, and model
  defaults from the published code.
