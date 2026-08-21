# Changelog

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
