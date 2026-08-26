# Evaluation protocols

This document describes the benchmark protocol introduced by `v0.1.0a1` and
retained by the scheduler-neutral `v0.1.0a11` work. Result metadata is
authoritative when it is more specific than this overview.

## Common contract

- Backend: vLLM only.
- Global/few-shot/sampling seed: 42.
- Model registration is versioned and immutable.
- Resume is enabled, but only within the same recorded protocol.
- Inference and scoring artifacts retain source/model fingerprints.

## GSM8K

The workflow consumes a user-supplied standard input JSON whose SHA-256 must be
`295395763cbe551cbe41481c1a9ad16b491768c8d50911bd6ab1cd4f69e2b265`.
It selects the fixed OLMo paper 8-shot task and validates the rendered prompt,
the 1,319-example test set, generation arguments, stop strings, and one-sample
coverage. Inference uses eight GPUs, batch size 8, and seed 42.

## Core88

Core88 uses a cost-aware four-machine dispatch plan, one eight-GPU vLLM job per
machine, plus a fresh standalone GSM8K job. vLLM max model length is 8,192.
Batch size is 8. `generation_samples_cap=0` means that each task retains its
protocol-defined sample count; it does not mean zero samples.

Scoring separates answer matching from sandboxed execution. MultiPL-E uses 32
shards; Python, BigCodeBench, and DS-1000 use eight shards each. Finalization
emits the full 88-column CSV and the fixed 30-column summary.

## SciQ

- Source profile/task: `all_supported_local`, task order 348,
  `olmo_eval_sciq`.
- Source file: `all_supported_local/348_olmo_eval_sciq.jsonl.gz`, SHA-256
  `b0bf31832d352e29b846f0e05893b1ffcef9d7ba2d4c350fdb36fad1f9fa0db3`.
- Test set: exactly 1,000 unique examples; zero-shot official
  `Question:/Answer:` prompt and four continuation candidates.
- Inference: `loglikelihood`, batch size 8, one eight-GPU vLLM job, seed 42,
  no generation sampling.
- Metric: official raw `acc`. The ordered upstream metric contract is retained
  as evidence, but `acc_norm` is never substituted as the primary score.
- Scoring: a CPU task reopens sealed gold and prediction artifacts, checks full
  coverage and hashes, and emits `score.json` plus `sciq-score.csv`.

SciQ remains a standalone benchmark. It does not change Core88's 88 tasks and
does not silently populate the optional SciQ column in a Core88 30-column table.

## RULER

- Upstream RULER revision: `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`.
- Data source: fixed AllenAI/OLMES `ruler_data` archive, verified by size/hash.
- Lengths: 4,096; 8,192; 16,384; 32,768; 65,536.
- Batch size: 4; one sample per example; seed 42; EOS stopping enabled.
- Prompt, QA contract, per-task generation budgets, normalization, and scoring
  follow the pinned OLMES implementation captured in the manifest.

RULER inputs are sealed and tokenizer-specific. Reusing data prepared for a
different tokenizer is rejected.

## HELMET

- Official revision: `af609c4d51b97fc35012099380aa889da961c42d`.
- Lengths: 8,192; 16,384; 32,768; 65,536.
- Batch size: 4; one sample per example; seed 42.
- The `8k-64k` profile validates task families instead of hard-coding the
  full/128K profile names.
- ALCE ASQA/QAMPARI citation tasks are recognized by task family.

HELMET preparation is content-addressed and can reuse an exact sealed dataset.
Citation NLI, NLTK data, and judge outputs are external caches and are recorded
in scoring evidence rather than downloaded inside inference jobs.
