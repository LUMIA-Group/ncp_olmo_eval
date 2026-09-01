# NCP DFlash speculative decoding

Version 0.1.0a12 adds an opt-in NCP DFlash path for NCP OLMo on the pinned
vLLM 0.13.0 runtime. It uses vLLM's upstream speculative scheduler and
rejection sampler, but replaces the n-gram proposer with the NCP DFlash draft
model. Installed vLLM files are not patched.

This path is fail-closed:

- the target must use the packaged NCP OLMo vLLM plugin;
- the draft directory must contain `config.json` and one
  `model.safetensors`;
- the draft architecture and target-layer contract are validated at
  registration;
- formal inference requires a target/draft-bound A/B artifact produced by the
  same vLLM 0.13.0 runtime;
- the artifact must cover at least eight prompts and 1,024 forced generated
  tokens at seed 42;
- exact modes must match target-only output token by token;
- the approximate mode requires an explicit opt-in and remains a separate
  evaluation identity that requires matched downstream scoring;
- speculative registrations are accepted only by GSM8K and Core88. SciQ,
  RULER, and HELMET reject them.

## Current correctness boundary

The stateful NCP OLMo target used while preparing `0.1.0a12` did **not** pass
target-only parity in either exact-labelled bring-up mode.  A release smoke on
eight prompts and 1,024 forced tokens matched 7/8 prompt streams in
`sequential_exact`; the comparison artifact was correctly persisted as
`NCP_DFLASH_VLLM_EXACT_MATCH_FAILED` and rejected by registration.  The source
integration used for this port records the same limitation, including failures
in batch-one isolation runs.

Consequently, `sequential_exact` and `intra_chunk_exact` are diagnostic gates,
not production claims in this release.  A target/draft pair may use one only
after its own artifact passes every exact-match requirement below.  The tuned
DFlash path that has downstream evidence is `segmented_kv_approx`; it must be
registered with explicit approximate opt-in and evaluated against a matched
target-only registration.  Do not report an approximate run as target-exact or
mix it into an ordinary target-only result identity.

## Verification modes

| Mode | Contract | Intended use |
|---|---|---|
| `sequential_exact` | must prove target-exact; at most one draft token | diagnostic correctness bring-up |
| `intra_chunk_exact` | must prove target-exact within one active HLM chunk | diagnostic multi-token experiment |
| `segmented_kv_approx` | output may diverge; downstream A/B required | isolated speed/quality experiment |

`chunk_parallel` and `transactional_exact` are accepted only as legacy input
aliases for `intra_chunk_exact` and `segmented_kv_approx`, respectively.
Manifests always record the canonical name.

## Produce a correctness artifact

Prepare a JSONL file with at least eight representative prompts in an
`input` field. Use fresh output directories for every run.

```bash
export TARGET=/models/ncp-olmo-target
export DRAFT=/models/ncp-dflash-draft
export PROMPTS=/data/dflash-verification-prompts.jsonl
export OUT=/shared/dflash-verification

ncp-olmo-eval-dflash-benchmark run \
  --mode target \
  --model "$TARGET" \
  --prompt-jsonl "$PROMPTS" \
  --output "$OUT/target" \
  --limit 8 \
  --batch-size 1 \
  --max-new-tokens 128 \
  --ignore-eos \
  --seed 42

ncp-olmo-eval-dflash-benchmark run \
  --mode speculative \
  --model "$TARGET" \
  --draft-model "$DRAFT" \
  --prompt-jsonl "$PROMPTS" \
  --output "$OUT/speculative" \
  --limit 8 \
  --batch-size 1 \
  --max-new-tokens 128 \
  --ignore-eos \
  --seed 42 \
  --speculative-verification-mode sequential_exact

ncp-olmo-eval-dflash-benchmark compare \
  --target-result "$OUT/target/benchmark.json" \
  --speculative-result "$OUT/speculative/benchmark.json" \
  --output "$OUT/comparison.json"
```

A passing parallel exact artifact additionally has to show
`throughput_speedup > 1`. To compare `segmented_kv_approx`, pass
`--allow-output-divergence` to `compare`; registration then also requires
`--vllm-speculative-allow-approximate`.

## Register and emit evaluation tasks

```bash
ncp-olmo-eval --root "$EVAL_ROOT" register \
  --checkpoint "$TARGET" \
  --backend vllm \
  --vllm-speculative-draft-model "$DRAFT" \
  --vllm-speculative-verification "$OUT/comparison.json" \
  --vllm-speculative-allow-approximate
```

Omit `--vllm-speculative-allow-approximate` only when the supplied artifact is
a genuinely passing exact artifact.  The flag never converts a failing exact
artifact into an accepted one.

The registration stores immutable target, draft, and comparison identities.
If any referenced artifact changes, later commands stop and require a new
registration version. The verified active batch size is propagated to the
GSM8K/Core88 generation tasks instead of silently using the ordinary batch-8
contract.

## Experimental performance controls

The proposer reads the following optional environment variables. Task specs
copy explicitly configured values so external schedulers can reproduce them:

- `CONCEPTLM_DFLASH_ATTENTION_BACKEND`
- `CONCEPTLM_DFLASH_CONTEXT_KV_CACHE`
- `CONCEPTLM_DFLASH_SPARSE_CONTEXT_PROJECTION`
- `CONCEPTLM_DFLASH_MIN_ELIGIBLE_BATCH`
- `CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_ROW`
- `CONCEPTLM_DFLASH_MIN_PROPOSAL_TOKENS_PER_BATCH`
- `CONCEPTLM_DFLASH_RUNTIME_BLOCK_SIZE`
- `CONCEPTLM_DFLASH_ACTIVE_BATCH_WIDTHS`
- `CONCEPTLM_DFLASH_DYNAMIC_RUNTIME_BLOCK_SIZE`
- `CONCEPTLM_DFLASH_RUNTIME_LAYER_COUNT`
- `CONCEPTLM_DFLASH_RUNTIME_LOCAL_MIXER`
- `CONCEPTLM_DFLASH_MIXER_COMPILE_MODE`
- `CONCEPTLM_DFLASH_TELEMETRY_FLUSH_INTERVAL`

These knobs are experimental and are not promoted to a default quality preset.
Each changed configuration needs a fresh matched target/spec comparison and a
new evaluation registration. Proposal, rejection, rollback, and acceptance
lower-bound telemetry is written under each inference shard.
