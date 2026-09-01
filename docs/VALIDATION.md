# Final alpha validation

This page records the pre-release evidence used for `0.1.0a14`. It does not
change the frozen GSM8K, SciQ, Core88, RULER, or HELMET protocols. The raw
checkpoint paths and cluster identifiers are intentionally omitted from the
public repository.

## Continuous-batching throughput

The target-only and DFlash paths were measured on matched NVIDIA H200 devices
with vLLM 0.13.0 and the `0.1.0a13` runtime code that is retained by a14.
Model loading and warmup were excluded from steady-state throughput. Generation
was greedy at seed 42 with EOS ignored so every request performed its assigned
amount of decode work.

The fixed sweep used four repeats of 256 generated tokens. The continuous run
used `max_num_seqs=8`, a queue of 32 requests, two repeats, and request output
lengths cycling through 64, 128, 256, and 512 tokens.

| Active batch | Draft width | Target tok/s | DFlash tok/s | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 60.22 | 222.09 | 3.69x |
| 2 | 8 | 108.59 | 336.74 | 3.10x |
| 4 | 4 | 207.24 | 391.86 | 1.89x |
| 8 | 2 | 385.32 | 541.34 | 1.40x |
| Continuous queue | adaptive | 286.14 | 409.16 | 1.43x |

The telemetry observed 23,422 proposed tokens and a conservative 97.25%
accepted-token lower bound. Dynamic active-batch telemetry exercised all of
the sealed width ranges, including batch sizes 1 through 8. This verifies that
the continuous queue is active; it is not an inference from fixed-batch runs.

Cold model loading took 26.05 seconds for target-only and 40.87 seconds for
target plus draft. The additional 14.82 seconds can erase the steady-state
gain in short one-shot processes. A persistent service or sufficiently long
evaluation is therefore required before interpreting the `1.43x` result as an
end-to-end gain.

## Downstream quality boundary

`segmented_kv_approx` is intentionally approximate. A fresh matched A/B on the
same target/draft pair produced:

| Benchmark | Target-only | DFlash | Delta |
|---|---:|---:|---:|
| GSM8K | 1092/1319 (82.79%) | 1080/1319 (81.88%) | -0.91 pp |
| HumanEval | 76/164 (46.34%) | 69/164 (42.07%) | -4.27 pp |

These pair-specific results are release evidence, not quality tolerances for a
different model. Every new target/draft identity or operating-point change
still requires a fresh comparison artifact, explicit approximate registration,
and matched downstream scoring. Exact-labelled modes remain fail-closed.

## Machine-readable record

The sanitized measurements are in
[`validation/dflash_a14_synthetic.json`](validation/dflash_a14_synthetic.json).
The record includes the SHA-256 of the unredacted source evidence so an
authorized operator can bind the public summary back to the retained internal
artifact without publishing private paths.
