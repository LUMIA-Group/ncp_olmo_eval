# Offline Hugging Face and data assets

Formal evaluation must not depend on a mutable Hub cache or worker network.
`ncp-olmo-eval-assets` downloads pinned snapshots into named directories and
writes a per-file size/SHA-256 lock. `verify` detects missing or changed files.
Asset roots in the lock are relative to the lock itself, so the sealed bundle
can be moved or mounted at a different absolute path on another cluster.

The input manifest requires an immutable revision for every model or dataset.
Use a full Hub commit SHA, not `main`. Credentials may be present in the host
process during `prepare`; they are never written to the lock or task specs.

Recommended layout:

```text
/shared/ncp-olmo-eval/
  assets/assets.lock.json
  cache/huggingface/{hub,datasets}/
  cache/nltk/
  data/{core88,gsm8k,prepared}/
  sources/{HELMET,OLMo-Eval}/
  results/
```

Model checkpoints can be local directories registered directly and need not
be copied into the Hub cache. Their config, weight shard list/sizes, tokenizer
files, and stable path identity are recorded by model registration.

The optional `docker/offline-assets.Dockerfile` demonstrates baking a sealed
asset bundle into a derived runtime image. Do this only when licenses permit;
for large or restricted assets prefer a read-only shared volume. Never publish
model weights merely because the evaluator image is public.
