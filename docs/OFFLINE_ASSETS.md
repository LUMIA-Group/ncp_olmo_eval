# Offline Hugging Face and data assets

Formal evaluation must not depend on a mutable Hub cache or worker network.
`ncp-olmo-eval-assets` downloads pinned snapshots into named directories and
writes a per-file size/SHA-256 lock. `verify` detects missing or changed files.
Asset roots in the lock are relative to the lock itself, so the sealed bundle
can be moved or mounted at a different absolute path on another cluster.

The input manifest requires an immutable revision for every model or dataset.
Use a full Hub commit SHA, not `main`. The checked-in
[`configs/assets.example.json`](../configs/assets.example.json) is immediately
usable and pins the public AutoAIS, RULER archive, and HELMET classic-data
snapshots used by this release. Credentials may be present in the host process
during `prepare`; they are never written to the lock or task specs.

HELMET also requires the official Llama 2 tokenizer. That repository is gated,
so the project cannot lawfully mirror it as an unrestricted public asset. After
accepting Meta's terms, prepare the separately pinned
[`configs/assets.gated.example.json`](../configs/assets.gated.example.json)
with an authenticated Hugging Face session. This is an explicit
external-license boundary, not a missing public mirror.

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

Pinned source checkouts are obtained directly from their public repositories:

```bash
git clone https://github.com/allenai/OLMo-Eval.git sources/OLMo-Eval
git -C sources/OLMo-Eval checkout --detach f8816eea36563f27b4a9dd2533d68d34f3c67d3f

git clone https://github.com/allenai/olmes.git sources/olmes
git -C sources/olmes checkout --detach 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e

git clone https://github.com/princeton-nlp/HELMET.git sources/HELMET
git -C sources/HELMET checkout --detach af609c4d51b97fc35012099380aa889da961c42d
```

The evaluator validates the relevant files and commits before a formal run.
Core88 task exports and HELMET's auxiliary Hub datasets remain separate data
preparation inputs; do not bake them into a public image unless their
individual licenses permit redistribution.
