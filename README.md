# ncp_olmo_eval

`ncp_olmo_eval` is a scheduler-neutral, vLLM-only evaluation workflow for
stock OLMo and NCP OLMo checkpoints. One CLI handles immutable model
registration, inference plans, scoring plans, status validation, and final
results for GSM8K, Core88, RULER, and HELMET.

The repository contains no cluster endpoint, private mount, registry, proxy,
account, credential, or checkpoint default. It emits ordinary JSON task specs;
execute them in an existing allocation or translate them with a thin site
adapter.

## Supported protocols

| Benchmark | Fixed contract | Portable task plan | Result |
|---|---|---:|---|
| GSM8K | standard fixed 8-shot prompt, seed 42, batch 8, one sample | 1 x 8 GPU | aligned score |
| Core88 | seed 42, batch 8, official per-task samples, context 8192 | 4 x 8 GPU plus GSM8K | 88/30-column CSV |
| RULER | fixed OLMES data, 4K/8K/16K/32K/64K, seed 42, batch 4 | 1 x 8 GPU | task/length scores |
| HELMET | pinned official profile, 8K/16K/32K/64K, seed 42, batch 4 | 1 x 8 GPU | family/length scores |

Stock OLMo uses vLLM's built-in model implementation. NCP OLMo is registered
through the installed `vllm.general_plugins` entry point and remains behind an
explicit experimental opt-in gate.

## Install

Python 3.12 and a CUDA-capable runtime are expected.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[vllm,helmet,dev]'
```

For protocol/unit tests without CUDA, `pip install -e '.[dev]'` is sufficient.
The NCP vLLM plugin must be installed, not only placed on `PYTHONPATH`.

## Reproducible assets and environment

1. Copy [`configs/runtime.env.example`](configs/runtime.env.example) and replace
   every placeholder with a shared path or immutable OCI digest.
2. Pin external Hugging Face assets in a copy of
   [`configs/assets.example.json`](configs/assets.example.json).
3. Download once on a connected host and verify before every formal run:

```bash
ncp-olmo-eval-assets prepare \
  --manifest configs/assets.json \
  --output-root /shared/ncp-olmo-eval/assets
ncp-olmo-eval-assets verify \
  --lock /shared/ncp-olmo-eval/assets/assets.lock.json
```

Inference defaults to `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`; missing inputs fail instead of downloading during a
measurement. See [offline assets](docs/OFFLINE_ASSETS.md) and
[images](docs/IMAGES.md).

## Unified entry and task plans

```bash
# Register one immutable checkpoint/backend identity.
ncp-olmo-eval --root /shared/evaluations register \
  --checkpoint /models/olmo-or-ncp-olmo \
  --backend vllm

# Validate only; write nothing.
ncp-olmo-eval --root /shared/evaluations infer \
  --evaluation vllm-abc123-v1 --benchmark core88 --dry-run

# Materialize task.json/status.json and one plan.json. No scheduler is called.
ncp-olmo-eval --root /shared/evaluations infer \
  --evaluation vllm-abc123-v1 --benchmark core88 --executor emit

# Inspect task and artifact state.
ncp-olmo-eval --root /shared/evaluations status \
  --evaluation vllm-abc123-v1 --benchmark core88
```

The command returns the exact `task_plan`. Run that plan in one of three ways:

```bash
# Current machine/allocation; tasks run sequentially.
# If a task declares an image, set NCP_OLMO_ACTIVE_IMAGE when already inside
# it, or set NCP_OLMO_CONTAINER_WRAPPER to scripts/run-task-apptainer.sh.
scripts/run-plan-local.sh /shared/evaluations/.../tasks/plan.json

# Slurm example; site policy stays in environment variables.
NCP_OLMO_SLURM_PARTITION=gpu \
NCP_OLMO_SLURM_CONTAINER_WRAPPER=scripts/run-task-apptainer.sh \
scripts/submit-plan-slurm.sh /path/to/plan.json

# Kubernetes Job JSON; apply after adding any required security policy.
scripts/render-kubernetes-jobs.py /path/to/plan.json \
  --pvc shared-storage --mount-path /shared --output-dir /tmp/jobs
```

`--executor local` is convenient when the CLI itself already runs inside an
8-GPU allocation. For Core88 it executes the four 8-GPU machine tasks and the
GSM8K companion sequentially; use an external executor for parallelism.

After inference succeeds, scoring and finalization use the same mechanism:

```bash
ncp-olmo-eval --root /shared/evaluations score \
  --evaluation vllm-abc123-v1 --benchmark core88 --executor emit
ncp-olmo-eval --root /shared/evaluations final \
  --evaluation vllm-abc123-v1 --benchmark core88 --executor emit
```

Scoring always creates a fresh attempt. Core88 finalization fails closed when
the evaluator source revision/tree digest, schema, coverage, hashes, or
aggregates do not match; if cached evidence is ineligible, it reopens raw
predictions/results and performs a complete rescore. This works from either a
clean Git checkout or an installed wheel/OCI image without requiring `.git` in
the worker.

## Prepared long-context data

RULER and HELMET require sealed tokenizer/profile-specific data:

```bash
ncp-olmo-eval-prepare ruler --help
ncp-olmo-eval-prepare helmet --help
ncp-olmo-eval --root /shared/evaluations infer \
  --evaluation vllm-abc123-v1 --benchmark ruler \
  --data-root /shared/prepared/ruler/tokenizer-hash --executor emit
```

Protocol pins are documented in [PROTOCOLS.md](docs/PROTOCOLS.md). Portable
deployment and task status semantics are in [PORTABILITY.md](docs/PORTABILITY.md).

## Alpha limitations

- Only vLLM is public in this repository.
- The runtime image recipe is reproducible, but CUDA driver compatibility and
  registry publication remain site responsibilities.
- Core88 execution scorers require separately built sandbox images and pinned
  upstream runtime artifacts. Images are hints in task specs; an executor must
  actually run each task in the matching image.
- HELMET citation and judge components require sealed local assets and, where
  applicable, credentials injected at execution time. Secrets are rejected in
  serialized task environments.
- Strict numerical parity with a native Megatron implementation is not claimed.
  The optional parity helper can consume native reference artifacts, but the
  private training stack needed to generate them is intentionally not bundled.

## License

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
