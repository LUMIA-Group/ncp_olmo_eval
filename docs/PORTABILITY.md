# Portable execution contract

The evaluator separates protocol planning from resource allocation. `infer`,
`score`, and `final` write a plan containing absolute paths to task specs. A
task spec contains only:

- an argument vector (`argv`), never an interpolated shell command;
- non-secret environment variables;
- requested GPUs, CPUs, memory, and nodes;
- working/output/log/status paths;
- an optional immutable OCI image hint.

No cluster SDK or command is imported by the Python package.

## State model

`status.json` moves through `Planned -> Running -> Succeeded|Failed`. The task
runner writes every transition atomically. Cluster adapters must invoke:

```bash
python -m ncp_olmo_eval.task_runner /absolute/path/to/task.json
```

They must not synthesize `Succeeded`; artifact validation is a separate gate.
`status` reports both the task state and benchmark-specific completeness.

## Filesystem contract

All task-spec paths must resolve identically inside every worker/container.
Use a shared filesystem, or stage a complete immutable input bundle and map it
at the same absolute path. Results, status files, logs, prepared data, model
weights, and offline caches must survive worker termination.

## Executor requirements

- Local: caller already owns the requested devices. Plans run sequentially.
  A declared image must match `NCP_OLMO_ACTIVE_IMAGE` or be executed through
  `NCP_OLMO_CONTAINER_WRAPPER`.
- Slurm: adapt partition/account/QoS outside the task spec. Request the exact
  resources and set `NCP_OLMO_SLURM_CONTAINER_WRAPPER` when tasks declare
  images. The provided Apptainer wrapper resolves exact OCI references through
  a user-maintained immutable ref-to-SIF map.
- Kubernetes: mount a PVC at the path used in specs, use NVIDIA device resource
  limits, and apply your own security/network policy.
- Other systems: read the stable JSON schema and run the same task runner.

Secrets, judge API keys, registry credentials, proxies, and scheduler options
must be injected by the executor. They are intentionally absent from plans.

## Formal-run checklist

1. Pin this repository revision and evaluator source-tree digest, and pin every
   OCI image by digest. Installed wheels do not need a `.git` directory.
2. Verify checkpoint registration and `assets.lock.json`.
3. Keep all Hugging Face/Transformers dataset downloads offline.
4. Materialize a plan, archive it, then submit every listed task exactly once.
5. Refresh status and require both `Succeeded` and complete artifacts.
6. Score into a fresh attempt; never edit prediction files.
7. Archive final CSV/JSON plus task specs, locks, and source/image revisions.
