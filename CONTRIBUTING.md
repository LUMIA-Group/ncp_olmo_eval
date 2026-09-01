# Contributing

Please keep benchmark protocol changes explicit and reviewable. A protocol
change must update tests, result metadata/schema where necessary,
`docs/PROTOCOLS.md`, and the changelog. Do not mix predictions from different
protocol revisions under resume.

Before submitting a change, run:

```bash
python -m compileall -q src tests
python -m pytest
ruff check src tests scripts
git diff --check
for script in scripts/*.sh; do bash -n "$script"; done
python -m py_compile scripts/*.py
```

Release candidates must also follow
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md), including clean-wheel,
distribution-content, public-tree, and formal sandbox-image validation. Version
changes, tags, and publication are explicit release actions and should not be
mixed into routine cleanup.

Never commit model weights, benchmark data, credentials, private registry
names, internal mounts, or judge API keys.
