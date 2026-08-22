# Third-party notices

This repository integrates protocol logic and compatibility code derived from
or validated against the following upstream projects. Their datasets, model
weights, source checkouts, and container images are not redistributed here.

- AllenAI OLMo and OLMo-Eval
- vLLM
- lm-evaluation-harness, pinned to official commit
  `95d580638385578c1c07fa554cf16ad7f5b5f460` (MIT license), supplies the
  `TaskManager` used to materialize the fixed GSM8K/Core88 prompt protocol.
- RULER and AllenAI/OLMES RULER data
- HELMET
- BigCodeBench, DS-1000, HumanEval, MBPP, and MultiPL-E

Pinned revisions and artifact hashes are recorded in source constants and
generated manifests. Users are responsible for obtaining each external asset
under its original license and terms.
