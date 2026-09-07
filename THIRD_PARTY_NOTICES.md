# Third-party notices

The original code in this repository is licensed under Apache-2.0. That license
does not replace the terms of upstream code, datasets, model weights, or
container images. The evaluator pins the following projects for protocol or
runtime compatibility:

| Upstream | Use in this project | Upstream license |
|---|---|---|
| [AllenAI OLMo](https://github.com/allenai/OLMo) and [OLMo-Eval](https://github.com/allenai/OLMo-Eval) | model/evaluation compatibility and pinned scorer behavior | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | inference runtime and plugin API | Apache-2.0 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at `95d580638385578c1c07fa554cf16ad7f5b5f460` | GSM8K/Core88 task materialization | MIT |
| [AllenAI OLMES](https://github.com/allenai/olmes) at `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e` and [NVIDIA RULER](https://github.com/NVIDIA/RULER) | RULER protocol and prepared data contract | Apache-2.0 |
| [HELMET](https://github.com/princeton-nlp/HELMET) at `af609c4d51b97fc35012099380aa889da961c42d` | HELMET preparation and scoring behavior | MIT |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | execution environment and benchmark protocol | Apache-2.0 |
| [MultiPL-E](https://github.com/nuprl/MultiPL-E) | language runtimes and execution protocol | BSD 3-Clause with an additional machine-learning-training restriction; see upstream `LICENSE` |
| [DS-1000](https://github.com/xlang-ai/DS-1000) | benchmark and execution protocol | CC BY-SA 4.0 |

Some NCP-ArchPreview architecture compatibility code has historical roots in
NVIDIA/Megatron-derived code. For any such portions, the following BSD
3-Clause notice is preserved:

> Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

The repository does not redistribute third-party model weights, benchmark
datasets, or pinned source checkouts. Public OCI images may contain installed
third-party runtimes; each image records its base reference and remains subject
to the licenses included by that base. In particular, the published MultiPL-E
sandbox inherits MultiPL-E's additional restriction and must not be treated as
an unqualified OSI-open image.

License names above summarize upstream metadata for convenience. Consult each
linked upstream license before redistribution.
