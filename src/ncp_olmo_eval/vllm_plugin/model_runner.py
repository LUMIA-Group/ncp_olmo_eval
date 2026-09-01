"""vLLM 0.13 model-runner adapter for the NCP DFlash proposer."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .ncp_dflash_proposer import ConceptLMDFlashProposer
from .ncp_dflash_state import append_telemetry, target_model

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.utils import CommonAttentionMetadata


VLLM_DFLASH_VERSION = "0.13.0"


def dflash_vllm_013_enabled() -> bool:
    """Return whether this process explicitly requested the DFlash adapter."""

    return os.environ.get("CONCEPTLM_VLLM_ENABLE_NCP_DFLASH", "0") == "1"


def require_vllm_013() -> None:
    """Fail closed instead of silently running against another vLLM ABI."""

    installed = version("vllm")
    if installed != VLLM_DFLASH_VERSION:
        raise RuntimeError(
            "NCP DFlash is implemented against vLLM "
            f"{VLLM_DFLASH_VERSION}, got {installed}"
        )


class ConceptLMDFlashGPUModelRunner(GPUModelRunner):
    """Replace vLLM 0.13's ngram placeholder with the NCP DFlash proposer.

    vLLM 0.13 has the complete speculative scheduler and rejection sampler but
    no public custom-proposer method.  The engine is configured with ``ngram``
    solely to enable that upstream plumbing; this runner owns proposal
    generation and never calls the ngram implementation.
    """

    def __init__(self, vllm_config: Any, device: "torch.device") -> None:
        require_vllm_013()
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "ngram":
            raise ValueError(
                "the vLLM 0.13 DFlash adapter requires ngram scheduler plumbing"
            )
        super().__init__(vllm_config, device)
        self.drafter = ConceptLMDFlashProposer(vllm_config)
        append_telemetry(
            "vllm_013_model_runner_initialized",
            vllm_version=VLLM_DFLASH_VERSION,
            scheduler_method="ngram",
            proposer="ncp_dflash",
        )

    def _bookkeeping_sync(
        self,
        scheduler_output: Any,
        sampler_output: Any,
        logits: Any,
        hidden_states: Any,
        num_scheduled_tokens: int,
        spec_decode_metadata: Any,
    ) -> Any:
        """Finalize target state after rejection and before the next proposal."""

        result = super()._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            num_scheduled_tokens,
            spec_decode_metadata,
        )
        if spec_decode_metadata is not None:
            target_model().finalize_dflash_transactions(
                [str(req_id) for req_id in result[4]],
                result[2],
            )
        return result

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: "torch.Tensor | list[list[int]]",
        sampling_metadata: "SamplingMetadata",
        hidden_states: "torch.Tensor",
        sample_hidden_states: "torch.Tensor",
        aux_hidden_states: "list[torch.Tensor] | None",
        spec_decode_metadata: "SpecDecodeMetadata | None",
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> "list[list[int]] | torch.Tensor":
        """Generate drafts from target-owned features, not prompt ngrams."""
        del (
            scheduler_output,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            common_attn_metadata,
        )
        if not isinstance(sampled_token_ids, list):
            raise TypeError(
                "vLLM 0.13 NCP DFlash requires CPU-side sampled token IDs"
            )
        input_batch: InputBatch = self.input_batch
        return self.drafter.propose(
            sampled_token_ids,
            input_batch.num_tokens_no_spec,
            input_batch.token_ids_cpu,
            request_ids=[str(req_id) for req_id in input_batch.req_ids],
        )

