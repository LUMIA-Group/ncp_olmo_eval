"""vLLM general-plugin registration hook."""

from __future__ import annotations

from .contract import ARCHITECTURE, MODEL_TYPE, STANDALONE_HF_ARCHITECTURE
from .hf_config import ConceptLMV22VQConfig


def _register_transformers_config() -> None:
    from transformers import AutoConfig

    AutoConfig.register(
        MODEL_TYPE,
        ConceptLMV22VQConfig,
        exist_ok=True,
    )


def register() -> None:
    """Register the native ConceptLM class lazily in every vLLM process."""

    from vllm.model_executor.models import ModelRegistry

    _register_transformers_config()
    implementation = "ncp_olmo_eval.vllm_plugin.model:ConceptLMV22VQForCausalLM"
    supported = set(ModelRegistry.get_supported_archs())
    for architecture in (ARCHITECTURE, STANDALONE_HF_ARCHITECTURE):
        if architecture not in supported:
            ModelRegistry.register_model(architecture, implementation)
