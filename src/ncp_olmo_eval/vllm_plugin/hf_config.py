"""Transformers config shim for native ConceptLM artifacts."""

from typing import Any

from transformers import PretrainedConfig

from .contract import MODEL_TYPE


class ConceptLMV22VQConfig(PretrainedConfig):
    """Generic field-preserving config used before vLLM builds the model."""

    model_type = MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        aliases = (
            ("num_layers", "num_hidden_layers", int),
            ("num_query_groups", "num_key_value_heads", int),
            ("ffn_hidden_size", "intermediate_size", int),
            ("layernorm_epsilon", "rms_norm_eps", float),
        )
        for source, target, converter in aliases:
            if hasattr(self, source):
                setattr(self, target, converter(getattr(self, source)))
        self.tie_word_embeddings = False
