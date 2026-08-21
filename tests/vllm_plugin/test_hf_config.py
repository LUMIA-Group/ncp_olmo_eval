"""Tests for the serializable Transformers config shim."""

from __future__ import annotations

import tempfile
import unittest

try:
    from ncp_olmo_eval.vllm_plugin.hf_config import ConceptLMV22VQConfig
    from ncp_olmo_eval.vllm_plugin.plugin import _register_transformers_config
    from transformers import AutoConfig
except ModuleNotFoundError:
    AutoConfig = None
    ConceptLMV22VQConfig = None
    _register_transformers_config = None


@unittest.skipIf(AutoConfig is None, "transformers is not installed")
class TestConceptLMHFConfig(unittest.TestCase):
    def test_vllm_dimension_aliases(self) -> None:
        config = ConceptLMV22VQConfig(
            num_layers=32,
            num_query_groups=32,
            ffn_hidden_size=11008,
            layernorm_epsilon=1.0e-6,
        )

        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_key_value_heads, 32)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.rms_norm_eps, 1.0e-6)
        self.assertFalse(config.tie_word_embeddings)

    def test_auto_config_round_trip(self) -> None:
        _register_transformers_config()
        config = ConceptLMV22VQConfig(
            num_layers=32,
            num_query_groups=32,
            ffn_hidden_size=11008,
            layernorm_epsilon=1.0e-6,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            config.save_pretrained(output_dir)
            loaded = AutoConfig.from_pretrained(output_dir)

        self.assertIsInstance(loaded, ConceptLMV22VQConfig)
        self.assertEqual(loaded.num_hidden_layers, 32)
        self.assertEqual(loaded.num_key_value_heads, 32)


if __name__ == "__main__":
    unittest.main()
