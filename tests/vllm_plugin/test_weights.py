"""Unit tests for native Megatron token-tower weight mapping."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ncp_olmo_eval.vllm_plugin.qkv import deinterleave_megatron_qkv

    try:
        from ncp_olmo_eval.vllm_plugin.model import ConceptLMV22VQForCausalLM
    except ImportError:
        ConceptLMV22VQForCausalLM = None
else:
    ConceptLMV22VQForCausalLM = None

from ncp_olmo_eval.vllm_plugin.weights import (
    Stage3WeightConfig,
    TokenWeightConfig,
    audit_stage3_weight_shapes,
    audit_token_weight_shapes,
    expected_stage3_weight_shapes,
    expected_token_weight_shapes,
    resolve_checkpoint_weight,
)


def token_config(*, qk_norm_weight_size: int | None = None) -> TokenWeightConfig:
    """Return a small full-MHA token-tower shape contract."""

    return TokenWeightConfig(
        hidden_size=8,
        intermediate_size=12,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        encoder_layers=2,
        decoder_layers=2,
        qk_norm_weight_size=qk_norm_weight_size,
    )


class TestTokenWeightAudit(unittest.TestCase):
    def test_exact_manifest_has_full_coverage(self) -> None:
        expected = expected_token_weight_shapes(token_config())
        manifest = dict(expected)
        manifest["decoder.final_layernorm._extra_state"] = (0,)
        manifest["concept_predictor.hlm_block.layers.0.mlp.linear_fc1.weight"] = (
            24,
            8,
        )

        report = audit_token_weight_shapes(manifest, token_config())

        self.assertTrue(report.ok)
        self.assertEqual(report.expected_parameter_count, 35)
        self.assertEqual(report.matched_parameter_count, 35)
        self.assertEqual(
            report.ignored_token_metadata,
            ("decoder.final_layernorm._extra_state",),
        )
        self.assertEqual(report.non_token_tensor_count, 1)

    def test_missing_and_shape_mismatch_fail(self) -> None:
        manifest = expected_token_weight_shapes(token_config())
        del manifest["encoder.layers.1.mlp.linear_fc2.weight"]
        manifest["output_layer.weight"] = (31, 8)

        report = audit_token_weight_shapes(manifest, token_config())

        self.assertFalse(report.ok)
        self.assertEqual(report.missing_parameters, ("encoder.layers.1.mlp.linear_fc2.weight",))
        self.assertEqual(
            report.shape_mismatches, ("output_layer.weight: expected (32, 8), found (31, 8)",)
        )

    def test_hf_wrapper_prefix_has_full_coverage(self) -> None:
        expected = expected_token_weight_shapes(token_config())
        manifest = {f"model.{name}": shape for name, shape in expected.items()}
        manifest["model.decoder.final_layernorm._extra_state"] = (0,)

        report = audit_token_weight_shapes(manifest, token_config())

        self.assertTrue(report.ok)
        self.assertEqual(report.matched_parameter_count, len(expected))

    def test_standard_hf_split_projections_have_full_coverage(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        manifest: dict[str, tuple[int, ...]] = {}
        for name, shape in expected.items():
            hf_name = f"model.{name}"
            if name == "embedding.word_embeddings.weight":
                manifest["model.embed_tokens.weight"] = shape
            elif name == "output_layer.weight":
                manifest["lm_head.weight"] = shape
            elif name.endswith(".self_attention.linear_qkv.weight"):
                prefix = hf_name.removesuffix(".self_attention.linear_qkv.weight")
                manifest[f"{prefix}.self_attn.q_proj.weight"] = (8, 8)
                manifest[f"{prefix}.self_attn.k_proj.weight"] = (8, 8)
                manifest[f"{prefix}.self_attn.v_proj.weight"] = (8, 8)
            elif name.endswith(".self_attention.linear_proj.weight"):
                manifest[hf_name.replace(".self_attention.linear_proj.", ".self_attn.o_proj.")] = (
                    shape
                )
            elif name.endswith(".self_attention.q_layernorm.weight"):
                manifest[hf_name.replace(".self_attention.q_layernorm.", ".self_attn.q_norm.")] = (
                    shape
                )
            elif name.endswith(".self_attention.k_layernorm.weight"):
                manifest[hf_name.replace(".self_attention.k_layernorm.", ".self_attn.k_norm.")] = (
                    shape
                )
            elif name.endswith(".mlp.linear_fc1.weight"):
                prefix = hf_name.removesuffix(".mlp.linear_fc1.weight")
                manifest[f"{prefix}.mlp.gate_proj.weight"] = (12, 8)
                manifest[f"{prefix}.mlp.up_proj.weight"] = (12, 8)
            elif name.endswith(".mlp.linear_fc2.weight"):
                manifest[hf_name.replace(".mlp.linear_fc2.", ".mlp.down_proj.")] = shape
            else:
                manifest[hf_name] = shape

        report = audit_token_weight_shapes(manifest, config)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.matched_parameter_count, len(expected))

    def test_incomplete_hf_qkv_is_reported_missing(self) -> None:
        config = token_config()
        expected = expected_token_weight_shapes(config)
        name = "encoder.layers.0.self_attention.linear_qkv.weight"
        manifest = dict(expected)
        del manifest[name]
        manifest["model.encoder.layers.0.self_attn.q_proj.weight"] = (8, 8)
        manifest["model.encoder.layers.0.self_attn.k_proj.weight"] = (8, 8)

        report = audit_token_weight_shapes(manifest, config)

        self.assertFalse(report.ok)
        self.assertIn(name, report.missing_parameters)


class TestCheckpointKeyResolution(unittest.TestCase):
    def test_arbitrary_hf_wrapper_prefix_is_stripped(self) -> None:
        parameter = "decoder_read_encoder_routes.0.w1.weight"
        target = resolve_checkpoint_weight(f"module.model.{parameter}", {parameter})

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)
        self.assertIsNone(target.shard_id)

    def test_hf_q_projection_maps_to_qkv_shard(self) -> None:
        parameter = "encoder.layers.0.self_attention.linear_qkv.weight"
        target = resolve_checkpoint_weight(
            "model.encoder.layers.0.self_attn.q_proj.weight", {parameter}
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)
        self.assertEqual(target.shard_id, "q")
        self.assertFalse(target.deinterleave_megatron_qkv)

    def test_hf_tower_norm_maps_to_final_layernorm(self) -> None:
        parameter = "decoder.final_layernorm.weight"
        target = resolve_checkpoint_weight("model.decoder.norm.weight", {parameter})

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.parameter_name, parameter)


class TestStage3WeightAudit(unittest.TestCase):
    def test_stage3_dimensions_define_722_parameters(self) -> None:
        config = Stage3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=4096,
                intermediate_size=11008,
                vocab_size=100278,
                num_attention_heads=32,
                num_key_value_heads=32,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=128,
            num_codebooks=32,
        )

        self.assertEqual(len(expected_stage3_weight_shapes(config)), 722)

    def test_exact_full_manifest_has_722_parameters(self) -> None:
        config = Stage3WeightConfig(
            token=token_config(),
            hlm_layers=2,
            codebook_size=4,
            num_codebooks=2,
        )
        manifest = expected_stage3_weight_shapes(config)
        manifest["decoder.final_layernorm._extra_state"] = (0,)
        manifest["concept_predictor.hlm_block.final_layernorm._extra_state"] = (0,)

        report = audit_stage3_weight_shapes(manifest, config)

        self.assertTrue(report.ok)
        self.assertEqual(report.expected_parameter_count, 114)
        self.assertEqual(report.matched_parameter_count, 114)
        self.assertEqual(len(report.ignored_metadata), 2)

    def test_hf_wrapped_full_manifest_has_complete_route_coverage(self) -> None:
        config = Stage3WeightConfig(
            token=token_config(), hlm_layers=2, codebook_size=4, num_codebooks=2
        )
        expected = expected_stage3_weight_shapes(config)
        manifest = {f"model.{name}": shape for name, shape in expected.items()}

        report = audit_stage3_weight_shapes(manifest, config)

        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.matched_parameter_count, 114)
        self.assertEqual(report.ignored_metadata, ())

    def test_cumsum_manifest_uses_scalar_route_contract(self) -> None:
        config = Stage3WeightConfig(
            token=token_config(),
            hlm_layers=2,
            codebook_size=4,
            num_codebooks=2,
            dd_self_mode="cumsum",
        )

        expected = expected_stage3_weight_shapes(config)

        self.assertIn("dd_encoder_self_dd.alpha", expected)
        self.assertIn("concept_predictor.concept_self_dd.alpha", expected)
        self.assertIn("dd_two_route_add.decoder_cumsum_dd.alpha", expected)
        self.assertIn("concept_predictor.concept_read_encoder_routes.1.beta", expected)
        self.assertIn("decoder_read_encoder_routes.1.beta", expected)
        self.assertIn("decoder_read_concept_routes.1.beta", expected)
        self.assertIn("dd_two_route_add.concept_routes.1.final_beta", expected)
        self.assertNotIn("dd_encoder_self_dd.depth_dds.0.static_a", expected)
        self.assertNotIn("dd_two_route_add.concept_routes.1.final_diag", expected)

        report = audit_stage3_weight_shapes(dict(expected), config)
        self.assertTrue(report.ok, report.to_dict())

    def test_per_head_cumsum_manifest_uses_head_dim_qk_norms(self) -> None:
        config = Stage3WeightConfig(
            token=token_config(qk_norm_weight_size=4),
            hlm_layers=2,
            codebook_size=4,
            num_codebooks=2,
            dd_self_mode="cumsum",
        )

        expected = expected_stage3_weight_shapes(config)

        self.assertEqual(
            expected["encoder.layers.0.self_attention.q_layernorm.weight"],
            (4,),
        )
        self.assertEqual(
            expected[
                "concept_predictor.hlm_block.layers.0.self_attention.k_layernorm.weight"
            ],
            (4,),
        )
        report = audit_stage3_weight_shapes(dict(expected), config)
        self.assertTrue(report.ok, report.to_dict())


@unittest.skipIf(
    torch is None or ConceptLMV22VQForCausalLM is None,
    "torch and the pinned vLLM runtime are required",
)
class TestModelHFWeightLoading(unittest.TestCase):
    class _Parameter:
        def __init__(self) -> None:
            self.shards: list[str | int | None] = []
            self.loaded: dict[str | int | None, object] = {}

        def weight_loader(
            self, parameter: object, loaded_weight: object, shard_id: str | int | None = None
        ) -> None:
            if parameter is not self:
                raise AssertionError("weight loader received the wrong parameter")
            self.shards.append(shard_id)
            self.loaded[shard_id] = loaded_weight.clone()

        def canonical_weight(self):
            if None in self.loaded:
                return self.loaded[None]
            if set(self.loaded) == {"q", "k", "v"}:
                return torch.cat(
                    (self.loaded["q"], self.loaded["k"], self.loaded["v"]),
                    dim=0,
                )
            if set(self.loaded) == {0, 1}:
                return torch.cat((self.loaded[0], self.loaded[1]), dim=0)
            raise AssertionError(f"incomplete parameter shards: {sorted(map(str, self.loaded))}")

    class _Module:
        def __init__(self, params: dict[str, object]) -> None:
            self.params = params

        def named_parameters(self, remove_duplicate: bool = False):
            del remove_duplicate
            return iter(self.params.items())

    class _BackendConfig:
        num_attention_heads = 2
        num_key_value_heads = 2
        dd_self_mode = "dd"

    class _Model:
        def named_parameters(self):
            for root in ("token_backbone", "highlevel", "routes"):
                module = getattr(self, root)
                for name, parameter in module.named_parameters():
                    yield f"{root}.{name}", parameter

    def _build_fake_model(self, names: dict[str, tuple[int, ...]]):
        params = {name: self._Parameter() for name in names}
        token = {
            name: parameter
            for name, parameter in params.items()
            if name.startswith(("embedding.", "encoder.", "decoder.", "output_layer."))
        }
        highlevel = {
            name: parameter
            for name, parameter in params.items()
            if name.startswith(
                ("concept_vq_input_norm.", "concept_quantizer.", "concept_predictor.")
            )
        }
        routes = {
            name: parameter
            for name, parameter in params.items()
            if name not in token and name not in highlevel
        }
        model = self._Model()
        model.token_backbone = self._Module(token)
        model.highlevel = self._Module(highlevel)
        model.routes = self._Module(routes)
        model.backend_config = self._BackendConfig()
        return model, params

    @staticmethod
    def _standard_hf_name(name: str) -> str:
        if name == "embedding.word_embeddings.weight":
            return "model.embed_tokens.weight"
        if name == "output_layer.weight":
            return "lm_head.weight"
        hf_name = f"model.{name}"
        for native_part, hf_part in (
            (".self_attention.linear_proj.", ".self_attn.o_proj."),
            (".self_attention.q_layernorm.", ".self_attn.q_norm."),
            (".self_attention.k_layernorm.", ".self_attn.k_norm."),
            (".mlp.linear_fc2.", ".mlp.down_proj."),
        ):
            hf_name = hf_name.replace(native_part, hf_part)
        if hf_name.endswith(".final_layernorm.weight"):
            hf_name = hf_name.removesuffix(".final_layernorm.weight") + ".norm.weight"
        return hf_name

    def test_full_standard_hf_manifest_loads_all_722_parameters(self) -> None:
        config = Stage3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=4096,
                intermediate_size=11008,
                vocab_size=100278,
                num_attention_heads=32,
                num_key_value_heads=32,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=128,
            num_codebooks=32,
        )
        names = expected_stage3_weight_shapes(config)
        model, params = self._build_fake_model(names)

        weights = []
        for name in names:
            if name == "embedding.word_embeddings.weight":
                weights.append(("model.embed_tokens.weight", torch.zeros(1)))
            elif name == "output_layer.weight":
                weights.append(("lm_head.weight", torch.zeros(1)))
            elif name.endswith(".self_attention.linear_qkv.weight"):
                prefix = f"model.{name}".removesuffix(".self_attention.linear_qkv.weight")
                for projection in ("q", "k", "v"):
                    weights.append((f"{prefix}.self_attn.{projection}_proj.weight", torch.zeros(1)))
            elif name.endswith(".mlp.linear_fc1.weight"):
                prefix = f"model.{name}".removesuffix(".mlp.linear_fc1.weight")
                weights.extend(
                    (
                        (f"{prefix}.mlp.gate_proj.weight", torch.zeros(1)),
                        (f"{prefix}.mlp.up_proj.weight", torch.zeros(1)),
                    )
                )
            else:
                weights.append((self._standard_hf_name(name), torch.zeros(1)))

        loaded = ConceptLMV22VQForCausalLM.load_weights(model, weights)

        self.assertEqual(len(loaded), 722)
        qkv = params["encoder.layers.0.self_attention.linear_qkv.weight"]
        fc1 = params["encoder.layers.0.mlp.linear_fc1.weight"]
        self.assertEqual(qkv.shards, ["q", "k", "v"])
        self.assertEqual(fc1.shards, [0, 1])

    def test_full_cumsum_manifest_loads_every_parameter(self) -> None:
        config = Stage3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=32,
                intermediate_size=12,
                vocab_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=4,
            num_codebooks=32,
            dd_self_mode="cumsum",
        )
        names = expected_stage3_weight_shapes(config)
        model, _ = self._build_fake_model(names)
        model.backend_config.dd_self_mode = "cumsum"

        loaded = ConceptLMV22VQForCausalLM.load_weights(
            model,
            ((name, torch.zeros(shape)) for name, shape in names.items()),
        )

        self.assertEqual(len(loaded), len(names))
        self.assertEqual(len(names), 525)

    def test_native_and_standard_hf_keys_load_identical_values(self) -> None:
        config = Stage3WeightConfig(
            token=TokenWeightConfig(
                hidden_size=32,
                intermediate_size=12,
                vocab_size=32,
                num_attention_heads=2,
                num_key_value_heads=2,
                encoder_layers=16,
                decoder_layers=16,
            ),
            hlm_layers=8,
            codebook_size=4,
            num_codebooks=32,
        )
        shapes = expected_stage3_weight_shapes(config)
        native_weights = []
        hf_weights = []
        for index, (name, shape) in enumerate(shapes.items()):
            numel = 1
            for dimension in shape:
                numel *= dimension
            value = torch.arange(numel, dtype=torch.float32).reshape(shape) + index
            native_weights.append((name, value))
            if name.endswith(".self_attention.linear_qkv.weight"):
                query, key, value_shard = deinterleave_megatron_qkv(
                    value,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
                prefix = f"model.{name}".removesuffix(
                    ".self_attention.linear_qkv.weight"
                )
                hf_weights.extend(
                    (
                        (f"{prefix}.self_attn.q_proj.weight", query),
                        (f"{prefix}.self_attn.k_proj.weight", key),
                        (f"{prefix}.self_attn.v_proj.weight", value_shard),
                    )
                )
            elif name.endswith(".mlp.linear_fc1.weight"):
                gate, up = value.chunk(2, dim=0)
                prefix = f"model.{name}".removesuffix(".mlp.linear_fc1.weight")
                hf_weights.extend(
                    (
                        (f"{prefix}.mlp.gate_proj.weight", gate),
                        (f"{prefix}.mlp.up_proj.weight", up),
                    )
                )
            else:
                hf_weights.append((self._standard_hf_name(name), value))

        native_model, native_params = self._build_fake_model(shapes)
        hf_model, hf_params = self._build_fake_model(shapes)
        ConceptLMV22VQForCausalLM.load_weights(native_model, native_weights)
        ConceptLMV22VQForCausalLM.load_weights(hf_model, hf_weights)

        self.assertEqual(set(native_params), set(hf_params))
        for name in native_params:
            self.assertTrue(
                torch.equal(
                    native_params[name].canonical_weight(),
                    hf_params[name].canonical_weight(),
                ),
                name,
            )


@unittest.skipIf(torch is None, "torch is not installed in the host-only test environment")
class TestMegatronQKVMapping(unittest.TestCase):
    def test_full_mha_rows_are_deinterleaved_by_head(self) -> None:
        row_ids = torch.arange(12, dtype=torch.float32).reshape(6, 2)

        query, key, value = deinterleave_megatron_qkv(
            row_ids,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

        self.assertEqual(query[:, 0].tolist(), [0.0, 6.0])
        self.assertEqual(key[:, 0].tolist(), [2.0, 8.0])
        self.assertEqual(value[:, 0].tolist(), [4.0, 10.0])


if __name__ == "__main__":
    unittest.main()
