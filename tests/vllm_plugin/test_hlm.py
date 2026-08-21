"""Focused math tests for the request-scoped HLM routes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ncp_olmo_eval.vllm_plugin.hlm import (
        ConceptLMCrossCumsumRoute,
        ConceptLMDepthDD,
        ConceptLMDiagResidualRoute,
        ConceptLMProductCodebook,
        ConceptLMSelfCumsumDD,
        _append_hlm_kv,
        _causal_prefill_attention,
        _legacy_decode_attention,
    )
    from ncp_olmo_eval.vllm_plugin.model import _append_graph_padding_rows
    from ncp_olmo_eval.vllm_plugin.routes import ConceptLMStage3Routes
    from ncp_olmo_eval.vllm_plugin.state import HLMKVState


@unittest.skipIf(torch is None, "torch is not installed in the host-only test environment")
class TestIncrementalHLMMath(unittest.TestCase):
    def test_graph_padding_preserves_request_rows_and_appends_zeros(self) -> None:
        first = torch.tensor([1.0, 2.0])
        second = torch.tensor([3.0, 4.0])
        zero = torch.zeros(2)
        rows = [first, second]

        _append_graph_padding_rows(rows, target_length=4, zero=zero)

        self.assertEqual(len(rows), 4)
        self.assertIs(rows[0], first)
        self.assertIs(rows[1], second)
        self.assertIs(rows[2], zero)
        self.assertIs(rows[3], zero)

    def test_graph_padding_rejects_more_request_rows_than_inputs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceed"):
            _append_graph_padding_rows(
                [torch.zeros(2), torch.zeros(2)],
                target_length=1,
                zero=torch.zeros(2),
            )

    def test_hlm_kv_append_reuses_then_grows_storage(self) -> None:
        state = HLMKVState()
        first_key = torch.randn(2, 3, 4)
        first_value = torch.randn(2, 3, 4)
        cached_key, cached_value = _append_hlm_kv(
            state,
            first_key,
            first_value,
        )
        first_pointer = state.key.data_ptr()

        self.assertEqual(state.length, 3)
        self.assertEqual(state.key.shape, (2, 16, 4))
        torch.testing.assert_close(cached_key, first_key)
        torch.testing.assert_close(cached_value, first_value)

        second_key = torch.randn(2, 2, 4)
        second_value = torch.randn(2, 2, 4)
        cached_key, cached_value = _append_hlm_kv(
            state,
            second_key,
            second_value,
        )

        self.assertEqual(state.length, 5)
        self.assertEqual(state.key.data_ptr(), first_pointer)
        torch.testing.assert_close(
            cached_key,
            torch.cat((first_key, second_key), dim=1),
        )
        torch.testing.assert_close(
            cached_value,
            torch.cat((first_value, second_value), dim=1),
        )

        third_key = torch.randn(2, 12, 4)
        third_value = torch.randn(2, 12, 4)
        cached_key, cached_value = _append_hlm_kv(
            state,
            third_key,
            third_value,
        )

        self.assertEqual(state.length, 17)
        self.assertEqual(state.key.shape, (2, 32, 4))
        self.assertNotEqual(state.key.data_ptr(), first_pointer)
        torch.testing.assert_close(
            cached_key,
            torch.cat((first_key, second_key, third_key), dim=1),
        )
        torch.testing.assert_close(
            cached_value,
            torch.cat((first_value, second_value, third_value), dim=1),
        )

    def test_batched_legacy_decode_matches_serial_attention(self) -> None:
        torch.manual_seed(5)
        query = torch.randn(3, 2, 1, 4)
        key = torch.randn(3, 2, 7, 4)
        value = torch.randn(3, 2, 7, 4)

        expected = torch.cat(
            [
                _legacy_decode_attention(
                    query[index : index + 1],
                    key[index : index + 1],
                    value[index : index + 1],
                    scale=0.5,
                    output_dtype=query.dtype,
                )
                for index in range(3)
            ],
            dim=0,
        )
        result = _legacy_decode_attention(
            query,
            key,
            value,
            scale=0.5,
            output_dtype=query.dtype,
        )

        torch.testing.assert_close(result, expected)

    def test_batched_prefill_attention_matches_explicit_causal_math(self) -> None:
        torch.manual_seed(7)
        query = torch.randn(2, 5, 4)
        key = torch.randn(2, 5, 4)
        value = torch.randn(2, 5, 4)
        scale = 0.5

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        causal = torch.ones(5, 5, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        expected = torch.matmul(torch.softmax(scores, dim=-1), value)
        for flash_attn_version in (2, 3):
            with self.subTest(flash_attn_version=flash_attn_version):
                result = _causal_prefill_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    prefix_length=0,
                    sliding_window=16,
                    flash_attn_version=flash_attn_version,
                )
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

    def test_continuation_prefill_attention_uses_offset_causal_mask(self) -> None:
        torch.manual_seed(11)
        query = torch.randn(1, 3, 4)
        key = torch.randn(1, 5, 4)
        value = torch.randn(1, 5, 4)
        scale = 0.5

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        query_positions = torch.arange(2, 5)
        key_positions = torch.arange(5)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal, float("-inf"))
        expected = torch.matmul(torch.softmax(scores, dim=-1), value)
        for flash_attn_version in (2, 3):
            with self.subTest(flash_attn_version=flash_attn_version):
                result = _causal_prefill_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    prefix_length=2,
                    sliding_window=None,
                    flash_attn_version=flash_attn_version,
                )
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

    def test_decode_attention_applies_left_sliding_window(self) -> None:
        torch.manual_seed(13)
        query = torch.randn(1, 1, 4)
        key = torch.randn(1, 5, 4)
        value = torch.randn(1, 5, 4)
        scale = 0.5

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        window = torch.tensor([[False, False, True, True, True]])
        scores = scores.masked_fill(~window, float("-inf"))
        expected = torch.matmul(torch.softmax(scores, dim=-1), value)
        for flash_attn_version in (2, 3):
            with self.subTest(flash_attn_version=flash_attn_version):
                result = _causal_prefill_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    prefix_length=4,
                    sliding_window=2,
                    flash_attn_version=flash_attn_version,
                )
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

    def test_depth_dd_uses_input_and_all_raw_layers_without_softmax(self) -> None:
        route = ConceptLMDepthDD(hidden_size=2, layer_index=1)
        with torch.no_grad():
            route.w1.weight.zero_()
            route.w2.weight.zero_()
            route.static_a.copy_(torch.tensor([0.25, 0.5, 1.0]))
        history = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]]])

        result = route(history[:, -1], history)

        torch.testing.assert_close(result, torch.tensor([[5.25, 10.5]]))

    def test_residual_route_softmaxes_sources_then_applies_diagonal(self) -> None:
        route = ConceptLMDiagResidualRoute(hidden_size=2, num_sources=2)
        with torch.no_grad():
            route.w1.weight.zero_()
            route.w2.weight.zero_()
            route.residual_diag.copy_(torch.tensor([2.0, 3.0]))

        result = route(
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        )

        torch.testing.assert_close(result, torch.tensor([[14.0, 29.0]]))

    def test_residual_route_applies_gate_only_to_its_update(self) -> None:
        route = ConceptLMDiagResidualRoute(hidden_size=2, num_sources=2)
        with torch.no_grad():
            route.w1.weight.zero_()
            route.w2.weight.zero_()
            route.residual_diag.copy_(torch.tensor([2.0, 3.0]))

        result = route(
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            residual_scale=torch.tensor(0.25),
        )

        torch.testing.assert_close(result, torch.tensor([[11.0, 22.25]]))

    def test_decoder_depth_dd_softmaxes_static_routes(self) -> None:
        route = ConceptLMDepthDD(
            hidden_size=2,
            layer_index=0,
            use_softmax=True,
        )
        with torch.no_grad():
            route.w1.weight.zero_()
            route.w2.weight.zero_()
            route.static_a.copy_(torch.tensor([0.0, 0.0]))
        history = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

        result = route(history[:, -1], history)

        torch.testing.assert_close(result, torch.tensor([[4.0, 6.0]]))

    def test_self_cumsum_route_matches_reference_recurrence(self) -> None:
        route = ConceptLMSelfCumsumDD()
        with torch.no_grad():
            route.alpha.copy_(torch.tensor(0.5))
        current = torch.tensor([[2.0, 4.0]])
        previous = torch.tensor([[3.0, 5.0]])

        result, state = route(current, previous)

        expected = current + torch.tanh(torch.tensor(0.5)) * previous
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(state, expected)

    def test_cross_cumsum_route_reads_only_final_source(self) -> None:
        route = ConceptLMCrossCumsumRoute()
        with torch.no_grad():
            route.beta.copy_(torch.tensor(2.0))
        target = torch.tensor([[10.0, 20.0]])
        sources = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        result = route(target, sources, residual_scale=torch.tensor(0.25))

        torch.testing.assert_close(result, torch.tensor([[11.5, 22.0]]))

    def test_raw_logits_codebook_does_not_apply_softmax(self) -> None:
        quantizer = ConceptLMProductCodebook(
            hidden_size=4,
            num_codebooks=2,
            codebook_size=2,
        )
        with torch.no_grad():
            quantizer.codebook[0].copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            )
            quantizer.codebook[1].copy_(
                torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            )
        logits = torch.tensor([[[2.0, -1.0], [0.5, 3.0]]])

        result = torch.einsum(
            "bhk,hkd->bhd",
            logits,
            quantizer.stacked(),
        ).reshape(1, 4)

        torch.testing.assert_close(result, torch.tensor([[2.0, -1.0, 1.0, 6.0]]))

    def test_stage3_route_parameter_names_match_checkpoint(self) -> None:
        routes = ConceptLMStage3Routes(
            backend_config=SimpleNamespace(
                hidden_size=8,
                encoder_layers=2,
                decoder_layers=2,
                hlm_layers=2,
                dd_self_mode="dd",
            ),
            epsilon=1.0e-5,
        )

        names = set(dict(routes.named_parameters()))

        self.assertIn("dd_encoder_self_dd.depth_dds.1.static_a", names)
        self.assertIn("dd_two_route_add.decoder_dds.1.w2.weight", names)
        self.assertIn("dd_two_route_add.concept_routes.1.final_diag", names)
        self.assertIn("decoder_read_encoder_routes.1.residual_diag", names)
        self.assertIn("decoder_read_concept_routes.1.residual_diag", names)
        self.assertIn("final_read_concept_gate_logits", names)

    def test_cumsum_route_parameter_names_match_checkpoint(self) -> None:
        routes = ConceptLMStage3Routes(
            backend_config=SimpleNamespace(
                hidden_size=8,
                encoder_layers=2,
                decoder_layers=2,
                hlm_layers=2,
                dd_self_mode="cumsum",
            ),
            epsilon=1.0e-5,
        )

        names = set(dict(routes.named_parameters()))

        self.assertIn("dd_encoder_self_dd.alpha", names)
        self.assertIn("dd_two_route_add.decoder_cumsum_dd.alpha", names)
        self.assertIn("dd_two_route_add.concept_routes.1.final_beta", names)
        self.assertIn("decoder_read_encoder_routes.1.beta", names)
        self.assertIn("decoder_read_concept_routes.1.beta", names)
        self.assertNotIn("dd_encoder_self_dd.depth_dds.0.static_a", names)


if __name__ == "__main__":
    unittest.main()
