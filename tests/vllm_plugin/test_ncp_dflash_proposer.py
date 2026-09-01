"""Contract tests for the conservative NCP DFlash proposal window."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from sys import modules
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from ncp_olmo_eval.vllm_plugin.ncp_dflash_proposer import (
    ConceptLMDFlashProposer,
    _dflash_context_kv,
    _dflash_sdpa_mask,
    _dynamic_two_tap_functional,
    _flash_varlen_dflash_attention,
    _install_draft_attention_backend,
    _install_runtime_local_mixer,
    _parse_active_batch_widths,
)


class _CountingProjection(torch.nn.Linear):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size, hidden_size, bias=False)
        self.call_count = 0
        self.projected_tokens = 0
        with torch.no_grad():
            self.weight.copy_(torch.eye(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.projected_tokens += int(hidden_states.numel() // hidden_states.shape[-1])
        return super().forward(hidden_states)


def _context_kv_test_layer() -> SimpleNamespace:
    hidden_size = 4
    layer = SimpleNamespace(
        num_attention_heads=2,
        head_size=2,
        k_proj=_CountingProjection(hidden_size),
        v_proj=_CountingProjection(hidden_size),
        k_norm=torch.nn.Identity(),
    )
    layer._split_heads = lambda hidden: hidden.unflatten(-1, (2, 2))
    layer.rotary = lambda hidden, positions: hidden
    return layer


def _create_dflash_block_mask(*_args: object, **_kwargs: object) -> object:
    return object()


class _RemoteDraftStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(flex_attention_compile=True)
        self.layers = torch.nn.ModuleList()

    def forward(self) -> object:
        return _create_dflash_block_mask()


class TestConceptLMDFlashProposer(unittest.TestCase):
    def test_active_batch_width_policy_uses_current_scheduler_batch(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.num_speculative_tokens = 16
        proposer.active_batch_widths = _parse_active_batch_widths(
            "1:8,2:8,4:4,7:2,8:0", proposer.num_speculative_tokens
        )

        self.assertEqual(
            [proposer._active_batch_width(batch_size) for batch_size in range(1, 10)],
            [8, 8, 4, 4, 2, 2, 2, 0, 0],
        )

    def test_active_batch_width_policy_rejects_invalid_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero"):
            _parse_active_batch_widths("8:17", 16)

    def test_flash_varlen_converts_cached_head_major_rows(self) -> None:
        layer = _context_kv_test_layer()
        layer.q_proj = _CountingProjection(4)
        layer.q_norm = torch.nn.Identity()
        layer._ncp_dflash_request_ids = ["request-a", "request-b"]
        layer._ncp_dflash_causal_lengths = [2, 3]
        slots = torch.zeros(2, 1, 4, 4)
        context = torch.zeros(2, 3, 4)
        anchor_positions = torch.tensor([[2], [3]])
        captured: dict[str, torch.Tensor] = {}

        def fake_flash_attn_varlen_func(**kwargs: object) -> torch.Tensor:
            captured["query"] = kwargs["q"]  # type: ignore[assignment]
            captured["key"] = kwargs["k"]  # type: ignore[assignment]
            captured["value"] = kwargs["v"]  # type: ignore[assignment]
            return torch.zeros_like(captured["query"])

        vllm_module = ModuleType("vllm")
        flash_module = ModuleType("vllm.vllm_flash_attn")
        flash_module.flash_attn_varlen_func = fake_flash_attn_varlen_func
        with (
            patch.dict(modules, {"vllm": vllm_module, "vllm.vllm_flash_attn": flash_module}),
            patch.object(torch, "autocast", return_value=nullcontext()),
        ):
            actual = _flash_varlen_dflash_attention(layer, slots, context, anchor_positions, None)

        self.assertEqual(tuple(actual.shape), (2, 1, 4, 4))
        self.assertEqual(tuple(captured["query"].shape), (8, 2, 2))
        self.assertEqual(tuple(captured["key"].shape), (13, 2, 2))
        self.assertEqual(tuple(captured["value"].shape), (13, 2, 2))

    def test_functional_dynamic_mixer_matches_remote_formula(self) -> None:
        torch.manual_seed(42)
        hidden = torch.randn(2, 1, 4, 8)
        base = torch.randn(2, 8)
        down = torch.randn(2, 8)
        up = torch.randn(8, 2)
        previous = torch.cat((hidden[:, :, :1], hidden[:, :, :-1]), dim=2)
        correction = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(hidden, down)), up
        )
        correction = correction.unflatten(-1, (4, 2)).transpose(-1, -2)
        correction = correction.repeat_interleave(2, dim=-1)
        coefficients = base + 0.1 * torch.tanh(correction)
        expected = coefficients[..., 0, :] * hidden + coefficients[..., 1, :] * previous

        actual = _dynamic_two_tap_functional(hidden, base, down, up, 2)
        torch.testing.assert_close(actual, expected)

    def test_static_runtime_mixer_keeps_learned_two_tap_path(self) -> None:
        mixer = torch.nn.Module()
        mixer.base_kernel = torch.nn.Parameter(torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
        layer = SimpleNamespace(
            pre_attention_conv=mixer,
            post_attention_conv=None,
            pre_feedforward_conv=None,
            post_feedforward_conv=None,
        )
        model = SimpleNamespace(layers=[layer])
        hidden = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

        self.assertEqual(_install_runtime_local_mixer(model, "static"), "static")
        actual = layer.pre_attention_conv(hidden)
        expected = torch.tensor([[[[7.0, 20.0], [11.0, 26.0]]]])
        torch.testing.assert_close(actual, expected)

    def test_none_runtime_mixer_removes_all_sites(self) -> None:
        layer = SimpleNamespace(
            pre_attention_conv=object(),
            post_attention_conv=object(),
            pre_feedforward_conv=object(),
            post_feedforward_conv=object(),
        )
        model = SimpleNamespace(layers=[layer])

        self.assertEqual(_install_runtime_local_mixer(model, "none"), "none")
        self.assertIsNone(layer.pre_attention_conv)
        self.assertIsNone(layer.post_attention_conv)
        self.assertIsNone(layer.pre_feedforward_conv)
        self.assertIsNone(layer.post_feedforward_conv)

    def test_hybrid_runtime_mixer_keeps_only_first_pre_attention_dynamic(self) -> None:
        def mixer() -> torch.nn.Module:
            module = torch.nn.Module()
            module.base_kernel = torch.nn.Parameter(torch.ones(2, 2))
            return module

        first_pre = mixer()
        first_post = mixer()
        second_pre = mixer()
        model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    pre_attention_conv=first_pre,
                    post_attention_conv=first_post,
                    pre_feedforward_conv=None,
                    post_feedforward_conv=None,
                ),
                SimpleNamespace(
                    pre_attention_conv=second_pre,
                    post_attention_conv=None,
                    pre_feedforward_conv=None,
                    post_feedforward_conv=None,
                ),
            ]
        )

        _install_runtime_local_mixer(model, "hybrid_first_pre_attention")
        self.assertNotIn("forward", first_pre.__dict__)
        self.assertIn("forward", first_post.__dict__)
        self.assertIn("forward", second_pre.__dict__)

    def test_sdpa_backend_skips_unused_remote_flex_mask(self) -> None:
        original = _create_dflash_block_mask
        model = _RemoteDraftStub()
        self.assertIsNotNone(model())
        try:
            self.assertEqual(_install_draft_attention_backend(model), "sdpa")
            self.assertIsNone(model())
            self.assertTrue(model._ncp_dflash_unused_block_mask_disabled)
        finally:
            model.forward.__func__.__globals__["_create_dflash_block_mask"] = original

    def test_context_kv_cache_reuses_request_prefix_after_row_reordering(self) -> None:
        layer = _context_kv_test_layer()
        layer._ncp_dflash_request_ids = ["request-a", "request-b"]
        layer._ncp_dflash_causal_lengths = [3, 2]
        first = torch.zeros(2, 4, 4)
        first[0, :3] = torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4])
        first[1, :2] = torch.tensor([[10.0] * 4, [11.0] * 4])

        _dflash_context_kv(layer, first, torch.tensor([[3], [2]]))

        self.assertEqual(layer.k_proj.projected_tokens, 5)
        self.assertEqual(layer.v_proj.projected_tokens, 5)
        self.assertEqual(layer.k_proj.call_count, 1)
        self.assertEqual(layer.v_proj.call_count, 1)

        second = torch.zeros(2, 5, 4)
        second[0, :3] = torch.tensor([[10.0] * 4, [11.0] * 4, [12.0] * 4])
        second[1, :4] = torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4])
        layer._ncp_dflash_request_ids = ["request-b", "request-a"]
        layer._ncp_dflash_causal_lengths = [3, 4]
        cached_key, cached_value = _dflash_context_kv(layer, second, torch.tensor([[3], [4]]))

        self.assertEqual(layer.k_proj.projected_tokens, 7)
        self.assertEqual(layer.v_proj.projected_tokens, 7)
        self.assertEqual(layer.k_proj.call_count, 2)
        self.assertEqual(layer.v_proj.call_count, 2)
        reference = _context_kv_test_layer()
        expected_key, expected_value = _dflash_context_kv(
            reference, second[:, :4], torch.tensor([[3], [4]])
        )
        torch.testing.assert_close(cached_key, expected_key)
        torch.testing.assert_close(cached_value, expected_value)
        self.assertEqual(set(layer._ncp_dflash_context_kv_cache), {"request-a", "request-b"})

    def test_context_kv_cache_prunes_inactive_requests(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.context_kv_cache = True
        layer = SimpleNamespace(
            _ncp_dflash_context_kv_cache={"request-a": object(), "request-b": object()}
        )
        proposer._draft_model = SimpleNamespace(layers=[layer])

        proposer._prune_context_kv_cache(["request-b"])

        self.assertEqual(set(layer._ncp_dflash_context_kv_cache), {"request-b"})

    def test_context_kv_cache_accepts_compact_absolute_suffix(self) -> None:
        layer = _context_kv_test_layer()
        layer._ncp_dflash_request_ids = ["request-a"]
        layer._ncp_dflash_causal_lengths = [2]
        initial = torch.tensor([[[1.0] * 4, [2.0] * 4]])
        _dflash_context_kv(layer, initial, torch.tensor([[2]]))

        layer._ncp_dflash_causal_lengths = [3]
        layer._ncp_dflash_context_offsets = [2]
        compact_suffix = torch.tensor([[[3.0] * 4]])
        actual_key, actual_value = _dflash_context_kv(layer, compact_suffix, torch.tensor([[3]]))

        reference = _context_kv_test_layer()
        full = torch.tensor([[[1.0] * 4, [2.0] * 4, [3.0] * 4]])
        expected_key, expected_value = _dflash_context_kv(reference, full, torch.tensor([[3]]))
        self.assertEqual(layer.k_proj.projected_tokens, 3)
        self.assertEqual(layer.v_proj.projected_tokens, 3)
        torch.testing.assert_close(actual_key, expected_key)
        torch.testing.assert_close(actual_value, expected_value)

    def test_sparse_uniform_suffix_accepts_noncontiguous_layer_context(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        captured: dict[str, torch.Tensor] = {}

        def context_for_layer(
            _features: torch.Tensor, shared: torch.Tensor, _layer_index: int
        ) -> torch.Tensor:
            context = torch.arange(
                shared.numel(), dtype=shared.dtype, device=shared.device
            ).reshape(1, shared.shape[-1], shared.shape[1]).transpose(1, 2)
            self.assertFalse(context.is_contiguous())
            return context

        def layer(
            slots: torch.Tensor,
            context: torch.Tensor,
            _hlm: object,
            _anchor_positions: torch.Tensor,
            _mask: object,
        ) -> torch.Tensor:
            captured["context"] = context
            return slots

        draft = SimpleNamespace(
            config=SimpleNamespace(draft_hidden_size=4, block_size=2),
            layers=[layer],
            feature_projection=lambda features: features,
            feature_norm=lambda features: features,
            _context_for_layer=context_for_layer,
            input_projection=lambda slots: slots,
            _hlm_for_layer=lambda *_args: None,
            final_norm=lambda slots: slots,
            output_projection=lambda slots: slots,
        )
        contexts = [torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)]

        output = proposer._forward_sparse_context(
            draft,
            contexts,
            request_ids=["request-a", "request-b"],
            prefix_lengths=[3, 3],
            anchor_embeddings=torch.randn(2, 1, 4),
            mask_embedding=torch.randn(4),
            anchor_positions=torch.tensor([[2], [2]]),
            hlm_hidden_states=torch.randn(2, 1, 4),
        )

        self.assertEqual(tuple(captured["context"].shape), (2, 2, 4))
        self.assertEqual(tuple(output.last_hidden_state.shape), (2, 1, 2, 4))

    def test_safe_window_never_completes_hlm_chunk(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 8192
        proposer.verification_mode = "intra_chunk_exact"

        self.assertEqual(
            [proposer.safe_proposal_count(length) for length in (1, 2, 3, 4)], [2, 1, 0, 0]
        )
        for prefix_length in range(1, 129):
            proposal_count = proposer.safe_proposal_count(prefix_length)
            processed_before_anchor = prefix_length - 1
            processed_after_draft = prefix_length + proposal_count
            if proposal_count:
                self.assertEqual(
                    processed_before_anchor // proposer.chunk_size,
                    processed_after_draft // proposer.chunk_size,
                )

    def test_safe_window_respects_context_limit(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 6
        proposer.verification_mode = "intra_chunk_exact"

        self.assertEqual(proposer.safe_proposal_count(5), 1)
        self.assertEqual(proposer.safe_proposal_count(6), 0)

    def test_safe_window_can_skip_one_token_proposals(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 8192
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_proposal_tokens_per_row = 2

        self.assertEqual(
            [proposer.safe_proposal_count(length) for length in (1, 2, 3, 4)], [2, 0, 0, 0]
        )

    def test_sequential_exact_proposes_at_most_one_token(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 8192
        proposer.verification_mode = "sequential_exact"

        self.assertEqual(
            [proposer.safe_proposal_count(length) for length in (1, 2, 3, 4)], [1, 1, 0, 0]
        )

    def test_segmented_kv_approx_can_cross_hlm_chunks(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 20
        proposer.verification_mode = "segmented_kv_approx"

        self.assertEqual(
            [proposer.safe_proposal_count(length) for length in (1, 4, 19, 20)], [16, 16, 1, 0]
        )

    def test_safe_window_respects_active_batch_width_cap(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.max_model_len = 20
        proposer.verification_mode = "segmented_kv_approx"
        proposer.min_proposal_tokens_per_row = 1

        self.assertEqual(proposer.safe_proposal_count(1, proposal_cap=4), 4)
        self.assertEqual(proposer.safe_proposal_count(1, proposal_cap=0), 0)

    def test_runtime_block_can_shrink_exact_draft_slots(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.verification_mode = "intra_chunk_exact"
        proposer.runtime_block_size = 2

        self.assertEqual(proposer._resolve_runtime_block_size(16), 2)

    def test_runtime_block_rejects_truncated_proposal_window(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.verification_mode = "intra_chunk_exact"
        proposer.runtime_block_size = 1

        with self.assertRaisesRegex(ValueError, "maximum proposal window"):
            proposer._resolve_runtime_block_size(16)

    def test_runtime_block_keeps_checkpoint_width_by_default(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 16
        proposer.verification_mode = "intra_chunk_exact"
        proposer.runtime_block_size = 0

        self.assertEqual(proposer._resolve_runtime_block_size(16), 16)

    def test_runtime_layer_count_can_early_exit(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.runtime_layer_count = 3

        self.assertEqual(proposer._resolve_runtime_layer_count(5), 3)

    def test_runtime_layer_count_keeps_checkpoint_depth_by_default(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.runtime_layer_count = 0

        self.assertEqual(proposer._resolve_runtime_layer_count(5), 5)

    def test_runtime_layer_count_rejects_missing_depth(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.runtime_layer_count = 6

        with self.assertRaisesRegex(ValueError, "within the checkpoint"):
            proposer._resolve_runtime_layer_count(5)

    def test_sdpa_mask_matches_random_anchor_visibility(self) -> None:
        mask = _dflash_sdpa_mask(torch.tensor([[2, 5]]), context_length=6, block_size=2)[0, 0]

        self.assertEqual(tuple(mask.shape), (4, 10))
        self.assertEqual(mask[0, :6].tolist(), [True, True, False, False, False, False])
        self.assertEqual(mask[1, :6].tolist(), [True, True, False, False, False, False])
        self.assertEqual(mask[2, :6].tolist(), [True, True, True, True, True, False])
        self.assertEqual(mask[3, :6].tolist(), [True, True, True, True, True, False])
        self.assertEqual(mask[0, 6:].tolist(), [True, True, False, False])
        self.assertEqual(mask[2, 6:].tolist(), [False, False, True, True])

    def test_context_padding_preserves_rows_and_zero_fills_tail(self) -> None:
        first = torch.arange(12).view(1, 3, 2, 2)
        second = torch.arange(4).view(1, 1, 2, 2) + 100

        padded = ConceptLMDFlashProposer._pad_context_batch([first, second])

        self.assertEqual(tuple(padded.shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(padded[0], first[0]))
        self.assertTrue(torch.equal(padded[1, 0], second[0, 0]))
        self.assertTrue(torch.equal(padded[1, 1:], torch.zeros_like(padded[1, 1:])))

    def test_batched_path_selector_matches_rowwise_reference(self) -> None:
        torch.manual_seed(42)
        hidden_size = 8
        selector_rank = 4
        model = SimpleNamespace(
            config=SimpleNamespace(selector_top_k=3),
            selector_previous_projection=torch.nn.Linear(hidden_size, selector_rank, bias=False),
            selector_candidate_projection=torch.nn.Linear(hidden_size, selector_rank, bias=False),
            selector_context_projection=torch.nn.Linear(hidden_size, selector_rank, bias=False),
        )
        hidden = torch.randn(3, 4, hidden_size)
        output_weight = torch.randn(11, hidden_size)
        embedding_weight = torch.randn(11, hidden_size)
        previous = [1, 2, 3]
        counts = [2, 1, 2]
        expected = [
            ConceptLMDFlashProposer._path_selector_tokens(
                model,
                hidden[row_index],
                output_weight,
                embedding_weight,
                previous[row_index],
                counts[row_index],
            )
            for row_index in range(3)
        ]

        actual = ConceptLMDFlashProposer._path_selector_batch(
            model, hidden, output_weight, embedding_weight, previous, counts
        )

        self.assertEqual(actual, expected)

    def test_proposal_batch_carries_request_ids_across_row_reordering(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose(
            [[11], [12]],
            [2, 2],
            torch.tensor([[1, 11], [2, 12]]),
            request_ids=["request-b", "request-a"],
        )

        self.assertEqual(captured, [(0, "request-b", [1, 11], 1), (1, "request-a", [2, 12], 1)])
        self.assertEqual(results, [[100], [101]])

    def test_adaptive_batch_skips_sparse_multi_request_proposal(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_eligible_batch = 2
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose(
            [[11], [12]],
            [1, 3],
            torch.tensor([[11, 0, 0], [1, 2, 12]]),
            request_ids=["request-a", "request-b"],
        )

        self.assertEqual(captured, [])
        self.assertEqual(results, [[], []])

    def test_active_batch_policy_caps_each_continuous_scheduler_step(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 8
        proposer.max_model_len = 32
        proposer.verification_mode = "segmented_kv_approx"
        proposer.min_proposal_tokens_per_row = 1
        proposer.min_eligible_batch = 1
        proposer.min_proposal_tokens_per_batch = 1
        proposer.max_batch_size = 8
        proposer.context_kv_cache = False
        proposer._draft_model = None
        proposer.active_batch_widths = ((2, 8), (4, 4), (8, 0))
        captured: list[list[tuple[int, str, list[int], int]]] = []

        def record(entries):
            captured.append(list(entries))
            return {
                row_index: [100 + row_index] * proposal_count
                for row_index, _, _, proposal_count in entries
            }

        proposer._propose_batch = record
        first = proposer.propose(
            [[11], [12], [13], [14]],
            [1, 1, 1, 1],
            torch.tensor([[11], [12], [13], [14]]),
            request_ids=["request-a", "request-b", "request-c", "request-d"],
        )
        second = proposer.propose(
            [[21], [22]],
            [1, 1],
            torch.tensor([[21], [22]]),
            request_ids=["request-c", "request-e"],
        )
        third = proposer.propose(
            [[31]] * 8,
            [1] * 8,
            torch.tensor([[31]] * 8),
            request_ids=[f"request-{index}" for index in range(8)],
        )

        self.assertEqual([entry[3] for entry in captured[0]], [4, 4, 4, 4])
        self.assertEqual([entry[3] for entry in captured[1]], [8, 8])
        self.assertEqual([len(row) for row in first], [4, 4, 4, 4])
        self.assertEqual([len(row) for row in second], [8, 8])
        self.assertEqual(third, [[] for _ in range(8)])

    def test_adaptive_batch_keeps_single_request_proposal(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_eligible_batch = 4
        proposer.max_batch_size = 1
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose([[11]], [1], torch.tensor([[11]]), request_ids=["request-a"])

        self.assertEqual(captured, [(0, "request-a", [11], 2)])
        self.assertEqual(results, [[100]])

    def test_adaptive_batch_skips_single_active_tail_in_batch_engine(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_eligible_batch = 4
        proposer.max_batch_size = 8
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose([[11]], [1], torch.tensor([[11]]), request_ids=["request-a"])

        self.assertEqual(captured, [])
        self.assertEqual(results, [[]])

    def test_adaptive_proposal_budget_skips_low_value_batch(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_eligible_batch = 1
        proposer.min_proposal_tokens_per_batch = 5
        proposer.max_batch_size = 8
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose(
            [[11], [12]], [1, 1], torch.tensor([[11], [12]]), request_ids=["request-a", "request-b"]
        )

        self.assertEqual(captured, [])
        self.assertEqual(results, [[], []])

    def test_adaptive_proposal_budget_keeps_valuable_batch(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)
        proposer.chunk_size = 4
        proposer.num_speculative_tokens = 2
        proposer.max_model_len = 32
        proposer.verification_mode = "intra_chunk_exact"
        proposer.min_eligible_batch = 1
        proposer.min_proposal_tokens_per_batch = 4
        proposer.max_batch_size = 8
        captured = []

        def record(entries):
            captured.extend(entries)
            return {row_index: [100 + row_index] for row_index, *_ in entries}

        proposer._propose_batch = record
        results = proposer.propose(
            [[11], [12]], [1, 1], torch.tensor([[11], [12]]), request_ids=["request-a", "request-b"]
        )

        self.assertEqual(captured, [(0, "request-a", [11], 2), (1, "request-b", [12], 2)])
        self.assertEqual(results, [[100], [101]])

    def test_proposal_requires_request_ids(self) -> None:
        proposer = ConceptLMDFlashProposer.__new__(ConceptLMDFlashProposer)

        with self.assertRaisesRegex(ValueError, "request IDs are required"):
            proposer.propose([[11]], [1], torch.tensor([[11]]))


if __name__ == "__main__":
    unittest.main()

