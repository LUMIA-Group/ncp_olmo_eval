"""Cross-chunk rollback tests for transactional NCP DFlash state."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from ncp_olmo_eval.vllm_plugin.model import ConceptLMV22VQForCausalLM, _DFlashStateTransaction
from ncp_olmo_eval.vllm_plugin.state import (
    ConceptRequestStateStore,
    ScheduledRequestSegment,
    active_tensor_buffer,
    append_tensor_buffer,
    clear_tensor_buffer,
    snapshot_request_state,
)


class _FakeHighLevel(torch.nn.Module):
    def advance(self, state, encoder_chunk, layer_chunks) -> None:
        state.hlm_kv[0].length += 1
        append_tensor_buffer(state.hlm_raw_layer_states[0], layer_chunks[0].view(1, -1))
        append_tensor_buffer(state.predicted_concepts, encoder_chunk.view(1, -1))

    def advance_batch(self, states, encoder_chunks, layer_chunks) -> None:
        for row, state in enumerate(states):
            self.advance(state, encoder_chunks[row], tuple(values[row] for values in layer_chunks))


class TestDFlashStateTransaction(unittest.TestCase):
    def test_rejection_truncates_same_chunk_without_snapshot(self) -> None:
        model = ConceptLMV22VQForCausalLM.__new__(ConceptLMV22VQForCausalLM)
        torch.nn.Module.__init__(model)
        model.backend_config = SimpleNamespace(chunk_size=4, hidden_size=2, encoder_layers=1)
        model._draft_capture_layer_ids = (1,)
        model.request_states = ConceptRequestStateStore(
            encoder_layers=1, hlm_layers=1, speculative_chunk_size=4, draft_layers=1
        )
        state = model.request_states._new_state("request-a")
        state.hlm_kv[0].length = 1
        append_tensor_buffer(state.hlm_raw_layer_states[0], torch.ones(1, 2))
        append_tensor_buffer(state.predicted_concepts, torch.ones(1, 2))
        append_tensor_buffer(state.draft_decoder_layers[0], torch.ones(4, 2))
        state.next_token_position = 4
        segment = ScheduledRequestSegment(
            req_id="request-a",
            flat_start=0,
            flat_end=3,
            position_start=4,
            position_end=7,
            state=state,
        )
        transaction = _DFlashStateTransaction(segment=segment, snapshot=None)

        verifier_rows = torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
        append_tensor_buffer(state.pending_encoder_final, verifier_rows)
        append_tensor_buffer(state.pending_encoder_layers[0], verifier_rows + 10)
        append_tensor_buffer(state.draft_decoder_layers[0], verifier_rows + 20)
        state.next_token_position = 7
        model._dflash_transactions = {"request-a": transaction}

        model.finalize_dflash_transactions(["request-a"], [[101, 102]])

        self.assertEqual(state.next_token_position, 6)
        self.assertEqual(state.pending_encoder_final.length, 2)
        torch.testing.assert_close(
            active_tensor_buffer(state.pending_encoder_final), verifier_rows[:2]
        )
        self.assertEqual(state.hlm_kv[0].length, 1)
        self.assertEqual(state.predicted_concepts.length, 1)
        self.assertEqual(state.draft_decoder_layers[0].length, 6)
        self.assertFalse(model._dflash_transactions)

    def test_proposal_context_resolves_state_by_request_id(self) -> None:
        model = ConceptLMV22VQForCausalLM.__new__(ConceptLMV22VQForCausalLM)
        torch.nn.Module.__init__(model)
        model._ncp_dflash_enabled = True
        model._draft_capture_layer_ids = (1,)
        model.backend_config = SimpleNamespace(chunk_size=4, hidden_size=2, vocab_size=4)
        embedding_weight = torch.arange(8, dtype=torch.float32).view(4, 2)
        output_weight = embedding_weight + 100
        model.token_backbone = SimpleNamespace(
            embedding=SimpleNamespace(word_embeddings=SimpleNamespace(weight=embedding_weight)),
            output_layer=SimpleNamespace(weight=output_weight),
        )
        model.request_states = ConceptRequestStateStore(
            encoder_layers=1, hlm_layers=1, speculative_chunk_size=4, draft_layers=1
        )
        request_a = model.request_states._new_state("request-a")
        request_b = model.request_states._new_state("request-b")
        append_tensor_buffer(
            request_a.draft_decoder_layers[0], torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        )
        append_tensor_buffer(
            request_b.draft_decoder_layers[0], torch.tensor([[7.0, 7.0], [8.0, 8.0]])
        )
        request_a.next_token_position = 2
        request_b.next_token_position = 2
        model.request_states.bind_input_batch(SimpleNamespace(req_ids=["request-a", "request-b"]))

        context, hlm_state, actual_embedding, actual_output = model.ncp_dflash_proposal_context(
            "request-b", 3
        )

        torch.testing.assert_close(context[0, :2, 0], torch.tensor([[7.0, 7.0], [8.0, 8.0]]))
        torch.testing.assert_close(context[0, 2, 0], torch.zeros(2))
        torch.testing.assert_close(hlm_state, torch.zeros(1, 1, 2))
        torch.testing.assert_close(actual_embedding, embedding_weight)
        torch.testing.assert_close(actual_output, output_weight)
        self.assertEqual(actual_embedding.data_ptr(), embedding_weight.data_ptr())
        self.assertEqual(actual_output.data_ptr(), output_weight.data_ptr())

    def test_rejection_replays_only_accepted_prefix_across_chunk_boundary(self) -> None:
        model = ConceptLMV22VQForCausalLM.__new__(ConceptLMV22VQForCausalLM)
        torch.nn.Module.__init__(model)
        model.backend_config = SimpleNamespace(chunk_size=4, hidden_size=2, encoder_layers=1)
        model.highlevel = _FakeHighLevel()
        model._draft_capture_layer_ids = (1,)
        model.request_states = ConceptRequestStateStore(
            encoder_layers=1, hlm_layers=1, speculative_chunk_size=4, draft_layers=1
        )
        state = model.request_states._new_state("request-a")
        old_pending = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        append_tensor_buffer(state.pending_encoder_final, old_pending)
        append_tensor_buffer(state.pending_encoder_layers[0], old_pending + 10)
        append_tensor_buffer(state.draft_decoder_layers[0], old_pending + 20)
        state.next_token_position = 3
        segment = ScheduledRequestSegment(
            req_id="request-a",
            flat_start=0,
            flat_end=5,
            position_start=3,
            position_end=8,
            state=state,
        )
        transaction = _DFlashStateTransaction(
            segment=segment,
            snapshot=snapshot_request_state(state),
            encoder_final=torch.tensor(
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]
            ),
            encoder_layers=(
                torch.tensor(
                    [[14.0, 14.0], [15.0, 15.0], [16.0, 16.0], [17.0, 17.0], [18.0, 18.0]]
                ),
            ),
            decoder_layers=(
                torch.tensor(
                    [[24.0, 24.0], [25.0, 25.0], [26.0, 26.0], [27.0, 27.0], [28.0, 28.0]]
                ),
            ),
        )

        # Simulate the state after target verification processed all five
        # anchor/draft inputs and completed two HLM chunks.
        clear_tensor_buffer(state.pending_encoder_final)
        clear_tensor_buffer(state.pending_encoder_layers[0])
        state.hlm_kv[0].length = 2
        append_tensor_buffer(state.hlm_raw_layer_states[0], torch.randn(2, 2))
        append_tensor_buffer(state.predicted_concepts, torch.randn(2, 2))
        append_tensor_buffer(state.draft_decoder_layers[0], transaction.decoder_layers[0])
        state.next_token_position = 8
        model._dflash_transactions = {"request-a": transaction}

        # One accepted draft plus the verifier backup means only the anchor
        # and accepted draft inputs (two rows) belong in target state.
        model.finalize_dflash_transactions(["request-a"], [[101, 102]])

        self.assertEqual(state.next_token_position, 5)
        self.assertEqual(state.hlm_kv[0].length, 1)
        self.assertEqual(state.predicted_concepts.length, 1)
        self.assertEqual(state.pending_encoder_final.length, 1)
        torch.testing.assert_close(
            active_tensor_buffer(state.pending_encoder_final), torch.tensor([[5.0, 5.0]])
        )
        self.assertEqual(state.draft_decoder_layers[0].length, 5)
        torch.testing.assert_close(
            active_tensor_buffer(state.draft_decoder_layers[0])[-2:],
            transaction.decoder_layers[0][:2],
        )
        self.assertFalse(model._dflash_transactions)


if __name__ == "__main__":
    unittest.main()

