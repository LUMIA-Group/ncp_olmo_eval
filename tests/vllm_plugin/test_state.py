"""Unit tests for request-scoped ConceptLM scheduler state."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from ncp_olmo_eval.vllm_plugin.state import (
    ConceptRequestStateStore,
    RequestStateError,
    TensorBuffer,
    active_tensor_buffer,
    append_tensor_buffer,
    clear_tensor_buffer,
    restore_request_state,
    snapshot_request_state,
)


def scheduler_output(
    counts: dict[str, int],
    *,
    finished: tuple[str, ...] = (),
    speculative: dict[str, list[int]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        finished_req_ids=finished,
        scheduled_spec_decode_tokens=speculative or {},
    )


def model_runner(
    req_ids: list[str],
    computed_positions: list[int] | None = None,
) -> SimpleNamespace:
    if computed_positions is None:
        computed_positions = [0] * len(req_ids)
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=req_ids,
            num_computed_tokens_cpu=computed_positions,
        )
    )


class TestConceptRequestStateStore(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_speculative_snapshot_restores_mutated_cross_chunk_state(self) -> None:
        store = ConceptRequestStateStore(
            encoder_layers=1,
            hlm_layers=1,
            speculative_chunk_size=4,
            draft_layers=1,
        )
        state = store._new_state("request-a")
        pending = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        append_tensor_buffer(state.pending_encoder_final, pending)
        append_tensor_buffer(state.pending_encoder_layers[0], pending + 10)
        state.hlm_kv[0].length = 1
        append_tensor_buffer(state.hlm_raw_layer_states[0], torch.randn(1, 4))
        append_tensor_buffer(state.predicted_concepts, torch.randn(1, 4))
        append_tensor_buffer(state.draft_decoder_layers[0], torch.randn(6, 4))
        state.next_token_position = 6
        snapshot = snapshot_request_state(state)

        clear_tensor_buffer(state.pending_encoder_final)
        append_tensor_buffer(state.pending_encoder_final, torch.full((1, 4), -1.0))
        clear_tensor_buffer(state.pending_encoder_layers[0])
        append_tensor_buffer(state.pending_encoder_layers[0], torch.full((1, 4), -2.0))
        state.hlm_kv[0].length = 3
        append_tensor_buffer(state.hlm_raw_layer_states[0], torch.randn(2, 4))
        append_tensor_buffer(state.predicted_concepts, torch.randn(2, 4))
        append_tensor_buffer(state.draft_decoder_layers[0], torch.randn(8, 4))
        state.next_token_position = 14

        restore_request_state(state, snapshot)

        self.assertEqual(state.next_token_position, 6)
        self.assertEqual(state.hlm_kv[0].length, 1)
        self.assertEqual(state.hlm_raw_layer_states[0].length, 1)
        self.assertEqual(state.predicted_concepts.length, 1)
        self.assertEqual(state.draft_decoder_layers[0].length, 6)
        torch.testing.assert_close(active_tensor_buffer(state.pending_encoder_final), pending)
        torch.testing.assert_close(
            active_tensor_buffer(state.pending_encoder_layers[0]), pending + 10
        )

    def test_scheduler_exposes_speculative_draft_count(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=1, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output(
                {"request-a": 4, "request-b": 1},
                speculative={"request-a": [11, 12, 13]},
            ),
            model_runner(["request-a", "request-b"]),
        )

        self.assertEqual(store.speculative_draft_token_count("request-a"), 3)
        self.assertIsNone(store.speculative_draft_token_count("request-b"))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_tensor_buffer_reuses_and_grows_device_storage(self) -> None:
        buffer = TensorBuffer()
        first = torch.randn(3, 4)
        active = append_tensor_buffer(
            buffer,
            first,
            minimum_capacity=4,
        )
        first_pointer = buffer.data.data_ptr()

        self.assertEqual(buffer.length, 3)
        self.assertEqual(buffer.data.shape, (4, 4))
        torch.testing.assert_close(active, first)

        second = torch.randn(2, 4)
        active = append_tensor_buffer(buffer, second)

        self.assertEqual(buffer.length, 5)
        self.assertEqual(buffer.data.shape, (8, 4))
        torch.testing.assert_close(
            active,
            torch.cat((first, second), dim=0),
        )
        grown_pointer = buffer.data.data_ptr()

        clear_tensor_buffer(buffer)
        self.assertIsNone(active_tensor_buffer(buffer))
        active = append_tensor_buffer(buffer, second)

        self.assertEqual(buffer.data.data_ptr(), grown_pointer)
        self.assertNotEqual(buffer.data.data_ptr(), first_pointer)
        torch.testing.assert_close(active, second)

    def test_real_request_order_drives_flat_segments(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 2, "request-b": 1}),
            model_runner(["request-b", "request-a"]),
        )

        segments = store.resolve_segments()

        self.assertEqual(
            [(item.req_id, item.flat_start, item.flat_end) for item in segments],
            [("request-b", 0, 1), ("request-a", 1, 3)],
        )
        for segment in segments:
            store.commit(segment)
        self.assertEqual(store.states["request-a"].next_token_position, 2)
        self.assertEqual(store.states["request-b"].next_token_position, 1)

    def test_finished_request_state_is_released_without_forward(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"]),
        )
        segment = store.resolve_segments()[0]
        store.commit(segment)

        store.bind_scheduler_output(
            scheduler_output({}, finished=("request-a",)),
            model_runner([]),
        )

        self.assertNotIn("request-a", store.states)

    def test_computed_position_metadata_drives_continuation(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 2}),
            model_runner(["request-a"], [0]),
        )
        first = store.resolve_segments()[0]
        store.commit(first)

        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"], [2]),
        )
        continuation = store.resolve_segments()[0]

        self.assertEqual(continuation.position_start, 2)
        self.assertEqual(continuation.position_end, 3)

    def test_v2_bound_input_batch_drives_segments(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        runner_without_persistent_batch = SimpleNamespace()
        store.bind_scheduler_output(
            scheduler_output({"request-a": 2, "request-b": 1}),
            runner_without_persistent_batch,
        )
        store.bind_input_batch(
            SimpleNamespace(
                req_ids=["request-b", "request-a"],
                num_computed_tokens_np=[0, 0],
            )
        )

        segments = store.resolve_segments()

        self.assertEqual(
            [
                (
                    item.req_id,
                    item.flat_start,
                    item.flat_end,
                    item.position_start,
                )
                for item in segments
            ],
            [
                ("request-b", 0, 1, 0),
                ("request-a", 1, 3, 0),
            ],
        )

    def test_unscheduled_request_state_survives(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"]),
        )
        segment = store.resolve_segments()[0]
        store.commit(segment)

        store.bind_scheduler_output(
            scheduler_output({"request-b": 1}),
            model_runner(["request-b"]),
        )
        store.resolve_segments()

        self.assertEqual(store.states["request-a"].next_token_position, 1)

    def test_prefix_cache_gap_fails_closed(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"], [8]),
        )

        with self.assertRaisesRegex(RequestStateError, "prefix caching"):
            store.resolve_segments()

    def test_full_replay_resets_state(self) -> None:
        store = ConceptRequestStateStore(encoder_layers=2, hlm_layers=1)
        store.bind_scheduler_output(
            scheduler_output({"request-a": 2}),
            model_runner(["request-a"]),
        )
        first = store.resolve_segments()[0]
        store.commit(first)
        old_state = first.state

        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"]),
        )
        replay = store.resolve_segments()[0]

        self.assertIsNot(replay.state, old_state)
        self.assertEqual(replay.state.next_token_position, 0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_speculative_rejection_truncates_only_current_chunk(self) -> None:
        store = ConceptRequestStateStore(
            encoder_layers=2,
            hlm_layers=1,
            speculative_chunk_size=4,
            draft_layers=5,
        )
        store.bind_scheduler_output(
            scheduler_output({"request-a": 7}),
            model_runner(["request-a"], [0]),
        )
        first = store.resolve_segments()[0]
        state = first.state
        for buffer in [state.pending_encoder_final, *state.pending_encoder_layers]:
            append_tensor_buffer(buffer, torch.randn(3, 4))
        state.hlm_kv[0].length = 1
        state.hlm_raw_layer_states[0].length = 1
        state.predicted_concepts.length = 1
        for buffer in state.draft_decoder_layers:
            append_tensor_buffer(buffer, torch.randn(7, 4))
        store.commit(first)

        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"], [5]),
        )
        continuation = store.resolve_segments()[0]

        self.assertEqual(continuation.position_start, 5)
        self.assertEqual(state.next_token_position, 5)
        self.assertEqual(state.pending_encoder_final.length, 1)
        self.assertTrue(all(buffer.length == 1 for buffer in state.pending_encoder_layers))
        self.assertTrue(all(buffer.length == 5 for buffer in state.draft_decoder_layers))
        self.assertEqual(state.predicted_concepts.length, 1)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_speculative_rejection_across_chunk_boundary_fails_closed(self) -> None:
        store = ConceptRequestStateStore(
            encoder_layers=1,
            hlm_layers=1,
            speculative_chunk_size=4,
            draft_layers=1,
        )
        store.bind_scheduler_output(
            scheduler_output({"request-a": 8}),
            model_runner(["request-a"], [0]),
        )
        first = store.resolve_segments()[0]
        state = first.state
        state.hlm_kv[0].length = 2
        state.hlm_raw_layer_states[0].length = 2
        state.predicted_concepts.length = 2
        append_tensor_buffer(state.draft_decoder_layers[0], torch.randn(8, 4))
        store.commit(first)

        store.bind_scheduler_output(
            scheduler_output({"request-a": 1}),
            model_runner(["request-a"], [5]),
        )
        with self.assertRaisesRegex(RequestStateError, "crossed an HLM chunk"):
            store.resolve_segments()


if __name__ == "__main__":
    unittest.main()
