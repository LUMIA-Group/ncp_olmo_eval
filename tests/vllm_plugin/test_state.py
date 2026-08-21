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
)


def scheduler_output(
    counts: dict[str, int],
    *,
    finished: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        finished_req_ids=finished,
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


if __name__ == "__main__":
    unittest.main()
