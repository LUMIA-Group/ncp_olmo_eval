"""Request-scoped incremental state for ConceptLM's chunk-rate HLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class RequestStateError(RuntimeError):
    """Raised when vLLM scheduling would make ConceptLM state ambiguous."""


@dataclass
class HLMKVState:
    """Dense K/V history for one HLM layer and one request."""

    key: Any | None = None
    value: Any | None = None
    length: int = 0


@dataclass
class TensorBuffer:
    """Geometrically growing device tensor with an explicit active length."""

    data: Any | None = None
    length: int = 0


def active_tensor_buffer(buffer: TensorBuffer) -> Any | None:
    """Return the active prefix without exposing unused capacity."""

    if buffer.length == 0:
        return None
    if buffer.data is None or buffer.length > int(buffer.data.shape[0]):
        raise ValueError("tensor buffer length exceeds storage capacity")
    return buffer.data[: buffer.length]


def append_tensor_buffer(
    buffer: TensorBuffer,
    values: Any,
    *,
    minimum_capacity: int = 1,
) -> Any:
    """Append one or more leading-dimension values without Python tensor lists."""

    if values.ndim == 0:
        raise ValueError("tensor buffer values require a leading dimension")
    num_values = int(values.shape[0])
    if num_values <= 0:
        raise ValueError("cannot append an empty tensor buffer segment")
    old_length = int(buffer.length)
    required_length = old_length + num_values
    if buffer.data is None:
        if old_length != 0:
            raise ValueError("empty tensor buffer has non-zero length")
        target = max(int(minimum_capacity), required_length)
        capacity = 1 << (target - 1).bit_length()
        buffer.data = values.new_empty((capacity, *values.shape[1:]))
    else:
        if (
            buffer.data.shape[1:] != values.shape[1:]
            or buffer.data.dtype != values.dtype
            or buffer.data.device != values.device
        ):
            raise ValueError("tensor buffer append does not match storage")
        capacity = int(buffer.data.shape[0])
        if old_length < 0 or old_length > capacity:
            raise ValueError("tensor buffer length exceeds storage capacity")
        if required_length > capacity:
            target = max(required_length, capacity * 2)
            new_capacity = 1 << (target - 1).bit_length()
            new_data = values.new_empty((new_capacity, *values.shape[1:]))
            new_data[:old_length].copy_(buffer.data[:old_length])
            buffer.data = new_data

    buffer.data[old_length:required_length].copy_(values)
    buffer.length = required_length
    return buffer.data[:required_length]


def clear_tensor_buffer(buffer: TensorBuffer) -> None:
    """Clear the active prefix while retaining device storage."""

    buffer.length = 0


@dataclass
class ConceptRequestState:
    """All non-PagedAttention state owned by one request."""

    req_id: str
    next_token_position: int = 0
    pending_encoder_final: TensorBuffer = field(default_factory=TensorBuffer)
    pending_encoder_layers: list[TensorBuffer] = field(default_factory=list)
    hlm_kv: list[HLMKVState] = field(default_factory=list)
    hlm_raw_layer_states: list[TensorBuffer] = field(default_factory=list)
    predicted_concepts: TensorBuffer = field(default_factory=TensorBuffer)
    draft_decoder_layers: list[TensorBuffer] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        req_id: str,
        *,
        encoder_layers: int,
        hlm_layers: int,
        draft_layers: int = 0,
    ) -> "ConceptRequestState":
        """Allocate empty per-layer containers without allocating tensors."""

        return cls(
            req_id=req_id,
            pending_encoder_layers=[
                TensorBuffer() for _ in range(encoder_layers)
            ],
            hlm_kv=[HLMKVState() for _ in range(hlm_layers)],
            hlm_raw_layer_states=[
                TensorBuffer() for _ in range(hlm_layers)
            ],
            draft_decoder_layers=[
                TensorBuffer() for _ in range(draft_layers)
            ],
        )


@dataclass(frozen=True)
class RequestStateSnapshot:
    """Small rollback checkpoint taken before one speculative target pass."""

    next_token_position: int
    pending_encoder_final: Any | None
    pending_encoder_layers: tuple[Any | None, ...]
    hlm_kv_lengths: tuple[int, ...]
    hlm_raw_layer_lengths: tuple[int, ...]
    predicted_concepts_length: int
    draft_decoder_lengths: tuple[int, ...]


def snapshot_request_state(state: ConceptRequestState) -> RequestStateSnapshot:
    """Capture mutable lengths plus the at-most-one-chunk pending encoder rows."""

    def clone_active(buffer: TensorBuffer) -> Any | None:
        values = active_tensor_buffer(buffer)
        return None if values is None else values.detach().clone()

    return RequestStateSnapshot(
        next_token_position=int(state.next_token_position),
        pending_encoder_final=clone_active(state.pending_encoder_final),
        pending_encoder_layers=tuple(
            clone_active(buffer) for buffer in state.pending_encoder_layers
        ),
        hlm_kv_lengths=tuple(int(item.length) for item in state.hlm_kv),
        hlm_raw_layer_lengths=tuple(
            int(buffer.length) for buffer in state.hlm_raw_layer_states
        ),
        predicted_concepts_length=int(state.predicted_concepts.length),
        draft_decoder_lengths=tuple(
            int(buffer.length) for buffer in state.draft_decoder_layers
        ),
    )


def restore_request_state(
    state: ConceptRequestState,
    snapshot: RequestStateSnapshot,
) -> None:
    """Restore a speculative checkpoint before replaying its accepted prefix."""

    def restore_pending(buffer: TensorBuffer, values: Any | None) -> None:
        clear_tensor_buffer(buffer)
        if values is not None:
            append_tensor_buffer(buffer, values)

    restore_pending(state.pending_encoder_final, snapshot.pending_encoder_final)
    if len(state.pending_encoder_layers) != len(snapshot.pending_encoder_layers):
        raise RequestStateError("encoder layer count changed during speculation")
    for buffer, values in zip(
        state.pending_encoder_layers,
        snapshot.pending_encoder_layers,
        strict=True,
    ):
        restore_pending(buffer, values)
    if len(state.hlm_kv) != len(snapshot.hlm_kv_lengths):
        raise RequestStateError("HLM K/V layer count changed during speculation")
    for kv_state, length in zip(state.hlm_kv, snapshot.hlm_kv_lengths, strict=True):
        if kv_state.length < length:
            raise RequestStateError("HLM K/V history is shorter than its checkpoint")
        kv_state.length = length
    if len(state.hlm_raw_layer_states) != len(snapshot.hlm_raw_layer_lengths):
        raise RequestStateError("HLM raw layer count changed during speculation")
    for buffer, length in zip(
        state.hlm_raw_layer_states,
        snapshot.hlm_raw_layer_lengths,
        strict=True,
    ):
        if buffer.length < length:
            raise RequestStateError("HLM raw history is shorter than its checkpoint")
        buffer.length = length
    if state.predicted_concepts.length < snapshot.predicted_concepts_length:
        raise RequestStateError("predicted concepts are shorter than their checkpoint")
    state.predicted_concepts.length = snapshot.predicted_concepts_length
    if len(state.draft_decoder_layers) != len(snapshot.draft_decoder_lengths):
        raise RequestStateError("draft decoder layer count changed during speculation")
    for buffer, length in zip(
        state.draft_decoder_layers,
        snapshot.draft_decoder_lengths,
        strict=True,
    ):
        if buffer.length < length:
            raise RequestStateError("draft decoder history is shorter than its checkpoint")
        buffer.length = length
    state.next_token_position = snapshot.next_token_position


@dataclass(frozen=True)
class ScheduledRequestSegment:
    """One contiguous flattened token segment scheduled for a request."""

    req_id: str
    flat_start: int
    flat_end: int
    position_start: int
    position_end: int
    state: ConceptRequestState


class ConceptRequestStateStore:
    """Bind vLLM scheduler output to persistent ConceptLM request state.

    The worker calls :meth:`bind_scheduler_output` before vLLM executes the
    model runner. The model resolves segments during ``forward``; by then the
    runner has admitted, removed, condensed, and potentially reordered its
    persistent batch.
    """

    def __init__(
        self,
        *,
        encoder_layers: int,
        hlm_layers: int,
        speculative_chunk_size: int | None = None,
        draft_layers: int = 0,
    ) -> None:
        self.encoder_layers = int(encoder_layers)
        self.hlm_layers = int(hlm_layers)
        self.speculative_chunk_size = (
            int(speculative_chunk_size) if speculative_chunk_size is not None else None
        )
        self.draft_layers = int(draft_layers)
        if self.speculative_chunk_size is not None and self.speculative_chunk_size <= 1:
            raise ValueError("speculative_chunk_size must be greater than one")
        if self.draft_layers < 0:
            raise ValueError("draft_layers must be non-negative")
        self._states: dict[str, ConceptRequestState] = {}
        self._scheduler_output: Any | None = None
        self._model_runner: Any | None = None
        self._input_batch: Any | None = None

    @property
    def states(self) -> Mapping[str, ConceptRequestState]:
        """Expose a read-only mapping view for diagnostics."""

        return self._states

    def bind_scheduler_output(self, scheduler_output: Any, model_runner: Any) -> None:
        """Bind one scheduler step and eagerly release finished requests."""

        for req_id in scheduler_output.finished_req_ids:
            self._states.pop(str(req_id), None)
        self._scheduler_output = scheduler_output
        self._model_runner = model_runner
        self._input_batch = None

    def bind_input_batch(self, input_batch: Any) -> None:
        """Bind the concrete vLLM input batch prepared for this forward.

        vLLM 0.13 keeps the active batch on ``model_runner.input_batch``.
        vLLM 0.25's V2 model runner instead builds a short-lived ``InputBatch``
        inside ``execute_model``.  The worker hook calls this method from the
        latter's ``prepare_inputs`` boundary so request ordering and computed
        positions still come from vLLM rather than being reconstructed.
        """

        self._input_batch = input_batch

    def _bound_input_batch(self) -> Any:
        if self._input_batch is not None:
            return self._input_batch
        if self._model_runner is None:
            raise RequestStateError("the scheduler output is not bound")
        input_batch = getattr(self._model_runner, "input_batch", None)
        if input_batch is None:
            raise RequestStateError(
                "vLLM did not expose or bind the concrete input batch"
            )
        return input_batch

    def bound_request_ids(self) -> tuple[str, ...]:
        """Return the model-runner batch order used by sampled-token outputs."""

        input_batch = self._bound_input_batch()
        return tuple(str(req_id) for req_id in input_batch.req_ids)

    def speculative_draft_token_count(self, req_id: str) -> int | None:
        """Return this step's verifier draft length for one request, if any."""

        if self._scheduler_output is None:
            raise RequestStateError("the scheduler output is not bound")
        scheduled = getattr(
            self._scheduler_output,
            "scheduled_spec_decode_tokens",
            {},
        )
        tokens = scheduled.get(req_id)
        return None if not tokens else len(tokens)

    def _new_state(self, req_id: str) -> ConceptRequestState:
        state = ConceptRequestState.empty(
            req_id,
            encoder_layers=self.encoder_layers,
            hlm_layers=self.hlm_layers,
            draft_layers=self.draft_layers,
        )
        self._states[req_id] = state
        return state

    def resolve_segments(self) -> tuple[ScheduledRequestSegment, ...]:
        """Resolve segments from vLLM's CPU scheduler metadata after reorder."""

        if self._scheduler_output is None or self._model_runner is None:
            raise RequestStateError(
                "ConceptLM worker did not bind scheduler output before model forward"
            )
        input_batch = self._bound_input_batch()
        req_ids = tuple(str(req_id) for req_id in input_batch.req_ids)
        scheduled_counts = self._scheduler_output.num_scheduled_tokens
        computed_positions = getattr(input_batch, "num_computed_tokens_cpu", None)
        if computed_positions is None:
            computed_positions = getattr(input_batch, "num_computed_tokens_np", None)
        if computed_positions is None:
            raise RequestStateError(
                "vLLM input batch is missing computed-position metadata"
            )
        expected_tokens = sum(int(scheduled_counts[req_id]) for req_id in req_ids)
        total_scheduled_tokens = int(
            self._scheduler_output.total_num_scheduled_tokens
        )
        if expected_tokens != total_scheduled_tokens:
            raise RequestStateError(
                "per-request scheduled token count does not match total: "
                f"{expected_tokens} != {total_scheduled_tokens}"
            )
        if len(computed_positions) < len(req_ids):
            raise RequestStateError(
                "vLLM computed-position metadata is shorter than the request batch"
            )

        segments = []
        flat_start = 0
        for req_index, req_id in enumerate(req_ids):
            token_count = int(scheduled_counts[req_id])
            if token_count <= 0:
                raise RequestStateError(
                    f"request {req_id!r} has an empty segment"
                )
            flat_end = flat_start + token_count
            position_start = int(computed_positions[req_index])
            if position_start < 0:
                raise RequestStateError(
                    f"request {req_id!r} has negative computed position "
                    f"{position_start}"
                )

            state = self._states.get(req_id)
            if state is None:
                state = self._new_state(req_id)
            if position_start < state.next_token_position:
                if position_start != 0:
                    if self.speculative_chunk_size is None:
                        raise RequestStateError(
                            f"request {req_id!r} rewound from "
                            f"{state.next_token_position} to {position_start}; only a "
                            "full replay from position 0 is supported"
                        )
                    self._rollback_speculative_suffix(state, position_start)
                else:
                    state = self._new_state(req_id)
            if position_start > state.next_token_position:
                raise RequestStateError(
                    f"request {req_id!r} starts at {position_start}, but its "
                    f"ConceptLM state ends at {state.next_token_position}; prefix "
                    "caching and partial replay must be disabled"
                )
            segments.append(
                ScheduledRequestSegment(
                    req_id=req_id,
                    flat_start=flat_start,
                    flat_end=flat_end,
                    position_start=position_start,
                    position_end=position_start + token_count,
                    state=state,
                )
            )
            flat_start = flat_end
        return tuple(segments)

    def _rollback_speculative_suffix(
        self,
        state: ConceptRequestState,
        position: int,
    ) -> None:
        """Discard rejected draft tokens without crossing an HLM chunk boundary.

        The initial integration deliberately limits proposals so the verifier's
        speculative segment cannot complete a chunk. Therefore rejection only
        shortens the current pending encoder chunk and decoder-feature histories;
        no HLM K/V or predicted-concept value needs to be reconstructed.
        """

        chunk_size = self.speculative_chunk_size
        if chunk_size is None:
            raise RequestStateError("speculative rollback is not enabled")
        old_position = int(state.next_token_position)
        if not 0 < position < old_position:
            raise RequestStateError(
                f"invalid speculative rollback position: {position} from {old_position}"
            )
        if position // chunk_size != old_position // chunk_size:
            raise RequestStateError(
                "NCP DFlash rollback crossed an HLM chunk boundary; this is "
                f"outside the correctness-gated proposal window: {old_position} -> {position}"
            )
        pending_length = position % chunk_size
        completed_chunks = position // chunk_size
        pending_buffers = [state.pending_encoder_final, *state.pending_encoder_layers]
        for buffer in pending_buffers:
            if buffer.length < pending_length:
                raise RequestStateError("pending encoder state is shorter than rollback target")
            buffer.length = pending_length
        for kv_state in state.hlm_kv:
            if kv_state.length != completed_chunks:
                raise RequestStateError("HLM K/V length changed inside a draft-only suffix")
        for buffer in [*state.hlm_raw_layer_states, state.predicted_concepts]:
            if buffer.length != completed_chunks:
                raise RequestStateError("HLM state length changed inside a draft-only suffix")
        for buffer in state.draft_decoder_layers:
            if buffer.length < position:
                raise RequestStateError("draft decoder history is shorter than rollback target")
            buffer.length = position
        state.next_token_position = position
        from .ncp_dflash_state import append_telemetry

        append_telemetry(
            "target_state_rollback",
            request_id=state.req_id,
            old_position=old_position,
            new_position=position,
            rejected_tokens=old_position - position,
        )

    @staticmethod
    def commit(segment: ScheduledRequestSegment) -> None:
        """Commit one successfully processed request segment."""

        if segment.state.next_token_position != segment.position_start:
            raise RequestStateError(
                f"request {segment.req_id!r} state changed before commit"
            )
        segment.state.next_token_position = segment.position_end
