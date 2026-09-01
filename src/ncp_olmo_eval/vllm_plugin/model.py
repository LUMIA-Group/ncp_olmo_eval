"""vLLM model entry point for ConceptLM V2.2-VQ."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .contract import ConceptLMBackendConfig
from .hlm import ConceptLMHighLevelBranch
from .qkv import deinterleave_megatron_qkv
from .routes import ConceptLMStage3Routes
from .state import (
    ConceptRequestState,
    ConceptRequestStateStore,
    RequestStateSnapshot,
    ScheduledRequestSegment,
    active_tensor_buffer,
    append_tensor_buffer,
    clear_tensor_buffer,
    restore_request_state,
    snapshot_request_state,
)
from .token_tower import ConceptLMTokenBackbone, _config_value
from .weights import (
    Stage3WeightConfig,
    expected_stage3_weight_shapes,
    required_checkpoint_shards,
    resolve_checkpoint_weight,
)

_HLMAdvance = tuple[ConceptRequestState, torch.Tensor, tuple[torch.Tensor, ...]]


@dataclass
class _DFlashStateTransaction:
    """Target-state mutation staged until vLLM reports accepted draft tokens."""

    segment: ScheduledRequestSegment
    snapshot: RequestStateSnapshot | None
    encoder_final: torch.Tensor | None = None
    encoder_layers: tuple[torch.Tensor, ...] = ()
    decoder_layers: tuple[torch.Tensor, ...] = ()


def _enabled_for_parity() -> bool:
    return os.environ.get("CONCEPTLM_VLLM_ENABLE_UNVERIFIED", "0") == "1"


def _ncp_dflash_enabled(vllm_config: Any) -> bool:
    """Recognize only the explicitly gated vLLM 0.13 proposer integration."""

    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return False
    return (
        os.environ.get("CONCEPTLM_VLLM_ENABLE_NCP_DFLASH", "0") == "1"
        and getattr(speculative_config, "method", None) == "ngram"
        and getattr(speculative_config, "model", None) == "ngram"
    )


def _append_graph_padding_rows(
    rows: list[torch.Tensor], *, target_length: int, zero: torch.Tensor
) -> None:
    """Pad request-owned rows to vLLM's PIECEWISE capture batch size."""

    if len(rows) > target_length:
        raise RuntimeError(
            "request-owned ConceptLM rows exceed the model input batch: "
            f"{len(rows)} > {target_length}"
        )
    rows.extend([zero] * (target_length - len(rows)))


class ConceptLMV22VQForCausalLM(nn.Module):
    """Native Stage3 ConceptLM model, gated until all parity checks pass."""

    backend_status = "full_forward_parity_gated"

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        if prefix:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        if not _enabled_for_parity():
            raise RuntimeError(
                "ConceptLM native vLLM construction is parity-gated. Set "
                "CONCEPTLM_VLLM_ENABLE_UNVERIFIED=1 only in the controlled "
                "batch-1 parity harness."
            )
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("the first ConceptLM backend requires TP=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        if (
            not vllm_config.model_config.enforce_eager
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.PIECEWISE
        ):
            raise ValueError(
                "ConceptLM non-eager runs require PIECEWISE CUDA graphs; "
                "full-model capture cannot safely own request-scoped HLM state"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("ConceptLM request-scoped HLM requires prefix caching to be disabled")
        self._ncp_dflash_enabled = _ncp_dflash_enabled(vllm_config)
        if vllm_config.speculative_config is not None and not self._ncp_dflash_enabled:
            raise NotImplementedError(
                "speculative decoding is not enabled before target-model parity"
            )

        if self._ncp_dflash_enabled:
            target_layers = tuple(
                int(value)
                for value in os.environ.get("CONCEPTLM_DFLASH_TARGET_LAYERS", "1,4,7,10,13").split(
                    ","
                )
            )
            if target_layers != (1, 4, 7, 10, 13):
                raise ValueError(
                    "the first NCP DFlash integration requires target layers " "1,4,7,10,13"
                )
            draft_chunk_size = int(os.environ.get("CONCEPTLM_DFLASH_CHUNK_SIZE", "4"))
        else:
            target_layers = ()
            draft_chunk_size = None
        self._draft_capture_layer_ids = target_layers

        hf_config = vllm_config.model_config.hf_config
        raw_config = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(vars(hf_config))
        self.backend_config = ConceptLMBackendConfig.from_mapping(raw_config)
        self.stage3_weight_config = Stage3WeightConfig.from_mapping(raw_config)
        if self.backend_config.chunk_merge_method != "meanpooling":
            raise NotImplementedError("the first Stage3 backend requires meanpooling chunks")
        epsilon = float(_config_value(hf_config, "layernorm_epsilon"))
        self.token_backbone = ConceptLMTokenBackbone(
            vllm_config=vllm_config, backend_config=self.backend_config
        )
        self.highlevel = ConceptLMHighLevelBranch(
            vllm_config=vllm_config, backend_config=self.backend_config
        )
        self.routes = ConceptLMStage3Routes(backend_config=self.backend_config, epsilon=epsilon)
        self.request_states = ConceptRequestStateStore(
            encoder_layers=self.backend_config.encoder_layers,
            hlm_layers=self.backend_config.hlm_layers,
            speculative_chunk_size=draft_chunk_size,
            draft_layers=len(target_layers),
        )
        self._profile_mode = False
        self._dflash_transactions: dict[str, _DFlashStateTransaction] = {}
        self._logits_trace_index = 0
        self._pending_stage_trace: dict[str, torch.Tensor] | None = None
        trace_dir = os.environ.get("CONCEPTLM_VLLM_LOGITS_TRACE_DIR")
        self._logits_trace_dir = Path(trace_dir) if trace_dir else None
        if self._ncp_dflash_enabled:
            from .ncp_dflash_state import register_target_model

            register_target_model(self)

    def bind_scheduler_output(self, scheduler_output: Any, model_runner: Any) -> None:
        """Bind the current vLLM scheduler step before runner preprocessing."""

        if self._dflash_transactions:
            raise RuntimeError(
                "NCP DFlash target transactions were not finalized before the next step: "
                f"{sorted(self._dflash_transactions)}"
            )
        self.request_states.bind_scheduler_output(scheduler_output, model_runner)

    def _begin_dflash_transaction(self, segment: ScheduledRequestSegment) -> None:
        draft_count = self.request_states.speculative_draft_token_count(segment.req_id)
        if draft_count is None:
            return
        segment_length = segment.position_end - segment.position_start
        if segment_length != draft_count + 1:
            raise RuntimeError(
                "NCP DFlash verifier segment length does not match scheduled drafts: "
                f"request={segment.req_id} segment={segment_length} drafts={draft_count}"
            )
        if segment.req_id in self._dflash_transactions:
            raise RuntimeError(f"duplicate NCP DFlash transaction for {segment.req_id!r}")
        chunk_size = int(self.backend_config.chunk_size)
        crosses_chunk_boundary = (
            segment.position_start // chunk_size != segment.position_end // chunk_size
        )
        self._dflash_transactions[segment.req_id] = _DFlashStateTransaction(
            segment=segment,
            # A verifier segment contained within one HLM chunk can be rolled
            # back by truncating request-local buffer lengths.  Only segments
            # that may advance HLM state need the expensive encoder-row copy
            # required for cross-chunk replay.
            snapshot=(snapshot_request_state(segment.state) if crosses_chunk_boundary else None),
        )

    def _record_dflash_transaction_features(
        self,
        segment: ScheduledRequestSegment,
        encoder_hidden: torch.Tensor,
        encoder_raw_layers: Sequence[torch.Tensor],
        draft_layer_outputs: dict[int, torch.Tensor],
    ) -> None:
        transaction = self._dflash_transactions.get(segment.req_id)
        if transaction is None or transaction.snapshot is None:
            return
        token_slice = slice(segment.flat_start, segment.flat_end)
        # Finalization runs after sampling and before the next model forward.
        # These forward outputs are not mutated in that interval, so retaining
        # detached views is sufficient for rejection replay.  Cloning every
        # captured layer launched dozens of small device copies per verifier
        # step and discarded all of them on a fully accepted proposal.
        transaction.encoder_final = encoder_hidden[token_slice].detach()
        transaction.encoder_layers = tuple(
            values[token_slice].detach() for values in encoder_raw_layers
        )
        transaction.decoder_layers = tuple(
            draft_layer_outputs[layer_id][token_slice].detach()
            for layer_id in self._draft_capture_layer_ids
        )

    def finalize_dflash_transactions(
        self, request_ids: Sequence[str], sampled_token_ids: Sequence[Sequence[int]]
    ) -> None:
        """Commit only the verifier-accepted target prefix before new proposals."""

        if not self._dflash_transactions:
            return
        if len(request_ids) != len(sampled_token_ids):
            raise RuntimeError("NCP DFlash sampled rows do not match the target request batch")
        sampled_by_request = {
            str(req_id): tuple(int(token) for token in tokens)
            for req_id, tokens in zip(request_ids, sampled_token_ids, strict=True)
        }
        replay: list[tuple[_DFlashStateTransaction, ScheduledRequestSegment, int]] = []
        hlm_advances: list[_HLMAdvance] = []
        for req_id, transaction in tuple(self._dflash_transactions.items()):
            tokens = sampled_by_request.get(req_id)
            if tokens is None:
                raise RuntimeError(f"NCP DFlash verifier omitted request {req_id!r}")
            if not tokens:
                # vLLM clears the sampled row when a request reaches its stop
                # condition.  No proposal or later target step can consume the
                # request-local state, so simply release the transaction.
                from .ncp_dflash_state import append_telemetry

                append_telemetry(
                    "target_state_transaction_discard",
                    request_id=req_id,
                    position_start=transaction.segment.position_start,
                    position_end=transaction.segment.position_end,
                    verifier_tokens=(
                        transaction.segment.position_end - transaction.segment.position_start
                    ),
                    reason="request_finished",
                )
                del self._dflash_transactions[req_id]
                continue
            full_length = transaction.segment.position_end - transaction.segment.position_start
            accepted_input_count = len(tokens)
            if accepted_input_count > full_length:
                raise RuntimeError(
                    "NCP DFlash accepted prefix exceeds its verifier segment: "
                    f"request={req_id} accepted={accepted_input_count} full={full_length}"
                )
            if accepted_input_count == full_length:
                if (
                    transaction.segment.state.next_token_position
                    != transaction.segment.position_end
                ):
                    raise RuntimeError(
                        f"fully accepted DFlash state is not committed for {req_id!r}"
                    )
                from .ncp_dflash_state import append_telemetry

                append_telemetry(
                    "target_state_transaction_commit",
                    request_id=req_id,
                    position_start=transaction.segment.position_start,
                    position_end=transaction.segment.position_end,
                    committed_position=transaction.segment.position_end,
                    verifier_tokens=full_length,
                    committed_tokens=full_length,
                    rolled_back_tokens=0,
                )
                del self._dflash_transactions[req_id]
                continue
            if transaction.snapshot is None:
                accepted_position = transaction.segment.position_start + accepted_input_count
                self.request_states._rollback_speculative_suffix(
                    transaction.segment.state, accepted_position
                )
                from .ncp_dflash_state import append_telemetry

                append_telemetry(
                    "target_state_transaction_commit",
                    request_id=req_id,
                    position_start=transaction.segment.position_start,
                    position_end=transaction.segment.position_end,
                    committed_position=accepted_position,
                    verifier_tokens=full_length,
                    committed_tokens=accepted_input_count,
                    rolled_back_tokens=full_length - accepted_input_count,
                    rollback_mode="truncate_same_chunk",
                )
                del self._dflash_transactions[req_id]
                continue
            if (
                transaction.encoder_final is None
                or not transaction.encoder_layers
                or len(transaction.decoder_layers) != len(self._draft_capture_layer_ids)
            ):
                raise RuntimeError(f"NCP DFlash transaction features are incomplete for {req_id!r}")
            if transaction.snapshot is None:
                raise RuntimeError(f"NCP DFlash cross-chunk snapshot is missing for {req_id!r}")
            restore_request_state(transaction.segment.state, transaction.snapshot)
            accepted_segment = ScheduledRequestSegment(
                req_id=req_id,
                flat_start=0,
                flat_end=accepted_input_count,
                position_start=transaction.segment.position_start,
                position_end=(transaction.segment.position_start + accepted_input_count),
                state=transaction.segment.state,
            )
            advance = self._advance_completed_chunks(
                accepted_segment,
                transaction.encoder_final,
                transaction.encoder_layers,
                incremental_exact=True,
            )
            if advance is not None:
                hlm_advances.append(advance)
            replay.append((transaction, accepted_segment, accepted_input_count))

        self._advance_hlm_batches(hlm_advances)
        for transaction, accepted_segment, accepted_input_count in replay:
            for buffer, values in zip(
                accepted_segment.state.draft_decoder_layers, transaction.decoder_layers, strict=True
            ):
                append_tensor_buffer(
                    buffer,
                    values[:accepted_input_count],
                    minimum_capacity=max(16, accepted_segment.position_end),
                )
            self.request_states.commit(accepted_segment)
            rolled_back = transaction.segment.position_end - accepted_segment.position_end
            from .ncp_dflash_state import append_telemetry

            append_telemetry(
                "target_state_transaction_commit",
                request_id=accepted_segment.req_id,
                position_start=transaction.segment.position_start,
                position_end=transaction.segment.position_end,
                committed_position=accepted_segment.position_end,
                verifier_tokens=(
                    transaction.segment.position_end - transaction.segment.position_start
                ),
                committed_tokens=accepted_input_count,
                rolled_back_tokens=rolled_back,
            )
            del self._dflash_transactions[accepted_segment.req_id]

    def set_profile_mode(self, enabled: bool) -> None:
        """Allow vLLM's allocation-only dummy run without scheduler requests."""

        self._profile_mode = bool(enabled)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings."""

        return self.token_backbone.embed_input_ids(input_ids)

    def ncp_dflash_proposal_context(
        self, request_id: str, prefix_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the trained DFlash inputs for one newly sampled anchor token.

        vLLM proposes after sampling the next target token, so the target has
        processed ``prefix_length - 1`` tokens. The drafter excludes the anchor
        row from attention; an explicit zero row preserves its sequence layout.
        """

        if not self._ncp_dflash_enabled:
            raise RuntimeError("NCP DFlash proposal context is not enabled")
        request_id = str(request_id)
        bound_request_ids = self.request_states.bound_request_ids()
        if request_id not in bound_request_ids:
            raise KeyError(
                "DFlash request is not present in the bound input batch: "
                f"{request_id!r} not in {bound_request_ids!r}"
            )
        state = self.request_states.states.get(request_id)
        if state is None:
            raise KeyError(f"DFlash request state is missing for {request_id!r}")
        context_length = int(prefix_length) - 1
        if context_length < 0 or context_length > state.next_token_position:
            raise RuntimeError(
                "DFlash prefix is inconsistent with target state: "
                f"prefix={prefix_length} target={state.next_token_position}"
            )
        layer_values = []
        for layer_id, buffer in zip(
            self._draft_capture_layer_ids, state.draft_decoder_layers, strict=True
        ):
            values = active_tensor_buffer(buffer)
            if values is None or int(values.shape[0]) < context_length:
                raise RuntimeError(
                    f"missing DFlash decoder history for layer {layer_id}: "
                    f"need={context_length} have={buffer.length}"
                )
            layer_values.append(values[:context_length])
        embedding_weight = self.token_backbone.embedding.word_embeddings.weight[
            : self.backend_config.vocab_size
        ]
        output_weight = self.token_backbone.output_layer.weight[: self.backend_config.vocab_size]
        if context_length:
            context = torch.stack(layer_values, dim=1).unsqueeze(0)
            anchor_row = context.new_zeros((1, 1, len(layer_values), context.shape[-1]))
            context = torch.cat((context, anchor_row), dim=1)
        else:
            context = embedding_weight.new_zeros(
                (1, 1, len(layer_values), self.backend_config.hidden_size)
            )
        completed_chunks = context_length // self.backend_config.chunk_size
        if completed_chunks == 0:
            hlm_state = embedding_weight.new_zeros((1, 1, self.backend_config.hidden_size))
        else:
            source_index = completed_chunks - 1
            concepts = active_tensor_buffer(state.predicted_concepts)
            if concepts is None or source_index >= int(concepts.shape[0]):
                raise RuntimeError(
                    "DFlash is missing its causal HLM source: "
                    f"source={source_index} concepts={state.predicted_concepts.length}"
                )
            hlm_state = concepts[source_index].view(1, 1, -1)
        return context, hlm_state, embedding_weight, output_weight

    def _advance_completed_chunks(
        self,
        segment: ScheduledRequestSegment,
        encoder_hidden: torch.Tensor,
        encoder_raw_layers: Sequence[torch.Tensor],
        *,
        incremental_exact: bool = False,
    ) -> _HLMAdvance | None:
        state = segment.state
        chunk_size = self.backend_config.chunk_size
        advance: _HLMAdvance | None = None
        new_final = encoder_hidden[segment.flat_start : segment.flat_end]
        new_layers = tuple(
            raw_layer[segment.flat_start : segment.flat_end] for raw_layer in encoder_raw_layers
        )
        if state.pending_encoder_final.length:
            pending_final = active_tensor_buffer(state.pending_encoder_final)
            if pending_final is None:
                raise RuntimeError("missing pending encoder final tensor")
            combined_final = torch.cat((pending_final, new_final), dim=0)
            combined_layers = tuple(
                torch.cat((active_tensor_buffer(pending), new_layer), dim=0)
                for pending, new_layer in zip(state.pending_encoder_layers, new_layers)
            )
        else:
            combined_final = new_final
            combined_layers = new_layers

        num_completed = int(combined_final.shape[0]) // chunk_size
        completed_tokens = num_completed * chunk_size
        if num_completed:
            encoder_chunks = (
                combined_final[:completed_tokens]
                .reshape(num_completed, chunk_size, self.backend_config.hidden_size)
                .mean(dim=1)
            )
            layer_chunks = tuple(
                values[:completed_tokens]
                .reshape(num_completed, chunk_size, self.backend_config.hidden_size)
                .mean(dim=1)
                for values in combined_layers
            )
            if num_completed == 1:
                advance = (state, encoder_chunks[0], tuple(values[0] for values in layer_chunks))
            elif incremental_exact:
                # Speculative verification must reproduce target-only decode's
                # per-chunk accumulation order. Bulk HLM prefill is causally
                # equivalent but can drift enough numerically to change a
                # rejection decision.
                for chunk_index in range(num_completed):
                    self.highlevel.advance(
                        state,
                        encoder_chunks[chunk_index],
                        tuple(values[chunk_index] for values in layer_chunks),
                    )
            else:
                self.highlevel.prefill(state, encoder_chunks, layer_chunks)

        clear_tensor_buffer(state.pending_encoder_final)
        for layer_values in state.pending_encoder_layers:
            clear_tensor_buffer(layer_values)
        if completed_tokens < int(combined_final.shape[0]):
            append_tensor_buffer(
                state.pending_encoder_final,
                combined_final[completed_tokens:],
                minimum_capacity=chunk_size,
            )
            for pending, values in zip(state.pending_encoder_layers, combined_layers):
                append_tensor_buffer(
                    pending, values[completed_tokens:], minimum_capacity=chunk_size
                )
        return advance

    def _advance_hlm_batches(self, advances: Sequence[_HLMAdvance]) -> None:
        """Batch one-chunk HLM updates at the same concept position."""

        batches: dict[int, list[_HLMAdvance]] = {}
        for advance in advances:
            concept_position = advance[0].predicted_concepts.length
            batches.setdefault(concept_position, []).append(advance)

        for batch in batches.values():
            if len(batch) == 1:
                state, encoder_chunk, layer_chunks = batch[0]
                self.highlevel.advance(state, encoder_chunk, layer_chunks)
                continue
            self.highlevel.advance_batch(
                [advance[0] for advance in batch],
                torch.stack([advance[1] for advance in batch], dim=0),
                tuple(
                    torch.stack([advance[2][layer_index] for advance in batch], dim=0)
                    for layer_index in range(self.backend_config.encoder_layers)
                ),
            )

    def _concept_index_for_position(self, position: int) -> int | None:
        if self.backend_config.shift_feature:
            shifted_chunk = (position + 1) // self.backend_config.chunk_size
        else:
            shifted_chunk = position // self.backend_config.chunk_size
        return shifted_chunk - 1 if shifted_chunk > 0 else None

    def _request_concept_at(
        self,
        state: ConceptRequestState,
        *,
        position: int,
        layer_index: int | None,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        concept_index = self._concept_index_for_position(position)
        if concept_index is None:
            return zero
        if concept_index >= state.predicted_concepts.length:
            raise RuntimeError(
                f"request {state.req_id!r} is missing concept {concept_index} "
                f"at token position {position}"
            )
        if layer_index is None:
            concepts = state.predicted_concepts.data
            if concepts is None:
                raise RuntimeError("missing predicted concept storage")
            return concepts[concept_index]
        layer_states = state.hlm_raw_layer_states[layer_index]
        if concept_index >= layer_states.length:
            raise RuntimeError(
                f"request {state.req_id!r} is missing HLM layer {layer_index} "
                f"concept {concept_index}"
            )
        if layer_states.data is None:
            raise RuntimeError("missing HLM raw layer storage")
        return layer_states.data[concept_index]

    def _build_decoder_concept_inputs(
        self, segments: Sequence[ScheduledRequestSegment], encoder_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        zero = encoder_hidden.new_zeros(self.backend_config.hidden_size)
        final_values: list[torch.Tensor] = []
        layer_values: list[list[torch.Tensor]] = [[] for _ in range(self.backend_config.hlm_layers)]
        for segment in segments:
            for local_index in range(segment.flat_end - segment.flat_start):
                position = segment.position_start + local_index
                final_values.append(
                    self._request_concept_at(
                        segment.state, position=position, layer_index=None, zero=zero
                    )
                )
                for layer_index in range(self.backend_config.hlm_layers):
                    layer_values[layer_index].append(
                        self._request_concept_at(
                            segment.state, position=position, layer_index=layer_index, zero=zero
                        )
                    )
        target_length = int(encoder_hidden.shape[0])
        _append_graph_padding_rows(final_values, target_length=target_length, zero=zero)
        for values in layer_values:
            _append_graph_padding_rows(values, target_length=target_length, zero=zero)
        return (
            torch.stack(final_values, dim=0),
            tuple(torch.stack(values, dim=0) for values in layer_values),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run token encoder, chunk HLM, fusion, and routed token decoder."""

        if intermediate_tensors is not None:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.token_backbone.embed_input_ids(input_ids)
        )
        stage_trace: dict[str, torch.Tensor] | None = (
            {} if not self._profile_mode and self._logits_trace_dir is not None else None
        )

        def record_stage(name: str, value: torch.Tensor) -> None:
            if stage_trace is not None:
                stage_trace[name] = value[-1].detach().float().cpu()

        segments = () if self._profile_mode else self.request_states.resolve_segments()
        if self._ncp_dflash_enabled:
            for segment in segments:
                self._begin_dflash_transaction(segment)

        encoder_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.encoder_layers + 1,
                hidden_states.shape[-1],
            )
        )
        encoder_history[..., 0, :].copy_(hidden_states)
        encoder_cumsum_state = (
            hidden_states if self.backend_config.dd_self_mode == "cumsum" else None
        )
        for layer_index, layer in enumerate(self.token_backbone.encoder.layers):
            record_stage(f"encoder.input.{layer_index}", hidden_states)
            raw_output = layer(positions, hidden_states)
            record_stage(f"encoder.raw.{layer_index}", raw_output)
            encoder_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, encoder_cumsum_state = self.routes.encoder_after_layer(
                layer_index,
                raw_output,
                encoder_history[..., : layer_index + 2, :],
                encoder_cumsum_state,
            )
            record_stage(f"encoder.routed.{layer_index}", hidden_states)
        encoder_hidden = hidden_states
        record_stage("encoder_hidden", encoder_hidden)
        encoder_raw_layers = encoder_history[..., 1:, :].unbind(dim=-2)

        if self._profile_mode:
            final_concepts = torch.zeros_like(encoder_hidden)
            concept_raw_layers = tuple(
                torch.zeros_like(encoder_hidden) for _ in range(self.backend_config.hlm_layers)
            )
        else:
            hlm_advances = []
            for segment in segments:
                advance = self._advance_completed_chunks(
                    segment,
                    encoder_hidden,
                    encoder_raw_layers,
                    incremental_exact=(segment.req_id in self._dflash_transactions),
                )
                if advance is not None:
                    hlm_advances.append(advance)
            self._advance_hlm_batches(hlm_advances)
            final_concepts, concept_raw_layers = self._build_decoder_concept_inputs(
                segments, encoder_hidden
            )
        hidden_states = self.routes.fuse(encoder_hidden, final_concepts)
        record_stage("final_concepts", final_concepts)
        record_stage("fusion_output", hidden_states)
        encoder_sources, concept_sources = self.routes.normalize_decoder_sources(
            encoder_raw_layers, concept_raw_layers
        )
        decoder_gates = self.routes.decoder_gates()

        decoder_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.decoder_layers + 1,
                hidden_states.shape[-1],
            )
        )
        decoder_history[..., 0, :].copy_(hidden_states)
        decoder_cumsum_state = (
            hidden_states if self.backend_config.dd_self_mode == "cumsum" else None
        )
        draft_layer_outputs: dict[int, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.token_backbone.decoder.layers):
            record_stage(f"decoder.input.{layer_index}", hidden_states)
            raw_output = layer(positions, hidden_states)
            record_stage(f"decoder.raw.{layer_index}", raw_output)
            decoder_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, decoder_cumsum_state = self.routes.decoder_after_layer(
                layer_index=layer_index,
                raw_output=raw_output,
                history_states=decoder_history[..., : layer_index + 2, :],
                final_concepts=final_concepts,
                encoder_sources=encoder_sources,
                concept_sources=concept_sources,
                gate=decoder_gates[layer_index],
                cumsum_state=decoder_cumsum_state,
            )
            if layer_index in self._draft_capture_layer_ids:
                draft_layer_outputs[layer_index] = hidden_states
            record_stage(f"decoder.routed.{layer_index}", hidden_states)
        final_layernorm = self.token_backbone.decoder.final_layernorm
        if final_layernorm is None:
            raise RuntimeError("ConceptLM decoder final_layernorm is missing")
        record_stage("decoder.pre_final_norm", hidden_states)
        hidden_states = final_layernorm(hidden_states)
        record_stage("decoder_final", hidden_states)
        if not self._profile_mode:
            for segment in segments:
                self._record_dflash_transaction_features(
                    segment, encoder_hidden, encoder_raw_layers, draft_layer_outputs
                )
                for buffer, layer_id in zip(
                    segment.state.draft_decoder_layers, self._draft_capture_layer_ids, strict=True
                ):
                    if buffer.length != segment.position_start:
                        raise RuntimeError(
                            "DFlash decoder history is not aligned with its segment: "
                            f"layer={layer_id} history={buffer.length} "
                            f"start={segment.position_start}"
                        )
                    append_tensor_buffer(
                        buffer,
                        draft_layer_outputs[layer_id][segment.flat_start : segment.flat_end],
                        minimum_capacity=max(16, segment.position_end),
                    )
                self.request_states.commit(segment)
        if stage_trace is not None:
            stage_trace.update(self.highlevel.stage_trace())
            self._pending_stage_trace = stage_trace
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Project decoder states to vocabulary logits."""

        logits = self.token_backbone.compute_logits(hidden_states)
        if logits is not None and self._logits_trace_dir is not None and not self._profile_mode:
            output_dir = self._logits_trace_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"logits-{self._logits_trace_index:05d}.pt"
            torch.save(logits.detach().float().cpu(), output_path)
            if self._pending_stage_trace is None:
                raise RuntimeError("missing ConceptLM stage trace for logits")
            stage_path = output_dir / f"stages-{self._logits_trace_index:05d}.pt"
            torch.save(self._pending_stage_trace, stage_path)
            self._pending_stage_trace = None
            self._logits_trace_index += 1
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load all parameters from native or dedicated-HF SafeTensors keys."""

        params: dict[str, tuple[str, nn.Parameter]] = {}
        for module_name, module in (
            ("token_backbone", self.token_backbone),
            ("highlevel", self.highlevel),
            ("routes", self.routes),
        ):
            for name, parameter in module.named_parameters(remove_duplicate=False):
                if name in params:
                    raise RuntimeError(f"duplicate native parameter name: {name}")
                params[name] = (f"{module_name}.{name}", parameter)

        stage3_weight_config = getattr(self, "stage3_weight_config", None)
        expected_parameter_count = (
            len(expected_stage3_weight_shapes(stage3_weight_config))
            if stage3_weight_config is not None
            else len(params)
        )
        if len(params) != expected_parameter_count:
            raise RuntimeError(
                "constructed Stage3 model parameter count does not match its "
                f"{self.backend_config.dd_self_mode} weight contract: "
                f"expected {expected_parameter_count}, got {len(params)}"
            )

        loaded_checkpoint_names: set[str] = set()
        loaded_targets: dict[str, set[str | int | None]] = {}
        ignored_metadata = set()
        parameter_names = frozenset(params)
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.endswith("._extra_state"):
                ignored_metadata.add(checkpoint_name)
                continue
            if checkpoint_name in loaded_checkpoint_names:
                raise ValueError(f"duplicate ConceptLM checkpoint tensor: {checkpoint_name}")
            target = resolve_checkpoint_weight(checkpoint_name, parameter_names)
            if target is None:
                raise ValueError(f"unexpected ConceptLM checkpoint tensor: {checkpoint_name}")
            target_shards = loaded_targets.setdefault(target.parameter_name, set())
            if target.shard_id in target_shards:
                raise ValueError(
                    "duplicate ConceptLM checkpoint target: "
                    f"{checkpoint_name} -> {target.parameter_name}"
                    f"[{target.shard_id}]"
                )
            if target.shard_id is None and target_shards:
                raise ValueError(
                    "cannot mix full and split ConceptLM checkpoint tensors for "
                    f"{target.parameter_name}"
                )
            if target.shard_id is not None and None in target_shards:
                raise ValueError(
                    "cannot mix split and full ConceptLM checkpoint tensors for "
                    f"{target.parameter_name}"
                )
            _, parameter = params[target.parameter_name]
            weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
            if target.deinterleave_megatron_qkv:
                query, key, value = deinterleave_megatron_qkv(
                    loaded_weight,
                    num_attention_heads=self.backend_config.num_attention_heads,
                    num_key_value_heads=self.backend_config.num_key_value_heads,
                )
                weight_loader(parameter, query, "q")
                weight_loader(parameter, key, "k")
                weight_loader(parameter, value, "v")
            elif target.shard_id is not None:
                weight_loader(parameter, loaded_weight, target.shard_id)
            else:
                weight_loader(parameter, loaded_weight)
            loaded_checkpoint_names.add(checkpoint_name)
            target_shards.add(target.shard_id)

        complete_parameters: set[str] = set()
        incomplete = []
        for parameter_name, loaded_shards in loaded_targets.items():
            if loaded_shards == {None}:
                complete_parameters.add(parameter_name)
                continue
            required = required_checkpoint_shards(parameter_name)
            if required is not None and loaded_shards == required:
                complete_parameters.add(parameter_name)
                continue
            incomplete.append(
                f"{parameter_name}: loaded={sorted(map(str, loaded_shards))}, "
                f"required={sorted(map(str, required or ()))}"
            )
        if incomplete:
            raise ValueError(
                "incomplete split ConceptLM checkpoint parameters: " + "; ".join(sorted(incomplete))
            )
        missing = sorted(set(params) - complete_parameters)
        if missing:
            raise ValueError("missing ConceptLM checkpoint parameters: " + ", ".join(missing))
        if len(ignored_metadata) not in (0, 2):
            raise ValueError(
                "Stage3 checkpoint must contain zero or two _extra_state tensors, "
                f"got {sorted(ignored_metadata)}"
            )
        loaded_model_names = {params[name][0] for name in complete_parameters}
        expected_model_names = {name for name, _ in self.named_parameters()}
        if loaded_model_names != expected_model_names:
            raise ValueError(
                "loaded vLLM parameter names do not match the constructed model: "
                f"missing={sorted(expected_model_names - loaded_model_names)} "
                f"unexpected={sorted(loaded_model_names - expected_model_names)}"
            )
        return loaded_model_names
