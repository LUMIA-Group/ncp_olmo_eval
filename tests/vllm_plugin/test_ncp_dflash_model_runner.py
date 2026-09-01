"""Contract tests for the vLLM 0.13 DFlash model-runner adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ncp_olmo_eval.vllm_plugin.model_runner import ConceptLMDFlashGPUModelRunner


class _RecordingDrafter:
    def __init__(self) -> None:
        self.call = None
        self.calls = []

    def propose(
        self,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        request_ids=None,
    ):
        self.call = (
            sampled_token_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            request_ids,
        )
        self.calls.append(self.call)
        return [[101], []]


class TestConceptLMDFlashGPUModelRunner(unittest.TestCase):
    def test_rejection_result_finalizes_target_before_proposal(self) -> None:
        runner = ConceptLMDFlashGPUModelRunner.__new__(
            ConceptLMDFlashGPUModelRunner
        )
        parent_result = (
            {},
            None,
            [[11, 12], [21]],
            {},
            ["request-a", "request-b"],
            {"request-a": 0, "request-b": 1},
            [],
        )
        target = SimpleNamespace(finalize_dflash_transactions=lambda *_: None)
        calls = []
        target.finalize_dflash_transactions = lambda reqs, tokens: calls.append(
            (reqs, tokens)
        )

        with (
            patch(
                "ncp_olmo_eval.vllm_plugin.model_runner.GPUModelRunner._bookkeeping_sync",
                return_value=parent_result,
            ),
            patch(
                "ncp_olmo_eval.vllm_plugin.model_runner.target_model",
                return_value=target,
            ),
        ):
            result = runner._bookkeeping_sync(
                object(), object(), object(), object(), 3, object()
            )

        self.assertIs(result, parent_result)
        self.assertEqual(
            calls,
            [(["request-a", "request-b"], [[11, 12], [21]])],
        )

    def test_proposer_bypasses_ngram_implementation(self) -> None:
        runner = ConceptLMDFlashGPUModelRunner.__new__(
            ConceptLMDFlashGPUModelRunner
        )
        runner.drafter = _RecordingDrafter()
        runner.input_batch = SimpleNamespace(
            num_tokens_no_spec=[4, 7],
            token_ids_cpu="token-buffer",
            req_ids=["request-a", "request-b"],
        )
        sampled = [[11], [12]]

        proposals = runner.propose_draft_token_ids(
            scheduler_output=object(),
            sampled_token_ids=sampled,
            sampling_metadata=object(),
            hidden_states=object(),
            sample_hidden_states=object(),
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=object(),
        )

        self.assertEqual(proposals, [[101], []])
        self.assertEqual(
            runner.drafter.call,
            (sampled, [4, 7], "token-buffer", ["request-a", "request-b"]),
        )

    def test_proposer_tracks_continuous_batch_admission_and_reordering(self) -> None:
        runner = ConceptLMDFlashGPUModelRunner.__new__(ConceptLMDFlashGPUModelRunner)
        runner.drafter = _RecordingDrafter()
        runner.input_batch = SimpleNamespace(
            num_tokens_no_spec=[4, 7],
            token_ids_cpu="first-token-buffer",
            req_ids=["request-a", "request-b"],
        )

        runner.propose_draft_token_ids(
            scheduler_output=object(),
            sampled_token_ids=[[11], [12]],
            sampling_metadata=object(),
            hidden_states=object(),
            sample_hidden_states=object(),
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=object(),
        )
        runner.input_batch = SimpleNamespace(
            num_tokens_no_spec=[8, 1],
            token_ids_cpu="second-token-buffer",
            req_ids=["request-b", "request-c"],
        )
        runner.propose_draft_token_ids(
            scheduler_output=object(),
            sampled_token_ids=[[13], [14]],
            sampling_metadata=object(),
            hidden_states=object(),
            sample_hidden_states=object(),
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=object(),
        )

        self.assertEqual(
            runner.drafter.calls,
            [
                (
                    [[11], [12]],
                    [4, 7],
                    "first-token-buffer",
                    ["request-a", "request-b"],
                ),
                (
                    [[13], [14]],
                    [8, 1],
                    "second-token-buffer",
                    ["request-b", "request-c"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

