"""Tests for matched ConceptLM backend A/B artifact comparison."""

from __future__ import annotations

import copy
import unittest

from ncp_olmo_eval.vllm_plugin.compare_backend_runs import compare


def _payload() -> dict:
    tokens = [11, 12, 13]
    request_hashes = ["request-a", "request-b"]
    return {
        "max_model_len": 7,
        "vllm_version": "0.13.0",
        "torch_version": "2.9.0",
        "device": "NVIDIA H200",
        "cases": [
            {
                "batch_size": 2,
                "repeat_token_hashes_match": True,
                "median_output_tokens_per_second": 100.0,
                "median_ttft_seconds": 0.2,
                "median_tpot_seconds": 0.02,
                "median_per_request_decode_tokens_per_second": 50.0,
                "measurements": [
                    {
                        "requests": [
                            {"token_sha256": value}
                            for value in request_hashes
                        ]
                    }
                ],
            }
        ],
        "sampled_accuracy": {
            "token_ids": tokens,
            "token_sha256": "sample",
            "step_logprobs": [
                [
                    {"token_id": token_id, "logprob": -0.1, "rank": 1},
                    {"token_id": token_id + 10, "logprob": -1.1, "rank": 2},
                ]
                for token_id in tokens
            ],
        },
    }


class TestCompareBackendRuns(unittest.TestCase):
    def test_exact_accuracy_and_speed_ratio(self) -> None:
        baseline = _payload()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["median_output_tokens_per_second"] = 125.0

        report = compare(baseline, candidate)

        self.assertTrue(report["accuracy_ok"])
        self.assertEqual(
            report["cases"]["2"]["output_tokens_per_second"][
                "candidate_over_baseline"
            ],
            1.25,
        )
        self.assertEqual(
            report["fixed_seed_sampling"]["shared_logprob_abs_max"],
            0.0,
        )

    def test_greedy_change_fails_accuracy(self) -> None:
        baseline = _payload()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["measurements"][0]["requests"][1][
            "token_sha256"
        ] = "changed"

        report = compare(baseline, candidate)

        self.assertFalse(report["accuracy_ok"])
        self.assertFalse(report["cases"]["2"]["greedy_exact"])


if __name__ == "__main__":
    unittest.main()
