"""Tests for matched FA2/FA3 artifact comparison."""

from __future__ import annotations

import copy
import unittest

from ncp_olmo_eval.vllm_plugin.compare_flash_versions import compare


def _payload(version: int) -> dict:
    tokens = [11, 12, 13]
    return {
        "backend_constraints": {"flash_attn_version": version},
        "max_model_len": 7,
        "vllm_version": "0.13.0",
        "torch_version": "2.9.0",
        "device": "NVIDIA H200",
        "cases": [
            {
                "batch_size": 1,
                "repeat_token_hashes_match": True,
                "median_output_tokens_per_second": 60.0 + version,
                "median_ttft_seconds": 0.2,
                "median_per_request_decode_tokens_per_second": 61.0,
                "measurements": [
                    {
                        "requests": [
                            {
                                "token_ids": tokens,
                                "token_sha256": f"greedy-fa{version}",
                            }
                        ]
                    }
                ],
            }
        ],
        "sampled_accuracy": {
            "prompt_length": 4,
            "prompt_offset": 0,
            "max_new_tokens": 3,
            "temperature": 0.8,
            "top_p": 0.95,
            "seed": 7,
            "requested_logprobs": 2,
            "token_ids": tokens,
            "token_sha256": f"sampled-fa{version}",
            "step_logprobs": [
                [
                    {"token_id": 11, "logprob": -0.1, "rank": 1},
                    {"token_id": 21, "logprob": -1.1, "rank": 2},
                ],
                [
                    {"token_id": 12, "logprob": -0.2, "rank": 1},
                    {"token_id": 22, "logprob": -1.2, "rank": 2},
                ],
                [
                    {"token_id": 13, "logprob": -0.3, "rank": 1},
                    {"token_id": 23, "logprob": -1.3, "rank": 2},
                ],
            ],
        },
    }


class TestCompareFlashVersions(unittest.TestCase):
    def test_exact_trajectories_compare_all_steps(self) -> None:
        report = compare(_payload(2), _payload(3))

        self.assertTrue(report["greedy"]["exact_match"])
        self.assertTrue(report["fixed_seed_sampling"]["exact_match"])
        self.assertEqual(
            report["fixed_seed_sampling"]["aligned_logprobs"]["aligned_steps"],
            3,
        )

    def test_sampled_divergence_compares_only_aligned_contexts(self) -> None:
        fa2 = _payload(2)
        fa3 = copy.deepcopy(_payload(3))
        fa3["sampled_accuracy"]["token_ids"][1] = 99

        report = compare(fa2, fa3)

        sampled = report["fixed_seed_sampling"]
        self.assertFalse(sampled["exact_match"])
        self.assertEqual(sampled["first_divergence_index"], 1)
        self.assertEqual(sampled["aligned_logprobs"]["aligned_steps"], 2)


if __name__ == "__main__":
    unittest.main()
