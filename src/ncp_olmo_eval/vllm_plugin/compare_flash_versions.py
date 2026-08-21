"""Compare matched FA2 and FA3 4K+1K throughput artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa2-json", type=Path, required=True)
    parser.add_argument("--fa3-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _request_tokens(payload: dict[str, Any]) -> list[int]:
    cases = payload["cases"]
    if len(cases) != 1 or int(cases[0]["batch_size"]) != 1:
        raise ValueError("FA comparison requires one batch-1 throughput case")
    measurements = cases[0]["measurements"]
    if not cases[0]["repeat_token_hashes_match"]:
        raise ValueError("greedy throughput repeats do not have matching tokens")
    return [int(token_id) for token_id in measurements[0]["requests"][0]["token_ids"]]


def _validate_pair(
    fa2: dict[str, Any],
    fa3: dict[str, Any],
) -> None:
    if int(fa2["backend_constraints"]["flash_attn_version"]) != 2:
        raise ValueError("FA2 artifact does not record flash_attn_version=2")
    if int(fa3["backend_constraints"]["flash_attn_version"]) != 3:
        raise ValueError("FA3 artifact does not record flash_attn_version=3")
    for field in (
        "prompt_length",
        "prompt_offset",
        "max_new_tokens",
        "temperature",
        "top_p",
        "seed",
        "requested_logprobs",
    ):
        if fa2["sampled_accuracy"][field] != fa3["sampled_accuracy"][field]:
            raise ValueError(f"sampled accuracy field differs: {field}")
    for field in ("max_model_len", "vllm_version", "torch_version", "device"):
        if fa2[field] != fa3[field]:
            raise ValueError(f"runtime field differs: {field}")


def _aligned_logprob_metrics(
    fa2_steps: list[list[dict[str, Any]]],
    fa3_steps: list[list[dict[str, Any]]],
    aligned_steps: int,
) -> dict[str, Any]:
    overlaps = []
    shared_differences = []
    exact_top_ids = 0
    for index in range(aligned_steps):
        left = {
            int(row["token_id"]): float(row["logprob"])
            for row in fa2_steps[index]
        }
        right = {
            int(row["token_id"]): float(row["logprob"])
            for row in fa3_steps[index]
        }
        left_ids = set(left)
        right_ids = set(right)
        shared = left_ids & right_ids
        overlaps.append(len(shared))
        exact_top_ids += int(left_ids == right_ids)
        shared_differences.extend(
            abs(left[token_id] - right[token_id])
            for token_id in shared
        )
    return {
        "aligned_steps": aligned_steps,
        "exact_top_id_set_steps": exact_top_ids,
        "top_id_overlap_min": min(overlaps) if overlaps else None,
        "top_id_overlap_mean": (
            statistics.mean(overlaps) if overlaps else None
        ),
        "shared_logprob_abs_max": (
            max(shared_differences) if shared_differences else None
        ),
        "shared_logprob_abs_mean": (
            statistics.mean(shared_differences)
            if shared_differences
            else None
        ),
    }


def compare(
    fa2: dict[str, Any],
    fa3: dict[str, Any],
) -> dict[str, Any]:
    _validate_pair(fa2, fa3)
    fa2_case = fa2["cases"][0]
    fa3_case = fa3["cases"][0]
    fa2_greedy = _request_tokens(fa2)
    fa3_greedy = _request_tokens(fa3)
    greedy_prefix = _common_prefix_length(fa2_greedy, fa3_greedy)
    fa2_sampled = [
        int(token_id) for token_id in fa2["sampled_accuracy"]["token_ids"]
    ]
    fa3_sampled = [
        int(token_id) for token_id in fa3["sampled_accuracy"]["token_ids"]
    ]
    sampled_prefix = _common_prefix_length(fa2_sampled, fa3_sampled)
    sampled_exact = fa2_sampled == fa3_sampled
    aligned_steps = (
        len(fa2_sampled)
        if sampled_exact
        else min(sampled_prefix + 1, len(fa2_sampled), len(fa3_sampled))
    )
    return {
        "runtime": {
            "device": fa2["device"],
            "vllm_version": fa2["vllm_version"],
            "torch_version": fa2["torch_version"],
            "prompt_length": fa2["sampled_accuracy"]["prompt_length"],
            "max_new_tokens": fa2["sampled_accuracy"]["max_new_tokens"],
        },
        "speed": {
            "fa2_output_tokens_per_second": fa2_case[
                "median_output_tokens_per_second"
            ],
            "fa3_output_tokens_per_second": fa3_case[
                "median_output_tokens_per_second"
            ],
            "fa3_over_fa2_output_tps": (
                fa3_case["median_output_tokens_per_second"]
                / fa2_case["median_output_tokens_per_second"]
            ),
            "fa2_ttft_seconds": fa2_case["median_ttft_seconds"],
            "fa3_ttft_seconds": fa3_case["median_ttft_seconds"],
            "fa2_decode_tokens_per_second": fa2_case[
                "median_per_request_decode_tokens_per_second"
            ],
            "fa3_decode_tokens_per_second": fa3_case[
                "median_per_request_decode_tokens_per_second"
            ],
        },
        "greedy": {
            "exact_match": fa2_greedy == fa3_greedy,
            "matching_prefix_tokens": greedy_prefix,
            "first_divergence_index": (
                None if fa2_greedy == fa3_greedy else greedy_prefix
            ),
            "fa2_token_sha256": fa2_case["measurements"][0][
                "requests"
            ][0]["token_sha256"],
            "fa3_token_sha256": fa3_case["measurements"][0][
                "requests"
            ][0]["token_sha256"],
        },
        "fixed_seed_sampling": {
            "temperature": fa2["sampled_accuracy"]["temperature"],
            "top_p": fa2["sampled_accuracy"]["top_p"],
            "seed": fa2["sampled_accuracy"]["seed"],
            "exact_match": sampled_exact,
            "matching_prefix_tokens": sampled_prefix,
            "first_divergence_index": (
                None if sampled_exact else sampled_prefix
            ),
            "fa2_token_sha256": fa2["sampled_accuracy"]["token_sha256"],
            "fa3_token_sha256": fa3["sampled_accuracy"]["token_sha256"],
            "aligned_logprobs": _aligned_logprob_metrics(
                fa2["sampled_accuracy"]["step_logprobs"],
                fa3["sampled_accuracy"]["step_logprobs"],
                aligned_steps,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    fa2 = json.loads(args.fa2_json.read_text())
    fa3 = json.loads(args.fa3_json.read_text())
    result = compare(fa2, fa3)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
