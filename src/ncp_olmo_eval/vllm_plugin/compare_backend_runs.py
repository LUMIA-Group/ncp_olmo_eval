"""Compare matched baseline and candidate ConceptLM throughput artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _cases_by_batch(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    cases = {
        int(case["batch_size"]): case
        for case in payload["cases"]
    }
    if len(cases) != len(payload["cases"]):
        raise ValueError("throughput artifact contains duplicate batch sizes")
    return cases


def _request_hashes(case: dict[str, Any]) -> list[str]:
    measurements = case["measurements"]
    if not case["repeat_token_hashes_match"]:
        raise ValueError(
            f"batch {case['batch_size']} repeats do not have matching tokens"
        )
    return [
        str(request["token_sha256"])
        for request in measurements[0]["requests"]
    ]


def _metric_pair(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    name: str,
) -> dict[str, float | None]:
    left = baseline.get(name)
    right = candidate.get(name)
    if left is None or right is None:
        return {
            "baseline": left,
            "candidate": right,
            "candidate_over_baseline": None,
        }
    return {
        "baseline": float(left),
        "candidate": float(right),
        "candidate_over_baseline": float(right) / float(left),
    }


def _sample_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left_tokens = [int(token_id) for token_id in baseline["token_ids"]]
    right_tokens = [int(token_id) for token_id in candidate["token_ids"]]
    prefix = _common_prefix_length(left_tokens, right_tokens)
    exact = left_tokens == right_tokens
    aligned_steps = (
        len(left_tokens)
        if exact
        else min(prefix + 1, len(left_tokens), len(right_tokens))
    )
    overlaps = []
    logprob_differences = []
    for left_step, right_step in zip(
        baseline["step_logprobs"][:aligned_steps],
        candidate["step_logprobs"][:aligned_steps],
    ):
        left = {
            int(row["token_id"]): float(row["logprob"])
            for row in left_step
        }
        right = {
            int(row["token_id"]): float(row["logprob"])
            for row in right_step
        }
        shared = set(left) & set(right)
        overlaps.append(len(shared))
        logprob_differences.extend(
            abs(left[token_id] - right[token_id])
            for token_id in shared
        )
    return {
        "exact_tokens": exact,
        "matching_prefix_tokens": prefix,
        "baseline_token_sha256": baseline["token_sha256"],
        "candidate_token_sha256": candidate["token_sha256"],
        "aligned_steps": aligned_steps,
        "top_id_overlap_min": min(overlaps) if overlaps else None,
        "top_id_overlap_mean": (
            statistics.mean(overlaps) if overlaps else None
        ),
        "shared_logprob_abs_max": (
            max(logprob_differences) if logprob_differences else None
        ),
        "shared_logprob_abs_mean": (
            statistics.mean(logprob_differences)
            if logprob_differences
            else None
        ),
    }


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    for name in ("max_model_len", "vllm_version", "torch_version", "device"):
        if baseline[name] != candidate[name]:
            raise ValueError(f"runtime field differs: {name}")
    baseline_cases = _cases_by_batch(baseline)
    candidate_cases = _cases_by_batch(candidate)
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("baseline and candidate batch sizes differ")

    cases = {}
    greedy_ok = True
    for batch_size in baseline_cases:
        left = baseline_cases[batch_size]
        right = candidate_cases[batch_size]
        left_hashes = _request_hashes(left)
        right_hashes = _request_hashes(right)
        exact = left_hashes == right_hashes
        greedy_ok &= exact
        cases[str(batch_size)] = {
            "greedy_exact": exact,
            "baseline_request_hashes": left_hashes,
            "candidate_request_hashes": right_hashes,
            "output_tokens_per_second": _metric_pair(
                left,
                right,
                "median_output_tokens_per_second",
            ),
            "per_request_decode_tokens_per_second": _metric_pair(
                left,
                right,
                "median_per_request_decode_tokens_per_second",
            ),
            "ttft_seconds": _metric_pair(
                left,
                right,
                "median_ttft_seconds",
            ),
            "tpot_seconds": _metric_pair(
                left,
                right,
                "median_tpot_seconds",
            ),
        }

    sampling = _sample_metrics(
        baseline["sampled_accuracy"],
        candidate["sampled_accuracy"],
    )
    return {
        "runtime": {
            "device": baseline["device"],
            "vllm_version": baseline["vllm_version"],
            "torch_version": baseline["torch_version"],
            "max_model_len": baseline["max_model_len"],
        },
        "cases": cases,
        "fixed_seed_sampling": sampling,
        "accuracy_ok": bool(greedy_ok and sampling["exact_tokens"]),
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_json.read_text())
    candidate = json.loads(args.candidate_json.read_text())
    result = compare(baseline, candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if not result["accuracy_ok"]:
        raise RuntimeError("ConceptLM backend A/B accuracy comparison failed")


if __name__ == "__main__":
    main()
