from __future__ import annotations

from ncp_olmo_eval.helmet_api_judge import (
    HELMET_JUDGE_OFFICIAL_REFERENCE_MODEL, HELMET_JUDGE_SEED,
    HELMET_JUDGE_TEMPERATURE, HELMET_JUDGE_TOP_P, HELMET_LONGQA_MAX_TOKENS,
    HELMET_SUMMARY_MAX_TOKENS, parsed_json, required_keys)


def test_helmet_api_judge_parameters_match_pinned_official_scripts() -> None:
    assert HELMET_JUDGE_OFFICIAL_REFERENCE_MODEL == "gpt-4o-2024-05-13"
    assert HELMET_JUDGE_TEMPERATURE == 0.1
    assert HELMET_JUDGE_TOP_P == 0.9
    assert HELMET_JUDGE_SEED == 42
    assert HELMET_LONGQA_MAX_TOKENS == 2048
    assert HELMET_SUMMARY_MAX_TOKENS == 4096


def test_helmet_api_judge_rejects_structurally_incomplete_responses() -> None:
    precision_prompt = 'Return {"precision": 2, "sentence_count": 3}'
    recall_prompt = 'Return {"supported_key_points": [1], "recall": 1}'
    longqa_prompt = 'Return {"fluency": 1, "correctness": 3}'
    fluency_prompt = 'Return {"fluency": 1}'

    assert required_keys(precision_prompt) == {"precision", "sentence_count"}
    assert required_keys(recall_prompt) == {"recall"}
    assert required_keys(longqa_prompt) == {"fluency", "correctness"}
    assert required_keys(fluency_prompt) == {"fluency"}
    assert parsed_json("reasoning\n{\"fluency\": 1}") == {"fluency": 1}
    assert parsed_json("reasoning only") is None
