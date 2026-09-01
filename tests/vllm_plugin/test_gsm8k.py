"""Host-only tests for the native-vLLM GSM8K evaluation helpers."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from ncp_olmo_eval.vllm_plugin.gsm8k import (
    STANDARD_MAX_GEN_TOKS,
    STANDARD_NUM_FEWSHOT,
    STANDARD_TEMPERATURE,
    STANDARD_TOP_P,
    STOP_STRINGS,
    _load_and_validate_prepared_inputs,
    _model_fingerprint,
    _prompt_sha256,
    _rank_schedule,
    _require_consistent_model_artifact,
    _request_seed,
    _score_prediction,
    _validate_standard_task,
)


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            task="olmo_eval_paper_gsm8k_main",
            num_fewshot=STANDARD_NUM_FEWSHOT,
            generation_kwargs={
                "do_sample": True,
                "temperature": STANDARD_TEMPERATURE,
                "top_p": STANDARD_TOP_P,
                "max_gen_toks": STANDARD_MAX_GEN_TOKS,
                "until": list(STOP_STRINGS),
            },
        ),
        fewshot_cfg=SimpleNamespace(sampler="first_n", samples=[{"x": 1}]),
    )


def test_prompt_sha256_is_order_and_boundary_sensitive() -> None:
    assert _prompt_sha256(["ab", "c"]) != _prompt_sha256(["a", "bc"])
    assert _prompt_sha256(["a", "b"]) != _prompt_sha256(["b", "a"])
    assert _prompt_sha256(["a", "b"]) == _prompt_sha256(["a", "b"])


def test_prepared_inputs_reject_actual_jsonl_drift(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config = tmp_path / "standard.json"
    dataset = tmp_path / "test.parquet"
    config.write_text("{}\n", encoding="utf-8")
    dataset.write_bytes(b"dataset")
    rows = [{"doc_index": 0, "prompt": "prompt zero"}, {"doc_index": 1, "prompt": "prompt one"}]
    inputs = input_dir / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "dataset_sample_count": 2,
        "prompt_sha256": _prompt_sha256([row["prompt"] for row in rows]),
        "inputs_jsonl": str(inputs.resolve()),
        "inputs_jsonl_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "standard_input_config": str(config),
        "standard_input_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "dataset_test_file": str(dataset),
        "dataset_test_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (input_dir / "_SUCCESS").touch()

    assert _load_and_validate_prepared_inputs(input_dir)[1] == rows
    rows[0]["prompt"] = "tampered but still valid JSON"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inputs JSONL changed"):
        _load_and_validate_prepared_inputs(input_dir)


def _write_model_metadata(model_dir) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_model_fingerprint_covers_all_three_shard_olmo_weights(tmp_path) -> None:
    model_dir = tmp_path / "olmo"
    _write_model_metadata(model_dir)
    (model_dir / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    weight_names = [f"model-{shard:05d}-of-00003.safetensors" for shard in range(1, 4)]
    for shard, name in enumerate(weight_names, start=1):
        (model_dir / name).write_bytes(bytes([shard]))

    before = _model_fingerprint(model_dir)

    fingerprinted_names = {row["name"] for row in before["files"]}
    assert set(weight_names) <= fingerprinted_names
    (model_dir / weight_names[1]).write_bytes(b"changed-weight")
    assert _model_fingerprint(model_dir) != before


def test_model_fingerprint_supports_one_unindexed_safetensor(tmp_path) -> None:
    model_dir = tmp_path / "single-file"
    _write_model_metadata(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"weights")

    fingerprint = _model_fingerprint(model_dir)

    assert "model.safetensors.index.json" not in {row["name"] for row in fingerprint["files"]}
    assert any(row["name"] == "model.safetensors" for row in fingerprint["files"])


def test_model_fingerprint_rejects_missing_safetensor_weights(tmp_path) -> None:
    model_dir = tmp_path / "no-weights"
    _write_model_metadata(model_dir)

    with pytest.raises(RuntimeError, match="no safetensor weights"):
        _model_fingerprint(model_dir)


def test_resume_aggregation_ignores_model_overlay_mtime_only() -> None:
    artifact = {
        "model_dir": "/evaluation/model",
        "files": [
            {
                "name": "config.json",
                "resolved_path": "/evaluation/model/config.json",
                "size": 100,
                "mtime_ns": 1,
                "sha256": "abc",
            }
        ],
    }
    resumed = {
        "model_dir": artifact["model_dir"],
        "files": [{**artifact["files"][0], "mtime_ns": 2}],
    }

    assert _require_consistent_model_artifact(
        [{"model_artifact": artifact}, {"model_artifact": resumed}]
    ) == artifact


def test_resume_aggregation_rejects_model_content_change() -> None:
    artifact = {
        "model_dir": "/evaluation/model",
        "files": [
            {
                "name": "config.json",
                "resolved_path": "/evaluation/model/config.json",
                "size": 100,
                "mtime_ns": 1,
                "sha256": "abc",
            }
        ],
    }
    changed = {
        "model_dir": artifact["model_dir"],
        "files": [{**artifact["files"][0], "mtime_ns": 2, "sha256": "def"}],
    }

    with pytest.raises(ValueError, match="inconsistent shard field model_artifact"):
        _require_consistent_model_artifact(
            [{"model_artifact": artifact}, {"model_artifact": changed}]
        )


def test_standard_task_contract_accepts_paper_protocol() -> None:
    contract = _validate_standard_task(_task())
    assert contract["num_fewshot"] == 8
    assert contract["fewshot_sampler"] == "first_n"
    assert contract["generation_kwargs"]["temperature"] == 0.6


def test_standard_task_contract_rejects_generation_drift() -> None:
    task = _task()
    task.config.generation_kwargs["top_p"] = 0.95
    with pytest.raises(ValueError, match="generation contract changed"):
        _validate_standard_task(task)


def test_rank_schedule_has_exact_doc_sample_coverage() -> None:
    schedules = [
        _rank_schedule(doc_count=3, samples_per_doc=8, rank=rank, world_size=8) for rank in range(8)
    ]
    observed = sorted(request for schedule in schedules for request in schedule)
    expected = [(doc_index, sample_index) for doc_index in range(3) for sample_index in range(8)]

    assert observed == expected
    assert all(len(schedule) == 3 for schedule in schedules)


def test_request_seed_is_stable_and_request_specific() -> None:
    seeds = {
        _request_seed(42, doc_index, sample_index)
        for doc_index in range(3)
        for sample_index in range(8)
    }

    assert len(seeds) == 24
    assert _request_seed(42, 2, 7) == _request_seed(42, 2, 7)
    assert _request_seed(42, 2, 7) != _request_seed(43, 2, 7)


def test_prediction_scoring_uses_official_last_number_contract() -> None:
    score = _score_prediction(
        {"answer": "The answer is #### 20"}, "The final ratio is 1/5, or 20%."
    )

    assert score["pass_at_1"] == 1.0
    assert score["standard_correct"] is True
    assert score["official_prediction"] == "20"
    assert score["official_gold"] == "20"
    assert score["answer_scorer"] == "olmo_eval_gsm_last_number_exact_match"


def test_prediction_scoring_preserves_official_numeric_string_exactness() -> None:
    score = _score_prediction({"answer": "The answer is #### 26"}, "The answer is 26.00.")

    assert score["pass_at_1"] == 0.0
    assert score["standard_correct"] is False
    assert score["official_prediction"] == "26.00"
    assert score["official_gold"] == "26"
