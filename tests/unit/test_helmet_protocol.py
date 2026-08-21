from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ncp_olmo_eval.helmet_prepare import (
    _reuse_prepared_root,
    find_reusable_helmet_root,
)
from ncp_olmo_eval.helmet_protocol import (
    HELMET_CATEGORY_CONFIGS,
    HELMET_HUB_SOURCE_SPLITS,
    HELMET_LLAMA2_TOKENIZER_MODEL_SHA256,
    HELMET_PINNED_FILE_SHA256,
    HELMET_PROFILES,
    HELMET_SHORT_CATEGORY_CONFIGS,
    HELMET_SWEEP_CATEGORY_CONFIGS,
    _prompt_with_official_truncation,
    _raw_prompt_with_official_truncation,
    _read_checkout_head_without_git,
    _read_checkpoint_rows,
    _redirect_datasets_cache,
    _write_checkpoint_rows,
    promote_checkpoint_rows,
)
from ncp_olmo_eval.helmet_score import (
    HELMET_CATEGORY_PRIMARY_METRICS,
    HELMET_PRIMARY_METRICS,
    _configure_official_scoring_runtime,
    _helmet_task_family,
    _judge_metrics,
    _length_label,
    _tasks_by_family,
)
from ncp_olmo_eval.helmet_trec_eval_compat import RelevanceEvaluator
from ncp_olmo_eval.long_context_inference import _select_rows
from ncp_olmo_eval.long_context_protocol import (
    HELMET_BENCHMARK,
    HELMET_CATEGORY_COUNTS,
    HELMET_FORMAL_EXAMPLE_COUNT,
    HELMET_INPUT_MAX_LENGTH,
    HELMET_MAX_NEW_TOKENS_BY_TASK,
    HELMET_NEWLINE_STOP_TASKS,
    HELMET_OFFICIAL_COMMIT,
    HELMET_SEED,
    HELMET_SHORT_CATEGORY_COUNTS,
    HELMET_SHORT_FORMAL_EXAMPLE_COUNT,
    HELMET_SHORT_INPUT_LENGTHS,
    HELMET_SWEEP_CATEGORY_COUNTS,
    HELMET_SWEEP_FORMAL_EXAMPLE_COUNT,
    HELMET_SWEEP_INPUT_LENGTHS,
    benchmark_sampling_contract,
    file_sha256,
    tokenizer_artifact_contract,
    write_json_atomic,
)


class OffsetTokenizer:
    is_fast = True

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[list[int]] | list[list[tuple[int, int]]]]:
        text = texts[0]
        words = text.split()
        ids = list(range(len(words) + (1 if add_special_tokens else 0)))
        result: dict[str, list[list[int]] | list[list[tuple[int, int]]]] = {"input_ids": [ids]}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for word in words:
                start = text.index(word, cursor)
                end = start + len(word)
                offsets.append((start, end))
                cursor = end
            result["offset_mapping"] = [offsets]
        return result


class ChatOffsetTokenizer(OffsetTokenizer):
    def apply_chat_template(
        self, chat: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert chat == [{"role": "user", "content": "Ask alpha beta"}]
        return "<user> Ask alpha beta <assistant>"


def test_helmet_protocol_is_pinned_to_official_formal_suite() -> None:
    contract = benchmark_sampling_contract(HELMET_BENCHMARK)

    assert HELMET_OFFICIAL_COMMIT == "af609c4d51b97fc35012099380aa889da961c42d"
    assert HELMET_SEED == 42
    assert HELMET_LLAMA2_TOKENIZER_MODEL_SHA256 == (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    )
    assert HELMET_FORMAL_EXAMPLE_COUNT == 5823
    assert HELMET_CATEGORY_COUNTS["RAG"] == 2100
    assert HELMET_CATEGORY_COUNTS["Re-rank"] == 123
    assert sum(HELMET_CATEGORY_COUNTS.values()) == HELMET_FORMAL_EXAMPLE_COUNT
    assert len(HELMET_CATEGORY_CONFIGS) == 7
    assert len(HELMET_MAX_NEW_TOKENS_BY_TASK) == 21
    assert contract["temperature"] == 0.0
    assert contract["top_p"] == 1.0
    assert contract["max_new_tokens"] == 1200
    assert contract["max_new_tokens_by_task"] == HELMET_MAX_NEW_TOKENS_BY_TASK
    assert set(contract["stop_by_task"]) == HELMET_NEWLINE_STOP_TASKS
    assert all(stops == ["\n", "\n\n"] for stops in contract["stop_by_task"].values())


def test_helmet_short_profile_is_pinned_to_official_four_length_sweep() -> None:
    assert HELMET_SHORT_INPUT_LENGTHS == (8192, 16384, 32768, 65536)
    assert HELMET_SHORT_FORMAL_EXAMPLE_COUNT == 4 * HELMET_FORMAL_EXAMPLE_COUNT
    assert HELMET_SHORT_CATEGORY_COUNTS == {
        category: 4 * count for category, count in HELMET_CATEGORY_COUNTS.items()
    }
    assert len(HELMET_SHORT_CATEGORY_CONFIGS) == 7
    assert HELMET_PROFILES["short"]["configs"] == HELMET_SHORT_CATEGORY_CONFIGS
    assert HELMET_PROFILES["short"]["input_lengths"] == HELMET_SHORT_INPUT_LENGTHS
    for _category, config_name in HELMET_SHORT_CATEGORY_CONFIGS:
        assert f"configs/{config_name}" in HELMET_PINNED_FILE_SHA256


def test_helmet_default_profile_is_full_8k_through_64k_sweep() -> None:
    assert HELMET_INPUT_MAX_LENGTH == 65536
    assert HELMET_SWEEP_INPUT_LENGTHS == (8192, 16384, 32768, 65536)
    assert HELMET_SWEEP_FORMAL_EXAMPLE_COUNT == 4 * HELMET_FORMAL_EXAMPLE_COUNT
    assert HELMET_SWEEP_CATEGORY_COUNTS == {
        category: 4 * count for category, count in HELMET_CATEGORY_COUNTS.items()
    }
    assert HELMET_SWEEP_CATEGORY_CONFIGS == HELMET_SHORT_CATEGORY_CONFIGS
    assert HELMET_PROFILES["8k-64k"]["configs"] == HELMET_SWEEP_CATEGORY_CONFIGS
    assert HELMET_PROFILES["8k-64k"]["input_lengths"] == HELMET_SWEEP_INPUT_LENGTHS
    assert "8k-128k" not in HELMET_PROFILES
    assert "128k" not in HELMET_PROFILES


def test_helmet_score_keeps_sweep_lengths_separate(tmp_path: Path) -> None:
    assert [_length_label(value) for value in HELMET_SHORT_INPUT_LENGTHS] == [
        "8k",
        "16k",
        "32k",
        "64k",
    ]
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "input_max_length": 8192,
                        "dataset": "narrativeqa_130772",
                        "metric": "gpt-4-score",
                        "score_percent": 50.0,
                    },
                    {
                        "input_max_length": 16384,
                        "dataset": "narrativeqa_130772",
                        "metric": "gpt-4-score",
                        "score_percent": 60.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _judge_metrics(judge_path, (8192, 16384)) == {
        (8192, "narrativeqa_130772", "gpt-4-score"): 50.0,
        (16384, "narrativeqa_130772", "gpt-4-score"): 60.0,
    }


def test_helmet_scorer_resolves_short_and_full_profile_task_families() -> None:
    for task in ("alce_asqa_30", "alce_asqa_75", "alce_asqa_165", "alce_asqa_345"):
        assert _helmet_task_family(task) == "alce_asqa"
    assert _helmet_task_family("alce_asqa_700") == "alce_asqa"
    for task in ("alce_qampari_30", "alce_qampari_75", "alce_qampari_165", "alce_qampari_345"):
        assert _helmet_task_family(task) == "alce_qampari"
    assert _helmet_task_family("alce_qampari_700") == "alce_qampari"
    assert _helmet_task_family("icl_trec_coarse_400shot_balance") == "icl_trec_coarse"
    assert _helmet_task_family("icl_trec_coarse_6600shot_balance") == "icl_trec_coarse"
    assert _helmet_task_family("narrativeqa_7892") == "narrativeqa"
    assert _helmet_task_family("narrativeqa_130772") == "narrativeqa"
    assert HELMET_PRIMARY_METRICS["alce_asqa"][0] == "str_em"
    assert HELMET_PRIMARY_METRICS["alce_qampari"][0] == "qampari_rec_top5"
    assert ("alce_asqa", "str_em") in HELMET_CATEGORY_PRIMARY_METRICS["Cite"]
    assert ("alce_qampari", "qampari_rec_top5") in HELMET_CATEGORY_PRIMARY_METRICS["Cite"]


def test_helmet_scorer_keeps_exact_profile_task_ids_in_family_index() -> None:
    rows = [
        {"task": "alce_asqa_30", "metadata": {"input_max_length": 8192, "category": "Cite"}},
        {"task": "alce_qampari_30", "metadata": {"input_max_length": 8192, "category": "Cite"}},
        {"task": "alce_asqa_75", "metadata": {"input_max_length": 16384, "category": "Cite"}},
        {"task": "alce_qampari_75", "metadata": {"input_max_length": 16384, "category": "Cite"}},
    ]

    tasks, categories = _tasks_by_family(rows)

    assert tasks == {
        8192: {"alce_asqa": "alce_asqa_30", "alce_qampari": "alce_qampari_30"},
        16384: {"alce_asqa": "alce_asqa_75", "alce_qampari": "alce_qampari_75"},
    }
    assert categories[(8192, "alce_asqa_30")] == "Cite"


def test_helmet_scorer_rejects_unknown_or_colliding_task_families() -> None:
    with pytest.raises(ValueError, match="unsupported HELMET task family"):
        _helmet_task_family("alce_asqa_nocite_30")
    with pytest.raises(ValueError, match="multiple HELMET task IDs"):
        _tasks_by_family(
            [
                {
                    "task": "alce_asqa_30",
                    "metadata": {"input_max_length": 8192, "category": "Cite"},
                },
                {
                    "task": "alce_asqa_75",
                    "metadata": {"input_max_length": 8192, "category": "Cite"},
                },
            ]
        )


def test_helmet_scorer_uses_explicit_local_autoais_and_nltk_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autoais_root = tmp_path / "autoais"
    nltk_root = tmp_path / "nltk_data"
    autoais_root.mkdir()
    nltk_root.mkdir()
    nltk_search_path: list[str] = []
    monkeypatch.setenv("HELMET_AUTOAIS_MODEL", str(autoais_root))
    monkeypatch.setenv("NLTK_DATA", str(nltk_root))
    monkeypatch.setitem(
        sys.modules, "nltk", SimpleNamespace(data=SimpleNamespace(path=nltk_search_path))
    )
    eval_alce = SimpleNamespace(AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture")

    runtime = _configure_official_scoring_runtime(eval_alce)

    assert eval_alce.AUTOAIS_MODEL == str(autoais_root.resolve())
    assert runtime == {
        "autoais_model": str(autoais_root.resolve()),
        "autoais_model_source": "local_override",
        "nltk_data": str(nltk_root.resolve()),
    }
    assert nltk_search_path == [str(nltk_root.resolve())]


def test_helmet_multi_length_judge_rows_require_length(tmp_path: Path) -> None:
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "dataset": "narrativeqa_130772",
                        "metric": "gpt-4-score",
                        "score_percent": 50.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require input_max_length"):
        _judge_metrics(judge_path, (8192, 16384))


def test_helmet_offline_source_contract_covers_every_official_hub_dataset() -> None:
    assert set(HELMET_HUB_SOURCE_SPLITS) == {
        "narrativeqa",
        "multi_lexsum-v20230518",
        "trec",
        "banking77",
        "clinc_oos-plus",
        "nlu_evaluation_data",
        "infinitebench",
    }
    assert HELMET_HUB_SOURCE_SPLITS["infinitebench"]["longbook_qa_eng"] == 351


def test_helmet_official_infinitebench_features_remain_source_pinned() -> None:
    """Keep the adapter out of official dataset loading and schema selection."""

    protocol_source = (
        Path(__file__).resolve().parents[3] / "experiments/vllm_inference/helmet_protocol.py"
    ).read_text(encoding="utf-8")
    assert "datasets_module.disable_caching()" in protocol_source
    assert "datasets_module.enable_caching()" in protocol_source

    assert 'path == "xinrongzhang2022/infinitebench"' in protocol_source
    assert 'options.pop("features", None)' in protocol_source
    assert 'key = "infinitebench"' in protocol_source
    assert "Dataset.from_file" not in protocol_source


def test_helmet_base_prompt_truncation_matches_official_context_tail_rule() -> None:
    sample = {
        "context": "zero one two three four five six seven eight nine",
        "question": "where",
        "answer": ["here"],
    }
    data = {
        "prompt_template": "Instruction {context} Question {question}\nAnswer:",
        "system_template": "Answer:",
    }

    prompt, prepared, contract = _raw_prompt_with_official_truncation(
        sample, data, OffsetTokenizer(), input_max_length=10, generation_max_length=2
    )

    assert contract["original_token_count"] == 15
    assert contract["prepared_token_count"] == 8
    assert contract["truncated_token_count"] == 7
    # HELMET slices at the first character of the removed token and therefore
    # retains the preceding separator exactly as the upstream implementation.
    assert prepared["context"] == "zero one two "
    assert prompt == "Instruction zero one two  Question where\nAnswer:"
    assert sample["context"].endswith("nine")


def test_helmet_chat_prompt_matches_official_task_template_path() -> None:
    prompt, prepared, contract = _prompt_with_official_truncation(
        {"context": "alpha beta", "answer": ["ok"]},
        {
            "prompt_template": "unused {context}",
            "user_template": "Ask {context}",
            "system_template": "Answer:",
        },
        ChatOffsetTokenizer(),
        input_max_length=16,
        generation_max_length=2,
        use_chat_template=True,
    )

    assert prompt == "<user> Ask alpha beta <assistant>"
    assert prepared["context"] == "alpha beta"
    assert contract["use_chat_template"] is True
    assert contract["add_special_tokens"] is False
    assert contract["strategy"] == "official_model_utils_chat_template_context_tail"


def test_helmet_checkout_head_fallback_resolves_loose_ref(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs/heads/main"
    ref.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref.write_text(f"{HELMET_OFFICIAL_COMMIT}\n", encoding="utf-8")

    assert _read_checkout_head_without_git(tmp_path) == HELMET_OFFICIAL_COMMIT


def test_helmet_portable_scoring_keeps_source_and_cache_boundaries() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/ncp_olmo_eval"
    scorer = (package_root / "evaluation_cli.py").read_text(encoding="utf-8")
    prepare = (package_root / "helmet_prepare.py").read_text(encoding="utf-8")

    assert HELMET_PINNED_FILE_SHA256["data.py"] in (
        package_root / "helmet_protocol.py"
    ).read_text(encoding="utf-8")
    assert "--run-citation-nli" in scorer
    assert "--judge-results" in scorer
    assert '"HELMET_AUTOAIS_MODEL": os.environ.get("HELMET_AUTOAIS_MODEL", "")' in scorer
    assert '"NLTK_DATA": os.environ.get("NLTK_DATA", "")' in scorer
    assert "--official-llama2-tokenizer" in prepare
    assert "--hub-source-root" in prepare
    assert "--reuse-under" in prepare
    assert "--derived-cache-root" in prepare
    inference = (package_root / "long_context_inference.py").read_text(encoding="utf-8")
    assert 'get("formal_data_validation_passed")' in inference


def _reusable_helmet_manifest(root: Path, tokenizer_root: Path) -> dict[str, object]:
    inputs_path = root / "inputs.jsonl"
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    inputs_path.write_text('{"example_id":"fixture"}\n', encoding="utf-8")
    tokenizer_contract = tokenizer_artifact_contract(tokenizer_root)
    return {
        "status": "LONG_CONTEXT_DATA_READY",
        "benchmark": HELMET_BENCHMARK,
        "example_count": HELMET_SHORT_FORMAL_EXAMPLE_COUNT,
        "inputs_jsonl_sha256": file_sha256(inputs_path),
        "tokenizer": tokenizer_contract,
        "protocol": {
            "input_lengths": list(HELMET_SHORT_INPUT_LENGTHS),
            "formal_data_validation_passed": True,
            "seed": HELMET_SEED,
            "chat_template_policy": "force_raw",
        },
        "source_data": {
            "formal_data_validation_passed": True,
            "checkout": {
                "commit": HELMET_OFFICIAL_COMMIT,
                "files": {
                    relative: {"sha256": sha256}
                    for relative, sha256 in HELMET_PINNED_FILE_SHA256.items()
                },
            },
            "official_llama2_tokenizer": {
                "tokenizer_model_sha256": HELMET_LLAMA2_TOKENIZER_MODEL_SHA256
            },
            "hub_source_snapshots": {
                "validation": "HELMET_HUB_SOURCE_SNAPSHOTS_OK",
                "contract_sha256": "source-contract",
            },
        },
    }


def test_helmet_preparation_reuses_exact_tokenizer_and_protocol(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
    prepared_parent = tmp_path / "prepared"
    source_root = prepared_parent / "source"
    manifest = _reusable_helmet_manifest(source_root, tokenizer_root)
    write_json_atomic(source_root / "manifest.json", manifest)

    found = find_reusable_helmet_root(
        prepared_parent,
        tokenizer_contract_sha256=tokenizer_artifact_contract(tokenizer_root)["contract_sha256"],
        profile="8k-64k",
        chat_template_policy="force_raw",
    )

    assert found == source_root
    output_root = prepared_parent / "alias"
    reused = _reuse_prepared_root(
        source_root, output_root, tokenizer_artifact_contract(tokenizer_root)
    )
    assert reused["prepared_reuse"]["source_root"] == str(source_root.resolve())
    assert file_sha256(output_root / "inputs.jsonl") == manifest["inputs_jsonl_sha256"]


def test_helmet_preparation_does_not_reuse_prompt_policy_mismatch(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
    prepared_parent = tmp_path / "prepared"
    source_root = prepared_parent / "source"
    manifest = _reusable_helmet_manifest(source_root, tokenizer_root)
    write_json_atomic(source_root / "manifest.json", manifest)

    assert (
        find_reusable_helmet_root(
            prepared_parent,
            tokenizer_contract_sha256=tokenizer_artifact_contract(tokenizer_root)[
                "contract_sha256"
            ],
            profile="8k-64k",
            chat_template_policy="official",
        )
        is None
    )


def test_helmet_preparation_shards_are_hash_validated_and_resumable(tmp_path: Path) -> None:
    rows = [{"example_id": "fixture", "metadata": {"category": "Recall"}}]
    dataset_contract = {"dataset": "fixture", "example_count": 1}
    _write_checkpoint_rows(tmp_path, "001-fixture", "checkpoint-contract", rows, dataset_contract)

    assert _read_checkpoint_rows(tmp_path, "001-fixture", "checkpoint-contract") == (
        rows,
        dataset_contract,
    )
    final_path = tmp_path / "final" / "inputs.jsonl"
    assert promote_checkpoint_rows(tmp_path, final_path, expected_example_count=1)
    assert json.loads(final_path.read_text(encoding="utf-8")) == rows[0]
    # Recreate the checkpoint to exercise corruption rejection independently.
    _write_checkpoint_rows(tmp_path, "001-fixture", "checkpoint-contract", rows, dataset_contract)
    (tmp_path / "inputs.partial.jsonl").write_text("{}\n", encoding="utf-8")
    assert _read_checkpoint_rows(tmp_path, "001-fixture", "checkpoint-contract") is None


def test_helmet_derived_cache_does_not_mutate_frozen_dataset(tmp_path: Path) -> None:
    from datasets import Dataset, load_from_disk

    source_root = tmp_path / "source"
    Dataset.from_dict({"value": [1, 2]}).save_to_disk(source_root)
    source_files_before = {
        path.relative_to(source_root) for path in source_root.rglob("*") if path.is_file()
    }
    dataset = load_from_disk(source_root)

    with _redirect_datasets_cache(tmp_path / "derived"):
        mapped = dataset.map(lambda row: {"doubled": row["value"] * 2})

    assert mapped["doubled"] == [2, 4]
    assert all(
        Path(cache_file["filename"]).is_relative_to(tmp_path / "derived")
        for cache_file in mapped.cache_files
    )
    assert {
        path.relative_to(source_root) for path in source_root.rglob("*") if path.is_file()
    } == source_files_before


def test_helmet_trec_eval_compat_ndcg_at_ten() -> None:
    evaluator = RelevanceEvaluator(
        {"q": {"1": 0, "2": 1, "3": 1}},
        {"ndcg_cut.10", "map_cut.10", "recall.10", "P.10", "recip_rank"},
    )

    metrics = evaluator.evaluate({"q": {"1": 3.0, "2": 2.0, "3": 1.0}})["q"]

    assert metrics["ndcg_cut_10"] == pytest.approx(0.6934264036)
    assert metrics["map_cut_10"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics["recall_10"] == 1.0
    assert metrics["P_10"] == 0.2
    assert metrics["recip_rank"] == 0.5


def test_helmet_smoke_selection_keeps_one_example_per_task() -> None:
    rows = [
        {"task": "a", "example_id": "a0"},
        {"task": "a", "example_id": "a1"},
        {"task": "b", "example_id": "b0"},
        {"task": "b", "example_id": "b1"},
    ]

    selected = _select_rows(
        SimpleNamespace(ruler_sequence_length=0, limit=0, limit_per_task=1),
        {"benchmark": HELMET_BENCHMARK},
        rows,
    )

    assert [row["example_id"] for row in selected] == ["a0", "b0"]
