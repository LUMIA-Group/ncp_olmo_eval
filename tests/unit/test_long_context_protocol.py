from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ncp_olmo_eval import benchmark, long_context_inference, long_context_prepare
from ncp_olmo_eval.long_context_prepare import _parse_sequence_lengths
from ncp_olmo_eval.long_context_protocol import (
    HELMET_BENCHMARK,
    LONGBENCH_V2_BENCHMARK,
    LONGBENCH_V2_PROMPT,
    LONGBENCH_V2_PROMPT_SHA256,
    PREDICTION_STATUS,
    RULER_BENCHMARK,
    RULER_DATA_ARCHIVE_SHA256,
    RULER_DATA_FILES_CONTRACT_SHA256,
    RULER_DATA_GENERATION_CONTRACT,
    RULER_DATASET,
    RULER_DATASET_REVISION,
    RULER_MAX_NEW_TOKENS,
    RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH,
    RULER_MAX_NEW_TOKENS_BY_TASK,
    RULER_QA_PROMPT_CONTRACT,
    RULER_SEQUENCE_LENGTHS,
    RULER_SOURCE_DATA_KIND,
    RULER_TASKS,
    SCHEMA_VERSION,
    SHARD_STATUS,
    benchmark_sampling_contract,
    build_prepared_manifest,
    extract_longbench_v2_answer,
    file_sha256,
    model_context_capability,
    prepare_longbench_v2_rows,
    prepare_ruler_rows,
    prepared_official_sampling_contract,
    resolve_model_context_contract,
    ruler_max_new_tokens,
    score_longbench_v2,
    score_ruler,
    score_ruler_generation,
    truncate_head_tail,
    validate_backend_shape,
    validate_prepared_dataset,
    write_json_atomic,
    write_jsonl_atomic,
)
from ncp_olmo_eval.long_context_score import score_run
from ncp_olmo_eval.prepare_long_context_model_overlay import (
    create_long_context_model_overlay,
)


class WordTokenizer:
    chat_template = "fake"

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{value}" for value in token_ids)

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"USER: {messages[0]['content']}\nASSISTANT:"


class ExpandingRoundTripTokenizer(WordTokenizer):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens) + " extra"


def _olmes_source_fixture() -> dict[str, str]:
    return {
        "kind": RULER_SOURCE_DATA_KIND,
        "dataset": RULER_DATASET,
        "dataset_revision": RULER_DATASET_REVISION,
        "archive_sha256": RULER_DATA_ARCHIVE_SHA256,
        "files_contract_sha256": RULER_DATA_FILES_CONTRACT_SHA256,
    }


def test_contract_tokenizer_falls_back_from_tokenizers_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise ValueError("Tokenizer class TokenizersBackend does not exist")

    class FakeFastTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> object:
            assert kwargs == {"local_files_only": True}
            return sentinel

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", FakeFastTokenizer)

    assert long_context_inference._load_contract_tokenizer(tmp_path) is sentinel


def test_prepare_tokenizer_falls_back_from_tokenizers_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise ValueError("Tokenizer class TokenizersBackend does not exist")

    class FakeFastTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> object:
            assert kwargs == {"local_files_only": True}
            return sentinel

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", FakeFastTokenizer)

    assert long_context_prepare._load_tokenizer(tmp_path) is sentinel
def test_long_context_hf_model_parallel_size_reaches_model_loader() -> None:
    args = SimpleNamespace(
        checkpoint_root="",
        ckpt_step=0,
        train_wandb_config="",
        tokenizer_model=Path("tokenizer"),
        hf_model_path="model",
        hf_compile_routes=False,
        hf_align_dcp_runtime_config=False,
        allow_context_extension=True,
        batch_size=1,
        hf_model_parallel_size=2,
    )

    model_args = long_context_inference._backend_model_args(args, "transformers", 131072)

    assert model_args.hf_model_parallel_size == 2


def test_hf_model_parallel_worker_uses_process_local_primary_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_torchrun_layout() -> int:
        raise AssertionError("model-parallel HF workers are not torchrun ranks")

    monkeypatch.setattr(benchmark, "local_cuda_device_index", reject_torchrun_layout)

    device = benchmark._transformers_hf_primary_device(2)

    assert device.type == "cuda"
    assert device.index == 0


def test_hf_transformers_runtime_contract_separates_native_and_ncp_olmo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_OLMO3_TRANSFORMERS_VERSION", raising=False)
    monkeypatch.delenv("HF_NCP_OLMO_TRANSFORMERS_VERSION", raising=False)
    native_config = {"model_type": "olmo3", "architectures": ["Olmo3ForCausalLM"]}
    ncp_config = {
        "model_type": "ncp_olmo3",
        "architectures": ["NCPOlmo3ForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_ncp_olmo3.NCPOlmo3ForCausalLM"},
    }

    assert benchmark._validate_transformers_runtime(native_config, "4.57.6") == (
        "native_olmo",
        "4.57.6",
    )
    assert benchmark._validate_transformers_runtime(ncp_config, "5.8.0") == ("ncp_olmo", "5.8.0")
    with pytest.raises(RuntimeError, match="must not share an environment"):
        benchmark._validate_transformers_runtime(native_config, "5.8.0")
    with pytest.raises(RuntimeError, match="must not share an environment"):
        benchmark._validate_transformers_runtime(ncp_config, "4.57.6")


def test_ncp_hf_runtime_requires_standalone_remote_code() -> None:
    with pytest.raises(RuntimeError, match="auto_map.AutoModelForCausalLM"):
        benchmark._validate_transformers_runtime(
            {"model_type": "ncp_olmo3", "architectures": ["NCPOlmo3ForCausalLM"]}, "5.8.0"
        )


def test_generic_transformers_model_has_no_project_specific_pin() -> None:
    assert benchmark._validate_transformers_runtime(
        {"model_type": "generic_causal_lm"}, "99.0.0"
    ) == ("generic_transformers", "")


def test_independent_hf_shards_do_not_join_a_process_group() -> None:
    assert not long_context_inference._uses_torchrun_process_group(
        SimpleNamespace(backend="hf", shard_index=0)
    )
    assert long_context_inference._uses_torchrun_process_group(
        SimpleNamespace(backend="hf", shard_index=-1)
    )
    assert long_context_inference._uses_torchrun_process_group(
        SimpleNamespace(backend="native_megatron", shard_index=0)
    )


def test_long_context_inference_rejects_noncanonical_seed() -> None:
    with pytest.raises(ValueError, match="fixed to 42"):
        long_context_inference.run(SimpleNamespace(global_seed=7))


def test_checkpoint_context_extension_preserves_megatron_encoder_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_args = SimpleNamespace(
        seq_length=65536, encoder_seq_length=None, max_position_embeddings=65536
    )
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    checkpointing.load_args_from_checkpoint = lambda *_args, **_kwargs: (
        checkpoint_args,
        "checkpoint",
    )
    training = types.ModuleType("megatron.training")
    training.checkpointing = checkpointing
    megatron = types.ModuleType("megatron")
    megatron.training = training
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)

    benchmark._force_checkpoint_context_length(131072)
    result, _ = checkpointing.load_args_from_checkpoint()

    assert result.seq_length == 131072
    assert result.max_position_embeddings == 131072
    assert result.encoder_seq_length is None


def _manifest(benchmark: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return build_prepared_manifest(
        benchmark=benchmark,
        rows=rows,
        tokenizer_contract={"contract_sha256": "tokenizer"},
        source_data={"fixture": True},
        protocol={"fixture": True},
    )


def _prediction(
    row: dict[str, object], generation: str, *, sample_index: int = 0, samples_per_example: int = 1
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": PREDICTION_STATUS,
        "benchmark": row["benchmark"],
        "example_id": row["example_id"],
        "sample_index": sample_index,
        "samples_per_example": samples_per_example,
        "prompt_sha256": row["prompt_sha256"],
        "generation": generation,
    }


def test_longbench_prompt_is_exactly_pinned() -> None:
    assert hashlib.sha256(LONGBENCH_V2_PROMPT.encode()).hexdigest() == (LONGBENCH_V2_PROMPT_SHA256)


def test_ruler_formal_sweep_is_4k_through_64k() -> None:
    assert RULER_SEQUENCE_LENGTHS == (4096, 8192, 16384, 32768, 65536)
    assert _parse_sequence_lengths("4096,8192,16384,32768,65536") == RULER_SEQUENCE_LENGTHS
    assert 131072 not in RULER_SEQUENCE_LENGTHS
    with pytest.raises(ValueError, match="unique and increasing"):
        _parse_sequence_lengths("8192,4096")


def test_prepared_ruler_rejects_128k_length(tmp_path: Path) -> None:
    prompt = "find the needle"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": RULER_BENCHMARK,
        "example_id": "131072:niah_single_1:0",
        "task": "niah_single_1",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "expected_answer": ["needle"],
        "metadata": {
            "sequence_length": 131072,
            "max_new_tokens": 50,
            "data_generation_contract": RULER_DATA_GENERATION_CONTRACT,
        },
    }
    output_root = tmp_path / "ruler-128k"
    write_jsonl_atomic(output_root / "inputs.jsonl", [row])
    manifest = build_prepared_manifest(
        benchmark=RULER_BENCHMARK,
        rows=[row],
        tokenizer_contract={"contract_sha256": "tokenizer"},
        source_data=_olmes_source_fixture(),
        protocol={
            **benchmark_sampling_contract(RULER_BENCHMARK),
            "qa_prompt_contract": RULER_QA_PROMPT_CONTRACT,
            "data_generation_contract": RULER_DATA_GENERATION_CONTRACT,
            "sequence_lengths": [131072],
        },
    )
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="exceed the current 64K protocol"):
        validate_prepared_dataset(output_root)


def test_prepared_helmet_rejects_128k_length(tmp_path: Path) -> None:
    prompt = "long context prompt"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": HELMET_BENCHMARK,
        "example_id": "helmet:131072:0",
        "task": "ruler_niah_mk_2",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "expected_answer": ["needle"],
        "metadata": {"input_max_length": 131072},
    }
    output_root = tmp_path / "helmet-128k"
    write_jsonl_atomic(output_root / "inputs.jsonl", [row])
    manifest = build_prepared_manifest(
        benchmark=HELMET_BENCHMARK,
        rows=[row],
        tokenizer_contract={"contract_sha256": "tokenizer"},
        source_data={"fixture": True},
        protocol={"input_lengths": [131072]},
    )
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="exceed the current 64K protocol"):
        validate_prepared_dataset(output_root)


def test_prepared_helmet_short_profile_uses_its_length_specific_task_ids(tmp_path: Path) -> None:
    tasks_by_length = {
        8192: "narrativeqa_7892",
        16384: "narrativeqa_16084",
        32768: "narrativeqa_32468",
        65536: "narrativeqa_65236",
    }
    rows = []
    for input_max_length, task in tasks_by_length.items():
        prompt = f"short HELMET prompt {input_max_length}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": HELMET_BENCHMARK,
                "example_id": f"LongQA:{task}:{input_max_length}:0",
                "task": task,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "expected_answer": "answer",
                "metadata": {
                    "category": "LongQA",
                    "input_max_length": input_max_length,
                    "max_new_tokens": 100,
                    "stop": [],
                },
            }
        )
    output_root = tmp_path / "helmet-short"
    write_jsonl_atomic(output_root / "inputs.jsonl", rows)
    task_budgets = {task: 100 for task in tasks_by_length.values()}
    manifest = build_prepared_manifest(
        benchmark=HELMET_BENCHMARK,
        rows=rows,
        tokenizer_contract={"contract_sha256": "tokenizer"},
        source_data={
            "profile": "8k-64k",
            "input_lengths": list(tasks_by_length),
            "datasets": [
                {
                    "dataset": task,
                    "input_max_length": input_max_length,
                    "generation_max_length": 100,
                    "stop_new_line": False,
                }
                for input_max_length, task in tasks_by_length.items()
            ],
        },
        protocol={
            "helmet_profile": "8k-64k",
            "input_lengths": list(tasks_by_length),
            "formal_data_validation_passed": False,
            "seed": 42,
            "max_new_tokens": 100,
            "max_new_tokens_by_task": task_budgets,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [],
            "stop_by_task": {},
        },
    )
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)

    validated, _ = validate_prepared_dataset(output_root)
    sampling = prepared_official_sampling_contract(validated)

    assert sampling["max_new_tokens_by_task"] == task_budgets
    assert "narrativeqa_130772" not in sampling["max_new_tokens_by_task"]


def test_ruler_uses_olmes_task_and_length_generation_budgets() -> None:
    contract = benchmark_sampling_contract(RULER_BENCHMARK)

    expected = {
        "niah_single_1": 50,
        "niah_single_2": 50,
        "niah_single_3": 50,
        "niah_multikey_1": 50,
        "niah_multikey_2": 50,
        "niah_multikey_3": 100,
        "niah_multivalue": 250,
        "niah_multiquery": 100,
        "vt": 50,
        "cwe": 100,
        "fwe": 50,
        "qa_1": 50,
        "qa_2": 50,
    }
    assert contract["max_new_tokens"] == RULER_MAX_NEW_TOKENS == 300
    assert contract["max_new_tokens_by_task"] == RULER_MAX_NEW_TOKENS_BY_TASK
    assert RULER_MAX_NEW_TOKENS_BY_TASK == expected
    assert contract["max_new_tokens_by_sequence_length"] == (
        RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH
    )
    assert ruler_max_new_tokens("niah_multivalue", 4096) == 300
    assert ruler_max_new_tokens("niah_multivalue", 8192) == 250
    assert ruler_max_new_tokens("niah_multikey_3", 65536) == 100
    assert contract["eos_stopping"] is True


def test_ruler_qa_prefers_full_input_over_short_question(tmp_path: Path) -> None:
    setup_root = tmp_path / "ruler"
    write_jsonl_atomic(
        setup_root / "qa_1" / "validation_4096.jsonl",
        [
            {
                "index": 0,
                "input": "document evidence\n\nQuestion: where? Answer:",
                "question": "where?",
                "outputs": ["France"],
                "length": 4096,
            }
        ],
    )

    rows = prepare_ruler_rows(setup_root, WordTokenizer(), sequence_length=4096, tasks=("qa_1",))

    assert rows[0]["prompt"] == "document evidence\n\nQuestion: where? Answer:"
    assert rows[0]["metadata"]["prompt_source"] == "input"
    assert rows[0]["metadata"]["prompt_token_count"] > len(WordTokenizer().encode("where?"))


def test_ruler_duplicate_source_indices_use_stable_row_ids(tmp_path: Path) -> None:
    setup_root = tmp_path / "ruler"
    write_jsonl_atomic(
        setup_root / "qa_1" / "validation_4096.jsonl",
        [
            {
                "index": 13865,
                "input": "first document\n\nQuestion: where? Answer:",
                "outputs": ["France"],
                "length": 4096,
            },
            {
                "index": 13865,
                "input": "second document\n\nQuestion: where? Answer:",
                "outputs": ["Spain"],
                "length": 4096,
            },
        ],
    )

    rows = prepare_ruler_rows(setup_root, WordTokenizer(), sequence_length=4096, tasks=("qa_1",))

    assert [row["example_id"] for row in rows] == ["4096:qa_1:0", "4096:qa_1:1"]
    assert [row["metadata"]["source_index"] for row in rows] == ["13865", "13865"]
    assert [row["metadata"]["source_row_index"] for row in rows] == [0, 1]


def test_ruler_appends_explicit_answer_prefix(tmp_path: Path) -> None:
    setup_root = tmp_path / "ruler"
    write_jsonl_atomic(
        setup_root / "qa_2" / "validation_4096.jsonl",
        [
            {
                "index": 0,
                "input": "document evidence\n\nQuestion: where?",
                "question": "where?",
                "answer_prefix": " Answer:",
                "outputs": ["France"],
                "length": 4096,
            }
        ],
    )

    rows = prepare_ruler_rows(setup_root, WordTokenizer(), sequence_length=4096, tasks=("qa_2",))

    assert rows[0]["prompt"].endswith("Question: where? Answer:")
    assert rows[0]["metadata"]["prompt_source"] == "input+answer_prefix"


def test_ruler_prepare_builds_complete_five_length_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    setup_root = tmp_path / "ruler-all"
    for sequence_length in RULER_SEQUENCE_LENGTHS:
        for task in RULER_TASKS:
            rows = [
                {
                    "index": index,
                    "input": f"find needle {index}",
                    "outputs": ["needle"],
                    "length": sequence_length,
                }
                for index in range(100)
            ]
            write_jsonl_atomic(setup_root / task / f"validation_{sequence_length}.jsonl", rows)
    monkeypatch.setattr(long_context_prepare, "_load_tokenizer", lambda _path: WordTokenizer())
    monkeypatch.setattr(
        long_context_prepare,
        "_validate_olmes_ruler_data_root",
        lambda _root: (setup_root, _olmes_source_fixture()),
    )
    output_root = tmp_path / "prepared"

    manifest = long_context_prepare.prepare(
        SimpleNamespace(
            benchmark=RULER_BENCHMARK,
            olmes_ruler_data_root=tmp_path / "allenai-ruler-data",
            sequence_lengths=",".join(str(value) for value in RULER_SEQUENCE_LENGTHS),
            tokenizer_model=tokenizer_root,
            output_root=output_root,
            prompt_transport="raw",
            allow_nonstandard_count=False,
        )
    )

    assert manifest["example_count"] == 6500
    assert manifest["protocol"]["sequence_lengths"] == list(RULER_SEQUENCE_LENGTHS)
    assert manifest["protocol"]["qa_prompt_contract"] == RULER_QA_PROMPT_CONTRACT
    assert manifest["protocol"]["data_generation_contract"] == (RULER_DATA_GENERATION_CONTRACT)
    assert "generation_tokenizer_contract_sha256" not in manifest["protocol"]
    assert manifest["protocol"]["formal_data_validation_passed"] is True
    assert manifest["protocol"]["full_sweep_validation_passed"] is True
    validated_manifest, validated_rows = validate_prepared_dataset(output_root)
    assert validated_manifest == manifest
    assert len(validated_rows) == 6500


def test_ruler_validates_fixed_olmes_cache_and_all_official_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "allenai--RULER"
    archive_path = cache_root / "data_100_samples.tgz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"official archive fixture")
    setup_root = cache_root / "data/ruler"
    for sequence_length in RULER_SEQUENCE_LENGTHS:
        for task in RULER_TASKS:
            write_jsonl_atomic(
                setup_root / task / f"validation_{sequence_length}.jsonl",
                [{"input": "prompt", "outputs": ["answer"]} for _ in range(100)],
            )
    monkeypatch.setattr(
        long_context_prepare, "RULER_DATA_ARCHIVE_SIZE_BYTES", archive_path.stat().st_size
    )
    monkeypatch.setattr(
        long_context_prepare, "RULER_DATA_ARCHIVE_SHA256", file_sha256(archive_path)
    )
    file_contract = {
        str(path.relative_to(cache_root)): file_sha256(path)
        for path in sorted(setup_root.glob("*/validation_*.jsonl"))
        if int(path.stem.rsplit("_", 1)[1]) in RULER_SEQUENCE_LENGTHS
    }
    monkeypatch.setattr(
        long_context_prepare,
        "RULER_DATA_FILES_CONTRACT_SHA256",
        long_context_prepare.canonical_json_sha256(file_contract),
    )

    validated_root, provenance = long_context_prepare._validate_olmes_ruler_data_root(cache_root)

    assert validated_root == setup_root
    assert provenance["kind"] == RULER_SOURCE_DATA_KIND
    assert provenance["dataset_revision"] == RULER_DATASET_REVISION
    assert len(provenance["files_by_sequence_length"]) == len(RULER_SEQUENCE_LENGTHS)


def test_ruler_rejects_legacy_question_only_prepared_data(tmp_path: Path) -> None:
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": RULER_BENCHMARK,
        "example_id": "4096:qa_1:0",
        "task": "qa_1",
        "prompt": "In what country is Normandy located?",
        "prompt_sha256": hashlib.sha256(b"In what country is Normandy located?").hexdigest(),
        "expected_answer": ["France"],
        "metadata": {
            "sequence_length": 4096,
            "prompt_token_count": 8,
            "prompt_transport": "raw",
            "match_type": "part",
        },
    }
    output_root = tmp_path / "legacy-ruler"
    write_jsonl_atomic(output_root / "inputs.jsonl", [row])
    manifest = build_prepared_manifest(
        benchmark=RULER_BENCHMARK,
        rows=[row],
        tokenizer_contract={"contract_sha256": "tokenizer"},
        source_data={"fixture": True},
        protocol={"qa_prompt_contract": "legacy-question-only"},
    )
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="QA prompt contract mismatch"):
        validate_prepared_dataset(output_root)


def test_ruler_rejects_import_only_tokenizer_retagged_cache(tmp_path: Path) -> None:
    prompt = "documents and evidence\n\nQuestion: where? Answer:"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": RULER_BENCHMARK,
        "example_id": "4096:qa_1:0",
        "task": "qa_1",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "expected_answer": ["France"],
        "metadata": {
            "sequence_length": 4096,
            "prompt_token_count": 6,
            "prompt_transport": "raw",
            "prompt_source": "input",
            "match_type": "part",
        },
    }
    output_root = tmp_path / "retagged-ruler"
    write_jsonl_atomic(output_root / "inputs.jsonl", [row])
    manifest = build_prepared_manifest(
        benchmark=RULER_BENCHMARK,
        rows=[row],
        tokenizer_contract={"contract_sha256": "current-tokenizer"},
        source_data={"kind": "imported-legacy-cache"},
        protocol={
            **benchmark_sampling_contract(RULER_BENCHMARK),
            "qa_prompt_contract": RULER_QA_PROMPT_CONTRACT,
            "formal_data_validation_passed": True,
        },
    )
    manifest["inputs_jsonl_sha256"] = file_sha256(output_root / "inputs.jsonl")
    write_json_atomic(output_root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="data contract mismatch"):
        validate_prepared_dataset(output_root)


def test_ruler_qa_uses_olmes_normalized_substring_exact_match() -> None:
    assert score_ruler_generation("It is The United States!", ["United States"], "part") == 1.0
    assert score_ruler_generation("It is Canada.", ["United States"], "part") == 0.0


def test_longbench_prepare_uses_head_tail_truncation() -> None:
    source = {
        "_id": "example-1",
        "domain": "Single-Document QA",
        "sub_domain": "Academic Papers",
        "difficulty": "hard",
        "length": "long",
        "question": "which answer",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": "C",
        "context": " ".join(f"word-{index}" for index in range(100)),
    }

    rows = prepare_longbench_v2_rows([source], WordTokenizer(), max_input_tokens=20)

    assert len(rows) == 1
    assert rows[0]["metadata"]["truncation"]["strategy"] == "official_head_tail"
    assert rows[0]["metadata"]["truncation"]["selected_token_count"] == 20
    assert rows[0]["expected_answer"] == "C"


def test_longbench_truncation_accounts_for_tokenizer_roundtrip_growth() -> None:
    prompt, metadata = truncate_head_tail(
        " ".join(f"word-{index}" for index in range(100)),
        ExpandingRoundTripTokenizer(),
        max_input_tokens=20,
    )

    assert len(ExpandingRoundTripTokenizer().encode(prompt)) == 20
    assert metadata["selected_token_count"] == 19
    assert metadata["roundtrip_trimmed_token_count"] == 1


def test_backend_shape_keeps_hf_and_megatron_batch_one() -> None:
    validate_backend_shape("hf", 1, 1)
    validate_backend_shape("native_megatron", 1, 1)
    validate_backend_shape("native_vllm", 8, 8)
    validate_backend_shape("lmdeploy", 8, 8)

    with pytest.raises(ValueError, match="batch_size=1"):
        validate_backend_shape("hf", 8, 1)
    with pytest.raises(ValueError, match="samples_per_example=1"):
        validate_backend_shape("native_megatron", 1, 2)


def test_model_context_contract_uses_native_config_and_requires_explicit_extension(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(
        config_path,
        {
            "max_sequence_length": 65536,
            "max_position_embeddings": 65536,
            "position_embedding_type": "yarn",
            "yarn_original_max_position_embeddings": 8192,
            "yarn_rotary_scaling_factor": 8.0,
            "window_size": [4096, 0],
        },
    )

    capability = model_context_capability(config_path)
    contract = resolve_model_context_contract(capability=capability, required_context_length=64000)

    assert capability["native_context_length"] == 65536
    assert contract["runtime_context_length"] == 64000
    assert contract["validation"] == "MODEL_CONTEXT_CONTRACT_OK"
    with pytest.raises(ValueError, match="explicitly allow"):
        resolve_model_context_contract(capability=capability, required_context_length=65537)

    extended = resolve_model_context_contract(
        capability=capability,
        required_context_length=131072,
        requested_context_length=131072,
        allow_context_extension=True,
    )
    assert extended["runtime_context_length"] == 131072
    assert extended["uses_context_extrapolation"] is True
    assert extended["context_mode"] == "explicit_unvalidated_extrapolation"


def test_model_context_contract_allows_only_request_derived_generation_overhang(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, {"max_sequence_length": 65536, "max_position_embeddings": 65536})
    capability = model_context_capability(config_path)

    contract = resolve_model_context_contract(
        capability=capability, required_context_length=69546, max_generation_overhang=4096
    )

    assert contract["runtime_context_length"] == 69546
    assert contract["generation_overhang_tokens"] == 4010
    assert contract["max_generation_overhang"] == 4096
    assert contract["bounded_generation_overhang"] is True
    assert contract["context_mode"] == "official_generation_overhang"

    with pytest.raises(ValueError, match="explicitly allow"):
        resolve_model_context_contract(
            capability=capability, required_context_length=69633, max_generation_overhang=4096
        )
    with pytest.raises(ValueError, match="explicitly allow"):
        resolve_model_context_contract(
            capability=capability,
            required_context_length=69546,
            requested_context_length=69632,
            max_generation_overhang=4096,
        )


def test_ruler_runtime_context_allows_task_specific_generation_overhang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, {"max_sequence_length": 65536, "max_position_embeddings": 65536})
    rows = [
        {
            "example_id": "65536:vt:0",
            "task": "vt",
            "prompt": "sealed prompt",
            "metadata": {"truncation": {"prepared_token_count": 65520}},
        }
    ]
    args = SimpleNamespace(
        model_config_path=str(config_path),
        model_identity_path=str(tmp_path),
        hf_model_path=str(tmp_path),
        model_context_length=0,
        allow_context_extension=False,
        backend="native_vllm",
    )

    contract = long_context_inference._runtime_context_contract(
        args,
        rows,
        WordTokenizer(),
        max_new_tokens=128,
        max_new_tokens_by_task={"vt": 30},
        benchmark_name=RULER_BENCHMARK,
    )

    assert contract["required_context_length"] == 65550
    assert contract["context_mode"] == "official_generation_overhang"
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0")
    long_context_inference._apply_native_vllm_context_policy(contract)
    assert long_context_inference.os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"


def test_non_ruler_runtime_extension_still_requires_an_overlay(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, {"max_sequence_length": 65536, "max_position_embeddings": 65536})
    rows = [
        {
            "example_id": "helmet:0",
            "task": "ruler_niah_mk_2",
            "prompt": "sealed prompt",
            "metadata": {"truncation": {"prepared_token_count": 65536}},
        }
    ]
    args = SimpleNamespace(
        model_config_path=str(config_path),
        model_identity_path=str(tmp_path),
        hf_model_path=str(tmp_path),
        model_context_length=65586,
        allow_context_extension=True,
        backend="hf",
    )

    with pytest.raises(ValueError, match="create an explicit long-context model overlay"):
        long_context_inference._runtime_context_contract(
            args,
            rows,
            WordTokenizer(),
            max_new_tokens=50,
            max_new_tokens_by_task={"ruler_niah_mk_2": 50},
            benchmark_name="helmet",
        )


def test_model_context_contract_fails_when_declared_fields_disagree(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(
        config_path, {"max_sequence_length": 131072, "max_position_embeddings": 65536}
    )

    with pytest.raises(ValueError, match="context fields disagree"):
        model_context_capability(config_path)


def test_runtime_context_contract_uses_sealed_prompt_counts_without_retokenizing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(
        config_path, {"max_sequence_length": 131072, "max_position_embeddings": 131072}
    )

    class RejectingTokenizer:
        def encode(self, _text: str) -> list[int]:
            raise AssertionError("sealed HELMET prompts must not be retokenized at startup")

    rows = [
        {
            "example_id": "helmet:0",
            "task": "ruler_niah_mk_2",
            "prompt": "large sealed prompt",
            "metadata": {"truncation": {"prepared_token_count": 131000}},
        }
    ]
    args = SimpleNamespace(
        model_config_path=str(config_path),
        model_identity_path=str(tmp_path),
        hf_model_path=str(tmp_path),
        model_context_length=131072,
        allow_context_extension=False,
        backend="hf",
    )

    contract = long_context_inference._runtime_context_contract(
        args,
        rows,
        RejectingTokenizer(),
        max_new_tokens=50,
        max_new_tokens_by_task={"ruler_niah_mk_2": 50},
    )

    assert contract["required_context_length"] == 131050
    assert contract["prompt_token_count_sources"] == ["sealed_prepared_metadata"]


def test_runtime_context_contract_falls_back_for_legacy_prepared_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, {"max_sequence_length": 128, "max_position_embeddings": 128})
    rows = [{"example_id": "legacy:0", "task": "legacy", "prompt": "one two three"}]
    args = SimpleNamespace(
        model_config_path=str(config_path),
        model_identity_path=str(tmp_path),
        hf_model_path=str(tmp_path),
        model_context_length=0,
        allow_context_extension=False,
        backend="hf",
    )

    contract = long_context_inference._runtime_context_contract(
        args, rows, WordTokenizer(), max_new_tokens=4, max_new_tokens_by_task={}
    )

    assert contract["required_context_length"] == 7
    assert contract["prompt_token_count_sources"] == ["runtime_tokenizer_fallback"]


def test_long_context_overlay_links_weights_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_json_atomic(
        source / "config.json",
        {
            "max_sequence_length": 65536,
            "max_position_embeddings": 65536,
            "yarn_rotary_scaling_factor": 8.0,
        },
    )
    (source / "model.safetensors").write_bytes(b"weights")
    source_config_before = (source / "config.json").read_bytes()

    output = tmp_path / "overlay"
    manifest = create_long_context_model_overlay(source, output, 131072)

    overlay_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert overlay_config["max_sequence_length"] == 131072
    assert overlay_config["max_position_embeddings"] == 131072
    assert overlay_config["yarn_rotary_scaling_factor"] == 8.0
    assert (output / "model.safetensors").is_symlink()
    assert (source / "config.json").read_bytes() == source_config_before
    assert manifest["context_mode"] == "explicit_unvalidated_extrapolation"

    with pytest.raises(FileExistsError):
        create_long_context_model_overlay(source, output, 131072)


def test_ruler_matcher_reproduces_all_and_part() -> None:
    assert score_ruler_generation("alpha and beta", ["alpha", "beta"], "all") == 1.0
    assert score_ruler_generation("alpha", ["alpha", "beta"], "all") == 0.5
    assert score_ruler_generation("contains BETA", ["alpha", "beta"], "part") == 1.0


def test_ruler_macro_uses_all_thirteen_tasks_and_keeps_sample_zero_primary() -> None:
    rows = []
    predictions = []
    for sequence_length in RULER_SEQUENCE_LENGTHS:
        for index, task in enumerate(RULER_TASKS):
            row = {
                "schema_version": SCHEMA_VERSION,
                "benchmark": RULER_BENCHMARK,
                "example_id": f"{sequence_length}:{task}:0",
                "task": task,
                "prompt": "prompt",
                "prompt_sha256": "prompt-hash",
                "expected_answer": ["needle"],
                "metadata": {
                    "sequence_length": sequence_length,
                    "match_type": "part" if task.startswith("qa") else "all",
                },
            }
            rows.append(row)
            predictions.append(_prediction(row, "needle", sample_index=0, samples_per_example=2))
            predictions.append(
                _prediction(
                    row,
                    "missing" if index == 0 else "needle",
                    sample_index=1,
                    samples_per_example=2,
                )
            )

    score = score_ruler(_manifest(RULER_BENCHMARK, rows), rows, predictions)

    assert score["official_protocol_compatible"] is False
    assert [row["sequence_length"] for row in score["summary_rows"]] == list(RULER_SEQUENCE_LENGTHS)
    for summary in score["summary_rows"]:
        assert summary["official_sample0_macro_accuracy"] == 1.0
        assert summary["sample_mean_macro_accuracy"] == pytest.approx((1.0 + 12 / 13) / 2)


def test_longbench_extractor_and_official_sample_zero_score() -> None:
    assert extract_longbench_v2_answer("The correct answer is (B)") == "B"
    assert extract_longbench_v2_answer("**The correct answer is C**") == "C"
    assert extract_longbench_v2_answer("I choose D") is None
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": LONGBENCH_V2_BENCHMARK,
        "example_id": "lb-1",
        "task": "QA",
        "prompt": "prompt",
        "prompt_sha256": "prompt-hash",
        "expected_answer": "A",
        "metadata": {
            "domain": "QA",
            "sub_domain": "Single-doc",
            "difficulty": "easy",
            "length": "short",
        },
    }
    predictions = [
        _prediction(row, "The correct answer is (B)", sample_index=0, samples_per_example=3),
        _prediction(row, "The correct answer is (A)", sample_index=1, samples_per_example=3),
        _prediction(row, "The correct answer is (A)", sample_index=2, samples_per_example=3),
    ]

    score = score_longbench_v2(_manifest(LONGBENCH_V2_BENCHMARK, [row]), [row], predictions)

    assert score["official_sample0_accuracy"] == 0.0
    assert score["sample_mean_accuracy"] == pytest.approx(2 / 3)
    assert score["pass_at_k"] == 1.0
    assert score["majority_vote_accuracy"] == 1.0
    assert score["official_protocol_compatible"] is False


def test_vllm_multi_sample_inference_uses_stable_per_request_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from ncp_olmo_eval import long_context_inference as inference

    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    tokenizer_root = tmp_path / "tokenizer"
    data_root.mkdir()
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    rows = []
    for index in range(2):
        prompt = f"prompt {index}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": LONGBENCH_V2_BENCHMARK,
                "example_id": f"lb-{index}",
                "task": "QA",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "expected_answer": "A",
                "metadata": {
                    "domain": "QA",
                    "sub_domain": "Single-doc",
                    "difficulty": "easy",
                    "length": "short",
                },
            }
        )
    tokenizer_contract = {
        "path": str(tokenizer_root),
        "files": {"tokenizer.json": "fake"},
        "contract_sha256": "tokenizer-contract",
    }
    manifest = _manifest(LONGBENCH_V2_BENCHMARK, rows)
    manifest["tokenizer"] = tokenizer_contract
    write_jsonl_atomic(data_root / "inputs.jsonl", rows)
    manifest["inputs_jsonl_sha256"] = file_sha256(data_root / "inputs.jsonl")
    write_json_atomic(data_root / "manifest.json", manifest)

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    class FakeInferencer:
        tokenizer = FakeTokenizer()

        def __init__(self) -> None:
            self.seeds: list[int] = []

        def generate_batch(
            self, prompts: list[str], samplings: list[object]
        ) -> list[SimpleNamespace]:
            self.seeds.extend(int(sampling.seed) for sampling in samplings)
            return [
                SimpleNamespace(
                    text="The correct answer is (A)", token_ids=[1, 2], finish_reason="stop"
                )
                for _ in prompts
            ]

    fake = FakeInferencer()
    monkeypatch.setattr(inference, "tokenizer_artifact_contract", lambda _path: tokenizer_contract)
    monkeypatch.setattr(
        inference,
        "_build_inferencer",
        lambda *_args: (
            fake,
            fake.tokenizer,
            "native_vllm",
            {"backend": "native_vllm"},
            SimpleNamespace(),
            {"artifact": "unchanged"},
        ),
    )
    monkeypatch.setattr(
        inference.benchmark, "_artifact_fingerprint", lambda _args: {"artifact": "unchanged"}
    )
    args = SimpleNamespace(
        backend="native_vllm",
        batch_size=4,
        samples_per_example=2,
        data_root=data_root,
        output_root=output_root,
        limit=0,
        shard_index=0,
        shard_count=1,
        max_new_tokens=0,
        temperature=None,
        top_p=None,
        global_seed=42,
        tokenizer_model=tokenizer_root,
        resume=True,
        model_label="fixture",
    )

    result = inference.run(args)

    predictions = [
        json.loads(line)
        for line in (output_root / "shard-000-of-001" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["prediction_count"] == 4
    assert len(fake.seeds) == len(set(fake.seeds)) == 4
    assert {row["sample_index"] for row in predictions} == {0, 1}
    assert result["official_protocol_compatible"] is False


def test_partial_longbench_score_writes_all_report_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    inference_root = tmp_path / "inference"
    output_root = tmp_path / "score"
    shard_root = inference_root / "shard-000-of-001"
    data_root.mkdir()
    shard_root.mkdir(parents=True)
    prompt = "fixture prompt"
    row = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": LONGBENCH_V2_BENCHMARK,
        "example_id": "lb-fixture",
        "task": "Single-Document QA",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "expected_answer": "A",
        "metadata": {
            "domain": "Single-Document QA",
            "sub_domain": "Academic Papers",
            "difficulty": "easy",
            "length": "short",
        },
    }
    manifest = build_prepared_manifest(
        benchmark=LONGBENCH_V2_BENCHMARK,
        rows=[row],
        tokenizer_contract={"contract_sha256": "fixture-tokenizer"},
        source_data={"fixture": True},
        protocol={
            "formal_example_count": 503,
            "formal_data_validation_passed": False,
            "prompt_transport": "raw",
        },
    )
    write_jsonl_atomic(data_root / "inputs.jsonl", [row])
    manifest["inputs_jsonl_sha256"] = file_sha256(data_root / "inputs.jsonl")
    write_json_atomic(data_root / "manifest.json", manifest)
    prediction = _prediction(row, "The correct answer is (A)")
    write_jsonl_atomic(shard_root / "predictions.jsonl", [prediction])
    write_json_atomic(
        shard_root / "result.json",
        {
            "status": SHARD_STATUS,
            "benchmark": LONGBENCH_V2_BENCHMARK,
            "model_label": "fixture",
            "backend": "native_vllm",
            "resolved_backend": "native_vllm",
            "batch_size": 8,
            "samples_per_example": 1,
            "global_seed": 42,
            "sampling": {"max_new_tokens": 128, "temperature": 0.1, "top_p": 1.0, "stop": []},
            "official_protocol_compatible": True,
            "shard_index": 0,
            "shard_count": 1,
            "artifact_mutated": False,
            "prepared_manifest_sha256": file_sha256(data_root / "manifest.json"),
            "prepared_inputs_sha256": file_sha256(data_root / "inputs.jsonl"),
        },
    )

    score = score_run(
        SimpleNamespace(
            data_root=data_root,
            inference_root=inference_root,
            output_root=output_root,
            allow_partial=True,
        )
    )

    assert score["official_sample0_accuracy"] == 1.0
    assert score["official_protocol_compatible"] is False
    assert (output_root / "score.json").is_file()
    assert (output_root / "longbench-v2-official-summary.csv").is_file()
    assert (output_root / "longbench-v2-breakdown.csv").is_file()
    assert (output_root / "longbench-v2-scored.jsonl").is_file()
