from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ncp_olmo_eval import core_native_pool
from ncp_olmo_eval.core_native_eval import (
    _max_inference_batch_size,
    _model_manifest_paths,
    _prepare_scoring_requests,
)
from ncp_olmo_eval.inference import SamplingParams
from ncp_olmo_eval.native_vllm_inference import (
    NativeVLLMInferencer,
    native_vllm_max_model_len,
    prepare_native_vllm_model,
    validate_native_vllm_args,
)


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    add_bos_token = False

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            return [1, *[ord(character) for character in text]]
        return [ord(character) for character in text]


class _Engine:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict, bool]] = []

    @staticmethod
    def get_tokenizer() -> _Tokenizer:
        return _Tokenizer()

    def generate(self, prompts, sampling, *, use_tqdm: bool):
        self.calls.append((prompts, sampling, use_tqdm))
        if isinstance(sampling, dict) and sampling.get("prompt_logprobs") is not None:
            outputs = []
            for prompt in prompts:
                token_ids = prompt["prompt_token_ids"]
                prompt_logprobs = [None]
                for index, token_id in enumerate(token_ids[1:], start=1):
                    prompt_logprobs.append(
                        {
                            token_id: SimpleNamespace(
                                logprob=-0.1 * index, rank=1 if token_id != 21 else 2
                            )
                        }
                    )
                outputs.append(SimpleNamespace(prompt_logprobs=prompt_logprobs, outputs=[]))
            return outputs
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"answer-{index}", token_ids=[7, 8 + index], finish_reason="stop"
                    )
                ]
            )
            for index, _ in enumerate(prompts)
        ]


def _inferencer(tmp_path: Path) -> tuple[NativeVLLMInferencer, _Engine]:
    engine = _Engine()
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=32,
        seed=1234,
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    return inferencer, engine


def test_core88_engine_capacity_covers_scoring_and_generation_batches() -> None:
    assert (
        _max_inference_batch_size(SimpleNamespace(score_batch_size=8, generation_batch_size=1)) == 8
    )
    assert (
        _max_inference_batch_size(SimpleNamespace(score_batch_size=1, generation_batch_size=8)) == 8
    )


def test_core_pool_validates_only_the_selected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_calls: list[tuple[int, int]] = []

    def reject_lmdeploy(*_args, **_kwargs) -> None:
        raise AssertionError("LMDeploy validation must not run for native_vllm")

    monkeypatch.setattr(
        core_native_pool,
        "validate_native_vllm_args",
        lambda _args, *, batch_size, processes_per_gpu: native_calls.append(
            (batch_size, processes_per_gpu)
        ),
    )
    monkeypatch.setattr(
        core_native_pool,
        "validate_lmdeploy_args",
        reject_lmdeploy,
    )
    monkeypatch.setattr(core_native_pool, "_file_sha256", lambda _path: "sealed-summary")
    args = SimpleNamespace(
        score_batch_size=8,
        generation_batch_size=8,
        pad_multiple=128,
        hf_backend="native_vllm",
        gpus=8,
        processes_per_gpu=1,
        machine_count=4,
        global_seed=42,
        profile="core88",
        limit_per_task=0,
        generation_samples_cap=0,
        max_gen_tokens_cap=0,
        data_root=tmp_path,
    )
    plan = {
        "machine_count": 4,
        "global_seed": 42,
        "profile": "core88",
        "limit_per_task": 0,
        "generation_samples_cap": 0,
        "max_gen_tokens_cap": 0,
        "data_root": str(tmp_path.resolve()),
        "data_summary_sha256": "sealed-summary",
    }

    core_native_pool._validate_args(args, plan)

    assert native_calls == [(8, 1)]


def test_native_vllm_generation_preserves_local_completion_schema(tmp_path: Path) -> None:
    inferencer, engine = _inferencer(tmp_path)

    completions = inferencer.generate(
        ["prompt"],
        SamplingParams(max_tokens=8, temperature=0.6, top_p=0.7, stop=["Question:"], seed=42),
    )

    assert completions[0].text == "answer-0"
    assert completions[0].token_ids == [7, 8]
    assert completions[0].finish_reason == "stop"
    assert completions[0].kv_stats["cache_backend"] == ("native_vllm_paged_attention")
    assert engine.calls[0][1] == [
        {
            "temperature": 0.6,
            "top_p": 0.7,
            "max_tokens": 8,
            "stop": ["Question:"],
            "ignore_eos": False,
            "detokenize": True,
            "seed": 42,
        }
    ]


def test_native_vllm_generation_batches_preserve_request_seeds(tmp_path: Path) -> None:
    engine = _Engine()
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=32,
        seed=1234,
        max_batch_size=8,
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    completions = inferencer.generate_batch(
        ["first", "second"],
        [
            SamplingParams(max_tokens=8, temperature=0.6, seed=41),
            SamplingParams(max_tokens=16, temperature=0.7, seed=42),
        ],
    )

    assert [completion.text for completion in completions] == ["answer-0", "answer-1"]
    assert [parameters["seed"] for parameters in engine.calls[0][1]] == [41, 42]
    assert [parameters["max_tokens"] for parameters in engine.calls[0][1]] == [8, 16]


def test_native_vllm_records_tensor_parallel_engine_topology(tmp_path: Path) -> None:
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=32,
        seed=1234,
        tensor_parallel_size=2,
        engine=_Engine(),
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert inferencer.runtime_metadata["tensor_parallel_size"] == 2


def test_native_vllm_scores_every_continuation_token_from_prompt_logprobs(tmp_path: Path) -> None:
    inferencer, engine = _inferencer(tmp_path)
    requests = [
        {
            "row_index": 0,
            "query_ids": [10, 11, 20],
            "targets": [{"candidate_index": 0, "positions": [1, 2], "token_ids": [20, 21]}],
            "fast_mc": False,
        }
    ]
    candidate_results: list[list[dict | None]] = [[None]]

    stats = inferencer.score_requests(requests, candidate_results)

    assert engine.calls[0][0] == [{"prompt_token_ids": [10, 11, 20, 21]}]
    assert engine.calls[0][1]["prompt_logprobs"] == 1
    assert candidate_results[0][0] == {
        "token_ids": [20, 21],
        "token_logprobs": pytest.approx([-0.2, -0.3]),
        "sum_logprob": pytest.approx(-0.5),
        "num_tokens": 2,
        "is_greedy": False,
    }
    assert stats["continuation_token_count"] == 2


def test_native_vllm_scoring_capacity_covers_full_protocol_window(tmp_path: Path) -> None:
    seq_length = 2048
    max_model_len = native_vllm_max_model_len(
        SimpleNamespace(seq_length=seq_length, vllm_max_model_len=0)
    )
    assert max_model_len == 2050

    engine = _Engine()
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=max_model_len,
        seed=1234,
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    requests = [
        {
            "row_index": 0,
            "query_ids": [10] * seq_length,
            "targets": [{"candidate_index": 0, "positions": [seq_length - 1], "token_ids": [20]}],
            "fast_mc": False,
        }
    ]
    candidate_results: list[list[dict | None]] = [[None]]

    inferencer.score_requests(requests, candidate_results)

    prompt_ids = engine.calls[0][0][0]["prompt_token_ids"]
    sampling = engine.calls[0][1]
    assert len(prompt_ids) == seq_length + 1
    assert sampling["max_tokens"] == 1
    assert len(prompt_ids) + sampling["max_tokens"] == max_model_len


def test_native_vllm_scoring_rejects_capacity_without_output_slot(tmp_path: Path) -> None:
    seq_length = 2048
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=seq_length + 1,
        seed=1234,
        engine=_Engine(),
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    requests = [
        {
            "row_index": 0,
            "query_ids": [10] * seq_length,
            "targets": [{"candidate_index": 0, "positions": [seq_length - 1], "token_ids": [20]}],
            "fast_mc": False,
        }
    ]

    with pytest.raises(ValueError, match=r"2049 \+ 1 = 2050 > 2049"):
        inferencer.score_requests(requests, [[None]])


def test_native_vllm_scores_requests_in_bounded_batches(tmp_path: Path) -> None:
    engine = _Engine()
    inferencer = NativeVLLMInferencer(
        model_path=tmp_path,
        max_model_len=32,
        seed=1234,
        max_batch_size=2,
        engine=engine,
        sampling_params_factory=lambda **kwargs: kwargs,
    )
    requests = [
        {
            "row_index": row_index,
            "query_ids": [10, 11, token_id],
            "targets": [{"candidate_index": 0, "positions": [1], "token_ids": [token_id]}],
            "fast_mc": False,
        }
        for row_index, token_id in enumerate((20, 22, 23))
    ]
    candidate_results: list[list[dict | None]] = [[None] for _ in requests]

    stats = inferencer.score_requests(requests, candidate_results)

    assert stats["forward_calls"] == 2
    assert [len(call[0]) for call in engine.calls] == [2, 1]
    assert all(result[0] is not None for result in candidate_results)


def test_core88_native_scoring_disables_single_token_fast_mc() -> None:
    rows = [
        {
            "task": "choice",
            "example_id": "one",
            "input": "Q",
            "choices": ["A", "B"],
            "metric": "acc",
        }
    ]

    requests, candidate_results = _prepare_scoring_requests(
        rows, _Tokenizer(), 32, enable_fast_mc=False
    )

    assert len(requests) == 2
    assert all(len(request["targets"]) == 1 for request in requests)
    assert candidate_results == [[None, None]]


def test_native_vllm_contract_requires_explicit_bounded_batch_opt_in() -> None:
    args = SimpleNamespace(
        hf_backend="native_vllm",
        allow_unverified_native_vllm=True,
        vllm_max_model_len=0,
        vllm_gpu_memory_utilization=0.85,
    )

    validate_native_vllm_args(args, batch_size=1, processes_per_gpu=1)
    validate_native_vllm_args(args, batch_size=8, processes_per_gpu=1)

    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        validate_native_vllm_args(args, batch_size=9, processes_per_gpu=1)
    with pytest.raises(ValueError, match="one engine process"):
        validate_native_vllm_args(args, batch_size=1, processes_per_gpu=4)
    args.allow_unverified_native_vllm = False
    with pytest.raises(ValueError, match="allow-unverified"):
        validate_native_vllm_args(args, batch_size=1, processes_per_gpu=1)


def _native_runtime_config() -> dict[str, object]:
    return {
        "architectures": ["ConceptLMV22VQForCausalLM"],
        "model_type": "conceptlm_v22_vq",
        "weight_key_format": "native_megatron_state_dict",
        "conceptlm_backbone": "olmo3",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_query_groups": 32,
        "ffn_hidden_size": 11008,
        "vocab_size": 100278,
        "max_sequence_length": 65536,
        "conceptlm_encoder_layers": 16,
        "conceptlm_decoder_layers": 16,
        "conceptlm_special_layers": 8,
        "conceptlm_chunk_size": 4,
        "conceptlm_shift_feature": True,
        "conceptlm_chunk_merge_method": "meanpooling",
        "conceptlm_layer_norm_option": "normed_add",
        "conceptlm_hlm_attention_mode": "backbone_window",
        "conceptlm_hlm_ffn_hidden_size": None,
        "window_size": [4096, 0],
        "window_attn_skip_freq": 4,
        "conceptlm_v22_vq_codebook_size": 128,
        "conceptlm_v22_vq_num_codebooks": 32,
        "conceptlm_v22_vq_merge_mode": "raw_logits",
        "conceptlm_v21_dd_self_dd_mode": "dd",
        "conceptlm_v21_dd_encoder_self_dd": True,
        "conceptlm_v21_dd_concept_self_dd": True,
        "conceptlm_v21_dd_two_route_add": True,
        "conceptlm_v21_enable_concept_read_encoder": True,
        "conceptlm_v21_enable_decoder_read_encoder": True,
        "conceptlm_v21_enable_decoder_read_concept": True,
        "conceptlm_v21_final_read_concept_gate": True,
    }


def test_native_vllm_overlay_links_weights_and_preserves_model_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}) + "\n", encoding="utf-8"
    )
    runtime_config = tmp_path / "runtime-config.json"
    runtime_config.write_text(json.dumps(_native_runtime_config()) + "\n", encoding="utf-8")

    overlay, manifest = prepare_native_vllm_model(
        source_model=source, runtime_config=runtime_config, overlay_dir=tmp_path / "overlay"
    )

    linked_weight = overlay / "model-00001-of-00001.safetensors"
    assert linked_weight.is_symlink()
    assert linked_weight.resolve() == (source / "model-00001-of-00001.safetensors")
    tokenizer_config = json.loads((overlay / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert manifest["weight_files_mutated"] is False
    assert manifest["tokenizer_config_compatibility_rewrite"] is True
    identity, runtime = _model_manifest_paths(
        SimpleNamespace(hf_model_path=str(overlay), model_identity_path=str(source))
    )
    assert identity == str(source.resolve())
    assert runtime == str(overlay.resolve())


def test_native_vllm_tokenizer_overlay_rejects_invalid_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(_native_runtime_config()) + "\n", encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}) + "\n", encoding="utf-8"
    )
    overlay, manifest = prepare_native_vllm_model(
        source_model=source, runtime_config=None, overlay_dir=tmp_path / "overlay"
    )
    manifest_path = overlay / "native_vllm_overlay_manifest.json"

    manifest["tokenizer_config_compatibility_rewrite"] = False
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid compatibility-overlay manifest"):
        _model_manifest_paths(
            SimpleNamespace(hf_model_path=str(overlay), model_identity_path=str(source))
        )

    manifest["tokenizer_config_compatibility_rewrite"] = True
    manifest["runtime_config"] = str(tmp_path / "wrong-config.json")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid compatibility-overlay manifest"):
        _model_manifest_paths(
            SimpleNamespace(hf_model_path=str(overlay), model_identity_path=str(source))
        )


def test_native_vllm_auto_overlays_forward_tokenizer_without_runtime_config(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(_native_runtime_config()) + "\n", encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}) + "\n", encoding="utf-8"
    )

    overlay, manifest = prepare_native_vllm_model(
        source_model=source, runtime_config=None, overlay_dir=tmp_path / "overlay"
    )

    assert overlay != source
    assert manifest["mode"] == "overlay_tokenizer_compatibility"
    assert manifest["runtime_config"] == str((source / "config.json").resolve())
    assert manifest["tokenizer_config_compatibility_rewrite"] is True
    assert (overlay / "model.safetensors").resolve() == (source / "model.safetensors").resolve()
    tokenizer_config = json.loads((overlay / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    identity, runtime = _model_manifest_paths(
        SimpleNamespace(hf_model_path=str(overlay), model_identity_path=str(source))
    )
    assert identity == str(source.resolve())
    assert runtime == str(overlay.resolve())


def test_native_vllm_overlay_can_resume_from_symlinked_source(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    weight = model / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"weights")
    readme = model / "README.md"
    readme.write_text("model card\n", encoding="utf-8")

    source = tmp_path / "source-overlay"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / weight.name).symlink_to(weight)
    (source / readme.name).symlink_to(readme)
    runtime_config = tmp_path / "runtime-config.json"
    runtime_config.write_text(json.dumps(_native_runtime_config()) + "\n", encoding="utf-8")
    destination = tmp_path / "runtime-overlay"

    prepare_native_vllm_model(
        source_model=source, runtime_config=runtime_config, overlay_dir=destination
    )
    prepare_native_vllm_model(
        source_model=source, runtime_config=runtime_config, overlay_dir=destination
    )

    assert (destination / weight.name).resolve() == weight.resolve()
    assert (destination / readme.name).resolve() == readme.resolve()
