#!/usr/bin/env python3
"""Versioned RULER, LongBench v2, and HELMET evaluation contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "conceptlm-long-context-eval-v1"
PREPARED_STATUS = "LONG_CONTEXT_DATA_READY"
PREDICTION_STATUS = "LONG_CONTEXT_PREDICTION_OK"
SHARD_STATUS = "LONG_CONTEXT_INFERENCE_SHARD_OK"
SCORE_STATUS = "LONG_CONTEXT_SCORE_OK"

RULER_BENCHMARK = "ruler"
LONGBENCH_V2_BENCHMARK = "longbench_v2"
HELMET_BENCHMARK = "helmet"
SUPPORTED_BENCHMARKS = (RULER_BENCHMARK, LONGBENCH_V2_BENCHMARK, HELMET_BENCHMARK)
SUPPORTED_BACKENDS = ("hf", "native_megatron", "native_vllm", "lmdeploy")
MODEL_CONTEXT_FIELDS = ("max_sequence_length", "max_position_embeddings")

RULER_OFFICIAL_REPOSITORY = "https://github.com/allenai/olmes"
RULER_OFFICIAL_COMMIT = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
RULER_OFFICIAL_TASK_CONFIG_SHA256 = (
    "a68c6e202e1e9f299831331b8dfef421393be7693236e11dab98108a6cbebc00"
)
RULER_OFFICIAL_TASK_IMPL_SHA256 = "81aa9db5fe4933f50a28ae18eeff377223c2990c0d13592c57b9b6287979b8b6"
RULER_OFFICIAL_DATA_LOADER_SHA256 = (
    "85936be625fc5d82b35c867c1bb6a6c9159a39dc6aa38883ae869622074f8213"
)
RULER_DATASET = "allenai/ruler_data"
RULER_DATASET_REVISION = "b2d1d948dc4f6ca37c0135123f3b3afbed741130"
RULER_DATA_ARCHIVE = "data_100_samples.tgz"
RULER_DATA_ARCHIVE_SIZE_BYTES = 622128854
RULER_DATA_ARCHIVE_SHA256 = "a8da67ff3a7e6866a8012b81c79084f763111be54bd3a44e86eb18a377eb53c3"
RULER_DATA_FILES_CONTRACT_SHA256 = (
    "d8560982cde0e016b40a801ac8f536c74b725bb902185176a4d70da73fd64262"
)
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_MATCH_TYPES = {"niah": "all", "vt": "all", "cwe": "all", "fwe": "all", "qa": "part"}
RULER_SEQUENCE_LENGTHS = (4096, 8192, 16384, 32768, 65536)
RULER_EXAMPLES_PER_TASK = 100
RULER_QA_PROMPT_CONTRACT = "official-full-input-v1"
RULER_DATA_GENERATION_CONTRACT = "allenai-olmes-ruler-data-100-samples-b2d1d94-v1"
RULER_SOURCE_DATA_KIND = "allenai_olmes_ruler_data_100_samples"
RULER_MAX_NEW_TOKENS_BY_TASK = {
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
RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH = {
    str(sequence_length): {
        **RULER_MAX_NEW_TOKENS_BY_TASK,
        "niah_multivalue": 300 if sequence_length == 4096 else 250,
    }
    for sequence_length in RULER_SEQUENCE_LENGTHS
}
RULER_MAX_NEW_TOKENS = max(
    value
    for task_budgets in RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH.values()
    for value in task_budgets.values()
)
RULER_TEMPERATURE = 0.0
RULER_TOP_P = 1.0

LONGBENCH_V2_REPOSITORY = "https://github.com/THUDM/LongBench"
LONGBENCH_V2_OFFICIAL_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONGBENCH_V2_DATASET = "zai-org/LongBench-v2"
LONGBENCH_V2_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONGBENCH_V2_SPLIT = "train"
LONGBENCH_V2_EXAMPLE_COUNT = 503
LONGBENCH_V2_MAX_NEW_TOKENS = 128
LONGBENCH_V2_TEMPERATURE = 0.1
LONGBENCH_V2_TOP_P = 1.0
LONGBENCH_V2_PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\"."""
LONGBENCH_V2_PROMPT_SHA256 = "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950"

HELMET_OFFICIAL_REPOSITORY = "https://github.com/princeton-nlp/HELMET"
HELMET_OFFICIAL_COMMIT = "af609c4d51b97fc35012099380aa889da961c42d"
HELMET_DATASET_REPOSITORY = "https://huggingface.co/datasets/princeton-nlp/HELMET"
HELMET_CLASSIC_DATASET_REVISION = "bc560a6b8165c696ad4bc3d1612c64b5794ba328"
HELMET_CLASSIC_DATA_ARCHIVE_SHA256 = (
    "9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9"
)
HELMET_SEED = 42
HELMET_INPUT_MAX_LENGTH = 65536
HELMET_SHORT_INPUT_LENGTHS = (8192, 16384, 32768, 65536)
# The pinned loader applies ``max_test_samples`` to unique query keys for RAG
# and MS MARCO, then evaluates every row matching those keys.  Those files
# contain repeated keys, so the official eval.py consumes more rows than the
# naive sum of the per-config maxima (4,100).  Do not post-hoc truncate them.
HELMET_FORMAL_EXAMPLE_COUNT = 5823
HELMET_CATEGORY_COUNTS = {
    "Recall": 400,
    "RAG": 2100,
    "Re-rank": 123,
    "Cite": 200,
    "LongQA": 300,
    "Summ": 200,
    "ICL": 2500,
}
HELMET_SHORT_FORMAL_EXAMPLE_COUNT = HELMET_FORMAL_EXAMPLE_COUNT * len(HELMET_SHORT_INPUT_LENGTHS)
HELMET_SHORT_CATEGORY_COUNTS = {
    category: count * len(HELMET_SHORT_INPUT_LENGTHS)
    for category, count in HELMET_CATEGORY_COUNTS.items()
}
HELMET_SWEEP_INPUT_LENGTHS = HELMET_SHORT_INPUT_LENGTHS
HELMET_SWEEP_FORMAL_EXAMPLE_COUNT = HELMET_FORMAL_EXAMPLE_COUNT * len(HELMET_SWEEP_INPUT_LENGTHS)
HELMET_SWEEP_CATEGORY_COUNTS = {
    category: count * len(HELMET_SWEEP_INPUT_LENGTHS)
    for category, count in HELMET_CATEGORY_COUNTS.items()
}
HELMET_MAX_NEW_TOKENS_BY_TASK = {
    "ruler_niah_mk_2": 50,
    "ruler_niah_mk_3": 100,
    "ruler_niah_mv": 50,
    "json_kv": 100,
    "kilt_nq": 20,
    "kilt_triviaqa": 20,
    "kilt_hotpotqa": 20,
    "kilt_popqa_3": 20,
    "msmarco_rerank_psg": 200,
    "alce_asqa_700": 300,
    "alce_qampari_700": 300,
    "narrativeqa_130772": 100,
    "infbench_qa_eng_130862": 10,
    "infbench_choice_eng_130862": 10,
    "infbench_sum_eng_129672": 1200,
    "multi_lexsum_130372": 400,
    "icl_trec_coarse_6600shot_balance": 20,
    "icl_trec_fine_6400shot_balance": 20,
    "icl_banking77_5900shot_balance": 20,
    "icl_clinic150_7050shot_balance": 20,
    "icl_nlu_8296shot_balance": 20,
}
HELMET_NEWLINE_STOP_TASKS = frozenset(
    {
        "kilt_nq",
        "kilt_triviaqa",
        "kilt_hotpotqa",
        "kilt_popqa_3",
        "msmarco_rerank_psg",
        "icl_trec_coarse_6600shot_balance",
        "icl_trec_fine_6400shot_balance",
        "icl_banking77_5900shot_balance",
        "icl_clinic150_7050shot_balance",
        "icl_nlu_8296shot_balance",
    }
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash JSON with stable key ordering and separators."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_artifact_contract(path: Path) -> dict[str, Any]:
    """Fingerprint tokenizer metadata without depending on its directory name."""

    root = path.resolve()
    files = {name: file_sha256(root / name) for name in TOKENIZER_FILES if (root / name).is_file()}
    if not files:
        raise FileNotFoundError(f"no supported tokenizer artifacts under {root}")
    return {"path": str(root), "files": files, "contract_sha256": canonical_json_sha256(files)}


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON list or a JSONL file into object rows."""

    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return list(payload)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write a pretty-printed JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prompt_sha256(prompt: str) -> str:
    """Hash one exact rendered prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def render_longbench_v2_prompt(row: dict[str, Any]) -> str:
    """Render the pinned official LongBench v2 zero-shot prompt."""

    required = ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"LongBench v2 row is missing fields: {missing}")
    return (
        LONGBENCH_V2_PROMPT.replace("$DOC$", str(row["context"]).strip())
        .replace("$Q$", str(row["question"]).strip())
        .replace("$C_A$", str(row["choice_A"]).strip())
        .replace("$C_B$", str(row["choice_B"]).strip())
        .replace("$C_C$", str(row["choice_C"]).strip())
        .replace("$C_D$", str(row["choice_D"]).strip())
    )


def truncate_head_tail(
    prompt: str, tokenizer: Any, max_input_tokens: int
) -> tuple[str, dict[str, Any]]:
    """Apply LongBench v2's tokenizer-aware head-plus-tail truncation."""

    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    token_ids = [int(value) for value in tokenizer.encode(prompt)]
    original_count = len(token_ids)
    if original_count <= max_input_tokens:
        return prompt, {
            "strategy": "none",
            "original_token_count": original_count,
            "prepared_token_count": original_count,
            "max_input_tokens": max_input_tokens,
        }
    selected_budget = max_input_tokens
    while True:
        head_count = selected_budget // 2
        tail_count = selected_budget - head_count
        selected = token_ids[:head_count] + token_ids[-tail_count:]
        truncated = tokenizer.decode(selected, skip_special_tokens=True)
        prepared_count = len(tokenizer.encode(truncated))
        if prepared_count <= max_input_tokens:
            break
        selected_budget -= max(1, prepared_count - max_input_tokens)
        if selected_budget <= 0:
            raise RuntimeError("could not fit decoded LongBench v2 prompt in token budget")
    return truncated, {
        "strategy": "official_head_tail",
        "original_token_count": original_count,
        "selected_token_count": len(selected),
        "roundtrip_trimmed_token_count": max_input_tokens - len(selected),
        "prepared_token_count": prepared_count,
        "max_input_tokens": max_input_tokens,
    }


def apply_prompt_transport(prompt: str, tokenizer: Any, prompt_transport: str) -> str:
    """Apply an explicit raw or tokenizer chat-template transport."""

    if prompt_transport == "raw":
        return prompt
    if prompt_transport != "chat_template":
        raise ValueError(f"unsupported prompt transport: {prompt_transport}")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError("chat_template transport requires tokenizer.chat_template")
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    )


def prepare_longbench_v2_rows(
    source_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_input_tokens: int,
    prompt_transport: str = "raw",
) -> list[dict[str, Any]]:
    """Normalize LongBench v2 rows into the shared inference contract."""

    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        example_id = str(row.get("_id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"invalid or duplicate LongBench v2 _id: {example_id!r}")
        seen.add(example_id)
        answer = str(row.get("answer", ""))
        if answer not in "ABCD" or len(answer) != 1:
            raise ValueError(f"invalid LongBench v2 answer for {example_id}: {answer!r}")
        rendered = render_longbench_v2_prompt(row)
        truncated, truncation = truncate_head_tail(rendered, tokenizer, max_input_tokens)
        prompt = apply_prompt_transport(truncated, tokenizer, prompt_transport)
        truncation["final_transport_token_count"] = len(tokenizer.encode(prompt))
        prepared.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": LONGBENCH_V2_BENCHMARK,
                "example_id": example_id,
                "task": str(row.get("domain", "unknown")),
                "prompt": prompt,
                "prompt_sha256": prompt_sha256(prompt),
                "expected_answer": answer,
                "metadata": {
                    "domain": str(row.get("domain", "unknown")),
                    "sub_domain": str(row.get("sub_domain", "unknown")),
                    "difficulty": str(row.get("difficulty", "unknown")),
                    "length": str(row.get("length", "unknown")),
                    "prompt_transport": prompt_transport,
                    "truncation": truncation,
                },
            }
        )
    return prepared


def ruler_match_type(task: str) -> str:
    """Return the OLMES RULER match type for one task."""

    short_name = task.split("_", 1)[0]
    try:
        return RULER_MATCH_TYPES[short_name]
    except KeyError as error:
        raise ValueError(f"unsupported RULER task: {task}") from error


def ruler_max_new_tokens(task: str, sequence_length: int) -> int:
    """Return OLMES' exact generation budget for one task and length."""

    try:
        return int(RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH[str(sequence_length)][task])
    except KeyError as error:
        raise ValueError(
            f"unsupported OLMES RULER task/length: task={task} length={sequence_length}"
        ) from error


def prepare_ruler_rows(
    setup_root: Path,
    tokenizer: Any,
    *,
    sequence_length: int,
    prompt_transport: str = "raw",
    tasks: Sequence[str] = RULER_TASKS,
    data_generation_contract: str | None = None,
) -> list[dict[str, Any]]:
    """Import fixed AllenAI/OLMES RULER JSONL files."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        data_path = ruler_source_path(setup_root, task, sequence_length)
        if not data_path.is_file():
            raise FileNotFoundError(f"missing prepared RULER task: {data_path}")
        for row_number, row in enumerate(read_json_or_jsonl(data_path), start=1):
            source_index = str(row.get("index", row_number - 1))
            # The official generator samples source indices with replacement, so
            # ``index`` is provenance rather than a unique row identifier.  The
            # JSONL row order is deterministic for a fixed seed and therefore is
            # the stable identity used by inference/resume/scoring.
            row_index = row_number - 1
            example_id = f"{sequence_length}:{task}:{row_index}"
            if example_id in seen:
                raise ValueError(f"duplicate RULER example ID: {example_id}")
            seen.add(example_id)
            expected = row.get("expected_answer", row.get("outputs", row.get("answer")))
            if isinstance(expected, str):
                expected = [expected]
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"RULER row has no expected answers: {example_id}")
            # Official RULER QA rows carry both a short ``question`` field and
            # the actual model input in ``input``.  The latter contains the
            # long evidence documents, task instruction, question, and answer
            # cue.  Prefer it whenever present; selecting ``question`` first
            # silently turns qa_1/qa_2 into closed-book QA at every length.
            source_prompt = row.get("input")
            prompt_source = "input"
            if source_prompt is None:
                source_prompt = row.get("question", "")
                prompt_source = "question"
            answer_prefix = row.get("answer_prefix")
            if answer_prefix:
                source_prompt = f"{source_prompt}{answer_prefix}"
                prompt_source = f"{prompt_source}+answer_prefix"
            prompt = apply_prompt_transport(str(source_prompt), tokenizer, prompt_transport)
            if not prompt:
                raise ValueError(f"RULER row has no prompt: {example_id}")
            prompt_token_count = len(tokenizer.encode(prompt))
            if prompt_token_count > sequence_length:
                raise ValueError(
                    f"RULER prompt exceeds prepared length for {example_id}: "
                    f"{prompt_token_count} > {sequence_length}"
                )
            prepared.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": RULER_BENCHMARK,
                    "example_id": example_id,
                    "task": task,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256(prompt),
                    "expected_answer": [str(value) for value in expected],
                    "metadata": {
                        "sequence_length": sequence_length,
                        "source_index": source_index,
                        "source_row_index": row_index,
                        "source_length": row.get("length"),
                        "prompt_token_count": prompt_token_count,
                        "prompt_transport": prompt_transport,
                        "prompt_source": prompt_source,
                        "match_type": ruler_match_type(task),
                        "max_new_tokens": ruler_max_new_tokens(task, sequence_length),
                        "data_generation_contract": data_generation_contract,
                    },
                }
            )
    return prepared


def ruler_source_path(setup_root: Path, task: str, sequence_length: int) -> Path:
    """Resolve either NeMo Skills or AllenAI/OLMo prepared RULER layout."""

    nemo_path = setup_root / task / "test.jsonl"
    allenai_path = setup_root / task / f"validation_{sequence_length}.jsonl"
    candidates = [path for path in (nemo_path, allenai_path) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "expected exactly one prepared RULER source for "
            f"task={task} length={sequence_length}: {nemo_path}, {allenai_path}"
        )
    return candidates[0]


def benchmark_sampling_contract(benchmark: str) -> dict[str, Any]:
    """Return pinned generation defaults for one benchmark."""

    if benchmark == RULER_BENCHMARK:
        return {
            "max_new_tokens": RULER_MAX_NEW_TOKENS,
            "max_new_tokens_by_task": dict(RULER_MAX_NEW_TOKENS_BY_TASK),
            "max_new_tokens_by_sequence_length": {
                length: dict(task_budgets)
                for length, task_budgets in RULER_MAX_NEW_TOKENS_BY_SEQUENCE_LENGTH.items()
            },
            "eos_stopping": True,
            "temperature": RULER_TEMPERATURE,
            "top_p": RULER_TOP_P,
            "stop": [],
        }
    if benchmark == LONGBENCH_V2_BENCHMARK:
        return {
            "max_new_tokens": LONGBENCH_V2_MAX_NEW_TOKENS,
            "temperature": LONGBENCH_V2_TEMPERATURE,
            "top_p": LONGBENCH_V2_TOP_P,
            "stop": [],
        }
    if benchmark == HELMET_BENCHMARK:
        return {
            "max_new_tokens": max(HELMET_MAX_NEW_TOKENS_BY_TASK.values()),
            "max_new_tokens_by_task": dict(HELMET_MAX_NEW_TOKENS_BY_TASK),
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [],
            "stop_by_task": {task: ["\n", "\n\n"] for task in sorted(HELMET_NEWLINE_STOP_TASKS)},
        }
    raise ValueError(f"unsupported benchmark: {benchmark}")


def prepared_official_sampling_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the official sampling contract for one sealed prepared dataset.

    HELMET's pinned ``*_short.yaml`` files use length-specific task IDs (for
    example ``narrativeqa_7892`` through ``narrativeqa_65236``), while the
    upstream full profile uses IDs such as ``narrativeqa_130772``.  The
    prepared manifest seals the exact official profile and its task budgets,
    so comparing a short run against the full-profile task map is invalid.
    ``validate_prepared_dataset`` verifies this manifest contract before this
    helper is used.
    """

    benchmark = str(manifest.get("benchmark", ""))
    contract = benchmark_sampling_contract(benchmark)
    if benchmark != HELMET_BENCHMARK:
        return contract
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("prepared HELMET protocol is missing")
    for key in (
        "max_new_tokens",
        "max_new_tokens_by_task",
        "temperature",
        "top_p",
        "stop",
        "stop_by_task",
    ):
        if key not in protocol:
            raise ValueError(f"prepared HELMET sampling contract is missing: {key}")
        contract[key] = protocol[key]
    return contract


def validate_prepared_helmet_manifest_contract(manifest: dict[str, Any]) -> None:
    """Validate HELMET's profile-specific task map without reading large prompts."""

    protocol = manifest.get("protocol")
    source_data = manifest.get("source_data")
    if not isinstance(protocol, dict):
        raise ValueError("prepared HELMET protocol is missing")
    if not isinstance(source_data, dict):
        raise ValueError("prepared HELMET source provenance is missing")
    profile = protocol.get("helmet_profile")
    if profile not in {"short", "8k-64k"} or source_data.get("profile") != profile:
        raise ValueError("prepared HELMET profile mismatch")
    if int(protocol.get("seed", -1)) != HELMET_SEED:
        raise ValueError("prepared HELMET seed mismatch")

    datasets = source_data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("prepared HELMET profile has no dataset contracts")
    dataset_budgets: dict[str, int] = {}
    dataset_stops: dict[str, list[str]] = {}
    dataset_lengths: set[int] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("prepared HELMET dataset contract is invalid")
        task = str(dataset.get("dataset", ""))
        budget = dataset.get("generation_max_length")
        input_length = dataset.get("input_max_length")
        if (
            not task
            or not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget <= 0
            or not isinstance(input_length, int)
            or isinstance(input_length, bool)
            or input_length <= 0
        ):
            raise ValueError("prepared HELMET dataset task/length/budget is invalid")
        previous_budget = dataset_budgets.setdefault(task, budget)
        if previous_budget != budget:
            raise ValueError(f"prepared HELMET dataset budget differs by length: {task}")
        stops = ["\n", "\n\n"] if dataset.get("stop_new_line") is True else []
        previous_stops = dataset_stops.setdefault(task, stops)
        if previous_stops != stops:
            raise ValueError(f"prepared HELMET dataset stops differ by length: {task}")
        dataset_lengths.add(input_length)

    declared_lengths = tuple(int(value) for value in protocol.get("input_lengths", ()))
    if tuple(sorted(dataset_lengths)) != declared_lengths:
        raise ValueError("prepared HELMET dataset lengths do not match its profile")
    if tuple(int(value) for value in source_data.get("input_lengths", ())) != declared_lengths:
        raise ValueError("prepared HELMET source/profile lengths mismatch")

    raw_budgets = protocol.get("max_new_tokens_by_task")
    if not isinstance(raw_budgets, dict):
        raise ValueError("prepared HELMET task budget map is missing")
    protocol_budgets = {str(task): int(value) for task, value in raw_budgets.items()}
    if protocol_budgets != dataset_budgets:
        raise ValueError("prepared HELMET task budget map does not match its profile")
    if int(protocol.get("max_new_tokens", -1)) != max(dataset_budgets.values()):
        raise ValueError("prepared HELMET maximum generation budget mismatch")
    raw_stops = protocol.get("stop_by_task")
    if not isinstance(raw_stops, dict):
        raise ValueError("prepared HELMET task stop map is missing")
    protocol_stops = {
        str(task): [str(value) for value in values] for task, values in raw_stops.items()
    }
    expected_stops = {task: stops for task, stops in dataset_stops.items() if stops}
    if protocol_stops != expected_stops:
        raise ValueError("prepared HELMET task stop map does not match its profile")
    if (
        float(protocol.get("temperature", -1.0)) != 0.0
        or float(protocol.get("top_p", -1.0)) != 1.0
        or list(protocol.get("stop", []))
    ):
        raise ValueError("prepared HELMET sampling parameters mismatch")


def official_source_contract(benchmark: str) -> dict[str, Any]:
    """Return immutable upstream provenance for one benchmark."""

    if benchmark == RULER_BENCHMARK:
        return {
            "repository": RULER_OFFICIAL_REPOSITORY,
            "commit": RULER_OFFICIAL_COMMIT,
            "data_generation_contract": RULER_DATA_GENERATION_CONTRACT,
            "task_config_sha256": RULER_OFFICIAL_TASK_CONFIG_SHA256,
            "task_impl_sha256": RULER_OFFICIAL_TASK_IMPL_SHA256,
            "data_loader_sha256": RULER_OFFICIAL_DATA_LOADER_SHA256,
            "dataset": RULER_DATASET,
            "dataset_revision": RULER_DATASET_REVISION,
            "data_archive": RULER_DATA_ARCHIVE,
            "data_archive_size_bytes": RULER_DATA_ARCHIVE_SIZE_BYTES,
            "data_archive_sha256": RULER_DATA_ARCHIVE_SHA256,
            "data_files_contract_sha256": RULER_DATA_FILES_CONTRACT_SHA256,
        }
    if benchmark == LONGBENCH_V2_BENCHMARK:
        return {
            "repository": LONGBENCH_V2_REPOSITORY,
            "commit": LONGBENCH_V2_OFFICIAL_COMMIT,
            "dataset": LONGBENCH_V2_DATASET,
            "dataset_revision": LONGBENCH_V2_DATASET_REVISION,
            "split": LONGBENCH_V2_SPLIT,
            "prompt_sha256": LONGBENCH_V2_PROMPT_SHA256,
        }
    if benchmark == HELMET_BENCHMARK:
        return {
            "repository": HELMET_OFFICIAL_REPOSITORY,
            "commit": HELMET_OFFICIAL_COMMIT,
            "dataset_repository": HELMET_DATASET_REPOSITORY,
            "classic_dataset_revision": HELMET_CLASSIC_DATASET_REVISION,
            "classic_data_archive_sha256": HELMET_CLASSIC_DATA_ARCHIVE_SHA256,
        }
    raise ValueError(f"unsupported benchmark: {benchmark}")


def build_prepared_manifest(
    *,
    benchmark: str,
    rows: Sequence[dict[str, Any]],
    tokenizer_contract: dict[str, Any],
    source_data: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Build a manifest that seals exact prepared examples and protocol."""

    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prepared example IDs are not unique")
    prompt_hashes = [str(row["prompt_sha256"]) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        "benchmark": benchmark,
        "example_count": len(rows),
        "example_ids_sha256": canonical_json_sha256(ids),
        "prompt_hashes_sha256": canonical_json_sha256(prompt_hashes),
        "tokenizer": tokenizer_contract,
        "source_data": source_data,
        "official_source": official_source_contract(benchmark),
        "protocol": protocol,
    }


def validate_prepared_dataset(data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and verify one sealed long-context prepared dataset."""

    manifest_path = data_root / "manifest.json"
    inputs_path = data_root / "inputs.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_json_or_jsonl(inputs_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("prepared manifest schema version mismatch")
    if manifest.get("status") != PREPARED_STATUS:
        raise ValueError("prepared manifest status mismatch")
    if int(manifest.get("example_count", -1)) != len(rows):
        raise ValueError("prepared example count mismatch")
    if manifest.get("inputs_jsonl_sha256") != file_sha256(inputs_path):
        raise ValueError("prepared inputs file hash mismatch")
    ids = [str(row.get("example_id")) for row in rows]
    hashes = [str(row.get("prompt_sha256")) for row in rows]
    if manifest.get("example_ids_sha256") != canonical_json_sha256(ids):
        raise ValueError("prepared example ID hash mismatch")
    if manifest.get("prompt_hashes_sha256") != canonical_json_sha256(hashes):
        raise ValueError("prepared prompt hash mismatch")
    if len(ids) != len(set(ids)):
        raise ValueError("prepared example IDs are duplicated")
    for row in rows:
        if row.get("benchmark") != manifest.get("benchmark"):
            raise ValueError("prepared row benchmark mismatch")
        if prompt_sha256(str(row.get("prompt", ""))) != row.get("prompt_sha256"):
            raise ValueError(f"prepared prompt changed: {row.get('example_id')}")
    benchmark = manifest.get("benchmark")
    protocol = manifest.get("protocol")
    if manifest.get("benchmark") == RULER_BENCHMARK:
        if (
            not isinstance(protocol, dict)
            or protocol.get("qa_prompt_contract") != RULER_QA_PROMPT_CONTRACT
        ):
            raise ValueError("prepared RULER QA prompt contract mismatch")
        for row in rows:
            if row.get("task") not in {"qa_1", "qa_2"}:
                continue
            metadata = row.get("metadata")
            prompt_source = metadata.get("prompt_source") if isinstance(metadata, dict) else None
            if prompt_source not in {"input", "input+answer_prefix"}:
                raise ValueError(
                    "prepared RULER QA row does not use the official full input: "
                    f"{row.get('example_id')}"
                )
        if protocol.get("data_generation_contract") != RULER_DATA_GENERATION_CONTRACT:
            raise ValueError(
                "prepared RULER data contract mismatch; tokenizer-generated or legacy "
                "RULER caches are not OLMES-compatible"
            )
        source_data = manifest.get("source_data")
        if not isinstance(source_data, dict):
            raise ValueError("prepared RULER source provenance is missing")
        expected_source_fields = {
            "kind": RULER_SOURCE_DATA_KIND,
            "dataset": RULER_DATASET,
            "dataset_revision": RULER_DATASET_REVISION,
            "archive_sha256": RULER_DATA_ARCHIVE_SHA256,
            "files_contract_sha256": RULER_DATA_FILES_CONTRACT_SHA256,
        }
        for key, expected in expected_source_fields.items():
            if source_data.get(key) != expected:
                raise ValueError(f"prepared RULER OLMES source mismatch: {key}")
        official_sampling = benchmark_sampling_contract(RULER_BENCHMARK)
        for key in (
            "max_new_tokens",
            "max_new_tokens_by_task",
            "max_new_tokens_by_sequence_length",
            "eos_stopping",
            "temperature",
            "top_p",
            "stop",
        ):
            if protocol.get(key) != official_sampling[key]:
                raise ValueError(f"prepared RULER sampling contract mismatch: {key}")
        for row in rows:
            metadata = row.get("metadata")
            row_contract = (
                metadata.get("data_generation_contract") if isinstance(metadata, dict) else None
            )
            if row_contract != RULER_DATA_GENERATION_CONTRACT:
                raise ValueError(
                    "prepared RULER row lacks fixed OLMES provenance: " f"{row.get('example_id')}"
                )
            sequence_length = int(metadata.get("sequence_length", 0))
            expected_budget = (
                ruler_max_new_tokens(str(row.get("task")), sequence_length)
                if sequence_length in RULER_SEQUENCE_LENGTHS
                else None
            )
            if expected_budget is not None and metadata.get("max_new_tokens") != expected_budget:
                raise ValueError(
                    "prepared RULER row generation budget mismatch: " f"{row.get('example_id')}"
                )
    if benchmark in {RULER_BENCHMARK, HELMET_BENCHMARK}:
        if not isinstance(protocol, dict):
            raise ValueError("prepared long-context protocol is missing")
        field = "sequence_lengths" if benchmark == RULER_BENCHMARK else "input_lengths"
        allowed = (
            RULER_SEQUENCE_LENGTHS if benchmark == RULER_BENCHMARK else HELMET_SWEEP_INPUT_LENGTHS
        )
        declared = tuple(int(value) for value in protocol.get(field, ()))
        if not declared or declared != tuple(sorted(set(declared))):
            raise ValueError(f"prepared {benchmark} {field} must be unique and increasing")
        unsupported = tuple(value for value in declared if value not in allowed)
        if unsupported:
            raise ValueError(
                f"prepared {benchmark} lengths exceed the current 64K protocol: "
                f"{unsupported}; allowed={allowed}"
            )
        metadata_field = "sequence_length" if benchmark == RULER_BENCHMARK else "input_max_length"
        observed = tuple(
            sorted(
                {
                    int(row["metadata"][metadata_field])
                    for row in rows
                    if isinstance(row.get("metadata"), dict) and metadata_field in row["metadata"]
                }
            )
        )
        if observed != declared:
            raise ValueError(
                f"prepared {benchmark} row lengths do not match manifest: "
                f"{observed} != {declared}"
            )
    if benchmark == HELMET_BENCHMARK:
        validate_prepared_helmet_manifest_contract(manifest)

        row_budgets: dict[str, int] = {}
        row_stops: dict[str, list[str]] = {}
        for row in rows:
            task = str(row.get("task", ""))
            metadata = row.get("metadata")
            if not task or not isinstance(metadata, dict):
                raise ValueError("prepared HELMET row lacks task metadata")
            budget = metadata.get("max_new_tokens")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ValueError(f"prepared HELMET row has invalid generation budget: {task}")
            previous_budget = row_budgets.setdefault(task, budget)
            if previous_budget != budget:
                raise ValueError(f"prepared HELMET task budget differs across rows: {task}")
            stops = [str(value) for value in metadata.get("stop", [])]
            previous_stops = row_stops.setdefault(task, stops)
            if previous_stops != stops:
                raise ValueError(f"prepared HELMET task stops differ across rows: {task}")

        assert isinstance(protocol, dict)
        raw_budgets = protocol["max_new_tokens_by_task"]
        assert isinstance(raw_budgets, dict)
        protocol_budgets = {str(task): int(value) for task, value in raw_budgets.items()}
        if protocol_budgets != row_budgets:
            raise ValueError("prepared HELMET task budget map does not match its rows")
        if int(protocol.get("max_new_tokens", -1)) != max(row_budgets.values()):
            raise ValueError("prepared HELMET maximum generation budget mismatch")
        raw_stops = protocol["stop_by_task"]
        assert isinstance(raw_stops, dict)
        protocol_stops = {
            str(task): [str(value) for value in values] for task, values in raw_stops.items()
        }
        expected_stops = {task: stops for task, stops in row_stops.items() if stops}
        if protocol_stops != expected_stops:
            raise ValueError("prepared HELMET task stop map does not match its rows")
    return manifest, rows


def stable_sample_seed(global_seed: int, example_id: str, sample_index: int) -> int:
    """Derive a batch-order-independent 31-bit sampling seed."""

    digest = hashlib.sha256(
        f"{int(global_seed)}\0{example_id}\0{int(sample_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def validate_backend_shape(backend: str, batch_size: int, samples_per_example: int) -> None:
    """Enforce the validated batching and repeated-sampling boundary."""

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if batch_size <= 0 or samples_per_example <= 0:
        raise ValueError("batch_size and samples_per_example must be positive")
    if backend in {"hf", "native_megatron"}:
        if batch_size != 1 or samples_per_example != 1:
            raise ValueError(f"{backend} requires batch_size=1 and samples_per_example=1")
        return
    if batch_size > 8:
        raise ValueError(f"{backend} batch_size must be in [1, 8]")


def model_context_capability(model_config_path: Path) -> dict[str, Any]:
    """Read a model's native context contract without guessing from its name.

    The two supported config fields must agree when both are present.  This is
    intentionally strict: silently choosing the larger field could make an
    evaluator run beyond the context window used by another backend.
    """

    config_path = model_config_path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"model config is not a JSON object: {config_path}")
    fields: dict[str, int] = {}
    for name in MODEL_CONTEXT_FIELDS:
        value = payload.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid {name} in {config_path}: {value!r}")
        fields[name] = int(value)
    if not fields:
        raise ValueError(
            f"model config has none of the context fields {MODEL_CONTEXT_FIELDS}: " f"{config_path}"
        )
    unique_lengths = set(fields.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"model context fields disagree in {config_path}: {fields}")
    native_context_length = unique_lengths.pop()
    return {
        "model_config_path": str(config_path),
        "model_config_sha256": file_sha256(config_path),
        "native_context_length": native_context_length,
        "context_fields": fields,
        "position_embedding_type": payload.get("position_embedding_type"),
        "rope_scaling": payload.get("rope_scaling"),
        "rope_theta": payload.get("rope_theta"),
        "yarn_original_max_position_embeddings": payload.get(
            "yarn_original_max_position_embeddings"
        ),
        "yarn_rotary_scaling_factor": payload.get("yarn_rotary_scaling_factor"),
        "window_size": payload.get("window_size"),
        "window_attn_skip_freq": payload.get("window_attn_skip_freq"),
    }


def resolve_model_context_contract(
    *,
    capability: dict[str, Any],
    required_context_length: int,
    requested_context_length: int = 0,
    allow_context_extension: bool = False,
    max_generation_overhang: int = 0,
) -> dict[str, Any]:
    """Resolve and validate the exact context length used by every backend."""

    required = int(required_context_length)
    requested = int(requested_context_length)
    native = int(capability["native_context_length"])
    overhang_limit = int(max_generation_overhang)
    if required <= 0 or requested < 0 or overhang_limit < 0:
        raise ValueError("context lengths must be positive (or zero for auto)")
    runtime = requested or required
    if runtime < required:
        raise ValueError(
            "requested model context is smaller than prompt plus generation: "
            f"{runtime} < {required}"
        )
    extrapolating = runtime > native
    generation_overhang = max(0, runtime - native)
    bounded_generation_overhang = bool(
        extrapolating
        and runtime == required
        and overhang_limit > 0
        and generation_overhang <= overhang_limit
    )
    if extrapolating and not allow_context_extension and not bounded_generation_overhang:
        raise ValueError(
            "requested model context exceeds the model's native declared context: "
            f"{runtime} > {native}; use a model artifact whose config explicitly "
            "declares the extended context or explicitly allow an unvalidated "
            "context extrapolation"
        )
    return {
        **capability,
        "required_context_length": required,
        "requested_context_length": requested,
        "runtime_context_length": runtime,
        "allow_context_extension": bool(allow_context_extension),
        "uses_context_extrapolation": extrapolating,
        "generation_overhang_tokens": generation_overhang,
        "max_generation_overhang": overhang_limit,
        "bounded_generation_overhang": bounded_generation_overhang,
        "context_mode": (
            "official_generation_overhang"
            if bounded_generation_overhang
            else (
                "explicit_unvalidated_extrapolation" if extrapolating else "native_declared_context"
            )
        ),
        "validation": "MODEL_CONTEXT_CONTRACT_OK",
    }


def _olmes_normalize_answer(text: str) -> str:
    """Reproduce the normalization used by OLMES' HELMET QA scorer."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def score_ruler_generation(generation: str, expected: Sequence[str], match_type: str) -> float:
    """Reproduce the pinned OLMES RULER metrics."""

    if not expected:
        raise ValueError("RULER expected answers must not be empty")
    lowered = generation.lower()
    matches = [float(str(reference).lower() in lowered) for reference in expected]
    if match_type == "all":
        return sum(matches) / len(matches)
    if match_type == "part":
        normalized_generation = _olmes_normalize_answer(generation)
        return max(
            float(_olmes_normalize_answer(str(reference)) in normalized_generation)
            for reference in expected
        )
    raise ValueError(f"unsupported RULER match type: {match_type}")


def extract_longbench_v2_answer(response: str) -> str | None:
    """Reproduce LongBench v2's pinned answer extractor."""

    cleaned = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", cleaned)
    return match.group(1) if match else None


def _safe_accuracy(correct: float, total: int) -> float | None:
    return correct / total if total else None


def _percentage(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def _validate_prediction_coverage(
    manifest: dict[str, Any], rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> tuple[int, dict[tuple[str, int], dict[str, Any]]]:
    samples_per_example_values = {
        int(prediction.get("samples_per_example", 0)) for prediction in predictions
    }
    if len(samples_per_example_values) != 1:
        raise ValueError("predictions disagree on samples_per_example")
    samples_per_example = next(iter(samples_per_example_values), 0)
    if samples_per_example <= 0:
        raise ValueError("predictions have no valid samples_per_example")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for prediction in predictions:
        if prediction.get("status") != PREDICTION_STATUS:
            raise ValueError("prediction status mismatch")
        key = (str(prediction.get("example_id")), int(prediction.get("sample_index", -1)))
        if key in by_key:
            raise ValueError(f"duplicate prediction key: {key}")
        by_key[key] = prediction
    expected_keys = {
        (str(row["example_id"]), sample_index)
        for row in rows
        for sample_index in range(samples_per_example)
    }
    actual_keys = set(by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:8]
        extra = sorted(actual_keys - expected_keys)[:8]
        raise ValueError(f"prediction coverage mismatch: missing={missing} extra={extra}")
    for row in rows:
        for sample_index in range(samples_per_example):
            prediction = by_key[(str(row["example_id"]), sample_index)]
            if prediction.get("benchmark") != manifest.get("benchmark"):
                raise ValueError("prediction benchmark mismatch")
            if prediction.get("prompt_sha256") != row.get("prompt_sha256"):
                raise ValueError(f"prediction prompt hash mismatch: {row['example_id']}")
    return samples_per_example, by_key


def score_ruler(
    manifest: dict[str, Any], rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Score RULER by task and sequence length, preserving sample diagnostics."""

    samples_per_example, by_key = _validate_prediction_coverage(manifest, rows, predictions)
    task_sample_scores: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    example_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        sequence_length = int(row["metadata"]["sequence_length"])
        task = str(row["task"])
        match_type = str(row["metadata"]["match_type"])
        expected = [str(value) for value in row["expected_answer"]]
        for sample_index in range(samples_per_example):
            prediction = by_key[(str(row["example_id"]), sample_index)]
            score = score_ruler_generation(
                str(prediction.get("generation", "")), expected, match_type
            )
            task_sample_scores[(sequence_length, task, sample_index)].append(score)
            example_scores[str(row["example_id"])].append(score)

    task_rows = []
    for (sequence_length, task, sample_index), values in sorted(task_sample_scores.items()):
        task_rows.append(
            {
                "sequence_length": sequence_length,
                "task": task,
                "sample_index": sample_index,
                "example_count": len(values),
                "accuracy": sum(values) / len(values),
                "accuracy_percent": 100.0 * sum(values) / len(values),
            }
        )
    length_rows = []
    lengths = sorted({int(row["metadata"]["sequence_length"]) for row in rows})
    for sequence_length in lengths:
        sample_macros = []
        for sample_index in range(samples_per_example):
            task_accuracies = [
                task_row["accuracy"]
                for task_row in task_rows
                if task_row["sequence_length"] == sequence_length
                and task_row["sample_index"] == sample_index
            ]
            if len(task_accuracies) != len(RULER_TASKS):
                raise ValueError(f"RULER length {sequence_length} does not contain all 13 tasks")
            sample_macros.append(sum(task_accuracies) / len(task_accuracies))
        length_example_ids = [
            str(row["example_id"])
            for row in rows
            if int(row["metadata"]["sequence_length"]) == sequence_length
        ]
        pass_at_k = sum(
            float(max(example_scores[example_id]) >= 1.0) for example_id in length_example_ids
        ) / len(length_example_ids)
        length_rows.append(
            {
                "sequence_length": sequence_length,
                "official_sample0_macro_accuracy": sample_macros[0],
                "official_sample0_macro_accuracy_percent": 100.0 * sample_macros[0],
                "sample_mean_macro_accuracy": sum(sample_macros) / len(sample_macros),
                "sample_mean_macro_accuracy_percent": 100.0
                * sum(sample_macros)
                / len(sample_macros),
                "pass_at_k": pass_at_k,
                "pass_at_k_percent": 100.0 * pass_at_k,
                "samples_per_example": samples_per_example,
            }
        )
    return {
        "status": SCORE_STATUS,
        "schema_version": SCHEMA_VERSION,
        "benchmark": RULER_BENCHMARK,
        "official_protocol_compatible": samples_per_example == 1,
        "samples_per_example": samples_per_example,
        "task_rows": task_rows,
        "summary_rows": length_rows,
    }


def _majority_vote(values: Sequence[str | None]) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value is not None:
            counts[value] += 1
    if not counts:
        return None
    highest = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == highest)
    return winners[0] if len(winners) == 1 else None


def score_longbench_v2(
    manifest: dict[str, Any], rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Score official LongBench v2 slices plus domain diagnostics."""

    samples_per_example, by_key = _validate_prediction_coverage(manifest, rows, predictions)
    slice_totals: dict[str, int] = defaultdict(int)
    slice_correct: dict[str, int] = defaultdict(int)
    sample_correct = [0 for _ in range(samples_per_example)]
    pass_count = 0
    majority_count = 0
    scored_rows = []
    for row in rows:
        expected = str(row["expected_answer"])
        extracted = [
            extract_longbench_v2_answer(
                str(by_key[(str(row["example_id"]), sample_index)].get("generation", ""))
            )
            for sample_index in range(samples_per_example)
        ]
        correct = [value == expected for value in extracted]
        for sample_index, value in enumerate(correct):
            sample_correct[sample_index] += int(value)
        pass_count += int(any(correct))
        majority = _majority_vote(extracted)
        majority_count += int(majority == expected)
        metadata = row["metadata"]
        slices = {
            "overall": "all",
            "difficulty": str(metadata["difficulty"]),
            "length": str(metadata["length"]),
            "domain": str(metadata["domain"]),
            "sub_domain": str(metadata["sub_domain"]),
        }
        for kind, value in slices.items():
            key = f"{kind}:{value}"
            slice_totals[key] += 1
            slice_correct[key] += int(correct[0])
        scored_rows.append(
            {
                "example_id": str(row["example_id"]),
                "expected_answer": expected,
                "extracted_answers": extracted,
                "sample_correct": correct,
                "majority_answer": majority,
            }
        )

    breakdown_rows = []
    order = {"overall": 0, "difficulty": 1, "length": 2, "domain": 3, "sub_domain": 4}
    for key, total in sorted(
        slice_totals.items(), key=lambda item: (order[item[0].split(":", 1)[0]], item[0])
    ):
        kind, value = key.split(":", 1)
        accuracy = slice_correct[key] / total
        breakdown_rows.append(
            {
                "slice": kind,
                "value": value,
                "example_count": total,
                "official_sample0_accuracy": accuracy,
                "official_sample0_accuracy_percent": 100.0 * accuracy,
            }
        )
    total = len(rows)
    sample_accuracies = [value / total for value in sample_correct]
    return {
        "status": SCORE_STATUS,
        "schema_version": SCHEMA_VERSION,
        "benchmark": LONGBENCH_V2_BENCHMARK,
        "official_protocol_compatible": samples_per_example == 1,
        "samples_per_example": samples_per_example,
        "example_count": total,
        "official_sample0_accuracy": sample_accuracies[0],
        "official_sample0_accuracy_percent": _percentage(sample_accuracies[0]),
        "sample_mean_accuracy": sum(sample_accuracies) / len(sample_accuracies),
        "sample_mean_accuracy_percent": 100.0 * sum(sample_accuracies) / len(sample_accuracies),
        "pass_at_k": pass_count / total,
        "pass_at_k_percent": 100.0 * pass_count / total,
        "majority_vote_accuracy": majority_count / total,
        "majority_vote_accuracy_percent": 100.0 * majority_count / total,
        "breakdown_rows": breakdown_rows,
        "scored_rows": scored_rows,
    }


def score_predictions(
    manifest: dict[str, Any], rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Dispatch scoring through the pinned benchmark implementation."""

    benchmark = str(manifest.get("benchmark"))
    if benchmark == RULER_BENCHMARK:
        return score_ruler(manifest, rows, predictions)
    if benchmark == LONGBENCH_V2_BENCHMARK:
        return score_longbench_v2(manifest, rows, predictions)
    raise ValueError(f"unsupported benchmark: {benchmark}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a stable CSV from homogeneous dictionaries."""

    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_finite_scores(payload: Any) -> None:
    """Fail if a score payload contains a non-finite float."""

    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError(f"score payload contains a non-finite value: {payload}")
    if isinstance(payload, dict):
        for value in payload.values():
            assert_finite_scores(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_finite_scores(value)
