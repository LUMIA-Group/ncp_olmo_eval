#!/usr/bin/env python3
"""Pinned HELMET data preparation and reporting metadata."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .helmet_trec_eval_compat import install_if_missing
from .long_context_protocol import (
    HELMET_BENCHMARK,
    HELMET_OFFICIAL_COMMIT,
    HELMET_SEED,
    HELMET_SHORT_CATEGORY_COUNTS,
    HELMET_SHORT_FORMAL_EXAMPLE_COUNT,
    HELMET_SHORT_INPUT_LENGTHS,
    HELMET_SWEEP_CATEGORY_COUNTS,
    HELMET_SWEEP_FORMAL_EXAMPLE_COUNT,
    HELMET_SWEEP_INPUT_LENGTHS,
    SCHEMA_VERSION,
    file_sha256,
    prompt_sha256,
    write_json_atomic,
)

HELMET_PREPARATION_CACHE_SCHEMA_VERSION = 1

HELMET_CATEGORY_CONFIGS = (
    ("Recall", "recall.yaml"),
    ("RAG", "rag.yaml"),
    ("Re-rank", "rerank.yaml"),
    ("Cite", "cite.yaml"),
    ("LongQA", "longqa.yaml"),
    ("Summ", "summ.yaml"),
    ("ICL", "icl.yaml"),
)
HELMET_SHORT_CATEGORY_CONFIGS = tuple(
    (category, config_name.replace(".yaml", "_short.yaml"))
    for category, config_name in HELMET_CATEGORY_CONFIGS
)
HELMET_SWEEP_CATEGORY_CONFIGS = HELMET_SHORT_CATEGORY_CONFIGS
HELMET_PROFILES = {
    "short": {
        "configs": HELMET_SHORT_CATEGORY_CONFIGS,
        "input_lengths": HELMET_SHORT_INPUT_LENGTHS,
        "example_count": HELMET_SHORT_FORMAL_EXAMPLE_COUNT,
        "category_counts": HELMET_SHORT_CATEGORY_COUNTS,
    },
    "8k-64k": {
        "configs": HELMET_SWEEP_CATEGORY_CONFIGS,
        "input_lengths": HELMET_SWEEP_INPUT_LENGTHS,
        "example_count": HELMET_SWEEP_FORMAL_EXAMPLE_COUNT,
        "category_counts": HELMET_SWEEP_CATEGORY_COUNTS,
    },
}
HELMET_LLAMA2_TOKENIZER_MODEL_SHA256 = (
    "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
)
HELMET_PINNED_FILE_SHA256 = {
    "configs/recall.yaml": "98b7f40da75ce06796c77fb95c37c3db191628676028ab3b29704b47dca48dce",
    "configs/rag.yaml": "9d7378b56785c4b3bf7910252c4dd5a3160f593c1d36acee95b9a575b8bdcee2",
    "configs/rerank.yaml": "e6cac9e13124fbfffdb2ae9cbe6ba0a912258dda0cd1ff9ef7c88ddfb0caf7e5",
    "configs/cite.yaml": "58edbcc7605ce45ae4a1050f50e643045fcca450c59b0c54b5d86ed322800755",
    "configs/longqa.yaml": "69976fa381c1a6f7232a0f8a00ad98c7e1b8e40f9d7ef16394ad540dffa3b850",
    "configs/summ.yaml": "921255979c5c7169d630aafb9de61b0b521204feb43a85464a3ea97e74b70554",
    "configs/icl.yaml": "95b3b9817561b8962c10f73bd0cd24de2b840f21cf342898c6755ba8500ef82f",
    "configs/recall_short.yaml": "438fa82ed1279bf28c82d2f7cf88333d760aa49ce0227e43210002fc4d555367",
    "configs/rag_short.yaml": "6a5d7304934b213254126b64587ce780cd4b51bcc02ca5849cb7ae86be5a72a5",
    "configs/rerank_short.yaml": "d87fea7fd9831cd58a7ea52a08ac9fd1be786508f3db9987b33c1f6515b0921d",
    "configs/cite_short.yaml": "b2a8b0d38272498bb75639d08b6b09439f9c776bcf942f68d5b045118806df96",
    "configs/longqa_short.yaml": "b671fd74b22e5ef3287d52ebad97ae0b50ee46db6d8a874e560093c35d95dc36",
    "configs/summ_short.yaml": "b17847187a0da2f36952572a95dd8dc5921da130a4d1dcf04096da2b9576797d",
    "configs/icl_short.yaml": "e0bd0bcbf243ff587171b5722f0c3596965070902751ccbabedaaf997d7c4969",
    "data.py": "559977d8f97357c77f2bc1a554e7d02dffebce99f450ba034864ca3e07c09348",
    "model_utils.py": "9fd15826773fb7f213ac0629eb0db681419241c6d1773e16f1f51f01483eb848",
    "utils.py": "86ceef1f3acbd9865eec8d95a5464a6e2d315e1d82ea6b9ae690582bf0ad1a50",
    "eval_alce.py": "fd64faa8e36e508b55983532f6d61c43a6c224190fd43068d4dbb90e330739cf",
    "scripts/collect_results.py": (
        "dd9c0c4e128a693267a48da249d50f7dbc8cecb66b3ec99ee08e668b0f6b74c2"
    ),
}
HELMET_HUB_SOURCE_SPLITS = {
    "narrativeqa": {"train": 32747, "test": 10557, "validation": 3461},
    "multi_lexsum-v20230518": {"train": 3177, "validation": 454, "test": 908},
    "trec": {"train": 5452, "test": 500},
    "banking77": {"train": 10003, "test": 3080},
    "clinc_oos-plus": {"train": 15250, "validation": 3100, "test": 5500},
    "nlu_evaluation_data": {"train": 25715},
    "infinitebench": {
        "passkey": 590,
        "kv_retrieval": 500,
        "number_string": 590,
        "code_run": 400,
        "code_debug": 394,
        "math_find": 350,
        "math_calc": 50,
        "longdialogue_qa_eng": 200,
        "longbook_qa_eng": 351,
        "longbook_sum_eng": 103,
        "longbook_choice_eng": 229,
        "longbook_qa_chn": 189,
    },
}


def validate_helmet_hub_source_snapshots(root: Path) -> dict[str, Any]:
    """Validate frozen Hub-only datasets before an offline HELMET run."""

    resolved = root.resolve()
    manifest_path = resolved / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "HELMET_HUB_SOURCE_SNAPSHOTS_OK":
        raise ValueError(f"invalid HELMET Hub source status: {manifest.get('status')}")
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError("unsupported HELMET Hub source manifest schema")
    declared_contract = str(manifest.get("contract_sha256", ""))
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("contract_sha256", None)
    actual_contract = hashlib.sha256(
        json.dumps(
            unsigned_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if declared_contract != actual_contract:
        raise ValueError(
            f"HELMET Hub source contract changed: {declared_contract} != {actual_contract}"
        )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(HELMET_HUB_SOURCE_SPLITS):
        raise ValueError("HELMET Hub source dataset set does not match the pinned contract")
    for dataset_name, expected_splits in HELMET_HUB_SOURCE_SPLITS.items():
        dataset_contract = datasets[dataset_name]
        if dataset_contract.get("splits") != expected_splits:
            raise ValueError(f"HELMET Hub source split mismatch: {dataset_name}")
        declared_files = dataset_contract.get("files")
        if not isinstance(declared_files, list) or not declared_files:
            raise ValueError(f"HELMET Hub source has no files: {dataset_name}")
        dataset_root = (resolved / dataset_name).resolve()
        actual_relative_files = {
            str(path.relative_to(resolved)) for path in dataset_root.rglob("*") if path.is_file()
        }
        declared_relative_files = {str(entry["path"]) for entry in declared_files}
        if actual_relative_files != declared_relative_files:
            raise ValueError(f"HELMET Hub source file set changed: {dataset_name}")
        for entry in declared_files:
            path = (resolved / str(entry["path"])).resolve()
            if not path.is_relative_to(dataset_root):
                raise ValueError(f"HELMET Hub source path escaped root: {path}")
            if path.stat().st_size != int(entry["size"]):
                raise ValueError(f"HELMET Hub source size changed: {path}")
            if file_sha256(path) != str(entry["sha256"]):
                raise ValueError(f"HELMET Hub source content changed: {path}")
    return {
        "path": str(resolved),
        "manifest_sha256": file_sha256(manifest_path),
        "contract_sha256": actual_contract,
        "datasets": {
            name: {"splits": dict(splits)} for name, splits in HELMET_HUB_SOURCE_SPLITS.items()
        },
        "validation": "HELMET_HUB_SOURCE_SNAPSHOTS_OK",
    }


def _read_checkout_head_without_git(root: Path) -> str:
    """Resolve a normal Git HEAD using only its read-only metadata files."""

    git_dir = root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ").strip()
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, packed_ref = line.split(" ", 1)
            if packed_ref == ref:
                return commit
    raise ValueError(f"cannot resolve HELMET checkout HEAD ref: {ref}")


def validate_official_helmet_checkout(official_root: Path) -> dict[str, Any]:
    """Require the exact official revision and source hashes used by this adapter."""

    root = official_root.resolve()
    head_validation_method = "git_rev_parse"
    try:
        head = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # Some containers reject a shared-filesystem checkout even after a
        # safe.directory override. Resolve the same read-only HEAD ref without
        # trusting the container's Git configuration; source hashes below are
        # still mandatory.
        head = _read_checkout_head_without_git(root)
        head_validation_method = "git_metadata_fallback"
    if head != HELMET_OFFICIAL_COMMIT:
        raise ValueError(f"HELMET checkout mismatch: {head} != {HELMET_OFFICIAL_COMMIT}")
    files: dict[str, dict[str, str]] = {}
    for relative, expected in HELMET_PINNED_FILE_SHA256.items():
        path = root / relative
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"pinned HELMET file changed: {relative}: {actual} != {expected}")
        files[relative] = {"path": str(path), "sha256": actual}
    return {
        "path": str(root),
        "commit": head,
        "head_validation_method": head_validation_method,
        "files": files,
    }


def _load_official_modules(official_root: Path) -> tuple[Any, Any]:
    install_if_missing()
    root = official_root.resolve()
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    data_module = importlib.import_module("data")
    utils_module = importlib.import_module("utils")
    if Path(data_module.__file__).resolve() != root / "data.py":
        raise RuntimeError(f"imported the wrong HELMET data.py: {data_module.__file__}")
    if Path(utils_module.__file__).resolve() != root / "utils.py":
        raise RuntimeError(f"imported the wrong HELMET utils.py: {utils_module.__file__}")
    return data_module, utils_module


@contextlib.contextmanager
def _cap_datasets_parallelism(max_processes: int) -> Iterable[None]:
    """Bound upstream hard-coded Dataset map/filter worker counts.

    Pinned HELMET uses as many as 40 workers in a few ICL preprocessing paths.
    That is a performance choice, not part of the benchmark semantics, and can
    make one worker get killed on memory-constrained preparation hosts. Keep
    the official functions and arguments intact while lowering only the
    requested worker count for the duration of data preparation.
    """

    if max_processes <= 0:
        raise ValueError("HELMET preparation processes must be positive")
    from datasets import Dataset

    original_map = Dataset.map
    original_filter = Dataset.filter

    def bounded_map(self: Any, *args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("num_proc")
        if requested is not None:
            kwargs["num_proc"] = min(int(requested), int(max_processes))
        return original_map(self, *args, **kwargs)

    def bounded_filter(self: Any, *args: Any, **kwargs: Any) -> Any:
        requested = kwargs.get("num_proc")
        if requested is not None:
            kwargs["num_proc"] = min(int(requested), int(max_processes))
        return original_filter(self, *args, **kwargs)

    Dataset.map = bounded_map
    Dataset.filter = bounded_filter
    try:
        yield
    finally:
        Dataset.map = original_map
        Dataset.filter = original_filter


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Iterable[None]:
    """Serialize writers that share deterministic Hugging Face cache files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _redirect_datasets_cache(cache_root: Path | None) -> Iterable[None]:
    """Keep derived Arrow caches outside the sealed source and reuse them.

    Datasets loaded with ``load_from_disk`` normally place ``map``/``filter``
    cache files next to their Arrow source.  HELMET repeatedly applies the same
    expensive source-only transforms for every tokenizer and context length.
    Redirect those content-fingerprinted files to a shared cache namespace so
    the frozen source remains immutable while later calls become cache hits.
    """

    if cache_root is None:
        yield
        return

    import datasets as datasets_module
    from datasets import Dataset

    resolved = cache_root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    original_get_cache_file_path = Dataset._get_cache_file_path
    original_hf_datasets_cache = datasets_module.config.HF_DATASETS_CACHE
    caching_was_enabled = datasets_module.is_caching_enabled()

    def redirected_get_cache_file_path(self: Any, fingerprint: str) -> str:
        source_fingerprint = str(getattr(self, "_fingerprint", "unknown"))
        cache_directory = resolved / source_fingerprint[:2]
        cache_directory.mkdir(parents=True, exist_ok=True)
        return str(cache_directory / f"cache-{source_fingerprint}-{fingerprint}.arrow")

    Dataset._get_cache_file_path = redirected_get_cache_file_path
    datasets_module.config.HF_DATASETS_CACHE = str(resolved / "builders")
    Path(datasets_module.config.HF_DATASETS_CACHE).mkdir(parents=True, exist_ok=True)
    datasets_module.enable_caching()
    try:
        yield
    finally:
        Dataset._get_cache_file_path = original_get_cache_file_path
        datasets_module.config.HF_DATASETS_CACHE = original_hf_datasets_cache
        if not caching_was_enabled:
            datasets_module.disable_caching()


@contextlib.contextmanager
def _use_frozen_helmet_hub_datasets(
    data_module: Any, hub_source_root: Path, cache: dict[str, Any], *, disable_derived_caching: bool
) -> Iterable[None]:
    """Route every official Hub dataset request to a sealed local snapshot."""

    import datasets as datasets_module
    from datasets import load_from_disk

    original_datasets_loader = datasets_module.load_dataset
    original_data_loader = data_module.load_dataset
    caching_was_enabled = datasets_module.is_caching_enabled()

    def frozen_dataset(key: str) -> Any:
        if key not in cache:
            cache[key] = load_from_disk(str(hub_source_root.resolve() / key))
        return cache[key]

    def load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        arguments = list(args)
        options = dict(kwargs)
        options.pop("trust_remote_code", None)
        if path == "json":
            return original_datasets_loader(path, *args, **kwargs)
        if path == "narrativeqa":
            key = "narrativeqa"
        elif path == "allenai/multi_lexsum":
            name = options.pop("name", arguments.pop(0) if arguments else None)
            if name != "v20230518":
                raise ValueError(f"unexpected MultiLexSum configuration: {name}")
            key = "multi_lexsum-v20230518"
        elif path == "CogComp/trec":
            key = "trec"
        elif path == "PolyAI/banking77":
            key = "banking77"
        elif path == "clinc/clinc_oos":
            name = options.pop("name", arguments.pop(0) if arguments else None)
            if name != "plus":
                raise ValueError(f"unexpected CLINC configuration: {name}")
            key = "clinc_oos-plus"
        elif path == "xingkunliuxtracta/nlu_evaluation_data":
            key = "nlu_evaluation_data"
        elif path == "xinrongzhang2022/infinitebench":
            options.pop("features", None)
            key = "infinitebench"
        else:
            raise RuntimeError("pinned HELMET requested an unsealed network dataset: " f"{path}")
        if arguments or options:
            raise ValueError(
                f"unexpected loader arguments for pinned HELMET dataset {path}: "
                f"args={arguments!r} kwargs={options!r}"
            )
        return frozen_dataset(key)

    data_module.load_dataset = load_dataset
    # load_infbench imports this name inside the function body, so patch the
    # datasets module for the same narrowly scoped call as well.
    datasets_module.load_dataset = load_dataset
    # Without an explicit redirected cache, a Dataset restored with
    # load_from_disk would write derived map/filter files beside the sealed
    # Arrow source. Keep the legacy temporary-cache fallback for callers that
    # do not opt into the shared content-addressed cache.
    if disable_derived_caching:
        datasets_module.disable_caching()
    try:
        yield
    finally:
        if disable_derived_caching and caching_was_enabled:
            datasets_module.enable_caching()
        data_module.load_dataset = original_data_loader
        datasets_module.load_dataset = original_datasets_loader


def _split_config_value(value: Any, count: int) -> list[str]:
    parts = str(value).split(",")
    if len(parts) == 1:
        return parts * count
    if len(parts) != count:
        raise ValueError(f"HELMET config arity mismatch: {len(parts)} != {count}: {value!r}")
    return parts


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"HELMET sample contains a non-JSON value: {type(value)!r}")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _helmet_derived_cache_namespace(
    cache_root: Path | None,
    *,
    checkout: dict[str, Any],
    hub_source_contract: dict[str, Any],
    llama2_tokenizer_model_sha256: str,
) -> tuple[Path | None, dict[str, Any]]:
    """Build the model-independent namespace for official HELMET transforms."""

    try:
        import datasets as datasets_module

        datasets_version = str(datasets_module.__version__)
    except ImportError:
        datasets_version = "unavailable"
    contract = {
        "schema_version": HELMET_PREPARATION_CACHE_SCHEMA_VERSION,
        "official_commit": str(checkout["commit"]),
        "official_files": {
            name: str(metadata["sha256"]) for name, metadata in sorted(checkout["files"].items())
        },
        "hub_source_contract_sha256": str(hub_source_contract["contract_sha256"]),
        "official_llama2_tokenizer_model_sha256": llama2_tokenizer_model_sha256,
        "datasets_version": datasets_version,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {**contract, "contract_sha256": contract_sha256}
    if cache_root is None:
        return None, metadata
    return cache_root.resolve() / contract_sha256, metadata


def _read_checkpoint_rows(
    checkpoint_root: Path | None, checkpoint_key: str, checkpoint_contract_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if checkpoint_root is None:
        return None
    metadata_path = checkpoint_root / "index.json"
    rows_path = checkpoint_root / "inputs.partial.jsonl"
    if not metadata_path.is_file() or not rows_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "HELMET_PREPARE_CHECKPOINTS_OK":
            return None
        entries = metadata.get("entries")
        if not isinstance(entries, dict):
            return None
        entry = entries.get(checkpoint_key)
        if not isinstance(entry, dict):
            return None
        if entry.get("checkpoint_contract_sha256") != checkpoint_contract_sha256:
            return None
        start_offset = int(entry.get("start_offset", -1))
        end_offset = int(entry.get("end_offset", -1))
        if start_offset < 0 or end_offset <= start_offset:
            return None
        digest = hashlib.sha256()
        rows: list[dict[str, Any]] = []
        with rows_path.open("rb") as handle:
            handle.seek(start_offset)
            remaining = end_offset - start_offset
            while remaining > 0:
                line = handle.readline(remaining)
                if not line:
                    return None
                remaining -= len(line)
                digest.update(line)
                if line.strip():
                    rows.append(json.loads(line))
        if digest.hexdigest() != entry.get("rows_sha256"):
            return None
        if len(rows) != int(entry.get("example_count", -1)):
            return None
        dataset_contract = entry.get("dataset_contract")
        if not isinstance(dataset_contract, dict):
            return None
        return rows, dataset_contract
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_checkpoint_rows(
    checkpoint_root: Path | None,
    checkpoint_key: str,
    checkpoint_contract_sha256: str,
    rows: list[dict[str, Any]],
    dataset_contract: dict[str, Any],
) -> None:
    if checkpoint_root is None:
        return
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_root / "index.json"
    rows_path = checkpoint_root / "inputs.partial.jsonl"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "HELMET_PREPARE_CHECKPOINTS_OK":
            raise ValueError("invalid HELMET checkpoint index status")
    else:
        metadata = {
            "status": "HELMET_PREPARE_CHECKPOINTS_OK",
            "schema_version": HELMET_PREPARATION_CACHE_SCHEMA_VERSION,
            "entries": {},
        }
    entries = metadata["entries"]
    if not isinstance(entries, dict):
        raise ValueError("invalid HELMET checkpoint entries")
    sequence = int(checkpoint_key.split("-", 1)[0])
    stale_entries = [key for key in entries if int(str(key).split("-", 1)[0]) >= sequence]
    committed_end = max(
        (int(entry["end_offset"]) for key, entry in entries.items() if key not in stale_entries),
        default=0,
    )
    with rows_path.open("a+b") as handle:
        handle.truncate(committed_end)
        handle.seek(committed_end)
        digest = hashlib.sha256()
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
        end_offset = handle.tell()
    for key in stale_entries:
        del entries[key]
    entries[checkpoint_key] = {
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "example_count": len(rows),
        "start_offset": committed_end,
        "end_offset": end_offset,
        "rows_sha256": digest.hexdigest(),
        "dataset_contract": dataset_contract,
    }
    write_json_atomic(metadata_path, metadata)


def promote_checkpoint_rows(
    checkpoint_root: Path, output_path: Path, *, expected_example_count: int
) -> bool:
    """Promote a complete append-only checkpoint spool without reserializing it."""

    metadata_path = checkpoint_root / "index.json"
    rows_path = checkpoint_root / "inputs.partial.jsonl"
    if not metadata_path.is_file() or not rows_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata.get("entries")
    if metadata.get("status") != "HELMET_PREPARE_CHECKPOINTS_OK" or not isinstance(entries, dict):
        return False
    if sum(int(entry["example_count"]) for entry in entries.values()) != int(
        expected_example_count
    ):
        return False
    cursor = 0
    for entry in sorted(entries.values(), key=lambda item: int(item["start_offset"])):
        if int(entry["start_offset"]) != cursor:
            return False
        cursor = int(entry["end_offset"])
    committed_end = cursor
    if rows_path.stat().st_size != committed_end:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.replace(output_path)
    return True


def _raw_prompt_with_official_truncation(
    sample: dict[str, Any],
    data: dict[str, Any],
    tokenizer: Any,
    *,
    input_max_length: int,
    generation_max_length: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Mirror HELMET model_utils.tokenize for a base model exactly."""

    return _prompt_with_official_truncation(
        sample,
        data,
        tokenizer,
        input_max_length=input_max_length,
        generation_max_length=generation_max_length,
        use_chat_template=False,
    )


def _prompt_with_official_truncation(
    sample: dict[str, Any],
    data: dict[str, Any],
    tokenizer: Any,
    *,
    input_max_length: int,
    generation_max_length: int,
    use_chat_template: bool,
    system_message: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Mirror pinned HELMET tokenization for raw and chat-template tasks."""

    prepared = copy.deepcopy(sample)

    def render() -> tuple[str, list[int]]:
        if use_chat_template:
            user_message = str(data["user_template"]).format(**prepared)
            chat = [{"role": "user", "content": user_message}]
            if system_message is not None:
                chat.insert(0, {"role": "system", "content": system_message})
            try:
                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                if system_message is None:
                    raise
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            add_special_tokens = False
        else:
            prompt = str(data["prompt_template"]).format(**prepared)
            add_special_tokens = True
        encoded = tokenizer([prompt], add_special_tokens=add_special_tokens)["input_ids"][0]
        return prompt, [int(token_id) for token_id in encoded]

    prompt, token_ids = render()
    original_token_count = len(token_ids)
    budget = int(input_max_length) - int(generation_max_length)
    if budget <= 0:
        raise ValueError("HELMET input length must exceed the generation length")
    truncated_tokens = 0
    safety_truncated_tokens = 0
    if len(token_ids) > budget:
        truncated_tokens = len(token_ids) - budget
        context = str(prepared.get("context", ""))
        offsets = tokenizer([context], return_offsets_mapping=True)["offset_mapping"][0]
        if truncated_tokens > len(offsets):
            raise ValueError(
                "official HELMET context truncation cannot remove enough tokens: "
                f"need={truncated_tokens} context={len(offsets)}"
            )
        prepared["context"] = context[: int(offsets[-truncated_tokens][0])]
        prompt, token_ids = render()
    # Pinned upstream HELMET performs the offset crop only once.  Token-boundary
    # interactions between the context and the surrounding prompt can leave an
    # occasional one-token overflow.  Preserve that first crop exactly, then
    # apply only the observed residual overflow so prompt+generation remains
    # inside the benchmark's declared input-length window.
    while len(token_ids) > budget:
        overflow = len(token_ids) - budget
        context = str(prepared.get("context", ""))
        offsets = tokenizer([context], return_offsets_mapping=True)["offset_mapping"][0]
        if overflow > len(offsets):
            raise ValueError(
                "HELMET residual context truncation cannot remove enough tokens: "
                f"need={overflow} context={len(offsets)}"
            )
        next_context = context[: int(offsets[-overflow][0])]
        if next_context == context:
            raise ValueError("HELMET residual context truncation made no progress")
        prepared["context"] = next_context
        safety_truncated_tokens += overflow
        prompt, token_ids = render()
    return (
        prompt,
        prepared,
        {
            "strategy": (
                "official_model_utils_chat_template_context_tail"
                if use_chat_template
                else "official_model_utils_tokenize_context_tail"
            ),
            "original_token_count": original_token_count,
            "prepared_token_count": len(token_ids),
            "truncated_token_count": truncated_tokens,
            "safety_truncated_token_count": safety_truncated_tokens,
            "max_input_tokens": budget,
            "input_max_length": int(input_max_length),
            "generation_max_length": int(generation_max_length),
            "add_special_tokens": not use_chat_template,
            "use_chat_template": bool(use_chat_template),
            "system_message": system_message,
        },
    )


def _source_identifier(sample: dict[str, Any], fallback: int) -> str:
    for key in ("id", "qid", "_id"):
        if key in sample and sample[key] not in (None, ""):
            return str(sample[key])
    return str(fallback)


def prepare_helmet_rows(
    official_root: Path,
    tokenizer: Any,
    official_llama2_tokenizer: Any,
    official_llama2_tokenizer_path: Path,
    hub_source_root: Path,
    *,
    limit_per_dataset: int = 0,
    prepare_processes: int = 8,
    profile: str = "8k-64k",
    chat_template_policy: str = "force_raw",
    derived_cache_root: Path | None = None,
    checkpoint_root: Path | None = None,
    tokenizer_contract_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load official HELMET data and seal tokenizer-specific prompts."""

    import yaml

    if profile not in HELMET_PROFILES:
        raise ValueError(f"unsupported HELMET profile: {profile}")
    if chat_template_policy not in {"force_raw", "official"}:
        raise ValueError(f"unsupported HELMET chat-template policy: {chat_template_policy}")
    profile_contract = HELMET_PROFILES[profile]
    expected_input_lengths = set(profile_contract["input_lengths"])
    checkout = validate_official_helmet_checkout(official_root)
    hub_source_contract = validate_helmet_hub_source_snapshots(hub_source_root)
    data_module, _ = _load_official_modules(official_root)
    llama2_tokenizer_model = official_llama2_tokenizer_path.resolve() / "tokenizer.model"
    llama2_tokenizer_model_sha256 = file_sha256(llama2_tokenizer_model)
    if llama2_tokenizer_model_sha256 != HELMET_LLAMA2_TOKENIZER_MODEL_SHA256:
        raise ValueError(
            "HELMET requires the official Llama 2 tokenizer.model: "
            f"{llama2_tokenizer_model_sha256} != {HELMET_LLAMA2_TOKENIZER_MODEL_SHA256}"
        )
    if checkpoint_root is not None and not tokenizer_contract_sha256:
        raise ValueError("HELMET resumable shards require a tokenizer contract hash")
    derived_cache_namespace, derived_cache_contract = _helmet_derived_cache_namespace(
        derived_cache_root,
        checkout=checkout,
        hub_source_contract=hub_source_contract,
        llama2_tokenizer_model_sha256=llama2_tokenizer_model_sha256,
    )
    if derived_cache_namespace is not None:
        derived_cache_namespace.mkdir(parents=True, exist_ok=True)
        contract_path = derived_cache_namespace / "contract.json"
        with _exclusive_file_lock(derived_cache_namespace / ".contract.lock"):
            if contract_path.is_file():
                existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
                if existing_contract != derived_cache_contract:
                    raise ValueError("HELMET derived cache contract collision")
            else:
                write_json_atomic(contract_path, derived_cache_contract)

    class _PinnedLlama2TokenizerFactory:
        @staticmethod
        def from_pretrained(name: str, *_args: Any, **_kwargs: Any) -> Any:
            if name != "meta-llama/Llama-2-7b-hf":
                raise ValueError(f"unexpected tokenizer requested by pinned HELMET: {name}")
            return official_llama2_tokenizer

    # Pinned HELMET data.py hardcodes the gated Meta repository for length
    # filtering and document truncation.  Route only that exact request to a
    # caller-supplied, hash-verified local copy; the official checkout itself
    # remains byte-for-byte unchanged.
    data_module.AutoTokenizer = _PinnedLlama2TokenizerFactory
    frozen_hub_datasets: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    dataset_contracts: list[dict[str, Any]] = []
    max_new_tokens_by_task: dict[str, int] = {}
    newline_stop_tasks: set[str] = set()
    seen: set[str] = set()
    dataset_sequence = 0
    for category, config_name in profile_contract["configs"]:
        config_path = official_root / "configs" / config_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        datasets = str(config["datasets"]).split(",")
        test_files = _split_config_value(config["test_files"], len(datasets))
        demo_files = _split_config_value(config["demo_files"], len(datasets))
        input_lengths = [
            int(value) for value in _split_config_value(config["input_max_length"], len(datasets))
        ]
        generation_lengths = [
            int(value)
            for value in _split_config_value(config["generation_max_length"], len(datasets))
        ]
        for dataset, test_file, demo_file, input_length, generation_length in zip(
            datasets, test_files, demo_files, input_lengths, generation_lengths, strict=True
        ):
            dataset_sequence += 1
            if input_length not in expected_input_lengths:
                raise ValueError(f"unexpected HELMET input length for {dataset}: {input_length}")
            previous_generation_length = max_new_tokens_by_task.setdefault(
                dataset, generation_length
            )
            if previous_generation_length != generation_length:
                raise ValueError(f"unexpected HELMET generation length for {dataset}")
            stop_new_line = bool(config["stop_new_line"])
            official_use_chat_template = bool(config["use_chat_template"])
            use_chat_template = (
                official_use_chat_template if chat_template_policy == "official" else False
            )
            if stop_new_line:
                newline_stop_tasks.add(dataset)
            args = type(
                "HelmetArgs",
                (),
                {
                    "max_test_samples": (
                        min(int(config["max_test_samples"]), limit_per_dataset)
                        if limit_per_dataset > 0
                        else int(config["max_test_samples"])
                    ),
                    "shots": int(config["shots"]),
                    "seed": HELMET_SEED,
                },
            )()
            test_path = str((official_root / test_file).resolve()) if test_file else ""
            demo_path = str((official_root / demo_file).resolve()) if demo_file else ""
            checkpoint_contract = {
                "schema_version": HELMET_PREPARATION_CACHE_SCHEMA_VERSION,
                "source_cache_contract_sha256": derived_cache_contract["contract_sha256"],
                "tokenizer_contract_sha256": tokenizer_contract_sha256,
                "category": category,
                "config": config_name,
                "config_sha256": checkout["files"][f"configs/{config_name}"]["sha256"],
                "dataset": dataset,
                "test_file": test_file,
                "demo_file": demo_file,
                "input_max_length": input_length,
                "generation_max_length": generation_length,
                "max_test_samples": int(args.max_test_samples),
                "shots": int(args.shots),
                "seed": HELMET_SEED,
                "chat_template_policy": chat_template_policy,
                "use_chat_template": use_chat_template,
            }
            checkpoint_contract_sha256 = _canonical_sha256(checkpoint_contract)
            checkpoint_key = f"{dataset_sequence:03d}-{checkpoint_contract_sha256[:20]}"
            checkpoint = _read_checkpoint_rows(
                checkpoint_root, checkpoint_key, checkpoint_contract_sha256
            )
            if checkpoint is not None:
                checkpoint_rows, dataset_contract = checkpoint
                for row in checkpoint_rows:
                    example_id = str(row["example_id"])
                    if example_id in seen:
                        raise ValueError(f"duplicate HELMET example ID: {example_id}")
                    seen.add(example_id)
                rows.extend(checkpoint_rows)
                dataset_contracts.append(dataset_contract)
                print(
                    "helmet_prepare_shard=cache_hit "
                    f"dataset={dataset} input_max_length={input_length} "
                    f"examples={len(checkpoint_rows)}",
                    flush=True,
                )
                continue
            random.seed(HELMET_SEED)
            with (
                _cap_datasets_parallelism(prepare_processes),
                _use_frozen_helmet_hub_datasets(
                    data_module,
                    hub_source_root,
                    frozen_hub_datasets,
                    disable_derived_caching=derived_cache_namespace is None,
                ),
                _redirect_datasets_cache(derived_cache_namespace),
                (
                    _exclusive_file_lock(derived_cache_namespace / ".write.lock")
                    if derived_cache_namespace is not None
                    else contextlib.nullcontext()
                ),
            ):
                loaded = data_module.load_data(args, dataset, test_path, demo_path)
            samples: Iterable[dict[str, Any]] = loaded["data"]
            dataset_count = 0
            dataset_fingerprint = getattr(samples, "_fingerprint", None)
            dataset_rows: list[dict[str, Any]] = []
            for sample_index, source_sample in enumerate(samples):
                sample = dict(source_sample)
                prompt, prepared_sample, truncation = _prompt_with_official_truncation(
                    sample,
                    loaded,
                    tokenizer,
                    input_max_length=input_length,
                    generation_max_length=generation_length,
                    use_chat_template=use_chat_template,
                )
                source_id = _source_identifier(sample, sample_index)
                example_id_parts = [category, dataset]
                example_id_parts.append(str(input_length))
                example_id_parts.extend((str(sample_index), source_id))
                example_id = ":".join(example_id_parts)
                if example_id in seen:
                    raise ValueError(f"duplicate HELMET example ID: {example_id}")
                seen.add(example_id)
                scoring_payload = copy.deepcopy(prepared_sample)
                scoring_payload.pop("context", None)
                scoring_payload.pop("input_ids", None)
                prefix = (
                    ""
                    if use_chat_template
                    else str(loaded["system_template"]).format(**prepared_sample)
                )
                dataset_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "benchmark": HELMET_BENCHMARK,
                        "example_id": example_id,
                        "task": dataset,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                        "expected_answer": _jsonable(prepared_sample.get("answer")),
                        "metadata": {
                            "category": category,
                            "dataset": dataset,
                            "source_index": sample_index,
                            "source_id": source_id,
                            "prompt_transport": (
                                "official_chat_template" if use_chat_template else "raw_base_model"
                            ),
                            "use_chat_template": use_chat_template,
                            "completion_prefix": prefix,
                            "input_max_length": input_length,
                            "max_new_tokens": generation_length,
                            "stop": (["\n", "\n\n"] if stop_new_line else []),
                            "truncation": truncation,
                            "scoring_payload": _jsonable(scoring_payload),
                        },
                    }
                )
                dataset_count += 1
            dataset_contract = {
                "category": category,
                "dataset": dataset,
                "example_count": dataset_count,
                "dataset_fingerprint": dataset_fingerprint,
                "test_file": test_file,
                "demo_file": demo_file,
                "config": config_name,
                "input_max_length": input_length,
                "generation_max_length": generation_length,
                "shots": int(config["shots"]),
                "stop_new_line": stop_new_line,
                "official_use_chat_template": official_use_chat_template,
                "use_chat_template": use_chat_template,
                "chat_template_policy": chat_template_policy,
                "base_model_use_chat_template_override": (
                    False if chat_template_policy == "force_raw" else None
                ),
                # The pinned upstream loader calls Dataset.shuffle()
                # without a seed while constructing demonstrations for
                # these two datasets.  The adapter does not invent a
                # different prompt; it seals the one official run emitted
                # so every backend sees byte-identical text.
                "upstream_unseeded_demo_shuffle": dataset.startswith(
                    ("narrativeqa_", "multi_lexsum_")
                ),
            }
            _write_checkpoint_rows(
                checkpoint_root,
                checkpoint_key,
                checkpoint_contract_sha256,
                dataset_rows,
                dataset_contract,
            )
            rows.extend(dataset_rows)
            dataset_contracts.append(dataset_contract)
            print(
                "helmet_prepare_shard=written "
                f"dataset={dataset} input_max_length={input_length} "
                f"examples={len(dataset_rows)}",
                flush=True,
            )
    counts = Counter(str(row["metadata"]["category"]) for row in rows)
    budget_guard_rows = sum(
        int(row["metadata"]["truncation"]["safety_truncated_token_count"]) > 0 for row in rows
    )
    budget_guard_tokens = sum(
        int(row["metadata"]["truncation"]["safety_truncated_token_count"]) for row in rows
    )
    return rows, {
        "checkout": checkout,
        "official_llama2_tokenizer": {
            "declared_repository": "meta-llama/Llama-2-7b-hf",
            "path": str(official_llama2_tokenizer_path.resolve()),
            "tokenizer_model_sha256": llama2_tokenizer_model_sha256,
        },
        "hub_source_snapshots": hub_source_contract,
        "preparation_processes": int(prepare_processes),
        "derived_cache": {
            **derived_cache_contract,
            "path": (str(derived_cache_namespace) if derived_cache_namespace is not None else None),
            "enabled": derived_cache_namespace is not None,
        },
        "profile": profile,
        "input_lengths": list(profile_contract["input_lengths"]),
        "datasets": dataset_contracts,
        "category_counts": dict(counts),
        "formal_example_count": int(profile_contract["example_count"]),
        "formal_category_counts": dict(profile_contract["category_counts"]),
        "max_new_tokens_by_task": max_new_tokens_by_task,
        "newline_stop_tasks": sorted(newline_stop_tasks),
        "seed": HELMET_SEED,
        "prompt_profile": (
            "official_task_chat_template_config"
            if chat_template_policy == "official"
            else "official_base_model_slurm_override"
        ),
        "chat_template_policy": chat_template_policy,
        "official_one_pass_budget_guard": {
            "affected_example_count": budget_guard_rows,
            "additional_truncated_token_count": budget_guard_tokens,
        },
        "formal_data_validation_passed": (
            len(rows) == int(profile_contract["example_count"])
            and dict(counts) == dict(profile_contract["category_counts"])
        ),
    }


def write_official_compatible_results(
    path: Path, rows: list[dict[str, Any]], generations: dict[str, str]
) -> None:
    """Write a compact result payload consumable by pinned HELMET judge scripts."""

    data = []
    for row in rows:
        payload = dict(row["metadata"]["scoring_payload"])
        payload["output"] = (
            str(row["metadata"]["completion_prefix"]) + generations[str(row["example_id"])]
        )
        data.append(payload)
    write_json_atomic(path, {"data": data})
