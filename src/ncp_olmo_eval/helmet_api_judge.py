#!/usr/bin/env python3
"""Run pinned HELMET LongQA/Summary API judges with an alternate API model."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

TASKS = (
    "narrativeqa_130772",
    "infbench_sum_eng_129672",
    "multi_lexsum_130372",
)
EXPECTED_COUNT = {task: 100 for task in TASKS}
HELMET_JUDGE_OFFICIAL_REFERENCE_MODEL = "gpt-4o-2024-05-13"
HELMET_JUDGE_TEMPERATURE = 0.1
HELMET_JUDGE_TOP_P = 0.9
HELMET_JUDGE_SEED = 42
HELMET_LONGQA_MAX_TOKENS = 2048
HELMET_SUMMARY_MAX_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official HELMET module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_inputs(prediction_root: Path, output_root: Path) -> dict[str, Path]:
    rows: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    seen: set[str] = set()
    shard_paths = sorted(prediction_root.glob("shard-*-of-*/predictions.jsonl"))
    if not shard_paths:
        raise RuntimeError("no prediction shards found")
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                task = str(item.get("task"))
                if task not in rows:
                    continue
                if item.get("status") != "LONG_CONTEXT_PREDICTION_OK":
                    raise RuntimeError(f"invalid prediction status in {shard_path}")
                if int(item.get("sample_index", -1)) != 0:
                    raise RuntimeError(f"unexpected sample_index for {item.get('example_id')}")
                example_id = str(item["example_id"])
                if example_id in seen:
                    raise RuntimeError(f"duplicate prediction: {example_id}")
                seen.add(example_id)
                metadata = item["metadata"]
                payload = dict(metadata["scoring_payload"])
                payload["output"] = str(metadata.get("completion_prefix", "")) + str(
                    item["generation"]
                )
                rows[task].append(
                    {
                        "source_index": int(metadata["source_index"]),
                        "payload": payload,
                    }
                )
    sources: dict[str, Path] = {}
    for task in TASKS:
        task_rows = sorted(rows[task], key=lambda row: row["source_index"])
        if len(task_rows) != EXPECTED_COUNT[task]:
            raise RuntimeError(
                f"prediction coverage mismatch for {task}: "
                f"{len(task_rows)} != {EXPECTED_COUNT[task]}"
            )
        source_indices = [row["source_index"] for row in task_rows]
        if len(set(source_indices)) != len(source_indices):
            raise RuntimeError(f"duplicate source_index for {task}")
        source = output_root / f"{task}.json"
        atomic_json(
            source,
            {
                "data": [row["payload"] for row in task_rows],
                "averaged_metrics": {},
            },
        )
        sources[task] = source
    return sources


def parsed_json(content: str) -> dict[str, Any] | None:
    matches = re.findall(r"\{.*?\}", content, re.DOTALL)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def required_keys(prompt: str) -> set[str]:
    if '"sentence_count"' in prompt:
        return {"precision", "sentence_count"}
    if '"recall"' in prompt and '"fluency"' not in prompt:
        return {"recall"}
    if '"correctness"' in prompt:
        return {"fluency", "correctness"}
    return {"fluency"}


class CachedCompatibleJudge:
    def __init__(
        self,
        model_name: str,
        output_root: Path,
        *,
        generation_max_length: int,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name
        self.output_root = output_root
        self.generation_max_length = generation_max_length
        self.concurrency = max(1, int(os.environ.get("HELMET_JUDGE_CONCURRENCY", "10")))
        self.disable_thinking = os.environ.get(
            "HELMET_JUDGE_DISABLE_THINKING", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.json_output = os.environ.get(
            "HELMET_JUDGE_JSON_OUTPUT", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._append_lock = threading.Lock()

    def _cache_key(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": HELMET_JUDGE_TEMPERATURE,
                "top_p": HELMET_JUDGE_TOP_P,
                "max_tokens": self.generation_max_length,
                "seed": HELMET_JUDGE_SEED,
                "thinking_disabled": self.disable_thinking,
                "json_output": self.json_output,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_cache(self, path: Path) -> dict[str, dict[str, Any]]:
        cached: dict[str, dict[str, Any]] = {}
        if not path.is_file():
            return cached
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item.get("response"), dict):
                    cached[str(item["key"])] = item["response"]
        return cached

    def _append_cache(self, path: Path, key: str, response: dict[str, Any]) -> None:
        record = json.dumps({"key": key, "response": response}, ensure_ascii=False)
        with self._append_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(record + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _generate_one(self, prompt: str) -> dict[str, Any]:
        expected = required_keys(prompt)
        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                kwargs: dict[str, Any] = {}
                if self.disable_thinking:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                if self.json_output:
                    kwargs["response_format"] = {"type": "json_object"}
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.generation_max_length,
                    temperature=HELMET_JUDGE_TEMPERATURE,
                    top_p=HELMET_JUDGE_TOP_P,
                    seed=HELMET_JUDGE_SEED,
                    **kwargs,
                )
                content = result.choices[0].message.content or ""
                parsed = parsed_json(content)
                if parsed is None or not expected.issubset(parsed):
                    raise ValueError(
                        f"judge response missing keys {sorted(expected)}: "
                        f"found={sorted(parsed) if parsed else []}"
                    )
                usage = getattr(result, "usage", None)
                return {
                    "output": content,
                    "input_len": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "output_len": int(getattr(usage, "completion_tokens", 0) or 0),
                    "input_text": [{"role": "user", "content": prompt}],
                    "system_fingerprint": getattr(result, "system_fingerprint", None),
                }
            except Exception as error:  # provider/network errors need bounded retry
                last_error = error
                if attempt < 8:
                    time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        raise RuntimeError(f"judge request failed after retries: {last_error}")

    def generate_batch(
        self,
        inputs: list[Any] | None = None,
        prompt: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        prompts = prompt if prompt is not None else [str(value) for value in inputs or []]
        cache_name = hashlib.sha256(
            "\n".join(self._cache_key(value) for value in prompts).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = self.output_root / f"responses-{cache_name}.jsonl"
        cached = self._load_cache(cache_path)
        outputs: list[dict[str, Any] | None] = [None] * len(prompts)
        missing: list[tuple[int, str, str]] = []
        for index, value in enumerate(prompts):
            key = self._cache_key(value)
            if key in cached:
                outputs[index] = cached[key]
            else:
                missing.append((index, key, value))
        print(
            f"judge_batch total={len(prompts)} cached={len(prompts)-len(missing)} "
            f"missing={len(missing)} concurrency={self.concurrency}",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            future_to_item = {
                pool.submit(self._generate_one, value): (index, key)
                for index, key, value in missing
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_item):
                index, key = future_to_item[future]
                response = future.result()
                outputs[index] = response
                self._append_cache(cache_path, key, response)
                completed += 1
                if completed % 10 == 0 or completed == len(missing):
                    print(
                        f"judge_batch_progress completed={completed}/{len(missing)}",
                        flush=True,
                    )
        if any(output is None for output in outputs):
            raise RuntimeError("judge batch contains missing responses")
        return [output for output in outputs if output is not None]


def judge_output_complete(source: Path, destination: Path) -> bool:
    if not destination.is_file():
        return False
    try:
        source_rows = json.loads(source.read_text(encoding="utf-8"))["data"]
        destination_rows = json.loads(destination.read_text(encoding="utf-8"))["data"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return len(source_rows) == len(destination_rows) and all(
        isinstance(item.get("gpt-4-scores"), dict) for item in destination_rows
    )


def wait_for_file(path: Path, timeout_seconds: int = 3600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file() or path.stat().st_size == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for judge asset: {path}")
        print(f"waiting_for_judge_asset path={path}", flush=True)
        time.sleep(10)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".host-api-judge.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another host API judge is already running") from error

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required")
        model_name = os.environ.get("HELMET_JUDGE_MODEL", "gpt-4o-2024-05-13")
        official_root = args.official_root.resolve()
        scripts = official_root / "scripts"
        sys.path[:0] = [str(official_root), str(scripts)]
        # The official judge modules only need the OpenAIModel symbol for their
        # standalone CLI block. check_metrics receives our API-compatible model
        # explicitly, so avoid importing HELMET model_utils (torch/transformers)
        # on a connected control host.
        model_utils_shim = types.ModuleType("model_utils")
        model_utils_shim.OpenAIModel = object
        sys.modules["model_utils"] = model_utils_shim
        longqa = load_module("helmet_official_eval_gpt4_longqa", scripts / "eval_gpt4_longqa.py")
        summ = load_module("helmet_official_eval_gpt4_summ", scripts / "eval_gpt4_summ.py")
        sources = collect_inputs(args.prediction_root.resolve(), output_root)
        # Match pinned HELMET af609c4 exactly except for the configured API
        # endpoint/model: LongQA keeps OpenAIModel's 2048-token default while
        # Summary explicitly requests 4096. Both use temperature=0.1,
        # top_p=0.9, and seed=42.
        longqa_model = CachedCompatibleJudge(
            model_name,
            output_root,
            generation_max_length=HELMET_LONGQA_MAX_TOKENS,
        )
        summ_model = CachedCompatibleJudge(
            model_name,
            output_root,
            generation_max_length=HELMET_SUMMARY_MAX_TOKENS,
        )

        previous_cwd = Path.cwd()
        outputs: dict[str, Path] = {}
        try:
            os.chdir(official_root)
            for task in TASKS:
                source = sources[task]
                destination = source.with_name(source.stem + "-gpt4eval_o.json")
                module = longqa if task == "narrativeqa_130772" else summ
                model = longqa_model if task == "narrativeqa_130772" else summ_model
                if task == "infbench_sum_eng_129672":
                    wait_for_file(
                        official_root / "data/infbench/longbook_sum_eng_keypoints.jsonl"
                    )
                elif task == "multi_lexsum_130372":
                    wait_for_file(
                        official_root / "data/multi_lexsum/multi_lexsum_val.jsonl"
                    )
                if not judge_output_complete(source, destination):
                    module.check_metrics(model, str(source), str(destination))
                if not judge_output_complete(source, destination):
                    raise RuntimeError(f"incomplete official judge output: {destination}")
                outputs[task] = destination
        finally:
            os.chdir(previous_cwd)

        judged = {
            task: json.loads(path.read_text(encoding="utf-8")) for task, path in outputs.items()
        }
        longqa_raw = float(judged["narrativeqa_130772"]["averaged_metrics"]["gpt-4-score"])
        infbench_raw = float(judged["infbench_sum_eng_129672"]["averaged_metrics"]["gpt-4-f1"])
        multilexsum_raw = float(judged["multi_lexsum_130372"]["averaged_metrics"]["gpt-4-f1"])
        metrics = [
            {
                "dataset": "narrativeqa_130772",
                "metric": "gpt-4-score",
                "raw_score": longqa_raw,
                "score_percent": longqa_raw * (100.0 / 3.0),
            },
            {
                "dataset": "infbench_sum_eng_129672",
                "metric": "gpt-4-f1",
                "raw_score": infbench_raw,
                "score_percent": infbench_raw * 100.0,
            },
            {
                "dataset": "multi_lexsum_130372",
                "metric": "gpt-4-f1",
                "raw_score": multilexsum_raw,
                "score_percent": multilexsum_raw * 100.0,
            },
        ]
        result = {
            "status": "HELMET_EXTERNAL_JUDGE_OK",
            "judge_model": model_name,
            "judge_base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "official_reference_model": HELMET_JUDGE_OFFICIAL_REFERENCE_MODEL,
            "official_comparable": model_name == HELMET_JUDGE_OFFICIAL_REFERENCE_MODEL,
            "protocol": "pinned_official_helmet_prompts_and_parsers",
            "judge_parameters": {
                "temperature": HELMET_JUDGE_TEMPERATURE,
                "top_p": HELMET_JUDGE_TOP_P,
                "seed": HELMET_JUDGE_SEED,
                "longqa_max_tokens": HELMET_LONGQA_MAX_TOKENS,
                "summary_max_tokens": HELMET_SUMMARY_MAX_TOKENS,
                "thinking": "disabled" if longqa_model.disable_thinking else "provider_default",
                "response_format": "json_object" if longqa_model.json_output else "provider_default",
            },
            "prediction_root": str(args.prediction_root.resolve()),
            "official_root": str(official_root),
            "official_commit": "af609c4d51b97fc35012099380aa889da961c42d",
            "official_script_sha256": {
                "eval_gpt4_longqa.py": file_sha256(scripts / "eval_gpt4_longqa.py"),
                "eval_gpt4_summ.py": file_sha256(scripts / "eval_gpt4_summ.py"),
            },
            "metrics": metrics,
            "outputs": {task: str(path) for task, path in outputs.items()},
        }
        atomic_json(output_root / "judge-results.json", result)
        print("status=HELMET_EXTERNAL_JUDGE_OK", flush=True)
        for metric in metrics:
            print(
                f"{metric['dataset']} {metric['metric']} "
                f"score_percent={metric['score_percent']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
