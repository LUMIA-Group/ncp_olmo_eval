"""Inspect an exported ConceptLM artifact and optional vLLM registration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .contract import (
    ARCHITECTURE,
    SUPPORTED_VLLM_VERSION,
    BackendContractError,
    ConceptLMBackendConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--check-registry", action="store_true")
    return parser.parse_args()


def _registry_payload() -> dict[str, Any]:
    installed = importlib.metadata.version("vllm")
    if installed != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"backend requires vLLM {SUPPORTED_VLLM_VERSION}, got {installed}"
        )
    from vllm.model_executor.models import ModelRegistry

    from .plugin import register

    register()
    return {
        "vllm_version": installed,
        "registered": ARCHITECTURE in ModelRegistry.get_supported_archs(),
    }


def main() -> None:
    """Validate the exported config and report a machine-readable status."""

    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    config_path = model_dir / "config.json"
    payload: dict[str, Any] = {
        "model_dir": str(model_dir),
        "config_path": str(config_path),
        "architecture": ARCHITECTURE,
        "backend_status": "contract_only",
    }
    exit_code = 0
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        normalized = ConceptLMBackendConfig.from_mapping(raw_config)
        payload["status"] = "CONCEPTLM_VLLM_EXPORT_CONTRACT_OK"
        payload["normalized_config"] = normalized.to_dict()
        if args.check_registry:
            payload["registry"] = _registry_payload()
            if not payload["registry"]["registered"]:
                raise RuntimeError("ConceptLM architecture was not registered")
    except (BackendContractError, FileNotFoundError, json.JSONDecodeError, RuntimeError) as error:
        payload["status"] = "CONCEPTLM_VLLM_EXPORT_CONTRACT_FAILED"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        exit_code = 1

    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
