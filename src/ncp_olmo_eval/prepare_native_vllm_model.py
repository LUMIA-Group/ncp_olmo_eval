"""Create or validate the immutable-weight model overlay used by vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .native_vllm_inference import prepare_native_vllm_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=("conceptlm", "auto"), required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> Path:
    args = _parser().parse_args(argv)
    model_dir, manifest = prepare_native_vllm_model(
        source_model=args.source_model,
        runtime_config=args.runtime_config,
        overlay_dir=args.overlay_dir,
        model_family=args.model_family,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model_dir


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
