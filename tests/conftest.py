"""Test-profile selection for CPU documentation and full GPU environments."""

from __future__ import annotations

import importlib.util


collect_ignore: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore.extend(
        [
            "unit/test_core_native_cached_finalize.py",
            "unit/test_core_native_summary.py",
            "unit/test_gsm8k_eval.py",
            "unit/test_helmet_protocol.py",
            "unit/test_long_context_protocol.py",
            "unit/test_native_vllm_inference.py",
            "vllm_plugin/test_model_transaction.py",
            "vllm_plugin/test_ncp_dflash_proposer.py",
            "vllm_plugin/test_ncp_dflash_state.py",
            "vllm_plugin/test_state.py",
        ]
    )
if importlib.util.find_spec("vllm") is None:
    collect_ignore.extend(
        [
            "vllm_plugin/test_model_transaction.py",
            "vllm_plugin/test_ncp_dflash_model_runner.py",
        ]
    )
