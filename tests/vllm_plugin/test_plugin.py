"""Tests for vLLM registration in both front-end and EngineCore processes."""

from __future__ import annotations

import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, call, patch

from ncp_olmo_eval.vllm_plugin import plugin
from ncp_olmo_eval.vllm_plugin.contract import ARCHITECTURE, STANDALONE_HF_ARCHITECTURE


class _FakeModelRegistry:
    def __init__(self) -> None:
        self.architectures: set[str] = set()
        self.register_model = Mock(side_effect=self._register_model)

    def get_supported_archs(self) -> set[str]:
        return self.architectures

    def _register_model(self, architecture: str, _model_class: str) -> None:
        self.architectures.add(architecture)


class TestConceptLMPlugin(unittest.TestCase):
    def test_register_is_idempotent(self) -> None:
        registry = _FakeModelRegistry()
        models_module = ModuleType("vllm.model_executor.models")
        models_module.ModelRegistry = registry
        fake_modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.models": models_module,
        }

        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(plugin, "_register_transformers_config") as register_config,
        ):
            plugin.register()
            plugin.register()

        self.assertEqual(register_config.call_count, 2)
        implementation = "ncp_olmo_eval.vllm_plugin.model:ConceptLMV22VQForCausalLM"
        self.assertEqual(
            registry.register_model.call_args_list,
            [
                call(ARCHITECTURE, implementation),
                call(STANDALONE_HF_ARCHITECTURE, implementation),
            ],
        )

    def test_worker_import_registers_engine_core_process(self) -> None:
        worker_module = ModuleType("vllm.v1.worker.gpu_worker")
        worker_module.Worker = type("Worker", (), {})
        fake_modules = {
            "vllm": ModuleType("vllm"),
            "vllm.v1": ModuleType("vllm.v1"),
            "vllm.v1.worker": ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu_worker": worker_module,
        }

        sys.modules.pop("ncp_olmo_eval.vllm_plugin.worker", None)
        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(plugin, "register") as register,
        ):
            importlib.import_module("ncp_olmo_eval.vllm_plugin.worker")

        register.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
