"""Unified vLLM evaluation for OLMo and NCP OLMo."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ncp-olmo-eval")
except PackageNotFoundError:  # pragma: no cover - editable source tree
    __version__ = "0.1.0a7"

__all__ = ["__version__"]
