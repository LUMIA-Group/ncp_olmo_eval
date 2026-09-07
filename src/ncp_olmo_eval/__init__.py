"""Unified vLLM evaluation for OLMo and NCP-ArchPreview."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ncp-olmo-eval")
except PackageNotFoundError:  # pragma: no cover - editable source tree
    __version__ = "0.1.0a15"

__all__ = ["__version__"]
