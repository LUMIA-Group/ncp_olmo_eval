"""Process-local bridge between the ConceptLM target and custom DFlash proposer."""

from __future__ import annotations

import atexit
import json
import os
import time
import weakref
from pathlib import Path
from typing import Any, TextIO

_TARGET_MODEL: weakref.ReferenceType[Any] | None = None
_TELEMETRY_HANDLE: TextIO | None = None
_TELEMETRY_PATH: Path | None = None
_TELEMETRY_PENDING_RECORDS = 0


def close_telemetry() -> None:
    """Flush and close the process-local telemetry stream."""

    global _TELEMETRY_HANDLE, _TELEMETRY_PATH, _TELEMETRY_PENDING_RECORDS
    if _TELEMETRY_HANDLE is not None:
        _TELEMETRY_HANDLE.close()
    _TELEMETRY_HANDLE = None
    _TELEMETRY_PATH = None
    _TELEMETRY_PENDING_RECORDS = 0


atexit.register(close_telemetry)


def register_target_model(model: Any) -> None:
    """Expose the loaded target model to the lazy custom proposer."""

    global _TARGET_MODEL
    _TARGET_MODEL = weakref.ref(model)


def target_model() -> Any:
    """Return the live target model or fail before loading a second target."""

    model = _TARGET_MODEL() if _TARGET_MODEL is not None else None
    if model is None:
        raise RuntimeError("the ConceptLM DFlash target model is not registered")
    return model


def append_telemetry(event: str, **payload: Any) -> None:
    """Buffer one diagnostic record when the caller configured a path.

    A proposer emits a record on every decode step.  Opening and closing a
    GPFS file for every record puts metadata I/O directly on the hot path and
    can erase the speculative-decoding speedup.  Keep one userspace buffer per
    engine process and flush it in small groups.  Periodic flushing is needed
    because vLLM's EngineCore may terminate with ``os._exit``, which bypasses
    Python ``atexit`` handlers.
    """

    global _TELEMETRY_HANDLE, _TELEMETRY_PATH, _TELEMETRY_PENDING_RECORDS
    raw_path = os.environ.get("CONCEPTLM_DFLASH_TELEMETRY_PATH", "")
    if not raw_path:
        return
    path = Path(raw_path)
    if _TELEMETRY_HANDLE is None or _TELEMETRY_PATH != path:
        close_telemetry()
        path.parent.mkdir(parents=True, exist_ok=True)
        _TELEMETRY_HANDLE = path.open("a", encoding="utf-8", buffering=1 << 20)
        _TELEMETRY_PATH = path
    record = {"event": event, "monotonic_time": time.monotonic(), "pid": os.getpid(), **payload}
    assert _TELEMETRY_HANDLE is not None
    _TELEMETRY_HANDLE.write(json.dumps(record, sort_keys=True) + "\n")
    _TELEMETRY_PENDING_RECORDS += 1
    flush_interval = int(os.environ.get("CONCEPTLM_DFLASH_TELEMETRY_FLUSH_INTERVAL", "8"))
    if flush_interval < 1:
        raise ValueError("CONCEPTLM_DFLASH_TELEMETRY_FLUSH_INTERVAL must be positive")
    if _TELEMETRY_PENDING_RECORDS >= flush_interval:
        _TELEMETRY_HANDLE.flush()
        _TELEMETRY_PENDING_RECORDS = 0

