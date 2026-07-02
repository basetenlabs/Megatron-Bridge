"""Env-gated per-rank JSONL probe for PP-export forensics (debug branch only).

Enable by setting ``BT_PROBE_BCAST_LOG=<directory>``. Each rank appends
line-buffered JSONL records to ``{dir}/probe_rank{RANK}.jsonl`` so that a hang
leaves the last attempted collective visible on every rank.

Zero-dependency on trainers_server; safe to import from vendored bridge code.
When the env var is unset every call is a cheap no-op.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time

_LOCK = threading.Lock()
_FILE = None
_SEQ = itertools.count()


def _probe_dir() -> str | None:
    return os.environ.get("BT_PROBE_BCAST_LOG") or None


def probe_enabled() -> bool:
    return _probe_dir() is not None


def _get_file():
    global _FILE
    if _FILE is None:
        directory = _probe_dir()
        assert directory is not None
        os.makedirs(directory, exist_ok=True)
        rank = os.environ.get("RANK", "na")
        path = os.path.join(directory, f"probe_rank{rank}.jsonl")
        # Line-buffered append: a wedged rank's file ends at its last event.
        _FILE = open(path, "a", buffering=1)
    return _FILE


def probe_log(event: str, **fields) -> None:
    """Append one JSONL record. No-op unless BT_PROBE_BCAST_LOG is set."""
    if not probe_enabled():
        return
    try:
        import torch

        record = {
            "event": event,
            "seq": next(_SEQ),
            "t": time.time(),
            "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "thread": threading.current_thread().name,
        }
        if torch.cuda.is_available():
            record["current_device"] = torch.cuda.current_device()
        record.update(fields)
        line = json.dumps(record, default=str)
        with _LOCK:
            _get_file().write(line + "\n")
    except Exception:
        # The probe must never break or desync the run it is observing.
        pass


def tensor_spec(tensor) -> dict | None:
    """JSON-safe summary of a tensor (or None)."""
    if tensor is None:
        return None
    try:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
            "elem_bytes": tensor.element_size(),
        }
    except Exception:
        return {"repr": repr(type(tensor))}
