"""Small atomic file-writing primitives used by experiment artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import json


def atomic_write_json(path: str | Path, value: object) -> Path:
    return _atomic_write(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    serialized: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping.")
        serialized.append(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
    return _atomic_write(path, "\n".join(serialized) + ("\n" if serialized else ""))


def _atomic_write(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
