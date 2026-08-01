"""JSONL persistence for complete rollout records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rewardscope.schemas import RolloutRecord
from rewardscope.io.atomic import atomic_write_jsonl


def write_rollouts_jsonl(
    path: str | Path,
    records: Iterable[RolloutRecord],
    *,
    append: bool = False,
) -> int:
    """Write complete rollout records as UTF-8 JSON Lines and return their count."""
    if not isinstance(append, bool):
        raise TypeError("append must be a boolean.")

    destination = Path(path)
    serialized_records = _serialize_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not append:
        atomic_write_jsonl(destination, [json.loads(record) for record in serialized_records])
        return len(serialized_records)

    mode = "a"
    with destination.open(mode, encoding="utf-8", newline="\n") as output_file:
        for serialized_record in serialized_records:
            output_file.write(serialized_record)
            output_file.write("\n")

    return len(serialized_records)


def read_rollouts_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read rollout JSONL into nested dictionaries, preserving file order."""
    source = Path(path)
    records: list[dict[str, Any]] = []

    with source.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {source} on line {line_number}."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {source} on line {line_number}."
                )
            records.append(record)

    return records


def _serialize_records(records: Iterable[RolloutRecord]) -> list[str]:
    serialized_records: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, RolloutRecord):
            raise TypeError(f"records[{index}] must be a RolloutRecord.")
        serialized_records.append(
            json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False)
        )
    return serialized_records
