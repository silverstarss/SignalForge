from __future__ import annotations

import json

import pandas as pd

from signal_forge.data.prepare_hive_formal_validation import (
    EXPECTED_TOTAL,
    build_formal_validation_rows,
    write_formal_validation,
)
from signal_forge.data.validation_decontamination import (
    EXPECTED_BENCHMARK_COUNTS,
    ValidationProblem,
)


def _problems():
    rows = []
    for benchmark, count in EXPECTED_BENCHMARK_COUNTS.items():
        for index in range(count):
            multiple = benchmark == "olympiadbench" and index == 0
            rows.append(
                ValidationProblem(
                    benchmark=benchmark,
                    benchmark_id=str(index),
                    row_index=index,
                    raw_question=f"raw {benchmark} {index}",
                    question=f"Find {index} for {benchmark}.",
                    source_answer=["1", "2"] if multiple else "1",
                    ground_truth=r"\boxed{1, 2}" if multiple else r"\boxed{1}",
                    is_multiple_answer=multiple,
                    multiple_answers=("1", "2") if multiple else (),
                )
            )
    return tuple(rows)


def test_formal_validation_rows_preserve_sources_ids_and_multiple_answers():
    rows = build_formal_validation_rows(_problems())

    assert len(rows) == EXPECTED_TOTAL
    assert len({row["extra_info"]["prompt_id"] for row in rows}) == EXPECTED_TOTAL
    assert {row["data_source"] for row in rows} == set(EXPECTED_BENCHMARK_COUNTS)
    assert all(row["extra_info"]["source_dataset"] == "math" for row in rows)
    assert all(row["extra_info"]["formal_partition"] == "validation" for row in rows)
    assert all(
        row["prompt"][0]["content"].startswith("Solve the following math problem step by step.")
        for row in rows
    )
    multiple = next(row for row in rows if row["extra_info"]["is_multiple_answer"])
    assert json.loads(multiple["extra_info"]["multiple_answers_json"]) == ["1", "2"]


def test_formal_validation_parquet_round_trip_and_manifest(tmp_path):
    rows = build_formal_validation_rows(_problems())
    parquet_path, manifest_path = write_formal_validation(tmp_path, rows=rows)

    frame = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text())
    assert len(frame) == EXPECTED_TOTAL
    assert manifest["row_count"] == EXPECTED_TOTAL
    assert manifest["benchmark_counts"] == EXPECTED_BENCHMARK_COUNTS
    assert manifest["parquet_sha256"]
