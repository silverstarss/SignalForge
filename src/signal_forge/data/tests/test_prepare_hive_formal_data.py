from __future__ import annotations

import json

import pandas as pd
import pytest

from signal_forge.calibration.hive_dataset import CalibrationPrompt
from signal_forge.data.prepare_hive_formal_data import (
    build_verl_rows,
    load_dapo_prompts_streaming,
    load_frozen_dapo_repeated_snapshot,
    maximum_exact_ratio_counts,
    scan_prompt_token_lengths,
    select_maximum_exact_three_to_one,
)


def _prompt(source: str, row_id: str) -> CalibrationPrompt:
    canonical = (
        "Solve the following math problem step by step.\n"
        "Put your final answer in \\boxed{...}.\n\n"
        f"Problem {row_id}"
    )
    return CalibrationPrompt(
        prompt_id=f"{source}:{row_id}",
        dataset_source=source,
        source_row_id=row_id,
        raw_prompt=({"role": "user", "content": f"raw {row_id}"},),
        canonical_prompt=canonical,
        messages=({"role": "user", "content": canonical},),
        source_ground_truth="1",
        ground_truth="1" if source == "math" else r"\boxed{1}",
    )


def _dapo_source_row(row_id: str, *, answer: str = "1") -> dict:
    content = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n\n"
        f"Problem {row_id}\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    return {
        "data_source": "math",
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"index": row_id},
    }


@pytest.mark.parametrize(
    ("math_available", "dapo_available", "expected"),
    [(10, 20, (9, 3)), (30, 2, (6, 2)), (7496, 17917, (7494, 2498))],
)
def test_maximum_exact_ratio_counts(math_available, dapo_available, expected):
    assert maximum_exact_ratio_counts(math_available, dapo_available) == expected


def test_maximum_exact_ratio_requires_complete_unit():
    with pytest.raises(ValueError, match="complete ratio unit"):
        maximum_exact_ratio_counts(2, 20)


def test_selection_is_exact_unique_and_deterministic():
    math_prompts = tuple(_prompt("math", str(index)) for index in range(10))
    dapo_prompts = tuple(_prompt("dapo", str(index)) for index in range(9))
    first = select_maximum_exact_three_to_one(math_prompts, dapo_prompts, seed=42)
    second = select_maximum_exact_three_to_one(math_prompts, dapo_prompts, seed=42)
    assert [item.prompt_id for item in first] == [item.prompt_id for item in second]
    assert sum(item.dataset_source == "math" for item in first) == 9
    assert sum(item.dataset_source == "dapo" for item in first) == 3
    assert len({item.prompt_id for item in first}) == 12


def test_streaming_dapo_loader_deduplicates_identical_rows(tmp_path):
    path = tmp_path / "dapo.parquet"
    rows = [_dapo_source_row("a"), _dapo_source_row("b"), _dapo_source_row("a")]
    pd.DataFrame(rows).to_parquet(path, index=False)
    prompts, metadata = load_dapo_prompts_streaming(path, batch_size=1)
    assert [item.prompt_id for item in prompts] == ["dapo:a", "dapo:b"]
    assert metadata == {
        "source_rows": 3,
        "unique_prompts": 2,
        "duplicate_rows_removed": 1,
    }


def test_streaming_dapo_loader_rejects_conflicting_duplicates(tmp_path):
    path = tmp_path / "dapo.parquet"
    pd.DataFrame(
        [_dapo_source_row("a", answer="1"), _dapo_source_row("a", answer="2")]
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="conflicting content"):
        load_dapo_prompts_streaming(path, batch_size=1)


def test_length_scan_reports_sources_thresholds_and_longest():
    prompts = (_prompt("math", "a"), _prompt("math", "b"), _prompt("dapo", "c"))
    lengths = {"math:a": 100, "math:b": 600, "dapo:c": 900}
    counts, report = scan_prompt_token_lengths(
        prompts, lambda prompt: lengths[prompt.prompt_id]
    )
    assert counts == lengths
    assert report["overall"]["p50"] == 600
    assert report["overall"]["thresholds"]["512"] == {
        "count_above": 2,
        "ratio_above": pytest.approx(2 / 3),
    }
    assert report["by_dataset_source"]["dapo"]["max"] == 900
    assert report["longest_prompts"][0]["prompt_id"] == "dapo:c"


def test_verl_rows_preserve_identity_audit_fields_and_shared_reward_source():
    prompts = (_prompt("math", "a"), _prompt("dapo", "b"))
    rows = build_verl_rows(prompts, {"math:a": 101, "dapo:b": 202})
    assert [row["data_source"] for row in rows] == ["math", "math"]
    assert [row["extra_info"]["prompt_id"] for row in rows] == ["math:a", "dapo:b"]
    assert [row["extra_info"]["dataset_source"] for row in rows] == ["math", "dapo"]
    assert rows[1]["extra_info"]["prompt_token_count"] == 202
    assert json.loads(rows[1]["extra_info"]["raw_prompt_json"])[0]["content"] == "raw b"
    assert rows[1]["prompt"] == list(prompts[1].messages)


def test_frozen_dapo_loader_validates_and_materializes_repeated_id_cycles(tmp_path):
    import hashlib

    path = tmp_path / "dapo-repeated.parquet"
    cycle = [_dapo_source_row("a"), _dapo_source_row("b")]
    pd.DataFrame(cycle * 3).to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prompts, metadata = load_frozen_dapo_repeated_snapshot(
        path,
        expected_sha256=digest,
        unique_prompts=2,
        repeat_factor=3,
        batch_size=1,
    )
    assert [prompt.prompt_id for prompt in prompts] == ["dapo:a", "dapo:b"]
    assert metadata == {
        "source_rows": 6,
        "unique_prompts": 2,
        "duplicate_rows_removed": 4,
        "verified_repeat_factor": 3,
    }


def test_frozen_dapo_loader_rejects_non_frozen_bytes(tmp_path):
    path = tmp_path / "dapo.parquet"
    pd.DataFrame([_dapo_source_row("a")]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_dapo_repeated_snapshot(
            path, expected_sha256="0" * 64, unique_prompts=1, repeat_factor=1
        )
