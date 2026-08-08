from __future__ import annotations

import pytest

from signal_forge.data.build_signal_forge_v1 import (
    CanonicalRecord,
    canonical_question,
    select_splits,
    stable_sha256,
    verl_row,
)


def _record(source: str, split: str, index: int, question: str | None = None) -> CanonicalRecord:
    question_text = question or f"{source} {split} question {index}"
    if source == "gsm8k":
        return CanonicalRecord(
            data_source="gsm8k",
            source_dataset="openai/gsm8k",
            source_config="main",
            source_split=split,
            source_index=index,
            source_key=f"{index:06d}",
            question=question_text,
            ground_truth=str(index),
            reference_solution=f"work #### {index}",
            math_level=None,
            math_type=None,
            source_file=None,
            source_revision="test",
            question_sha256=stable_sha256(canonical_question(question_text)),
        )
    return CanonicalRecord(
        data_source="math_level_3",
        source_dataset="competition_math",
        source_config="all",
        source_split=split,
        source_index=index,
        source_key=f"algebra/{index}.json",
        question=question_text,
        ground_truth=str(index),
        reference_solution=f"solution \\boxed{{{index}}}",
        math_level=3,
        math_type="Algebra",
        source_file=f"algebra/{index}.json",
        source_revision="test",
        question_sha256=stable_sha256(canonical_question(question_text)),
    )


def _records(source: str, split: str, count: int, offset: int = 0) -> list[CanonicalRecord]:
    return [_record(source, split, offset + index) for index in range(count)]


def test_select_splits_uses_exact_60_40_train_mixture():
    selection = select_splits(
        gsm_train=_records("gsm8k", "train", 20),
        gsm_test=_records("gsm8k", "test", 4, 1000),
        math_train=_records("math_level_3", "train", 12),
        math_test=_records("math_level_3", "test", 4, 1000),
        validation_gsm=2,
        validation_math=2,
        seed=7,
    )

    assert len(selection.train) == 25
    assert sum(r.data_source == "gsm8k" for r in selection.train) == 15
    assert sum(r.data_source == "math_level_3" for r in selection.train) == 10
    assert len(selection.validation) == 4
    assert len(selection.test) == 8


def test_select_splits_rounds_odd_math_train_count_down_for_exact_ratio():
    selection = select_splits(
        gsm_train=_records("gsm8k", "train", 20),
        gsm_test=[],
        math_train=_records("math_level_3", "train", 13),
        math_test=[],
        validation_gsm=2,
        validation_math=2,
        seed=7,
    )

    assert sum(r.data_source == "math_level_3" for r in selection.train) == 10
    assert sum(r.data_source == "gsm8k" for r in selection.train) == 15


def test_select_splits_removes_train_duplicate_with_test():
    leaked_question = "same problem text"
    gsm_train = [_record("gsm8k", "train", 0, leaked_question), *_records("gsm8k", "train", 10, 10)]
    gsm_test = [_record("gsm8k", "test", 0, leaked_question)]

    selection = select_splits(
        gsm_train=gsm_train,
        gsm_test=gsm_test,
        math_train=_records("math_level_3", "train", 6),
        math_test=[],
        validation_gsm=1,
        validation_math=1,
        seed=11,
    )

    assert all(record.question != leaked_question for record in selection.train)
    assert any(rejection.reason == "duplicate_with_test" for rejection in selection.rejections)
    assert selection.duplicate_report


def test_select_splits_fails_when_validation_pool_is_too_small():
    with pytest.raises(ValueError, match="MATH L3 validation"):
        select_splits(
            gsm_train=_records("gsm8k", "train", 20),
            gsm_test=[],
            math_train=_records("math_level_3", "train", 1),
            math_test=[],
            validation_gsm=2,
            validation_math=2,
            seed=7,
        )


def test_verl_row_schema_preserves_required_metadata():
    record = _record("math_level_3", "train", 12)
    row = verl_row(record, "train", 123)

    assert row["data_source"] == "math_level_3"
    assert row["ability"] == "math"
    assert row["prompt"][-1]["content"].endswith("\\boxed{}.")
    assert row["reward_model"] == {"style": "rule", "ground_truth": "12"}
    assert row["extra_info"]["prompt_id"] == "math:competition_math:train:algebra:12"
    assert row["extra_info"]["source_key"] == "algebra/12.json"
    assert row["extra_info"]["selection_seed"] == 123
    assert row["extra_info"]["dataset_version"] == "signal_forge_v1"
