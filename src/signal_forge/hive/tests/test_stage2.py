from __future__ import annotations

import itertools
import math
import random

import pytest

from signal_forge.hive.prompt_entropy import PromptEntropyRecord
from signal_forge.hive.stage2 import (
    Stage2Config,
    Stage2PromptRecord,
    Stage2Selector,
    compute_stage2_counts,
)


def _records(entropies: list[float]) -> list[Stage2PromptRecord]:
    return [
        Stage2PromptRecord(prompt_id=f"prompt:{index:02d}", entropy=entropy)
        for index, entropy in enumerate(entropies)
    ]


def _ids(records) -> tuple[str, ...]:
    return tuple(record.prompt_id for record in records)


def test_canonical_entropy_band_is_correct_before_group_rounding():
    result = Stage2Selector().select(_records([10, 9, 8, 7, 6, 5, 4, 3]))

    assert [record.entropy for record in result.upper_trimmed] == [10, 9]
    assert [record.entropy for record in result.pre_round_kept] == [8, 7, 6, 5]
    assert [record.entropy for record in result.low_entropy_rejected] == [4, 3]
    assert result.kept == ()
    assert result.rounding_dropped == result.pre_round_kept


def test_entropy_ties_are_ordered_by_stable_prompt_id():
    records = [
        Stage2PromptRecord(prompt_id=prompt_id, entropy=1.0)
        for prompt_id in ["d", "b", "a", "c"]
    ]
    selector = Stage2Selector(Stage2Config(group_size=1))

    result = selector.select(records)

    assert _ids(result.upper_trimmed) == ("a",)
    assert _ids(result.kept) == ("b", "c")
    assert _ids(result.low_entropy_rejected) == ("d",)


def test_selection_is_invariant_to_input_order():
    records = _records([2.0, 5.0, 5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    shuffled = records.copy()
    random.Random(41).shuffle(shuffled)
    selector = Stage2Selector(Stage2Config(group_size=1))

    first = selector.select(records)
    second = selector.select(shuffled)

    assert _ids(first.upper_trimmed) == _ids(second.upper_trimmed)
    assert _ids(first.kept) == _ids(second.kept)
    assert _ids(first.low_entropy_rejected) == _ids(second.low_entropy_rejected)
    assert _ids(first.rounding_dropped) == _ids(second.rounding_dropped)


def test_exact_default_counts_without_rounding_loss():
    counts = compute_stage2_counts(16, Stage2Config())

    assert counts.input_count == 16
    assert counts.requested_upper_trim_count == 4
    assert counts.actual_upper_trim_count == 4
    assert counts.requested_keep_count == 8
    assert counts.pre_round_keep_count == 8
    assert counts.post_round_keep_count == 8
    assert counts.rounding_dropped_count == 0
    assert counts.low_entropy_reject_count == 4


def test_group_multiple_rounding_drops_tail_of_retained_band():
    result = Stage2Selector().select(_records(list(range(20, 0, -1))))

    assert result.diagnostics.pre_round_keep_count == 10
    assert result.diagnostics.post_round_keep_count == 8
    assert result.diagnostics.rounding_dropped_count == 2
    assert [record.entropy for record in result.rounding_dropped] == [7, 6]
    assert len(result.kept) % 8 == 0


def test_small_pool_can_round_to_zero_without_forcing_group_size():
    result = Stage2Selector().select(_records([8, 7, 6, 5, 4, 3, 2, 1]))

    assert result.diagnostics.pre_round_keep_count == 4
    assert result.diagnostics.post_round_keep_count == 0
    assert result.diagnostics.rounding_dropped_count == 4
    assert result.kept == ()


def test_non_default_group_size_is_respected():
    result = Stage2Selector(Stage2Config(group_size=3)).select(_records(list(range(10, 0, -1))))

    assert result.diagnostics.pre_round_keep_count == 5
    assert result.diagnostics.post_round_keep_count == 3
    assert result.diagnostics.rounding_dropped_count == 2
    assert len(result.kept) == 3


def test_non_default_ratios_use_floor_semantics():
    config = Stage2Config(upper_trim_ratio=0.2, keep_ratio=0.6, group_size=2)
    result = Stage2Selector(config).select(_records(list(range(10, 0, -1))))

    assert result.diagnostics.requested_upper_trim_count == math.floor(10 * 0.2)
    assert result.diagnostics.requested_keep_count == math.floor(10 * 0.6)
    assert len(result.upper_trimmed) == 2
    assert len(result.kept) == 6
    assert len(result.low_entropy_rejected) == 2


def test_prompt_entropy_records_can_be_consumed_without_conversion():
    records = tuple(
        PromptEntropyRecord(
            prompt_id=f"prompt:{index}",
            entropy=float(index),
            valid_token_count=4,
            predictive_position_count=3,
        )
        for index in range(4)
    )

    result = Stage2Selector(Stage2Config(group_size=1)).select(records)

    assert all(isinstance(record, PromptEntropyRecord) for record in result.kept)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upper_trim_ratio": -0.01},
        {"upper_trim_ratio": 1.0},
        {"keep_ratio": 0.0},
        {"keep_ratio": 1.01},
        {"upper_trim_ratio": 0.6, "keep_ratio": 0.5},
        {"group_size": 0},
        {"group_size": -1},
        {"group_size": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Stage2Config(**kwargs)


def test_nan_entropy_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        Stage2Selector().select([Stage2PromptRecord(prompt_id="nan", entropy=float("nan"))])


@pytest.mark.parametrize("entropy", [float("inf"), float("-inf")])
def test_infinite_entropy_is_rejected(entropy):
    with pytest.raises(ValueError, match="finite"):
        Stage2Selector().select([Stage2PromptRecord(prompt_id="infinite", entropy=entropy)])


def test_duplicate_prompt_ids_are_rejected():
    records = [
        Stage2PromptRecord(prompt_id="duplicate", entropy=1.0),
        Stage2PromptRecord(prompt_id="duplicate", entropy=2.0),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        Stage2Selector().select(records)


def test_empty_input_returns_empty_partition_and_zero_diagnostics():
    result = Stage2Selector().select([])

    assert result.kept == ()
    assert result.upper_trimmed == ()
    assert result.low_entropy_rejected == ()
    assert result.rounding_dropped == ()
    assert set(vars(result.diagnostics).values()) == {0}


def test_result_partitions_every_input_exactly_once():
    records = _records([float(value) for value in range(23)])

    result = Stage2Selector().select(records)

    partitions = (
        result.kept,
        result.upper_trimmed,
        result.low_entropy_rejected,
        result.rounding_dropped,
    )
    partitioned_ids = [record.prompt_id for record in itertools.chain.from_iterable(partitions)]
    assert len(partitioned_ids) == len(records)
    assert len(set(partitioned_ids)) == len(records)
    assert set(partitioned_ids) == {record.prompt_id for record in records}
    assert sum(len(partition) for partition in partitions) == result.diagnostics.input_count
