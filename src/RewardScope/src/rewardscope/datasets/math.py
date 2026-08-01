"""Thin MATH adapter with audited final-boxed LaTeX gold answers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any

from rewardscope.datasets.schema import DatasetExample, DatasetLoadResult
from rewardscope.verification import extract_final_boxed_latex_gold


MATH_DATASET_ID = "EleutherAI/hendrycks_math"
MODELSCOPE_MATH_DATASET_ID = "opencompass/competition_math"
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE = """Solve the problem step by step and put your final answer within \\boxed{{}}.

Question: {question}
"""


def load_math_examples(
    split: str,
    *,
    config_name: str | None = "all",
    revision: str | None = None,
    max_examples: int | None = None,
    selection: str = "first",
    dataset_seed: int = 0,
    source_indices: tuple[int, ...] | None = None,
    levels: tuple[str, ...] | None = None,
    hf_endpoint: str | None = None,
    data_source: str = "huggingface",
    prompt_template: str = MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
) -> list[DatasetExample]:
    return list(
        load_math_result(
            config_name=config_name,
            split=split,
            revision=revision,
            max_examples=max_examples,
            selection=selection,
            dataset_seed=dataset_seed,
            source_indices=source_indices,
            levels=levels,
            hf_endpoint=hf_endpoint,
            data_source=data_source,
            prompt_template=prompt_template,
        ).examples
    )


def load_math_result(
    *,
    config_name: str | None,
    split: str,
    revision: str | None,
    max_examples: int | None,
    selection: str,
    dataset_seed: int,
    source_indices: tuple[int, ...] | None = None,
    levels: tuple[str, ...] | None = None,
    hf_endpoint: str | None = None,
    data_source: str = "huggingface",
    prompt_template: str = MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
) -> DatasetLoadResult:
    """Load MATH, audit LaTeX gold before filtering, then select valid prompts."""
    _require_non_empty_str("split", split)
    _require_optional_non_empty_str("config_name", config_name)
    _require_optional_non_empty_str("revision", revision)
    _require_optional_positive_int("max_examples", max_examples)
    _require_selection(selection)
    _require_non_negative_int("dataset_seed", dataset_seed)
    _require_optional_source_indices(source_indices)
    _require_optional_levels(levels)
    _require_optional_non_empty_str("hf_endpoint", hf_endpoint)
    if data_source not in {"huggingface", "modelscope"}:
        raise ValueError("data_source must be huggingface or modelscope.")
    _require_prompt_template(prompt_template)

    loaded = _load_math_datasets(
        split=split,
        config_name=config_name,
        revision=revision,
        hf_endpoint=hf_endpoint,
        data_source=data_source,
    )
    eligible_rows = [
        (source_index, category, row)
        for source_index, (category, row) in enumerate(_flatten_rows(loaded))
        if levels is None or row.get("level") in levels
    ]
    parseable: list[tuple[int, str, dict[str, Any], str]] = []
    gold_failures = 0
    for source_index, category, row in eligible_rows:
        gold = _normalize_math_gold(row, split=split, source_index=source_index)
        if gold is None:
            gold_failures += 1
            continue
        parseable.append((source_index, category, row, gold))

    selected = _select_rows(
        parseable,
        max_examples=max_examples,
        selection=selection,
        dataset_seed=dataset_seed,
        source_indices=source_indices,
    )
    examples = tuple(
        _to_example(
            source_index=source_index,
            category=category,
            row=row,
            gold=gold,
            split=split,
            prompt_template=prompt_template,
        )
        for source_index, category, row, gold in selected
    )
    fingerprints = [
        value for value in (getattr(dataset, "_fingerprint", None) for _, dataset in loaded)
        if isinstance(value, str) and value
    ]
    return DatasetLoadResult(
        examples=examples,
        source_count=len(eligible_rows),
        fingerprint="|".join(fingerprints) or None,
        gold_parse_attempt_count=len(eligible_rows),
        gold_parse_failure_count=gold_failures,
    )


def _load_math_datasets(
    *, split: str, config_name: str | None, revision: str | None, hf_endpoint: str | None,
    data_source: str,
) -> list[tuple[str, Sequence[dict[str, Any]]]]:
    if data_source == "modelscope":
        return _load_modelscope_math_dataset(
            split=split, config_name=config_name, revision=revision
        )
    try:
        if hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'MATH loading requires the optional data dependency. Run: pip install -e ".[data]"'
        ) from error

    configs = MATH_CONFIGS if config_name in {None, "all"} else (config_name,)
    return [
        (
            category,
            load_dataset(MATH_DATASET_ID, category, split=split, revision=revision),
        )
        for category in configs
    ]


def _load_modelscope_math_dataset(
    *, split: str, config_name: str | None, revision: str | None,
) -> list[tuple[str, Sequence[dict[str, Any]]]]:
    if config_name not in {None, "all"}:
        raise ValueError("ModelScope MATH loading supports only dataset.config: all.")
    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ModelScope MATH loading requires the optional modelscope dependency. "
            "Run: pip install 'modelscope>=1.13.2'."
        ) from error
    try:
        return [("all", _load_modelscope_cached_arrow_split(split))]
    except RuntimeError:
        pass
    # The inspected competition_math script only downloads data/MATH.zip and
    # yields JSON rows with problem, level, type, and solution fields.
    try:
        dataset = MsDataset.load(
            MODELSCOPE_MATH_DATASET_ID,
            split=split,
            version=revision or "master",
            trust_remote_code=True,
        )
    except TypeError as error:
        if "verification_mode" not in str(error):
            raise
        dataset = _load_modelscope_cached_arrow_split(split)
    return [("all", dataset)]


def _load_modelscope_cached_arrow_split(split: str) -> Sequence[dict[str, Any]]:
    """Read ModelScope's already-built Arrow split for datasets API compatibility.

    ModelScope 1.38 passes a removed ``verification_mode`` keyword to newer
    Hugging Face Datasets. The dataset script has already completed safely at
    this point, so reading its Arrow artifact avoids re-running or modifying
    the script while preserving the exact rows it produced.
    """
    try:
        from datasets import Dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'MATH loading requires the optional data dependency. Run: pip install -e ".[data]"'
        ) from error
    cache_root = Path(
        os.environ.get("MODELSCOPE_CACHE", "~/.cache/modelscope")
    ).expanduser() / "hub" / "datasets"
    paths = sorted(
        cache_root.glob(
            f"opencompass___competition_math/**/competition_math-{split}.arrow"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise RuntimeError(
            "ModelScope built MATH data but its Arrow split was not found. "
            "Retry the load after the download completes."
        )
    return Dataset.from_file(str(paths[-1]))


def _flatten_rows(
    datasets: Sequence[tuple[str, Sequence[dict[str, Any]]]]
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for category, dataset in datasets:
        for row in dataset:
            if not isinstance(row, dict):
                raise ValueError("MATH rows must be mappings.")
            rows.append((category, row))
    return rows


def _normalize_math_gold(row: dict[str, Any], *, split: str, source_index: int) -> str | None:
    solution = row.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"MATH {split} example {source_index} has an invalid solution.")
    return extract_final_boxed_latex_gold(solution)


def _to_example(
    *, source_index: int, category: str, row: dict[str, Any], gold: str,
    split: str, prompt_template: str,
) -> DatasetExample:
    problem = row.get("problem")
    solution = row.get("solution")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"MATH {split} example {source_index} has an invalid problem.")
    assert isinstance(solution, str)
    return DatasetExample(
        dataset_name="math",
        split=split,
        source_index=source_index,
        prompt_id=f"math-{split}-{source_index:06d}",
        question=problem,
        prompt=prompt_template.format(question=problem),
        ground_truth=gold,
        reference_solution=solution,
    )


def _select_rows(
    rows: list[tuple[int, str, dict[str, Any], str]], *, max_examples: int | None,
    selection: str, dataset_seed: int, source_indices: tuple[int, ...] | None,
) -> list[tuple[int, str, dict[str, Any], str]]:
    if source_indices is not None:
        by_source_index = {row[0]: row for row in rows}
        missing = [index for index in source_indices if index not in by_source_index]
        if missing:
            raise ValueError(
                "source_indices contains unavailable or gold-unparseable MATH examples: "
                f"{missing}."
            )
        return [by_source_index[index] for index in source_indices]
    limit = len(rows) if max_examples is None else min(len(rows), max_examples)
    if selection == "first":
        return rows[:limit]
    return [rows[index] for index in sorted(Random(dataset_seed).sample(range(len(rows)), limit))]


def _require_non_empty_str(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_non_empty_str(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer or None.")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_selection(value: object) -> None:
    if value not in {"first", "random"}:
        raise ValueError("selection must be one of: first, random.")


def _require_optional_source_indices(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or not value:
        raise ValueError("source_indices must be a non-empty tuple of non-negative integers or None.")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in value):
        raise ValueError("source_indices must be a non-empty tuple of non-negative integers or None.")
    if len(set(value)) != len(value):
        raise ValueError("source_indices must not contain duplicates.")


def _require_optional_levels(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or not value or any(not isinstance(level, str) or not level.strip() for level in value):
        raise ValueError("levels must be a non-empty tuple of non-empty strings or None.")


def _require_prompt_template(prompt_template: object) -> None:
    _require_non_empty_str("prompt_template", prompt_template)
    if "{question}" not in prompt_template:
        raise ValueError("prompt_template must include a {question} placeholder.")
    try:
        prompt_template.format(question="example")
    except (KeyError, ValueError) as error:
        raise ValueError("prompt_template must be format-compatible with {question}.") from error
