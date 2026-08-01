"""Strict YAML-to-dataclass loading for RewardScope experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rewardscope.config.schema import (
    AnalysisConfig,
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    SamplingConfig,
    VerificationConfig,
)
from rewardscope.rewards import RewardConfig


def load_run_config(path: str | Path) -> RunConfig:
    """Load one strict YAML experiment configuration into validated dataclasses."""
    config, _ = load_run_config_with_requested(path)
    return config


def load_run_config_with_requested(path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Load strict YAML and retain its requested fields for experiment snapshots."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as input_file:
            raw_config = yaml.safe_load(input_file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {source}.") from error

    config = _require_mapping(raw_config, "configuration")
    _require_exact_keys(
        config,
        required={"model", "dataset", "sampling", "output"},
        optional={"reward", "analysis", "verification"},
        context="configuration",
    )

    return RunConfig(
        model=_load_model_config(_require_mapping(config["model"], "model")),
        dataset=_load_dataset_config(_require_mapping(config["dataset"], "dataset")),
        sampling=_load_sampling_config(
            _require_mapping(config["sampling"], "sampling")
        ),
        reward=_load_reward_config(
            _require_mapping(config.get("reward", {}), "reward")
        ),
        output=_load_output_config(_require_mapping(config["output"], "output")),
        analysis=_load_analysis_config(
            _require_mapping(config.get("analysis", {}), "analysis")
        ),
        verification=_load_verification_config(
            _require_mapping(config.get("verification", {}), "verification")
        ),
    ), config


def _load_model_config(config: dict[str, Any]) -> ModelConfig:
    _require_exact_keys(
        config,
        required={"name"},
        optional={"tokenizer_name", "prompt_format", "context_window"},
        context="model",
    )
    return ModelConfig(**config)


def _load_dataset_config(config: dict[str, Any]) -> DatasetConfig:
    _require_exact_keys(
        config,
        required={"name", "config", "split"},
        optional={
            "revision", "max_examples", "selection", "dataset_seed", "source_indices",
            "levels", "hf_endpoint", "data_source", "prompt_template",
        },
        context="dataset",
    )
    values = dict(config)
    if "source_indices" in values:
        if not isinstance(values["source_indices"], list):
            raise ValueError("dataset.source_indices must be a list of non-negative integers.")
        values["source_indices"] = tuple(values["source_indices"])
    if "levels" in values:
        if not isinstance(values["levels"], list):
            raise ValueError("dataset.levels must be a list of non-empty strings.")
        values["levels"] = tuple(values["levels"])
    return DatasetConfig(**values)


def _load_sampling_config(config: dict[str, Any]) -> SamplingConfig:
    _require_exact_keys(
        config,
        required={
            "num_samples",
            "generation_seed",
            "temperature",
            "top_p",
            "max_new_tokens",
            "batch_size",
        },
        optional=set(),
        context="sampling",
    )
    return SamplingConfig(**config)


def _load_reward_config(config: dict[str, Any]) -> RewardConfig:
    _require_exact_keys(
        config,
        required=set(),
        optional={
            "correct_answer_reward",
            "incorrect_answer_reward",
            "format_compliance_reward",
            "length_penalty_per_token",
        },
        context="reward",
    )
    return RewardConfig(**config)


def _load_output_config(config: dict[str, Any]) -> OutputConfig:
    _require_exact_keys(
        config,
        required={"run_id", "output_dir"},
        optional={"keep_failed_run"},
        context="output",
    )
    output_dir = config["output_dir"]
    if not isinstance(output_dir, str):
        raise ValueError("output.output_dir must be a string.")
    return OutputConfig(
        run_id=config["run_id"],
        output_dir=Path(output_dir),
        keep_failed_run=config.get("keep_failed_run", False),
    )


def _load_analysis_config(config: dict[str, Any]) -> AnalysisConfig:
    _require_exact_keys(
        config,
        required=set(),
        optional={"strict", "k_values", "write_plots"},
        context="analysis",
    )
    values = dict(config)
    if "k_values" in values:
        if not isinstance(values["k_values"], list):
            raise ValueError("analysis.k_values must be a list of positive integers.")
        values["k_values"] = tuple(values["k_values"])
    return AnalysisConfig(**values)


def _load_verification_config(config: dict[str, Any]) -> VerificationConfig:
    _require_exact_keys(
        config,
        required=set(),
        optional={"backend", "mode"},
        context="verification",
    )
    return VerificationConfig(**config)


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _require_exact_keys(
    config: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}.")
    unknown = sorted(config.keys() - required - optional)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}.")
