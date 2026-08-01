"""Validated configuration objects for RewardScope runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from rewardscope.rewards import RewardConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    tokenizer_name: str | None = None
    prompt_format: Literal["chat", "plain", "auto"] = "auto"
    context_window: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_optional_non_empty_str("tokenizer_name", self.tokenizer_name)
        if self.prompt_format not in {"chat", "plain", "auto"}:
            raise ValueError("prompt_format must be one of: chat, plain, auto.")
        _require_optional_positive_int("context_window", self.context_window)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    config: str | None
    split: str
    revision: str | None = None
    max_examples: int | None = None
    selection: Literal["first", "random"] = "first"
    dataset_seed: int = 0
    source_indices: tuple[int, ...] | None = None
    levels: tuple[str, ...] | None = None
    hf_endpoint: str | None = None
    data_source: Literal["huggingface", "modelscope"] = "huggingface"
    prompt_template: Literal[
        "baseline", "strict", "zero_shot_boxed", "gsm8k_zero_shot_boxed", "gsm8k_cot_4shot", "gsm8k_cot_4shot_terminal",
        "gsm8k_cot_4shot_multiturn_terminal",
    ] = "baseline"

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_non_empty_str("split", self.split)
        _require_optional_non_empty_str("config", self.config)
        _require_optional_non_empty_str("revision", self.revision)
        _require_optional_non_empty_str("hf_endpoint", self.hf_endpoint)
        if self.data_source not in {"huggingface", "modelscope"}:
            raise ValueError("data_source must be huggingface or modelscope.")
        _require_optional_positive_int("max_examples", self.max_examples)
        if self.selection not in {"first", "random"}:
            raise ValueError("selection must be one of: first, random.")
        if self.prompt_template not in {
            "baseline", "strict", "zero_shot_boxed", "gsm8k_zero_shot_boxed", "gsm8k_cot_4shot", "gsm8k_cot_4shot_terminal",
            "gsm8k_cot_4shot_multiturn_terminal",
        }:
            raise ValueError(
                "prompt_template must be one of: baseline, strict, zero_shot_boxed, gsm8k_zero_shot_boxed, gsm8k_cot_4shot, "
                "gsm8k_cot_4shot_terminal, gsm8k_cot_4shot_multiturn_terminal."
            )
        _require_non_negative_int("dataset_seed", self.dataset_seed)
        _require_optional_source_indices(self.source_indices)
        _require_optional_levels(self.levels)
        if self.source_indices is not None and self.max_examples is not None:
            raise ValueError("max_examples must be None when source_indices is set.")
        if self.source_indices is not None and self.selection != "first":
            raise ValueError("selection must be first when source_indices is set.")


@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int
    generation_seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_size: int

    def __post_init__(self) -> None:
        _require_positive_int("num_samples", self.num_samples)
        _require_non_negative_int("generation_seed", self.generation_seed)
        _require_non_negative_finite_number("temperature", self.temperature)
        _require_probability("top_p", self.top_p)
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        _require_positive_int("batch_size", self.batch_size)
        if self.temperature == 0 and self.num_samples != 1:
            raise ValueError("num_samples must be 1 when temperature is 0.")


@dataclass(frozen=True)
class OutputConfig:
    run_id: str
    output_dir: Path
    keep_failed_run: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_str("run_id", self.run_id)
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path.")
        if not str(self.output_dir):
            raise ValueError("output_dir must not be empty.")
        if not isinstance(self.keep_failed_run, bool):
            raise ValueError("keep_failed_run must be a boolean.")


@dataclass(frozen=True)
class AnalysisConfig:
    strict: bool = False
    k_values: tuple[int, ...] = (1, 4, 8)
    write_plots: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be a boolean.")
        if not isinstance(self.write_plots, bool):
            raise ValueError("write_plots must be a boolean.")
        _require_k_values(self.k_values)


@dataclass(frozen=True)
class VerificationConfig:
    """Verification policy used by dataset-specific verification backends."""

    backend: Literal["numeric", "math_verify", "math_verify_latex"] = "numeric"
    mode: Literal["evaluation", "training"] = "evaluation"

    def __post_init__(self) -> None:
        if self.backend not in {"numeric", "math_verify", "math_verify_latex"}:
            raise ValueError("backend must be numeric, math_verify, or math_verify_latex.")
        if self.mode not in {"evaluation", "training"}:
            raise ValueError("mode must be evaluation or training.")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    dataset: DatasetConfig
    sampling: SamplingConfig
    reward: RewardConfig
    output: OutputConfig
    analysis: AnalysisConfig
    verification: VerificationConfig = VerificationConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig):
            raise ValueError("model must be a ModelConfig.")
        if not isinstance(self.dataset, DatasetConfig):
            raise ValueError("dataset must be a DatasetConfig.")
        if not isinstance(self.sampling, SamplingConfig):
            raise ValueError("sampling must be a SamplingConfig.")
        if not isinstance(self.reward, RewardConfig):
            raise ValueError("reward must be a RewardConfig.")
        if not isinstance(self.output, OutputConfig):
            raise ValueError("output must be an OutputConfig.")
        if not isinstance(self.analysis, AnalysisConfig):
            raise ValueError("analysis must be an AnalysisConfig.")
        if not isinstance(self.verification, VerificationConfig):
            raise ValueError("verification must be a VerificationConfig.")
        if (
            self.dataset.name.lower() == "math"
            and self.verification.backend != "math_verify_latex"
        ):
            raise ValueError("MATH runs require the math_verify_latex verification backend.")
        if (
            self.verification.backend == "math_verify_latex"
            and self.verification.mode != "training"
        ):
            raise ValueError("math_verify_latex runs require training mode for boxed-only verification.")
        if any(k > self.sampling.num_samples for k in self.analysis.k_values):
            raise ValueError("analysis.k_values cannot exceed sampling.num_samples.")


def _require_non_empty_str(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_non_empty_str(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is not None:
        _require_positive_int(name, value)


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


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
    if not isinstance(value, tuple) or not value:
        raise ValueError("levels must be a non-empty tuple of non-empty strings or None.")
    if any(not isinstance(level, str) or not level.strip() for level in value):
        raise ValueError("levels must be a non-empty tuple of non-empty strings or None.")
    if len(set(value)) != len(value):
        raise ValueError("levels must not contain duplicates.")


def _require_non_negative_finite_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number.")


def _require_probability(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be a number in the interval (0, 1].")


def _require_k_values(value: object) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in value):
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if len(set(value)) != len(value):
        raise ValueError("k_values must not contain duplicates.")
