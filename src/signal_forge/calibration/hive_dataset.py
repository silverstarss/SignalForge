"""Standalone HIVE dataset calibration without optimizer or GRPO execution."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from rewardscope.config import ModelConfig, SamplingConfig
from rewardscope.datasets.math import (
    MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
    load_math_result,
)
from rewardscope.io.atomic import atomic_write_json, atomic_write_jsonl
from rewardscope.sampling.schema import GeneratedResponse
from rewardscope.sampling.transformers import TransformersSampler
from rewardscope.verification import extract_final_boxed_latex_gold

from signal_forge.rewards.math_verify_adapter import compute_score


GROUP_SIZE = 8
DEFAULT_SAMPLE_SIZE = 256
DEFAULT_SEED = 42
DEFAULT_MAX_RESPONSE_LENGTH = 768
DAPO_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
CANONICAL_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    "Put your final answer in \\boxed{{...}}.\n\n"
    "{problem}"
)
DAPO_SOURCE_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_SOURCE_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
DatasetChoice = Literal["math", "dapo", "dapo_math"]


@dataclass(frozen=True)
class CalibrationPrompt:
    prompt_id: str
    dataset_source: Literal["math", "dapo"]
    source_row_id: str
    raw_prompt: tuple[dict[str, str], ...]
    canonical_prompt: str
    messages: tuple[dict[str, str], ...]
    source_ground_truth: str
    ground_truth: str

    def __post_init__(self) -> None:
        prefix = f"{self.dataset_source}:"
        if not isinstance(self.prompt_id, str) or not self.prompt_id.startswith(prefix):
            raise ValueError(f"prompt_id must start with {prefix!r}")
        if not isinstance(self.source_row_id, str) or not self.source_row_id.strip():
            raise ValueError("source_row_id must be a non-empty string")
        _validate_message_sequence("raw_prompt", self.raw_prompt)
        _validate_message_sequence("messages", self.messages)
        if not isinstance(self.canonical_prompt, str) or not self.canonical_prompt.strip():
            raise ValueError("canonical_prompt must be a non-empty string")
        expected_messages = ({"role": "user", "content": self.canonical_prompt},)
        if self.messages != expected_messages:
            raise ValueError("messages must contain only the canonical user prompt")
        if (
            not isinstance(self.source_ground_truth, str)
            or not self.source_ground_truth.strip()
        ):
            raise ValueError("source_ground_truth must be a non-empty string")
        if not isinstance(self.ground_truth, str) or not self.ground_truth.strip():
            raise ValueError("ground_truth must be a non-empty string")

    def selection_row(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "dataset_source": self.dataset_source,
            "source_row_id": self.source_row_id,
            "raw_prompt": list(self.raw_prompt),
            "canonical_prompt": self.canonical_prompt,
            "messages": list(self.messages),
            "source_ground_truth": self.source_ground_truth,
            "ground_truth": self.ground_truth,
        }


@dataclass(frozen=True)
class DatasetInventory:
    dataset: DatasetChoice
    source_rows: dict[str, int]
    unique_prompts: dict[str, int]
    duplicate_rows_removed: dict[str, int]
    math_gold_parse_failures: int = 0
    dapo_prompt_transform: str = "extract_problem_then_canonical_math_prompt_v1"
    dapo_ground_truth_transform: str = "wrap_boxed_and_validate_before_generation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_fixed_subset(
    prompts: Sequence[CalibrationPrompt],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[CalibrationPrompt, ...]:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ordered = sorted(prompts, key=lambda item: item.prompt_id)
    prompt_ids = [item.prompt_id for item in ordered]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("candidate prompts contain duplicate stable prompt_ids")
    if sample_size > len(ordered):
        raise ValueError(
            f"sample_size={sample_size} exceeds available unique prompts={len(ordered)}"
        )
    indices = random.Random(seed).sample(range(len(ordered)), sample_size)
    return tuple(ordered[index] for index in sorted(indices))


def format_canonical_math_prompt(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    return CANONICAL_MATH_PROMPT_TEMPLATE.format(problem=problem.strip())


def select_balanced_merged_subset(
    math_prompts: Sequence[CalibrationPrompt],
    dapo_prompts: Sequence[CalibrationPrompt],
    *,
    sample_size: int,
    seed: int,
) -> tuple[CalibrationPrompt, ...]:
    if sample_size % 2:
        raise ValueError("balanced merged calibration requires an even sample_size")
    per_source = sample_size // 2
    selected = (
        *select_fixed_subset(math_prompts, sample_size=per_source, seed=seed),
        *select_fixed_subset(dapo_prompts, sample_size=per_source, seed=seed),
    )
    return tuple(sorted(selected, key=lambda item: item.prompt_id))


def load_calibration_prompts(
    dataset: DatasetChoice,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    math_split: str = "train",
    dapo_split: str = "train",
    math_data_source: str = "huggingface",
    hf_endpoint: str | None = None,
    dapo_parquet: str | Path | None = None,
    balanced_merged: bool = False,
) -> tuple[tuple[CalibrationPrompt, ...], DatasetInventory]:
    if dataset not in {"math", "dapo", "dapo_math"}:
        raise ValueError("dataset must be one of: math, dapo, dapo_math")
    if balanced_merged and dataset != "dapo_math":
        raise ValueError("balanced_merged is supported only for dapo_math")

    math_prompts: tuple[CalibrationPrompt, ...] = ()
    dapo_prompts: tuple[CalibrationPrompt, ...] = ()
    source_rows: dict[str, int] = {}
    unique_prompts: dict[str, int] = {}
    duplicates: dict[str, int] = {}
    math_failures = 0

    if dataset in {"math", "dapo_math"}:
        math_prompts, math_metadata = load_math_calibration_prompts(
            split=math_split,
            data_source=math_data_source,
            hf_endpoint=hf_endpoint,
        )
        source_rows["math"] = math_metadata["source_rows"]
        unique_prompts["math"] = len(math_prompts)
        duplicates["math"] = 0
        math_failures = math_metadata["gold_parse_failures"]

    if dataset in {"dapo", "dapo_math"}:
        dapo_prompts, dapo_metadata = load_dapo_calibration_prompts(
            split=dapo_split,
            parquet_path=dapo_parquet,
        )
        source_rows["dapo"] = dapo_metadata["source_rows"]
        unique_prompts["dapo"] = len(dapo_prompts)
        duplicates["dapo"] = dapo_metadata["duplicate_rows_removed"]

    if balanced_merged:
        selected = select_balanced_merged_subset(
            math_prompts,
            dapo_prompts,
            sample_size=sample_size,
            seed=seed,
        )
    else:
        selected = select_fixed_subset(
            (*math_prompts, *dapo_prompts),
            sample_size=sample_size,
            seed=seed,
        )
    selected = _canonicalize_selected_ground_truths(selected)
    return selected, DatasetInventory(
        dataset=dataset,
        source_rows=source_rows,
        unique_prompts=unique_prompts,
        duplicate_rows_removed=duplicates,
        math_gold_parse_failures=math_failures,
    )


def load_math_calibration_prompts(
    *,
    split: str = "train",
    data_source: str = "huggingface",
    hf_endpoint: str | None = None,
) -> tuple[tuple[CalibrationPrompt, ...], dict[str, int]]:
    result = load_math_result(
        config_name="all",
        split=split,
        revision=None,
        max_examples=None,
        selection="first",
        dataset_seed=0,
        hf_endpoint=hf_endpoint,
        data_source=data_source,
        prompt_template=MATH_ZERO_SHOT_BOXED_PROMPT_TEMPLATE,
    )
    prompts = tuple(
        CalibrationPrompt(
            prompt_id=f"math:{example.source_index}",
            dataset_source="math",
            source_row_id=str(example.source_index),
            raw_prompt=({"role": "user", "content": example.prompt},),
            canonical_prompt=format_canonical_math_prompt(example.question),
            messages=(
                {
                    "role": "user",
                    "content": format_canonical_math_prompt(example.question),
                },
            ),
            source_ground_truth=example.ground_truth,
            ground_truth=example.ground_truth,
        )
        for example in result.examples
    )
    return prompts, {
        "source_rows": result.source_count,
        "gold_parse_failures": result.gold_parse_failure_count or 0,
    }


def load_dapo_calibration_prompts(
    *,
    split: str = "train",
    parquet_path: str | Path | None = None,
) -> tuple[tuple[CalibrationPrompt, ...], dict[str, int]]:
    dataset = _load_dapo_dataset(split=split, parquet_path=parquet_path)
    by_id: dict[str, CalibrationPrompt] = {}
    source_rows = 0
    for row in dataset:
        source_rows += 1
        prompt = normalize_dapo_row(row)
        previous = by_id.get(prompt.prompt_id)
        if previous is not None and previous != prompt:
            raise ValueError(
                f"DAPO duplicate row id has conflicting prompt or ground truth: {prompt.prompt_id}"
            )
        by_id[prompt.prompt_id] = prompt
    prompts = tuple(by_id[prompt_id] for prompt_id in sorted(by_id))
    return prompts, {
        "source_rows": source_rows,
        "duplicate_rows_removed": source_rows - len(prompts),
    }


def normalize_dapo_row(row: Mapping[str, Any]) -> CalibrationPrompt:
    if not isinstance(row, Mapping):
        raise ValueError("DAPO row must be a mapping")
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError("DAPO row is missing extra_info")
    row_id = extra_info.get("index")
    if row_id is None or not str(row_id).strip():
        raise ValueError("DAPO row is missing stable extra_info.index")
    raw_prompt = _normalize_messages(row.get("prompt"))
    problem = extract_dapo_problem(raw_prompt)
    canonical_prompt = format_canonical_math_prompt(problem)

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise ValueError(f"DAPO row {row_id!r} is missing reward_model")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(f"DAPO row {row_id!r} has invalid reward_model.ground_truth")
    normalized_id = str(row_id).strip()
    return CalibrationPrompt(
        prompt_id=f"dapo:{normalized_id}",
        dataset_source="dapo",
        source_row_id=normalized_id,
        raw_prompt=raw_prompt,
        canonical_prompt=canonical_prompt,
        messages=({"role": "user", "content": canonical_prompt},),
        source_ground_truth=ground_truth,
        ground_truth=normalize_dapo_ground_truth(ground_truth),
    )


def extract_dapo_problem(messages: Sequence[Mapping[str, str]]) -> str:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("DAPO raw prompt must end with a user message")
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("DAPO user prompt content must be a string")
    prefix = f"{DAPO_SOURCE_PREFIX}\n\n"
    suffix = f"\n\n{DAPO_SOURCE_SUFFIX}"
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise ValueError(
            "DAPO user prompt does not match the audited Answer-format source template"
        )
    problem = content[len(prefix) : -len(suffix)].strip()
    if not problem:
        raise ValueError("DAPO source prompt contains an empty math problem")
    return problem


def normalize_dapo_ground_truth(ground_truth: str) -> str:
    """Adapt DAPO's bare answer into the frozen boxed-LaTeX gold contract."""
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("DAPO ground_truth must be a non-empty string")
    normalized = ground_truth.strip()
    if r"\boxed{" in normalized:
        return normalized
    return rf"\boxed{{{normalized}}}"


def _canonicalize_selected_ground_truths(
    prompts: Sequence[CalibrationPrompt],
) -> tuple[CalibrationPrompt, ...]:
    canonical: list[CalibrationPrompt] = []
    for prompt in prompts:
        if prompt.dataset_source != "dapo":
            canonical.append(prompt)
            continue
        parsed = extract_final_boxed_latex_gold(prompt.ground_truth)
        if parsed is None:
            raise ValueError(
                "selected DAPO ground truth is not parseable by the frozen verifier: "
                f"prompt_id={prompt.prompt_id!r}, "
                f"source_ground_truth={prompt.source_ground_truth!r}"
            )
        canonical.append(replace(prompt, ground_truth=parsed))
    return tuple(canonical)


def evaluate_generated_responses(
    prompts: Sequence[CalibrationPrompt],
    responses: Sequence[GeneratedResponse],
    *,
    verifier: Callable[..., Mapping[str, Any]] = compute_score,
) -> list[dict[str, Any]]:
    from signal_forge.hive.state import ZeroVarianceType, classify_zero_variance

    if len(responses) != len(prompts) * GROUP_SIZE:
        raise ValueError("sampler must return exactly eight responses per prompt")
    grouped: list[list[GeneratedResponse]] = [[] for _ in prompts]
    seen: set[tuple[int, int]] = set()
    for response in responses:
        key = (response.prompt_index, response.sample_index)
        if key in seen:
            raise ValueError(f"duplicate generated response index: {key}")
        seen.add(key)
        if not 0 <= response.prompt_index < len(prompts):
            raise ValueError("generated response prompt_index is out of range")
        if not 0 <= response.sample_index < GROUP_SIZE:
            raise ValueError("generated response sample_index is out of range")
        grouped[response.prompt_index].append(response)

    rows: list[dict[str, Any]] = []
    for prompt_index, (prompt, group) in enumerate(zip(prompts, grouped, strict=True)):
        ordered = sorted(group, key=lambda item: item.sample_index)
        if [item.sample_index for item in ordered] != list(range(GROUP_SIZE)):
            raise ValueError(f"prompt {prompt_index} does not have sample indices 0..7")
        verifier_results = [
            dict(
                verifier(
                    "math",
                    item.response,
                    prompt.ground_truth,
                    extra_info={"source_dataset": "math"},
                )
            )
            for item in ordered
        ]
        for item in verifier_results:
            _validate_three_state_verifier_result(item)
        rewards = [float(item["reward"]) for item in verifier_results]
        classification = classify_zero_variance(rewards, group_size=GROUP_SIZE)
        prompt_token_counts = {item.prompt_tokens for item in ordered}
        if len(prompt_token_counts) != 1:
            raise ValueError("responses for one prompt disagree on prompt token count")
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "dataset_source": prompt.dataset_source,
                "raw_prompt": list(prompt.raw_prompt),
                "canonical_prompt": prompt.canonical_prompt,
                "source_ground_truth": prompt.source_ground_truth,
                "ground_truth": prompt.ground_truth,
                "rewards": rewards,
                "correct_count": sum(bool(item["correct"]) for item in verifier_results),
                "easy_zero_var": classification.zero_variance_type is ZeroVarianceType.EASY,
                "hard_zero_var": classification.zero_variance_type is ZeroVarianceType.HARD,
                "other_zero_var": classification.zero_variance_type is ZeroVarianceType.OTHER,
                "effective": not classification.zero_variance,
                "prompt_token_count": prompt_token_counts.pop(),
                "response_lengths": [item.response_tokens for item in ordered],
                "extraction_failure_count": sum(
                    not bool(item["extracted"]) for item in verifier_results
                ),
                "finish_reasons": [item.finish_reason for item in ordered],
                "responses": [item.response for item in ordered],
                "verifier_results": verifier_results,
            }
        )
    return rows


def summarize_calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("calibration summary requires at least one prompt")
    return {
        **_summarize_rows(rows),
        "by_dataset_source": {
            source: _summarize_rows(
                [row for row in rows if row["dataset_source"] == source]
            )
            for source in sorted({str(row["dataset_source"]) for row in rows})
        },
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_count = len(rows)
    rewards = [float(value) for row in rows for value in row["rewards"]]
    lengths = [int(value) for row in rows for value in row["response_lengths"]]
    histogram = Counter(int(row["correct_count"]) for row in rows)
    easy = sum(bool(row["easy_zero_var"]) for row in rows)
    hard = sum(bool(row["hard_zero_var"]) for row in rows)
    other = sum(bool(row["other_zero_var"]) for row in rows)
    effective = sum(bool(row["effective"]) for row in rows)
    finishes = [
        str(finish)
        for row in rows
        for finish in row["finish_reasons"]
    ]
    verifier_results = [
        result
        for row in rows
        for result in row["verifier_results"]
    ]
    eos_count = sum(finish == "eos" for finish in finishes)
    length_count = sum(finish == "length" for finish in finishes)
    extraction_failures = sum(not bool(result["extracted"]) for result in verifier_results)
    extraction_failures_eos = sum(
        finish == "eos" and not bool(result["extracted"])
        for finish, result in zip(finishes, verifier_results, strict=True)
    )
    extraction_failures_length = sum(
        finish == "length" and not bool(result["extracted"])
        for finish, result in zip(finishes, verifier_results, strict=True)
    )
    return {
        "prompt_count": prompt_count,
        "response_count": len(rewards),
        "generated_response_tokens": sum(lengths),
        "correct_count_histogram": {
            f"{count}/{GROUP_SIZE}": histogram.get(count, 0)
            for count in range(GROUP_SIZE + 1)
        },
        "easy_zero_var_ratio": easy / prompt_count,
        "hard_zero_var_ratio": hard / prompt_count,
        "other_zero_var_ratio": other / prompt_count,
        "effective_mixed_ratio": effective / prompt_count,
        "eos_finish_ratio": eos_count / len(rewards),
        "length_limit_finish_ratio": length_count / len(rewards),
        "truncation_ratio": length_count / len(rewards),
        "extraction_failure_ratio": extraction_failures / len(rewards),
        "extraction_failure_given_eos": _ratio_or_none(
            extraction_failures_eos, eos_count
        ),
        "extraction_failure_given_length_truncation": _ratio_or_none(
            extraction_failures_length, length_count
        ),
        "response_token_length_statistics": _length_statistics(lengths),
    }


def run_calibration(
    prompts: Sequence[CalibrationPrompt],
    *,
    model_path: str,
    seed: int,
    max_response_length: int,
    batch_size: int,
    sampler: TransformersSampler | None = None,
) -> list[dict[str, Any]]:
    if sampler is None:
        sampler = TransformersSampler.from_pretrained(
            ModelConfig(name=model_path, prompt_format="chat")
        )
    responses = sampler.generate(
        [list(prompt.messages) for prompt in prompts],
        SamplingConfig(
            num_samples=GROUP_SIZE,
            generation_seed=seed,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_response_length,
            batch_size=batch_size,
        ),
    )
    return evaluate_generated_responses(prompts, responses)


def write_calibration_inputs(
    output_dir: str | Path,
    *,
    prompts: Sequence[CalibrationPrompt],
    inventory: DatasetInventory,
    config: Mapping[str, Any],
) -> Path:
    destination = _prepare_output_dir(output_dir)
    atomic_write_json(destination / "config.json", dict(config))
    atomic_write_json(destination / "inventory.json", inventory.to_dict())
    atomic_write_jsonl(
        destination / "selected_prompts.jsonl",
        [prompt.selection_row() for prompt in prompts],
    )
    return destination


def write_calibration_results(
    output_dir: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    destination = Path(output_dir)
    atomic_write_jsonl(destination / "prompt_results.jsonl", rows)
    atomic_write_json(destination / "aggregate.json", summarize_calibration(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed G=8 HIVE dataset-calibration rollouts without training."
    )
    parser.add_argument("--dataset", choices=("math", "dapo", "dapo_math"), required=True)
    parser.add_argument("--model-path", default=os.environ.get("QWEN25_3B_LOCAL_DIR"))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-response-length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--math-split", default="train")
    parser.add_argument("--dapo-split", default="train")
    parser.add_argument(
        "--math-data-source", choices=("huggingface", "modelscope"), default="huggingface"
    )
    parser.add_argument("--hf-endpoint")
    parser.add_argument("--dapo-parquet")
    parser.add_argument(
        "--balanced-merged",
        action="store_true",
        help="For dapo_math, select exactly half MATH and half DAPO prompts.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only load/deduplicate/sample and write selected_prompts.jsonl.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit generation without CUDA; disabled by default for the 3B target.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_response_length <= 0 or args.batch_size <= 0:
        raise ValueError("max_response_length and batch_size must be positive")
    prompts, inventory = load_calibration_prompts(
        args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        math_split=args.math_split,
        dapo_split=args.dapo_split,
        math_data_source=args.math_data_source,
        hf_endpoint=args.hf_endpoint,
        dapo_parquet=args.dapo_parquet,
        balanced_merged=args.balanced_merged,
    )
    output_dir = args.output_dir or (
        f"artifacts/calibration/hive_dataset/{args.dataset}"
        f"_n{args.sample_size}_seed{args.seed}_r{args.max_response_length}"
    )
    config = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "temperature": 1.0,
        "group_size": GROUP_SIZE,
        "max_response_length": args.max_response_length,
        "batch_size": args.batch_size,
        "optimizer_updates": 0,
        "grpo_training": False,
        "math_split": args.math_split,
        "dapo_split": args.dapo_split,
        "math_data_source": args.math_data_source,
        "dapo_dataset_id": DAPO_DATASET_ID,
        "dapo_parquet": args.dapo_parquet,
        "balanced_merged": args.balanced_merged,
        "selected_source_counts": dict(
            sorted(Counter(prompt.dataset_source for prompt in prompts).items())
        ),
        "canonical_prompt_template": format_canonical_math_prompt("{problem}"),
        "dapo_prompt_transform": inventory.dapo_prompt_transform,
        "dapo_ground_truth_transform": inventory.dapo_ground_truth_transform,
    }
    if not args.prepare_only:
        if not args.model_path:
            raise ValueError("--model-path or QWEN25_3B_LOCAL_DIR is required")
        if not args.allow_cpu:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required unless --allow-cpu is supplied")
    destination = write_calibration_inputs(
        output_dir,
        prompts=prompts,
        inventory=inventory,
        config=config,
    )
    if args.prepare_only:
        return 0
    rows = run_calibration(
        prompts,
        model_path=args.model_path,
        seed=args.seed,
        max_response_length=args.max_response_length,
        batch_size=args.batch_size,
    )
    write_calibration_results(destination, rows=rows)
    return 0


def _validate_three_state_verifier_result(result: Mapping[str, Any]) -> None:
    if not isinstance(result.get("correct"), bool) or not isinstance(
        result.get("extracted"), bool
    ):
        raise ValueError("verifier result must contain boolean correct and extracted fields")
    reward = result.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("verifier result must contain a numeric reward")
    expected = 1.0 if result["correct"] else 0.1 if result["extracted"] else 0.0
    if float(reward) != expected:
        raise ValueError(
            "verifier result violates frozen three-state semantics: "
            f"correct={result['correct']}, extracted={result['extracted']}, "
            f"reward={reward}, expected={expected}"
        )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_message_sequence(
    name: str, messages: Sequence[Mapping[str, str]]
) -> None:
    if not isinstance(messages, tuple) or not messages:
        raise ValueError(f"{name} must be a non-empty tuple")
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError(f"{name} messages must contain only role and content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"{name} contains an invalid role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"{name} message content must be non-empty")


def _load_dapo_dataset(*, split: str, parquet_path: str | Path | None):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError("DAPO calibration requires the datasets package") from error
    if parquet_path is not None:
        return load_dataset("parquet", data_files=str(parquet_path), split="train")
    return load_dataset(DAPO_DATASET_ID, split=split)


def _normalize_messages(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("DAPO prompt must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("DAPO prompt messages must be mappings")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"DAPO prompt has invalid role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DAPO prompt message content must be non-empty")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("DAPO prompt must end with a user message")
    return tuple(messages)


def _length_statistics(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("response length statistics require non-empty values")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
    }


def _quantile(ordered: Sequence[int], probability: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _prepare_output_dir(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"calibration output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
