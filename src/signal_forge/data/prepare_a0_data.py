"""Prepare tiny Experiment A0 veRL parquet files.

A0 is a pipeline smoke test, not an accuracy experiment. The default export is
small but keeps the frozen A protocol shape: GSM8K + MATH Level 3, boxed prompt,
Math-Verify-compatible gold, independent validation, and veRL parquet schema.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BOXED_INSTRUCTION = "Please reason step by step and put your final answer within \\boxed{}."

_SIGNAL_FORGE_SRC = Path(os.environ.get("SIGNAL_FORGE_SRC", Path(__file__).resolve().parents[2]))
_REWARDSCOPE_DIR = _SIGNAL_FORGE_SRC / "RewardScope"
_REWARDSCOPE_OUTPUTS = _REWARDSCOPE_DIR / "outputs"

DEFAULT_GSM8K_TRAIN_INPUTS = _REWARDSCOPE_OUTPUTS / "gsm8k-qwen-grpo-train-zero-shot-boxed-128" / "inputs.jsonl"
DEFAULT_MATH_TRAIN_INPUTS = _REWARDSCOPE_OUTPUTS / "math-qwen-grpo-train-level-3-64-max768" / "inputs.jsonl"
DEFAULT_GSM8K_VAL_INPUTS = _REWARDSCOPE_OUTPUTS / "gsm8k-qwen-zero-shot-boxed-128" / "inputs.jsonl"
LOCAL_GSM8K_TRAIN_INPUTS = DEFAULT_GSM8K_TRAIN_INPUTS
LOCAL_MATH_TRAIN_INPUTS = DEFAULT_MATH_TRAIN_INPUTS
LOCAL_GSM8K_VAL_INPUTS = DEFAULT_GSM8K_VAL_INPUTS
DEFAULT_MATH_TEST_DIR = _REWARDSCOPE_DIR / "raw" / "competition_math" / "test"
LOCAL_MATH_TEST_DIR = DEFAULT_MATH_TEST_DIR


@dataclass(frozen=True)
class SelectionManifest:
    seed: int
    train_gsm8k: int
    train_math_level_3: int
    val_gsm8k: int
    val_math_level_3: int
    train_file: str
    val_file: str
    train_prompt_ids: list[str]
    val_prompt_ids: list[str]


def _existing(default_path: Path, local_path: Path) -> Path:
    if default_path.exists():
        return default_path
    return local_path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question.strip()}\n\n{BOXED_INSTRUCTION}"}]


def _verl_row(record: dict[str, Any], *, source: str, split: str, math_level: int | None) -> dict[str, Any]:
    question = str(record["question"]).strip()
    ground_truth = str(record["ground_truth"]).strip()
    return {
        "data_source": source,
        "ability": "math",
        "prompt": _prompt(question),
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "prompt_id": str(record["prompt_id"]),
            "source_index": int(record["source_index"]),
            "split": split,
            "math_level": math_level,
            "source_dataset": record.get("dataset_name"),
            "dataset_config": record.get("dataset_config"),
            "revision": record.get("revision"),
            "dataset_version": record.get("dataset_version"),
            "math_type": record.get("math_type"),
            "source_file": record.get("source_file"),
            "question": question,
        },
    }


def _sample_inputs(path: Path, count: int, *, used_prompt_ids: set[str], seed: int) -> list[dict[str, Any]]:
    records = [record for record in _iter_jsonl(path) if record["prompt_id"] not in used_prompt_ids]
    if len(records) < count:
        raise ValueError(f"Need {count} records from {path}, found {len(records)} after excluding overlaps")
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:count]


def _math_test_records(
    count: int,
    *,
    used_prompt_ids: set[str],
    seed: int,
    math_test_dir: Path,
) -> list[dict[str, Any]]:
    from rewardscope.verification.math_verify import extract_final_boxed_latex_gold

    if not math_test_dir.exists():
        raise FileNotFoundError(
            "MATH validation requires a local competition_math test directory. "
            f"Expected {math_test_dir}. Pass --math-test-dir if your cache lives elsewhere."
        )

    json_files = sorted(math_test_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under MATH test directory: {math_test_dir}")

    candidates: list[dict[str, Any]] = []
    skipped_unparseable_gold = 0
    for index, json_file in enumerate(json_files):
        item = json.loads(json_file.read_text(encoding="utf-8"))
        if item.get("level") != "Level 3":
            continue
        gold = extract_final_boxed_latex_gold(str(item["solution"]))
        if gold is None:
            skipped_unparseable_gold += 1
            continue
        prompt_id = f"math-test-{index:06d}"
        if prompt_id in used_prompt_ids:
            continue
        candidates.append(
            {
                "prompt_id": prompt_id,
                "source_index": index,
                "question": item["problem"],
                "ground_truth": gold,
                "dataset_name": "math",
                "dataset_config": "all",
                "split": "test",
                "revision": "modelscope/opencompass___competition_math:test",
                "math_type": item.get("type"),
                "source_file": str(json_file.relative_to(math_test_dir)),
            }
        )
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} parseable MATH Level 3 validation rows, found {len(candidates)} "
            f"under {math_test_dir}; skipped_unparseable_gold={skipped_unparseable_gold}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def _check_prompt_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [row["extra_info"]["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"Duplicate prompt_id in {label}")


def _check_no_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_ids = {row["extra_info"]["prompt_id"] for row in train_rows}
    val_ids = {row["extra_info"]["prompt_id"] for row in val_rows}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise AssertionError(f"train/validation prompt_id overlap: {overlap[:20]}")


def _check_train_ratio(train_rows: list[dict[str, Any]], expected_gsm8k: int, expected_math: int) -> None:
    gsm8k = sum(row["data_source"] == "gsm8k" for row in train_rows)
    math = sum(row["data_source"] == "math_level_3" for row in train_rows)
    if gsm8k != expected_gsm8k or math != expected_math:
        raise AssertionError(f"Unexpected train source counts: gsm8k={gsm8k}, math_level_3={math}")
    if expected_gsm8k * 2 != expected_math * 3:
        raise AssertionError("A protocol expects train GSM8K:MATH Level 3 = 60:40 exactly")


def _check_gold_parseable(rows: list[dict[str, Any]]) -> None:
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier, MathVerifyNumericVerifier

    numeric = MathVerifyNumericVerifier(mode="training")
    latex = MathVerifyLatexVerifier(mode="training")
    for row in rows:
        gt = row["reward_model"]["ground_truth"]
        response = gt if str(gt).startswith("\\boxed{") else f"\\boxed{{{gt}}}"
        if row["data_source"] == "gsm8k":
            result = numeric.verify(response=response, ground_truth=gt)
        else:
            result = latex.verify(response=response, ground_truth=gt)
        if not result.extraction.extraction_ok:
            raise AssertionError(f"Gold is not parseable for {row['extra_info']['prompt_id']}: {gt!r}")


def _check_prompt_text(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        content = row["prompt"][-1]["content"]
        if not content.endswith(BOXED_INSTRUCTION):
            raise AssertionError(f"Prompt missing boxed instruction: {row['extra_info']['prompt_id']}")


def _check_token_length(rows: list[dict[str, Any]], tokenizer_path: str | None, max_prompt_length: int) -> None:
    if not tokenizer_path:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    for row in rows:
        token_ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
        if len(token_ids) > max_prompt_length:
            raise AssertionError(
                f"Prompt too long for {row['extra_info']['prompt_id']}: {len(token_ids)} > {max_prompt_length}"
            )


def _check_parquet_roundtrip(path: Path, expected_rows: int) -> None:
    loaded = pd.read_parquet(path)
    if len(loaded) != expected_rows:
        raise AssertionError(f"Parquet roundtrip row mismatch for {path}: {len(loaded)} != {expected_rows}")


def _check_verl_loader(path: Path, tokenizer_path: str, max_prompt_length: int) -> None:
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": max_prompt_length,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "return_raw_chat": False,
        }
    )
    dataset = RLHFDataset(str(path), tokenizer=tokenizer, config=config)
    if len(dataset) == 0:
        raise AssertionError(f"veRL RLHFDataset loaded zero rows from {path}")
    sample = dataset[0]
    assert sample["reward_model"]["ground_truth"]
    assert sample["extra_info"]["prompt_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "signal_forge_a0")
    parser.add_argument("--gsm8k-train-inputs", type=Path, default=_existing(DEFAULT_GSM8K_TRAIN_INPUTS, LOCAL_GSM8K_TRAIN_INPUTS))
    parser.add_argument("--math-train-inputs", type=Path, default=_existing(DEFAULT_MATH_TRAIN_INPUTS, LOCAL_MATH_TRAIN_INPUTS))
    parser.add_argument("--gsm8k-val-inputs", type=Path, default=_existing(DEFAULT_GSM8K_VAL_INPUTS, LOCAL_GSM8K_VAL_INPUTS))
    parser.add_argument("--math-val-inputs", type=Path, default=None)
    parser.add_argument("--math-test-dir", type=Path, default=_existing(DEFAULT_MATH_TEST_DIR, LOCAL_MATH_TEST_DIR))
    parser.add_argument("--train-gsm8k", type=int, default=3)
    parser.add_argument("--train-math", type=int, default=2)
    parser.add_argument("--val-gsm8k", type=int, default=4)
    parser.add_argument("--val-math", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--check-verl-loader", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "train.parquet"
    val_file = args.output_dir / "val.parquet"

    train_gsm = _sample_inputs(args.gsm8k_train_inputs, args.train_gsm8k, used_prompt_ids=set(), seed=args.seed + 1)
    used_train_ids = {record["prompt_id"] for record in train_gsm}
    train_math = _sample_inputs(args.math_train_inputs, args.train_math, used_prompt_ids=used_train_ids, seed=args.seed + 2)
    used_train_ids.update(record["prompt_id"] for record in train_math)

    val_gsm = _sample_inputs(args.gsm8k_val_inputs, args.val_gsm8k, used_prompt_ids=used_train_ids, seed=args.seed + 3)
    used_val_ids = used_train_ids | {record["prompt_id"] for record in val_gsm}
    if args.math_val_inputs:
        val_math = _sample_inputs(args.math_val_inputs, args.val_math, used_prompt_ids=used_val_ids, seed=args.seed + 4)
    else:
        val_math = _math_test_records(
            args.val_math,
            used_prompt_ids=used_val_ids,
            seed=args.seed + 4,
            math_test_dir=args.math_test_dir,
        )

    train_rows = [
        *[_verl_row(record, source="gsm8k", split="train", math_level=None) for record in train_gsm],
        *[_verl_row(record, source="math_level_3", split="train", math_level=3) for record in train_math],
    ]
    val_rows = [
        *[_verl_row(record, source="gsm8k", split="validation", math_level=None) for record in val_gsm],
        *[_verl_row(record, source="math_level_3", split="validation", math_level=3) for record in val_math],
    ]

    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 99).shuffle(val_rows)

    _check_prompt_ids(train_rows, "train")
    _check_prompt_ids(val_rows, "validation")
    _check_no_overlap(train_rows, val_rows)
    _check_train_ratio(train_rows, args.train_gsm8k, args.train_math)
    _check_gold_parseable(train_rows + val_rows)
    _check_prompt_text(train_rows + val_rows)
    _check_token_length(train_rows + val_rows, args.tokenizer_path, args.max_prompt_length)

    pd.DataFrame(train_rows).to_parquet(train_file, index=False)
    pd.DataFrame(val_rows).to_parquet(val_file, index=False)
    _check_parquet_roundtrip(train_file, len(train_rows))
    _check_parquet_roundtrip(val_file, len(val_rows))

    if args.check_verl_loader:
        if not args.tokenizer_path:
            raise ValueError("--check-verl-loader requires --tokenizer-path")
        _check_verl_loader(train_file, args.tokenizer_path, args.max_prompt_length)
        _check_verl_loader(val_file, args.tokenizer_path, args.max_prompt_length)

    manifest = SelectionManifest(
        seed=args.seed,
        train_gsm8k=args.train_gsm8k,
        train_math_level_3=args.train_math,
        val_gsm8k=args.val_gsm8k,
        val_math_level_3=args.val_math,
        train_file=str(train_file),
        val_file=str(val_file),
        train_prompt_ids=[row["extra_info"]["prompt_id"] for row in train_rows],
        val_prompt_ids=[row["extra_info"]["prompt_id"] for row in val_rows],
    )
    manifest_file = args.output_dir / "selection_manifest.json"
    manifest_file.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
