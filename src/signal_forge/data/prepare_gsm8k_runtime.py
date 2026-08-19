"""Prepare a boxed GSM8K veRL dataset for runtime migration smoke tests.

This script downloads GSM8K through Hugging Face Datasets when the dataset is not
already cached. It intentionally uses the SignalForge boxed prompt and veRL row
schema so the existing Math-Verify reward adapter can be exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

BOXED_INSTRUCTION = r"Please reason step by step and put your final answer within \boxed{}."


@dataclass(frozen=True)
class Gsm8kRuntimeManifest:
    dataset_name: str
    dataset_config: str
    output_dir: str
    train_file: str
    test_file: str
    train_rows: int
    test_rows: int
    train_max_samples: int
    test_max_samples: int
    boxed_instruction: str
    hf_home: str | None
    hf_datasets_cache: str | None
    train_prompt_ids: list[str]
    test_prompt_ids: list[str]


def _extract_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer, flags=re.DOTALL)
    if not match:
        raise ValueError(f"GSM8K answer is missing #### final answer marker: {answer[:120]!r}")
    return match.group(1).replace(",", "").strip()


def _prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question.strip()}\n\n{BOXED_INSTRUCTION}"}]


def _row(item: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    question = str(item["question"]).strip()
    ground_truth = _extract_gsm8k_answer(str(item["answer"]))
    prompt_id = f"gsm8k:openai-main:{split}:{index:06d}"
    return {
        "data_source": "gsm8k",
        "ability": "math",
        "prompt": _prompt(question),
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "prompt_id": prompt_id,
            "source_index": index,
            "split": split,
            "math_level": None,
            "source_dataset": "openai/gsm8k",
            "dataset_config": "main",
            "dataset_version": "hf-openai-gsm8k-main-runtime",
            "question": question,
            "boxed_instruction": BOXED_INSTRUCTION,
        },
    }


def _rows(dataset: Any, *, split: str, max_samples: int) -> list[dict[str, Any]]:
    limit = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
    return [_row(dataset[i], split=split, index=i) for i in range(limit)]


def _check_rows(rows: list[dict[str, Any]], label: str) -> None:
    ids = [row["extra_info"]["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate prompt_id in {label}")
    for row in rows:
        prompt_text = row["prompt"][-1]["content"]
        if not prompt_text.endswith(BOXED_INSTRUCTION):
            raise AssertionError(f"prompt missing boxed instruction: {row['extra_info']['prompt_id']}")
        if not str(row["reward_model"].get("ground_truth", "")).strip():
            raise AssertionError(f"empty ground truth: {row['extra_info']['prompt_id']}")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    if len(loaded) != len(rows):
        raise AssertionError(f"parquet roundtrip mismatch for {path}: {len(loaded)} != {len(rows)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DATA_ROOT", "data")) / "gsm8k_boxed")
    parser.add_argument("--train-max-samples", type=int, default=-1)
    parser.add_argument("--test-max-samples", type=int, default=-1)
    args = parser.parse_args(argv)

    dataset = load_dataset("openai/gsm8k", "main")
    train_rows = _rows(dataset["train"], split="train", max_samples=args.train_max_samples)
    test_rows = _rows(dataset["test"], split="test", max_samples=args.test_max_samples)
    _check_rows(train_rows, "train")
    _check_rows(test_rows, "test")
    overlap = {row["extra_info"]["prompt_id"] for row in train_rows} & {row["extra_info"]["prompt_id"] for row in test_rows}
    if overlap:
        raise AssertionError(f"train/test prompt_id overlap: {sorted(overlap)[:10]}")

    output_dir = args.output_dir.resolve()
    train_file = output_dir / "train.parquet"
    test_file = output_dir / "test.parquet"
    _write_parquet(train_file, train_rows)
    _write_parquet(test_file, test_rows)

    manifest = Gsm8kRuntimeManifest(
        dataset_name="openai/gsm8k",
        dataset_config="main",
        output_dir=str(output_dir),
        train_file=str(train_file),
        test_file=str(test_file),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        train_max_samples=args.train_max_samples,
        test_max_samples=args.test_max_samples,
        boxed_instruction=BOXED_INSTRUCTION,
        hf_home=os.environ.get("HF_HOME"),
        hf_datasets_cache=os.environ.get("HF_DATASETS_CACHE"),
        train_prompt_ids=[row["extra_info"]["prompt_id"] for row in train_rows],
        test_prompt_ids=[row["extra_info"]["prompt_id"] for row in test_rows],
    )
    manifest_path = output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"train_file": str(train_file), "test_file": str(test_file), "manifest": str(manifest_path), "train_rows": len(train_rows), "test_rows": len(test_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
