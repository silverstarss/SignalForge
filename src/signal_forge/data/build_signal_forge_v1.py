"""Build the frozen Signal Forge v1 GSM8K/MATH Level 3 dataset.

The builder is intentionally offline-first. It consumes local GSM8K cache via
``datasets`` and local MATH JSON files, then exports the exact veRL parquet row
schema used by the A0 data path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BOXED_INSTRUCTION = "Please reason step by step and put your final answer within \\boxed{}."
DATASET_VERSION = "signal_forge_v1"
DEFAULT_SEED = 20260728


@dataclass(frozen=True)
class CanonicalRecord:
    data_source: str
    source_dataset: str
    source_config: str
    source_split: str
    source_index: int
    source_key: str
    question: str
    ground_truth: str
    reference_solution: str
    math_level: int | None
    math_type: str | None
    source_file: str | None
    source_revision: str
    question_sha256: str

    @property
    def prompt_id(self) -> str:
        if self.data_source == "gsm8k":
            return f"gsm8k:{self.source_config}:{self.source_split}:{self.source_index:06d}"
        if self.data_source == "math_level_3":
            clean_key = self.source_key.replace("/", ":").removesuffix(".json")
            return f"math:competition_math:{self.source_split}:{clean_key}"
        raise ValueError(f"Unsupported data_source: {self.data_source}")


@dataclass(frozen=True)
class Rejection:
    source_dataset: str
    source_split: str
    source_key: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class BuildSelection:
    train: list[CanonicalRecord]
    validation: list[CanonicalRecord]
    test: list[CanonicalRecord]
    rejections: list[Rejection]
    duplicate_report: list[dict[str, Any]]


def stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def prompt_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question.strip()}\n\n{BOXED_INSTRUCTION}"}]


def verl_row(record: CanonicalRecord, project_split: str, selection_seed: int) -> dict[str, Any]:
    return {
        "data_source": record.data_source,
        "ability": "math",
        "prompt": prompt_messages(record.question),
        "reward_model": {"style": "rule", "ground_truth": record.ground_truth},
        "extra_info": {
            "prompt_id": record.prompt_id,
            "source_index": record.source_index,
            "source_key": record.source_key,
            "source_dataset": record.source_dataset,
            "source_config": record.source_config,
            "source_split": record.source_split,
            "split": project_split,
            "selection_seed": selection_seed,
            "dataset_version": DATASET_VERSION,
            "revision": record.source_revision,
            "math_level": record.math_level,
            "math_type": record.math_type,
            "source_file": record.source_file,
            "question": record.question,
            "question_sha256": record.question_sha256,
        },
    }


def _rewardscope_src_path(project_root: Path) -> Path:
    return project_root / "src" / "RewardScope" / "src"


def _ensure_rewardscope_importable(project_root: Path) -> None:
    for path in (
        str(project_root / "src" / "vendor_python"),
        str(_rewardscope_src_path(project_root)),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


def extract_gsm8k_gold(reference_solution: str, *, project_root: Path | None = None) -> str | None:
    if project_root is not None:
        _ensure_rewardscope_importable(project_root)
    from rewardscope.extraction import extract_numeric_answer

    extraction = extract_numeric_answer(reference_solution)
    if extraction.extraction_ok and extraction.normalized_answer is not None:
        return str(extraction.normalized_answer)
    return None


def extract_math_gold(reference_solution: str, *, project_root: Path | None = None) -> str | None:
    if project_root is not None:
        _ensure_rewardscope_importable(project_root)
    from rewardscope.verification.math_verify import extract_final_boxed_latex_gold

    return extract_final_boxed_latex_gold(reference_solution)


def load_gsm8k_split(
    split: str,
    *,
    project_root: Path,
    config_name: str = "main",
    revision: str | None = None,
    offline: bool = True,
) -> tuple[list[CanonicalRecord], list[Rejection], dict[str, Any]]:
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", config_name, split=split, revision=revision)
    records: list[CanonicalRecord] = []
    rejections: list[Rejection] = []
    for index, row in enumerate(dataset):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        source_key = f"{index:06d}"
        if not question:
            rejections.append(Rejection("gsm8k", split, source_key, "empty_prompt"))
            continue
        if not answer:
            rejections.append(Rejection("gsm8k", split, source_key, "empty_ground_truth"))
            continue
        gold = extract_gsm8k_gold(answer, project_root=project_root)
        if gold is None:
            rejections.append(Rejection("gsm8k", split, source_key, "gold_unparseable"))
            continue
        records.append(
            CanonicalRecord(
                data_source="gsm8k",
                source_dataset="openai/gsm8k",
                source_config=config_name,
                source_split=split,
                source_index=index,
                source_key=source_key,
                question=question,
                ground_truth=gold,
                reference_solution=answer,
                math_level=None,
                math_type=None,
                source_file=None,
                source_revision=revision or "local_hf_cache",
                question_sha256=stable_sha256(canonical_question(question)),
            )
        )
    meta = {
        "source_count": len(dataset),
        "eligible_count": len(records),
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }
    return records, rejections, meta


def load_math_raw_split(
    split_dir: Path,
    *,
    split: str,
    project_root: Path,
    required_level: str = "Level 3",
    source_revision: str = "local_MATH_tar",
) -> tuple[list[CanonicalRecord], list[Rejection], dict[str, Any]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"MATH split directory not found: {split_dir}")

    records: list[CanonicalRecord] = []
    rejections: list[Rejection] = []
    level_counts: Counter[str] = Counter()
    json_files = sorted(split_dir.rglob("*.json"), key=lambda p: str(p.relative_to(split_dir)))
    for index, path in enumerate(json_files):
        source_file = str(path.relative_to(split_dir))
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            rejections.append(Rejection("competition_math", split, source_file, "json_parse_error", str(error)))
            continue

        level = str(item.get("level") or "")
        level_counts[level] += 1
        if level != required_level:
            continue
        question = str(item.get("problem") or "").strip()
        solution = str(item.get("solution") or "").strip()
        if not question:
            rejections.append(Rejection("competition_math", split, source_file, "empty_prompt"))
            continue
        if not solution:
            rejections.append(Rejection("competition_math", split, source_file, "empty_ground_truth"))
            continue
        gold = extract_math_gold(solution, project_root=project_root)
        if gold is None:
            rejections.append(Rejection("competition_math", split, source_file, "gold_unparseable"))
            continue
        records.append(
            CanonicalRecord(
                data_source="math_level_3",
                source_dataset="competition_math",
                source_config="all",
                source_split=split,
                source_index=index,
                source_key=source_file,
                question=question,
                ground_truth=gold,
                reference_solution=solution,
                math_level=3,
                math_type=item.get("type") or path.parent.name,
                source_file=source_file,
                source_revision=source_revision,
                question_sha256=stable_sha256(canonical_question(question)),
            )
        )
    meta = {
        "source_count": len(json_files),
        "eligible_count": len(records),
        "level_counts": dict(level_counts),
    }
    return records, rejections, meta


def duplicate_report(records: Iterable[CanonicalRecord]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.question_sha256].append(record)
    report = []
    for question_hash, group in sorted(by_hash.items()):
        if len(group) <= 1:
            continue
        report.append(
            {
                "question_sha256": question_hash,
                "count": len(group),
                "members": [
                    {
                        "prompt_id": record.prompt_id,
                        "data_source": record.data_source,
                        "source_split": record.source_split,
                        "source_key": record.source_key,
                    }
                    for record in group
                ],
            }
        )
    return report


def reject_training_duplicates(
    train_candidates: list[CanonicalRecord],
    test_records: list[CanonicalRecord],
) -> tuple[list[CanonicalRecord], list[Rejection]]:
    test_hashes = {record.question_sha256 for record in test_records}
    seen_train_hashes: set[str] = set()
    kept: list[CanonicalRecord] = []
    rejections: list[Rejection] = []
    for record in sorted(train_candidates, key=lambda r: (r.data_source, r.source_split, r.source_key)):
        if record.question_sha256 in test_hashes:
            rejections.append(
                Rejection(
                    record.source_dataset,
                    record.source_split,
                    record.source_key,
                    "duplicate_with_test",
                    record.question_sha256,
                )
            )
            continue
        if record.question_sha256 in seen_train_hashes:
            rejections.append(
                Rejection(
                    record.source_dataset,
                    record.source_split,
                    record.source_key,
                    "duplicate_with_train",
                    record.question_sha256,
                )
            )
            continue
        seen_train_hashes.add(record.question_sha256)
        kept.append(record)
    return kept, rejections


def select_splits(
    *,
    gsm_train: list[CanonicalRecord],
    gsm_test: list[CanonicalRecord],
    math_train: list[CanonicalRecord],
    math_test: list[CanonicalRecord],
    seed: int = DEFAULT_SEED,
    validation_gsm: int = 300,
    validation_math: int = 200,
    train_math_count: int | None = None,
) -> BuildSelection:
    all_records = [*gsm_train, *gsm_test, *math_train, *math_test]
    dup_report = duplicate_report(all_records)
    train_candidates, dup_rejections = reject_training_duplicates([*gsm_train, *math_train], [*gsm_test, *math_test])
    gsm_pool = [record for record in train_candidates if record.data_source == "gsm8k"]
    math_pool = [record for record in train_candidates if record.data_source == "math_level_3"]

    if len(gsm_pool) < validation_gsm:
        raise ValueError(f"Need {validation_gsm} GSM8K validation records, found {len(gsm_pool)}")
    if len(math_pool) < validation_math:
        raise ValueError(f"Need {validation_math} MATH L3 validation records, found {len(math_pool)}")

    rng_gsm = random.Random(seed + 101)
    rng_math = random.Random(seed + 202)
    validation_gsm_ids = set(rng_gsm.sample(range(len(gsm_pool)), validation_gsm))
    validation_math_ids = set(rng_math.sample(range(len(math_pool)), validation_math))
    val_gsm_records = [record for i, record in enumerate(gsm_pool) if i in validation_gsm_ids]
    val_math_records = [record for i, record in enumerate(math_pool) if i in validation_math_ids]
    remaining_gsm = [record for i, record in enumerate(gsm_pool) if i not in validation_gsm_ids]
    remaining_math = [record for i, record in enumerate(math_pool) if i not in validation_math_ids]

    desired_math_train = len(remaining_math) if train_math_count is None else min(train_math_count, len(remaining_math))
    if desired_math_train % 2:
        desired_math_train -= 1
    if desired_math_train <= 0:
        raise ValueError("No MATH L3 training records remain after validation and 60/40 rounding")
    desired_gsm_train = desired_math_train * 3 // 2
    if len(remaining_gsm) < desired_gsm_train:
        max_even_math = (len(remaining_gsm) * 2 // 3)
        if max_even_math % 2:
            max_even_math -= 1
        desired_math_train = min(desired_math_train, max_even_math)
        desired_gsm_train = desired_math_train * 3 // 2
    if desired_math_train <= 0 or len(remaining_gsm) < desired_gsm_train:
        raise ValueError(
            "Insufficient GSM8K records to satisfy exact 60/40 train mixture "
            f"after validation: gsm={len(remaining_gsm)}, math={len(remaining_math)}"
        )

    train_math_records = random.Random(seed + 303).sample(remaining_math, desired_math_train)
    train_gsm_records = random.Random(seed + 404).sample(remaining_gsm, desired_gsm_train)
    train = [*train_gsm_records, *train_math_records]
    validation = [*val_gsm_records, *val_math_records]
    test = [*gsm_test, *math_test]
    random.Random(seed + 505).shuffle(train)
    random.Random(seed + 606).shuffle(validation)

    if len({record.prompt_id for record in [*train, *validation, *test]}) != len([*train, *validation, *test]):
        raise AssertionError("prompt_id values are not globally unique")
    if len({record.question_sha256 for record in train} & {record.question_sha256 for record in validation}):
        raise AssertionError("train/validation duplicate question leakage detected")
    if len({record.question_sha256 for record in train} & {record.question_sha256 for record in test}):
        raise AssertionError("train/test duplicate question leakage detected")
    if sum(r.data_source == "gsm8k" for r in train) * 2 != sum(r.data_source == "math_level_3" for r in train) * 3:
        raise AssertionError("train mixture is not exactly GSM8K 60% / MATH Level 3 40%")

    rejections = list(dup_rejections)
    unused_math = [record for record in remaining_math if record not in train_math_records]
    if unused_math:
        rejections.extend(
            Rejection(record.source_dataset, record.source_split, record.source_key, "unused_after_mixture_selection")
            for record in unused_math
        )
    unused_gsm = [record for record in remaining_gsm if record not in train_gsm_records]
    if unused_gsm:
        rejections.extend(
            Rejection(record.source_dataset, record.source_split, record.source_key, "unused_after_mixture_selection")
            for record in unused_gsm
        )
    return BuildSelection(train=train, validation=validation, test=test, rejections=rejections, duplicate_report=dup_report)


def _count_by_source(records: list[CanonicalRecord]) -> dict[str, int]:
    return dict(Counter(record.data_source for record in records))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def write_outputs(
    selection: BuildSelection,
    *,
    output_dir: Path,
    seed: int,
    source_meta: dict[str, Any],
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing_output:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation_id.parquet"
    test_path = output_dir / "test_id.parquet"
    records_path = output_dir / "records_manifest.parquet"
    rejections_path = output_dir / "rejections.jsonl"
    duplicate_path = output_dir / "duplicate_report.jsonl"
    stats_path = output_dir / "statistics.json"
    manifest_path = output_dir / "manifest.json"

    rows = {
        "train": [verl_row(record, "train", seed) for record in selection.train],
        "validation_id": [verl_row(record, "validation_id", seed) for record in selection.validation],
        "test_id": [verl_row(record, "test_id", seed) for record in selection.test],
    }
    pd.DataFrame(rows["train"]).to_parquet(train_path, index=False)
    pd.DataFrame(rows["validation_id"]).to_parquet(validation_path, index=False)
    pd.DataFrame(rows["test_id"]).to_parquet(test_path, index=False)

    record_rows = []
    for split, records in (("train", selection.train), ("validation_id", selection.validation), ("test_id", selection.test)):
        for record in records:
            item = asdict(record)
            item["prompt_id"] = record.prompt_id
            item["project_split"] = split
            item["selection_seed"] = seed
            record_rows.append(item)
    pd.DataFrame(record_rows).to_parquet(records_path, index=False)

    with rejections_path.open("w", encoding="utf-8") as handle:
        for rejection in selection.rejections:
            handle.write(json.dumps(asdict(rejection), ensure_ascii=False, sort_keys=True) + "\n")
    with duplicate_path.open("w", encoding="utf-8") as handle:
        for item in selection.duplicate_report:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    statistics = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "splits": {
            "train": {"rows": len(selection.train), "by_source": _count_by_source(selection.train)},
            "validation_id": {"rows": len(selection.validation), "by_source": _count_by_source(selection.validation)},
            "test_id": {"rows": len(selection.test), "by_source": _count_by_source(selection.test)},
        },
        "train_mixture_check": {
            "gsm8k": sum(r.data_source == "gsm8k" for r in selection.train),
            "math_level_3": sum(r.data_source == "math_level_3" for r in selection.train),
            "ratio": "60/40",
        },
        "rejections_by_reason": dict(Counter(rejection.reason for rejection in selection.rejections)),
        "duplicate_cluster_count": len(selection.duplicate_report),
        "source_meta": source_meta,
    }
    stats_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_files = {
        "train": str(train_path),
        "validation_id": str(validation_path),
        "test_id": str(test_path),
        "records_manifest": str(records_path),
        "rejections": str(rejections_path),
        "duplicate_report": str(duplicate_path),
        "statistics": str(stats_path),
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "boxed_instruction": BOXED_INSTRUCTION,
        "train_mixture": {"gsm8k": 0.60, "math_level_3": 0.40},
        "splits": statistics["splits"],
        "output_files": output_files,
        "output_sha256": {name: _sha256_file(Path(path)) for name, path in output_files.items()},
        "source_meta": source_meta,
        "notes": [
            "validation_id is sampled from upstream train splits and is used for checkpoint selection.",
            "test_id preserves upstream official test splits and must not be used for checkpoint selection.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_from_local_sources(args: argparse.Namespace) -> tuple[BuildSelection, dict[str, Any]]:
    project_root = args.project_root.resolve()
    gsm_train, gsm_train_rej, gsm_train_meta = load_gsm8k_split(
        "train", project_root=project_root, revision=args.gsm8k_revision, offline=args.offline
    )
    gsm_test, gsm_test_rej, gsm_test_meta = load_gsm8k_split(
        "test", project_root=project_root, revision=args.gsm8k_revision, offline=args.offline
    )
    math_train, math_train_rej, math_train_meta = load_math_raw_split(
        args.math_root / "train",
        split="train",
        project_root=project_root,
        source_revision=args.math_revision,
    )
    math_test, math_test_rej, math_test_meta = load_math_raw_split(
        args.math_root / "test",
        split="test",
        project_root=project_root,
        source_revision=args.math_revision,
    )
    selection = select_splits(
        gsm_train=gsm_train,
        gsm_test=gsm_test,
        math_train=math_train,
        math_test=math_test,
        seed=args.seed,
        validation_gsm=args.validation_gsm,
        validation_math=args.validation_math,
        train_math_count=args.train_math_count,
    )
    selection.rejections.extend([*gsm_train_rej, *gsm_test_rej, *math_train_rej, *math_test_rej])  # type: ignore[attr-defined]
    math_archive = args.math_archive
    source_meta = {
        "gsm8k_train": gsm_train_meta,
        "gsm8k_test": gsm_test_meta,
        "math_train": math_train_meta,
        "math_test": math_test_meta,
        "math_root": str(args.math_root),
        "math_archive": str(math_archive) if math_archive else None,
        "math_archive_sha256": _sha256_file(math_archive) if math_archive and math_archive.exists() else None,
        "gsm8k_revision": args.gsm8k_revision or "local_hf_cache",
        "math_revision": args.math_revision,
        "signal_forge_git_head": _git_value(project_root, "rev-parse", "HEAD"),
        "signal_forge_git_dirty": _git_value(project_root, "status", "--short") is not None,
    }
    return selection, source_meta


def print_summary(selection: BuildSelection, source_meta: dict[str, Any]) -> None:
    summary = {
        "dataset_version": DATASET_VERSION,
        "train_rows": len(selection.train),
        "train_by_source": _count_by_source(selection.train),
        "validation_rows": len(selection.validation),
        "validation_by_source": _count_by_source(selection.validation),
        "test_rows": len(selection.test),
        "test_by_source": _count_by_source(selection.test),
        "rejections_by_reason": dict(Counter(rejection.reason for rejection in selection.rejections)),
        "duplicate_cluster_count": len(selection.duplicate_report),
        "source_meta": source_meta,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/signal_forge_v1"))
    parser.add_argument("--math-root", type=Path, default=Path("src/RewardScope/raw/competition_math"))
    parser.add_argument("--math-archive", type=Path, default=Path("_downloads/MATH.tar"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-gsm", type=int, default=300)
    parser.add_argument("--validation-math", type=int, default=200)
    parser.add_argument("--train-math-count", type=int, default=None)
    parser.add_argument("--gsm8k-revision", default=None)
    parser.add_argument("--math-revision", default="local_modelscope_MATH_tar")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selection, source_meta = build_from_local_sources(args)
    print_summary(selection, source_meta)
    if args.inventory_only:
        return 0
    manifest = write_outputs(
        selection,
        output_dir=args.output_dir,
        seed=args.seed,
        source_meta=source_meta,
        allow_existing_output=args.allow_existing_output,
    )
    print(f"Wrote {DATASET_VERSION} dataset to {args.output_dir}")
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
