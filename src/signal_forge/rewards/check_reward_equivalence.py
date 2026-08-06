"""Check RewardScope verifier and veRL reward adapter equivalence."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from signal_forge.rewards.math_verify_adapter import compute_score

_INFERRED_SIGNAL_FORGE_SRC = Path(__file__).resolve().parents[2]
_GSM8K_REWARDSCOPE_RUN = "gsm8k-qwen-grpo-train-zero-shot-boxed-128"
_MATH_REWARDSCOPE_RUN = "math-qwen-grpo-train-level-3-64-max768"


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            unique.append(path.expanduser())
            seen.add(key)
    return unique


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _source_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ["SIGNAL_FORGE_SRC", "SIGNAL_FORGE_ROOT", "ROOT_DIR", "WORKSPACE"]:
        path = _env_path(name)
        if path is None:
            continue
        roots.extend([path, path / "src"])
    roots.extend([_INFERRED_SIGNAL_FORGE_SRC, Path.cwd(), Path.cwd() / "src"])
    return _dedupe(roots)


def _artifact_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ["SIGNAL_FORGE_ARTIFACT_ROOT", "OUTPUT_ROOT"]:
        path = _env_path(name)
        if path is not None:
            roots.extend([path, path / "outputs"])
    for source_root in _source_roots():
        roots.extend(
            [
                source_root.parent / "signal_forge_artifacts",
                source_root.parent / "signal_forge_artifacts" / "outputs",
                source_root / "artifacts",
                source_root / "artifacts" / "outputs",
                source_root / "outputs",
            ]
        )
    return _dedupe(roots)


def _rewardscope_output_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ["REWARDSCOPE_OUTPUTS", "REWARDSCOPE_OUTPUT_DIR"]:
        path = _env_path(name)
        if path is not None:
            roots.append(path)
    for source_root in _source_roots():
        roots.extend(
            [
                source_root / "RewardScope" / "outputs",
                source_root.parent / "RewardScope" / "outputs",
            ]
        )
    for artifact_root in _artifact_roots():
        roots.extend([artifact_root / "RewardScope" / "outputs", artifact_root])
    return _dedupe(roots)


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _rollout_candidates(run_name: str) -> list[Path]:
    return [root / run_name / "rollouts.jsonl" for root in _rewardscope_output_roots()]


GSM8K_REWARDSCOPE_ROLLOUT_CANDIDATES = _rollout_candidates(_GSM8K_REWARDSCOPE_RUN)
MATH_REWARDSCOPE_ROLLOUT_CANDIDATES = _rollout_candidates(_MATH_REWARDSCOPE_RUN)
DEFAULT_GSM8K_ROLLOUTS = _first_existing(GSM8K_REWARDSCOPE_ROLLOUT_CANDIDATES)
DEFAULT_MATH_ROLLOUTS = _first_existing(MATH_REWARDSCOPE_ROLLOUT_CANDIDATES)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Rollout JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _jsonl_sort_key(path: Path) -> tuple[str, int, str]:
    try:
        stem_number = int(path.stem)
    except ValueError:
        stem_number = 10**9
    return (str(path.parent), stem_number, path.name)


def _expand_jsonl_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl"), key=_jsonl_sort_key))
        else:
            expanded.append(path)
    return _dedupe(expanded)


def _discover_verl_dump_files() -> list[Path]:
    explicit_dirs = [
        path
        for path in [_env_path("ROLLOUT_DIR"), _env_path("VAL_DIR"), _env_path("A0_ROLLOUT_DIR"), _env_path("A0_VAL_DIR")]
        if path
    ]
    explicit_files = _expand_jsonl_paths(explicit_dirs)
    if explicit_files:
        return explicit_files

    dump_dirs: list[Path] = []
    for root in _artifact_roots():
        if not root.exists() or not root.is_dir():
            continue
        for name in ["rollout_data", "validation_data"]:
            dump_dirs.extend(path for path in root.rglob(name) if path.is_dir())

    if not dump_dirs:
        return []

    newest_run = max(dump_dirs, key=lambda path: path.stat().st_mtime).parent
    return _expand_jsonl_paths([newest_run / "rollout_data", newest_run / "validation_data"])


def _default_rollout_files() -> list[Path]:
    rewardscope_files = [DEFAULT_GSM8K_ROLLOUTS, DEFAULT_MATH_ROLLOUTS]
    if all(path.exists() for path in rewardscope_files):
        return rewardscope_files
    return _discover_verl_dump_files()


def _missing_default_message() -> str:
    tried = [
        *GSM8K_REWARDSCOPE_ROLLOUT_CANDIDATES,
        *MATH_REWARDSCOPE_ROLLOUT_CANDIDATES,
        *_discover_verl_dump_files(),
    ]
    rendered = "\n".join(f"  - {path}" for path in _dedupe(tried)[:80])
    return (
        "No rollout JSONL files were found. Pass --rollouts/--gsm8k-rollouts/--math-rollouts, "
        "or set ROLLOUT_DIR/VAL_DIR/REWARDSCOPE_OUTPUTS. Tried:\n"
        f"{rendered}"
    )


def _record_response(record: dict[str, Any]) -> str:
    return str(record.get("response", record.get("output", "")))


def _record_ground_truth(record: dict[str, Any]) -> str:
    return str(record.get("ground_truth", record.get("gts", "")))


def _source(record: dict[str, Any]) -> str:
    dataset_name = record.get("dataset_name", record.get("reward_source", record.get("data_source")))
    if dataset_name == "gsm8k":
        return "gsm8k"
    if dataset_name in {"math", "math_level_3"}:
        return "math_level_3"
    raise ValueError(f"Unsupported rollout dataset_name: {dataset_name!r}")


def _direct_reward(record: dict[str, Any]) -> dict[str, Any]:
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier, MathVerifyNumericVerifier

    source = _source(record)
    if source == "gsm8k":
        result = MathVerifyNumericVerifier(mode="training").verify(
            response=_record_response(record), ground_truth=_record_ground_truth(record)
        )
    else:
        result = MathVerifyLatexVerifier(mode="training").verify(
            response=_record_response(record), ground_truth=_record_ground_truth(record)
        )
    extraction = result.extraction
    score = float(result.is_correct)
    return {
        "score": score,
        "raw_correctness": score,
        "extraction_ok": bool(extraction.extraction_ok),
        "format_ok": bool(extraction.format_ok),
        "verification_status": extraction.extraction_status.value,
        "verification_error_type": result.error_type or "",
    }


def _adapter_reward(record: dict[str, Any]) -> dict[str, Any]:
    return compute_score(
        data_source=_source(record),
        solution_str=_record_response(record),
        ground_truth=_record_ground_truth(record),
        extra_info={"prompt_id": record.get("prompt_id"), "source_dataset": record.get("dataset_name")},
    )


def _category(record: dict[str, Any]) -> set[str]:
    verification = record.get("verification") or {}
    extraction = verification.get("extraction") or {}
    status = str(extraction.get("extraction_status") or record.get("verification_status") or "")
    raw_correctness = verification.get("is_correct", record.get("raw_correctness", record.get("score", record.get("acc"))))
    categories = {_source(record)}
    categories.add("correct" if bool(raw_correctness) else "wrong")
    if status in {"missing", "parse_error", "ambiguous"}:
        categories.add("missing_or_parse")
    if bool(record.get("hit_max_length")) or record.get("finish_reason") == "length":
        categories.add("truncated")
    if len(extraction.get("valid_candidates") or []) > 1:
        categories.add("multi_value")
    return categories


def _has_truncation_signal(record: dict[str, Any]) -> bool:
    return "hit_max_length" in record or "finish_reason" in record


def _pick_diverse(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    wanted = ["gsm8k", "math_level_3", "correct", "wrong", "missing_or_parse", "truncated", "multi_value"]
    picked: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for category in _category(record):
            buckets[category].append(record)

    for category in wanted:
        for record in buckets.get(category, [])[:8]:
            key = _record_key(record)
            if key not in seen_keys:
                picked.append(record)
                seen_keys.add(key)

    for record in records:
        if len(picked) >= limit:
            break
        key = _record_key(record)
        if key not in seen_keys:
            picked.append(record)
            seen_keys.add(key)
    return picked[:limit]


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    prompt_id = str(record.get("prompt_id", record.get("input", "")))
    sample_id = record.get("sample_id", record.get("output", ""))
    return (prompt_id, hash(str(sample_id)))


def _compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in [
        "score",
        "raw_correctness",
        "extraction_ok",
        "format_ok",
        "verification_status",
        "verification_error_type",
    ]:
        if expected[key] != actual[key]:
            mismatches.append(f"{key}: expected={expected[key]!r} actual={actual[key]!r}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, action="append", default=None, help="Mixed rollout JSONL file or directory.")
    parser.add_argument("--gsm8k-rollouts", type=Path, default=None)
    parser.add_argument("--math-rollouts", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--require-categories", action="store_true")
    args = parser.parse_args()

    if args.rollouts:
        rollout_files = _expand_jsonl_paths(args.rollouts)
    elif args.gsm8k_rollouts or args.math_rollouts:
        rollout_files = [path for path in [args.gsm8k_rollouts, args.math_rollouts] if path is not None]
    else:
        rollout_files = _default_rollout_files()

    if not rollout_files:
        raise FileNotFoundError(_missing_default_message())

    records = [record for path in rollout_files for record in _iter_jsonl(path)]
    if len(records) < 100:
        raise AssertionError(f"Need at least 100 saved rollouts, found {len(records)}")

    picked = _pick_diverse(records, max(args.limit, 100))
    category_counts = Counter(category for record in picked for category in _category(record))

    if args.require_categories:
        required = {"gsm8k", "math_level_3", "correct", "wrong", "missing_or_parse"}
        if any(_has_truncation_signal(record) for record in records):
            required.add("truncated")
        missing = sorted(category for category in required if category_counts[category] == 0)
        if missing:
            raise AssertionError(f"Selected rollouts miss required categories: {missing}; counts={dict(category_counts)}")

    mismatches: list[str] = []
    start = time.perf_counter()
    for index, record in enumerate(picked):
        expected = _direct_reward(record)
        actual = _adapter_reward(record)
        diff = _compare(expected, actual)
        if diff:
            mismatches.append(
                f"#{index} {_record_label(record)} source={_source(record)}: "
                + "; ".join(diff)
            )
    elapsed = time.perf_counter() - start

    if mismatches:
        preview = "\n".join(mismatches[:20])
        raise AssertionError(f"Reward equivalence failed for {len(mismatches)} rows:\n{preview}")

    print(
        json.dumps(
            {
                "checked": len(picked),
                "elapsed_sec": round(elapsed, 3),
                "rows_per_sec": round(len(picked) / elapsed, 3) if elapsed else None,
                "categories": dict(sorted(category_counts.items())),
                "rollout_files": [str(path) for path in rollout_files],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _record_label(record: dict[str, Any]) -> str:
    if "prompt_id" in record:
        return f"{record.get('prompt_id')} sample={record.get('sample_id')}"
    return f"step={record.get('step', '?')}"


if __name__ == "__main__":
    main()
