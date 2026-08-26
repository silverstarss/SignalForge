"""Validation normalization and deterministic training-pool decontamination."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from rewardscope.verification import extract_final_boxed_latex_gold

from signal_forge.calibration.hive_dataset import CalibrationPrompt


QWEN_EVAL_REVISION = "a45202bd16f1ec06f433442dc1152d0074773465"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
EXPECTED_BENCHMARK_COUNTS = {
    "math500": 500,
    "aime24": 30,
    "amc23": 40,
    "minerva_math": 272,
    "gaokao2023en": 385,
    "olympiadbench": 675,
}
BENCHMARK_FILENAMES = {name: f"{name}.jsonl" for name in EXPECTED_BENCHMARK_COUNTS}
MatchType = Literal["A", "B", "C"]
Decision = Literal["remove", "keep"]


@dataclass(frozen=True)
class ValidationProblem:
    benchmark: str
    benchmark_id: str
    row_index: int
    raw_question: str
    question: str
    source_answer: Any
    ground_truth: str
    is_multiple_answer: bool = False
    multiple_answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecontaminationRecord:
    train_prompt_id: str
    benchmark: str
    benchmark_id: str
    benchmark_row_index: int
    match_type: MatchType
    similarity: float
    normalized_char_similarity: float
    decision: Decision
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_validation_suite(
    data_dir: str | Path, *, validate_counts: bool = True
) -> tuple[ValidationProblem, ...]:
    root = Path(data_dir)
    problems: list[ValidationProblem] = []
    for benchmark, filename in BENCHMARK_FILENAMES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"validation benchmark snapshot not found: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if validate_counts and len(rows) != EXPECTED_BENCHMARK_COUNTS[benchmark]:
            raise ValueError(
                f"unexpected {benchmark} row count: {len(rows)} != "
                f"{EXPECTED_BENCHMARK_COUNTS[benchmark]}"
            )
        problems.extend(
            normalize_validation_row(benchmark, row, row_index=index)
            for index, row in enumerate(rows)
        )
    identities = [(item.benchmark, item.benchmark_id) for item in problems]
    if len(identities) != len(set(identities)):
        raise ValueError("validation suite contains duplicate benchmark-qualified IDs")
    return tuple(problems)


def normalize_validation_row(
    benchmark: str, row: Mapping[str, Any], *, row_index: int
) -> ValidationProblem:
    if benchmark == "math500":
        return _single_answer_problem(
            benchmark,
            str(row.get("unique_id", row_index)),
            row_index,
            row["problem"],
            row["answer"],
        )
    if benchmark in {"aime24", "amc23"}:
        answer = row["answer"]
        if isinstance(answer, float) and answer.is_integer():
            answer = str(int(answer))
        return _single_answer_problem(
            benchmark,
            str(row.get("id", row_index)),
            row_index,
            row.get("problem") or row["question"],
            answer,
        )
    if benchmark == "minerva_math":
        gold = extract_final_boxed_latex_gold(str(row["solution"]))
        if gold is None:
            raise ValueError(f"Minerva row {row_index} has no parseable final boxed answer")
        return ValidationProblem(
            benchmark=benchmark,
            benchmark_id=str(row.get("idx", row_index)),
            row_index=row_index,
            raw_question=str(row["problem"]),
            question=str(row["problem"]).strip(),
            source_answer=row["solution"],
            ground_truth=gold,
        )
    if benchmark == "gaokao2023en":
        return normalize_gaokao_row(row, row_index=row_index)
    if benchmark == "olympiadbench":
        return normalize_olympiad_row(row, row_index=row_index)
    raise ValueError(f"unsupported validation benchmark: {benchmark!r}")


def normalize_gaokao_row(row: Mapping[str, Any], *, row_index: int) -> ValidationProblem:
    raw_question = str(row["question"])
    question = raw_question.strip()
    answer = str(row.get("answer", "")).strip()
    if not answer:
        match = re.search(r"(?i)(?:answers?|answer):\s*(.+?)\s*$", raw_question, re.DOTALL)
        if match is None:
            raise ValueError(f"Gaokao row {row_index} has neither gold nor audited answer suffix")
        answer = match.group(1).strip()
        question = raw_question[: match.start()].rstrip()
    return _single_answer_problem(
        "gaokao2023en",
        str(row_index),
        row_index,
        question,
        answer,
        raw_question=raw_question,
    )


def normalize_olympiad_row(
    row: Mapping[str, Any], *, row_index: int
) -> ValidationProblem:
    raw_answers = row["final_answer"]
    if not isinstance(raw_answers, list) or not raw_answers:
        raise ValueError(f"OlympiadBench row {row_index} has invalid final_answer")
    multiple = bool(row.get("is_multiple_answer", False))
    if multiple:
        answers: list[str] = []
        for value in raw_answers:
            answers.extend(split_top_level_answers(_strip_math_delimiters(str(value))))
        normalized = tuple(_strip_math_delimiters(value) for value in answers)
        if not normalized or any(not value for value in normalized):
            raise ValueError(f"OlympiadBench row {row_index} has empty multiple answer")
        gold = _boxed(", ".join(normalized))
    else:
        if len(raw_answers) != 1:
            raise ValueError(
                f"OlympiadBench row {row_index} is single-answer but has {len(raw_answers)} golds"
            )
        normalized = ()
        gold = _boxed(_strip_math_delimiters(str(raw_answers[0])))
    question = str(row["question"]).strip()
    return ValidationProblem(
        benchmark="olympiadbench",
        benchmark_id=str(row.get("id", row_index)),
        row_index=row_index,
        raw_question=str(row["question"]),
        question=question,
        source_answer=raw_answers,
        ground_truth=gold,
        is_multiple_answer=multiple,
        multiple_answers=normalized,
    )


def split_top_level_answers(value: str) -> tuple[str, ...]:
    text = value.strip()
    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if char in "([{":
            stack.append(char)
        elif char in ")]}" and stack and stack[-1] == pairs[char]:
            stack.pop()
        elif char == "," and not stack:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return tuple(piece for piece in pieces if piece)


def olympiad_multiple_answers_equal(
    response: str,
    expected_answers: Sequence[str],
    *,
    verifier: Callable[[str, str], bool] | None = None,
) -> bool:
    """Apply OlympiadBench's unordered top-level multiple-answer semantics.

    Ordered structures inside one answer, such as a coordinate tuple, remain a
    single expression. Only the source-declared top-level answers are unordered.
    """
    boxed = _last_boxed_content(response)
    if boxed is None:
        return False
    predictions = tuple(_strip_math_delimiters(x) for x in split_top_level_answers(boxed))
    expected = tuple(_strip_math_delimiters(x) for x in expected_answers)
    if len(predictions) != len(expected):
        return False
    compare = verifier or _latex_equivalent

    def match(remaining_predictions: tuple[str, ...], remaining_gold: tuple[str, ...]) -> bool:
        if not remaining_predictions:
            return True
        prediction = remaining_predictions[0]
        for index, gold in enumerate(remaining_gold):
            if compare(prediction, gold) and match(
                remaining_predictions[1:], remaining_gold[:index] + remaining_gold[index + 1 :]
            ):
                return True
        return False

    return match(predictions, expected)


def load_review_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported decontamination review schema")
    return payload


def audit_training_pool(
    train_prompts: Sequence[CalibrationPrompt],
    validation_problems: Sequence[ValidationProblem],
    review_config: Mapping[str, Any],
) -> tuple[DecontaminationRecord, ...]:
    train_by_id = {prompt.prompt_id: _training_problem(prompt) for prompt in train_prompts}
    if len(train_by_id) != len(train_prompts):
        raise ValueError("decontamination input contains duplicate stable prompt IDs")
    ngram_size = int(review_config.get("ngram_size", 13))
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    manual_b = {
        (str(item["benchmark"]), str(item["benchmark_id"]), str(item["train_prompt_id"]))
        for item in review_config.get("manual_same_problem_pairs", [])
    }
    word_index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    exact_index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for prompt_id, problem in train_by_id.items():
        words = lexical_tokens(problem)
        exact_index[semantic_tokens(problem)].add(prompt_id)
        for index in range(len(words) - ngram_size + 1):
            word_index[words[index : index + ngram_size]].add(prompt_id)

    records: list[DecontaminationRecord] = []
    seen_manual_b: set[tuple[str, str, str]] = set()
    candidate_rows: set[tuple[str, str]] = set()
    high_review_count = 0
    manual_threshold = float(review_config.get("manual_review_lcs_threshold", 0.8))
    char_threshold = float(review_config.get("manual_review_char_threshold", 0.9))
    for validation in validation_problems:
        words = lexical_tokens(validation.question)
        candidate_ids = set(exact_index.get(semantic_tokens(validation.question), ()))
        for index in range(len(words) - ngram_size + 1):
            candidate_ids.update(word_index.get(words[index : index + ngram_size], ()))
        if candidate_ids:
            candidate_rows.add((validation.benchmark, validation.benchmark_id))
        for prompt_id in sorted(candidate_ids):
            train_problem = train_by_id[prompt_id]
            train_words = lexical_tokens(train_problem)
            similarity = _lcs_similarity(words, train_words)
            char_similarity = difflib.SequenceMatcher(
                None,
                "".join(words),
                "".join(train_words),
                autojunk=False,
            ).ratio()
            key = (validation.benchmark, validation.benchmark_id, prompt_id)
            if semantic_tokens(validation.question) == semantic_tokens(train_problem):
                match_type: MatchType = "A"
                decision: Decision = "remove"
                reason = "operator-preserving normalized semantic tokens are identical"
            elif key in manual_b:
                match_type = "B"
                decision = "remove"
                reason = (
                    "manual review confirmed the same mathematical task; differences are "
                    "presentation, trivial wording, or equivalent source-preserving notation"
                )
                seen_manual_b.add(key)
            else:
                match_type = "C"
                decision = "keep"
                if similarity >= manual_threshold or char_similarity >= char_threshold:
                    reason = (
                        "manual review found a changed target, constraint, numeric instance, "
                        "operator, or genuinely different problem"
                    )
                else:
                    reason = "shared 13-word span only; similarity is below manual-review scope"
            if similarity >= manual_threshold or char_similarity >= char_threshold:
                high_review_count += 1
            records.append(
                DecontaminationRecord(
                    train_prompt_id=prompt_id,
                    benchmark=validation.benchmark,
                    benchmark_id=validation.benchmark_id,
                    benchmark_row_index=validation.row_index,
                    match_type=match_type,
                    similarity=round(similarity, 6),
                    normalized_char_similarity=round(char_similarity, 6),
                    decision=decision,
                    reason=reason,
                )
            )
    missing_manual = manual_b - seen_manual_b
    if missing_manual:
        raise ValueError(f"manual B decisions are not present in candidate set: {sorted(missing_manual)!r}")
    expected = review_config.get("expected_complete_source_audit", {})
    actual = {
        "candidate_pairs": len(records),
        "candidate_benchmark_rows": len(candidate_rows),
        "manual_review_pairs": high_review_count,
    }
    for name, value in actual.items():
        if name in expected and int(expected[name]) != value:
            raise ValueError(f"decontamination audit drift for {name}: {value} != {expected[name]}")
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.benchmark,
                item.benchmark_row_index,
                item.train_prompt_id,
            ),
        )
    )


def remove_confirmed_overlaps(
    prompts: Sequence[CalibrationPrompt], records: Sequence[DecontaminationRecord]
) -> tuple[tuple[CalibrationPrompt, ...], tuple[str, ...]]:
    remove_ids = {record.train_prompt_id for record in records if record.decision == "remove"}
    prompt_ids = {prompt.prompt_id for prompt in prompts}
    unknown = remove_ids - prompt_ids
    if unknown:
        raise ValueError(f"decontamination manifest references unknown training IDs: {sorted(unknown)!r}")
    clean = tuple(prompt for prompt in prompts if prompt.prompt_id not in remove_ids)
    return clean, tuple(sorted(remove_ids))


def _single_answer_problem(
    benchmark: str,
    benchmark_id: str,
    row_index: int,
    question: Any,
    answer: Any,
    *,
    raw_question: str | None = None,
) -> ValidationProblem:
    visible = str(question).strip()
    normalized_answer = _strip_math_delimiters(str(answer))
    if not visible or not normalized_answer:
        raise ValueError(f"{benchmark} row {row_index} has empty question or answer")
    return ValidationProblem(
        benchmark=benchmark,
        benchmark_id=benchmark_id,
        row_index=row_index,
        raw_question=raw_question if raw_question is not None else str(question),
        question=visible,
        source_answer=answer,
        ground_truth=_boxed(normalized_answer),
    )


def lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", value).lower()))


def semantic_tokens(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", value).lower()
    for presentation in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:"):
        text = text.replace(presentation, "")
    for delimiter in (r"\[", r"\]", r"\(", r"\)", "$$", "$"):
        text = text.replace(delimiter, " ")
    for source, target in (
        (r"\dfrac", r"\frac"),
        (r"\tfrac", r"\frac"),
        (r"\cdots", "..."),
        (r"\ldots", "..."),
        (r"\dots", "..."),
        (r"\leqslant", "<="),
        (r"\leq", "<="),
        (r"\geqslant", ">="),
        (r"\geq", ">="),
        (r"\cdot", "*"),
        (r"\times", "*"),
    ):
        text = text.replace(source, target)
    return tuple(
        re.findall(
            r"\\[a-z]+|\d+(?:\.\d+)?|[a-z]+|<=|>=|!=|==|[+\-*/=<>^|]|[(),\[\]]",
            text,
        )
    )


def _lcs_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    token_masks: dict[str, int] = {}
    for index, token in enumerate(left):
        token_masks[token] = token_masks.get(token, 0) | (1 << index)
    row = 0
    for token in right:
        matches = token_masks.get(token, 0)
        x = matches | row
        row = x & ~(x - ((row << 1) | 1))
    return 2 * row.bit_count() / (len(left) + len(right))


def _training_problem(prompt: CalibrationPrompt) -> str:
    marker = "Put your final answer in \\boxed{...}.\n\n"
    if marker not in prompt.canonical_prompt:
        raise ValueError(f"training prompt lacks canonical problem marker: {prompt.prompt_id}")
    return prompt.canonical_prompt.split(marker, 1)[1]


def _strip_math_delimiters(value: str) -> str:
    text = value.strip()
    while len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    return text


def _boxed(value: str) -> str:
    text = value.strip()
    return text if r"\boxed{" in text else rf"\boxed{{{text}}}"


def _last_boxed_content(value: str) -> str | None:
    contents: list[str] = []
    start = 0
    marker = r"\boxed{"
    while True:
        marker_index = value.find(marker, start)
        if marker_index < 0:
            break
        index = marker_index + len(marker)
        depth = 1
        cursor = index
        while cursor < len(value) and depth:
            if value[cursor] == "{":
                depth += 1
            elif value[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            contents.append(value[index : cursor - 1])
        start = marker_index + 1
    return contents[-1] if contents else None


def _latex_equivalent(prediction: str, gold: str) -> bool:
    from rewardscope.verification.math_verify import MathVerifyLatexVerifier

    result = MathVerifyLatexVerifier(mode="training").verify(
        response=_boxed(prediction), ground_truth=_boxed(gold)
    )
    return bool(result.is_correct)
