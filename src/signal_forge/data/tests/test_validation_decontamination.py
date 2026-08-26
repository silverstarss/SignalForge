from __future__ import annotations

import pytest

from signal_forge.calibration.hive_dataset import CalibrationPrompt
from signal_forge.data.validation_decontamination import (
    audit_training_pool,
    normalize_gaokao_row,
    normalize_olympiad_row,
    olympiad_multiple_answers_equal,
    remove_confirmed_overlaps,
    semantic_tokens,
    split_top_level_answers,
    ValidationProblem,
    _lcs_similarity,
)


def _prompt(prompt_id: str, problem: str) -> CalibrationPrompt:
    source, row_id = prompt_id.split(":", 1)
    canonical = (
        "Solve the following math problem step by step.\n"
        "Put your final answer in \\boxed{...}.\n\n"
        f"{problem}"
    )
    return CalibrationPrompt(
        prompt_id=prompt_id,
        dataset_source=source,
        source_row_id=row_id,
        raw_prompt=({"role": "user", "content": problem},),
        canonical_prompt=canonical,
        messages=({"role": "user", "content": canonical},),
        source_ground_truth="1",
        ground_truth=r"\boxed{1}",
    )


def _validation(benchmark_id: str, problem: str) -> ValidationProblem:
    return ValidationProblem(
        benchmark="math500",
        benchmark_id=benchmark_id,
        row_index=0,
        raw_question=problem,
        question=problem,
        source_answer="1",
        ground_truth=r"\boxed{1}",
    )


@pytest.mark.parametrize(
    ("row_index", "question", "gold", "forbidden"),
    [
        (
            167,
            "An entry in a grid is called a saddle point if it is the largest number "
            "in its row and the smallest number in its column. Suppose that each cell "
            "in a $3 \\times 3$ grid is filled with a real number, each chosen "
            "independently and uniformly at random from the interval [0,1]. Compute "
            "the probability that this grid has at least one saddle "
            "point.answers: $\\frac{3}{10}$",
            r"\boxed{\frac{3}{10}}",
            r"\frac{3}{10}",
        ),
        (
            192,
            "There are 435 voting members of the US House of Representatives. If b "
            "voting members are in favor of a certain bill, which expression represents "
            "the percentage of the voting members in favor of the bill?(A) "
            "$100\\left(\\frac{b}{435}\\right)$(B) "
            "$100\\left(\\frac{435}{b}\\right)$(C) "
            "$435\\left(\\frac{b}{100}\\right)$(D) $435(100b)$Answer: A",
            r"\boxed{A}",
            "Answer: A",
        ),
    ],
)
def test_gaokao_leaked_gold_is_removed_from_visible_prompt(
    row_index, question, gold, forbidden
):
    normalized = normalize_gaokao_row(
        {"question": question, "answer": ""}, row_index=row_index
    )
    assert normalized.raw_question == question
    assert normalized.ground_truth == gold
    assert forbidden not in normalized.question


def test_olympiad_multiple_answer_normalization_preserves_source_semantics():
    normalized = normalize_olympiad_row(
        {
            "id": 1709,
            "question": "Find all values.",
            "final_answer": ["$(1,8,19), (2,7,13), (4,5,7)$"],
            "is_multiple_answer": True,
        },
        row_index=3,
    )
    assert normalized.multiple_answers == ("(1,8,19)", "(2,7,13)", "(4,5,7)")
    assert normalized.ground_truth == r"\boxed{(1,8,19), (2,7,13), (4,5,7)}"


def test_top_level_answer_split_keeps_ordered_tuples_intact():
    assert split_top_level_answers("(1, 8, 19), (2, 7, 13), 4") == (
        "(1, 8, 19)",
        "(2, 7, 13)",
        "4",
    )


@pytest.mark.parametrize(
    "response",
    [
        r"\boxed{\frac{1}{2}, 2}",
        r"\boxed{2, \frac{1}{2}}",
        r"The result is \boxed{ 2 , \frac{2}{4} }.",
    ],
)
def test_olympiad_multiple_answers_accept_order_format_and_fraction_equivalence(response):
    assert olympiad_multiple_answers_equal(response, (r"\frac{1}{2}", "2"))


@pytest.mark.parametrize(
    "response",
    [r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}, 3}"],
)
def test_olympiad_multiple_answers_reject_missing_or_wrong_answer(response):
    assert not olympiad_multiple_answers_equal(response, (r"\frac{1}{2}", "2"))


def test_olympiad_top_level_is_unordered_but_tuple_components_remain_ordered():
    expected = ("(1,2)", "3")
    assert olympiad_multiple_answers_equal(r"\boxed{3, (1,2)}", expected)
    assert not olympiad_multiple_answers_equal(r"\boxed{3, (2,1)}", expected)


def test_semantic_normalization_preserves_math_operators():
    plus = "Find roots of $z^4 + z^2 + 1 = 0$."
    minus = "Find roots of $z^4 - z^2 + 1 = 0$."
    assert semantic_tokens(plus) != semantic_tokens(minus)


def test_audit_classifies_exact_manual_paraphrase_and_different_problem():
    train = (
        _prompt("math:a", "Find the value of $x+1$ when $x=2$ and explain your work."),
        _prompt("math:b", "Compute $x+1$ for $x=2$ and explain your work."),
        _prompt("math:c", "Find the value of $x-1$ when $x=2$ and explain your work."),
    )
    validation = (
        _validation("same", "Find the value of $x+1$ when $x=2$ and explain your work."),
    )
    review = {
        "schema_version": 1,
        "ngram_size": 3,
        "manual_review_lcs_threshold": 0.8,
        "manual_review_char_threshold": 0.9,
        "manual_same_problem_pairs": [
            {
                "benchmark": "math500",
                "benchmark_id": "same",
                "train_prompt_id": "math:b",
            }
        ],
    }
    records = audit_training_pool(train, validation, review)
    by_id = {record.train_prompt_id: record for record in records}
    assert (by_id["math:a"].match_type, by_id["math:a"].decision) == ("A", "remove")
    assert (by_id["math:b"].match_type, by_id["math:b"].decision) == ("B", "remove")
    assert (by_id["math:c"].match_type, by_id["math:c"].decision) == ("C", "keep")
    clean, removed = remove_confirmed_overlaps(train, records)
    assert [prompt.prompt_id for prompt in clean] == ["math:c"]
    assert removed == ("math:a", "math:b")


def test_bit_parallel_lcs_matches_reference_dynamic_programming():
    def reference(left, right):
        table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for i, lhs in enumerate(left, 1):
            for j, rhs in enumerate(right, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1
                    if lhs == rhs
                    else max(table[i - 1][j], table[i][j - 1])
                )
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return 2 * table[-1][-1] / (len(left) + len(right))

    examples = [
        ((), ()),
        (("a",), ()),
        (("a", "b", "c"), ("a", "b", "c")),
        (("a", "b", "c"), ("b", "a", "c")),
        (("x", "x", "y"), ("x", "y", "x")),
    ]
    for left, right in examples:
        assert _lcs_similarity(left, right) == reference(left, right)
