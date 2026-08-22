from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rewardscope.sampling.schema import GeneratedResponse

from signal_forge.calibration import hive_dataset
from signal_forge.calibration.hive_dataset import (
    CalibrationPrompt,
    evaluate_generated_responses,
    load_calibration_prompts,
    load_dapo_calibration_prompts,
    load_math_calibration_prompts,
    normalize_dapo_row,
    select_balanced_merged_subset,
    select_fixed_subset,
    summarize_calibration,
    write_calibration_inputs,
    write_calibration_results,
)


def _prompt(prompt_id: str, source: str = "math") -> CalibrationPrompt:
    row_id = prompt_id.split(":", 1)[1]
    problem = f"question {prompt_id}"
    canonical = hive_dataset.format_canonical_math_prompt(problem)
    return CalibrationPrompt(
        prompt_id=prompt_id,
        dataset_source=source,
        source_row_id=row_id,
        raw_prompt=({"role": "user", "content": f"raw {problem}"},),
        canonical_prompt=canonical,
        messages=({"role": "user", "content": canonical},),
        source_ground_truth=r"\boxed{1}",
        ground_truth=r"\boxed{1}",
    )


def _dapo_row(row_id: str, *, question: str = "Solve me", gold: str = "1"):
    source_prompt = (
        f"{hive_dataset.DAPO_SOURCE_PREFIX}\n\n"
        f"{question}\n\n"
        f"{hive_dataset.DAPO_SOURCE_SUFFIX}"
    )
    return {
        "prompt": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": source_prompt},
        ],
        "reward_model": {"ground_truth": gold},
        "extra_info": {"index": row_id},
    }


def _responses(
    patterns: list[list[str]],
    *,
    length_keys: set[tuple[int, int]] | None = None,
) -> list[GeneratedResponse]:
    length_keys = length_keys or set()
    output = []
    for prompt_index, pattern in enumerate(patterns):
        for sample_index, value in enumerate(pattern):
            output.append(
                GeneratedResponse(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    response=value,
                    prompt_tokens=10 + prompt_index,
                    response_tokens=sample_index + 1,
                    finish_reason=(
                        "length"
                        if (prompt_index, sample_index) in length_keys
                        else "eos"
                    ),
                )
            )
    return output


def _verifier(_source, response, _gold, **_kwargs):
    if response == "correct":
        return {"reward": 1.0, "correct": True, "extracted": True}
    if response == "wrong":
        return {"reward": 0.1, "correct": False, "extracted": True}
    return {"reward": 0.0, "correct": False, "extracted": False}


def test_fixed_subset_is_deterministic_and_input_order_invariant():
    prompts = [_prompt(f"math:{index}") for index in range(20)]

    first = select_fixed_subset(prompts, sample_size=7, seed=42)
    second = select_fixed_subset(list(reversed(prompts)), sample_size=7, seed=42)

    assert [item.prompt_id for item in first] == [item.prompt_id for item in second]
    assert len(first) == 7
    assert select_fixed_subset(prompts, sample_size=7, seed=43) != first


def test_stable_ids_must_be_dataset_qualified_and_unique():
    with pytest.raises(ValueError, match="must start"):
        _prompt("dapo:1", source="math")
    with pytest.raises(ValueError, match="duplicate stable prompt_ids"):
        select_fixed_subset([_prompt("math:1"), _prompt("math:1")], sample_size=1)


def test_dapo_normalization_preserves_raw_and_uses_one_canonical_requirement():
    prompt = normalize_dapo_row(_dapo_row("row-7"))

    assert prompt.prompt_id == "dapo:row-7"
    assert prompt.dataset_source == "dapo"
    assert prompt.raw_prompt[0] == {"role": "system", "content": "You are helpful."}
    assert "Answer: $Answer" in prompt.raw_prompt[-1]["content"]
    assert prompt.canonical_prompt == (
        "Solve the following math problem step by step.\n"
        "Put your final answer in \\boxed{...}.\n\n"
        "Solve me"
    )
    assert "Answer:" not in prompt.canonical_prompt
    assert prompt.messages == ({"role": "user", "content": prompt.canonical_prompt},)
    assert prompt.source_ground_truth == "1"
    assert prompt.ground_truth == r"\boxed{1}"


def test_math_uses_the_same_canonical_prompt_and_preserves_raw(monkeypatch):
    example = SimpleNamespace(
        source_index=3,
        question="What is 1+1?",
        prompt="Original MATH prompt",
        ground_truth=r"\boxed{2}",
    )
    result = SimpleNamespace(
        examples=(example,),
        source_count=1,
        gold_parse_failure_count=0,
    )
    monkeypatch.setattr(hive_dataset, "load_math_result", lambda **_kwargs: result)

    prompts, _metadata = load_math_calibration_prompts()
    prompt = prompts[0]

    assert prompt.raw_prompt == ({"role": "user", "content": "Original MATH prompt"},)
    assert prompt.canonical_prompt == (
        "Solve the following math problem step by step.\n"
        "Put your final answer in \\boxed{...}.\n\n"
        "What is 1+1?"
    )
    assert prompt.messages == ({"role": "user", "content": prompt.canonical_prompt},)


def test_dapo_source_template_mismatch_is_rejected():
    row = _dapo_row("bad")
    row["prompt"][-1]["content"] = "Solve me\n\nAnswer: 1"

    with pytest.raises(ValueError, match="audited Answer-format source template"):
        normalize_dapo_row(row)


def test_dapo_boxed_gold_is_preserved():
    prompt = normalize_dapo_row(_dapo_row("boxed", gold=r"\boxed{7}"))

    assert prompt.source_ground_truth == r"\boxed{7}"
    assert prompt.ground_truth == r"\boxed{7}"


def test_selected_dapo_gold_is_validated_before_generation(monkeypatch):
    prompt = normalize_dapo_row(_dapo_row("invalid"))
    monkeypatch.setattr(
        hive_dataset,
        "extract_final_boxed_latex_gold",
        lambda _value: None,
    )

    with pytest.raises(ValueError, match="prompt_id='dapo:invalid'"):
        hive_dataset._canonicalize_selected_ground_truths((prompt,))


def test_dapo_deduplicates_identity_and_rejects_conflicting_duplicates(monkeypatch):
    rows = [_dapo_row("same"), _dapo_row("same"), _dapo_row("other")]
    monkeypatch.setattr(hive_dataset, "_load_dapo_dataset", lambda **_kwargs: rows)

    prompts, metadata = load_dapo_calibration_prompts()

    assert [item.prompt_id for item in prompts] == ["dapo:other", "dapo:same"]
    assert metadata == {"source_rows": 3, "duplicate_rows_removed": 1}

    monkeypatch.setattr(
        hive_dataset,
        "_load_dapo_dataset",
        lambda **_kwargs: [_dapo_row("same"), _dapo_row("same", question="changed")],
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_dapo_calibration_prompts()


def test_merged_selection_preserves_source_ids(monkeypatch):
    monkeypatch.setattr(
        hive_dataset,
        "_canonicalize_selected_ground_truths",
        lambda prompts: tuple(prompts),
    )
    math_prompts = tuple(_prompt(f"math:{index}") for index in range(4))
    dapo_prompts = tuple(_prompt(f"dapo:{index}", source="dapo") for index in range(4))
    monkeypatch.setattr(
        hive_dataset,
        "load_math_calibration_prompts",
        lambda **_kwargs: (math_prompts, {"source_rows": 4, "gold_parse_failures": 0}),
    )
    monkeypatch.setattr(
        hive_dataset,
        "load_dapo_calibration_prompts",
        lambda **_kwargs: (
            dapo_prompts,
            {"source_rows": 8, "duplicate_rows_removed": 4},
        ),
    )

    selected, inventory = load_calibration_prompts("dapo_math", sample_size=6, seed=42)
    balanced, _ = load_calibration_prompts(
        "dapo_math",
        sample_size=6,
        seed=42,
        balanced_merged=True,
    )

    assert len(selected) == 6
    assert all(item.prompt_id.startswith(("math:", "dapo:")) for item in selected)
    assert sum(item.dataset_source == "math" for item in balanced) == 3
    assert sum(item.dataset_source == "dapo" for item in balanced) == 3
    assert inventory.unique_prompts == {"math": 4, "dapo": 4}
    assert inventory.duplicate_rows_removed["dapo"] == 4

    with pytest.raises(ValueError, match="only for dapo_math"):
        load_calibration_prompts("math", sample_size=2, balanced_merged=True)


def test_balanced_merged_selection_is_exact_and_reproducible():
    math_prompts = tuple(_prompt(f"math:{index}") for index in range(20))
    dapo_prompts = tuple(_prompt(f"dapo:{index}", source="dapo") for index in range(20))

    first = select_balanced_merged_subset(
        math_prompts,
        dapo_prompts,
        sample_size=32,
        seed=42,
    )
    second = select_balanced_merged_subset(
        tuple(reversed(math_prompts)),
        tuple(reversed(dapo_prompts)),
        sample_size=32,
        seed=42,
    )

    assert [item.prompt_id for item in first] == [item.prompt_id for item in second]
    assert sum(item.dataset_source == "math" for item in first) == 16
    assert sum(item.dataset_source == "dapo" for item in first) == 16
    with pytest.raises(ValueError, match="even sample_size"):
        select_balanced_merged_subset(
            math_prompts,
            dapo_prompts,
            sample_size=31,
            seed=42,
        )


def test_three_state_groups_and_aggregate_statistics():
    prompts = [
        _prompt("math:easy"),
        _prompt("math:hard"),
        _prompt("math:other"),
        _prompt("dapo:mixed", source="dapo"),
    ]
    patterns = [
        ["correct"] * 8,
        ["wrong"] * 8,
        ["missing"] * 8,
        ["correct"] * 4 + ["wrong"] * 2 + ["missing"] * 2,
    ]
    rows = evaluate_generated_responses(
        prompts,
        _responses(patterns, length_keys={(2, 0), (3, 0)}),
        verifier=_verifier,
    )

    assert rows[0]["rewards"] == [1.0] * 8
    assert rows[0]["easy_zero_var"] is True
    assert rows[1]["hard_zero_var"] is True
    assert rows[2]["other_zero_var"] is True
    assert rows[3]["effective"] is True
    assert rows[3]["correct_count"] == 4

    summary = summarize_calibration(rows)
    assert summary["correct_count_histogram"]["8/8"] == 1
    assert summary["correct_count_histogram"]["4/8"] == 1
    assert summary["correct_count_histogram"]["0/8"] == 2
    assert summary["easy_zero_var_ratio"] == pytest.approx(0.25)
    assert summary["hard_zero_var_ratio"] == pytest.approx(0.25)
    assert summary["other_zero_var_ratio"] == pytest.approx(0.25)
    assert summary["effective_mixed_ratio"] == pytest.approx(0.25)
    assert summary["eos_finish_ratio"] == pytest.approx(30 / 32)
    assert summary["length_limit_finish_ratio"] == pytest.approx(2 / 32)
    assert summary["truncation_ratio"] == pytest.approx(2 / 32)
    assert summary["extraction_failure_ratio"] == pytest.approx(10 / 32)
    assert summary["extraction_failure_given_eos"] == pytest.approx(9 / 30)
    assert summary["extraction_failure_given_length_truncation"] == pytest.approx(1 / 2)
    lengths = summary["response_token_length_statistics"]
    assert lengths["count"] == 32
    assert summary["generated_response_tokens"] == sum(
        length for row in rows for length in row["response_lengths"]
    )
    assert set(("p50", "p90", "p95", "max")).issubset(lengths)
    assert set(summary["by_dataset_source"]) == {"dapo", "math"}
    assert summary["by_dataset_source"]["math"]["truncation_ratio"] == pytest.approx(1 / 24)
    assert summary["by_dataset_source"]["dapo"]["truncation_ratio"] == pytest.approx(1 / 8)


def test_verifier_structured_status_must_match_frozen_reward_semantics():
    prompts = [_prompt("math:1")]

    def inconsistent(_source, _response, _gold, **_kwargs):
        return {"reward": 0.0, "correct": False, "extracted": True}

    with pytest.raises(ValueError, match="violates frozen three-state semantics"):
        evaluate_generated_responses(
            prompts,
            _responses([["wrong"] * 8]),
            verifier=inconsistent,
        )


def test_generation_contract_requires_exactly_eight_unique_samples():
    prompts = [_prompt("math:1")]
    with pytest.raises(ValueError, match="exactly eight"):
        evaluate_generated_responses(prompts, _responses([["correct"] * 7]), verifier=_verifier)

    duplicate = _responses([["correct"] * 8])
    duplicate[-1] = GeneratedResponse(
        prompt_index=0,
        sample_index=0,
        response="correct",
        prompt_tokens=10,
        response_tokens=1,
        finish_reason="eos",
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_generated_responses(prompts, duplicate, verifier=_verifier)


def test_output_files_contain_selection_raw_results_and_aggregate(tmp_path):
    prompts = [_prompt("math:1")]
    inventory = hive_dataset.DatasetInventory(
        dataset="math",
        source_rows={"math": 1},
        unique_prompts={"math": 1},
        duplicate_rows_removed={"math": 0},
    )
    destination = write_calibration_inputs(
        tmp_path / "run",
        prompts=prompts,
        inventory=inventory,
        config={"group_size": 8},
    )
    rows = evaluate_generated_responses(
        prompts,
        _responses([["correct"] * 8]),
        verifier=_verifier,
    )
    write_calibration_results(destination, rows=rows)

    assert (destination / "config.json").exists()
    assert (destination / "inventory.json").exists()
    selection = json.loads((destination / "selected_prompts.jsonl").read_text())
    raw = json.loads((destination / "prompt_results.jsonl").read_text())
    aggregate = json.loads((destination / "aggregate.json").read_text())
    assert selection["prompt_id"] == "math:1"
    assert selection["raw_prompt"] == [{"role": "user", "content": "raw question math:1"}]
    assert selection["canonical_prompt"] == prompts[0].canonical_prompt
    assert raw["canonical_prompt"] == prompts[0].canonical_prompt
    assert raw["rewards"] == [1.0] * 8
    assert aggregate["correct_count_histogram"]["8/8"] == 1


def test_cli_defaults_are_frozen_for_calibration():
    args = hive_dataset.build_parser().parse_args(["--dataset", "math", "--prepare-only"])

    assert args.sample_size == 256
    assert args.seed == 42
    assert args.max_response_length == 768
    assert args.batch_size == 1
    assert args.balanced_merged is False
