from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from signal_forge.hive.prompt_entropy import (
    PromptEntropyEvaluator,
    PromptEntropyInputBatch,
    full_categorical_entropy,
)


class TokenLogitModel(torch.nn.Module):
    """Tiny causal-LM-shaped fixture whose logits depend on each input token."""

    def __init__(self, logits_by_token: torch.Tensor):
        super().__init__()
        self.logits_by_token = torch.nn.Parameter(logits_by_token.clone())
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, input_ids, attention_mask, position_ids=None):
        del attention_mask, position_ids
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        return SimpleNamespace(logits=self.logits_by_token[input_ids])


def _batch(
    prompt_ids: list[str],
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
    *,
    prompt_token_mask: list[list[int]] | None = None,
) -> PromptEntropyInputBatch:
    attention = torch.tensor(attention_mask, dtype=torch.bool)
    return PromptEntropyInputBatch(
        prompt_ids=prompt_ids,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=attention,
        position_ids=attention.long().cumsum(dim=-1) - 1,
        prompt_token_mask=(
            torch.tensor(prompt_token_mask, dtype=torch.bool) if prompt_token_mask is not None else None
        ),
    )


def _model() -> TokenLogitModel:
    return TokenLogitModel(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
                [-3.0, -1.0, 2.0, 6.0],
                [0.5, 0.5, -0.5, -0.5],
                [8.0, -8.0, -8.0, -8.0],
                [2.0, 1.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
    )


def test_full_categorical_entropy_matches_explicit_reference():
    logits = torch.tensor([[0.3, -0.7, 2.1], [-4.0, 0.5, 0.5]], dtype=torch.float64)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    expected = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)

    actual = full_categorical_entropy(logits)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_uniform_distribution_entropy_equals_log_vocab_size():
    logits = torch.zeros(3, 11)

    entropy = full_categorical_entropy(logits)

    assert torch.allclose(entropy, torch.full((3,), math.log(11)), atol=1e-6)


def test_peaked_distribution_entropy_approaches_zero():
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])

    entropy = full_categorical_entropy(logits)

    assert entropy.item() == pytest.approx(0.0, abs=1e-6)


def test_padding_invariance_for_left_and_right_padded_batches():
    model = _model()
    evaluator = PromptEntropyEvaluator(model)
    alone = _batch(["p"], [[1, 2, 3]], [[1, 1, 1]])
    left_padded = _batch(["neighbor", "p"], [[4, 5, 6, 1], [0, 1, 2, 3]], [[1, 1, 1, 1], [0, 1, 1, 1]])
    right_padded = _batch(["p", "neighbor"], [[1, 2, 3, 0], [4, 5, 6, 1]], [[1, 1, 1, 0], [1, 1, 1, 1]])

    expected = evaluator.compute(alone).records[0].entropy
    left_actual = evaluator.compute(left_padded).by_prompt_id["p"].entropy
    right_actual = evaluator.compute(right_padded).by_prompt_id["p"].entropy

    assert left_actual == pytest.approx(expected)
    assert right_actual == pytest.approx(expected)


def test_causal_shift_uses_exactly_l_minus_one_prompt_positions():
    model = _model()
    evaluator = PromptEntropyEvaluator(model)
    batch = _batch(["p"], [[1, 2, 3, 5]], [[1, 1, 1, 1]])
    token_entropies = full_categorical_entropy(model.logits_by_token.detach())

    result = evaluator.compute(batch).records[0]

    assert result.valid_token_count == 4
    assert result.predictive_position_count == 3
    assert result.entropy == pytest.approx(token_entropies[[1, 2, 3]].mean().item())
    assert result.entropy != pytest.approx(token_entropies[[2, 3, 5]].mean().item())


def test_batch_neighbors_do_not_change_prompt_entropy():
    evaluator = PromptEntropyEvaluator(_model())
    alone = _batch(["p"], [[2, 3, 4]], [[1, 1, 1]])
    with_neighbors = _batch(
        ["a", "p", "b"],
        [[1, 1, 0, 0], [0, 2, 3, 4], [5, 6, 1, 2]],
        [[1, 1, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]],
    )

    expected = evaluator.compute(alone).records[0].entropy
    actual = evaluator.compute(with_neighbors).by_prompt_id["p"].entropy

    assert actual == pytest.approx(expected)


def test_micro_batching_matches_full_batch_evaluation():
    batch = _batch(
        ["a", "b", "c", "d", "e"],
        [[1, 2, 0, 0], [2, 3, 4, 0], [3, 4, 5, 6], [4, 5, 0, 0], [5, 6, 1, 0]],
        [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]],
    )

    full = PromptEntropyEvaluator(_model()).compute(batch)
    micro = PromptEntropyEvaluator(_model(), micro_batch_size=2).compute(batch)

    assert micro.entropies == pytest.approx(full.entropies)
    assert micro.diagnostics.forward_passes == 3


def test_entropy_remains_finite_for_representative_extreme_logits():
    logits = torch.tensor([[1.0e4, -1.0e4, 0.0], [-80.0, -80.0, -80.0], [90.0, 0.0, -90.0]])

    entropy = full_categorical_entropy(logits)

    assert torch.isfinite(entropy).all()


def test_current_model_change_changes_prompt_entropy():
    model = _model()
    evaluator = PromptEntropyEvaluator(model)
    batch = _batch(["p"], [[1, 2, 3]], [[1, 1, 1]])
    before = evaluator.compute(batch).records[0].entropy

    with torch.no_grad():
        model.logits_by_token[1:3].zero_()
    after = evaluator.compute(batch).records[0].entropy

    assert after != pytest.approx(before)


def test_response_tokens_are_rejected_by_prompt_only_contract():
    evaluator = PromptEntropyEvaluator(_model())
    batch = _batch(
        ["p"],
        [[1, 2, 3, 4, 5]],
        [[1, 1, 1, 1, 1]],
        prompt_token_mask=[[1, 1, 1, 0, 0]],
    )

    with pytest.raises(ValueError, match="response|non-prompt"):
        evaluator.compute(batch)


def test_compute_disables_autograd_and_does_not_populate_gradients():
    model = _model()
    evaluator = PromptEntropyEvaluator(model)

    evaluator.compute(_batch(["p"], [[1, 2, 3]], [[1, 1, 1]]))

    assert model.grad_enabled_during_forward == [False]
    assert model.logits_by_token.grad is None


def test_prompt_must_have_at_least_two_valid_contiguous_tokens():
    evaluator = PromptEntropyEvaluator(_model())

    with pytest.raises(ValueError, match="at least two"):
        evaluator.compute(_batch(["short"], [[0, 1]], [[0, 1]]))

    with pytest.raises(ValueError, match="contiguous"):
        evaluator.compute(_batch(["gapped"], [[1, 0, 2]], [[1, 0, 1]]))
