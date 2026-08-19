"""Current-policy prompt entropy evaluation for HIVE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from verl.utils import torch_functional as verl_F


def full_categorical_entropy(logits: torch.Tensor, *, chunk_size: int | None = None) -> torch.Tensor:
    """Return exact full-vocabulary categorical entropy in float32 or float64."""
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
        raise ValueError("logits must be a tensor with a vocabulary dimension")
    if logits.shape[-1] < 2:
        raise ValueError("logits vocabulary dimension must contain at least two entries")
    if not logits.is_floating_point():
        raise ValueError("logits must have a floating-point dtype")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")

    if chunk_size is None:
        stable_logits = logits if logits.dtype == torch.float64 else logits.float()
        return verl_F.entropy_from_logits(stable_logits)

    output_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_entropy = verl_F.entropy_from_logits_with_chunking(
        flat_logits,
        chunk_size=chunk_size,
    )
    return flat_entropy.reshape(output_shape)


@dataclass(frozen=True)
class PromptEntropyInputBatch:
    """Already-tokenized, prompt-only inputs from the rollout formatting path."""

    prompt_ids: Sequence[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None = None
    prompt_token_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PromptEntropyRecord:
    prompt_id: str
    entropy: float
    valid_token_count: int
    predictive_position_count: int


@dataclass(frozen=True)
class PromptEntropyDiagnostics:
    batch_size: int
    micro_batch_size: int
    forward_passes: int
    total_valid_tokens: int
    total_predictive_positions: int
    minimum_entropy: float
    maximum_entropy: float
    mean_entropy: float


@dataclass(frozen=True)
class PromptEntropyBatchResult:
    records: tuple[PromptEntropyRecord, ...]
    diagnostics: PromptEntropyDiagnostics

    @property
    def entropies(self) -> tuple[float, ...]:
        return tuple(record.entropy for record in self.records)

    @property
    def by_prompt_id(self) -> Mapping[str, PromptEntropyRecord]:
        return {record.prompt_id: record for record in self.records}


@dataclass(frozen=True)
class _ValidatedBatch:
    prompt_ids: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    valid_token_counts: tuple[int, ...]


class PromptEntropyEvaluator:
    """Teacher-force prompts through the current actor and average exact entropy."""

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        micro_batch_size: int | None = None,
        entropy_chunk_size: int = 2048,
        model_forward_kwargs: Mapping[str, Any] | None = None,
    ):
        if not isinstance(actor, torch.nn.Module):
            raise TypeError("actor must be a torch.nn.Module representing the current policy")
        if micro_batch_size is not None and (
            isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size <= 0
        ):
            raise ValueError("micro_batch_size must be a positive integer or None")
        if isinstance(entropy_chunk_size, bool) or not isinstance(entropy_chunk_size, int) or entropy_chunk_size <= 0:
            raise ValueError("entropy_chunk_size must be a positive integer")
        self.actor = actor
        self.micro_batch_size = micro_batch_size
        self.entropy_chunk_size = entropy_chunk_size
        self.model_forward_kwargs = dict(model_forward_kwargs or {})

    def compute(self, batch: PromptEntropyInputBatch) -> PromptEntropyBatchResult:
        """Compute V(x) over exactly L-1 predictive prompt positions per row."""
        validated = _validate_batch(batch)
        batch_size = len(validated.prompt_ids)
        micro_batch_size = min(self.micro_batch_size or batch_size, batch_size)
        records: list[PromptEntropyRecord] = []
        forward_passes = 0

        was_training = self.actor.training
        self.actor.eval()
        try:
            with torch.inference_mode():
                for start in range(0, batch_size, micro_batch_size):
                    end = min(start + micro_batch_size, batch_size)
                    input_ids = validated.input_ids[start:end]
                    attention_mask = validated.attention_mask[start:end]
                    model_kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    if validated.position_ids is not None:
                        model_kwargs["position_ids"] = validated.position_ids[start:end]
                    model_kwargs.update(self.model_forward_kwargs)

                    output = self.actor(**model_kwargs)
                    logits = _extract_logits(output)
                    _validate_logits(
                        logits,
                        expected_batch_size=end - start,
                        expected_sequence_length=input_ids.shape[1],
                    )

                    token_entropies = full_categorical_entropy(
                        logits, chunk_size=self.entropy_chunk_size
                    )
                    predictive_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
                    masked_entropies = token_entropies[:, :-1].masked_fill(
                        ~predictive_mask, 0.0
                    )
                    entropy_sums = masked_entropies.sum(dim=-1)
                    predictive_counts = predictive_mask.sum(dim=-1)
                    prompt_entropies = entropy_sums / predictive_counts
                    if not torch.isfinite(prompt_entropies).all():
                        raise FloatingPointError("prompt entropy computation produced NaN or infinity")

                    entropy_values = prompt_entropies.detach().cpu().tolist()
                    predictive_count_values = predictive_counts.detach().cpu().tolist()
                    for offset, entropy in enumerate(entropy_values):
                        batch_index = start + offset
                        records.append(
                            PromptEntropyRecord(
                                prompt_id=validated.prompt_ids[batch_index],
                                entropy=float(entropy),
                                valid_token_count=validated.valid_token_counts[batch_index],
                                predictive_position_count=int(predictive_count_values[offset]),
                            )
                        )
                    forward_passes += 1

                    del output, logits, token_entropies, entropy_sums, prompt_entropies
        finally:
            self.actor.train(was_training)

        entropy_values = tuple(record.entropy for record in records)
        diagnostics = PromptEntropyDiagnostics(
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            forward_passes=forward_passes,
            total_valid_tokens=sum(record.valid_token_count for record in records),
            total_predictive_positions=sum(record.predictive_position_count for record in records),
            minimum_entropy=min(entropy_values),
            maximum_entropy=max(entropy_values),
            mean_entropy=sum(entropy_values) / len(entropy_values),
        )
        return PromptEntropyBatchResult(records=tuple(records), diagnostics=diagnostics)


def _validate_batch(batch: PromptEntropyInputBatch) -> _ValidatedBatch:
    if not isinstance(batch, PromptEntropyInputBatch):
        raise TypeError("batch must be a PromptEntropyInputBatch")

    prompt_ids = tuple(batch.prompt_ids)
    if not prompt_ids:
        raise ValueError("prompt entropy batch must not be empty")
    for index, prompt_id in enumerate(prompt_ids):
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt_id at index {index} must be a non-empty string")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt entropy batch contains duplicate prompt_id values")

    if not isinstance(batch.input_ids, torch.Tensor) or batch.input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-2 tensor [batch, sequence]")
    if batch.input_ids.shape[0] != len(prompt_ids):
        raise ValueError("input_ids batch dimension must match prompt_ids")
    if not isinstance(batch.attention_mask, torch.Tensor) or batch.attention_mask.shape != batch.input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    attention_mask = _normalize_binary_mask(batch.attention_mask, "attention_mask")

    if batch.prompt_token_mask is not None:
        if (
            not isinstance(batch.prompt_token_mask, torch.Tensor)
            or batch.prompt_token_mask.shape != batch.input_ids.shape
        ):
            raise ValueError("prompt_token_mask must match input_ids shape")
        prompt_token_mask = _normalize_binary_mask(batch.prompt_token_mask, "prompt_token_mask")
        if not torch.equal(prompt_token_mask, attention_mask):
            raise ValueError("prompt-only entropy input contains response or other non-prompt valid tokens")

    if batch.position_ids is not None:
        if not isinstance(batch.position_ids, torch.Tensor):
            raise ValueError("position_ids must be a tensor or None")
        if (
            batch.position_ids.shape[0] != batch.input_ids.shape[0]
            or batch.position_ids.shape[-1] != batch.input_ids.shape[1]
        ):
            raise ValueError("position_ids must share input_ids batch and sequence dimensions")

    valid_token_counts: list[int] = []
    for index, row_mask in enumerate(attention_mask):
        valid_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        valid_count = int(valid_indices.numel())
        if valid_count < 2:
            raise ValueError(f"prompt at index {index} must contain at least two valid tokens")
        if int(valid_indices[-1] - valid_indices[0] + 1) != valid_count:
            raise ValueError(f"prompt at index {index} must occupy one contiguous token span")
        valid_token_counts.append(valid_count)

    return _ValidatedBatch(
        prompt_ids=prompt_ids,
        input_ids=batch.input_ids,
        attention_mask=attention_mask,
        position_ids=batch.position_ids,
        valid_token_counts=tuple(valid_token_counts),
    )


def _normalize_binary_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must contain only zero/one values")
    return mask.bool()


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, Mapping) and "logits" in output:
        logits = output["logits"]
    elif isinstance(output, tuple) and output:
        logits = output[0]
    else:
        raise TypeError("actor output must expose a logits tensor")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("actor output logits must be a tensor")
    return logits


def _validate_logits(logits: torch.Tensor, *, expected_batch_size: int, expected_sequence_length: int) -> None:
    if logits.ndim != 3:
        raise ValueError("actor logits must have shape [batch, sequence, vocabulary]")
    if logits.shape[:2] != (expected_batch_size, expected_sequence_length):
        raise ValueError("actor logits batch/sequence dimensions do not match prompt inputs")
    if logits.shape[-1] < 2:
        raise ValueError("actor logits vocabulary dimension must contain at least two entries")
    if not logits.is_floating_point():
        raise ValueError("actor logits must have a floating-point dtype")
