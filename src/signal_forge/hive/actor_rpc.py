"""TensorDict contract adapter for actor-side HIVE prompt entropy RPCs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tensordict import NonTensorData, TensorDict

from signal_forge.hive.prompt_entropy import (
    PromptEntropyBatchResult,
    PromptEntropyEvaluator,
    PromptEntropyInputBatch,
)
from verl.utils import tensordict_utils as tu


@dataclass(frozen=True)
class ActorEntropyCallDiagnostics:
    latency_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    micro_batch_size: int


def compute_prompt_entropy_rpc(
    actor: torch.nn.Module,
    data: TensorDict,
    *,
    device: torch.device | str | None = None,
) -> TensorDict:
    """Evaluate already-tokenized prompt-only inputs on the supplied current actor."""
    prompt_batch = _prepare_prompt_batch(data, device=device)
    micro_batch_size = tu.get(data, "prompt_entropy_micro_batch_size", 1)
    entropy_chunk_size = tu.get(data, "prompt_entropy_chunk_size", 2048)
    evaluator = PromptEntropyEvaluator(
        actor,
        micro_batch_size=micro_batch_size,
        entropy_chunk_size=entropy_chunk_size,
        model_forward_kwargs={"use_cache": False},
    )

    if len(prompt_batch.prompt_ids) == 0:
        return _build_output(
            prompt_ids=(),
            result=None,
            diagnostics=ActorEntropyCallDiagnostics(
                latency_seconds=0.0,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
                micro_batch_size=micro_batch_size,
            ),
        )

    cuda_device = _cuda_profile_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    started_at = time.perf_counter()
    result = evaluator.compute(prompt_batch)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    latency_seconds = time.perf_counter() - started_at

    diagnostics = ActorEntropyCallDiagnostics(
        latency_seconds=latency_seconds,
        peak_allocated_bytes=(torch.cuda.max_memory_allocated(cuda_device) if cuda_device is not None else 0),
        peak_reserved_bytes=(torch.cuda.max_memory_reserved(cuda_device) if cuda_device is not None else 0),
        micro_batch_size=micro_batch_size,
    )
    return _build_output(
        prompt_ids=prompt_batch.prompt_ids,
        result=result,
        diagnostics=diagnostics,
    )


def _prepare_prompt_batch(
    data: TensorDict,
    *,
    device: torch.device | str | None,
) -> PromptEntropyInputBatch:
    if not isinstance(data, TensorDict):
        raise TypeError("prompt entropy RPC data must be a TensorDict")
    for key in ("prompt_id", "input_ids", "attention_mask"):
        if key not in data:
            raise ValueError(f"prompt entropy RPC input is missing required field {key!r}")

    prompt_ids = tu.get(data, "prompt_id")
    if isinstance(prompt_ids, (str, bytes)) or not isinstance(prompt_ids, Sequence):
        raise ValueError("prompt_id must be a per-prompt sequence")
    prompt_ids = tuple(prompt_ids)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-2 tensor [batch, sequence]")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")
    if input_ids.shape[0] != len(prompt_ids):
        raise ValueError("prompt_id count must match input_ids batch dimension")

    position_ids = data.get("position_ids")
    prompt_token_mask = data.get("prompt_token_mask")
    for name, tensor in (("position_ids", position_ids), ("prompt_token_mask", prompt_token_mask)):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a tensor when provided")

    return PromptEntropyInputBatch(
        prompt_ids=prompt_ids,
        input_ids=_move_tensor(input_ids, device),
        attention_mask=_move_tensor(attention_mask, device),
        position_ids=_move_tensor(position_ids, device),
        prompt_token_mask=_move_tensor(prompt_token_mask, device),
    )


def _move_tensor(tensor: torch.Tensor | None, device: torch.device | str | None) -> torch.Tensor | None:
    if tensor is None or device is None:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _cuda_profile_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return resolved


def _build_output(
    *,
    prompt_ids: Sequence[str],
    result: PromptEntropyBatchResult | None,
    diagnostics: ActorEntropyCallDiagnostics,
) -> TensorDict:
    records = result.records if result is not None else ()
    batch_size = len(prompt_ids)
    if len(records) != batch_size:
        raise RuntimeError("prompt entropy evaluator output count does not match RPC input")

    tensor_fields = {
        "entropy": torch.tensor([record.entropy for record in records], dtype=torch.float32),
        "valid_token_count": torch.tensor(
            [record.valid_token_count for record in records], dtype=torch.int64
        ),
        "predictive_position_count": torch.tensor(
            [record.predictive_position_count for record in records], dtype=torch.int64
        ),
        "entropy_eval_latency_seconds": torch.full(
            (batch_size,), diagnostics.latency_seconds, dtype=torch.float64
        ),
        "entropy_eval_peak_allocated_bytes": torch.full(
            (batch_size,), diagnostics.peak_allocated_bytes, dtype=torch.int64
        ),
        "entropy_eval_peak_reserved_bytes": torch.full(
            (batch_size,), diagnostics.peak_reserved_bytes, dtype=torch.int64
        ),
        "prompt_entropy_micro_batch_size": torch.full(
            (batch_size,), diagnostics.micro_batch_size, dtype=torch.int64
        ),
    }
    if batch_size == 0:
        return TensorDict(
            {"prompt_id": NonTensorData(tuple()), **tensor_fields},
            batch_size=[0],
        )
    return tu.get_tensordict(
        tensor_dict={"prompt_id": list(prompt_ids), **tensor_fields}
    )
