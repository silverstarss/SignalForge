from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from signal_forge.hive import actor_rpc
from signal_forge.hive.actor_rpc import compute_prompt_entropy_rpc
from verl.utils import tensordict_utils as tu
from verl.workers import engine_workers
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


class TrackingTokenLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_by_token = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            )
        )
        self.forward_batch_sizes: list[int] = []

    def forward(self, input_ids, attention_mask, position_ids=None, use_cache=None):
        del attention_mask, position_ids, use_cache
        self.forward_batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.logits_by_token[input_ids])


def _batch(
    prompt_ids: list[str],
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
    *,
    micro_batch_size: int | None = None,
):
    if not prompt_ids:
        return TensorDict(
            {
                "prompt_id": NonTensorData(tuple()),
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.bool),
                "position_ids": torch.empty((0, 0), dtype=torch.long),
            },
            batch_size=[0],
        )
    attention = torch.tensor(attention_mask, dtype=torch.bool)
    non_tensor = {}
    if micro_batch_size is not None:
        non_tensor["prompt_entropy_micro_batch_size"] = micro_batch_size
    return tu.get_tensordict(
        tensor_dict={
            "prompt_id": prompt_ids,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention,
            "position_ids": attention.long().cumsum(dim=-1) - 1,
        },
        non_tensor_dict=non_tensor,
    )


def test_rpc_preserves_prompt_ids_result_order_and_valid_counts():
    model = TrackingTokenLogitModel()
    data = _batch(
        ["prompt:b", "prompt:a"],
        [[1, 2, 3], [0, 2, 3]],
        [[1, 1, 1], [0, 1, 1]],
        micro_batch_size=1,
    )

    result = compute_prompt_entropy_rpc(model, data)

    assert tu.get(result, "prompt_id") == ["prompt:b", "prompt:a"]
    assert tu.get(result, "valid_token_count").tolist() == [3, 2]
    assert tu.get(result, "predictive_position_count").tolist() == [2, 1]
    assert torch.isfinite(tu.get(result, "entropy")).all()
    assert model.forward_batch_sizes == [1, 1]


def test_rpc_exposes_cpu_latency_and_memory_diagnostics():
    result = compute_prompt_entropy_rpc(
        TrackingTokenLogitModel(),
        _batch(["prompt:1"], [[1, 2, 3]], [[1, 1, 1]]),
    )

    assert tu.get(result, "entropy_eval_latency_seconds").item() >= 0.0
    assert tu.get(result, "entropy_eval_peak_allocated_bytes").item() == 0
    assert tu.get(result, "entropy_eval_peak_reserved_bytes").item() == 0
    assert tu.get(result, "prompt_entropy_micro_batch_size").item() == 1


def test_rpc_empty_batch_returns_empty_output_without_model_forward():
    model = TrackingTokenLogitModel()
    data = _batch([], [], [])
    data["input_ids"] = torch.empty((0, 3), dtype=torch.long)
    data["attention_mask"] = torch.empty((0, 3), dtype=torch.bool)
    data["position_ids"] = torch.empty((0, 3), dtype=torch.long)

    result = compute_prompt_entropy_rpc(model, data)

    assert len(result) == 0
    assert tu.get(result, "prompt_id") == ()
    assert tu.get(result, "entropy").numel() == 0
    assert tu.get(result, "valid_token_count").numel() == 0
    assert model.forward_batch_sizes == []


@pytest.mark.parametrize("missing_key", ["prompt_id", "input_ids", "attention_mask"])
def test_rpc_rejects_missing_required_batch_fields(missing_key):
    data = _batch(["prompt:1"], [[1, 2]], [[1, 1]])
    del data[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        compute_prompt_entropy_rpc(TrackingTokenLogitModel(), data)


def test_rpc_propagates_prompt_entropy_evaluator_errors(monkeypatch):
    class RaisingEvaluator:
        def __init__(self, *args, **kwargs):
            pass

        def compute(self, batch):
            raise FloatingPointError("synthetic evaluator failure")

    monkeypatch.setattr(actor_rpc, "PromptEntropyEvaluator", RaisingEvaluator)

    with pytest.raises(FloatingPointError, match="synthetic evaluator failure"):
        compute_prompt_entropy_rpc(
            TrackingTokenLogitModel(),
            _batch(["prompt:1"], [[1, 2]], [[1, 1]]),
        )


@pytest.mark.parametrize("micro_batch_size", [0, -1, 1.5, True])
def test_rpc_rejects_invalid_micro_batch_configuration(micro_batch_size):
    data = _batch(
        ["prompt:1"],
        [[1, 2]],
        [[1, 1]],
        micro_batch_size=micro_batch_size,
    )

    with pytest.raises(ValueError, match="micro_batch_size"):
        compute_prompt_entropy_rpc(TrackingTokenLogitModel(), data)


def test_actor_ray_entry_point_delegates_to_current_actor_worker():
    sentinel = tu.get_tensordict(
        tensor_dict={
            "prompt_id": ["prompt:1"],
            "entropy": torch.tensor([1.0]),
            "valid_token_count": torch.tensor([2]),
        }
    )

    class FakeActorWorker:
        def __init__(self):
            self.received = None

        def compute_prompt_entropy(self, data):
            self.received = data
            return sentinel

    worker = object.__new__(ActorRolloutRefWorker)
    worker.actor = FakeActorWorker()
    data = _batch(["prompt:1"], [[1, 2]], [[1, 1]])

    result = worker.compute_prompt_entropy(data)

    assert result is sentinel
    assert worker.actor.received is data


def test_training_worker_uses_its_live_engine_module(monkeypatch):
    actor = TrackingTokenLogitModel()
    sentinel = tu.get_tensordict(
        tensor_dict={
            "prompt_id": ["prompt:1"],
            "entropy": torch.tensor([1.0]),
            "valid_token_count": torch.tensor([2]),
        }
    )
    captured = {}

    class FakeEngine:
        module = actor
        ulysses_sequence_parallel_size = 1
        _autocast_dtype = torch.float32

        def eval_mode(self, *, disable_auto_offload):
            captured["disable_auto_offload"] = disable_auto_offload
            return nullcontext()

        def is_mp_src_rank_with_outputs(self):
            return True

    def fake_compute(current_actor, data, *, device):
        captured.update(actor=current_actor, data=data, device=device)
        return sentinel

    monkeypatch.setattr(actor_rpc, "compute_prompt_entropy_rpc", fake_compute)
    monkeypatch.setattr(
        engine_workers,
        "get_torch_device",
        lambda: SimpleNamespace(current_device=lambda: 0),
    )

    worker = object.__new__(TrainingWorker)
    worker.engine = FakeEngine()
    worker.engine_config = SimpleNamespace(use_fused_kernels=False)
    worker.device_name = "cpu"
    data = _batch(["prompt:1"], [[1, 2]], [[1, 1]])

    result = worker.compute_prompt_entropy(data)

    assert tu.get(result, "prompt_id") == ["prompt:1"]
    assert captured["actor"] is actor
    assert captured["data"] is data
    assert captured["device"] == torch.device("cpu:0")
    assert captured["disable_auto_offload"] is False
