from __future__ import annotations

import asyncio
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from signal_forge.hive.pre_rollout import HivePreRolloutConfig, HivePreRolloutStep
from signal_forge.hive.prompt_preprocessing import HivePromptPreprocessor
from signal_forge.hive.stage1 import Stage1Config, Stage1StepSelector
from signal_forge.hive.stage2 import Stage2Config, Stage2Selector
from signal_forge.hive.state import HiveSelectorState, PromptVisit
from signal_forge.hive.tests.test_prompt_preprocessing import QWEN25_TEXT_CHAT_TEMPLATE, _run_rollout
from verl import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "<|endoftext|>": 1, "<|im_start|>": 2, "<|im_end|>": 3},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        chat_template=QWEN25_TEXT_CHAT_TEMPLATE,
    )
    tokenizer.padding_side = "left"
    return tokenizer


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def _raw_batch(start: int, count: int) -> DataProto:
    prompt_ids = [f"prompt:{index:03d}" for index in range(start, start + count)]
    raw_prompts = [[{"role": "user", "content": f"Question {prompt_id}"}] for prompt_id in prompt_ids]
    extra_infos = [{"prompt_id": prompt_id, "source_row": index} for index, prompt_id in enumerate(prompt_ids, start)]
    return DataProto(
        batch=TensorDict(
            {
                "dummy_tensor": torch.zeros((count, 1), dtype=torch.uint8),
                "original_row": torch.arange(start, start + count, dtype=torch.int64),
            },
            batch_size=[count],
        ),
        non_tensor_batch={
            "raw_prompt": _object_array(raw_prompts),
            "extra_info": _object_array(extra_infos),
            "data_source": np.asarray(["test"] * count, dtype=object),
        },
        meta_info={"temperature": 1.0},
    )


def _batch_dict(batch: DataProto) -> dict:
    return {**{key: value for key, value in batch.batch.items()}, **batch.non_tensor_batch}


def _state(*, p_easy: float = 0.5) -> HiveSelectorState:
    return HiveSelectorState.create(group_size=8, seed=17, p_easy=p_easy)


def _step(
    *,
    state: HiveSelectorState | None = None,
    effective_batch_size: int = 4,
    epsilon_p: float = 0.01,
) -> HivePreRolloutStep:
    state = state or _state()
    return HivePreRolloutStep(
        stage1_selector=Stage1StepSelector(
            state.snapshot(),
            Stage1Config(lambda_weight=1.0, epsilon_p=epsilon_p),
        ),
        prompt_preprocessor=HivePromptPreprocessor(_tokenizer(), max_prompt_length=128),
        stage2_selector=Stage2Selector(Stage2Config()),
        config=HivePreRolloutConfig(effective_batch_size=effective_batch_size),
    )


def _entropy_result(prepared, *, reverse: bool = False):
    prompt_ids = [prompt.prompt_id for prompt in prepared.canonical_prompts]
    if reverse:
        prompt_ids.reverse()
    valid_by_id = {prompt.prompt_id: prompt.valid_token_count for prompt in prepared.canonical_prompts}
    entropy = [float(int(prompt_id.split(":")[-1])) for prompt_id in prompt_ids]
    count = len(prompt_ids)
    return tu.get_tensordict(
        tensor_dict={
            "prompt_id": prompt_ids,
            "entropy": torch.tensor(entropy, dtype=torch.float32),
            "valid_token_count": torch.tensor([valid_by_id[prompt_id] for prompt_id in prompt_ids]),
            "predictive_position_count": torch.tensor([valid_by_id[prompt_id] - 1 for prompt_id in prompt_ids]),
            "entropy_eval_latency_seconds": torch.full((count,), 0.25, dtype=torch.float64),
            "entropy_eval_peak_allocated_bytes": torch.full((count,), 100, dtype=torch.int64),
            "entropy_eval_peak_reserved_bytes": torch.full((count,), 200, dtype=torch.int64),
            "prompt_entropy_micro_batch_size": torch.ones(count, dtype=torch.int64),
        }
    )


def _finish(step: HivePreRolloutStep, raw_batch: DataProto, *, reverse_rpc: bool = False):
    prepared = step.prepare_round(raw_batch)
    rpc_result = None if prepared.entropy_rpc_batch is None else _entropy_result(prepared, reverse=reverse_rpc)
    return prepared, step.finish_round(prepared, rpc_result)


def test_raw_batch_through_stage1_stage2_selects_original_rows_by_stable_id():
    step = _step()
    prepared, result = _finish(step, _raw_batch(0, 16), reverse_rpc=True)

    expected_ids = tuple(record.prompt_id for record in result.stage2.kept)
    assert prepared.stage1.accepted_prompt_ids == tuple(f"prompt:{index:03d}" for index in range(16))
    assert tuple(result.selected_batch.non_tensor_batch["prompt_id"]) == expected_ids
    assert result.selected_batch.batch["original_row"].tolist() == [int(value.split(":")[-1]) for value in expected_ids]
    assert [info["prompt_id"] for info in result.selected_batch.non_tensor_batch["extra_info"]] == list(expected_ids)


def test_stable_prompt_id_and_temporary_rollout_uid_remain_separate():
    step = _step()
    _, result = _finish(step, _raw_batch(0, 16))
    stable_ids = result.selected_batch.non_tensor_batch["prompt_id"].copy()
    result.selected_batch.non_tensor_batch["uid"] = np.asarray(
        [f"temporary:{index}" for index in range(len(result.selected_batch))], dtype=object
    )

    trainer = object.__new__(RayPPOTrainer)
    gen_batch = trainer._get_gen_batch(result.selected_batch)

    assert np.array_equal(result.selected_batch.non_tensor_batch["prompt_id"], stable_ids)
    assert np.array_equal(gen_batch.non_tensor_batch["prompt_id"], stable_ids)
    assert not np.array_equal(gen_batch.non_tensor_batch["uid"], stable_ids)


def test_stage1_all_reject_short_circuits_entropy_rpc_and_returns_empty_stage2():
    state = _state(p_easy=0.0)
    raw_batch = _raw_batch(0, 8)
    for prompt_id in [info["prompt_id"] for info in raw_batch.non_tensor_batch["extra_info"]]:
        state.append_visit(
            prompt_id,
            PromptVisit.from_rewards(step=1, rewards=[1.0] * 8, group_size=8),
        )
    step = _step(state=state, epsilon_p=0.0)

    prepared = step.prepare_round(raw_batch)
    result = step.finish_round(prepared, None)

    assert prepared.entropy_rpc_batch is None
    assert result.stage1.diagnostics.accepted == 0
    assert result.stage2.diagnostics.input_count == 0
    assert len(result.selected_batch) == 0


def test_stage2_round_to_zero_does_not_crash_or_complete_accumulator():
    step = _step()
    _, result = _finish(step, _raw_batch(0, 8))

    assert result.stage2.diagnostics.pre_round_keep_count == 4
    assert result.stage2.diagnostics.post_round_keep_count == 0
    assert len(result.selected_batch) == 0
    assert step.candidate_actual == 0
    assert step.is_complete is False


def test_complete_stage2_partitions_accumulate_in_arrival_order_and_keep_overshoot():
    step = _step(effective_batch_size=8)
    _, first = _finish(step, _raw_batch(0, 16))
    _, second = _finish(step, _raw_batch(16, 16))
    final = step.finalize()

    expected_ids = first.selected_prompt_ids + second.selected_prompt_ids
    assert tuple(final.selected_batch.non_tensor_batch["prompt_id"]) == expected_ids
    assert final.candidate_target == 12
    assert len(final.selected_batch) == 16
    assert final.metrics["hive/candidate_actual"] == 16
    assert final.metrics["hive/candidate_overshoot"] == 4
    assert final.metrics["hive/candidate_actual_ratio_to_Bt"] == 2.0
    assert final.metrics["hive/candidate_accumulation_rounds"] == 2
    assert len(set(expected_ids)) == len(expected_ids)


def test_canonical_entropy_tokens_equal_actual_rollout_conditioning_after_integration():
    tokenizer = _tokenizer()
    raw_batch = _raw_batch(0, 16)
    step = HivePreRolloutStep(
        stage1_selector=Stage1StepSelector(_state().snapshot()),
        prompt_preprocessor=HivePromptPreprocessor(tokenizer, max_prompt_length=128),
        stage2_selector=Stage2Selector(),
        config=HivePreRolloutConfig(effective_batch_size=4),
    )
    prepared = step.prepare_round(raw_batch)
    first_prompt = prepared.canonical_prompts[0]
    _, server = asyncio.run(
        _run_rollout(tokenizer, raw_batch.non_tensor_batch["raw_prompt"][0], prompt_length=128)
    )

    assert server.prompt_ids == list(first_prompt.input_ids)
    assert tu.get(prepared.entropy_rpc_batch, "prompt_id")[0] == first_prompt.prompt_id
    entropy_ids = prepared.entropy_rpc_batch["input_ids"][0][prepared.entropy_rpc_batch["attention_mask"][0]]
    assert entropy_ids.tolist() == list(first_prompt.input_ids)


def test_candidate_target_requires_exact_integer_three_halves_relationship():
    assert HivePreRolloutConfig(effective_batch_size=8).candidate_target == 12
    with pytest.raises(ValueError, match="divisible by 2"):
        HivePreRolloutConfig(effective_batch_size=7)


class _LifecycleCheckpointManager:
    def __init__(self):
        self.calls = []

    def sleep_replicas(self):
        self.calls.append("sleep")

    def wake_up_replicas(self):
        self.calls.append("wake")


class _EntropyActorWorkerGroup:
    def __init__(self):
        self.calls = 0

    def compute_prompt_entropy(self, batch):
        self.calls += 1
        ids = tu.get(batch, "prompt_id")
        count = len(ids)
        return tu.get_tensordict(
            tensor_dict={
                "prompt_id": ids,
                "entropy": torch.arange(count, dtype=torch.float32),
                "valid_token_count": batch["attention_mask"].sum(-1),
                "predictive_position_count": batch["attention_mask"].sum(-1) - 1,
                "entropy_eval_latency_seconds": torch.full((count,), 0.1, dtype=torch.float64),
                "entropy_eval_peak_allocated_bytes": torch.full((count,), 10, dtype=torch.int64),
                "entropy_eval_peak_reserved_bytes": torch.full((count,), 20, dtype=torch.int64),
            }
        )


def test_trainer_accumulates_raw_batches_and_uses_one_sleep_and_existing_weight_sync():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"hive": {"lambda_weight": 1.0, "epsilon_p": 0.01}},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
        }
    )
    trainer.global_steps = 3
    trainer.hive_selector_state = _state()
    trainer._hive_prompt_preprocessor = HivePromptPreprocessor(_tokenizer(), max_prompt_length=128)
    trainer._hive_stage2_selector = Stage2Selector()
    trainer._hive_pre_rollout_config = HivePreRolloutConfig(effective_batch_size=8)
    trainer.checkpoint_manager = _LifecycleCheckpointManager()
    trainer.actor_rollout_wg = _EntropyActorWorkerGroup()
    raw_batches = [_batch_dict(_raw_batch(0, 16)), _batch_dict(_raw_batch(16, 16))]

    rng_before = copy.deepcopy(trainer.hive_selector_state.selector_rng_state)

    result, selector = trainer._select_hive_pre_rollout_candidates(raw_batches[0], iter(raw_batches[1:]))

    assert len(result.selected_batch) == 16
    assert trainer.actor_rollout_wg.calls == 2
    assert trainer.checkpoint_manager.calls == ["sleep", "wake"]
    assert trainer.hive_selector_state.selector_rng_state == rng_before
    trainer._commit_hive_stage1_rng(selector)
    assert trainer.hive_selector_state.selector_rng_state != rng_before
    assert trainer.hive_selector_state.global_step == 3


def test_hive_disabled_initialization_bypasses_all_hive_components(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"algorithm": {"hive": {"enable": False}}})
    trainer._hive_configuration = None
    trainer.hive_selector_state = None
    trainer._hive_prompt_preprocessor = None
    trainer._hive_stage2_selector = None
    trainer._hive_pre_rollout_config = None
    monkeypatch.setattr(
        ray_trainer,
        "validate_hive_prompt_preprocessing_scope",
        lambda config: pytest.fail("HIVE-disabled path invoked HIVE preflight"),
    )

    trainer._initialize_hive_selector_state()

    assert trainer.hive_selector_state is None
    assert trainer._hive_prompt_preprocessor is None
