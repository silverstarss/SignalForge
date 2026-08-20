from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from signal_forge.hive.state import HiveSelectorState
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _prompt_batch() -> DataProto:
    prompt_ids = ["stable:0", "stable:1"]
    return DataProto(
        batch=TensorDict(
            {"source_row": torch.tensor([[0], [1]], dtype=torch.int64)},
            batch_size=[2],
        ),
        non_tensor_batch={
            "prompt_id": np.asarray(prompt_ids, dtype=object),
            "extra_info": _object_array([{"prompt_id": value} for value in prompt_ids]),
            "data_source": np.asarray(["test", "test"], dtype=object),
            "raw_prompt": _object_array(
                [
                    [{"role": "user", "content": "first"}],
                    [{"role": "user", "content": "second"}],
                ]
            ),
        },
    )


class _RolloutManager:
    def __init__(self):
        self.input_prompt_ids = None
        self.input_uids = None

    def generate_sequences(self, batch: DataProto) -> DataProto:
        self.input_prompt_ids = batch.non_tensor_batch["prompt_id"].copy()
        self.input_uids = batch.non_tensor_batch["uid"].copy()
        count = len(self.input_prompt_ids)
        rewards = np.asarray(([1.0] * 8) + ([0.1] * 8), dtype=object)
        return DataProto(
            batch=TensorDict(
                {
                    "responses": torch.ones((count, 2), dtype=torch.int64),
                    "response_mask": torch.ones((count, 2), dtype=torch.int64),
                    "attention_mask": torch.ones((count, 4), dtype=torch.int64),
                    "rm_scores": torch.tensor(rewards.astype(float), dtype=torch.float32).unsqueeze(-1),
                },
                batch_size=[count],
            ),
            non_tensor_batch={
                "reward": rewards,
                "extracted": np.asarray([True] * count, dtype=object),
                "correct": np.asarray(([True] * 8) + ([False] * 8), dtype=object),
            },
            meta_info={
                "timing": {"mock_rollout": 0.25},
                "reward_extra_keys": ["reward", "extracted", "correct"],
            },
        )


class _CheckpointManager:
    def __init__(self):
        self.calls = []

    def sleep_replicas(self):
        self.calls.append("sleep")


def test_trainer_topup_rollout_preserves_stable_identity_and_complete_groups():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": AdvantageEstimator.GRPO},
            "actor_rollout_ref": {"rollout": {"n": 8}},
        }
    )
    trainer.global_steps = 11
    trainer.hive_selector_state = HiveSelectorState.create(group_size=8, seed=7)
    trainer.async_rollout_manager = _RolloutManager()
    trainer.checkpoint_manager = _CheckpointManager()
    trainer.use_rm = False

    batch, reward_tensor, reward_infos, timing = trainer._rollout_hive_topup_candidates(
        _prompt_batch(), curr_step_profile=False
    )

    assert len(batch) == 16
    assert reward_tensor.shape == (16, 1)
    assert reward_infos["reward"][:8].tolist() == [1.0] * 8
    assert reward_infos["reward"][8:].tolist() == [0.1] * 8
    assert reward_infos["extracted"].tolist() == [True] * 16
    assert timing == {"mock_rollout": 0.25}
    assert trainer.checkpoint_manager.calls == ["sleep"]

    prompt_ids = trainer.async_rollout_manager.input_prompt_ids.tolist()
    temporary_uids = trainer.async_rollout_manager.input_uids.tolist()
    assert prompt_ids == (["stable:0"] * 8) + (["stable:1"] * 8)
    assert len(set(temporary_uids[:8])) == 1
    assert len(set(temporary_uids[8:])) == 1
    assert temporary_uids[0] != temporary_uids[8]
    assert set(temporary_uids).isdisjoint({"stable:0", "stable:1"})
