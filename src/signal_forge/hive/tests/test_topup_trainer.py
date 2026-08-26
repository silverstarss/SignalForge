from __future__ import annotations

import json

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from signal_forge.hive.stage1 import Stage1StepSelector
from signal_forge.hive.state import HiveSelectorState, PromptVisit
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _hive_step_start_metrics


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


def test_step_start_metrics_expose_frozen_history_and_controller_state():
    state = HiveSelectorState.create(
        group_size=8,
        seed=7,
        p_easy=0.47,
        p_hard=0.53,
        p_default=0.5,
    )
    state.append_visit(
        "math:1",
        PromptVisit.from_rewards(step=1, rewards=[1.0] * 8, group_size=8),
    )
    state.global_step = 1

    metrics = _hive_step_start_metrics(Stage1StepSelector(state.snapshot()))

    assert metrics["hive/selector_snapshot_global_step"] == 1.0
    assert metrics["hive/history_prompts_at_step_start"] == 1.0
    assert metrics["hive/history_visits_at_step_start"] == 1.0
    assert metrics["hive/p_easy_step_start"] == 0.47
    assert metrics["hive/p_hard_step_start"] == 0.53
    assert metrics["hive/p_default_step_start"] == 0.5


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
    assert timing["mock_rollout"] == 0.25
    assert timing["rollout_wall_seconds"] >= 0.0
    assert timing["reward_wall_seconds"] >= 0.0
    assert trainer.checkpoint_manager.calls == ["sleep"]

    prompt_ids = trainer.async_rollout_manager.input_prompt_ids.tolist()
    temporary_uids = trainer.async_rollout_manager.input_uids.tolist()
    assert prompt_ids == (["stable:0"] * 8) + (["stable:1"] * 8)
    assert len(set(temporary_uids[:8])) == 1
    assert len(set(temporary_uids[8:])) == 1
    assert temporary_uids[0] != temporary_uids[8]
    assert set(temporary_uids).isdisjoint({"stable:0", "stable:1"})


class _DiagnosticTokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(token) for token in row.tolist()) for row in values]


def test_hive_round_dump_preserves_reward_failure_evidence(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "hive_round_dump_enabled": True,
                "rollout_data_dir": str(tmp_path),
                "rollout_dump_max_records": None,
            }
        }
    )
    trainer.global_steps = 3
    trainer.hive_selector_state = HiveSelectorState.create(group_size=8, seed=7)
    trainer.tokenizer = _DiagnosticTokenizer()
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                "responses": torch.tensor([[5, 6], [7, 8]], dtype=torch.int64),
                "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.int64),
            },
            batch_size=[2],
        ),
        non_tensor_batch={
            "prompt_id": np.asarray(["math:1", "dapo:2"], dtype=object),
            "data_source": np.asarray(["math", "dapo"], dtype=object),
            "uid": np.asarray(["temporary-1", "temporary-2"], dtype=object),
            "reward_model": _object_array(
                [{"ground_truth": "\\boxed{1}"}, {"ground_truth": "\\boxed{2}"}]
            ),
        },
    )

    trainer._log_hive_round_data(
        batch=batch,
        reward_tensor=torch.tensor([[0.0, 0.0], [0.0, 0.1]], dtype=torch.float32),
        reward_extra_infos_dict={
            "fallback_used": [True, False],
            "failure_reason": ["parse_timeout", ""],
            "extracted": [False, True],
        },
        round_index=2,
    )

    dump_file = tmp_path / "hive_round_diagnostics" / "step_3" / "round_2" / "3.jsonl"
    records = [json.loads(line) for line in dump_file.read_text().splitlines()]
    assert [record["prompt_id"] for record in records] == ["math:1", "dapo:2"]
    np.testing.assert_allclose([record["score"] for record in records], [0.0, 0.1])
    assert [record["fallback_used"] for record in records] == [True, False]
    assert [record["failure_reason"] for record in records] == ["parse_timeout", ""]
    assert [record["response_token_count"] for record in records] == [2, 1]
    assert [record["hive_round_index"] for record in records] == [2, 2]
