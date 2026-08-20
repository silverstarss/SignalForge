from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from signal_forge.hive.tests.test_prompt_preprocessing import qwen25_tokenizer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _trainer(tokenizer, *, effective_batch_size=8, raw_batch_size=32):
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = tokenizer
    trainer.config = OmegaConf.create(
        {
            "trainer": {"nnodes": 1, "n_gpus_per_node": 1, "total_training_steps": 2},
            "algorithm": {
                "adv_estimator": "grpo",
                "hive": {
                    "enable": True,
                    "group_size": 8,
                    "seed": 42,
                    "p_easy_initial": 0.5,
                    "p_hard_initial": 0.5,
                    "p_default": 0.5,
                    "lambda_weight": 1.0,
                    "epsilon_p": 0.01,
                    "upper_trim_ratio": 0.25,
                    "keep_ratio": 0.5,
                    "prompt_entropy_micro_batch_size": 1,
                    # Reduced unit-test scale; this is not a formal reproduction value.
                    "b_min": 8,
                }
            },
            "actor_rollout_ref": {
                "actor": {"use_fused_kernels": False, "ulysses_sequence_parallel_size": 1},
                "rollout": {"name": "vllm", "n": 8, "prompt_length": 128},
            },
            "data": {
                "train_batch_size": effective_batch_size,
                "gen_batch_size": raw_batch_size,
                "apply_chat_template_kwargs": {},
                "continuous_token": {"enable": False},
            },
        }
    )
    trainer._hive_configuration = None
    trainer.hive_selector_state = None
    trainer._hive_prompt_preprocessor = None
    trainer._hive_stage2_selector = None
    trainer._hive_pre_rollout_config = None
    return trainer


def test_phase5c_preflight_builds_approved_configuration(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer)

    trainer._initialize_hive_selector_state()

    assert trainer.hive_selector_state is not None
    assert trainer._hive_pre_rollout_config.candidate_target == 12
    assert trainer._hive_pre_rollout_config.prompt_entropy_micro_batch_size == 1
    assert trainer._hive_topup_config.eta == 1.25
    assert trainer._hive_topup_config.b_min == 8
    assert trainer._hive_topup_config.max_topup_rounds == 8
    assert trainer._hive_topup_config.candidate_cap == 12


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("trainer.n_gpus_per_node", 2, "single GPU"),
        ("actor_rollout_ref.actor.use_fused_kernels", True, "fused"),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2, "Ulysses"),
        ("actor_rollout_ref.rollout.name", "sglang", "vLLM"),
        ("algorithm.hive.lambda_weight", 0.5, "lambda_weight=1.0"),
    ],
)
def test_phase5c_preflight_rejects_unsupported_paths(qwen25_tokenizer, path, value, message):
    trainer = _trainer(qwen25_tokenizer)
    OmegaConf.update(trainer.config, path, value)

    with pytest.raises(ValueError, match=message):
        trainer._initialize_hive_selector_state()


def test_hive_preflight_rejects_non_grpo(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer)
    trainer.config.algorithm.adv_estimator = "reinforce_plus_plus"

    with pytest.raises(ValueError, match="requires the GRPO"):
        trainer._initialize_hive_selector_state()


def test_hive_preflight_rejects_b_min_above_candidate_cap(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer)
    trainer.config.algorithm.hive.b_min = 13

    with pytest.raises(ValueError, match="b_min <= B_cand"):
        trainer._initialize_hive_selector_state()


def test_phase5c_preflight_rejects_non_integral_candidate_target(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer, effective_batch_size=7)

    with pytest.raises(ValueError, match="divisible by 2"):
        trainer._initialize_hive_selector_state()


def test_phase5c_preflight_rejects_raw_batch_that_always_rounds_to_zero(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer, raw_batch_size=8)

    with pytest.raises(ValueError, match="too small"):
        trainer._initialize_hive_selector_state()


def test_phase5c_preflight_requires_explicit_optimizer_step_count(qwen25_tokenizer):
    trainer = _trainer(qwen25_tokenizer)
    trainer.config.trainer.total_training_steps = None

    with pytest.raises(ValueError, match="total_training_steps"):
        trainer._initialize_hive_selector_state()
