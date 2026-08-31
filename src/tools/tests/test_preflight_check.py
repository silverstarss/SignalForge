from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tools import preflight_check as pf


def base_cfg(tmp_path: Path) -> dict:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    rows = []
    for i in range(30):
        rows.append(
            {
                "data_source": "gsm8k" if i % 2 else "math_level_3",
                "prompt": [{"role": "user", "content": f"q{i}\\n\\nPlease put final answer in \\boxed{{}}."}],
                "reward_model": {"style": "rule", "ground_truth": "2"},
                "extra_info": {"prompt_id": f"p{i}"},
            }
        )
    pd.DataFrame(rows).to_parquet(train, index=False)
    pd.DataFrame(rows[:4]).to_parquet(val, index=False)
    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"
    return {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "norm_adv_by_std_in_grpo": True},
        "data": {
            "train_files": str(train),
            "val_files": str(val),
            "train_batch_size": 5,
            "max_prompt_length": 512,
            "max_response_length": 768,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "prompt_key": "prompt",
            "train_max_samples": -1,
            "val_max_samples": -1,
        },
        "actor_rollout_ref": {
            "model": {"path": str(tmp_path / "model")},
            "actor": {
                "ppo_mini_batch_size": 5,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_epochs": 1,
                "use_dynamic_bsz": True,
                "use_kl_loss": False,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "entropy_coeff": 0,
                "ulysses_sequence_parallel_size": 1,
                "optim": {"lr": 1e-6},
            },
            "rollout": {
                "name": "vllm",
                "n": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.55,
                "max_model_len": 1280,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 64,
            },
        },
        "reward": {
            "reward_manager": {"source": "register", "name": "naive"},
            "custom_reward_function": {"path": "adapter.py", "name": "compute_score", "reward_kwargs": {}},
        },
        "trainer": {
            "logger": ["console"],
            "project_name": "p",
            "experiment_name": "e",
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "default_local_dir": str(ckpt),
            "rollout_data_dir": str(out / "rollout"),
            "validation_data_dir": str(out / "val"),
            "total_epochs": 9,
            "total_training_steps": 50,
            "save_freq": 10,
            "test_freq": 10,
            "resume_mode": "disable",
            "use_v1": False,
        },
    }


def statuses(rep: pf.Reporter, name: str) -> list[str]:
    return [c.status for c in rep.checks if c.name == name]


def test_previous_40_response_vs_160_minibatch_failure(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 20
    rep = pf.Reporter()
    out = pf.check_batches(rep, cfg, Path.cwd())
    assert out["responses_per_step_responses"] == 40
    assert out["normalized_ppo_mini_batch_size_responses"] == 160
    assert "FAIL" in statuses(rep, "E.batch.divisibility")


def test_six_steps_per_epoch_vs_fifty_requested_failure(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"]["total_epochs"] = 8
    rep = pf.Reporter()
    dataset = pf.check_dataset(rep, cfg, {"train": cfg["data"]["train_files"], "val": cfg["data"]["val_files"]}, False, None)
    assert dataset["steps_per_epoch"] == 6
    rep2 = pf.Reporter()
    pf.check_steps(rep2, cfg, dataset)
    assert "FAIL" in statuses(rep2, "D.steps.epoch_budget")


def test_val_only_does_not_require_training_epoch_or_save_test_frequency(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["trainer"].update(
        {
            "val_only": True,
            "save_freq": -1,
            "test_freq": -1,
            "total_training_steps": 100,
        }
    )
    rep = pf.Reporter()
    pf.check_steps(rep, cfg, {"steps_per_epoch": 0})

    assert statuses(rep, "D.steps.epoch_budget") == ["PASS"]
    assert statuses(rep, "D.steps.save_freq") == ["PASS"]
    assert statuses(rep, "D.steps.test_freq") == ["PASS"]


def test_zero_effective_dataset_rows_after_filtering(tmp_path):
    cfg = base_cfg(tmp_path)
    empty = tmp_path / "empty.parquet"
    pd.DataFrame([], columns=["data_source", "prompt", "reward_model", "extra_info"]).to_parquet(empty, index=False)
    cfg["data"]["train_files"] = str(empty)
    rep = pf.Reporter()
    pf.check_dataset(rep, cfg, {"train": str(empty), "val": cfg["data"]["val_files"]}, False, None)
    assert "FAIL" in statuses(rep, "C.dataset.train.effective_rows")


def test_invalid_micro_batch_divisibility(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 3
    rep = pf.Reporter()
    pf.check_batches(rep, cfg, Path.cwd())
    assert "FAIL" in statuses(rep, "E.batch.actor_micro")


def test_context_length_overflow(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["actor_rollout_ref"]["rollout"]["max_model_len"] = 100
    rep = pf.Reporter()
    pf.check_context_lengths(rep, cfg, {"max_position_embeddings": 100})
    assert "FAIL" in statuses(rep, "F.length.context")


def test_broken_artifact_symlink(tmp_path):
    cfg = base_cfg(tmp_path)
    broken = tmp_path / "broken_train.parquet"
    broken.symlink_to(tmp_path / "missing.parquet")
    cfg["data"]["train_files"] = str(broken)
    rep = pf.Reporter()
    pf.check_paths(rep, cfg, tmp_path, allow_existing_output=True)
    assert "FAIL" in statuses(rep, "B.path.train.symlink")


def test_existing_output_checkpoint_collision(tmp_path):
    cfg = base_cfg(tmp_path)
    ckpt = Path(cfg["trainer"]["default_local_dir"])
    ckpt.mkdir(parents=True)
    (ckpt / "sentinel.txt").write_text("old", encoding="utf-8")
    rep = pf.Reporter()
    pf.check_paths(rep, cfg, tmp_path, allow_existing_output=False)
    assert "FAIL" in statuses(rep, "B.path.checkpoint.collision")


def test_reward_returning_nan_fails():
    rep = pf.Reporter()
    pf.validate_reward_outputs(rep, [{"score": math.nan}], expected_count=1)
    assert "FAIL" in statuses(rep, "G.reward.finite")


def test_all_equal_group_rewards_are_finite():
    result = pf.simulate_grpo(8, [1.0] * 8)
    assert result["advantages_finite"]
    assert result["returns_finite"]
    assert result["advantage_std"] == 0.0


def test_wandb_disabled_when_logging_expected(tmp_path, monkeypatch):
    cfg = base_cfg(tmp_path)
    monkeypatch.setenv("EXPECT_WANDB", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\npython train.py | tee train.log\n", encoding="utf-8")
    rep = pf.Reporter()
    pf.check_logging(rep, cfg, script, {})
    assert "FAIL" in statuses(rep, "J.logging.wandb")
