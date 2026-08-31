from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from signal_forge.observability.diagnostic_validation import (
    build_validation_compute_metrics,
    validate_diagnostic_validation_contract,
)


def _config(tmp_path, **overrides):
    trainer = {
        "validation_label": "pilot_diagnostic_step80",
        "val_only": True,
        "val_before_train": True,
        "update_best_checkpoint_metadata": False,
        "resume_mode": "resume_path",
        "resume_from_path": str(tmp_path / "formal" / "global_step_80"),
        "default_local_dir": str(tmp_path / "diagnostic"),
        "del_local_ckpt_after_load": False,
    }
    trainer.update(overrides)
    return OmegaConf.create({"trainer": trainer})


def test_diagnostic_validation_contract_accepts_isolated_val_only_process(tmp_path):
    validate_diagnostic_validation_contract(_config(tmp_path))


@pytest.mark.parametrize(
    "override, message",
    [
        ({"val_only": False}, "val_only"),
        ({"update_best_checkpoint_metadata": True}, "best-checkpoint"),
        ({"resume_mode": "disable"}, "explicit checkpoint"),
        ({"del_local_ckpt_after_load": True}, "must not delete"),
    ],
)
def test_diagnostic_validation_contract_rejects_state_mutating_modes(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        validate_diagnostic_validation_contract(_config(tmp_path, **override))


def test_regular_training_without_label_is_unchanged(tmp_path):
    config = OmegaConf.create({"trainer": {"validation_label": None}})
    validate_diagnostic_validation_contract(config)


def test_validation_compute_metrics_are_separate_from_training_counters():
    metrics = build_validation_compute_metrics(
        generated_responses=8,
        generated_prompt_tokens=80,
        generated_response_tokens=120,
        validation_n=2,
        wall_time_seconds=3.5,
        label="pilot_diagnostic_step80",
    )

    assert metrics["validation/generated_prompts"] == 4.0
    assert metrics["validation/generated_responses"] == 8.0
    assert metrics["validation/rollout_tokens"] == 200.0
    assert metrics["physical_compute/validation_rollout_tokens"] == 200.0
    assert metrics["diagnostic/pilot_diagnostic_step80"] == 1.0
    assert all(not key.startswith("compute/") for key in metrics)
