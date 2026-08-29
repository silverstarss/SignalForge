from __future__ import annotations

import json

import pytest

from signal_forge.observability.best_checkpoint import load_best_checkpoint_metadata


METRIC_NAME = "val/six_benchmark_mean_accuracy"


def _write_metadata(root, **overrides):
    payload = {
        "checkpoint_path": None,
        "checkpoint_saved_on_update": False,
        "global_step": 0,
        "metric_name": METRIC_NAME,
        "metric_value": 0.375,
    }
    payload.update(overrides)
    (root / "best_checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resume_restores_step_zero_best_without_checkpoint_path(tmp_path):
    _write_metadata(tmp_path)

    metadata = load_best_checkpoint_metadata(
        str(tmp_path),
        expected_metric_name=METRIC_NAME,
        resume_global_step=100,
    )

    assert metadata is not None
    assert metadata.metric_name == METRIC_NAME
    assert metadata.metric_value == 0.375
    assert metadata.global_step == 0
    assert metadata.checkpoint_path is None
    assert metadata.checkpoint_saved_on_update is False


def test_resume_restores_existing_scheduled_best_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "global_step_50"
    checkpoint_path.mkdir()
    _write_metadata(
        tmp_path,
        checkpoint_path=str(checkpoint_path),
        global_step=50,
        metric_value=0.5,
    )

    metadata = load_best_checkpoint_metadata(
        str(tmp_path),
        expected_metric_name=METRIC_NAME,
        resume_global_step=100,
    )

    assert metadata is not None
    assert metadata.global_step == 50
    assert metadata.checkpoint_path == str(checkpoint_path.resolve())


def test_missing_metadata_supports_legacy_resume(tmp_path):
    assert (
        load_best_checkpoint_metadata(
            str(tmp_path),
            expected_metric_name=METRIC_NAME,
            resume_global_step=100,
        )
        is None
    )


def test_resume_rejects_future_best_step(tmp_path):
    _write_metadata(tmp_path, global_step=150)

    with pytest.raises(ValueError, match="between zero and the resume step"):
        load_best_checkpoint_metadata(
            str(tmp_path),
            expected_metric_name=METRIC_NAME,
            resume_global_step=100,
        )


def test_resume_rejects_metric_mismatch(tmp_path):
    _write_metadata(tmp_path, metric_name="val/wrong")

    with pytest.raises(ValueError, match="metric mismatch"):
        load_best_checkpoint_metadata(
            str(tmp_path),
            expected_metric_name=METRIC_NAME,
            resume_global_step=100,
        )


def test_resume_rejects_missing_referenced_best_checkpoint(tmp_path):
    missing_path = tmp_path / "global_step_50"
    _write_metadata(tmp_path, checkpoint_path=str(missing_path), global_step=50)

    with pytest.raises(ValueError, match="path does not exist"):
        load_best_checkpoint_metadata(
            str(tmp_path),
            expected_metric_name=METRIC_NAME,
            resume_global_step=100,
        )


def test_resume_rejects_checkpoint_path_step_mismatch(tmp_path):
    checkpoint_path = tmp_path / "global_step_50"
    checkpoint_path.mkdir()
    _write_metadata(tmp_path, checkpoint_path=str(checkpoint_path), global_step=25)

    with pytest.raises(ValueError, match="does not match its global_step"):
        load_best_checkpoint_metadata(
            str(tmp_path),
            expected_metric_name=METRIC_NAME,
            resume_global_step=100,
        )
