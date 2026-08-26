from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from signal_forge.hive.data_stream import HiveEpochSpanningDataStream
from verl import DataProto


def _batch(*prompt_ids: str) -> dict[str, np.ndarray]:
    return {
        "raw_prompt": np.asarray(
            [[{"role": "user", "content": prompt_id}] for prompt_id in prompt_ids], dtype=object
        ),
        "extra_info": np.asarray([{"prompt_id": prompt_id} for prompt_id in prompt_ids], dtype=object),
        "row_value": np.asarray(prompt_ids, dtype=object),
    }


def _ids(batch: DataProto) -> tuple[str, ...]:
    return tuple(np.asarray(batch.non_tensor_batch["prompt_id"], dtype=object).tolist())


class _StatefulEpochLoader:
    def __init__(self, epochs: list[list[dict[str, np.ndarray]]]) -> None:
        self.epochs = epochs
        self.epoch = 0
        self.offset = 0

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        epoch = self.epoch
        while self.offset < len(self.epochs[epoch]):
            batch = self.epochs[epoch][self.offset]
            self.offset += 1
            yield batch
        self.epoch += 1
        self.offset = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])


def test_step_crosses_epochs_in_sampler_order_without_duplicate_ids() -> None:
    loader = _StatefulEpochLoader(
        [
            [_batch("p0", "p1"), _batch("p2", "p3")],
            [_batch("p3", "p4"), _batch("p5", "p0"), _batch("p6", "p7")],
        ]
    )
    stream = HiveEpochSpanningDataStream(loader, total_epochs=2, raw_batch_size=2)
    step = stream.begin_step()

    assert [_ids(next(step)) for _ in range(4)] == [
        ("p0", "p1"),
        ("p2", "p3"),
        ("p4", "p5"),
        ("p6", "p7"),
    ]
    assert step.seen_prompt_ids == frozenset(f"p{index}" for index in range(8))


def test_duplicate_exclusion_resets_for_next_optimizer_step() -> None:
    loader = _StatefulEpochLoader([[_batch("p0", "p1")], [_batch("p0", "p1")]])
    stream = HiveEpochSpanningDataStream(loader, total_epochs=2, raw_batch_size=2)

    assert _ids(next(stream.begin_step())) == ("p0", "p1")
    assert _ids(next(stream.begin_step())) == ("p0", "p1")


def test_checkpoint_resume_preserves_sampler_epoch_and_pending_rows() -> None:
    epochs = [
        [_batch("p0", "p1"), _batch("p2", "p3")],
        [_batch("p3", "p4"), _batch("p5", "p6"), _batch("p7", "p8")],
    ]
    loader = _StatefulEpochLoader(epochs)
    stream = HiveEpochSpanningDataStream(loader, total_epochs=2, raw_batch_size=2)
    step = stream.begin_step()

    assert _ids(next(step)) == ("p0", "p1")
    assert _ids(next(step)) == ("p2", "p3")
    assert _ids(next(step)) == ("p4", "p5")
    checkpoint_buffer = io.BytesIO()
    torch.save(
        {"dataloader_state": loader.state_dict(), "stream_state": stream.state_dict()},
        checkpoint_buffer,
    )
    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, weights_only=False)

    restored_loader = _StatefulEpochLoader(epochs)
    restored_loader.load_state_dict(checkpoint["dataloader_state"])
    restored_stream = HiveEpochSpanningDataStream(
        restored_loader,
        total_epochs=2,
        raw_batch_size=2,
        state=checkpoint["stream_state"],
    )

    assert restored_stream.epoch_index == 1
    assert _ids(next(restored_stream.begin_step())) == ("p6", "p7")


def test_exhaustion_does_not_emit_an_incomplete_raw_batch() -> None:
    loader = _StatefulEpochLoader([[_batch("p0", "p1")]])
    stream = HiveEpochSpanningDataStream(loader, total_epochs=1, raw_batch_size=2)
    step = stream.begin_step()

    assert _ids(next(step)) == ("p0", "p1")
    try:
        next(step)
    except StopIteration:
        pass
    else:
        raise AssertionError("configured epoch exhaustion must stop the raw prompt stream")


def test_state_validation_rejects_mismatched_resume_configuration() -> None:
    loader = _StatefulEpochLoader([[_batch("p0", "p1")]])
    stream = HiveEpochSpanningDataStream(loader, total_epochs=1, raw_batch_size=2)
    state = stream.state_dict()

    try:
        HiveEpochSpanningDataStream(loader, total_epochs=2, raw_batch_size=2, state=state)
    except ValueError as exc:
        assert "total_epochs" in str(exc)
    else:
        raise AssertionError("resume must reject a changed total_epochs value")


def _make_stateful_random_loader(seed: int) -> StatefulDataLoader:
    dataset = [f"p{index}" for index in range(8)]
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(data_source=dataset, generator=generator)
    return StatefulDataLoader(
        dataset=dataset,
        batch_size=4,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda prompt_ids: _batch(*prompt_ids),
    )


def test_real_stateful_random_sampler_resume_matches_continuous_cross_epoch_order() -> None:
    loader = _make_stateful_random_loader(seed=42)
    stream = HiveEpochSpanningDataStream(loader, total_epochs=3, raw_batch_size=4)
    assert len(next(stream.begin_step())) == 4

    checkpoint_buffer = io.BytesIO()
    torch.save(
        {"dataloader_state": loader.state_dict(), "stream_state": stream.state_dict()},
        checkpoint_buffer,
    )
    continuous_ids = (
        _ids(next(stream.begin_step())),
        _ids(next(stream.begin_step())),
    )

    checkpoint_buffer.seek(0)
    checkpoint = torch.load(checkpoint_buffer, weights_only=False)
    restored_loader = _make_stateful_random_loader(seed=42)
    restored_loader.load_state_dict(checkpoint["dataloader_state"])
    restored_stream = HiveEpochSpanningDataStream(
        restored_loader,
        total_epochs=3,
        raw_batch_size=4,
        state=checkpoint["stream_state"],
    )
    resumed_ids = (
        _ids(next(restored_stream.begin_step())),
        _ids(next(restored_stream.begin_step())),
    )

    assert resumed_ids == continuous_ids
    assert restored_stream.epoch_index == stream.epoch_index == 1
