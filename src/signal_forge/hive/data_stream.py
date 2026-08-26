"""Epoch-spanning raw prompt stream for one HIVE optimizer step."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from signal_forge.hive.identity import attach_stable_prompt_ids
from verl import DataProto


HIVE_DATA_STREAM_STATE_VERSION = 1
HIVE_DATALOADER_CHECKPOINT_FORMAT = "signal_forge_hive_epoch_stream_v1"


class HiveEpochSpanningDataStream:
    """Read complete raw batches continuously across dataset epoch boundaries.

    A fresh :class:`HiveStepRawBatchIterator` is created for each optimizer step.
    It excludes duplicate stable prompt IDs within that step while retaining source
    order. Rows read past a reconstructed ``b_raw`` boundary are retained so that
    checkpoint/resume continues at the exact logical stream position.
    """

    def __init__(
        self,
        dataloader: Any,
        *,
        total_epochs: int,
        raw_batch_size: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool) or total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer")
        if not isinstance(raw_batch_size, int) or isinstance(raw_batch_size, bool) or raw_batch_size <= 0:
            raise ValueError("raw_batch_size must be a positive integer")
        self.dataloader = dataloader
        self.total_epochs = total_epochs
        self.raw_batch_size = raw_batch_size
        self.epoch_index = 0
        self._source_iterator: Iterator[Any] | None = None
        self._pending_batch: DataProto | None = None
        if state is not None:
            self.load_state_dict(state)

    def begin_step(self) -> "HiveStepRawBatchIterator":
        """Return an iterator with a new optimizer-step duplicate-ID scope."""
        return HiveStepRawBatchIterator(self)

    def state_dict(self) -> dict[str, Any]:
        """Return stream state stored alongside StatefulDataLoader state."""
        return {
            "version": HIVE_DATA_STREAM_STATE_VERSION,
            "epoch_index": self.epoch_index,
            "total_epochs": self.total_epochs,
            "raw_batch_size": self.raw_batch_size,
            "pending_batch": self._pending_batch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("HIVE data stream state must be a mapping")
        version = state.get("version")
        if version != HIVE_DATA_STREAM_STATE_VERSION:
            raise ValueError(f"unsupported HIVE data stream state version: {version!r}")
        if int(state.get("total_epochs", -1)) != self.total_epochs:
            raise ValueError("HIVE data stream total_epochs does not match checkpoint")
        if int(state.get("raw_batch_size", -1)) != self.raw_batch_size:
            raise ValueError("HIVE data stream raw_batch_size does not match checkpoint")
        epoch_index = state.get("epoch_index")
        if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
            raise ValueError("HIVE data stream epoch_index must be an integer")
        if not 0 <= epoch_index <= self.total_epochs:
            raise ValueError("HIVE data stream epoch_index is outside configured epochs")
        pending_batch = state.get("pending_batch")
        if pending_batch is not None:
            if not isinstance(pending_batch, DataProto):
                raise TypeError("HIVE pending raw batch must be a DataProto")
            if not 0 < len(pending_batch) < self.raw_batch_size:
                raise ValueError("HIVE pending raw batch must be smaller than b_raw")
        self.epoch_index = epoch_index
        self._pending_batch = pending_batch
        self._source_iterator = None

    def _next_source_batch(self) -> DataProto:
        while self.epoch_index < self.total_epochs:
            if self._source_iterator is None:
                self._source_iterator = iter(self.dataloader)
            try:
                batch = next(self._source_iterator)
            except StopIteration:
                self._source_iterator = None
                self.epoch_index += 1
                continue
            if isinstance(batch, DataProto):
                raw_batch = batch
            elif isinstance(batch, Mapping):
                raw_batch = DataProto.from_single_dict(dict(batch))
            else:
                raise TypeError(f"unsupported HIVE dataloader batch type: {type(batch)!r}")
            if len(raw_batch) == 0:
                continue
            attach_stable_prompt_ids(raw_batch.non_tensor_batch)
            return raw_batch
        raise StopIteration

    def _next_unique_batch(self, seen_prompt_ids: set[str]) -> DataProto:
        pieces: list[DataProto] = []
        remaining = self.raw_batch_size
        while remaining > 0:
            if self._pending_batch is not None:
                source = self._pending_batch
                self._pending_batch = None
            else:
                source = self._next_source_batch()

            prompt_ids = tuple(np.asarray(source.non_tensor_batch["prompt_id"], dtype=object).tolist())
            eligible_indices = [
                index for index, prompt_id in enumerate(prompt_ids) if prompt_id not in seen_prompt_ids
            ]
            if not eligible_indices:
                continue

            take_indices = eligible_indices[:remaining]
            pieces.append(source.select_idxs(np.asarray(take_indices, dtype=np.int64)))
            seen_prompt_ids.update(prompt_ids[index] for index in take_indices)
            remaining -= len(take_indices)

            if remaining == 0:
                cutoff = take_indices[-1]
                if cutoff + 1 < len(source):
                    self._pending_batch = source.slice(cutoff + 1, None)

        output = pieces[0] if len(pieces) == 1 else DataProto.concat(pieces)
        if len(output) != self.raw_batch_size:
            raise RuntimeError("HIVE raw data stream emitted an incomplete b_raw batch")
        return output


class HiveStepRawBatchIterator:
    """Raw-batch iterator whose stable-ID exclusion scope is one optimizer step."""

    def __init__(self, stream: HiveEpochSpanningDataStream) -> None:
        self._stream = stream
        self._seen_prompt_ids: set[str] = set()

    @property
    def seen_prompt_ids(self) -> frozenset[str]:
        return frozenset(self._seen_prompt_ids)

    def __iter__(self) -> "HiveStepRawBatchIterator":
        return self

    def __next__(self) -> DataProto:
        return self._stream._next_unique_batch(self._seen_prompt_ids)
