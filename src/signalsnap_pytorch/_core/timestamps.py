from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data_access import RuntimeSource, get_source_length, read_source, relative_float64_offsets
from .planning import WindowBatch

# Sequential reads of about 512 KiB for float64 timestamps.
_TIMESTAMP_READ_CHUNK_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class PreparedTimestampBatch:
    """Events assigned to one physical ``(B, m)`` window batch."""

    relative_event_times: NDArray[np.float64]
    window_indices: NDArray[np.int64]
    global_event_indices: NDArray[np.int64]
    estimate_count: int
    windows_per_estimate: int


class TimestampCursor:
    """Sequential bounded reader for one validated timestamp source."""

    def __init__(self, source: RuntimeSource, observation_start: float) -> None:
        self.source = source
        self.observation_start = observation_start
        self.source_length = get_source_length(source)
        self._last_interval_stop: float | None = None
        self._reset()

    def _reset(self) -> None:
        """Return to the beginning for a new placement traversal."""

        self._next_read_index = 0
        self._buffer_start = 0
        self._buffer = np.empty(0, dtype=np.float64)
        self._position = 0

    def _load_next_chunk(self) -> bool:
        """Load and rebase the next bounded source chunk."""

        if self._next_read_index >= self.source_length:
            self._buffer = np.empty(0, dtype=np.float64)
            self._position = 0
            return False

        start = self._next_read_index
        stop = min(start + _TIMESTAMP_READ_CHUNK_SIZE, self.source_length)
        raw_values = read_source(self.source, start, stop)

        self._buffer_start = start
        self._buffer = relative_float64_offsets(raw_values, self.observation_start)
        self._position = 0
        self._next_read_index = stop
        return True

    def _advance_to(self, target: float) -> bool:
        """Advance to the first timestamp offset not smaller than ``target``."""

        while True:
            if self._position >= self._buffer.size and not self._load_next_chunk():
                return False

            relative_position = int(
                np.searchsorted(self._buffer[self._position :], target, side="left")
            )
            self._position += relative_position

            if self._position < self._buffer.size:
                return True

    def read_interval(
        self,
        start: float,
        stop: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Read canonical offsets and stable indices in ``[start, stop)``."""

        if stop < start:
            raise ValueError("Timestamp cursor interval stop cannot precede its start.")

        if self._last_interval_stop is not None and start < self._last_interval_stop:
            self._reset()

        self._last_interval_stop = stop

        if not self._advance_to(start):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

        offset_pieces: list[NDArray[np.float64]] = []
        index_pieces: list[NDArray[np.int64]] = []

        while True:
            local_stop = int(np.searchsorted(self._buffer, stop, side="left"))

            if local_stop > self._position:
                offset_pieces.append(self._buffer[self._position : local_stop])
                index_pieces.append(
                    np.arange(
                        self._buffer_start + self._position,
                        self._buffer_start + local_stop,
                        dtype=np.int64,
                    )
                )

            self._position = local_stop

            if self._position < self._buffer.size:
                break

            if not self._load_next_chunk():
                break

        if not offset_pieces:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

        return np.concatenate(offset_pieces), np.concatenate(index_pieces)


def prepare_timestamp_batch(cursor: TimestampCursor, batch: WindowBatch) -> PreparedTimestampBatch:
    """Assign sequential source events to the batch's half-open windows."""

    starts = np.asarray(batch.relative_starts, dtype=np.float64)
    estimate_count, windows_per_estimate = starts.shape
    flat_starts = starts.reshape(-1)

    observation_offsets, global_indices = cursor.read_interval(
        float(flat_starts[0]),
        float(flat_starts[-1] + batch.duration),
    )

    relative_pieces: list[NDArray[np.float64]] = []
    window_pieces: list[NDArray[np.int64]] = []
    global_pieces: list[NDArray[np.int64]] = []

    for window_index, window_start in enumerate(flat_starts):
        window_stop = window_start + batch.duration
        local_start = int(np.searchsorted(observation_offsets, window_start, side="left"))
        local_stop = int(np.searchsorted(observation_offsets, window_stop, side="left"))
        event_count = local_stop - local_start

        if event_count == 0:
            continue

        relative_pieces.append(observation_offsets[local_start:local_stop] - window_start)
        window_pieces.append(np.full(event_count, window_index, dtype=np.int64))
        global_pieces.append(global_indices[local_start:local_stop])

    if relative_pieces:
        relative_times = np.concatenate(relative_pieces)
        window_indices = np.concatenate(window_pieces)
        selected_global_indices = np.concatenate(global_pieces)
    else:
        relative_times = np.empty(0, dtype=np.float64)
        window_indices = np.empty(0, dtype=np.int64)
        selected_global_indices = np.empty(0, dtype=np.int64)

    return PreparedTimestampBatch(
        relative_event_times=relative_times,
        window_indices=window_indices,
        global_event_indices=selected_global_indices,
        estimate_count=estimate_count,
        windows_per_estimate=windows_per_estimate,
    )
