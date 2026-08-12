# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .data_access import RuntimeSource, get_source_length, read_source, relative_float64_offsets
from .fft import TimestampWindow
from .planning import FrequencyPlan, RuntimeConfig, TimestampedChannelPlan, WindowBatch
from .spectra import ChannelCoefficients, ThirdOrderCoefficients, TimestampThirdOrderFrequencyCache

# Sequential reads of about 512 KiB for float64 timestamps.
_TIMESTAMP_READ_CHUNK_SIZE = 65_536

_MAX_DIRECT_PHASE_ELEMENTS = 1_048_576
_MAX_DIRECT_FREQUENCIES_PER_CHUNK = 256
_PHILOX_OUTPUTS_PER_COUNTER = 4
_OPEN_UNIT_FLOAT64_SCALE = 2.0**-52


@dataclass(frozen=True, slots=True)
class PreparedTimestampBatch:
    """Timestamp events assigned to one physical ``(B, m)`` window batch.

    Attributes
    ----------
    relative_event_times : NDArray[np.float64]
        Event times relative to their physical-window starts, with shape ``(E,)``. Values lie in the
        half-open interval ``[0, window_duration)``.
    window_indices : NDArray[np.int64]
        Row-major flattened physical-window indices with shape ``(E,)``. Values identify windows in
        ``range(B * m)``.
    global_event_indices : NDArray[np.int64]
        Stable indices into the original flattened timestamp source, with shape ``(E,)``. These
        identities are used to generate batching-independent random amplitudes.
    estimate_count : int
        Number of spectral estimates in the batch, equal to ``B``.
    windows_per_estimate : int
        Number of physical windows in each estimate, equal to ``m``.
    """

    relative_event_times: NDArray[np.float64]
    window_indices: NDArray[np.int64]
    global_event_indices: NDArray[np.int64]
    estimate_count: int
    windows_per_estimate: int


def _generate_keyed_exponential_amplitudes(
    prepared: PreparedTimestampBatch,
    realization_ids: range,
    *,
    resolved_seed: int,
    channel_index: int,
    scale: float,
) -> NDArray[np.float64]:
    """Generate CPU float64 exponential amplitudes keyed by event identity.

    NumPy's counter-based Philox 4x64 generator supplies the integer stream.
    The Philox key is derived from the calculation seed and channel index.
    Counter words identify the global event block and realization.

    The returned array has shape ``(R, E)``. Its values therefore do not
    depend on physical batching, repetition batching, or traversal order.
    """

    event_indices = prepared.global_event_indices
    realizations = np.asarray(tuple(realization_ids), dtype=np.int64)

    if realizations.size == 0:
        raise ValueError("At least one realization ID is required.")

    amplitudes = np.empty((realizations.size, event_indices.size), dtype=np.float64)

    if event_indices.size == 0:
        return amplitudes

    first_event = int(event_indices[0])
    last_event = int(event_indices[-1])
    expected_indices = np.arange(first_event, last_event + 1, dtype=np.int64)

    if not np.array_equal(event_indices, expected_indices):
        raise RuntimeError("Prepared timestamp events must retain contiguous global indices.")

    first_counter = first_event // _PHILOX_OUTPUTS_PER_COUNTER
    final_counter = last_event // _PHILOX_OUTPUTS_PER_COUNTER
    raw_count = (final_counter - first_counter + 1) * _PHILOX_OUTPUTS_PER_COUNTER
    event_offsets = event_indices - (first_counter * _PHILOX_OUTPUTS_PER_COUNTER)

    key = np.random.SeedSequence([resolved_seed, channel_index]).generate_state(2, dtype=np.uint64)

    for row, realization_id in enumerate(realizations):
        counter = np.array([first_counter, int(realization_id), 0, 0], dtype=np.uint64)
        bit_generator = np.random.Philox(counter=counter, key=key)
        raw_values = bit_generator.random_raw(raw_count)
        selected_values = raw_values[event_offsets]

        uniform_survival = (
            (selected_values >> np.uint64(12)).astype(np.float64) + 0.5
        ) * _OPEN_UNIT_FLOAT64_SCALE
        amplitudes[row] = -scale * np.log(uniform_survival)

    return amplitudes


class TimestampCursor:
    """Sequential bounded reader for one validated timestamp source.

    The cursor reads a source in bounded chunks, rebases timestamps relative to the observation
    start, and returns stable indices into the original flattened source. Forward, nonoverlapping
    interval reads reuse the current source position. If a requested interval starts before the
    preceding interval ended, the cursor resets; this supports the second traversal introduced by
    interlacing.

    Parameters
    ----------
    source : RuntimeSource
        Validated, nondecreasing timestamp source.
    observation_start : float
        Origin subtracted from timestamps to obtain observation-relative offsets.
    """

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
        """Read timestamp offsets and source indices from a half-open interval.

        ``start`` and ``stop`` are measured relative to the cursor's observation origin. Forward,
        nonoverlapping calls continue from the current source position. If ``start`` precedes the
        previous interval's stop, the cursor resets and traverses the source again.

        Parameters
        ----------
        start, stop : float
            Bounds of the observation-relative half-open interval ``[start, stop)``.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.int64]]
            Observation-relative timestamp offsets and their corresponding indices in the original
            flattened source. Both arrays have shape ``(E,)`` and preserve source order. Empty
            arrays are returned when the interval contains no events.

        Raises
        ------
        ValueError
            If ``stop`` is less than ``start``.
        """

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
    """Assign timestamp events to a batch of half-open physical windows.

    The standard batches produced by :func:`~signalsnap_pytorch._core.planning.iter_window_batches`
    contain consecutive, nonoverlapping windows. Events at a window start are included in that
    window; events at its stop are assigned to the following window.

    Parameters
    ----------
    cursor : TimestampCursor
        Cursor for a validated timestamp source.
    batch : WindowBatch
        Physical windows whose relative starts have shape ``(B, m)``.

    Returns
    -------
    PreparedTimestampBatch
        Flattened event data for the batch. ``relative_event_times`` has shape ``(E,)`` and contains
        times relative to each event's physical-window start. ``window_indices`` has shape ``(E,)``
        and identifies the row-major flattened window in ``range(B * m)``. ``global_event_indices``
        has shape ``(E,)`` and preserves each event's index in the source. ``estimate_count`` and
        ``windows_per_estimate`` retain ``B`` and ``m``.
    """

    starts = np.asarray(batch.relative_starts, dtype=np.float64)
    estimate_count, windows_per_estimate = starts.shape
    flat_starts = starts.reshape(-1)

    boundaries = np.empty(flat_starts.size + 1, dtype=np.float64)
    boundaries[:-1] = flat_starts
    boundaries[-1] = (
        batch.relative_stop if batch.relative_stop is not None else flat_starts[-1] + batch.duration
    )

    if np.any(boundaries[1:] <= boundaries[:-1]):
        raise RuntimeError("Timestamp window boundaries must be strictly increasing.")

    observation_offsets, global_indices = cursor.read_interval(
        float(boundaries[0]),
        float(boundaries[-1]),
    )

    window_indices = (np.searchsorted(boundaries, observation_offsets, side="right") - 1).astype(
        np.int64, copy=False
    )

    relative_times = observation_offsets - flat_starts[window_indices]

    return PreparedTimestampBatch(
        relative_event_times=relative_times,
        window_indices=window_indices,
        global_event_indices=global_indices,
        estimate_count=estimate_count,
        windows_per_estimate=windows_per_estimate,
    )


def _direct_timestamp_transform(
    prepared: PreparedTimestampBatch,
    frequencies: NDArray[np.floating[Any]],
    event_weights: Tensor,
    runtime: RuntimeConfig,
) -> Tensor:
    """Direct-transform prepared events using the positive-exponential convention.

    Parameters
    ----------
    prepared
        Window-relative event times and flattened ``(B, m)`` window indices.
    frequencies
        Frequencies to evaluate, with shape ``(F,)``.
    event_weights
        Window and amplitude weights with shape ``(R, E)``.
    runtime
        Calculation device and numeric dtypes.

    Returns
    -------
    Tensor
        Complex coefficients with shape ``(R, B, m, F)``.
    """

    if event_weights.ndim != 2:
        raise ValueError("Event weights must have shape (R, E).")

    realization_count, event_count = event_weights.shape
    if realization_count == 0:
        raise ValueError("At least one amplitude realization is required.")

    if event_count != prepared.relative_event_times.size:
        raise ValueError(
            f"Event weights contain {event_count} events; "
            f"the prepared batch contains {prepared.relative_event_times.size}."
        )

    frequency_values = torch.as_tensor(
        frequencies,
        dtype=runtime.real_dtype,
        device=runtime.device,
    )
    if frequency_values.ndim != 1:
        raise ValueError("Frequencies must be one-dimensional.")

    weights = event_weights.to(dtype=runtime.real_dtype, device=runtime.device)
    relative_times = torch.as_tensor(
        prepared.relative_event_times,
        dtype=runtime.real_dtype,
        device=runtime.device,
    )
    window_indices = torch.as_tensor(
        prepared.window_indices,
        dtype=torch.long,
        device=runtime.device,
    )

    window_count = prepared.estimate_count * prepared.windows_per_estimate
    frequency_count = frequency_values.numel()
    result = torch.zeros(
        realization_count,
        window_count,
        frequency_count,
        dtype=runtime.complex_dtype,
        device=runtime.device,
    )

    if event_count == 0 or frequency_count == 0:
        return result.reshape(
            realization_count,
            prepared.estimate_count,
            prepared.windows_per_estimate,
            frequency_count,
        )

    frequency_chunk_size = min(frequency_count, _MAX_DIRECT_FREQUENCIES_PER_CHUNK)

    for frequency_start in range(0, frequency_count, frequency_chunk_size):
        frequency_stop = min(frequency_start + frequency_chunk_size, frequency_count)
        frequency_chunk = frequency_values[frequency_start:frequency_stop]
        event_chunk_size = max(
            1,
            _MAX_DIRECT_PHASE_ELEMENTS // (realization_count * frequency_chunk.numel()),
        )
        frequency_result = result.new_zeros(
            realization_count,
            window_count,
            frequency_chunk.numel(),
        )

        for event_start in range(0, event_count, event_chunk_size):
            event_stop = min(event_start + event_chunk_size, event_count)
            time_chunk = relative_times[event_start:event_stop]
            angles = 2.0 * torch.pi * time_chunk[:, None] * frequency_chunk[None, :]
            phases = torch.polar(torch.ones_like(angles), angles)
            contributions = weights[:, event_start:event_stop, None] * phases[None, :, :]
            frequency_result.index_add_(1, window_indices[event_start:event_stop], contributions)

        result[:, :, frequency_start:frequency_stop] = frequency_result

    return result.reshape(
        realization_count,
        prepared.estimate_count,
        prepared.windows_per_estimate,
        frequency_count,
    )


def materialize_timestamp_coefficients(
    prepared: PreparedTimestampBatch,
    frequency_plan: FrequencyPlan,
    timestamp_window: TimestampWindow,
    runtime: RuntimeConfig,
    third_order_cache: TimestampThirdOrderFrequencyCache | None,
    event_amplitudes: NDArray[np.float64],
    *,
    needs_output: bool = True,
) -> ChannelCoefficients:
    """Apply event amplitudes and directly transform one prepared timestamp batch.

    Window weights are evaluated at the prepared relative event times and multiplied by the supplied
    amplitudes. Zero-frequency coefficients are always calculated. Output-band and compact
    third-order closing-frequency coefficients are calculated only when requested.

    Parameters
    ----------
    prepared : PreparedTimestampBatch
        Events assigned to a physical ``(B, m)`` window batch.
    frequency_plan : FFTFrequencyPlan | DirectFrequencyPlan
        Frequency grid used for output-band coefficients.
    timestamp_window : TimestampWindow
        Timestamp window used for event weighting.
    runtime : RuntimeConfig
        Calculation device and numeric dtypes.
    third_order_cache : TimestampThirdOrderFrequencyCache | None
        Compact closing frequencies and output-grid mapping, if required.
    event_amplitudes : NDArray[np.float64]
        CPU amplitude matrix with shape ``(R, E)``, where ``R`` is the realization
        count and ``E`` is the number of prepared events.
    needs_output : bool, default=True
        Whether to calculate coefficients on the output-frequency band.

    Returns
    -------
    ChannelCoefficients
        ``dc`` has shape ``(R, B, m)``. ``output`` has shape ``(R, B, m, F)`` when requested and is
        otherwise ``None``. When a third-order cache is supplied, ``third_order.values`` has shape
        ``(R, B, m, K)``.

    Raises
    ------
    ValueError
        If the amplitude matrix is not two-dimensional, contains no realizations, or has a different
        event count from ``prepared``.
    """

    event_count = prepared.global_event_indices.size

    if event_amplitudes.ndim != 2:
        raise ValueError("Event amplitudes must have shape (R, E).")

    if event_amplitudes.shape[0] == 0:
        raise ValueError("At least one amplitude realization is required.")

    if event_amplitudes.shape[1] != event_count:
        raise ValueError(
            f"Event amplitudes contain {event_amplitudes.shape[1]} events; "
            f"the prepared batch contains {event_count}."
        )

    relative_times = torch.as_tensor(
        prepared.relative_event_times,
        dtype=runtime.real_dtype,
        device=runtime.device,
    )
    amplitudes = torch.as_tensor(
        event_amplitudes,
        dtype=runtime.real_dtype,
        device=runtime.device,
    )
    window_weights = timestamp_window.evaluate(relative_times)
    event_weights = amplitudes * window_weights.unsqueeze(0)

    dc = _direct_timestamp_transform(
        prepared,
        frequencies=np.zeros(1, dtype=np.float64),
        event_weights=event_weights,
        runtime=runtime,
    )[..., 0]

    output = None
    if needs_output:
        output = _direct_timestamp_transform(
            prepared,
            frequencies=frequency_plan.band_frequencies,
            event_weights=event_weights,
            runtime=runtime,
        )

    third_order = None
    if third_order_cache is not None:
        closing_values = _direct_timestamp_transform(
            prepared,
            frequencies=third_order_cache.closing_frequencies,
            event_weights=event_weights,
            runtime=runtime,
        )
        third_order = ThirdOrderCoefficients(
            values=closing_values,
            gather_indices=third_order_cache.gather_indices,
            valid_mask=third_order_cache.valid_mask,
        )

    return ChannelCoefficients(dc=dc, output=output, third_order=third_order)


def materialize_timestamp_event_amplitudes(
    prepared: PreparedTimestampBatch,
    channel_index: int,
    channel_plan: TimestampedChannelPlan,
    realization_ids: range,
    runtime: RuntimeConfig,
) -> NDArray[np.float64]:
    """Materialize reusable amplitudes for one timestamp realization batch.

    Unit weighting assigns one to every realization-event pair. Exponential weighting generates CPU
    float64 amplitudes keyed by the resolved seed, channel index, realization ID, and stable global
    event index. The keyed construction makes exponential amplitudes independent of physical-window
    batching, repetition batching, and traversal order.

    Parameters
    ----------
    prepared : PreparedTimestampBatch
        Prepared events and their stable source indices.
    channel_index : int
        Channel identity used to derive an independent random stream.
    channel_plan : TimestampedChannelPlan
        Resolved weighting model and exponential scale.
    realization_ids : range
        Stable calculation-wide IDs for the current realization batch.
    runtime : RuntimeConfig
        Runtime repetition plan containing the resolved random seed.

    Returns
    -------
    NDArray[np.float64]
        CPU amplitude matrix with shape ``(R, E)``, where ``R`` is the number of realization IDs and
        ``E`` is the number of prepared events.

    Raises
    ------
    ValueError
        If exponential weighting receives no realization IDs.
    RuntimeError
        If exponential weighting lacks a scale or resolved seed, or if the weighting model is
        unsupported.
    """

    if channel_plan.weighting == "unit":
        return np.ones(
            (len(realization_ids), prepared.global_event_indices.size),
            dtype=np.float64,
        )

    if channel_plan.weighting == "exponential":
        resolved_seed = runtime.repetition_plan.resolved_seed

        if channel_plan.scale is None or resolved_seed is None:
            raise RuntimeError(
                "Exponential timestamp weighting requires a scale and resolved seed."
            )

        return _generate_keyed_exponential_amplitudes(
            prepared,
            realization_ids,
            resolved_seed=resolved_seed,
            channel_index=channel_index,
            scale=channel_plan.scale,
        )

    raise RuntimeError(f"Unknown timestamp weighting model {channel_plan.weighting!r}.")
