# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Generator, Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..config import DataConfig, HDF5Source, SampledChannel, TimestampedChannel

if TYPE_CHECKING:
    import h5py

NormalizedSelector = int | slice
_VALIDATION_CHUNK_SIZE = 65_536


def _require_h5py() -> Any:
    """Import and return ``h5py`` or raise an installation-oriented error."""
    try:
        import h5py
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise

        raise ModuleNotFoundError(
            "HDF5 support requires the optional dependency 'h5py'. "
            'Install it with: pip install "signalsnap-pytorch[hdf5]"'
        ) from None

    return h5py


@dataclass(frozen=True, slots=True)
class HDF5SourceState:
    """Opened runtime representation of a user-specified :class:`HDF5Source`.

    Attributes
    ----------
    dataset : h5py.Dataset
        Open HDF5 dataset. Its owning file must remain open while this state is used.
    selection : tuple[int | slice, ...]
        Normalized dataset selection with explicit nonnegative indices and slice bounds.
    selected_shape : tuple[int, ...]
        Shape of the one- or two-dimensional logical selection before C-order flattening.
    """

    dataset: h5py.Dataset
    selection: tuple[NormalizedSelector, ...]
    selected_shape: tuple[int, ...]

    @property
    def length(self) -> int:
        """Length of the flattened logical source."""
        return math.prod(self.selected_shape)

    def read(self, start: int, stop: int) -> np.ndarray:
        """Read a half-open range from the C-order-flattened logical source.

        Parameters
        ----------
        start, stop : int
            Validated zero-based bounds in flattened-source coordinates.

        Returns
        -------
        np.ndarray
            One-dimensional array containing exactly ``stop - start`` values. The public
            :func:`read_source` helper normalizes byte order and contiguity.

        Raises
        ------
        RuntimeError
            If the normalized selection has an unsupported number of logical dimensions.
        """
        if start == stop:
            return np.empty(0, dtype=self.dataset.dtype)

        if len(self.selected_shape) == 1:
            return self._read_1d(start, stop)

        if len(self.selected_shape) == 2:
            return self._read_2d(start, stop)

        raise RuntimeError(
            f"Unsupported selected shape {self.selected_shape}. "
            "At most two non-fixed dimensions are supported."
        )

    def _build_dataset_index(
        self, logical_selectors: tuple[int | slice, ...]
    ) -> tuple[int | slice, ...]:
        """Translate logical-source selectors back to selectors for the source dataset."""
        if len(logical_selectors) != len(self.selected_shape):
            raise ValueError(
                f"Expected {len(self.selected_shape)} logical selectors, "
                f"received {len(logical_selectors)}."
            )

        dataset_index: list[int | slice] = []
        logical_axis = 0

        for base_selector in self.selection:
            if isinstance(base_selector, int):
                dataset_index.append(base_selector)
                continue

            logical_selector = logical_selectors[logical_axis]
            logical_axis += 1

            # Slices were normalized earlier, so start is always an integer.
            base_start = base_selector.start
            assert base_start is not None

            if isinstance(logical_selector, int):
                dataset_index.append(base_start + logical_selector)
            else:
                logical_start = logical_selector.start
                logical_stop = logical_selector.stop

                if logical_start is None or logical_stop is None:
                    raise ValueError("Internal logical slices must have explicit bounds.")

                dataset_index.append(slice(base_start + logical_start, base_start + logical_stop))

        return tuple(dataset_index)

    def _allocate_output(self, size: int) -> np.ndarray:
        """Let HDF5 convert byte order while reading, avoiding a later full copy."""
        dtype = self.dataset.dtype.newbyteorder("=")
        return np.empty(size, dtype=dtype)

    def _read_direct(
        self, destination: np.ndarray, logical_selectors: tuple[int | slice, ...]
    ) -> None:
        """Read logical selectors directly into a preallocated NumPy destination."""
        dataset_index = self._build_dataset_index(logical_selectors)
        self.dataset.read_direct(destination, source_sel=dataset_index)

    def _read_1d(self, start: int, stop: int) -> np.ndarray:
        """Read a flattened range from a one-dimensional logical selection."""
        result = self._allocate_output(stop - start)
        self._read_direct(result, (slice(start, stop),))
        return result

    def _read_2d(self, start: int, stop: int) -> np.ndarray:
        """Read a flattened range from a two-dimensional logical selection in C-order."""
        _, column_count = self.selected_shape
        result = self._allocate_output(stop - start)

        cursor = start
        output_cursor = 0

        # Partial first row.
        start_row, start_column = divmod(cursor, column_count)

        if start_column:
            count = min(stop - cursor, column_count - start_column)

            self._read_direct(
                result[output_cursor : output_cursor + count],
                (start_row, slice(start_column, start_column + count)),
            )

            cursor += count
            output_cursor += count

        # Complete middle rows.
        complete_rows = (stop - cursor) // column_count

        if complete_rows:
            first_row = cursor // column_count
            count = complete_rows * column_count

            destination = result[output_cursor : output_cursor + count].reshape(
                complete_rows, column_count
            )

            self._read_direct(
                destination,
                (
                    slice(first_row, first_row + complete_rows),
                    slice(0, column_count),
                ),
            )

            cursor += count
            output_cursor += count

        # Partial final row.
        if cursor < stop:
            final_row = cursor // column_count
            count = stop - cursor

            self._read_direct(
                result[output_cursor : output_cursor + count],
                (final_row, slice(0, count)),
            )

            cursor += count
            output_cursor += count

        if cursor != stop or output_cursor != result.size:
            raise RuntimeError(
                f"HDF5 reading stopped at sample {cursor}; expected to stop at {stop}."
            )

        return result


def _normalize_hdf5_selection(
    dataset: h5py.Dataset, source: HDF5Source
) -> tuple[tuple[NormalizedSelector, ...], tuple[int, ...]]:
    """Normalize and validate an HDF5 source selection against its dataset.

    Negative integer indices and open slice bounds are resolved against the dataset shape. The
    selection must leave one or two logical dimensions, which form the logical source before
    C-order flattening. A logical source may be empty.

    Parameters
    ----------
    dataset : h5py.Dataset
        Dataset selected by ``source.dataset``.
    source : HDF5Source
        User selection to normalize.

    Returns
    -------
    tuple[tuple[int | slice, ...], tuple[int, ...]]
        Normalized dataset selectors and the remaining logical shape.

    Raises
    ------
    ValueError
        If the rank differs, the selection is scalar, a slice step is unsupported, or more
        than two dimensions remain unfixed.
    IndexError
        If an integer selector is outside its dataset dimension.
    TypeError
        If a selector is neither an integer nor a slice.
    """

    if len(source.selection) != dataset.ndim:
        raise ValueError(
            f"Selection for dataset {source.dataset!r} contains {len(source.selection)} entries, "
            f"but the dataset has {dataset.ndim} dimensions."
        )

    normalized = []
    selected_shape = []

    for axis, (dimension_size, selector) in enumerate(zip(dataset.shape, source.selection)):
        if isinstance(selector, int) and not isinstance(selector, bool):
            index = selector

            if index < 0:
                index += dimension_size

            if index < 0 or index >= dimension_size:
                raise IndexError(
                    f"Index {selector} is out of bounds for axis {axis} with size {dimension_size}."
                )

            normalized.append(index)
            continue

        elif isinstance(selector, slice):
            start, stop, step = selector.indices(dimension_size)
            if step != 1:
                raise ValueError("Only HDF5 slices with step 1 are supported.")

            normalized.append(slice(start, stop))
            selected_shape.append(max(0, stop - start))
        else:
            raise TypeError(
                f"Selection entry for axis {axis} must be an integer or slice; "
                f"received {type(selector).__name__}."
            )

    if not selected_shape:
        raise ValueError("The HDF5 selection selects a scalar, not a logical source.")

    if len(selected_shape) > 2:
        raise ValueError(
            "An HDF5 source may have at most two non-fixed dimensions. "
            "Use integer indices to fix additional dimensions."
        )

    return tuple(normalized), tuple(selected_shape)


def _validate_hdf5_dataset(file: h5py.File, source: HDF5Source) -> h5py.Dataset:
    """Return the selected dataset after validating its path, object type, and dtype."""
    h5py = _require_h5py()

    if source.dataset not in file:
        raise KeyError(
            f"Dataset {source.dataset!r} does not exist in HDF5 file {str(source.file)!r}."
        )

    dataset = file[source.dataset]

    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"HDF5 path {source.dataset!r} is not a dataset.")

    dataset = cast("h5py.Dataset", dataset)

    if np.issubdtype(dataset.dtype, np.complexfloating):
        raise TypeError("Complex HDF5 datasets are not supported.")

    if not np.issubdtype(dataset.dtype, np.number) and not np.issubdtype(dataset.dtype, np.bool_):
        raise TypeError(f"HDF5 dataset dtype {dataset.dtype} is not numeric.")

    return dataset


RuntimeSource = np.ndarray | torch.Tensor | HDF5SourceState


@contextmanager
def open_channels(
    data_config: DataConfig,
    channel_indices: Iterable[int],
) -> Generator[dict[int, RuntimeSource], None, None]:
    """Open selected channel sources and close shared HDF5 files on context exit.

    In-memory sources are returned unchanged. HDF5 sources that refer to the same resolved file
    path share one read-only file handle.

    Parameters
    ----------
    data_config : DataConfig
        User channel source definitions.
    channel_indices : Iterable[int]
        Validated indices of channels needed by the caller.

    Yields
    ------
    dict[int, RuntimeSource]
        Mapping from requested indices to in-memory sources or opened :class:`HDF5SourceState`
        objects.

    Raises
    ------
    ModuleNotFoundError
        If an HDF5 source is requested without the optional ``h5py`` dependency.
    OSError
        If an HDF5 file cannot be opened.
    KeyError
        If a configured dataset path does not exist.
    TypeError
        If the selected HDF5 object is not a supported real numeric or Boolean dataset.
    ValueError
        If an HDF5 selection is invalid for its dataset.
    """
    with ExitStack() as stack:
        files: dict[Path, h5py.File] = {}
        opened_channels: dict[int, RuntimeSource] = {}

        for channel_index in channel_indices:
            channel = data_config.channels[channel_index]

            if isinstance(channel, SampledChannel):
                source = channel.data
            elif isinstance(channel, TimestampedChannel):
                source = channel.timestamps
            else:
                raise TypeError(
                    f"Channel {channel_index} has unsupported type {type(channel).__name__}."
                )

            if not isinstance(source, HDF5Source):
                opened_channels[channel_index] = source
                continue

            h5py = _require_h5py()
            path = source.file.expanduser().resolve()

            if path not in files:
                files[path] = stack.enter_context(h5py.File(path, mode="r"))

            dataset = _validate_hdf5_dataset(files[path], source)
            selection, selected_shape = _normalize_hdf5_selection(dataset, source)

            opened_channels[channel_index] = HDF5SourceState(
                dataset=dataset,
                selection=selection,
                selected_shape=selected_shape,
            )

        yield opened_channels


def get_source_length(source: RuntimeSource) -> int:
    """Return the flattened length of an in-memory or opened HDF5 source."""
    if isinstance(source, HDF5SourceState):
        return source.length

    return int(source.shape[0])


def _validate_source_read_range(source: RuntimeSource, start: int, stop: int) -> tuple[int, int]:
    """Validate and normalize a half-open runtime-source read range.

    Parameters
    ----------
    source : RuntimeSource
        Source whose read range is to be validated.
    start, stop : int
        Half-open value bounds. NumPy integers are accepted; Booleans are rejected.

    Returns
    -------
    tuple[int, int]
        Bounds normalized to built-in integers.

    Raises
    ------
    TypeError
        If either bound is not an integer or the source returns a NumPy masked array.
    ValueError
        If the range is negative, reversed, or extends beyond the source.
    """
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise TypeError("start must be an integer.")

    if isinstance(stop, bool) or not isinstance(stop, (int, np.integer)):
        raise TypeError("stop must be an integer.")

    start = int(start)
    stop = int(stop)

    if start < 0:
        raise ValueError("start cannot be negative.")

    if stop < start:
        raise ValueError("stop cannot be smaller than start.")

    source_length = get_source_length(source)

    if stop > source_length:
        raise ValueError(
            f"Cannot read until position {stop}; the source contains {source_length} values."
        )

    return start, stop


def read_source(source: RuntimeSource, start: int, stop: int) -> np.ndarray:
    """Read a contiguous one-dimensional range from a runtime source.

    The returned array has native byte order and C-contiguous storage, copying only when required.

    Parameters
    ----------
    source : RuntimeSource
        In-memory source or opened HDF5 state.
    start, stop : int
        Half-open value bounds.

    Returns
    -------
    np.ndarray
        One-dimensional array containing exactly ``stop - start`` values.

    Raises
    ------
    TypeError
        If either bound is not an integer.
    ValueError
        If the requested range is invalid.
    RuntimeError
        If the underlying read returns an unexpected shape or number of values.
    """
    start, stop = _validate_source_read_range(source, start, stop)

    if isinstance(source, HDF5SourceState):
        result = source.read(start, stop)
    elif isinstance(source, torch.Tensor):
        tensor = source[start:stop].detach()

        try:
            result = tensor.numpy()
        except (TypeError, RuntimeError):
            if not tensor.is_floating_point():
                raise
            result = tensor.to(torch.float64).numpy()
    else:
        result = source[start:stop]

    if np.ma.isMaskedArray(result):
        raise TypeError("Runtime sources must not return NumPy masked arrays.")

    result = np.asarray(result)

    if result.ndim != 1:
        raise RuntimeError(
            f"Reading returned shape {result.shape}; expected a one-dimensional array."
        )

    expected_size = stop - start

    if result.shape[0] != expected_size:
        raise RuntimeError(f"Reading returned {result.shape[0]} values; expected {expected_size}.")

    if not result.dtype.isnative:
        native_dtype = result.dtype.newbyteorder("=")
        result = result.astype(native_dtype, copy=False)

    return np.ascontiguousarray(result)


def relative_float64_offsets(
    values: np.ndarray,
    observation_start: float,
) -> np.ndarray:
    """Rebase raw timestamps before conversion to canonical float64 offsets."""

    if np.issubdtype(values.dtype, np.integer):
        integral_origin = isinstance(observation_start, int) or (
            isinstance(observation_start, float) and observation_start.is_integer()
        )

        if integral_origin:
            origin = int(observation_start)
            offsets = (int(value) - origin for value in values)
            return np.fromiter(offsets, dtype=np.float64, count=values.size)

    return np.asarray(values - observation_start, dtype=np.float64)


def validate_sampled_hdf5_source(source: RuntimeSource, *, label: str) -> None:
    """Validate one sampled HDF5 source using bounded contiguous reads."""

    source_length = get_source_length(source)

    for start in range(0, source_length, _VALIDATION_CHUNK_SIZE):
        stop = min(start + _VALIDATION_CHUNK_SIZE, source_length)
        values = read_source(source, start, stop)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain only finite values.")


def validate_timestamp_source(
    source: RuntimeSource,
    observation_start: float,
    observation_stop: float,
    *,
    label: str,
) -> None:
    """Validate one active timestamp source using bounded contiguous reads."""

    if isinstance(source, HDF5SourceState):
        is_boolean = np.issubdtype(source.dataset.dtype, np.bool_)
    elif isinstance(source, torch.Tensor):
        is_boolean = source.dtype == torch.bool
    else:
        is_boolean = np.issubdtype(source.dtype, np.bool_)

    if is_boolean:
        raise TypeError(f"{label} must contain real numeric timestamps, not Boolean data.")

    previous_raw: float | None = None
    previous_offset: float | None = None

    for start in range(0, get_source_length(source), _VALIDATION_CHUNK_SIZE):
        stop = min(start + _VALIDATION_CHUNK_SIZE, get_source_length(source))
        values = read_source(source, start, stop)

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must contain only finite timestamps.")

        if values.size > 1 and np.any(values[1:] < values[:-1]):
            raise ValueError(f"{label} must be nondecreasing in flattened C order.")

        offsets = relative_float64_offsets(values, observation_start)

        if not np.all(np.isfinite(offsets)):
            raise ValueError(
                f"{label} cannot be represented as finite float64 offsets. "
                "Use a nearby time origin or a more appropriate time unit."
            )

        first_raw = values[0].item()
        last_raw = values[-1].item()
        first_offset = float(offsets[0])
        last_offset = float(offsets[-1])

        if previous_raw is not None and first_raw < previous_raw:
            raise ValueError(f"{label} must be nondecreasing in flattened C order.")

        if (
            previous_raw is not None
            and first_raw != previous_raw
            and first_offset == previous_offset
        ):
            raise ValueError(
                f"{label} contains distinct timestamps that collapse to one float64 "
                "offset. Use a nearby time origin or a more appropriate time unit."
            )

        distinct = values[1:] != values[:-1]
        collapsed = offsets[1:] == offsets[:-1]

        if np.any(distinct & collapsed):
            raise ValueError(
                f"{label} contains distinct timestamps that collapse to one float64 "
                "offset. Use a nearby time origin or a more appropriate time unit."
            )

        if first_raw < observation_start or last_raw >= observation_stop:
            raise ValueError(
                f"{label} must lie within the half-open observation interval "
                f"[{observation_start}, {observation_stop})."
            )

        previous_raw = last_raw
        previous_offset = last_offset
