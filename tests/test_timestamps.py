import h5py
import numpy as np

from signalsnap_pytorch import DataConfig, HDF5Source, TimestampedChannel
from signalsnap_pytorch._core import timestamps as _timestamps
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.planning import WindowBatch
from signalsnap_pytorch._core.timestamps import (
    TimestampCursor,
    prepare_timestamp_batch,
)


def _batch(starts, duration=1.0, shifted=False):
    starts_array = np.asarray(starts, dtype=np.float64)
    return WindowBatch(
        relative_starts=starts_array,
        duration=duration,
        estimate_count=starts_array.shape[0],
        shifted=shifted,
    )


def _assert_prepared_equal(actual, expected):
    np.testing.assert_allclose(
        actual.relative_event_times,
        expected.relative_event_times,
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(actual.window_indices, expected.window_indices)
    np.testing.assert_array_equal(
        actual.global_event_indices,
        expected.global_event_indices,
    )
    assert actual.estimate_count == expected.estimate_count
    assert actual.windows_per_estimate == expected.windows_per_estimate


def test_timestamp_batch_assigns_boundaries_duplicates_and_empty_windows(monkeypatch):
    monkeypatch.setattr(_timestamps, "_TIMESTAMP_READ_CHUNK_SIZE", 3)
    timestamps = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.999, 3.0, 3.5])
    cursor = TimestampCursor(timestamps, observation_start=0.0)

    prepared = prepare_timestamp_batch(cursor, _batch([[0.0, 1.0], [2.0, 3.0]]))

    np.testing.assert_allclose(
        prepared.relative_event_times,
        [0.0, 0.5, 0.0, 0.0, 0.5, 0.999, 0.0, 0.5],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(prepared.window_indices, [0, 0, 1, 1, 1, 2, 3, 3])
    np.testing.assert_array_equal(prepared.global_event_indices, np.arange(8))
    assert prepared.estimate_count == 2
    assert prepared.windows_per_estimate == 2


def test_timestamp_cursor_retains_lookahead_across_consecutive_batches(monkeypatch):
    monkeypatch.setattr(_timestamps, "_TIMESTAMP_READ_CHUNK_SIZE", 3)
    timestamps = np.arange(8, dtype=np.float64) + 0.25
    cursor = TimestampCursor(timestamps, observation_start=0.0)

    first = prepare_timestamp_batch(cursor, _batch([[0.0, 1.0]]))
    second = prepare_timestamp_batch(cursor, _batch([[2.0, 3.0]]))

    np.testing.assert_array_equal(first.global_event_indices, [0, 1])
    np.testing.assert_array_equal(second.global_event_indices, [2, 3])
    np.testing.assert_allclose(first.relative_event_times, [0.25, 0.25])
    np.testing.assert_allclose(second.relative_event_times, [0.25, 0.25])


def test_timestamp_cursor_resets_for_interlaced_traversal(monkeypatch):
    monkeypatch.setattr(_timestamps, "_TIMESTAMP_READ_CHUNK_SIZE", 3)
    timestamps = np.array([0.75, 1.25, 2.75, 3.25])
    cursor = TimestampCursor(timestamps, observation_start=0.0)

    prepare_timestamp_batch(cursor, _batch([[0.0, 1.0], [2.0, 3.0]]))
    shifted = prepare_timestamp_batch(
        cursor,
        _batch([[0.5, 1.5], [2.5, 3.5]], shifted=True),
    )

    np.testing.assert_allclose(shifted.relative_event_times, [0.25, 0.75, 0.25, 0.75])
    np.testing.assert_array_equal(shifted.window_indices, [0, 0, 2, 2])
    np.testing.assert_array_equal(shifted.global_event_indices, [0, 1, 2, 3])


def test_timestamp_batch_preserves_large_integer_origin_offsets():
    origin = 2**60 + 1
    timestamps = np.array([origin, origin + 1, origin + 3], dtype=np.int64)
    cursor = TimestampCursor(timestamps, observation_start=origin)

    prepared = prepare_timestamp_batch(cursor, _batch([[0.0, 1.0], [2.0, 3.0]]))

    np.testing.assert_array_equal(prepared.relative_event_times, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(prepared.window_indices, [0, 1, 3])
    np.testing.assert_array_equal(prepared.global_event_indices, [0, 1, 2])


def test_timestamp_batch_array_and_two_dimensional_hdf5_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(_timestamps, "_TIMESTAMP_READ_CHUNK_SIZE", 3)
    timestamps = np.array([[0.0, 0.5, 1.0], [1.0, 2.25, 3.75]])
    path = tmp_path / "timestamps.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("/timestamps", data=timestamps)

    config = DataConfig(
        channels=(
            TimestampedChannel(
                timestamps=HDF5Source(
                    file=path,
                    dataset="/timestamps",
                    selection=(slice(None), slice(None)),
                ),
            ),
        ),
    )
    batch = _batch([[0.0, 1.0], [2.0, 3.0]])
    expected = prepare_timestamp_batch(
        TimestampCursor(timestamps.reshape(-1), observation_start=0.0),
        batch,
    )

    with open_channels(config, (0,)) as sources:
        actual = prepare_timestamp_batch(
            TimestampCursor(sources[0], observation_start=0.0),
            batch,
        )

    _assert_prepared_equal(actual, expected)
