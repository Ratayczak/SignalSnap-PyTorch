from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from signalsnap_pytorch import DataConfig, HDF5Source, TimestampedChannel
from signalsnap_pytorch._core import timestamps as _timestamps
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.fft import prepare_default_timestamp_window
from signalsnap_pytorch._core.planning import TimestampFrequencyPlan, WindowBatch
from signalsnap_pytorch._core.spectra import build_timestamp_third_order_cache
from signalsnap_pytorch._core.timestamps import (
    PreparedTimestampBatch,
    TimestampCursor,
    direct_timestamp_transform,
    materialize_unit_timestamp_coefficients,
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


def _runtime(dtype=torch.float64, duration=1.0):
    return SimpleNamespace(
        real_dtype=dtype,
        complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        device=torch.device("cpu"),
        window_plan=SimpleNamespace(duration=duration),
    )


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


def test_direct_transform_uses_positive_sign_and_window_relative_phase():
    cursor = TimestampCursor(np.array([12.25]), observation_start=10.0)
    prepared = prepare_timestamp_batch(cursor, _batch([[2.0]], duration=1.0))

    actual = direct_timestamp_transform(
        prepared,
        frequencies=np.array([0.0, 1.0]),
        event_weights=torch.ones((1, 1), dtype=torch.float64),
        runtime=_runtime(),
    )

    expected = torch.tensor([[[[1.0, 1.0j]]]], dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)


def test_direct_transform_sums_duplicates_per_window_and_realization():
    relative_times = np.array([0.0, 0.25, 0.25, 0.5])
    window_indices = np.array([0, 0, 0, 1], dtype=np.int64)
    prepared = PreparedTimestampBatch(
        relative_event_times=relative_times,
        window_indices=window_indices,
        global_event_indices=np.arange(4, dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=2,
    )
    frequencies = np.array([-1.0, 0.0, 0.5])
    weights = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]])

    actual = direct_timestamp_transform(
        prepared,
        frequencies=frequencies,
        event_weights=torch.from_numpy(weights),
        runtime=_runtime(),
    )

    expected = np.zeros((2, 1, 2, 3), dtype=np.complex128)
    for realization in range(2):
        for event, (relative_time, window_index) in enumerate(
            zip(relative_times, window_indices)
        ):
            expected[realization, 0, window_index] += (
                weights[realization, event]
                * np.exp(1j * 2.0 * np.pi * frequencies * relative_time)
            )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-14, atol=1e-14)


def test_direct_transform_empty_events_preserve_shape_dtype_and_device():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.empty(0, dtype=np.float64),
        window_indices=np.empty(0, dtype=np.int64),
        global_event_indices=np.empty(0, dtype=np.int64),
        estimate_count=2,
        windows_per_estimate=3,
    )

    actual = direct_timestamp_transform(
        prepared,
        frequencies=np.array([-1.0, 0.0, 1.0]),
        event_weights=torch.empty((2, 0)),
        runtime=_runtime(torch.float32),
    )

    assert actual.shape == (2, 2, 3, 3)
    assert actual.dtype == torch.complex64
    assert actual.device == torch.device("cpu")
    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_direct_transform_is_invariant_to_event_and_frequency_chunking(monkeypatch):
    relative_times = np.linspace(0.0, 0.9, 10)
    prepared = PreparedTimestampBatch(
        relative_event_times=relative_times,
        window_indices=np.repeat(np.arange(5), 2),
        global_event_indices=np.arange(10, dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=5,
    )
    frequencies = np.linspace(-2.0, 2.0, 9)
    weights = torch.arange(1, 21, dtype=torch.float64).reshape(2, 10) / 10.0
    expected = direct_timestamp_transform(prepared, frequencies, weights, _runtime())

    monkeypatch.setattr(_timestamps, "_MAX_DIRECT_PHASE_ELEMENTS", 4)
    monkeypatch.setattr(_timestamps, "_MAX_DIRECT_FREQUENCIES_PER_CHUNK", 2)
    actual = direct_timestamp_transform(prepared, frequencies, weights, _runtime())

    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        pytest.param(torch.ones(2), "shape", id="missing-realization-axis"),
        pytest.param(torch.ones((0, 1)), "realization", id="empty-realization-axis"),
        pytest.param(torch.ones((1, 2)), "contains 1", id="event-count-mismatch"),
    ],
)
def test_direct_transform_rejects_invalid_event_weights(weights, message):
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.25]),
        window_indices=np.array([0], dtype=np.int64),
        global_event_indices=np.array([0], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=1,
    )

    with pytest.raises(ValueError, match=message):
        direct_timestamp_transform(
            prepared,
            frequencies=np.array([1.0]),
            event_weights=weights,
            runtime=_runtime(),
        )


def test_unit_timestamp_coefficients_apply_window_to_dc_output_and_closing_roles():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.0, 0.5]),
        window_indices=np.array([0, 0], dtype=np.int64),
        global_event_indices=np.array([0, 1], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=1,
    )
    grid_indices = np.arange(-1, 2, dtype=np.int64)
    frequency_plan = TimestampFrequencyPlan(
        actual_df=1.0,
        grid_indices=grid_indices,
        band_frequencies=grid_indices.astype(np.float64),
    )
    runtime = _runtime()
    timestamp_window = prepare_default_timestamp_window(runtime)
    third_order_cache = build_timestamp_third_order_cache(runtime, frequency_plan)

    coefficients = materialize_unit_timestamp_coefficients(
        prepared,
        frequency_plan,
        timestamp_window,
        runtime,
        third_order_cache,
    )

    output_expected = np.exp(1j * np.pi * frequency_plan.band_frequencies)
    closing_expected = np.exp(1j * np.pi * third_order_cache.closing_frequencies)
    np.testing.assert_allclose(coefficients.dc.numpy(), [[[1.0]]], atol=1e-14)
    np.testing.assert_allclose(
        coefficients.output.numpy(),
        output_expected.reshape(1, 1, 1, -1),
        rtol=0,
        atol=1e-14,
    )
    assert coefficients.third_order is not None
    np.testing.assert_allclose(
        coefficients.third_order.values.numpy(),
        closing_expected.reshape(1, 1, 1, -1),
        rtol=0,
        atol=1e-14,
    )
    assert coefficients.third_order.gather_indices is third_order_cache.gather_indices
    assert coefficients.third_order.valid_mask is third_order_cache.valid_mask


def test_unit_timestamp_coefficients_keep_empty_windows_and_omit_third_order_storage():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.empty(0, dtype=np.float64),
        window_indices=np.empty(0, dtype=np.int64),
        global_event_indices=np.empty(0, dtype=np.int64),
        estimate_count=2,
        windows_per_estimate=3,
    )
    frequency_plan = TimestampFrequencyPlan(
        actual_df=1.0,
        grid_indices=np.array([0, 1], dtype=np.int64),
        band_frequencies=np.array([0.0, 1.0]),
    )
    runtime = _runtime(torch.float32)

    coefficients = materialize_unit_timestamp_coefficients(
        prepared,
        frequency_plan,
        prepare_default_timestamp_window(runtime),
        runtime,
        third_order_cache=None,
    )

    assert coefficients.dc.shape == (1, 2, 3)
    assert coefficients.output.shape == (1, 2, 3, 2)
    assert coefficients.dc.dtype == torch.complex64
    assert coefficients.output.dtype == torch.complex64
    assert torch.count_nonzero(coefficients.dc) == 0
    assert torch.count_nonzero(coefficients.output) == 0
    assert coefficients.third_order is None
