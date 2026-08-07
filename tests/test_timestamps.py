from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from signalsnap_pytorch import DataConfig, HDF5Source, TimestampedChannel
from signalsnap_pytorch._core import timestamps as _timestamps
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.fft import _prepare_default_timestamp_window
from signalsnap_pytorch._core.planning import (
    DirectFrequencyPlan,
    FFTFrequencyPlan,
    TimestampedChannelPlan,
    WindowBatch,
)
from signalsnap_pytorch._core.spectra import build_timestamp_third_order_cache
from signalsnap_pytorch._core.timestamps import (
    PreparedTimestampBatch,
    TimestampCursor,
    _direct_timestamp_transform,
    _generate_keyed_exponential_amplitudes,
    materialize_timestamp_coefficients,
    materialize_timestamp_event_amplitudes,
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


def _prepared_events(global_event_indices):
    event_indices = np.asarray(global_event_indices, dtype=np.int64)
    return PreparedTimestampBatch(
        relative_event_times=np.zeros(event_indices.size, dtype=np.float64),
        window_indices=np.zeros(event_indices.size, dtype=np.int64),
        global_event_indices=event_indices,
        estimate_count=1,
        windows_per_estimate=1,
    )


def test_keyed_exponential_amplitudes_lock_philox_stream_and_transform():
    prepared = _prepared_events(np.arange(3, 9))

    actual = _generate_keyed_exponential_amplitudes(
        prepared,
        range(2, 5),
        resolved_seed=1234,
        channel_index=7,
        scale=1.5,
    )

    expected = np.array(
        [
            [
                0.48596626469593807,
                1.0130647510740087,
                2.8540173143180034,
                0.8989021372038755,
                4.314648392085424,
                0.478821649596433,
            ],
            [
                0.9165340274484812,
                5.177763660217787,
                0.9500213756313394,
                2.3599477517112843,
                5.66941143536925,
                0.504162025831843,
            ],
            [
                0.35576219255200214,
                0.7194604886838967,
                1.2081723507989828,
                1.1179264779233187,
                2.715905356925086,
                0.19679736546458293,
            ],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_keyed_exponential_amplitudes_are_batch_and_repetition_chunk_invariant():
    full = _generate_keyed_exponential_amplitudes(
        _prepared_events(np.arange(3, 12)),
        range(2, 5),
        resolved_seed=91,
        channel_index=4,
        scale=0.75,
    )
    event_chunks = np.concatenate(
        [
            _generate_keyed_exponential_amplitudes(
                _prepared_events(np.arange(3, 7)),
                range(2, 5),
                resolved_seed=91,
                channel_index=4,
                scale=0.75,
            ),
            _generate_keyed_exponential_amplitudes(
                _prepared_events(np.arange(7, 12)),
                range(2, 5),
                resolved_seed=91,
                channel_index=4,
                scale=0.75,
            ),
        ],
        axis=1,
    )
    repetition_chunks = np.concatenate(
        [
            _generate_keyed_exponential_amplitudes(
                _prepared_events(np.arange(3, 12)),
                realization_ids,
                resolved_seed=91,
                channel_index=4,
                scale=0.75,
            )
            for realization_ids in (range(2, 3), range(3, 5))
        ],
        axis=0,
    )

    np.testing.assert_array_equal(event_chunks, full)
    np.testing.assert_array_equal(repetition_chunks, full)


def test_keyed_exponential_amplitudes_separate_channels_events_and_scale():
    prepared = _prepared_events(np.arange(4))
    channel_zero = _generate_keyed_exponential_amplitudes(
        prepared,
        range(2),
        resolved_seed=17,
        channel_index=0,
        scale=1.0,
    )
    channel_one = _generate_keyed_exponential_amplitudes(
        prepared,
        range(2),
        resolved_seed=17,
        channel_index=1,
        scale=1.0,
    )
    scaled = _generate_keyed_exponential_amplitudes(
        prepared,
        range(2),
        resolved_seed=17,
        channel_index=0,
        scale=2.5,
    )

    assert np.all(channel_zero > 0)
    assert not np.array_equal(channel_zero, channel_one)
    assert np.unique(channel_zero[0]).size == prepared.global_event_indices.size
    np.testing.assert_array_equal(scaled, 2.5 * channel_zero)


def test_keyed_exponential_amplitudes_preserve_empty_event_axis():
    actual = _generate_keyed_exponential_amplitudes(
        _prepared_events(np.empty(0, dtype=np.int64)),
        range(3, 5),
        resolved_seed=8,
        channel_index=2,
        scale=1.0,
    )

    assert actual.shape == (2, 0)
    assert actual.dtype == np.float64


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

    actual = _direct_timestamp_transform(
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

    actual = _direct_timestamp_transform(
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

    actual = _direct_timestamp_transform(
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
    expected = _direct_timestamp_transform(prepared, frequencies, weights, _runtime())

    monkeypatch.setattr(_timestamps, "_MAX_DIRECT_PHASE_ELEMENTS", 4)
    monkeypatch.setattr(_timestamps, "_MAX_DIRECT_FREQUENCIES_PER_CHUNK", 2)
    actual = _direct_timestamp_transform(prepared, frequencies, weights, _runtime())

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
        _direct_timestamp_transform(
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
    frequency_plan = DirectFrequencyPlan(
        actual_df=1.0,
        grid_indices=grid_indices,
    )
    runtime = _runtime()
    timestamp_window = _prepare_default_timestamp_window(runtime)
    third_order_cache = build_timestamp_third_order_cache(runtime, frequency_plan)

    coefficients = materialize_timestamp_coefficients(
        prepared,
        frequency_plan,
        timestamp_window,
        runtime,
        third_order_cache,
        event_amplitudes=np.ones((1, 2)),
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


def test_timestamp_coefficients_reuse_each_amplitude_for_every_frequency_role():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.5, 0.5]),
        window_indices=np.array([0, 1], dtype=np.int64),
        global_event_indices=np.array([0, 1], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=2,
    )
    grid_indices = np.arange(-1, 2, dtype=np.int64)
    frequency_plan = DirectFrequencyPlan(
        actual_df=1.0,
        grid_indices=grid_indices,
    )
    runtime = _runtime()
    third_order_cache = build_timestamp_third_order_cache(runtime, frequency_plan)
    amplitudes = np.array([[2.0, 3.0], [5.0, 7.0]])

    coefficients = materialize_timestamp_coefficients(
        prepared,
        frequency_plan,
        _prepare_default_timestamp_window(runtime),
        runtime,
        third_order_cache,
        amplitudes,
    )

    output_phases = np.exp(1j * np.pi * frequency_plan.band_frequencies)
    expected_output = amplitudes[:, None, :, None] * output_phases[None, None, None, :]
    closing_phases = np.exp(1j * np.pi * third_order_cache.closing_frequencies)
    expected_closing = amplitudes[:, None, :, None] * closing_phases[None, None, None, :]

    np.testing.assert_allclose(coefficients.dc.numpy(), amplitudes[:, None, :], atol=1e-14)
    np.testing.assert_allclose(coefficients.output.numpy(), expected_output, atol=1e-14)
    assert coefficients.third_order is not None
    np.testing.assert_allclose(
        coefficients.third_order.values.numpy(),
        expected_closing,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("needs_output", "expected_frequencies"),
    [
        pytest.param(False, [[0.0]], id="dc-only"),
        pytest.param(
            True,
            [[0.0], [-1.0, 0.0, 1.0]],
            id="dc-and-output",
        ),
    ],
)
def test_timestamp_coefficients_always_transform_dc_and_only_requested_output(
    monkeypatch,
    needs_output,
    expected_frequencies,
):
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.5]),
        window_indices=np.array([0], dtype=np.int64),
        global_event_indices=np.array([0], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=1,
    )
    frequency_plan = DirectFrequencyPlan(
        actual_df=1.0,
        grid_indices=np.arange(-1, 2, dtype=np.int64),
    )
    runtime = _runtime()
    transformed_frequencies = []

    def recording_transform(prepared, frequencies, event_weights, runtime):
        transformed_frequencies.append(frequencies.tolist())
        return torch.zeros(
            event_weights.shape[0],
            prepared.estimate_count,
            prepared.windows_per_estimate,
            frequencies.size,
            dtype=runtime.complex_dtype,
            device=runtime.device,
        )

    monkeypatch.setattr(
        _timestamps,
        "_direct_timestamp_transform",
        recording_transform,
    )

    coefficients = materialize_timestamp_coefficients(
        prepared,
        frequency_plan,
        _prepare_default_timestamp_window(runtime),
        runtime,
        third_order_cache=None,
        event_amplitudes=np.ones((1, 1)),
        needs_output=needs_output,
    )

    assert transformed_frequencies == expected_frequencies
    assert coefficients.dc.shape == (1, 1, 1)
    assert (coefficients.output is not None) is needs_output


def test_timestamp_coefficients_accept_sampled_output_frequency_view():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.5]),
        window_indices=np.array([0], dtype=np.int64),
        global_event_indices=np.array([0], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=1,
    )
    full_frequencies = np.array([-2.0, -1.0, 0.0, 1.0])
    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=full_frequencies,
        band_start=1,
        band_stop=4,
    )
    runtime = _runtime()

    coefficients = materialize_timestamp_coefficients(
        prepared,
        frequency_plan,
        _prepare_default_timestamp_window(runtime),
        runtime,
        third_order_cache=None,
        event_amplitudes=np.array([[2.5]]),
    )

    expected = 2.5 * np.exp(1j * np.pi * frequency_plan.band_frequencies)
    np.testing.assert_allclose(
        coefficients.output.numpy(),
        expected.reshape(1, 1, 1, -1),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("amplitudes", "message"),
    [
        pytest.param(np.ones(2), "shape", id="missing-realization-axis"),
        pytest.param(np.ones((0, 2)), "realization", id="empty-realization-axis"),
        pytest.param(np.ones((1, 3)), "contains 2", id="event-count-mismatch"),
    ],
)
def test_timestamp_materializer_rejects_invalid_amplitude_shape(amplitudes, message):
    prepared = _prepared_events(np.arange(2))
    frequency_plan = DirectFrequencyPlan(
        actual_df=1.0,
        grid_indices=np.array([0], dtype=np.int64),
    )

    with pytest.raises(ValueError, match=message):
        materialize_timestamp_coefficients(
            prepared,
            frequency_plan,
            _prepare_default_timestamp_window(_runtime()),
            _runtime(),
            third_order_cache=None,
            event_amplitudes=amplitudes,
        )


def test_timestamp_event_materializer_creates_unit_realization_batch():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.5, 0.5]),
        window_indices=np.array([0, 1], dtype=np.int64),
        global_event_indices=np.array([4, 5], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=2,
    )
    runtime = _runtime()

    amplitudes = materialize_timestamp_event_amplitudes(
        prepared,
        channel_index=3,
        channel_plan=TimestampedChannelPlan(
            event_count=2,
            weighting="unit",
            scale=None,
        ),
        realization_ids=range(2),
        runtime=runtime,
    )

    assert amplitudes.shape == (2, 2)
    np.testing.assert_array_equal(amplitudes, np.ones((2, 2)))


def test_timestamp_event_materializer_keys_exponential_amplitudes(monkeypatch):
    prepared = PreparedTimestampBatch(
        relative_event_times=np.array([0.5, 0.5]),
        window_indices=np.array([0, 1], dtype=np.int64),
        global_event_indices=np.array([4, 5], dtype=np.int64),
        estimate_count=1,
        windows_per_estimate=2,
    )
    runtime = _runtime()
    runtime.repetition_plan = SimpleNamespace(resolved_seed=91)
    expected_amplitudes = np.array([[2.0, 3.0], [5.0, 7.0]])

    def fake_amplitudes(
        actual_prepared,
        realization_ids,
        *,
        resolved_seed,
        channel_index,
        scale,
    ):
        assert actual_prepared is prepared
        assert realization_ids == range(2, 4)
        assert resolved_seed == 91
        assert channel_index == 6
        assert scale == 1.5
        return expected_amplitudes

    monkeypatch.setattr(
        _timestamps,
        "_generate_keyed_exponential_amplitudes",
        fake_amplitudes,
    )

    amplitudes = materialize_timestamp_event_amplitudes(
        prepared,
        channel_index=6,
        channel_plan=TimestampedChannelPlan(
            event_count=2,
            weighting="exponential",
            scale=1.5,
        ),
        realization_ids=range(2, 4),
        runtime=runtime,
    )

    np.testing.assert_array_equal(amplitudes, expected_amplitudes)


def test_unit_timestamp_coefficients_keep_empty_windows_and_omit_third_order_storage():
    prepared = PreparedTimestampBatch(
        relative_event_times=np.empty(0, dtype=np.float64),
        window_indices=np.empty(0, dtype=np.int64),
        global_event_indices=np.empty(0, dtype=np.int64),
        estimate_count=2,
        windows_per_estimate=3,
    )
    frequency_plan = DirectFrequencyPlan(
        actual_df=1.0,
        grid_indices=np.array([0, 1], dtype=np.int64),
    )
    runtime = _runtime(torch.float32)

    coefficients = materialize_timestamp_coefficients(
        prepared,
        frequency_plan,
        _prepare_default_timestamp_window(runtime),
        runtime,
        third_order_cache=None,
        event_amplitudes=np.ones((1, 0)),
    )

    assert coefficients.dc.shape == (1, 2, 3)
    assert coefficients.output.shape == (1, 2, 3, 2)
    assert coefficients.dc.dtype == torch.complex64
    assert coefficients.output.dtype == torch.complex64
    assert torch.count_nonzero(coefficients.dc) == 0
    assert torch.count_nonzero(coefficients.output) == 0
    assert coefficients.third_order is None
