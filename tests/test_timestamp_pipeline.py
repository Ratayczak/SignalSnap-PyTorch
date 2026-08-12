import numpy as np

from signalsnap_pytorch._core import planning as _planning
from signalsnap_pytorch._core import timestamps as _timestamps
from signalsnap_pytorch import (
    DataConfig,
    TimestampOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    calculate_spectra,
)


def _default_window(normalized_times):
    sigma = 0.14

    def gaussian(values):
        return np.exp(-((values - 0.5) / (2.0 * sigma)) ** 2)

    edge = gaussian(0.0)
    denominator = gaussian(1.0) + gaussian(-1.0)
    raw = gaussian(normalized_times) - edge * (
        gaussian(normalized_times + 1.0)
        + gaussian(normalized_times - 1.0)
    ) / denominator
    midpoint_raw = gaussian(0.5) - edge * (
        gaussian(1.5) + gaussian(-0.5)
    ) / denominator
    return raw / midpoint_raw


def _normalization(order):
    nodes, weights = np.polynomial.legendre.leggauss(512)
    window = _default_window((nodes + 1.0) / 2.0)
    return 0.5 * np.dot(weights, window**order)


def _sampled_window(point_count):
    sigma = 0.14
    points = np.arange(point_count, dtype=np.float64)
    center = (point_count - 1) / 2.0
    denominator = 2.0 * point_count * sigma

    def gaussian(values):
        return np.exp(-((values - center) / denominator) ** 2)

    edge = gaussian(-0.5)
    raw = gaussian(points) - edge * (
        gaussian(points + point_count) + gaussian(points - point_count)
    ) / (gaussian(-0.5 + point_count) + gaussian(-0.5 - point_count))
    return raw / raw.max()


def _mixed_normalization(dt, point_count, sampled_power, timestamp_power):
    sample_times = np.arange(point_count, dtype=np.float64) * dt
    sampled_taper = _sampled_window(point_count)
    timestamp_taper = _default_window(sample_times / (point_count * dt))
    return dt * np.sum(sampled_taper**sampled_power * timestamp_taper**timestamp_power)


def _mean_outer(first, second):
    return np.einsum("bmf,bmg->bfg", first, second) / first.shape[1]


def _timestamp_reference():
    frequencies = np.array([-1.0, 0.0, 1.0])
    patterns = ((), (0.25,), (0.5, 0.5), (0.2, 0.75))
    timestamps = np.array(
        [
            window_index + relative_time
            for window_index in range(8)
            for relative_time in patterns[window_index % 4]
        ]
    )
    window_coefficients = []
    dc_coefficients = []

    for window_index in range(8):
        relative_times = np.asarray(patterns[window_index % 4])
        weights = _default_window(relative_times)
        dc_coefficients.append(weights.sum())
        window_coefficients.append(
            np.sum(
                weights[:, None]
                * np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * relative_times[:, None]
                    * frequencies[None, :]
                ),
                axis=0,
            )
        )

    dc = np.asarray(dc_coefficients).reshape(2, 4)
    output = np.asarray(window_coefficients).reshape(2, 4, 3)
    centered = output - output.mean(axis=1, keepdims=True)
    m = output.shape[1]

    order_one = (dc.mean(axis=1) / _normalization(1)).mean()
    order_two = (
        m
        / (m - 1)
        * np.mean(centered * np.conj(centered), axis=1)
        / _normalization(2)
    ).mean(axis=0)

    closing_frequencies = -(
        frequencies[:, None] + frequencies[None, :]
    )
    closing = []
    for window_index in range(8):
        relative_times = np.asarray(patterns[window_index % 4])
        weights = _default_window(relative_times)
        closing.append(
            np.sum(
                weights[:, None, None]
                * np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * relative_times[:, None, None]
                    * closing_frequencies[None, :, :]
                ),
                axis=0,
            )
        )
    closing = np.asarray(closing).reshape(2, 4, 3, 3)
    closing -= closing.mean(axis=1, keepdims=True)
    order_three = (
        m**2
        / ((m - 1) * (m - 2))
        * np.mean(
            centered[:, :, :, None]
            * centered[:, :, None, :]
            * closing,
            axis=1,
        )
        / _normalization(3)
    ).mean(axis=0)

    conjugated = np.conj(centered)
    centered_xy = centered * conjugated
    centered_zw = centered * conjugated
    order_four = (
        m**2
        / ((m - 1) * (m - 2) * (m - 3))
        * (
            (m + 1) * _mean_outer(centered_xy, centered_zw)
            - (m - 1)
            * (
                np.einsum(
                    "bf,bg->bfg",
                    centered_xy.mean(axis=1),
                    centered_zw.mean(axis=1),
                )
                + _mean_outer(centered, centered)
                * _mean_outer(conjugated, conjugated)
                + _mean_outer(centered, conjugated)
                * _mean_outer(conjugated, centered)
            )
        )
        / _normalization(4)
    ).mean(axis=0)

    return timestamps, frequencies, (
        np.asarray([order_one]),
        order_two,
        order_three,
        order_four,
    )


def _window_event_coefficients(patterns, frequencies):
    dc_coefficients = []
    frequency_coefficients = []

    for relative_event_times in patterns:
        relative_event_times = np.asarray(relative_event_times, dtype=np.float64)
        weights = _default_window(relative_event_times)
        phase_shape = (-1,) + (1,) * frequencies.ndim
        phases = np.exp(
            1j
            * 2.0
            * np.pi
            * relative_event_times.reshape(phase_shape)
            * frequencies[None, ...]
        )
        dc_coefficients.append(weights.sum())
        frequency_coefficients.append(
            np.sum(weights.reshape(phase_shape) * phases, axis=0)
        )

    dc = np.asarray(dc_coefficients).reshape(2, 4)
    coefficients = np.asarray(frequency_coefficients).reshape((2, 4) + frequencies.shape)
    return dc, coefficients


def _multichannel_timestamp_reference(channel_patterns, frequencies, channels):
    coefficient_data = {
        channel: _window_event_coefficients(patterns, frequencies)
        for channel, patterns in channel_patterns.items()
    }
    order = len(channels)
    normalization = _normalization(order)
    m = 4

    if order == 1:
        dc = coefficient_data[channels[0]][0]
        return np.asarray([dc.mean(axis=1).mean() / normalization])

    centered = {
        channel: coefficients - coefficients.mean(axis=1, keepdims=True)
        for channel, (_, coefficients) in coefficient_data.items()
    }

    if order == 2:
        estimates = (
            m
            / (m - 1)
            * np.mean(centered[channels[0]] * np.conj(centered[channels[1]]), axis=1)
        )
    elif order == 3:
        closing_frequencies = -(frequencies[:, None] + frequencies[None, :])
        _, closing = _window_event_coefficients(
            channel_patterns[channels[2]],
            closing_frequencies,
        )
        closing -= closing.mean(axis=1, keepdims=True)
        estimates = (
            m**2
            / ((m - 1) * (m - 2))
            * np.mean(
                centered[channels[0]][:, :, :, None]
                * centered[channels[1]][:, :, None, :]
                * closing,
                axis=1,
            )
        )
    else:
        first = centered[channels[0]]
        second = np.conj(centered[channels[1]])
        third = centered[channels[2]]
        fourth = np.conj(centered[channels[3]])
        first_pair = first * second
        second_pair = third * fourth
        estimates = m**2 / ((m - 1) * (m - 2) * (m - 3)) * (
            (m + 1) * _mean_outer(first_pair, second_pair)
            - (m - 1)
            * (
                np.einsum(
                    "bf,bg->bfg",
                    first_pair.mean(axis=1),
                    second_pair.mean(axis=1),
                )
                + _mean_outer(first, third) * _mean_outer(second, fourth)
                + _mean_outer(first, fourth) * _mean_outer(second, third)
            )
        )

    return estimates.mean(axis=0) / normalization


def test_unit_timestamp_pipeline_orders_one_through_four_match_numpy_reference():
    timestamps, frequencies, expected_spectra = _timestamp_reference()
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=timestamps),),
        observation_start=0.0,
        observation_stop=8.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=-1.0,
        f_max=1.0,
        m=4,
        spectral_estimates_per_batch=1,
        timestamp_options=TimestampOptions(weighting="unit"),
    )
    requested_spectra = [(0,) * order for order in range(1, 5)]

    results = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=requested_spectra,
        show_progress=False,
    )

    for order, (channels, expected) in enumerate(
        zip(requested_spectra, expected_spectra),
        start=1,
    ):
        result = results[channels]
        expected_frequency = np.array([0.0]) if order == 1 else frequencies
        np.testing.assert_array_equal(result.freq, expected_frequency)
        np.testing.assert_allclose(
            result.spectrum,
            expected,
            rtol=2e-12,
            atol=2e-12,
        )


def test_two_timestamp_channels_arbitrary_tuples_match_numpy_reference():
    channel_patterns = {
        0: (
            (),
            (0.2,),
            (0.5, 0.5),
            (0.15, 0.7),
            (0.1, 0.4),
            (),
            (0.25, 0.75),
            (0.35,),
        ),
        1: (
            (0.1,),
            (),
            (0.3, 0.3),
            (0.8,),
            (0.2, 0.6),
            (0.5,),
            (),
            (0.1, 0.9),
        ),
    }
    timestamp_channels = tuple(
        TimestampedChannel(
            timestamps=np.asarray(
                [
                    window_index + relative_time
                    for window_index, pattern in enumerate(patterns)
                    for relative_time in pattern
                ]
            )
        )
        for patterns in channel_patterns.values()
    )
    frequencies = np.array([-1.0, 0.0, 1.0])
    requested_spectra = [
        (0,),
        (1,),
        (0, 1),
        (1, 0),
        (1, 0, 1),
        (0, 1, 0, 1),
    ]
    results = calculate_spectra(
        DataConfig(
            channels=timestamp_channels,
            observation_start=0.0,
            observation_stop=8.0,
        ),
        SpectrumConfig(
            df=1.0,
            f_min=-1.0,
            f_max=1.0,
            m=4,
            spectral_estimates_per_batch=1,
            timestamp_options=TimestampOptions(weighting="unit"),
        ),
        requested_spectra=requested_spectra,
        show_progress=False,
    )

    for channels in requested_spectra:
        result = results[channels]
        expected_frequencies = np.array([0.0]) if len(channels) == 1 else frequencies
        expected_spectrum = _multichannel_timestamp_reference(
            channel_patterns,
            frequencies,
            channels,
        )

        np.testing.assert_array_equal(result.freq, expected_frequencies)
        np.testing.assert_allclose(
            result.spectrum,
            expected_spectrum,
            rtol=2e-12,
            atol=2e-12,
        )


def test_unit_timestamp_pipeline_counts_complete_event_free_tail_windows():
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.5])),),
        observation_start=0.0,
        observation_stop=4.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=0.0,
        f_max=1.0,
        m=2,
        timestamp_options=TimestampOptions(weighting="unit"),
    )

    result = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0,)],
        show_progress=False,
    )[(0,)]

    expected = np.array([0.25 / _normalization(1)])
    np.testing.assert_allclose(result.spectrum, expected, rtol=1e-13)


def test_exponential_pipeline_averages_spectra_instead_of_coefficients(monkeypatch):
    amplitudes = np.array(
        [
            [1.0, 3.0, 2.0, 4.0],
            [2.0, 6.0, 1.0, 5.0],
        ]
    )

    def fixed_amplitudes(
        prepared,
        realization_ids,
        *,
        resolved_seed,
        channel_index,
        scale,
    ):
        assert resolved_seed == 123
        assert channel_index == 0
        assert scale == 1.0
        rows = np.asarray(tuple(realization_ids), dtype=np.int64)
        return amplitudes[rows[:, None], prepared.global_event_indices[None, :]]

    monkeypatch.setattr(
        _timestamps,
        "_generate_keyed_exponential_amplitudes",
        fixed_amplitudes,
    )
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.arange(4) + 0.5),),
        observation_start=0.0,
        observation_stop=4.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=0.0,
        f_max=0.5,
        m=2,
        timestamp_options=TimestampOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=2,
            seed=123,
        ),
    )

    result = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 0)],
        show_progress=False,
    )[(0, 0)]

    expected = np.array([5.0 / _normalization(2)])
    coefficient_average_result = np.array([4.5 / _normalization(2)])
    np.testing.assert_allclose(result.spectrum, expected, rtol=1e-13)
    assert not np.allclose(result.spectrum, coefficient_average_result, rtol=1e-13)
    np.testing.assert_allclose(result.spectrum_uncertainty, 0.0, atol=1e-14)


def test_exponential_pipeline_is_physical_and_repetition_batch_invariant(monkeypatch):
    patterns = ((), (0.2,), (0.25, 0.75), (0.5, 0.5, 0.8))
    timestamps = np.array(
        [
            window_index + relative_time
            for window_index in range(16)
            for relative_time in patterns[window_index % len(patterns)]
        ]
    )
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=timestamps),),
        observation_start=0.0,
        observation_stop=16.0,
    )
    requested_spectra = [(0,) * order for order in range(1, 5)]

    def calculate(estimates_per_batch):
        return calculate_spectra(
            data_config,
            SpectrumConfig(
                df=1.0,
                f_min=-1.0,
                f_max=1.0,
                m=4,
                spectral_estimates_per_batch=estimates_per_batch,
                timestamp_options=TimestampOptions(
                    weighting="exponential",
                    scale=1.25,
                    repetitions=5,
                    seed=9876,
                ),
            ),
            requested_spectra=requested_spectra,
            show_progress=False,
        )

    reference = calculate(estimates_per_batch=1)
    physical_batched = calculate(estimates_per_batch=2)
    monkeypatch.setattr(_planning, "_MAX_AMPLITUDE_REPETITIONS_PER_BATCH", 2)
    repetition_batched = calculate(estimates_per_batch=2)

    for channels in requested_spectra:
        expected = reference[channels]

        for actual_store in (physical_batched, repetition_batched):
            actual = actual_store[channels]
            np.testing.assert_array_equal(actual.freq, expected.freq)
            np.testing.assert_allclose(
                actual.spectrum,
                expected.spectrum,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                actual.spectrum_uncertainty,
                expected.spectrum_uncertainty,
                rtol=2e-13,
                atol=2e-13,
            )


def test_mixed_unit_pipeline_matches_analytic_impulse_reference():
    dt = 0.25
    window_points = 4
    counts = np.tile(np.array([0, 1, 2, 4]), 2)
    sampled = np.zeros(counts.size * window_points)
    timestamps = []

    for window_index, count in enumerate(counts):
        sampled[window_index * window_points + 2] = count / dt
        timestamps.extend([window_index + 0.5] * count)

    data_config = DataConfig(
        channels=(
            SampledChannel(data=sampled, dt=dt),
            TimestampedChannel(timestamps=np.asarray(timestamps)),
        ),
        observation_start=0.0,
        observation_stop=8.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=-1.0,
        f_max=1.0,
        m=4,
        timestamp_options=TimestampOptions(weighting="unit"),
    )
    requested_spectra = [(0, 1), (0, 1, 1), (1, 1, 0)]

    results = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=requested_spectra,
        show_progress=False,
    )

    frequencies = np.array([-1.0, 0.0, 1.0])
    centered_counts = counts[:4] - counts[:4].mean()
    expected_second = np.sum(centered_counts**2) / 3.0 / _mixed_normalization(
        dt, window_points, sampled_power=1, timestamp_power=1
    )
    expected_third = 2.0 * np.sum(centered_counts**3) / 3.0 / _mixed_normalization(
        dt, window_points, sampled_power=1, timestamp_power=2
    )

    np.testing.assert_array_equal(results[(0, 1)].freq, frequencies)
    np.testing.assert_allclose(
        results[(0, 1)].spectrum,
        expected_second,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        results[(0, 1, 1)].spectrum,
        expected_third,
        rtol=2e-12,
        atol=2e-12,
    )

    sampled_closing = results[(1, 1, 0)].spectrum
    assert np.isnan(sampled_closing[0, 0])
    np.testing.assert_allclose(
        sampled_closing[~np.isnan(sampled_closing)],
        expected_third,
        rtol=2e-12,
        atol=2e-12,
    )


def test_mixed_third_order_pipeline_supports_one_point_fft_grid():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.arange(6.0), dt=1.0),
            TimestampedChannel(
                timestamps=np.arange(6.0) + 0.25,
            ),
        ),
        observation_start=0.0,
        observation_stop=6.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=0.0,
        f_max=0.5,
        m=3,
        old_window=True,
        timestamp_options=TimestampOptions(weighting="unit"),
    )

    results = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 0, 1)],
        show_progress=False,
    )

    result = results[(0, 0, 1)]
    np.testing.assert_array_equal(result.freq, np.array([0.0]))
    assert result.spectrum.shape == (1, 1)
    assert np.all(np.isfinite(result.spectrum))


def test_mixed_calculation_returns_result_specific_frequency_views():
    dt = 0.25
    window_points = 4
    counts = np.tile(np.array([0, 1, 2, 4]), 2)
    sampled = np.zeros(counts.size * window_points)
    timestamps = []

    for window_index, count in enumerate(counts):
        sampled[window_index * window_points + 2] = count / dt
        timestamps.extend([window_index + 0.5] * count)

    data_config = DataConfig(
        channels=(
            SampledChannel(data=sampled, dt=dt),
            TimestampedChannel(timestamps=np.asarray(timestamps)),
        ),
        observation_start=0.0,
        observation_stop=8.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=-3.0,
        f_max=3.0,
        m=4,
        timestamp_options=TimestampOptions(weighting="unit"),
    )

    results = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 1), (1, 1)],
        show_progress=False,
    )

    np.testing.assert_array_equal(
        results[(0, 1)].freq,
        np.array([-2.0, -1.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        results[(1, 1)].freq,
        np.arange(-3.0, 4.0),
    )
    expected_mixed_second = (
        np.sum((counts[:4] - counts[:4].mean()) ** 2)
        / 3.0
        / _mixed_normalization(dt, window_points, sampled_power=1, timestamp_power=1)
    )
    np.testing.assert_allclose(
        results[(0, 1)].spectrum,
        expected_mixed_second,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        results[(1, 1)].spectrum,
        np.sum((counts[:4] - counts[:4].mean()) ** 2) / 3.0 / _normalization(2),
        rtol=2e-12,
        atol=2e-12,
    )


def test_fixed_seed_timestamp_result_ignores_unrelated_mixed_request():
    dt = 0.25
    sampled = np.zeros(8 * 4)
    timestamps = []

    for window_index, count in enumerate(np.tile([0, 1, 2, 3], 2)):
        sampled[window_index * 4 + 2] = count / dt
        timestamps.extend([window_index + 0.5] * count)

    data_config = DataConfig(
        channels=(
            SampledChannel(data=sampled, dt=dt),
            TimestampedChannel(timestamps=np.asarray(timestamps)),
        ),
        observation_start=0.0,
        observation_stop=8.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=-3.0,
        f_max=3.0,
        m=4,
        timestamp_options=TimestampOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=5,
            seed=2468,
        ),
    )

    timestamp_only = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(1, 1)],
        show_progress=False,
    )[(1, 1)]
    with_mixed_request = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 1), (1, 1)],
        show_progress=False,
    )[(1, 1)]

    np.testing.assert_array_equal(with_mixed_request.freq, timestamp_only.freq)
    np.testing.assert_allclose(
        with_mixed_request.spectrum,
        timestamp_only.spectrum,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        with_mixed_request.spectrum_uncertainty,
        timestamp_only.spectrum_uncertainty,
        rtol=2e-13,
        atol=2e-13,
    )


def test_fixed_seed_targets_are_request_order_and_unrelated_request_invariant():
    dt = 0.25
    physical_window_count = 12
    sampled = np.sin(2.0 * np.pi * np.arange(physical_window_count * 4) / 8.0)
    patterns = (
        (),
        (0.2,),
        (0.3, 0.3),
        (0.1, 0.7),
        (0.5,),
        (0.25, 0.75),
    )
    timestamp_channels = []

    for channel_offset in (0.0, 0.05):
        timestamp_channels.append(
            np.asarray(
                [
                    window_index + relative_time + channel_offset
                    for window_index in range(physical_window_count)
                    for relative_time in patterns[window_index % len(patterns)]
                ]
            )
        )

    data_config = DataConfig(
        channels=(
            SampledChannel(data=sampled, dt=dt),
            TimestampedChannel(timestamps=timestamp_channels[0]),
            TimestampedChannel(timestamps=timestamp_channels[1]),
        ),
        observation_start=0.0,
        observation_stop=float(physical_window_count),
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=-1.0,
        f_max=1.0,
        m=4,
        timestamp_options=TimestampOptions(
            weighting="exponential",
            scale=0.75,
            repetitions=4,
            seed=13579,
        ),
    )
    targets = [(1, 2), (2, 1, 2)]

    def calculate(requested_spectra):
        return calculate_spectra(
            data_config,
            spectrum_config,
            requested_spectra=requested_spectra,
            show_progress=False,
        )

    reference = calculate(targets)
    variants = (
        calculate(list(reversed(targets))),
        calculate([(1, 2, 1, 2), *targets]),
        calculate([(0, 0), (0, 1, 0), *targets]),
    )

    for channels in targets:
        expected = reference[channels]

        for results in variants:
            actual = results[channels]
            np.testing.assert_array_equal(actual.freq, expected.freq)
            np.testing.assert_allclose(
                actual.spectrum,
                expected.spectrum,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                actual.spectrum_uncertainty,
                expected.spectrum_uncertainty,
                rtol=2e-13,
                atol=2e-13,
            )
