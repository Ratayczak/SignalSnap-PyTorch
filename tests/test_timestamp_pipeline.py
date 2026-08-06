import numpy as np

from signalsnap_pytorch._core import planning as _planning
from signalsnap_pytorch._core import timestamps as _timestamps
from signalsnap_pytorch import (
    DataConfig,
    PhotonOptions,
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
        photon_options=PhotonOptions(weighting="unit"),
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
        photon_options=PhotonOptions(weighting="unit"),
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
        "generate_keyed_exponential_amplitudes",
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
        photon_options=PhotonOptions(
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
                photon_options=PhotonOptions(
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
