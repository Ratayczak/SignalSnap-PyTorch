import numpy as np

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
