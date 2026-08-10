import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signalsnap_pytorch import (
    DataConfig,
    PhotonOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    calculate_spectra,
)
from signalsnap_pytorch._core import fft as _fft
from signalsnap_pytorch._core.planning import SampledChannelPlan, TimestampedChannelPlan
from signalsnap_pytorch._core.spectra import (
    _COEFFICIENT_ROLE_CONJUGATIONS,
    ChannelCoefficients,
    ThirdOrderCoefficients,
    compute_spectral_estimates,
    prepare_spectrum_normalizations,
)

_EXPECTED_ROLE_CONJUGATIONS = {
    1: (False,),
    2: (False, True),
    3: (False, False, False),
    4: (False, True, False, True),
}


@dataclass(frozen=True)
class ArtificialTimestampWindow:
    values: torch.Tensor
    norm_all_orders: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def evaluate(self, relative_times):
        assert relative_times.shape == self.values.shape
        return self.values

    def norm(self, order):
        return self.norm_all_orders[order - 1]


def _runtime(requested_spectra, channel_plans, *, duration=1.5, old_window=False, m=5):
    return SimpleNamespace(
        requested_spectra=tuple(requested_spectra),
        channel_plans=channel_plans,
        real_dtype=torch.float64,
        complex_dtype=torch.complex128,
        device=torch.device("cpu"),
        old_window=old_window,
        window_plan=SimpleNamespace(duration=duration, windows_per_estimate=m),
    )


def _artificial_windows(sampled_values, timestamp_values, *, dt=0.5):
    sampled_values = torch.as_tensor(sampled_values)
    timestamp_values = torch.as_tensor(timestamp_values)
    sampled_norms = tuple(dt * torch.sum(sampled_values**order) for order in range(1, 5))
    timestamp_norms = tuple(torch.tensor(10.0 + order) for order in range(1, 5))
    return (
        _fft.SampledWindow(sampled_values, sampled_norms),
        ArtificialTimestampWindow(timestamp_values, timestamp_norms),
    )


def _expected_overlap(channels, sampled_values, timestamp_values, *, dt=0.5):
    factors = []
    for channel, conjugated in zip(channels, _EXPECTED_ROLE_CONJUGATIONS[len(channels)]):
        factor = sampled_values if channel == 0 else timestamp_values
        factors.append(np.conj(factor) if conjugated else factor)
    return dt * np.prod(np.stack(factors), axis=0).sum()


@pytest.mark.parametrize(
    "channels",
    [
        (0, 1),
        (1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (0, 1, 0, 1),
    ],
)
def test_mixed_real_overlap_uses_every_coefficient_role(channels):
    sampled_values = np.array([1.0, 2.0, 4.0])
    timestamp_values = 1.0 + np.array([0.0, 0.5, 1.0])
    sampled_window, timestamp_window = _artificial_windows(
        sampled_values, timestamp_values
    )
    runtime = _runtime(
        [channels],
        {
            0: SampledChannelPlan(sample_count=30, dt=0.5),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    actual = prepare_spectrum_normalizations(
        runtime, sampled_window, timestamp_window
    )[channels]
    expected = _expected_overlap(channels, sampled_values, timestamp_values)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "channels",
    [
        (0, 1),
        (1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ],
)
def test_complex_overlap_follows_explicit_coefficient_conjugations(channels):
    sampled_values = np.array([1.0 + 2.0j, 2.0 - 0.5j, -0.25 + 1.0j])
    timestamp_values = np.array([0.5 - 1.0j, 1.5 + 0.25j, 2.0 + 0.75j])
    sampled_window, timestamp_window = _artificial_windows(
        sampled_values, timestamp_values
    )
    runtime = _runtime(
        [channels],
        {
            0: SampledChannelPlan(sample_count=30, dt=0.5),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    actual = prepare_spectrum_normalizations(
        runtime, sampled_window, timestamp_window
    )[channels]
    expected = _expected_overlap(channels, sampled_values, timestamp_values)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_coefficient_role_conjugation_mapping_matches_estimator_roles(order):
    assert _COEFFICIENT_ROLE_CONJUGATIONS[order] == _EXPECTED_ROLE_CONJUGATIONS[order]


def test_third_order_closing_taper_is_not_conjugated():
    sampled_values = np.array([1.0 + 1.0j, 2.0 - 0.5j, 0.5 + 0.25j])
    timestamp_values = np.array([0.5 + 2.0j, 1.0 - 1.0j, 3.0 + 0.5j])
    channels = (1, 1, 0)
    sampled_window, timestamp_window = _artificial_windows(
        sampled_values, timestamp_values
    )
    runtime = _runtime(
        [channels],
        {
            0: SampledChannelPlan(sample_count=30, dt=0.5),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    actual = prepare_spectrum_normalizations(
        runtime, sampled_window, timestamp_window
    )[channels].numpy()
    direct_closing = 0.5 * np.sum(timestamp_values**2 * sampled_values)
    conjugated_closing = 0.5 * np.sum(timestamp_values**2 * np.conj(sampled_values))

    np.testing.assert_allclose(actual, direct_closing, rtol=1e-15, atol=1e-15)
    assert not np.isclose(actual, conjugated_closing)


@pytest.mark.parametrize("old_window", [False, True])
def test_homogeneous_paths_return_existing_norms_exactly(old_window):
    requested = tuple(
        [(0,) * order for order in range(1, 5)]
        + [(1,) * order for order in range(1, 5)]
    )
    runtime = _runtime(
        requested,
        {
            0: SampledChannelPlan(sample_count=64, dt=0.125),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
        duration=1.0,
        old_window=old_window,
    )
    sampled_window = _fft.prepare_window(runtime, dt=0.125, window_points=8)
    timestamp_window = _fft.prepare_timestamp_window(runtime)

    actual = prepare_spectrum_normalizations(runtime, sampled_window, timestamp_window)

    for order in range(1, 5):
        assert torch.equal(actual[(0,) * order], sampled_window.norm(order))
        assert torch.equal(actual[(1,) * order], timestamp_window.norm(order))


def test_timestamp_window_is_evaluated_on_sample_grid_once():
    class CountingTimestampWindow(ArtificialTimestampWindow):
        calls = 0

        def evaluate(self, relative_times):
            type(self).calls += 1
            np.testing.assert_array_equal(relative_times.numpy(), np.array([0.0, 0.5, 1.0]))
            return super().evaluate(relative_times)

    sampled_window, timestamp_window_base = _artificial_windows(
        [1.0, 2.0, 4.0], [1.0, 1.5, 2.0]
    )
    timestamp_window = CountingTimestampWindow(
        timestamp_window_base.values, timestamp_window_base.norm_all_orders
    )
    requested = ((0, 1), (1, 0), (0, 1, 1), (1, 1, 1, 0))
    runtime = _runtime(
        requested,
        {
            0: SampledChannelPlan(sample_count=30, dt=0.5),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    prepare_spectrum_normalizations(runtime, sampled_window, timestamp_window)

    assert CountingTimestampWindow.calls == 1


@pytest.mark.parametrize(
    "timestamp_values",
    [
        np.array([1.0, -1.0]),
        np.array([1.0, -1.0 + np.finfo(np.float64).eps]),
    ],
)
def test_zero_or_negligible_mixed_overlap_is_rejected(timestamp_values):
    sampled_window, timestamp_window = _artificial_windows(
        np.ones(2), timestamp_values, dt=1.0
    )
    runtime = _runtime(
        [(0, 1)],
        {
            0: SampledChannelPlan(sample_count=20, dt=1.0),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    with pytest.raises(ValueError, match="zero or numerically negligible"):
        prepare_spectrum_normalizations(runtime, sampled_window, timestamp_window)


def _coefficient_fixture(sampled_scale, timestamp_scale):
    sampled_base = torch.tensor(
        [-2.0 + 0.5j, -0.5 - 1.0j, 0.25 + 0.75j, 1.0 - 0.25j, 2.5 + 0.0j],
        dtype=torch.complex128,
    ).reshape(1, 1, 5, 1)
    timestamp_base = torch.tensor(
        [1.5 - 0.5j, -1.0 + 0.25j, 0.5 + 1.0j, 2.0 - 0.75j, -0.25 - 0.5j],
        dtype=torch.complex128,
    ).reshape(1, 1, 5, 1)
    timestamp_closing = torch.tensor(
        [-0.5 + 1.0j, 1.25 - 0.5j, 2.0 + 0.25j, -1.5 - 0.75j, 0.75 + 0.5j],
        dtype=torch.complex128,
    ).reshape(1, 1, 5, 1)
    gather_indices = torch.zeros((1, 1), dtype=torch.long)
    valid_mask = torch.ones((1, 1), dtype=torch.bool)

    return {
        0: ChannelCoefficients(
            dc=torch.zeros((1, 1, 5), dtype=torch.complex128),
            output=sampled_scale * sampled_base,
        ),
        1: ChannelCoefficients(
            dc=torch.zeros((1, 1, 5), dtype=torch.complex128),
            output=timestamp_scale * timestamp_base,
            third_order=ThirdOrderCoefficients(
                values=timestamp_scale * timestamp_closing,
                gather_indices=gather_indices,
                valid_mask=valid_mask,
            ),
        ),
    }


@pytest.mark.parametrize("channels", [(0, 1), (0, 1, 1), (0, 1, 0, 1)])
def test_relative_taper_and_coefficient_rescaling_leaves_spectrum_unchanged(channels):
    sampled_values = torch.tensor(
        [1.0 + 0.2j, 2.0 - 0.1j, 1.3 + 0.4j], dtype=torch.complex128
    )
    timestamp_values = torch.tensor(
        [0.8 - 0.3j, 1.4 + 0.5j, 0.9 + 0.2j], dtype=torch.complex128
    )
    alpha = 1.7 + 0.4j
    beta = 0.8 - 0.3j
    runtime = _runtime(
        [channels],
        {
            0: SampledChannelPlan(sample_count=30, dt=0.5),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
    )

    base_windows = _artificial_windows(sampled_values, timestamp_values)
    scaled_windows = _artificial_windows(alpha * sampled_values, beta * timestamp_values)
    base_normalization = prepare_spectrum_normalizations(runtime, *base_windows)[channels]
    scaled_normalization = prepare_spectrum_normalizations(runtime, *scaled_windows)[channels]

    base = compute_spectral_estimates(
        channels, _coefficient_fixture(1.0, 1.0), base_normalization, runtime
    )
    scaled = compute_spectral_estimates(
        channels, _coefficient_fixture(alpha, beta), scaled_normalization, runtime
    )

    torch.testing.assert_close(scaled, base, rtol=2e-13, atol=2e-13)


def _old_calc_window_numpy(values, n_windows, length):
    center = n_windows * 0.5
    denominator = 2.0 * length * 0.14

    def gaussian(points):
        return np.exp(-((points - center) / denominator) ** 2)

    edge = gaussian(-0.5)
    return gaussian(values) - edge * (
        gaussian(values + length) + gaussian(values - length)
    ) / (gaussian(-0.5 + length) + gaussian(-0.5 - length))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_legacy_mixed_overlap_uses_sample_grid_instead_of_fixed_reference_norm(order):
    point_count = 8
    dt = 0.125
    duration = point_count * dt
    channels = (0,) + (1,) * (order - 1)
    runtime = _runtime(
        [channels],
        {
            0: SampledChannelPlan(sample_count=64, dt=dt),
            1: TimestampedChannelPlan(event_count=4, weighting="unit", scale=None),
        },
        duration=duration,
        old_window=True,
    )
    sampled_window = _fft.prepare_window(runtime, dt=dt, window_points=point_count)
    timestamp_window = _fft.prepare_timestamp_window(runtime)

    actual = prepare_spectrum_normalizations(
        runtime, sampled_window, timestamp_window
    )[channels].numpy()

    sampled_raw = _old_calc_window_numpy(
        np.linspace(0.0, point_count, point_count), point_count, point_count + 1
    )
    sampled_taper = sampled_raw / np.sqrt(np.sum(sampled_raw**2))
    reference_points = 70
    reference_dt = duration / reference_points
    timestamp_reference_raw = _old_calc_window_numpy(
        np.linspace(0.0, reference_points, reference_points),
        reference_points,
        reference_points + 1,
    )
    timestamp_scale = 1.0 / np.sqrt(
        reference_dt * np.sum(timestamp_reference_raw**2)
    )
    timestamp_taper = timestamp_scale * _old_calc_window_numpy(
        np.arange(point_count) * dt / reference_dt,
        reference_points,
        reference_points + 1,
    )
    expected = dt * np.sum(sampled_taper * timestamp_taper ** (order - 1))

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
    assert not np.isclose(actual, timestamp_window.norm(order).numpy(), rtol=1e-4)


def _default_window_numpy(normalized_times):
    sigma = 0.14

    def gaussian(values):
        return np.exp(-((values - 0.5) / (2.0 * sigma)) ** 2)

    edge = gaussian(0.0)
    denominator = gaussian(1.0) + gaussian(-1.0)
    raw = gaussian(normalized_times) - edge * (
        gaussian(normalized_times + 1.0) + gaussian(normalized_times - 1.0)
    ) / denominator
    midpoint = gaussian(0.5) - edge * (
        gaussian(1.5) + gaussian(-0.5)
    ) / denominator
    return raw / midpoint


def _sampled_window_numpy(point_count):
    points = np.arange(point_count, dtype=np.float64)
    center = (point_count - 1) / 2.0

    def gaussian(values):
        return np.exp(-((values - center) / (2.0 * point_count * 0.14)) ** 2)

    edge = gaussian(-0.5)
    raw = gaussian(points) - edge * (
        gaussian(points + point_count) + gaussian(points - point_count)
    ) / (gaussian(-0.5 + point_count) + gaussian(-0.5 - point_count))
    return raw / raw.max()


def test_discrete_mixed_overlap_converges_to_continuous_product():
    nodes, weights = np.polynomial.legendre.leggauss(1024)
    normalized_nodes = (nodes + 1.0) / 2.0
    continuous_window = _default_window_numpy(normalized_nodes)
    continuous = 0.5 * np.dot(weights, continuous_window**3)
    errors = []
    ratios = []

    for point_count in (8, 16, 32, 64, 128):
        dt = 1.0 / point_count
        sample_times = np.arange(point_count, dtype=np.float64) * dt
        sampled_taper = _sampled_window_numpy(point_count)
        timestamp_taper = _default_window_numpy(sample_times)
        discrete = dt * np.sum(sampled_taper * timestamp_taper**2)
        errors.append(abs(discrete - continuous))
        ratios.append(discrete / continuous)

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert abs(ratios[-1] - 1.0) < 2e-3


def test_mixed_pipeline_permutations_and_sampled_closing_support():
    dt = 0.25
    point_count = 4
    counts = np.tile(np.array([0, 1, 2, 4]), 2)
    sampled = np.zeros(counts.size * point_count)
    timestamps = []
    for window_index, count in enumerate(counts):
        sampled[window_index * point_count + 2] = count / dt
        timestamps.extend([window_index + 0.5] * count)

    requested = [
        (0, 1),
        (1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ]
    results = calculate_spectra(
        DataConfig(
            channels=(
                SampledChannel(data=sampled, dt=dt),
                TimestampedChannel(timestamps=np.asarray(timestamps)),
            ),
            observation_start=0.0,
            observation_stop=8.0,
        ),
        SpectrumConfig(
            df=1.0,
            f_min=-1.0,
            f_max=1.0,
            m=4,
            photon_options=PhotonOptions(weighting="unit"),
        ),
        requested_spectra=requested,
        show_progress=False,
    )

    np.testing.assert_allclose(results[(0, 1)].spectrum, results[(1, 0)].spectrum)
    third_reference = results[(0, 1, 1)].spectrum
    np.testing.assert_allclose(results[(1, 0, 1)].spectrum, third_reference)
    sampled_closing = results[(1, 1, 0)].spectrum
    assert np.isnan(sampled_closing[0, 0])
    np.testing.assert_allclose(
        sampled_closing[~np.isnan(sampled_closing)],
        third_reference[~np.isnan(sampled_closing)],
    )
    fourth_reference = results[(0, 1, 1, 1)].spectrum
    for channels in requested[6:]:
        np.testing.assert_allclose(results[channels].spectrum, fourth_reference)


def test_grid_aligned_timestamp_and_sampled_count_rate_equivalence(monkeypatch):
    dt = 0.25
    taper_values = torch.tensor([0.5, 1.25, 0.75, 1.5], dtype=torch.float64)
    norms = tuple(dt * torch.sum(taper_values**order) for order in range(1, 5))

    @dataclass(frozen=True)
    class GridTimestampWindow:
        def evaluate(self, relative_times):
            indices = torch.round(relative_times / dt).to(torch.long)
            return taper_values.to(relative_times.device)[indices]

        def norm(self, order):
            return norms[order - 1]

    def prepare_sampled(runtime, dt, window_points):
        assert window_points == taper_values.numel()
        return _fft.SampledWindow(taper_values.to(runtime.device), norms)

    def prepare_timestamp(runtime):
        return GridTimestampWindow()

    monkeypatch.setattr(_fft, "prepare_window", prepare_sampled)
    monkeypatch.setattr(_fft, "prepare_timestamp_window", prepare_timestamp)

    # Independently verify coefficient equivalence with repeated events, non-unit marks, and an
    # empty Fourier node.
    event_nodes = np.array([0, 1, 1, 3])
    event_marks = np.array([1.5, -0.25, 2.0, 0.75])
    mark_sums = np.zeros(4)
    np.add.at(mark_sums, event_nodes, event_marks)
    sampled_coefficients = _fft.compute_fft(
        torch.tensor(mark_sums / dt).reshape(1, 1, 4), taper_values, dt
    ).numpy()[0, 0, 0]
    frequencies = np.fft.fftshift(np.fft.fftfreq(4, dt))
    timestamp_coefficients = np.sum(
        taper_values.numpy()[event_nodes, None]
        * event_marks[:, None]
        * np.exp(2j * np.pi * event_nodes[:, None] * dt * frequencies[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(sampled_coefficients, timestamp_coefficients, atol=2e-15)

    node_counts = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 2, 0],
            [0, 3, 0, 1],
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 2, 0, 3],
            [4, 0, 1, 0],
        ]
    )
    sampled = (node_counts / dt).reshape(-1)
    timestamps = np.asarray(
        [
            window_index + node_index * dt
            for window_index, window_counts in enumerate(node_counts)
            for node_index, count in enumerate(window_counts)
            for _ in range(count)
        ]
    )
    requested = [
        (0, 0),
        (1, 1),
        (0, 1),
        (0, 0, 0),
        (1, 1, 1),
        (0, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (0, 1, 0, 1),
    ]
    results = calculate_spectra(
        DataConfig(
            channels=(
                SampledChannel(data=sampled, dt=dt),
                TimestampedChannel(timestamps=timestamps),
            ),
            observation_start=0.0,
            observation_stop=8.0,
        ),
        SpectrumConfig(
            df=1.0,
            f_min=-1.0,
            f_max=1.0,
            m=4,
            photon_options=PhotonOptions(weighting="unit"),
        ),
        requested_spectra=requested,
        show_progress=False,
    )

    for sampled_channels, timestamp_channels, mixed_channels in (
        ((0, 0), (1, 1), (0, 1)),
        ((0, 0, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1)),
    ):
        np.testing.assert_allclose(
            results[mixed_channels].spectrum,
            results[timestamp_channels].spectrum,
            rtol=2e-13,
            atol=2e-13,
        )
        sampled_spectrum = results[sampled_channels].spectrum
        valid = ~np.isnan(sampled_spectrum)
        np.testing.assert_allclose(
            results[timestamp_channels].spectrum[valid],
            sampled_spectrum[valid],
            rtol=2e-13,
            atol=2e-13,
        )


@pytest.mark.parametrize(
    ("weighting", "scale", "repetitions"),
    [("unit", None, None), ("exponential", 0.7, 24)],
)
def test_homogeneous_poisson_timestamp_spectra_match_flat_cumulant_reference(
    weighting, scale, repetitions
):
    rate = 2.0
    duration = 2048.0
    requested = [(0,) * order for order in range(1, 5)]
    replicate_results = []

    for replicate_seed in (1201, 1202, 1203, 1204):
        rng = np.random.default_rng(replicate_seed)
        event_count = rng.poisson(rate * duration)
        timestamps = np.sort(rng.uniform(0.0, duration, event_count))
        photon_options = (
            PhotonOptions(weighting="unit")
            if weighting == "unit"
            else PhotonOptions(
                weighting="exponential",
                scale=scale,
                repetitions=repetitions,
                seed=9000 + replicate_seed,
            )
        )
        results = calculate_spectra(
            DataConfig(
                channels=(TimestampedChannel(timestamps=timestamps),),
                observation_start=0.0,
                observation_stop=duration,
            ),
            SpectrumConfig(
                df=1.0,
                f_min=-0.1,
                f_max=0.1,
                m=16,
                spectral_estimates_per_batch=32,
                photon_options=photon_options,
            ),
            requested_spectra=requested,
            show_progress=False,
        )
        replicate_results.append(
            [
                float(np.real(np.asarray(results[channels].spectrum)).reshape(-1)[0])
                for channels in requested
            ]
        )

    replicate_results = np.asarray(replicate_results)
    if weighting == "unit":
        expected = np.full(4, rate)
    else:
        expected = np.asarray(
            [rate * math.factorial(order) * scale**order for order in range(1, 5)]
        )
    replicate_means = replicate_results.mean(axis=0)
    standard_errors = replicate_results.std(axis=0, ddof=1) / np.sqrt(len(replicate_results))

    np.testing.assert_array_less(
        np.abs(replicate_means - expected),
        6.0 * standard_errors + 0.05 * expected,
    )


def test_unsupported_order_is_rejected_by_normalization_preparation():
    channels = (0, 0, 0, 0, 0)
    sampled_window, _ = _artificial_windows([1.0, 2.0, 4.0], [1.0, 1.5, 2.0])
    runtime = _runtime(
        [channels],
        {0: SampledChannelPlan(sample_count=30, dt=0.5)},
    )

    with pytest.raises(ValueError, match="Unsupported spectrum order: 5"):
        prepare_spectrum_normalizations(runtime, sampled_window, None)
