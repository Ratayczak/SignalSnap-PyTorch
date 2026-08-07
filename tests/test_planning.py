from contextlib import nullcontext

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
from signalsnap_pytorch._core.accumulation import initialize_accumulator_store
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.planning import (
    _MAX_AMPLITUDE_REPETITIONS_PER_BATCH,
    DirectFrequencyPlan,
    FFTFrequencyPlan,
    SampledChannelPlan,
    TimestampedChannelPlan,
    _build_channel_plans,
    _count_complete_windows,
    _resolve_device,
    _resolve_repetition_plan,
    build_runtime_config,
    iter_window_batches,
    physical_estimate_count,
    resolve_requested_spectra,
    resolve_sampled_frequencies,
    resolve_timestamp_frequencies,
)
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH, sampled_data_config

auto_spectra = [(0,), (0, 0)]


def _build_runtime(data_config, spectrum_config, requested_spectra):
    spectra_channels, active_data_channels = resolve_requested_spectra(
        requested_spectra,
        channel_count=len(data_config.channels),
    )

    with open_channels(data_config, active_data_channels) as opened_channels:
        return build_runtime_config(
            data_config=data_config,
            opened_channels=opened_channels,
            spectrum_config=spectrum_config,
            spectra_channels=spectra_channels,
        )


@pytest.mark.parametrize(
    (
        "n_data_points",
        "spectral_estimates_max",
        "auto_spectra_channels",
        "frequency_points",
        "f_max",
        "m",
        "expected_unshifted_estimates",
        "expected_m",
    ),
    [
        pytest.param(80, None, auto_spectra, 9, 0.5, 4, 1, 4, id="uncapped"),
        pytest.param(136, 1, auto_spectra, 9, 0.5, 4, 1, 4, id="cap-below-available"),
        pytest.param(128, 2, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-equals-available"),
        pytest.param(136, 10, auto_spectra, 9, 0.5, 4, 2, 4, id="cap-above-available"),
        pytest.param(127, None, auto_spectra, 9, 0.5, 4, 1, 4, id="one-before-next-base"),
        pytest.param(63, None, auto_spectra, 9, 0.5, 4, 1, 3, id="m-reduced-at-short-boundary"),
        pytest.param(
            256,
            3,
            auto_spectra + [(0, 0, 0), (0, 0, 0, 0)],
            9,
            0.5,
            4,
            3,
            4,
            id="higher-orders-capped",
        ),
    ],
)
def test_spectral_estimates_in_runtime_config(
    n_data_points,
    spectral_estimates_max,
    auto_spectra_channels,
    frequency_points,
    f_max,
    m,
    expected_unshifted_estimates,
    expected_m,
):

    df = (f_max - 0.0) / (frequency_points - 1)
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=f_max,
        df=df,
        m=m,
        spectral_estimates_max=spectral_estimates_max,
    )
    data_config = sampled_data_config(channels=(np.ones(n_data_points),), dt=1.0)

    warning_context = (
        pytest.warns(UserWarning, match=f"using m={expected_m} instead")
        if expected_m != m
        else nullcontext()
    )

    with warning_context:
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra_channels)

    assert runtime.window_plan.windows_per_estimate == expected_m
    assert runtime.window_plan.unshifted_estimate_count == expected_unshifted_estimates


def test_runtime_config_propagates_short_term_uncertainty_configuration():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="short_term",
        m_var=3,
        spectral_estimates_per_batch=2,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(256),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.uncertainty_estimation == "short_term"
    assert runtime.m_var == 3
    assert runtime.window_plan.unshifted_estimate_count == 4
    assert runtime.window_plan.estimates_per_batch == 2


@pytest.mark.parametrize(
    (
        "uncertainty_estimation",
        "configured_batch_size",
        "expected_batch_size",
    ),
    [
        pytest.param("short_term", 8, 6, id="short-term-rounded-down"),
        pytest.param("short_term", 6, 6, id="short-term-already-aligned"),
        pytest.param("short_term", 2, 2, id="short-term-smaller-than-m-var"),
        pytest.param("global", 8, 8, id="global-unchanged"),
    ],
)
def test_runtime_config_aligns_short_term_batch_size_with_m_var(
    uncertainty_estimation,
    configured_batch_size,
    expected_batch_size,
):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=4,
        uncertainty_estimation=uncertainty_estimation,
        m_var=3,
        spectral_estimates_max=8,
        spectral_estimates_per_batch=configured_batch_size,
    )
    data_config = sampled_data_config(channels=(np.ones(10_000),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.window_plan.estimates_per_batch == expected_batch_size
    assert spectrum_config.spectral_estimates_per_batch == configured_batch_size


def test_runtime_config_aligns_batch_size_with_reduced_m_var():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=4,
        uncertainty_estimation="short_term",
        m_var=10,
        spectral_estimates_max=3,
        spectral_estimates_per_batch=8,
    )
    data_config = sampled_data_config(channels=(np.ones(10_000),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=3 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.m_var == 3
    assert runtime.window_plan.estimates_per_batch == 6


def test_runtime_config_reduces_short_term_m_var_to_available_estimates():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="short_term",
        m_var=10,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(128),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=2 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.window_plan.unshifted_estimate_count == 2
    assert runtime.m_var == 2
    assert spectrum_config.m_var == 10


def test_runtime_config_applies_estimate_cap_before_reducing_m_var():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="short_term",
        m_var=8,
        spectral_estimates_max=3,
    )
    data_config = sampled_data_config(channels=(np.ones(640),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=3 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.window_plan.unshifted_estimate_count == 3
    assert runtime.m_var == 3


def test_runtime_config_does_not_reduce_short_term_m_var_to_one():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="short_term",
        m_var=10,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(64),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.window_plan.unshifted_estimate_count == 1
    assert runtime.m_var == 10


def test_global_runtime_keeps_configured_m_var_when_fewer_estimates_are_available():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="global",
        m_var=10,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(128),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.uncertainty_estimation == "global"
    assert runtime.window_plan.unshifted_estimate_count == 2
    assert runtime.m_var == 10


def test_accumulator_store_receives_resolved_uncertainty_configuration():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        uncertainty_estimation="short_term",
        m_var=3,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(256),), dt=1.0)
    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    store = initialize_accumulator_store(runtime)

    assert len(tuple(store)) == len(runtime.requested_spectra)
    for accumulator in store:
        assert accumulator.uncertainty_estimation == "short_term"
        assert accumulator.m_var == 3

        frequency_plan = runtime.frequency_plan_for(accumulator.channels)
        expected_frequencies = (
            np.asarray([0.0])
            if len(accumulator.channels) == 1
            else frequency_plan.band_frequencies
        )
        np.testing.assert_array_equal(accumulator.freq, expected_frequencies)


@pytest.mark.parametrize(
    (
        "n_data_points",
        "frequency_points",
        "f_max",
        "m",
        "spectral_estimates_per_batch",
        "interlacing",
        "expected_slices",
        "expected_spectral_estimates",
    ),
    [
        pytest.param(
            136,
            9,
            0.5,
            4,
            2,
            True,
            [(0, 128, 2, False), (8, 136, 2, True)],
            2,
            id="even-window-interlacing-enabled",
        ),
        pytest.param(
            136,
            9,
            0.5,
            4,
            3,
            False,
            [(0, 128, 2, False)],
            2,
            id="even-window-interlacing-disabled",
        ),
        pytest.param(
            96,
            6,
            1 / 3,
            3,
            2,
            True,
            [(0, 90, 2, False), (7, 52, 1, True)],
            2,
            id="odd-window-before-second-shifted-estimate",
        ),
        pytest.param(
            97,
            6,
            1 / 3,
            3,
            3,
            True,
            [(0, 90, 2, False), (7, 97, 2, True)],
            2,
            id="odd-window-at-second-shifted-estimate",
        ),
        pytest.param(
            320,
            9,
            0.5,
            4,
            2,
            False,
            [
                (0, 128, 2, False),
                (128, 256, 2, False),
                (256, 320, 1, False),
            ],
            5,
            id="incomplete-final-batch",
        ),
    ],
)
def test_window_slices_respect_interlacing(
    n_data_points,
    frequency_points,
    f_max,
    m,
    spectral_estimates_per_batch,
    interlacing,
    expected_slices,
    expected_spectral_estimates,
):
    df = (f_max - 0.0) / (frequency_points - 1)
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=f_max,
        df=df,
        m=m,
        spectral_estimates_per_batch=spectral_estimates_per_batch,
        spectral_estimates_max=None,
        interlacing=interlacing,
    )
    data_config = sampled_data_config(channels=(np.ones(n_data_points),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)
    batches = list(iter_window_batches(runtime.window_plan))
    frequency_plan = runtime.fft_frequency_plan
    assert frequency_plan is not None

    assert runtime.window_plan.unshifted_estimate_count == expected_spectral_estimates
    assert physical_estimate_count(runtime.window_plan) == sum(
        expected[2] for expected in expected_slices
    )
    assert len(batches) == len(expected_slices)

    for batch, (start, end, estimate_count, shifted) in zip(batches, expected_slices):
        expected_starts = np.arange(
            start,
            end,
            frequency_plan.window_points,
            dtype=np.float64,
        )
        expected_starts = expected_starts.reshape(
            estimate_count,
            runtime.window_plan.windows_per_estimate,
        )

        np.testing.assert_array_equal(batch.relative_starts, expected_starts)
        assert batch.duration == runtime.window_plan.duration
        assert batch.estimate_count == estimate_count
        assert batch.shifted is shifted

    assert (runtime.window_plan.shifted_estimate_count > 0) is interlacing


def test_short_term_batch_alignment_is_applied_separately_to_each_placement():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=4,
        uncertainty_estimation="short_term",
        m_var=3,
        spectral_estimates_max=8,
        spectral_estimates_per_batch=8,
        interlacing=True,
    )
    data_config = sampled_data_config(channels=(np.ones(10_000),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)
    batches = list(iter_window_batches(runtime.window_plan))

    assert [batch.estimate_count for batch in batches] == [6, 2, 6, 2]
    assert [batch.shifted for batch in batches] == [False, False, True, True]
    assert sum(batch.estimate_count for batch in batches if not batch.shifted) == 8
    assert sum(batch.estimate_count for batch in batches if batch.shifted) == 8


def test_pipeline_returns_full_axis_third_order_spectrum_with_invalid_points_masked():
    spectrum_config = SpectrumConfig(
        f_min=-0.25,
        f_max=0.25,
        df=0.125,
        m=4,
        spectral_estimates_max=1,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    )
    data_config = sampled_data_config(channels=(np.ones(64),), dt=1.0)

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result_store = calculate_spectra(
            data_config, spectrum_config, requested_spectra=[(0, 0, 0)]
        )
    result = result_store[(0, 0, 0)]
    assert result is not None

    assert result.spectrum.shape == (result.freq.size, result.freq.size)

    assert spectrum_config.df is not None
    dt = data_config.channels[0].dt
    window_points = int(np.round(1 / (spectrum_config.df * dt)))
    full_fft_freq = np.fft.fftshift(np.fft.fftfreq(window_points, dt))
    third_factor_freq = -(result.freq[:, None] + result.freq[None, :])
    expected_valid_mask = np.isclose(
        third_factor_freq[..., None],
        full_fft_freq,
        rtol=0.0,
        atol=1e-12,
    ).any(axis=-1)

    np.testing.assert_array_equal(np.isnan(result.spectrum), ~expected_valid_mask)
    assert np.isfinite(result.spectrum[expected_valid_mask]).all()


def test_pipeline_produces_short_term_uncertainty():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=2,
        uncertainty_estimation="short_term",
        m_var=2,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(
        channels=(np.arange(32, dtype=np.float64),),
        dt=1.0,
    )

    result_store = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0,)],
        show_progress=False,
    )
    result = result_store[(0,)]

    assert result.spectrum.shape == (1,)
    assert result.spectrum_uncertainty is not None
    assert result.spectrum_uncertainty.shape == result.spectrum.shape
    assert np.isfinite(result.spectrum_uncertainty).all()


@pytest.mark.parametrize(
    ("f_min", "f_max", "expected_start", "expected_stop", "expected_band"),
    [
        pytest.param(
            -0.25,
            0.25,
            2,
            7,
            [-0.25, -0.125, 0.0, 0.125, 0.25],
            id="exact-endpoints-are-inclusive",
        ),
        pytest.param(
            -0.249,
            0.249,
            3,
            6,
            [-0.125, 0.0, 0.125],
            id="neighboring-bins-outside-hard-bounds-are-excluded",
        ),
        pytest.param(
            -0.5,
            0.5,
            0,
            8,
            [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375],
            id="even-grid-has-negative-but-not-positive-nyquist-bin",
        ),
        pytest.param(
            -1.0,
            0.25,
            0,
            7,
            [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25],
            id="lower-bound-beyond-sampled-support-is-intersected",
        ),
        pytest.param(
            -0.25,
            1.0,
            2,
            8,
            [-0.25, -0.125, 0.0, 0.125, 0.25, 0.375],
            id="upper-bound-beyond-sampled-support-is-intersected",
        ),
    ],
)
def test_fft_frequency_plan_applies_exact_inclusive_hard_bounds(
    f_min,
    f_max,
    expected_start,
    expected_stop,
    expected_band,
):
    spectrum_config = SpectrumConfig(
        f_min=f_min,
        f_max=f_max,
        df=0.125,
    )

    window_points, frequency_plan = resolve_sampled_frequencies(spectrum_config, dt=1.0)

    assert window_points == 8
    assert frequency_plan.band_start == expected_start
    assert frequency_plan.band_stop == expected_stop
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        np.asarray(expected_band),
    )
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        frequency_plan.shifted_full_fft_frequencies[expected_start:expected_stop],
    )
    assert np.all(frequency_plan.band_frequencies >= f_min)
    assert np.all(frequency_plan.band_frequencies <= f_max)

    if expected_start > 0:
        assert frequency_plan.shifted_full_fft_frequencies[expected_start - 1] < f_min

    if expected_stop < window_points:
        assert frequency_plan.shifted_full_fft_frequencies[expected_stop] > f_max


def test_fft_frequency_plan_uses_actual_odd_fft_support():
    spectrum_config = SpectrumConfig(
        f_min=-0.5,
        f_max=0.5,
        df=0.2,
    )

    window_points, frequency_plan = resolve_sampled_frequencies(spectrum_config, dt=1.0)

    assert window_points == 5
    np.testing.assert_allclose(
        frequency_plan.shifted_full_fft_frequencies,
        [-0.4, -0.2, 0.0, 0.2, 0.4],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        frequency_plan.shifted_full_fft_frequencies,
    )


def test_resolve_sampled_frequencies_rejects_band_without_fft_frequency():
    spectrum_config = SpectrumConfig(
        f_min=0.1,
        f_max=0.2,
        df=0.5,
    )

    with pytest.raises(ValueError, match="does not contain any FFT frequencies"):
        resolve_sampled_frequencies(spectrum_config, dt=1.0)


@pytest.mark.parametrize(
    ("f_min", "f_max"),
    [
        pytest.param(0.6, 1.0, id="entirely-above-sampled-support"),
        pytest.param(-1.0, -0.6, id="entirely-below-sampled-support"),
    ],
)
def test_fft_frequency_plan_rejects_band_disjoint_from_fft_support(f_min, f_max):
    spectrum_config = SpectrumConfig(
        f_min=f_min,
        f_max=f_max,
        df=0.125,
    )

    with pytest.raises(ValueError, match="does not contain any FFT frequencies"):
        resolve_sampled_frequencies(spectrum_config, dt=1.0)


@pytest.mark.parametrize(
    ("f_min", "f_max", "expected_band"),
    [
        pytest.param(
            -0.25,
            0.25,
            [-0.25, -0.125, 0.0, 0.125, 0.25],
            id="exact-endpoints-are-inclusive",
        ),
        pytest.param(
            np.nextafter(-0.25, np.inf),
            np.nextafter(0.25, -np.inf),
            [-0.125, 0.0, 0.125],
            id="neighboring-grid-points-outside-hard-bounds-are-excluded",
        ),
        pytest.param(
            -0.75,
            0.75,
            [
                -0.75,
                -0.625,
                -0.5,
                -0.375,
                -0.25,
                -0.125,
                0.0,
                0.125,
                0.25,
                0.375,
                0.5,
                0.625,
                0.75,
            ],
            id="grid-is-not-limited-by-sampled-nyquist",
        ),
    ],
)
def test_direct_frequency_plan_applies_exact_inclusive_hard_bounds(
    f_min,
    f_max,
    expected_band,
):
    frequency_plan = resolve_timestamp_frequencies(
        f_min=f_min,
        f_max=f_max,
        window_duration=8.0,
    )

    assert isinstance(frequency_plan, DirectFrequencyPlan)
    assert frequency_plan.actual_df == 0.125
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        np.asarray(expected_band),
    )
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        frequency_plan.grid_indices.astype(np.float64) * frequency_plan.actual_df,
    )
    assert np.all(frequency_plan.band_frequencies >= f_min)
    assert np.all(frequency_plan.band_frequencies <= f_max)


def test_timestamp_grid_indices_give_compact_closing_frequency_identity():
    frequency_plan = resolve_timestamp_frequencies(
        f_min=-2.0,
        f_max=2.0,
        window_duration=10.0,
    )

    closing_indices = -(
        frequency_plan.grid_indices[:, None]
        + frequency_plan.grid_indices[None, :]
    )
    compact_indices = np.unique(closing_indices)
    expected_indices = np.arange(-40, 41, dtype=np.int64)

    assert compact_indices.size == 2 * frequency_plan.grid_indices.size - 1
    np.testing.assert_array_equal(compact_indices, expected_indices)
    np.testing.assert_array_equal(
        compact_indices.astype(np.float64) * frequency_plan.actual_df,
        expected_indices.astype(np.float64) * 0.1,
    )


def test_direct_frequency_plan_rejects_band_without_grid_frequency():
    with pytest.raises(ValueError, match="does not contain any timestamp frequencies"):
        resolve_timestamp_frequencies(
            f_min=0.1,
            f_max=0.2,
            window_duration=2.0,
        )


@pytest.mark.parametrize(
    ("observation_start", "observation_stop", "expected_start", "expected_stop"),
    [
        pytest.param(None, None, 0.0, 16.0, id="infer-both-bounds"),
        pytest.param(100.0, None, 100.0, 116.0, id="infer-stop"),
        pytest.param(None, 16.0, 0.0, 16.0, id="default-start"),
        pytest.param(100.0, 116.0, 100.0, 116.0, id="explicit-bounds"),
    ],
)
def test_runtime_config_resolves_sampled_observation_interval(
    observation_start,
    observation_stop,
    expected_start,
    expected_stop,
):
    data_config = DataConfig(
        channels=(SampledChannel(data=np.ones(64), dt=0.25),),
        observation_start=observation_start,
        observation_stop=observation_stop,
    )
    spectrum_config = SpectrumConfig(
        df=0.25,
        f_min=0.0,
        f_max=2.0,
        m=2,
        interlacing=False,
    )

    runtime = _build_runtime(data_config, spectrum_config, [(0,)])
    first_batch = next(iter_window_batches(runtime.window_plan))

    assert runtime.window_plan.observation_start == expected_start
    assert runtime.window_plan.observation_stop == expected_stop
    assert first_batch.relative_starts[0, 0] == 0.0


def test_runtime_config_rejects_sampled_observation_interval_with_wrong_duration():
    data_config = DataConfig(
        channels=(SampledChannel(data=np.ones(64), dt=0.25),),
        observation_start=100.0,
        observation_stop=115.0,
    )
    spectrum_config = SpectrumConfig(
        df=0.25,
        f_min=0.0,
        f_max=2.0,
        m=2,
        interlacing=False,
    )

    with pytest.raises(ValueError, match="configured observation interval has duration"):
        _build_runtime(data_config, spectrum_config, [(0,)])


def test_runtime_config_rejects_duration_error_hidden_by_relative_tolerance():
    data_config = DataConfig(
        channels=(SampledChannel(data=np.ones(1), dt=1e9),),
        observation_start=0.0,
        observation_stop=1e9 + 5e-4,
    )
    spectrum_config = SpectrumConfig(
        df=1e-9,
        f_min=0.0,
        f_max=5e-10,
        m=1,
        interlacing=False,
    )

    with pytest.raises(ValueError, match="configured observation interval has duration"):
        _build_runtime(data_config, spectrum_config, [(0,)])


def test_runtime_config_accepts_ulp_scale_observation_duration_rounding():
    data_config = DataConfig(
        channels=(SampledChannel(data=np.ones(3), dt=0.1),),
        observation_start=0.0,
        observation_stop=0.3,
    )
    spectrum_config = SpectrumConfig(
        df=10.0,
        f_min=0.0,
        f_max=5.0,
        m=1,
        interlacing=False,
    )

    runtime = _build_runtime(data_config, spectrum_config, [(0,)])

    assert runtime.window_plan.observation_stop == 0.3


@pytest.mark.parametrize(
    ("available_duration", "window_duration", "expected_count"),
    [
        pytest.param(10.0, 2.0, 5, id="exact-boundary"),
        pytest.param(
            np.nextafter(10.0, -np.inf),
            2.0,
            5,
            id="ulp-below-boundary",
        ),
        pytest.param(10.0 - 1e-6, 2.0, 4, id="physically-incomplete-window"),
        pytest.param(1.0, 2.0, 0, id="no-complete-window"),
    ],
)
def test_complete_window_count_uses_only_ulp_scale_boundary_tolerance(
    available_duration,
    window_duration,
    expected_count,
):
    assert _count_complete_windows(available_duration, window_duration) == expected_count


def test_runtime_config_keeps_m_for_exact_unshifted_fit_without_interlacing():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        interlacing=False,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(64),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.window_plan.windows_per_estimate == 4
    assert runtime.window_plan.unshifted_estimate_count == 1


def test_runtime_config_raises_when_interlacing_has_no_shifted_estimate():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        interlacing=True,
        spectral_estimates_max=None,
    )
    data_config = sampled_data_config(channels=(np.ones(64),), dt=1.0)

    with pytest.raises(ValueError, match="Interlacing was requested"):
        _build_runtime(data_config, spectrum_config, auto_spectra)


@pytest.mark.parametrize(
    ("auto_spectra_channels", "m"),
    [
        pytest.param([(0, 0)], 1, id="order-2-needs-at-least-two-windows"),
        pytest.param([(0, 0, 0)], 2, id="order-3-needs-at-least-three-windows"),
        pytest.param([(0, 0, 0, 0)], 3, id="order-4-needs-at-least-four-windows"),
    ],
)
def test_runtime_config_rejects_m_below_requested_order(auto_spectra_channels, m):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=20.0,
        df=20 / 15,
        m=m,
    )
    data_config = sampled_data_config(channels=(np.ones(50000),), dt=0.001)

    with pytest.raises(ValueError, match="Not enough data points"):
        _build_runtime(data_config, spectrum_config, auto_spectra_channels)


def test_runtime_config_defaults_to_all_auto_spectra_for_all_channels():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
    )
    data_config = sampled_data_config(
        channels=(
            np.ones(136),
            np.ones(136),
        ),
        dt=1.0,
    )

    runtime = _build_runtime(data_config, spectrum_config, None)

    assert runtime.active_data_channels == (0, 1)
    assert runtime.requested_spectra == (
        (0,),
        (0, 0),
        (0, 0, 0),
        (0, 0, 0, 0),
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
    )
    assert isinstance(runtime.fft_frequency_plan, FFTFrequencyPlan)
    assert runtime.direct_frequency_plan is None
    assert all(
        runtime.frequency_plan_for(channels) is runtime.fft_frequency_plan
        for channels in runtime.requested_spectra
    )


def test_runtime_config_validates_dt_for_active_sampled_channels_only():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.ones(136), dt=1.0),
            SampledChannel(data=np.ones(136), dt=2.0),
        )
    )
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=2)

    runtime = _build_runtime(data_config, spectrum_config, [(0, 0)])
    channel_plan = runtime.channel_plans[0]
    assert isinstance(channel_plan, SampledChannelPlan)
    assert channel_plan.dt == 1.0

    with pytest.raises(ValueError, match=r"Channel 1 has dt=2.0"):
        _build_runtime(data_config, spectrum_config, [(0, 1)])


def test_sampled_runtime_uses_neutral_repetition_plan():
    data_config = sampled_data_config(channels=(np.ones(136),), dt=1.0)
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=2)

    runtime = _build_runtime(data_config, spectrum_config, [(0, 0)])

    assert runtime.repetition_plan.count == 1
    assert runtime.repetition_plan.batch_size == 1
    assert runtime.repetition_plan.resolved_seed is None


def test_sampled_only_runtime_rejects_photon_options():
    data_config = sampled_data_config(channels=(np.ones(136),), dt=1.0)
    spectrum_config = SpectrumConfig(
        df=0.125,
        f_min=0.0,
        f_max=0.5,
        m=2,
        photon_options=PhotonOptions(weighting="unit"),
    )

    with pytest.raises(ValueError, match="sampled-only calculation"):
        _build_runtime(data_config, spectrum_config, [(0, 0)])


def test_active_timestamped_channel_requires_photon_options():
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),),
        observation_start=0.0,
        observation_stop=2.0,
    )
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=2)

    with pytest.raises(ValueError, match="required when an active channel is timestamped"):
        _build_runtime(data_config, spectrum_config, [(0,)])


@pytest.mark.parametrize(
    ("observation_start", "observation_stop"),
    [
        pytest.param(None, 2.0, id="missing-start"),
        pytest.param(0.0, None, id="missing-stop"),
        pytest.param(None, None, id="missing-both"),
    ],
)
def test_active_timestamped_channel_requires_explicit_observation_bounds(
    observation_start,
    observation_stop,
):
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),),
        observation_start=observation_start,
        observation_stop=observation_stop,
    )
    spectrum_config = SpectrumConfig(
        df=0.125,
        f_min=0.0,
        f_max=0.5,
        m=2,
        photon_options=PhotonOptions(weighting="unit"),
    )

    with pytest.raises(ValueError, match="require explicit observation_start"):
        _build_runtime(data_config, spectrum_config, [(0,)])


def test_timestamp_runtime_preserves_large_integral_observation_origin():
    origin = 2**60 + 1
    data_config = DataConfig(
        channels=(
            TimestampedChannel(
                timestamps=np.array([origin, origin + 1], dtype=np.int64),
            ),
        ),
        observation_start=origin,
        observation_stop=origin + 8,
    )
    spectrum_config = SpectrumConfig(
        df=1.0,
        f_min=0.0,
        f_max=1.0,
        m=2,
        photon_options=PhotonOptions(weighting="unit"),
    )

    runtime = _build_runtime(data_config, spectrum_config, [(0,)])

    assert runtime.window_plan.observation_start == origin
    assert runtime.window_plan.observation_stop == origin + 8
    assert type(runtime.window_plan.observation_start) is int
    assert type(runtime.window_plan.observation_stop) is int


@pytest.mark.parametrize(
    ("df", "f_max"),
    [
        pytest.param(None, 0.5, id="missing-df"),
        pytest.param(0.125, None, id="missing-f-max"),
        pytest.param(None, None, id="missing-both"),
    ],
)
def test_timestamp_only_runtime_requires_explicit_frequency_grid(df, f_max):
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),),
        observation_start=0.0,
        observation_stop=2.0,
    )
    spectrum_config = SpectrumConfig(
        df=df,
        f_min=0.0,
        f_max=f_max,
        m=2,
        photon_options=PhotonOptions(weighting="unit"),
    )

    with pytest.raises(ValueError, match="require explicit df and f_max"):
        _build_runtime(data_config, spectrum_config, [(0,)])


def test_mixed_runtime_rejects_interval_that_does_not_match_sampled_duration():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.ones(64), dt=0.25),
            TimestampedChannel(timestamps=np.array([100.0, 101.0])),
        ),
        observation_start=100.0,
        observation_stop=115.0,
    )
    spectrum_config = SpectrumConfig(
        df=0.25,
        f_min=0.0,
        f_max=2.0,
        m=2,
        photon_options=PhotonOptions(weighting="unit"),
    )

    with pytest.raises(ValueError, match="configured observation interval has duration"):
        _build_runtime(data_config, spectrum_config, [(0, 1)])


def test_configured_timestamped_channel_may_remain_inactive():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.ones(136), dt=1.0),
            TimestampedChannel(timestamps=np.array([0.0, 1.0])),
        )
    )
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=2)

    runtime = _build_runtime(data_config, spectrum_config, [(0, 0)])

    assert runtime.active_data_channels == (0,)
    assert tuple(runtime.channel_plans) == (0,)


def test_timestamp_only_runtime_uses_complete_event_free_tail_windows():
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),),
        observation_start=0.0,
        observation_stop=10.0,
    )
    spectrum_config = SpectrumConfig(
        df=0.5,
        f_min=-0.5,
        f_max=1.0,
        m=1,
        interlacing=True,
        spectral_estimates_max=None,
        photon_options=PhotonOptions(weighting="unit"),
    )

    runtime = _build_runtime(data_config, spectrum_config, [(0,)])
    frequency_plan = runtime.frequency_plan_for((0,))
    batches = list(iter_window_batches(runtime.window_plan))

    assert isinstance(frequency_plan, DirectFrequencyPlan)
    np.testing.assert_array_equal(
        frequency_plan.band_frequencies,
        [-0.5, 0.0, 0.5, 1.0],
    )
    assert runtime.window_plan.duration == 2.0
    assert runtime.window_plan.interlacing_offset == 1.0
    assert runtime.window_plan.unshifted_estimate_count == 5
    assert runtime.window_plan.shifted_estimate_count == 4
    assert physical_estimate_count(runtime.window_plan) == 9

    unshifted_starts = np.concatenate(
        [batch.relative_starts.ravel() for batch in batches if not batch.shifted]
    )
    shifted_starts = np.concatenate(
        [batch.relative_starts.ravel() for batch in batches if batch.shifted]
    )
    np.testing.assert_array_equal(unshifted_starts, [0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_array_equal(shifted_starts, [1.0, 3.0, 5.0, 7.0])


def test_mixed_runtime_assigns_per_spectrum_views_and_sampled_odd_offset():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.ones(10), dt=1.0),
            TimestampedChannel(timestamps=np.array([0.0, 1.0])),
        ),
        observation_start=0.0,
        observation_stop=10.0,
    )
    spectrum_config = SpectrumConfig(
        df=1.0 / 3.0,
        f_min=-1.0,
        f_max=1.0,
        m=2,
        interlacing=True,
        spectral_estimates_max=None,
        photon_options=PhotonOptions(weighting="unit"),
    )

    runtime = _build_runtime(
        data_config,
        spectrum_config,
        [(1, 1), (0, 1), (0, 0)],
    )

    timestamp_plan = runtime.frequency_plan_for((1, 1))
    mixed_plan = runtime.frequency_plan_for((0, 1))
    sampled_plan = runtime.frequency_plan_for((0, 0))

    assert isinstance(timestamp_plan, DirectFrequencyPlan)
    assert isinstance(mixed_plan, FFTFrequencyPlan)
    assert timestamp_plan is runtime.direct_frequency_plan
    assert mixed_plan is runtime.fft_frequency_plan
    assert mixed_plan is sampled_plan
    assert timestamp_plan.band_frequencies[0] == -1.0
    assert timestamp_plan.band_frequencies[-1] == 1.0
    assert mixed_plan.band_frequencies[0] > -1.0
    assert mixed_plan.band_frequencies[-1] < 1.0
    assert runtime.window_plan.duration == 3.0
    assert runtime.window_plan.interlacing_offset == 1.0
    assert runtime.window_plan.unshifted_estimate_count == 1
    assert runtime.window_plan.shifted_estimate_count == 1

    batches = list(iter_window_batches(runtime.window_plan))
    np.testing.assert_array_equal(batches[0].relative_starts, [[0.0, 3.0]])
    np.testing.assert_array_equal(batches[1].relative_starts, [[1.0, 4.0]])


@pytest.mark.parametrize(
    ("photon_options", "expected_weighting", "expected_scale"),
    [
        pytest.param(
            PhotonOptions(weighting="unit"),
            "unit",
            None,
            id="unit",
        ),
        pytest.param(
            PhotonOptions(
                weighting="exponential",
                scale=1.5,
                repetitions=4,
                seed=123,
            ),
            "exponential",
            1.5,
            id="exponential",
        ),
    ],
)
def test_timestamped_channel_plan_owns_amplitude_instructions(
    photon_options,
    expected_weighting,
    expected_scale,
):
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),),
    )

    with open_channels(data_config, (0,)) as opened_channels:
        channel_plans, _ = _build_channel_plans(
            data_config=data_config,
            opened_channels=opened_channels,
            photon_options=photon_options,
        )

    channel_plan = channel_plans[0]
    assert isinstance(channel_plan, TimestampedChannelPlan)
    assert channel_plan.event_count == 2
    assert channel_plan.weighting == expected_weighting
    assert channel_plan.scale == expected_scale


@pytest.mark.parametrize(
    "photon_options",
    [
        None,
        PhotonOptions(weighting="unit"),
    ],
)
def test_repetition_plan_is_neutral_without_exponential_weighting(photon_options):
    plan = _resolve_repetition_plan(photon_options)

    assert plan.count == 1
    assert plan.batch_size == 1
    assert plan.resolved_seed is None


def test_exponential_repetition_plan_preserves_count_and_explicit_seed():
    options = PhotonOptions(
        weighting="exponential",
        scale=1.0,
        repetitions=4,
        seed=0,
    )

    plan = _resolve_repetition_plan(options)

    assert plan.count == 4
    assert plan.batch_size == 4
    assert plan.resolved_seed == 0


def test_exponential_repetition_plan_bounds_internal_batch_size():
    repetition_count = _MAX_AMPLITUDE_REPETITIONS_PER_BATCH + 1
    options = PhotonOptions(
        weighting="exponential",
        scale=1.0,
        repetitions=repetition_count,
        seed=123,
    )

    plan = _resolve_repetition_plan(options)

    assert plan.count == repetition_count
    assert plan.batch_size == _MAX_AMPLITUDE_REPETITIONS_PER_BATCH


def test_exponential_repetition_plan_resolves_one_63_bit_seed(monkeypatch):
    calls = []

    def fake_randbits(bit_count):
        calls.append(bit_count)
        return 456

    monkeypatch.setattr(
        "signalsnap_pytorch._core.planning.secrets.randbits",
        fake_randbits,
    )
    options = PhotonOptions(
        weighting="exponential",
        scale=1.0,
        repetitions=2,
    )

    plan = _resolve_repetition_plan(options)

    assert calls == [63]
    assert plan.resolved_seed == 456


def test_runtime_config_rejects_unequal_active_sampled_lengths():
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.ones(136), dt=1.0),
            SampledChannel(data=np.ones(128), dt=1.0),
        )
    )
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=2)

    with pytest.raises(ValueError, match=r"Channel 1 contains 128 samples"):
        _build_runtime(data_config, spectrum_config, [(0, 1)])


def test_runtime_config_rejects_out_of_bounds_spectra_channel_indices():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
    )
    data_config = sampled_data_config(channels=(np.ones(136),), dt=1.0)

    with pytest.raises(ValueError, match="out of bounds"):
        _build_runtime(data_config, spectrum_config, [(1,)])


@pytest.mark.parametrize(
    ("requested_spectra", "exception_type", "message"),
    [
        pytest.param([], ValueError, "at least one spectrum", id="empty-request"),
        pytest.param([[]], TypeError, "must be a tuple", id="non-tuple-spectrum"),
        pytest.param([()], ValueError, "between 1 and 4", id="order-zero"),
        pytest.param([(0, 0, 0, 0, 0)], ValueError, "between 1 and 4", id="order-five"),
        pytest.param([(True,)], TypeError, "must be integers", id="boolean-index"),
        pytest.param([(0.0,)], TypeError, "must be integers", id="float-index"),
        pytest.param([(-1,)], ValueError, "nonnegative", id="negative-index"),
        pytest.param([(2,)], ValueError, "out of bounds", id="out-of-bounds-index"),
        pytest.param(
            [(0,), (0,)], ValueError, "cannot contain duplicates", id="duplicate-spectrum"
        ),
    ],
)
def test_resolve_requested_spectra_rejects_invalid_requests(
    requested_spectra,
    exception_type,
    message,
):
    with pytest.raises(exception_type, match=message):
        resolve_requested_spectra(requested_spectra, channel_count=2)


def test_resolve_requested_spectra_normalizes_numpy_integer_indices():
    requested_spectra = [(np.int64(0),), (np.int32(0), np.int64(1))]

    spectra_channels, active_data_channels = resolve_requested_spectra(
        requested_spectra,
        channel_count=2,
    )

    assert spectra_channels == ((0,), (0, 1))
    assert active_data_channels == (0, 1)
    assert all(type(channel) is int for spectrum in spectra_channels for channel in spectrum)


def test_resolve_requested_spectra_preserves_first_seen_active_data_channel_order():
    spectra_channels, active_data_channels = resolve_requested_spectra(
        [(2, 0), (1, 2, 1)],
        channel_count=3,
    )

    assert spectra_channels == ((2, 0), (1, 2, 1))
    assert active_data_channels == (2, 0, 1)


def test_resolve_requested_spectra_rejects_missing_data_channels():
    with pytest.raises(ValueError, match="At least one.*channel"):
        resolve_requested_spectra(None, channel_count=0)


def test_resolve_device_accepts_cpu():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _resolve_device("cuda")


def test_resolve_device_accepts_existing_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert _resolve_device("cuda:1") == torch.device("cuda:1")


def test_resolve_device_rejects_missing_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        _resolve_device("cuda:1")


def test_resolve_plain_cuda_to_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert _resolve_device("cuda") == torch.device("cuda:1")


def test_resolve_device_rejects_unavailable_xpu(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="XPU is not available"):
        _resolve_device("xpu")


def test_resolve_device_accepts_existing_xpu_index(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)

    assert _resolve_device("xpu:1") == torch.device("xpu:1")


def test_resolve_device_rejects_missing_xpu_index(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="only 1 XPU device"):
        _resolve_device("xpu:1")


def test_resolve_plain_xpu_to_current_device(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)

    assert _resolve_device("xpu") == torch.device("xpu:1")


def test_runtime_config_auto_precision_uses_single_on_xpu(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)

    runtime = _build_runtime(
        sampled_data_config(channels=(np.ones(136),), dt=1.0),
        SpectrumConfig(
            f_min=0.0,
            f_max=0.5,
            df=0.0625,
            m=4,
            device="xpu",
        ),
        auto_spectra,
    )

    assert runtime.device == torch.device("xpu:0")
    assert runtime.real_dtype == torch.float32
    assert runtime.complex_dtype == torch.complex64
