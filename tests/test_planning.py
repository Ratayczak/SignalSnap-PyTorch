from contextlib import nullcontext

import numpy as np
import pytest
import torch

from signalsnap_pytorch import DataConfig, SpectrumConfig, calculate_spectra
from signalsnap_pytorch._core.accumulation import initialize_accumulator_store
from signalsnap_pytorch._core.data_access import open_channels
from signalsnap_pytorch._core.planning import (
    _resolve_device,
    build_runtime_config,
    iter_window_slices,
    resolve_channels,
    resolve_frequencies,
)
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH

auto_spectra = [(0,), (0, 0)]


def _build_runtime(data_config, spectrum_config, requested_spectra):
    spectra_channels, active_data_channels = resolve_channels(
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
    data_config = DataConfig(channels=(np.ones(n_data_points),), dt=1.0)

    warning_context = (
        pytest.warns(UserWarning, match=f"using m={expected_m} instead")
        if expected_m != m
        else nullcontext()
    )

    with warning_context:
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra_channels)

    assert runtime.m == expected_m
    assert runtime.spectral_estimates == expected_unshifted_estimates


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
    data_config = DataConfig(channels=(np.ones(256),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.uncertainty_estimation == "short_term"
    assert runtime.m_var == 3
    assert runtime.spectral_estimates == 4
    assert runtime.spectral_estimates_per_batch == 2


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
    data_config = DataConfig(channels=(np.ones(10_000),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.spectral_estimates_per_batch == expected_batch_size
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
    data_config = DataConfig(channels=(np.ones(10_000),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=3 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.m_var == 3
    assert runtime.spectral_estimates_per_batch == 6


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
    data_config = DataConfig(channels=(np.ones(128),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=2 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.spectral_estimates == 2
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
    data_config = DataConfig(channels=(np.ones(640),), dt=1.0)

    with pytest.warns(UserWarning, match="using m_var=3 instead"):
        runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.spectral_estimates == 3
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
    data_config = DataConfig(channels=(np.ones(64),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.spectral_estimates == 1
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
    data_config = DataConfig(channels=(np.ones(128),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.uncertainty_estimation == "global"
    assert runtime.spectral_estimates == 2
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
    data_config = DataConfig(channels=(np.ones(256),), dt=1.0)
    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    store = initialize_accumulator_store(runtime)

    assert len(tuple(store)) == len(runtime.spectra_channels)
    for accumulator in store:
        assert accumulator.uncertainty_estimation == "short_term"
        assert accumulator.m_var == 3


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
    data_config = DataConfig(channels=(np.ones(n_data_points),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.interlacing is interlacing
    assert runtime.spectral_estimates == expected_spectral_estimates
    assert list(iter_window_slices(runtime)) == expected_slices


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
    data_config = DataConfig(channels=(np.ones(10_000),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)
    slices = list(iter_window_slices(runtime))

    assert [estimate_count for _, _, estimate_count, _ in slices] == [6, 2, 6, 2]
    assert [shifted for _, _, _, shifted in slices] == [False, False, True, True]
    assert sum(estimate_count for _, _, estimate_count, shifted in slices if not shifted) == 8
    assert sum(estimate_count for _, _, estimate_count, shifted in slices if shifted) == 8


def test_pipeline_returns_full_axis_third_order_spectrum_with_invalid_points_masked():
    spectrum_config = SpectrumConfig(
        f_min=-0.25,
        f_max=0.25,
        df=0.125,
        m=4,
        spectral_estimates_max=1,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    )
    data_config = DataConfig(channels=(np.ones(64),), dt=1.0)

    with pytest.warns(RuntimeWarning, match="at least two spectral estimates"):
        result_store = calculate_spectra(
            data_config, spectrum_config, requested_spectra=[(0, 0, 0)]
        )
    result = result_store[(0, 0, 0)]
    assert result is not None

    assert result.spectrum.shape == (result.freq.size, result.freq.size)

    assert spectrum_config.df is not None
    window_points = int(np.round(1 / (spectrum_config.df * data_config.dt)))
    full_fft_freq = np.fft.fftshift(np.fft.fftfreq(window_points, data_config.dt))
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
    data_config = DataConfig(
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


def test_resolve_frequencies_rejects_band_without_fft_frequency():
    spectrum_config = SpectrumConfig(
        f_min=0.1,
        f_max=0.2,
        df=0.5,
    )

    with pytest.raises(ValueError, match="does not contain any FFT frequencies"):
        resolve_frequencies(spectrum_config, dt=1.0)


def test_runtime_config_keeps_m_for_exact_unshifted_fit_without_interlacing():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        interlacing=False,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(channels=(np.ones(64),), dt=1.0)

    runtime = _build_runtime(data_config, spectrum_config, auto_spectra)

    assert runtime.m == 4
    assert runtime.spectral_estimates == 1


def test_runtime_config_raises_when_interlacing_has_no_shifted_estimate():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
        interlacing=True,
        spectral_estimates_max=None,
    )
    data_config = DataConfig(channels=(np.ones(64),), dt=1.0)

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
    data_config = DataConfig(channels=(np.ones(50000),), dt=0.001)

    with pytest.raises(ValueError, match="Not enough data points"):
        _build_runtime(data_config, spectrum_config, auto_spectra_channels)


def test_runtime_config_defaults_to_all_auto_spectra_for_all_channels():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
    )
    data_config = DataConfig(
        channels=(
            np.ones(136),
            np.ones(136),
        ),
        dt=1.0,
    )

    runtime = _build_runtime(data_config, spectrum_config, None)

    assert runtime.active_data_channels == (0, 1)
    assert runtime.spectra_channels == (
        (0,),
        (0, 0),
        (0, 0, 0),
        (0, 0, 0, 0),
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
    )


def test_runtime_config_rejects_out_of_bounds_spectra_channel_indices():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.0625,
        m=4,
    )
    data_config = DataConfig(channels=(np.ones(136),), dt=1.0)

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
def test_resolve_channels_rejects_invalid_requests(requested_spectra, exception_type, message):
    with pytest.raises(exception_type, match=message):
        resolve_channels(requested_spectra, channel_count=2)


def test_resolve_channels_normalizes_numpy_integer_indices():
    requested_spectra = [(np.int64(0),), (np.int32(0), np.int64(1))]

    spectra_channels, active_data_channels = resolve_channels(
        requested_spectra,
        channel_count=2,
    )

    assert spectra_channels == ((0,), (0, 1))
    assert active_data_channels == (0, 1)
    assert all(type(channel) is int for spectrum in spectra_channels for channel in spectrum)


def test_resolve_channels_preserves_first_seen_active_data_channel_order():
    spectra_channels, active_data_channels = resolve_channels(
        [(2, 0), (1, 2, 1)],
        channel_count=3,
    )

    assert spectra_channels == ((2, 0), (1, 2, 1))
    assert active_data_channels == (2, 0, 1)


def test_resolve_channels_rejects_missing_data_channels():
    with pytest.raises(ValueError, match="At least one.*channel"):
        resolve_channels(None, channel_count=0)


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
        DataConfig(channels=(np.ones(136),), dt=1.0),
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
