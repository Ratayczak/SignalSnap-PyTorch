from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from signalsnap_pytorch import DataConfig, HDF5Channel, SpectrumConfig


def test_spectrum_config_accepts_negative_frequency_band():
    SpectrumConfig(f_min=-1, f_max=1)


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False


def test_spectrum_config_defaults_to_automatic_frequency_spacing():
    assert SpectrumConfig().df is None


def test_spectrum_config_defaults_to_global_uncertainty_estimation():
    config = SpectrumConfig()

    assert config.uncertainty_estimation == "global"
    assert config.m_var == 10


def test_spectrum_config_defaults_to_one_spectral_estimate_per_batch():
    assert SpectrumConfig().spectral_estimates_per_batch == 1


def test_spectrum_config_accepts_multiple_spectral_estimates_per_batch():
    assert SpectrumConfig(spectral_estimates_per_batch=4).spectral_estimates_per_batch == 4


@pytest.mark.parametrize(
    ("batch_size", "expected_error"),
    [
        pytest.param(0, ValidationError, id="zero"),
        pytest.param(-1, ValidationError, id="negative"),
        pytest.param(1.5, ValidationError, id="non-integer"),
        pytest.param(True, TypeError, id="boolean"),
    ],
)
def test_spectrum_config_rejects_invalid_spectral_estimates_per_batch(
    batch_size,
    expected_error,
):
    with pytest.raises(expected_error, match="spectral_estimates_per_batch"):
        SpectrumConfig(spectral_estimates_per_batch=batch_size)


def test_spectrum_config_accepts_short_term_uncertainty_estimation():
    config = SpectrumConfig(uncertainty_estimation="short_term", m_var=4)

    assert config.uncertainty_estimation == "short_term"
    assert config.m_var == 4


def test_spectrum_config_rejects_unknown_uncertainty_estimation():
    with pytest.raises(ValidationError, match="uncertainty_estimation"):
        SpectrumConfig(uncertainty_estimation="local")


@pytest.mark.parametrize("m_var", [1, 0, -1, 1.5, True])
def test_spectrum_config_rejects_invalid_m_var(m_var):
    with pytest.raises(ValidationError, match="m_var"):
        SpectrumConfig(uncertainty_estimation="short_term", m_var=m_var)


def test_spectrum_config_accepts_positive_frequency_spacing():
    assert SpectrumConfig(df=0.125).df == 0.125


@pytest.mark.parametrize("df", [0.0, -0.125, np.inf, np.nan])
def test_spectrum_config_rejects_invalid_frequency_spacing(df):
    with pytest.raises(ValidationError, match="df"):
        SpectrumConfig(df=df)


def test_data_config_accepts_array_channels():
    config = DataConfig(
        channels=(np.arange(10), np.arange(10) * 2),
        dt=0.1,
    )

    assert len(config.channels) == 2
    assert isinstance(config.channels, tuple)
    assert config.dt == 0.1
    assert config.t_unit == "s"


def test_data_config_accepts_array_and_hdf5_channels():
    hdf5_channel = HDF5Channel(
        file=Path("data.h5"),
        dataset="/signals",
        selection=(slice(None), slice(None), 0),
    )

    config = DataConfig(
        channels=(np.arange(10), hdf5_channel),
        dt=0.1,
    )

    assert len(config.channels) == 2
    assert config.channels[1] == hdf5_channel


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ([], "at least 1 item"),
        ([None], "cannot be None"),
        ([object()], "shape attribute"),
        ([np.ones((2, 3))], "one-dimensional"),
        ([np.array([])], "cannot be empty"),
        ([np.array([1 + 2j])], "cannot be complex"),
        ([np.array(["a", "b"])], "must be numeric"),
        ([np.array([object()], dtype=object)], "must be numeric"),
    ],
)
def test_data_config_rejects_invalid_array_channels(channels, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        DataConfig(channels=channels, dt=1.0)


@pytest.mark.parametrize(
    ("dataset", "selection", "message"),
    [
        ("", (slice(None),), "dataset cannot be empty"),
        ("/signals", (), "selection cannot be empty"),
        ("/signals", (True,), "integers or slices"),
        ("/signals", ("channel-0",), "integers or slices"),
        ("/signals", (slice(None, None, 2),), "steps other than 1"),
        ("/signals", (slice(None, None, -1),), "steps other than 1"),
    ],
)
def test_hdf5_channel_rejects_invalid_configuration(dataset, selection, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        HDF5Channel(
            file=Path("data.h5"),
            dataset=dataset,
            selection=selection,
        )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("cuda:1", "cuda:1"),
        ("xpu", "xpu"),
        ("xpu:0", "xpu:0"),
        ("xpu:1", "xpu:1"),
    ],
)
def test_spectrum_config_accepts_supported_devices(device, expected):
    config = SpectrumConfig(device=device)

    assert config.device == expected


@pytest.mark.parametrize(
    "device",
    [
        "",
        "CPU",
        "cuda:",
        "cuda:01",
        "cuda:-1",
        "cuda:abc",
        "cudafoo",
        "xpu:",
        "xpu:01",
        "xpu:-1",
        "xpu:abc",
        "xpufoo",
        "cpu:0",
        "mps:0",
        "xla",
        "meta",
    ],
)
def test_spectrum_config_rejects_unsupported_devices(device):
    with pytest.raises(ValidationError, match="device|device type|numbered"):
        SpectrumConfig(device=device)
