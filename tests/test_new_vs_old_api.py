from __future__ import annotations

import h5py
import numpy as np
import pytest

from signalsnap_pytorch import DataConfig, SpectrumConfig, calculate_spectra
from tests._helpers import align_legacy_spectrum_region

AUTO_CHANNELS = [
    (0,),
    (0, 0),
    (0, 0, 0),
    (0, 0, 0, 0),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, 1, 1, 1),
]
CROSS_CHANNELS_24 = [(0, 1), (1, 0), (1, 0, 0, 1), (1, 1, 0, 0)]
CROSS_CHANNELS_3 = [(0, 1, 1), (1, 0, 0), (0, 0, 1)]

LEGACY_CASES = [
    pytest.param(
        "auto",
        "./tests/test_data/references/5Qubit_short_data_auto_corr.npz",
        AUTO_CHANNELS,
        id="auto",
    ),
    pytest.param(
        "cross_ch24",
        "./tests/test_data/references/5Qubit_short_data_cross_corr_ch124.npz",
        CROSS_CHANNELS_24,
        id="cross_ch24",
    ),
    pytest.param(
        "cross_ch3",
        "./tests/test_data/references/5Qubit_short_data_cross_corr_ch3.npz",
        CROSS_CHANNELS_3,
        id="cross_ch3",
    ),
]


@pytest.fixture(scope="module")
def prepared_data():
    with h5py.File("./tests/test_data/datasets/5Qubit_short_data.h5", "r") as file:
        dataset = file["/X_test"]
        assert isinstance(dataset, h5py.Dataset)
        data = dataset[...]

    return DataConfig(
        channels=(
            data[:1000, :, 0].reshape(-1),
            data[:1000, :, 1].reshape(-1),
        ),
        dt=2.0,
        t_unit="ns",
    )


def _build_spectrum_config(name: str, channels, legacy_freqs) -> SpectrumConfig:
    if name == "auto":
        legacy_s3_freq = np.asarray(legacy_freqs[0][3])
        return SpectrumConfig(
            f_min=float(legacy_s3_freq[0]),
            f_max=float(legacy_s3_freq[-1]),
            device="cpu",
            df=legacy_s3_freq[1] - legacy_s3_freq[0],
            interlacing=True,
            old_window=True,
        )

    if name == "cross_ch24":
        return SpectrumConfig(
            f_min=-0.25,
            f_max=0.25,
            device="cpu",
            df=0.5 / 99,
            interlacing=True,
            old_window=True,
        )

    if name == "cross_ch3":
        legacy_s3_freq = np.asarray(legacy_freqs[channels[0]][3])
        return SpectrumConfig(
            f_min=float(legacy_s3_freq[0]),
            f_max=float(legacy_s3_freq[-1]),
            device="cpu",
            df=legacy_s3_freq[1] - legacy_s3_freq[0],
            interlacing=True,
            old_window=True,
        )

    raise AssertionError(f"Update test parameters to include a test for {name}")


def _calculate_legacy_case(name, reference_file, channels, prepared_data):
    with np.load(reference_file, allow_pickle=True) as benchmark:
        legacy_spectra = benchmark["spectra"].item()
        legacy_uncertainties = benchmark["error"].item()
        legacy_freqs = benchmark["freqs"].item()

    spectrum_config = _build_spectrum_config(name, channels, legacy_freqs)
    result_store = calculate_spectra(prepared_data, spectrum_config, requested_spectra=channels)
    return result_store, legacy_spectra, legacy_uncertainties, legacy_freqs


def _legacy_channel_key(name: str, channels: tuple[int, ...]):
    return channels[0] if name == "auto" else channels


def _expected_current_freq(name: str, legacy_key, order: int, legacy_freqs) -> np.ndarray:
    if order == 1:
        return np.asarray([0.0])
    if name == "auto":
        return np.asarray(legacy_freqs[legacy_key][3])
    return np.asarray(legacy_freqs[legacy_key][order])


@pytest.mark.parametrize(("name", "reference_file", "channels"), LEGACY_CASES)
def test_new_api_matches_legacy_spectra(name, reference_file, channels, prepared_data):
    """The modular pipeline reproduces stored legacy spectra for auto- and cross-correlation."""
    result_store, legacy_spectra, _, legacy_freqs = _calculate_legacy_case(
        name, reference_file, channels, prepared_data
    )

    for channel_tuple in channels:
        order = len(channel_tuple)
        legacy_key = _legacy_channel_key(name, channel_tuple)
        result = result_store[channel_tuple]
        expected_freq = _expected_current_freq(name, legacy_key, order, legacy_freqs)
        np.testing.assert_allclose(result.freq, expected_freq, rtol=0.0, atol=1e-12)

        expected_shape = {
            1: (1,),
            2: (expected_freq.size,),
            3: (expected_freq.size, expected_freq.size),
            4: (expected_freq.size, expected_freq.size),
        }[order]
        assert result.spectrum.shape == expected_shape

        spectrum_result = np.asarray(result.spectrum)
        if order == 3:
            spectrum_result = spectrum_result.transpose()

        actual_spectrum, expected_spectrum = align_legacy_spectrum_region(
            spectrum_result,
            expected_freq,
            np.asarray(legacy_spectra[legacy_key][order]),
            np.asarray(legacy_freqs[legacy_key][order]),
            order,
        )
        assert actual_spectrum.shape == expected_spectrum.shape

        np.testing.assert_allclose(
            actual_spectrum,
            expected_spectrum,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Spectrum at order {order} for channel {channel_tuple} doesn't match.",
        )


@pytest.mark.skip(reason="Uncertainty calculation was intentionally redesigned in new API")
@pytest.mark.parametrize(("name", "reference_file", "channels"), LEGACY_CASES)
def test_new_api_uncertainties_match_legacy(name, reference_file, channels, prepared_data):
    result_store, _, legacy_uncertainties, legacy_freqs = _calculate_legacy_case(
        name, reference_file, channels, prepared_data
    )

    for channel_tuple in channels:
        order = len(channel_tuple)
        legacy_key = _legacy_channel_key(name, channel_tuple)
        result = result_store[channel_tuple]
        assert result.spectrum_uncertainty is not None
        expected_freq = _expected_current_freq(name, legacy_key, order, legacy_freqs)
        np.testing.assert_allclose(result.freq, expected_freq, rtol=0.0, atol=1e-12)

        spectrum_uncertainty = np.asarray(result.spectrum_uncertainty)
        if order == 3:
            spectrum_uncertainty = spectrum_uncertainty.transpose()

        actual_uncertainty, expected_uncertainty = align_legacy_spectrum_region(
            spectrum_uncertainty,
            expected_freq,
            np.asarray(legacy_uncertainties[legacy_key][order]),
            np.asarray(legacy_freqs[legacy_key][order]),
            order,
        )
        np.testing.assert_allclose(
            actual_uncertainty,
            expected_uncertainty,
            rtol=1e-6,
            atol=1e-8,
            err_msg=(
                f"Spectrum uncertainty at order {order} for channel {channel_tuple} doesn't match."
            ),
        )
