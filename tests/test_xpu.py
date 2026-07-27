import numpy as np
import pytest
import torch

from signalsnap_pytorch import DataConfig, SpectrumConfig, calculate_spectra
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH

_XPU_AVAILABLE = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()


@pytest.mark.skipif(not _XPU_AVAILABLE, reason="requires an available Intel XPU device")
def test_xpu_spectra_match_cpu_in_single_precision():
    rng = np.random.default_rng(42)
    samples = np.arange(512)
    channel_0 = np.sin(2 * np.pi * samples / 16) + 0.05 * rng.standard_normal(samples.size)
    channel_1 = 0.5 * channel_0 + 0.05 * rng.standard_normal(samples.size)
    data_config = DataConfig(channels=(channel_0, channel_1), dt=1.0)
    requested_spectra = [(0,), (0, 1), (0, 1, 0), (0, 1, 0, 1)]

    common_config = {
        "f_min": -0.25,
        "f_max": 0.25,
        "df": 0.0625,
        "m": 5,
        "precision": "single",
        "spectral_estimates_max": 3,
        "spectral_estimates_per_batch": TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    }
    cpu_results = calculate_spectra(
        data_config,
        SpectrumConfig(device="cpu", **common_config),
        requested_spectra=requested_spectra,
        show_progress=False,
    )
    xpu_results = calculate_spectra(
        data_config,
        SpectrumConfig(device="xpu", **common_config),
        requested_spectra=requested_spectra,
        show_progress=False,
    )

    assert tuple(result.channels for result in xpu_results) == tuple(requested_spectra)

    for channels in requested_spectra:
        cpu_result = cpu_results[channels]
        xpu_result = xpu_results[channels]

        np.testing.assert_array_equal(xpu_result.freq, cpu_result.freq)
        np.testing.assert_allclose(
            xpu_result.spectrum,
            cpu_result.spectrum,
            rtol=2e-4,
            atol=2e-5,
            equal_nan=True,
        )
        assert xpu_result.spectrum_uncertainty is not None
        assert cpu_result.spectrum_uncertainty is not None
        np.testing.assert_allclose(
            xpu_result.spectrum_uncertainty,
            cpu_result.spectrum_uncertainty,
            rtol=2e-4,
            atol=2e-5,
            equal_nan=True,
        )
