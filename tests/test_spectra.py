import numpy as np
import pytest

from signalsnap_pytorch import SpectrumConfig, calculate_spectra, pipelines
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH, sampled_data_config


def test_c1_returns_correct_mean():
    """The first-order cumulant equals the signal mean."""
    dt = 0.01
    samples_per_window = 1_000
    centered_samples = np.arange(samples_per_window) - (samples_per_window - 1) / 2
    window_signal = np.sin(2 * np.pi * 10 * centered_samples / samples_per_window) + 2
    signal = np.tile(window_signal, 20)

    data_config = sampled_data_config(channels=(signal,), dt=dt, t_unit="s")
    spectrum_config = SpectrumConfig(
        f_min=0,
        f_max=2,
        device="cpu",
        df=0.1,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    )

    result = calculate_spectra(data_config, spectrum_config, requested_spectra=[(0,)])[(0,)]

    np.testing.assert_allclose(result.spectrum, np.asarray([2.0 + 0.0j]), atol=1e-12)


def test_c1_returns_mean_when_selected_band_excludes_dc():
    signal = np.full(10_000, 2.0)
    data_config = sampled_data_config(channels=(signal,), dt=0.01, t_unit="s")
    spectrum_config = SpectrumConfig(
        f_min=1,
        f_max=2,
        device="cpu",
        df=0.5,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    )

    result = calculate_spectra(data_config, spectrum_config, requested_spectra=[(0,)])[(0,)]

    np.testing.assert_allclose(result.spectrum, np.asarray([2.0 + 0.0j]), atol=1e-12)
    np.testing.assert_array_equal(result.freq, np.asarray([0.0]))


@pytest.mark.parametrize("interlacing, expected_total", [(False, 2), (True, 4)])
@pytest.mark.parametrize("show_progress", [True, False])
def test_calculate_spectra_reports_progress(
    monkeypatch, interlacing, expected_total, show_progress
):
    progress_call = {}
    progress_updates = []

    class RecordingProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def update(self, estimate_count):
            progress_updates.append(estimate_count)

    def recording_progress(**kwargs):
        progress_call.update(kwargs)
        return RecordingProgress()

    monkeypatch.setattr(pipelines, "tqdm", recording_progress)

    signal = np.ones(40)
    data_config = sampled_data_config(channels=(signal,), dt=1.0)
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=2,
        spectral_estimates_max=2,
        spectral_estimates_per_batch=2,
        interlacing=interlacing,
    )

    calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0,)],
        show_progress=show_progress,
    )

    assert progress_call == {
        "total": expected_total,
        "desc": "Calculating spectra",
        "unit": "estimate",
        "disable": not show_progress,
    }
    assert progress_updates == ([2, 2] if interlacing else [2])
    assert sum(progress_updates) == expected_total
