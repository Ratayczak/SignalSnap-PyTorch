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
from signalsnap_pytorch._core import timestamps as _timestamps
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH, sampled_data_config

_XPU_AVAILABLE = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()


@pytest.mark.skipif(not _XPU_AVAILABLE, reason="requires an available Intel XPU device")
def test_xpu_spectra_match_cpu_in_single_precision():
    rng = np.random.default_rng(42)
    samples = np.arange(512)
    channel_0 = np.sin(2 * np.pi * samples / 16) + 0.05 * rng.standard_normal(samples.size)
    channel_1 = 0.5 * channel_0 + 0.05 * rng.standard_normal(samples.size)
    data_config = sampled_data_config(channels=(channel_0, channel_1), dt=1.0)
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


@pytest.mark.parametrize("case", ["timestamp-unit", "mixed-exponential"])
@pytest.mark.skipif(not _XPU_AVAILABLE, reason="requires an available Intel XPU device")
def test_xpu_timestamp_spectra_and_cpu_amplitudes_match(case, monkeypatch):
    physical_window_count = 12
    patterns = (
        (),
        (0.2,),
        (0.3, 0.3),
        (0.1, 0.7),
        (0.5,),
        (0.25, 0.75),
    )

    def timestamps(offset):
        return np.asarray(
            [
                window_index + relative_time + offset
                for window_index in range(physical_window_count)
                for relative_time in patterns[window_index % len(patterns)]
            ]
        )

    if case == "timestamp-unit":
        data_config = DataConfig(
            channels=(
                TimestampedChannel(timestamps=timestamps(0.0)),
                TimestampedChannel(timestamps=timestamps(0.05)),
            ),
            observation_start=0.0,
            observation_stop=float(physical_window_count),
        )
        requested_spectra = [(0,), (0, 1), (1, 0, 1), (0, 1, 0, 1)]
        photon_options = PhotonOptions(weighting="unit")
    else:
        sample_times = np.arange(physical_window_count * 4)
        data_config = DataConfig(
            channels=(
                SampledChannel(
                    data=np.sin(2.0 * np.pi * sample_times / 8.0),
                    dt=0.25,
                ),
                TimestampedChannel(timestamps=timestamps(0.0)),
            ),
            observation_start=0.0,
            observation_stop=float(physical_window_count),
        )
        requested_spectra = [(1,), (0, 1), (0, 1, 1), (0, 1, 0, 1)]
        photon_options = PhotonOptions(
            weighting="exponential",
            scale=1.25,
            repetitions=3,
            seed=8642,
        )

    recorded_amplitudes = {"cpu": [], "xpu": []}
    original_materializer = _timestamps.materialize_timestamp_event_amplitudes

    def recording_materializer(
        prepared,
        channel_index,
        channel_plan,
        realization_ids,
        runtime,
    ):
        amplitudes = original_materializer(
            prepared,
            channel_index,
            channel_plan,
            realization_ids,
            runtime,
        )
        recorded_amplitudes[runtime.device.type].append(
            (
                channel_index,
                tuple(realization_ids),
                prepared.global_event_indices.copy(),
                amplitudes.copy(),
            )
        )
        return amplitudes

    monkeypatch.setattr(
        _timestamps,
        "materialize_timestamp_event_amplitudes",
        recording_materializer,
    )
    common_config = {
        "df": 1.0,
        "f_min": -1.0,
        "f_max": 1.0,
        "m": 4,
        "precision": "single",
        "spectral_estimates_max": 2,
        "spectral_estimates_per_batch": TEST_SPECTRAL_ESTIMATES_PER_BATCH,
        "photon_options": photon_options,
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

    assert len(recorded_amplitudes["xpu"]) == len(recorded_amplitudes["cpu"])
    assert recorded_amplitudes["cpu"]

    for cpu_record, xpu_record in zip(
        recorded_amplitudes["cpu"],
        recorded_amplitudes["xpu"],
    ):
        assert xpu_record[:2] == cpu_record[:2]
        np.testing.assert_array_equal(xpu_record[2], cpu_record[2])
        np.testing.assert_array_equal(xpu_record[3], cpu_record[3])

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
        np.testing.assert_allclose(
            xpu_result.spectrum_uncertainty,
            cpu_result.spectrum_uncertainty,
            rtol=2e-4,
            atol=2e-5,
            equal_nan=True,
        )
