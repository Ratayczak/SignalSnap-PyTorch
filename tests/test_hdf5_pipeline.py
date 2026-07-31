import h5py
import numpy as np
import pytest

from signalsnap_pytorch import (
    DataConfig,
    HDF5Source,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    calculate_spectra,
)
from tests._helpers import TEST_SPECTRAL_ESTIMATES_PER_BATCH, sampled_data_config

REQUESTED_SPECTRA = [
    (0,),
    (0, 0),
    (0, 1),
    (0, 1, 0),
    (0, 1, 0, 1),
]


def _assert_result_stores_equal(actual_store, expected_store):
    for channels in REQUESTED_SPECTRA:
        actual = actual_store[channels]
        expected = expected_store[channels]

        np.testing.assert_array_equal(actual.freq, expected.freq)
        assert actual.freq_unit == expected.freq_unit
        assert actual.channels == expected.channels
        np.testing.assert_allclose(
            actual.spectrum,
            expected.spectrum,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

        if expected.spectrum_uncertainty is None:
            assert actual.spectrum_uncertainty is None
        else:
            assert actual.spectrum_uncertainty is not None
            np.testing.assert_allclose(
                actual.spectrum_uncertainty,
                expected.spectrum_uncertainty,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )


def test_hdf5_and_mixed_channels_match_eager_array_pipeline(tmp_path):
    rng = np.random.default_rng(12345)
    stored = rng.normal(size=(10, 64, 3))
    path = tmp_path / "signals.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=stored, chunks=(2, 64, 1))

    eager_channels = (
        stored[1:9, :, 0].reshape(-1),
        stored[1:9, :, 1].reshape(-1),
    )
    eager_config = sampled_data_config(channels=eager_channels, dt=0.1, t_unit="s")
    hdf5_config = sampled_data_config(
        channels=(
            HDF5Source(
                file=path,
                dataset="/signals",
                selection=(slice(1, 9), slice(None), 0),
            ),
            HDF5Source(
                file=path,
                dataset="/signals",
                selection=(slice(1, 9), slice(None), 1),
            ),
        ),
        dt=0.1,
        t_unit="s",
    )
    mixed_config = sampled_data_config(
        channels=(
            hdf5_config.channels[0],
            eager_channels[1],
        ),
        dt=0.1,
        t_unit="s",
    )
    spectrum_config = SpectrumConfig(
        f_min=-2.5,
        f_max=2.5,
        df=0.625,
        m=4,
        spectral_estimates_max=3,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
        interlacing=True,
    )

    expected = calculate_spectra(
        eager_config,
        spectrum_config,
        requested_spectra=REQUESTED_SPECTRA,
    )

    for actual_config in (hdf5_config, mixed_config):
        actual = calculate_spectra(
            actual_config,
            spectrum_config,
            requested_spectra=REQUESTED_SPECTRA,
        )
        _assert_result_stores_equal(actual, expected)


def test_pipeline_does_not_open_unrequested_hdf5_channel(tmp_path):
    data_config = sampled_data_config(
        channels=(
            np.arange(128, dtype=float),
            HDF5Source(
                file=tmp_path / "missing.h5",
                dataset="/signals",
                selection=(slice(None),),
            ),
        ),
        dt=1.0,
    )
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
        m=4,
        spectral_estimates_per_batch=TEST_SPECTRAL_ESTIMATES_PER_BATCH,
    )

    result_store = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 0)],
    )

    assert result_store[(0, 0)].channels == (0, 0)

def test_sampled_pipeline_ignores_unrequested_timestamped_hdf5_source(tmp_path):
    data_config = DataConfig(
        channels=(
            SampledChannel(data=np.arange(128, dtype=float), dt=1.0),
            TimestampedChannel(
                timestamps=HDF5Source(
                    file=tmp_path / "missing.h5",
                    dataset="/timestamps",
                    selection=(slice(None),),
                )
            ),
        )
    )
    spectrum_config = SpectrumConfig(df=0.125, f_min=0.0, f_max=0.5, m=4)

    result_store = calculate_spectra(
        data_config,
        spectrum_config,
        requested_spectra=[(0, 0)],
        show_progress=False,
    )

    assert result_store[(0, 0)].channels == (0, 0)

    with pytest.raises(NotImplementedError, match="Timestamped channel"):
        calculate_spectra(
            data_config,
            spectrum_config,
            requested_spectra=[(1,)],
            show_progress=False,
        )
