from typing import Literal

import numpy as np
import pytest

from signalsnap_pytorch import DataConfig, SpectrumConfig, calculate_spectra

REQUESTED_SPECTRA = [
    (0,),
    (0, 1),
    (0, 1, 0),
    (0, 1, 0, 1),
]

UncertaintyEstimation = Literal["global", "short_term"]


def _calculate_with_batch_size(
    data_config: DataConfig,
    *,
    batch_size: int,
    uncertainty_estimation: UncertaintyEstimation,
):
    return calculate_spectra(
        data_config,
        SpectrumConfig(
            f_min=-0.25,
            f_max=0.5,
            df=0.125,
            m=4,
            uncertainty_estimation=uncertainty_estimation,
            m_var=4,
            spectral_estimates_max=5,
            spectral_estimates_per_batch=batch_size,
            interlacing=True,
        ),
        requested_spectra=REQUESTED_SPECTRA,
        show_progress=False,
    )


@pytest.mark.parametrize("uncertainty_estimation", ["global", "short_term"])
@pytest.mark.parametrize("batch_size", [3, 10])
def test_pipeline_batching_matches_single_estimate_batches(
    uncertainty_estimation: UncertaintyEstimation,
    batch_size: int,
):
    rng = np.random.default_rng(4815)
    data_config = DataConfig(
        channels=(
            rng.normal(size=170),
            rng.normal(size=170),
        ),
        dt=1.0,
    )

    expected = _calculate_with_batch_size(
        data_config,
        batch_size=1,
        uncertainty_estimation=uncertainty_estimation,
    )
    actual = _calculate_with_batch_size(
        data_config,
        batch_size=batch_size,
        uncertainty_estimation=uncertainty_estimation,
    )

    for channels in REQUESTED_SPECTRA:
        expected_result = expected[channels]
        actual_result = actual[channels]

        np.testing.assert_array_equal(actual_result.freq, expected_result.freq)
        assert actual_result.freq_unit == expected_result.freq_unit
        assert actual_result.channels == expected_result.channels
        np.testing.assert_allclose(
            actual_result.spectrum,
            expected_result.spectrum,
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )

        assert actual_result.spectrum_uncertainty is not None
        assert expected_result.spectrum_uncertainty is not None
        np.testing.assert_allclose(
            actual_result.spectrum_uncertainty,
            expected_result.spectrum_uncertainty,
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )
