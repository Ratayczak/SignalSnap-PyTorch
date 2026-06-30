
import numpy as np
import pytest

from multichss.configurators import CrossConfig, DataConfig, SpectrumConfig
from multichss.fft import iter_window_slices
from multichss.pipelines import calculate_spectra
from multichss.planning import build_runtime_config


@pytest.mark.parametrize(
    (
        "n_data_points",
        "spectral_estimates_max",
        "orders",
        "frequency_points",
        "f_max",
        "m",
        "expected_available_estimates",
        "expected_m",
    ),
    [
        pytest.param(80, None, [1, 2], 9, 0.5, 4, 2, 4, id="uncapped"),
        pytest.param(80, 1, [1, 2], 9, 0.5, 4, 2, 4, id="capped-below-available"),
        pytest.param(128, 2, [1, 2], 9, 0.5, 4, 3, 4, id="cap-below-boundary"),
        pytest.param(136, 10, [1, 2], 9, 0.5, 4, 4, 4, id="cap-above-available"),
        pytest.param(136, 4, [1, 2], 9, 0.5, 4, 4, 4, id="cap-equals-available"),
        pytest.param(127, None, [1, 2], 9, 0.5, 4, 2, 4, id="one-before-next-base"),
        pytest.param(64, None, [1, 2], 9, 0.5, 4, 2, 3, id="m-reduced-at-short-boundary"),
        pytest.param(256, 3, [1, 2, 3, 4], 9, 0.5, 4, 7, 4, id="higher-orders-capped"),
        pytest.param(96, None, [1, 2], 6, 1 / 3, 3, 3, 3, id="odd-window-before-half-shift"),
        pytest.param(97, None, [1, 2], 6, 1 / 3, 3, 4, 3, id="odd-window-at-half-shift"),
    ],
)
def test_spectral_estimates_in_runtime_config(
    n_data_points,
    spectral_estimates_max,
    orders,
    frequency_points,
    f_max,
    m,
    expected_available_estimates,
    expected_m,
):
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=f_max,
        frequency_points=frequency_points,
        orders=orders,
        m=m,
        spectral_estimates_max=spectral_estimates_max,
    )
    data_config = DataConfig(data=np.ones(n_data_points), dt=1.0)

    runtime = build_runtime_config(spectrum_config, [data_config])
    available_estimates = len(list(iter_window_slices(runtime)))

    assert runtime.m == expected_m
    assert available_estimates == expected_available_estimates

    if spectral_estimates_max is None:
        expected_estimates = available_estimates
    else:
        expected_estimates = min(spectral_estimates_max, available_estimates)

    assert runtime.spectral_estimates == expected_estimates
    assert runtime.spectral_estimates <= available_estimates


def test_pipeline_processes_runtime_spectral_estimates():
    spectrum_config = SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        frequency_points=9,
        orders=[1, 2],
        m=4,
        spectral_estimates_max=3,
    )
    cross_config = CrossConfig(auto_corr=True)
    data_config = DataConfig(data=np.ones(128), dt=1.0)

    runtime = build_runtime_config(spectrum_config, [data_config])
    result_store = calculate_spectra(spectrum_config, cross_config, [data_config])

    assert runtime.spectral_estimates == 3
    for result in result_store.results.values():
        assert result.chunks_processed == runtime.spectral_estimates
