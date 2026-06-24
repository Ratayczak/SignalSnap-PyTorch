# This code tries to compute the same spectrum in the new API as the old version in
# https://github.com/ArminGEtemad/SignalSnap-PyTorch/blob/main/Examples/Straightforward_Example.ipynb

import numpy as np
from multichss.configurators import DataConfig, SpectrumConfig, CrossConfig
from multichss.pipelines import calculate_spectra


def test_straightforward_example_matches_old_spectra():
    data = np.load("./tests/test_data/signals/straightforward_example_signal.npz")
    x = data["x"][0:1000000000]
    m = data["m"][0:1000000000]

    dconfig1 = DataConfig(data=x, dt=0.001, t_unit="us")
    dconfig2 = DataConfig(data=m, dt=0.001, t_unit="us")
    selected_data = [0, 1]

    sconfig = SpectrumConfig(
        f_min=0,
        f_max=5,
        s3_calc="1/4",
        backend="cuda",
        order_in=[1, 2, 3, 4],
        spectrum_size=1000,
        show_first_frame=False,
    )

    cconfig = CrossConfig(auto_corr=True)

    result_store = calculate_spectra(
        sconfig, cconfig, [dconfig1, dconfig2], selected=selected_data
    )
    result1 = result_store.get((0,), 1)
    result2 = result_store.get((0, 0), 2)
    result3 = result_store.get((0, 0, 0), 3)
    result4 = result_store.get((0, 0, 0, 0), 4)

    benchmark_spectra = np.load(
        "./tests/test_data/spectra/straightforward_example_spectrum.npz"
    )
    old_spectra = benchmark_spectra["spectra"]
    old_error = benchmark_spectra["error"]
    old_freqs = benchmark_spectra["freqs"]

    assert result1.spectrum is not None
    assert result2.spectrum is not None
    assert result3.spectrum is not None
    assert result4.spectrum is not None

    np.testing.assert_allclose(
        np.asarray(result1.spectrum),
        np.asarray(old_spectra[0][1]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result2.spectrum),
        np.asarray(old_spectra[0][2]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result3.spectrum),
        np.asarray(old_spectra[0][3]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result4.spectrum),
        np.asarray(old_spectra[0][4]),
        rtol=1e-6,
        atol=1e-8,
    )

    assert result1.spectrum_error is not None
    assert result2.spectrum_error is not None
    assert result3.spectrum_error is not None
    assert result4.spectrum_error is not None

    np.testing.assert_allclose(
        np.asarray(result1.spectrum_error),
        np.asarray(old_error[0][1]),
        rtol=1e-6,
        atol=1e-8,
        err_msg="First-order spectrum error doesn't match",
    )

    np.testing.assert_allclose(
        np.asarray(result2.spectrum_error),
        np.asarray(old_error[0][2]),
        rtol=1e-6,
        atol=1e-8,
        err_msg="Second-order spectrum error doesn't match",
    )

    np.testing.assert_allclose(
        np.asarray(result3.spectrum_error),
        np.asarray(old_error[0][3]),
        rtol=1e-6,
        atol=1e-8,
        err_msg="Third-order spectrum error doesn't match",
    )

    np.testing.assert_allclose(
        np.asarray(result4.spectrum_error),
        np.asarray(old_error[0][4]),
        rtol=1e-6,
        atol=1e-8,
        err_msg="Fourth-order spectrum error doesn't match",
    )

    assert result1.freq is not None
    assert result2.freq is not None
    assert result3.freq is not None
    assert result4.freq is not None

    np.testing.assert_allclose(
        np.asarray(result1.freq), np.asarray(old_freqs[0][1]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result2.freq), np.asarray(old_freqs[0][2]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result3.freq), np.asarray(old_freqs[0][3]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result4.freq), np.asarray(old_freqs[0][4]), rtol=0, atol=1e-12
    )

    result1_ch1 = result_store.get((1,), 1)
    result2_ch1 = result_store.get((1, 1), 2)
    result3_ch1 = result_store.get((1, 1, 1), 3)
    result4_ch1 = result_store.get((1, 1, 1, 1), 4)

    assert result1_ch1.spectrum is not None
    assert result2_ch1.spectrum is not None
    assert result3_ch1.spectrum is not None
    assert result4_ch1.spectrum is not None

    np.testing.assert_allclose(
        np.asarray(result1_ch1.spectrum),
        np.asarray(old_spectra[1][1]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result2_ch1.spectrum),
        np.asarray(old_spectra[1][2]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result3_ch1.spectrum),
        np.asarray(old_spectra[1][3]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result4_ch1.spectrum),
        np.asarray(old_spectra[1][4]),
        rtol=1e-6,
        atol=1e-8,
    )

    assert result1_ch1.spectrum_error is not None
    assert result2_ch1.spectrum_error is not None
    assert result3_ch1.spectrum_error is not None
    assert result4_ch1.spectrum_error is not None

    np.testing.assert_allclose(
        np.asarray(result1_ch1.spectrum_error),
        np.asarray(old_error[1][1]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result2_ch1.spectrum_error),
        np.asarray(old_error[1][2]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result3_ch1.spectrum_error),
        np.asarray(old_error[1][3]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result4_ch1.spectrum_error),
        np.asarray(old_error[1][4]),
        rtol=1e-6,
        atol=1e-8,
    )

    assert result1_ch1.freq is not None
    assert result2_ch1.freq is not None
    assert result3_ch1.freq is not None
    assert result4_ch1.freq is not None

    np.testing.assert_allclose(
        np.asarray(result1_ch1.freq), np.asarray(old_freqs[1][1]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result2_ch1.freq), np.asarray(old_freqs[1][2]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result3_ch1.freq), np.asarray(old_freqs[1][3]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(result4_ch1.freq), np.asarray(old_freqs[1][4]), rtol=0, atol=1e-12
    )
