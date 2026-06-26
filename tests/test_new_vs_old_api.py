# This code tries to compute the same spectrum in the new API as the old version in
# https://github.com/ArminGEtemad/SignalSnap-PyTorch/blob/main/Examples/Straightforward_Example.ipynb

import numpy as np
from multichss.configurators import DataConfig, SpectrumConfig, CrossConfig
from multichss.pipelines import calculate_spectra

import h5py


def test_new_vs_old_api_auto_corr_1():
    """
    Tests if the refactor implements the same calculations. To check if the
    implementation is correct, the old window function needs to be used in
    SpectrumConfig.
    """
    with h5py.File(
        "./tests/test_data/datasets/5Qubit_short_data.h5", "r"
    ) as f:
        x_test_dataset = f["/X_test"]
        assert isinstance(x_test_dataset, h5py.Dataset)
        X_test = x_test_dataset[...]

    signal_channel_0 = X_test[:1000, :, 0].reshape(-1)
    signal_channel_1 = X_test[:1000, :, 1].reshape(-1)
    
    dconfig1 = DataConfig(data=signal_channel_0, dt=2.0, t_unit="ns")
    dconfig2 = DataConfig(data=signal_channel_1, dt=2.0, t_unit="ns")
    selected_data = [0, 1]

    sconfig = SpectrumConfig(
        f_min=0,
        f_max=0.25,
        s3_calc="1/4",
        backend="cpu",
        order_in=[1, 2, 3, 4],
        spectrum_size=100,
        show_first_frame=False,
        old_window=True,
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
        "./tests/test_data/references/5Qubit_short_data_auto_corr.npz", allow_pickle=True
    )
    old_spectra = benchmark_spectra["spectra"].item()
    # old_error = benchmark_spectra["error"].item()
    old_freqs = benchmark_spectra["freqs"].item()

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

    # Old error estimation was wrong
    # assert result1.spectrum_error is not None
    # assert result2.spectrum_error is not None
    # assert result3.spectrum_error is not None
    # assert result4.spectrum_error is not None

    # np.testing.assert_allclose(
    #     np.asarray(result1.spectrum_error),
    #     np.asarray(old_error[0][1]),
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg="First-order spectrum error doesn't match",
    # )

    # np.testing.assert_allclose(
    #     np.asarray(result2.spectrum_error),
    #     np.asarray(old_error[0][2]),
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg="Second-order spectrum error doesn't match",
    # )

    # np.testing.assert_allclose(
    #     np.asarray(result3.spectrum_error),
    #     np.asarray(old_error[0][3]),
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg="Third-order spectrum error doesn't match",
    # )

    # np.testing.assert_allclose(
    #     np.asarray(result4.spectrum_error),
    #     np.asarray(old_error[0][4]),
    #     rtol=1e-6,
    #     atol=1e-8,
    #     err_msg="Fourth-order spectrum error doesn't match",
    # )

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

    # Old error estimation was wrong
    # assert result1_ch1.spectrum_error is not None
    # assert result2_ch1.spectrum_error is not None
    # assert result3_ch1.spectrum_error is not None
    # assert result4_ch1.spectrum_error is not None

    # np.testing.assert_allclose(
    #     np.asarray(result1_ch1.spectrum_error),
    #     np.asarray(old_error[1][1]),
    #     rtol=1e-6,
    #     atol=1e-8,
    # )
    # np.testing.assert_allclose(
    #     np.asarray(result2_ch1.spectrum_error),
    #     np.asarray(old_error[1][2]),
    #     rtol=1e-6,
    #     atol=1e-8,
    # )
    # np.testing.assert_allclose(
    #     np.asarray(result3_ch1.spectrum_error),
    #     np.asarray(old_error[1][3]),
    #     rtol=1e-6,
    #     atol=1e-8,
    # )
    # np.testing.assert_allclose(
    #     np.asarray(result4_ch1.spectrum_error),
    #     np.asarray(old_error[1][4]),
    #     rtol=1e-6,
    #     atol=1e-8,
    # )

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


def test_new_vs_old_api_cross_corr_1():
    """
    Tests if the refactor implements the same calculations. To check if the
    implementation is correct, the old window function needs to be used in
    SpectrumConfig.
    """
    with h5py.File(
        "./tests/test_data/datasets/5Qubit_short_data.h5", "r"
    ) as f:
        x_test_dataset = f["/X_test"]
        assert isinstance(x_test_dataset, h5py.Dataset)
        X_test = x_test_dataset[...]

    signal_channel_0 = X_test[:1000, :, 0].reshape(-1)
    signal_channel_1 = X_test[:1000, :, 1].reshape(-1)

    dconfig1 = DataConfig(data=signal_channel_0, dt=2.0, t_unit="ns")
    dconfig2 = DataConfig(data=signal_channel_1, dt=2.0, t_unit="ns")
    selected_data = [0, 1]

    sconfig = SpectrumConfig(
        f_min=-0.25,
        f_max=0.25,
        s3_calc="1/4",
        backend="cpu",
        order_in=[1, 2, 4],
        spectrum_size=100,
        show_first_frame=False,
        old_window=True,
    )

    cconfig = CrossConfig(
        auto_corr=False,
        cross_corr_2=[(0, 1), (1, 0)],
        cross_corr_4=[(1, 0, 0, 1), (1, 1, 0, 0)],
    )

    result_store = calculate_spectra(
        sconfig, cconfig, [dconfig1, dconfig2], selected=selected_data
    )

    result_01_2 = result_store.get((0, 1), 2)
    result_10_2 = result_store.get((1, 0), 2)
    result_1001_4 = result_store.get((1, 0, 0, 1), 4)
    result_1100_4 = result_store.get((1, 1, 0, 0), 4)

    sconfig2 = SpectrumConfig(
        f_min=0,
        f_max=0.25,
        s3_calc="1/2",
        backend="cpu",
        order_in=[3],
        spectrum_size=100,
        show_first_frame=False,
        old_window=True,
    )

    cconfig2 = CrossConfig(
        auto_corr=False, cross_corr_3=[(0, 1, 1), (1, 0, 0), (0, 0, 1)]
    )

    result_store2 = calculate_spectra(
        sconfig2, cconfig2, [dconfig1, dconfig2], selected=selected_data
    )

    result_011_3 = result_store2.get((0, 1, 1), 3)
    result_100_3 = result_store2.get((1, 0, 0), 3)
    result_001_3 = result_store2.get((0, 0, 1), 3)

    benchmark_spectra_ch124 = np.load(
        "./tests/test_data/references/5Qubit_short_data_cross_corr_ch124.npz",
        allow_pickle=True,
    )
    old_spectra_ch124 = benchmark_spectra_ch124["spectra"].item()
    # old_error_ch124 = benchmark_spectra_ch124["error"].item()
    old_freqs_ch124 = benchmark_spectra_ch124["freqs"].item()

    benchmark_spectra_ch3 = np.load(
        "./tests/test_data/references/5Qubit_short_data_cross_corr_ch3.npz",
        allow_pickle=True,
    )
    old_spectra_ch3 = benchmark_spectra_ch3["spectra"].item()
    # old_error_ch3 = benchmark_spectra_ch3["error"].item()
    old_freqs_ch3 = benchmark_spectra_ch3["freqs"].item()

    cross_corr_results_ch124 = [
        ((0, 1), 2, result_01_2),
        ((1, 0), 2, result_10_2),
        ((1, 0, 0, 1), 4, result_1001_4),
        ((1, 1, 0, 0), 4, result_1100_4),
    ]

    for channels, order, result in cross_corr_results_ch124:
        assert result.spectrum is not None
        np.testing.assert_allclose(
            np.asarray(result.spectrum),
            np.asarray(old_spectra_ch124[channels][order]),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{order}-order spectrum {channels} doesn't match",
        )

        # Old error estimation was wrong
        # assert result.spectrum_error is not None
        # np.testing.assert_allclose(
        #     np.asarray(result.spectrum_error),
        #     np.asarray(old_error_ch124[channels][order]),
        #     rtol=1e-6,
        #     atol=1e-8,
        #     err_msg=f"{order}-order spectrum error {channels} doesn't match",
        # )

        assert result.freq is not None
        np.testing.assert_allclose(
            np.asarray(result.freq),
            np.asarray(old_freqs_ch124[channels][order]),
            rtol=0,
            atol=1e-12,
            err_msg=f"{order}-order frequencies {channels} don't match",
        )

    cross_corr_results_ch3 = [
        ((0, 1, 1), 3, result_011_3),
        ((1, 0, 0), 3, result_100_3),
        ((0, 0, 1), 3, result_001_3),
    ]

    for channels, order, result in cross_corr_results_ch3:
        assert result.spectrum is not None
        np.testing.assert_allclose(
            np.asarray(result.spectrum),
            np.asarray(old_spectra_ch3[channels][order]),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{order}-order spectrum {channels} doesn't match",
        )

        # Old error estimation was wrong
        # assert result.spectrum_error is not None
        # np.testing.assert_allclose(
        #     np.asarray(result.spectrum_error),
        #     np.asarray(old_error_ch3[channels][order]),
        #     rtol=1e-6,
        #     atol=1e-8,
        #     err_msg=f"{order}-order spectrum error {channels} doesn't match",
        # )

        assert result.freq is not None
        np.testing.assert_allclose(
            np.asarray(result.freq),
            np.asarray(old_freqs_ch3[channels][order]),
            rtol=0,
            atol=1e-12,
            err_msg=f"{order}-order frequencies {channels} don't match",
        )
