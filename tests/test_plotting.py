import h5py
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from signalsnap_pytorch import DataConfig, HDF5Source, SpectrumConfig, TimestampedChannel
from signalsnap_pytorch.plotting import create_first_window_figure
from tests._helpers import sampled_data_config


@pytest.fixture
def first_window_spectrum_config():
    # dt=1, f_min=0, f_max=0.5 and five frequency points produce an eight-sample window.
    return SpectrumConfig(
        f_min=0.0,
        f_max=0.5,
        df=0.125,
    )


def test_create_first_window_figure_plots_all_array_channels(
    first_window_spectrum_config,
):
    channel_0 = np.arange(12, dtype=float)
    channel_1 = np.arange(12, dtype=float) + 100
    data_config = sampled_data_config(
        channels=(channel_0, channel_1),
        dt=1.0,
        t_unit="ms",
    )

    figure = create_first_window_figure(
        data_config,
        first_window_spectrum_config,
    )

    try:
        assert len(figure.axes) == 2
        expected_time = np.arange(8, dtype=float)

        for channel, expected_data in enumerate((channel_0, channel_1)):
            axis = figure.axes[channel]
            assert len(axis.lines) == 1
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), expected_time)
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), expected_data[:8])
            assert axis.get_title() == f"First window for channel {channel}"
            assert axis.get_ylabel() == "Amplitude"

        assert figure.axes[-1].get_xlabel() == "t / ms"
    finally:
        plt.close(figure)


def test_create_first_window_figure_respects_selected_channel_order(
    first_window_spectrum_config,
):
    channels = (
        np.arange(12, dtype=float),
        np.arange(12, dtype=float) + 100,
        np.arange(12, dtype=float) + 200,
    )
    data_config = sampled_data_config(channels=channels, dt=1.0)

    figure = create_first_window_figure(
        data_config,
        first_window_spectrum_config,
        channels=[2, 0],
    )

    try:
        assert len(figure.axes) == 2
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), channels[2][:8])
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), channels[0][:8])
        assert figure.axes[0].get_title() == "First window for channel 2"
        assert figure.axes[1].get_title() == "First window for channel 0"
    finally:
        plt.close(figure)


def test_create_first_window_figure_supports_mixed_array_and_hdf5_channels(
    tmp_path,
    first_window_spectrum_config,
):
    stored = np.arange(4 * 10 * 2, dtype=float).reshape(4, 10, 2)
    path = tmp_path / "signals.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=stored)

    array_channel = np.arange(20, dtype=float) + 1000
    data_config = sampled_data_config(
        channels=(
            array_channel,
            HDF5Source(
                file=path,
                dataset="/signals",
                selection=(slice(1, 3), slice(1, 10), 1),
            ),
        ),
        dt=1.0,
    )
    expected_hdf5 = stored[1:3, 1:10, 1].reshape(-1)

    figure = create_first_window_figure(
        data_config,
        first_window_spectrum_config,
    )

    try:
        np.testing.assert_array_equal(
            figure.axes[0].lines[0].get_ydata(),
            array_channel[:8],
        )
        np.testing.assert_array_equal(
            figure.axes[1].lines[0].get_ydata(),
            expected_hdf5[:8],
        )
    finally:
        plt.close(figure)

    # The plotting context must release the file after reading the first window.
    with h5py.File(path, "r+") as file:
        assert "/signals" in file


@pytest.mark.parametrize(
    ("selected_channels", "exception", "message"),
    [
        ([], ValueError, "At least one channel"),
        ([0, 0], ValueError, "selected more than once"),
        ([2], ValueError, "out of bounds"),
        ([-1], ValueError, "nonnegative"),
        ([True], TypeError, "must be integers"),
        ([0.0], TypeError, "must be integers"),
    ],
)
def test_create_first_window_figure_rejects_invalid_channel_selection(
    first_window_spectrum_config,
    selected_channels,
    exception,
    message,
):
    data_config = sampled_data_config(
        channels=(np.arange(12), np.arange(12)),
        dt=1.0,
    )

    with pytest.raises(exception, match=message):
        create_first_window_figure(
            data_config,
            first_window_spectrum_config,
            channels=selected_channels,
        )


def test_create_first_window_figure_rejects_short_array_channel(
    first_window_spectrum_config,
):
    data_config = sampled_data_config(channels=(np.arange(7),), dt=1.0)

    with pytest.raises(
        ValueError,
        match=r"Channel 0 contains 7 samples, but one window requires 8",
    ):
        create_first_window_figure(data_config, first_window_spectrum_config)


def test_create_first_window_figure_rejects_timestamped_channel(
    first_window_spectrum_config,
):
    data_config = DataConfig(
        channels=(TimestampedChannel(timestamps=np.array([0.0, 1.0])),)
    )

    with pytest.raises(TypeError, match="supports only sampled channels"):
        create_first_window_figure(data_config, first_window_spectrum_config)


def test_create_first_window_figure_rejects_short_hdf5_channel(
    tmp_path,
    first_window_spectrum_config,
):
    path = tmp_path / "short.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=np.arange(7))

    data_config = sampled_data_config(
        channels=(
            HDF5Source(
                file=path,
                dataset="/signals",
                selection=(slice(None),),
            ),
        ),
        dt=1.0,
    )

    with pytest.raises(
        ValueError,
        match=r"Channel 0 contains 7 samples, but one window requires 8",
    ):
        create_first_window_figure(data_config, first_window_spectrum_config)


def test_create_first_window_figure_does_not_open_unselected_hdf5_channel(
    tmp_path,
    first_window_spectrum_config,
):
    array_channel = np.arange(12, dtype=float)
    data_config = sampled_data_config(
        channels=(
            array_channel,
            HDF5Source(
                file=tmp_path / "missing.h5",
                dataset="/signals",
                selection=(slice(None),),
            ),
        ),
        dt=1.0,
    )

    figure = create_first_window_figure(
        data_config,
        first_window_spectrum_config,
        channels=[0],
    )

    try:
        np.testing.assert_array_equal(
            figure.axes[0].lines[0].get_ydata(),
            array_channel[:8],
        )
    finally:
        plt.close(figure)
