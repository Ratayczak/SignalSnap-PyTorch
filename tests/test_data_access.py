import h5py
import numpy as np
import pytest

from signalsnap_pytorch import DataConfig, HDF5Channel
from signalsnap_pytorch._core.data_access import (
    HDF5ChannelState,
    get_sample_count,
    open_channels,
    read_channel,
)


def _open_all_channels(config):
    return open_channels(config, range(len(config.channels)))


def test_array_sample_count():
    channel = np.arange(10)

    assert get_sample_count(channel) == 10


def test_array_read():
    channel = np.arange(10)

    result = read_channel(channel, 2, 6)

    np.testing.assert_array_equal(result, np.array([2, 3, 4, 5]))
    assert result.flags.c_contiguous


def test_array_empty_read():
    result = read_channel(np.arange(10), 4, 4)

    assert result.shape == (0,)


def test_array_read_accepts_numpy_integer_bounds():
    result = read_channel(np.arange(10), np.int64(2), np.int32(6))  # type: ignore

    np.testing.assert_array_equal(result, np.array([2, 3, 4, 5]))


@pytest.mark.parametrize(
    ("start", "stop", "exception"),
    [
        (-1, 3, ValueError),
        (5, 4, ValueError),
        (0, 11, ValueError),
        (True, 4, TypeError),
        (0, False, TypeError),
        (1.5, 4, TypeError),
        (0, 4.5, TypeError),
    ],
)
def test_read_channel_rejects_invalid_range(start, stop, exception):
    with pytest.raises(exception):
        read_channel(np.arange(10), start, stop)


@pytest.fixture
def hdf5_file(tmp_path):
    values = np.arange(4 * 6 * 3, dtype=np.float64).reshape(4, 6, 3)

    path = tmp_path / "signals.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=values)

    return path, values


def test_open_channels_builds_hdf5_state(hdf5_file):
    path, _ = hdf5_file

    config = DataConfig(
        channels=(
            HDF5Channel(file=path, dataset="/signals", selection=(slice(None), slice(None), 1)),
        ),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        assert len(channels) == 1
        assert isinstance(channels[0], HDF5ChannelState)
        assert channels[0].selected_shape == (4, 6)
        assert get_sample_count(channels[0]) == 24


@pytest.mark.parametrize(
    ("start", "stop"),
    [(0, 0), (0, 3), (1, 5), (4, 9), (6, 12), (5, 19), (0, 24), (18, 24)],
)
def test_flattened_hdf5_reads_match_numpy(hdf5_file, start, stop):
    path, values = hdf5_file

    config = DataConfig(
        channels=(
            HDF5Channel(file=path, dataset="/signals", selection=(slice(None), slice(None), 1)),
        ),
        dt=1.0,
    )

    expected = values[:, :, 1].reshape(-1)[start:stop]

    with _open_all_channels(config) as channels:
        actual = read_channel(channels[0], start, stop)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_all_flattened_hdf5_read_ranges_match_numpy(hdf5_file):
    path, values = hdf5_file
    config = DataConfig(
        channels=(
            HDF5Channel(file=path, dataset="/signals", selection=(slice(None), slice(None), 1)),
        ),
        dt=1.0,
    )
    expected = values[:, :, 1].reshape(-1)

    with _open_all_channels(config) as channels:
        for start in range(expected.size + 1):
            for stop in range(start, expected.size + 1):
                actual = read_channel(channels[0], start, stop)

                np.testing.assert_array_equal(actual, expected[start:stop])
                assert actual.flags.c_contiguous


def test_hdf5_read_respects_selection_offsets(tmp_path):
    values = np.arange(10 * 20 * 2).reshape(10, 20, 2)
    path = tmp_path / "offsets.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=values)

    config = DataConfig(
        channels=(
            HDF5Channel(file=path, dataset="/signals", selection=(slice(2, 8), slice(5, 15), 1)),
        ),
        dt=1.0,
    )

    expected = values[2:8, 5:15, 1].reshape(-1)

    with _open_all_channels(config) as channels:
        actual = read_channel(channels[0], 7, 43)

    np.testing.assert_array_equal(actual, expected[7:43])


def test_one_dimensional_hdf5_selection_matches_numpy(hdf5_file):
    path, values = hdf5_file
    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(1, 4), 2, 1),
            ),
        ),
        dt=1.0,
    )
    expected = values[1:4, 2, 1]

    with _open_all_channels(config) as channels:
        actual = read_channel(channels[0], 1, 3)

    np.testing.assert_array_equal(actual, expected[1:3])


def test_negative_integer_and_slice_bounds_are_normalized(hdf5_file):
    path, values = hdf5_file
    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(-3, None), slice(-5, -1), -1),
            ),
        ),
        dt=1.0,
    )
    expected = values[-3:, -5:-1, -1].reshape(-1)

    with _open_all_channels(config) as channels:
        actual = read_channel(channels[0], 2, 10)

    np.testing.assert_array_equal(actual, expected[2:10])


def test_open_channels_preserves_array_and_hdf5_channel_order(hdf5_file):
    path, values = hdf5_file
    array = np.arange(24)
    config = DataConfig(
        channels=(
            array,
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(None), slice(None), 1),
            ),
        ),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        assert channels[0] is array
        assert isinstance(channels[1], HDF5ChannelState)
        np.testing.assert_array_equal(
            read_channel(channels[1], 0, 4), values[:, :, 1].reshape(-1)[:4]
        )


def test_open_channels_closes_hdf5_file_after_context(hdf5_file):
    path, _ = hdf5_file
    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(None), slice(None), 1),
            ),
        ),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        dataset = channels[0].dataset
        assert dataset.id.valid

    assert not dataset.id.valid


def test_channels_from_same_file_share_file_handle(hdf5_file):
    path, _ = hdf5_file
    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(None), slice(None), 0),
            ),
            HDF5Channel(
                file=path,
                dataset="/signals",
                selection=(slice(None), slice(None), 1),
            ),
        ),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        assert channels[0].dataset.file.id == channels[1].dataset.file.id


@pytest.mark.parametrize(
    ("selection", "exception", "message"),
    [
        ((slice(None),), ValueError, "dataset has 3 dimensions"),
        ((slice(None), slice(None), 3), IndexError, "out of bounds"),
        ((slice(None), slice(None), -4), IndexError, "out of bounds"),
        ((0, 0, 0), ValueError, "selects a scalar"),
        ((slice(0, 0), slice(None), 0), ValueError, "selection is empty"),
        ((slice(None), slice(None), slice(None)), ValueError, "at most two"),
    ],
)
def test_open_channels_rejects_invalid_hdf5_selection(hdf5_file, selection, exception, message):
    path, _ = hdf5_file
    config = DataConfig(
        channels=(HDF5Channel(file=path, dataset="/signals", selection=selection),),
        dt=1.0,
    )

    with pytest.raises(exception, match=message), _open_all_channels(config):
        pass


def test_open_channels_rejects_missing_dataset(hdf5_file):
    path, _ = hdf5_file
    config = DataConfig(
        channels=(HDF5Channel(file=path, dataset="/missing", selection=(slice(None),)),),
        dt=1.0,
    )

    with pytest.raises(KeyError, match="does not exist"), _open_all_channels(config):
        pass


def test_open_channels_rejects_missing_file(tmp_path):
    config = DataConfig(
        channels=(
            HDF5Channel(
                file=tmp_path / "missing.h5",
                dataset="/signals",
                selection=(slice(None),),
            ),
        ),
        dt=1.0,
    )

    with pytest.raises(OSError), _open_all_channels(config):
        pass


def test_open_channels_rejects_group_instead_of_dataset(tmp_path):
    path = tmp_path / "group.h5"
    with h5py.File(path, "w") as file:
        file.create_group("/signals")
    config = DataConfig(
        channels=(HDF5Channel(file=path, dataset="/signals", selection=(slice(None),)),),
        dt=1.0,
    )

    with pytest.raises(TypeError, match="is not a dataset"), _open_all_channels(config):
        pass


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([1 + 2j]), "Complex HDF5 datasets"),
        (np.array(["a", "b"], dtype="S1"), "is not numeric"),
    ],
)
def test_open_channels_rejects_unsupported_dataset_dtype(tmp_path, values, message):
    path = tmp_path / "dtype.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=values)
    config = DataConfig(
        channels=(HDF5Channel(file=path, dataset="/signals", selection=(slice(None),)),),
        dt=1.0,
    )

    with pytest.raises(TypeError, match=message), _open_all_channels(config):
        pass


def test_boolean_hdf5_dataset_is_supported(tmp_path):
    values = np.array([True, False, True])
    path = tmp_path / "bool.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("/signals", data=values)
    config = DataConfig(
        channels=(HDF5Channel(file=path, dataset="/signals", selection=(slice(None),)),),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        actual = read_channel(channels[0], 0, 3)

    np.testing.assert_array_equal(actual, values)


def test_read_channel_converts_to_native_byte_order(tmp_path):
    path = tmp_path / "big_endian.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signal", data=np.arange(10, dtype=">f8"))

    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signal",
                selection=(slice(None),),
            ),
        ),
        dt=1.0,
    )

    with _open_all_channels(config) as channels:
        result = read_channel(channels[0], 2, 6)

    assert result.dtype.isnative
    np.testing.assert_array_equal(result, [2, 3, 4, 5])


def test_two_dimensional_hdf5_read_converts_to_native_byte_order(tmp_path):
    values = np.arange(4 * 3 * 6, dtype=">f8").reshape(4, 3, 6)
    path = tmp_path / "big_endian_2d.h5"

    with h5py.File(path, "w") as file:
        file.create_dataset("/signal", data=values)

    config = DataConfig(
        channels=(
            HDF5Channel(
                file=path,
                dataset="/signal",
                selection=(slice(None), 1, slice(None)),
            ),
        ),
        dt=1.0,
    )
    expected = values[:, 1, :].reshape(-1)

    with _open_all_channels(config) as channels:
        result = read_channel(channels[0], 1, expected.size - 1)

    assert result.dtype.isnative
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, expected[1:-1])


def test_hdf5_channel_normalizes_numpy_integer_selection():
    channel = HDF5Channel(
        file="data.h5",  # type: ignore
        dataset="/signals",
        selection=(slice(None), np.int64(2)),
    )

    assert channel.selection == (slice(None), 2)
    assert type(channel.selection[1]) is int
