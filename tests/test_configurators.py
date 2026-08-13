from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from signalsnap_pytorch import (
    DataConfig,
    HDF5Source,
    TimestampOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
)


def test_timestamp_options_accept_unit_weighting_without_mark_fields():
    options = TimestampOptions(weighting="unit")

    assert options.weighting == "unit"
    assert options.scale is None
    assert options.repetitions is None
    assert options.seed is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("scale", 1.0, id="scale"),
        pytest.param("repetitions", 2, id="repetitions"),
        pytest.param("seed", 0, id="seed"),
    ],
)
def test_unit_timestamp_weighting_rejects_exponential_fields(field, value):
    with pytest.raises(ValueError, match="does not accept"):
        TimestampOptions(weighting="unit", **{field: value})


def test_timestamp_options_accept_exponential_weighting():
    options = TimestampOptions(
        weighting="exponential",
        scale=1.5,
        repetitions=100,
        seed=1234,
    )

    assert options.scale == 1.5
    assert options.repetitions == 100
    assert options.seed == 1234


def test_timestamp_options_normalizes_numpy_integer_fields():
    options = TimestampOptions(
        weighting="exponential",
        scale=1.0,
        repetitions=np.int64(100),
        repetitions_per_batch=np.uint8(5),
        seed=np.int32(1234),
    )

    assert options.repetitions == 100
    assert type(options.repetitions) is int
    assert options.repetitions_per_batch == 5
    assert type(options.repetitions_per_batch) is int
    assert options.seed == 1234
    assert type(options.seed) is int


@pytest.mark.parametrize(
    "missing_field",
    ["scale", "repetitions"],
)
def test_exponential_timestamp_weighting_requires_scale_and_repetitions(missing_field):
    values = {"scale": 1.0, "repetitions": 2}
    del values[missing_field]

    with pytest.raises(ValueError, match="requires scale and repetitions"):
        TimestampOptions(weighting="exponential", **values)


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, np.inf, np.nan, True],
)
def test_exponential_timestamp_weighting_rejects_invalid_scale(scale):
    with pytest.raises((TypeError, ValueError), match="scale|finite"):
        TimestampOptions(
            weighting="exponential",
            scale=scale,
            repetitions=2,
        )


@pytest.mark.parametrize(
    "repetitions",
    [0, -1, 1.5, "2", True],
)
def test_exponential_timestamp_weighting_requires_strict_positive_repetitions(repetitions):
    with pytest.raises((TypeError, ValueError), match="repetitions"):
        TimestampOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=repetitions,
        )


@pytest.mark.parametrize(
    "seed",
    [-1, 1.5, "2", True],
)
def test_exponential_timestamp_weighting_requires_strict_nonnegative_seed(seed):
    with pytest.raises((TypeError, ValueError), match="seed"):
        TimestampOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=2,
            seed=seed,
        )


def test_timestamp_options_are_frozen():
    options = TimestampOptions(weighting="unit")

    with pytest.raises(FrozenInstanceError):
        options.weighting = "exponential"


def test_spectrum_config_accepts_negative_frequency_band():
    SpectrumConfig(f_min=-1, f_max=1)


def test_spectrum_config_defaults_to_no_interlacing():
    assert SpectrumConfig().interlacing is False


def test_spectrum_config_defaults_to_automatic_frequency_spacing():
    assert SpectrumConfig().df is None


def test_spectrum_config_defaults_to_global_uncertainty_estimation():
    config = SpectrumConfig()

    assert config.uncertainty_estimation == "global"
    assert config.m_var == 10


def test_spectrum_config_defaults_to_one_spectral_estimate_per_batch():
    assert SpectrumConfig().spectral_estimates_per_batch == 1


def test_spectrum_config_accepts_multiple_spectral_estimates_per_batch():
    assert SpectrumConfig(spectral_estimates_per_batch=4).spectral_estimates_per_batch == 4


def test_spectrum_config_normalizes_numpy_integer_fields():
    config = SpectrumConfig(
        m=np.int64(4),
        m_var=np.uint8(3),
        spectral_estimates_max=np.int32(20),
        spectral_estimates_per_batch=np.uint16(2),
    )

    for field, expected in (
        ("m", 4),
        ("m_var", 3),
        ("spectral_estimates_max", 20),
        ("spectral_estimates_per_batch", 2),
    ):
        value = getattr(config, field)
        assert value == expected
        assert type(value) is int


@pytest.mark.parametrize(
    "field",
    [
        "df",
        "f_min",
        "f_max",
        "m",
        "m_var",
        "spectral_estimates_max",
        "spectral_estimates_per_batch",
    ],
)
def test_spectrum_config_rejects_numeric_strings(field):
    with pytest.raises(TypeError, match=field):
        SpectrumConfig(**{field: "2"})


@pytest.mark.parametrize("field", ["interlacing", "old_window"])
@pytest.mark.parametrize("value", [0, 1, "false", np.bool_(False)])
def test_spectrum_config_requires_python_boolean_flags(field, value):
    with pytest.raises(TypeError, match=field):
        SpectrumConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "df",
        "f_min",
        "f_max",
        "m",
        "m_var",
        "spectral_estimates_max",
        "spectral_estimates_per_batch",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(np.bool_(True), id="numpy-true"),
        pytest.param(np.bool_(False), id="numpy-false"),
    ],
)
def test_spectrum_config_rejects_boolean_numeric_fields(field, value):
    with pytest.raises(
        TypeError,
        match=rf"^{field} must be (?:an integer|a finite real number)\.$",
    ):
        SpectrumConfig(**{field: value})


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="non-integer"),
    ],
)
def test_spectrum_config_rejects_invalid_spectral_estimates_per_batch(batch_size):
    with pytest.raises((TypeError, ValueError), match="spectral_estimates_per_batch"):
        SpectrumConfig(spectral_estimates_per_batch=batch_size)


def test_spectrum_config_accepts_short_term_uncertainty_estimation():
    config = SpectrumConfig(uncertainty_estimation="short_term", m_var=4)

    assert config.uncertainty_estimation == "short_term"
    assert config.m_var == 4


def test_spectrum_config_rejects_unknown_uncertainty_estimation():
    with pytest.raises(ValueError, match="uncertainty_estimation"):
        SpectrumConfig(uncertainty_estimation="local")


@pytest.mark.parametrize("m_var", [1, 0, -1, 1.5])
def test_spectrum_config_rejects_invalid_m_var(m_var):
    with pytest.raises((TypeError, ValueError), match="m_var"):
        SpectrumConfig(uncertainty_estimation="short_term", m_var=m_var)


def test_spectrum_config_accepts_positive_frequency_spacing():
    assert SpectrumConfig(df=0.125).df == 0.125


@pytest.mark.parametrize("df", [0.0, -0.125, np.inf, np.nan])
def test_spectrum_config_rejects_invalid_frequency_spacing(df):
    with pytest.raises(ValueError, match="df"):
        SpectrumConfig(df=df)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.arange(8), id="numpy"),
        pytest.param(torch.arange(8), id="cpu-tensor"),
        pytest.param(np.array([True, False]), id="boolean"),
    ],
)
def test_sampled_channel_accepts_supported_in_memory_data_without_copying(data):
    channel = SampledChannel(data=data, dt=0.125)

    assert channel.data is data
    assert channel.dt == 0.125


def test_sampled_channel_accepts_hdf5_source_without_opening_it():
    source = HDF5Source(
        file=Path("missing.h5"),
        dataset="/signals",
        selection=(slice(None),),
    )

    channel = SampledChannel(data=source, dt=0.125)

    assert channel.data is source


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.array([1.0, np.nan]), id="numpy-nan"),
        pytest.param(np.array([1.0, np.inf]), id="numpy-positive-inf"),
        pytest.param(np.array([1.0, -np.inf]), id="numpy-negative-inf"),
        pytest.param(torch.tensor([1.0, float("nan")]), id="tensor-nan"),
        pytest.param(torch.tensor([1.0, float("inf")]), id="tensor-positive-inf"),
        pytest.param(torch.tensor([1.0, -float("inf")]), id="tensor-negative-inf"),
    ],
)
def test_sampled_channel_rejects_nonfinite_in_memory_data(data):
    with pytest.raises(ValueError, match="only finite values"):
        SampledChannel(data=data, dt=1.0)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan, True])
def test_sampled_channel_rejects_invalid_dt(dt):
    with pytest.raises((TypeError, ValueError), match="dt"):
        SampledChannel(data=np.arange(8), dt=dt)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param(None, "NumPy array", id="none"),
        pytest.param([1.0, 2.0], "NumPy array", id="list"),
        pytest.param(np.ones((2, 3)), "one-dimensional", id="numpy-2d"),
        pytest.param(torch.ones((2, 3)), "one-dimensional", id="tensor-2d"),
        pytest.param(np.array([]), "cannot be empty", id="numpy-empty"),
        pytest.param(torch.tensor([]), "cannot be empty", id="tensor-empty"),
        pytest.param(np.array([1 + 2j]), "real numeric", id="numpy-complex"),
        pytest.param(torch.tensor([1 + 2j]), "real numeric", id="tensor-complex"),
        pytest.param(np.array(["a"]), "real numeric", id="numpy-string"),
        pytest.param(
            np.ma.array([1.0, 999.0, 3.0], mask=[False, True, False]),
            "masked arrays",
            id="numpy-masked",
        ),
    ],
)
def test_sampled_channel_rejects_invalid_data(data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        SampledChannel(data=data, dt=1.0)


def test_sampled_channel_rejects_masked_array_even_when_no_values_are_masked():
    data = np.ma.array([1.0, 2.0, 3.0], mask=np.ma.nomask)

    with pytest.raises(TypeError, match="masked arrays"):
        SampledChannel(data=data, dt=1.0)


def test_sampled_channel_rejects_non_cpu_tensor_storage():
    with pytest.raises(ValueError, match="stored on the CPU"):
        SampledChannel(data=torch.empty(2, device="meta"), dt=1.0)


@pytest.mark.parametrize(
    "timestamps",
    [
        pytest.param(np.array([], dtype=np.float64), id="numpy-empty"),
        pytest.param(torch.tensor([], dtype=torch.float64), id="tensor-empty"),
        pytest.param(np.array([0.0, 0.25, 0.25, 1.0]), id="numpy-duplicates"),
        pytest.param(torch.tensor([0.0, 0.25, 0.25, 1.0]), id="tensor-duplicates"),
    ],
)
def test_timestamped_channel_accepts_empty_and_nondecreasing_data(timestamps):
    channel = TimestampedChannel(timestamps=timestamps)

    assert channel.timestamps is timestamps


def test_timestamped_channel_accepts_hdf5_source_without_opening_it():
    source = HDF5Source(
        file="missing.h5",
        dataset="/timestamps",
        selection=(slice(None),),
    )

    channel = TimestampedChannel(timestamps=source)

    assert channel.timestamps is source


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        pytest.param([0.0, 1.0], "NumPy array", id="list"),
        pytest.param(np.ones((2, 2)), "one-dimensional", id="two-dimensional"),
        pytest.param(np.array([False, True]), "real numeric", id="boolean"),
        pytest.param(np.array([0.0, np.nan]), "finite", id="nan"),
        pytest.param(np.array([0.0, np.inf]), "finite", id="infinity"),
        pytest.param(np.array([0.5, 0.25]), "nondecreasing", id="unordered"),
        pytest.param(
            np.ma.array([0.0, 999.0, 1.0], mask=[False, True, False]),
            "masked arrays",
            id="numpy-masked",
        ),
        pytest.param(torch.empty(2, device="meta"), "stored on the CPU", id="non-cpu"),
    ],
)
def test_timestamped_channel_rejects_invalid_timestamps(timestamps, message):
    with pytest.raises((TypeError, ValueError), match=message):
        TimestampedChannel(timestamps=timestamps)


def test_explicit_data_models_are_frozen_without_freezing_referenced_arrays():
    data = np.arange(8)
    sampled = SampledChannel(data=data, dt=1.0)
    timestamped = TimestampedChannel(timestamps=np.array([0.0, 1.0]))
    source = HDF5Source(file="data.h5", dataset="/x", selection=(slice(None),))

    with pytest.raises(FrozenInstanceError):
        sampled.dt = 2.0
    with pytest.raises(FrozenInstanceError):
        timestamped.timestamps = np.array([])
    with pytest.raises(FrozenInstanceError):
        source.dataset = "/y"

    data[0] = 42
    assert sampled.data[0] == 42


def test_data_config_accepts_explicit_heterogeneous_channels_and_bounds():
    sampled = SampledChannel(data=np.arange(8), dt=0.25)
    timestamped = TimestampedChannel(timestamps=np.array([0.0, 1.0]))

    config = DataConfig(
        channels=(sampled, timestamped),
        observation_start=0.0,
        observation_stop=2.0,
        t_unit="ms",
    )

    assert config.channels == (sampled, timestamped)
    assert config.observation_start == 0.0
    assert config.observation_stop == 2.0
    assert config.t_unit == "ms"


def test_data_config_normalizes_channel_lists_to_tuples():
    channel = SampledChannel(data=np.arange(8), dt=0.25)

    config = DataConfig(channels=[channel])

    assert config.channels == (channel,)
    assert type(config.channels) is tuple


@pytest.mark.parametrize(
    "channel",
    [
        pytest.param(np.arange(8), id="numpy"),
        pytest.param(torch.arange(8), id="tensor"),
        pytest.param([1.0, 2.0], id="list"),
        pytest.param(
            HDF5Source(file="data.h5", dataset="/x", selection=(slice(None),)),
            id="hdf5-source",
        ),
    ],
)
def test_data_config_rejects_bare_channels(channel):
    with pytest.raises(TypeError, match="SampledChannel or TimestampedChannel"):
        DataConfig(channels=(channel,))


def test_data_config_rejects_empty_channels():
    with pytest.raises(ValueError, match="at least one channel"):
        DataConfig(channels=())


def test_data_config_rejects_old_global_dt():
    channel = SampledChannel(data=np.arange(8), dt=1.0)

    with pytest.raises(TypeError, match="dt"):
        DataConfig(channels=(channel,), dt=1.0)


@pytest.mark.parametrize(
    ("start", "stop"),
    [(0.0, 0.0), (1.0, 0.0)],
)
def test_data_config_rejects_unordered_observation_interval(start, stop):
    channel = SampledChannel(data=np.arange(8), dt=1.0)

    with pytest.raises(ValueError, match="observation_start"):
        DataConfig(
            channels=(channel,),
            observation_start=start,
            observation_stop=stop,
        )


def test_data_config_preserves_large_numpy_integer_observation_bounds():
    origin = 2**60 + 1
    channel = TimestampedChannel(timestamps=np.array([origin], dtype=np.int64))

    config = DataConfig(
        channels=(channel,),
        observation_start=np.int64(origin),
        observation_stop=np.int64(origin + 8),
    )

    assert config.observation_start == origin
    assert config.observation_stop == origin + 8
    assert type(config.observation_start) is int
    assert type(config.observation_stop) is int


@pytest.mark.parametrize(
    ("dataset", "selection", "message"),
    [
        ("", (slice(None),), "dataset cannot be empty"),
        ("/signals", (), "selection cannot be empty"),
        ("/signals", (True,), "integers or slices"),
        ("/signals", ("channel-0",), "integers or slices"),
        ("/signals", (slice(None, None, 2),), "steps other than 1"),
        ("/signals", (slice(None, None, -1),), "steps other than 1"),
    ],
)
def test_hdf5_source_rejects_invalid_configuration(dataset, selection, message):
    with pytest.raises((TypeError, ValueError), match=message):
        HDF5Source(
            file=Path("data.h5"),
            dataset=dataset,
            selection=selection,
        )


def test_hdf5_source_normalizes_path_selection_and_numpy_indices():
    source = HDF5Source(
        file="data.h5",
        dataset="/signals",
        selection=[slice(np.int64(1), np.int64(4)), np.int64(2)],
    )

    assert source.file == Path("data.h5")
    assert isinstance(source.file, Path)
    assert source.selection == (slice(1, 4), 2)
    assert type(source.selection) is tuple
    assert type(source.selection[0].start) is int
    assert type(source.selection[0].stop) is int
    assert type(source.selection[1]) is int


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("cuda:1", "cuda:1"),
        ("xpu", "xpu"),
        ("xpu:0", "xpu:0"),
        ("xpu:1", "xpu:1"),
    ],
)
def test_spectrum_config_accepts_supported_devices(device, expected):
    config = SpectrumConfig(device=device)

    assert config.device == expected


@pytest.mark.parametrize(
    "device",
    [
        "",
        "CPU",
        "cuda:",
        "cuda:01",
        "cuda:-1",
        "cuda:abc",
        "cudafoo",
        "xpu:",
        "xpu:01",
        "xpu:-1",
        "xpu:abc",
        "xpufoo",
        "cpu:0",
        "mps:0",
        "xla",
        "meta",
    ],
)
def test_spectrum_config_rejects_unsupported_devices(device):
    with pytest.raises(ValueError, match="device|device type|numbered"):
        SpectrumConfig(device=device)
