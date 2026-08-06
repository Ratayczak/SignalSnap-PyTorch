from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from signalsnap_pytorch import (
    DataConfig,
    HDF5Source,
    PhotonOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
)


def test_photon_options_accept_unit_weighting_without_mark_fields():
    options = PhotonOptions(weighting="unit")

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
def test_unit_photon_weighting_rejects_exponential_fields(field, value):
    with pytest.raises(ValidationError, match="does not accept"):
        PhotonOptions(weighting="unit", **{field: value})


def test_photon_options_accept_exponential_weighting():
    options = PhotonOptions(
        weighting="exponential",
        scale=1.5,
        repetitions=100,
        seed=1234,
    )

    assert options.scale == 1.5
    assert options.repetitions == 100
    assert options.seed == 1234


@pytest.mark.parametrize(
    "missing_field",
    ["scale", "repetitions"],
)
def test_exponential_photon_weighting_requires_scale_and_repetitions(missing_field):
    values = {"scale": 1.0, "repetitions": 2}
    del values[missing_field]

    with pytest.raises(ValidationError, match="requires scale and repetitions"):
        PhotonOptions(weighting="exponential", **values)


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, np.inf, np.nan, True],
)
def test_exponential_photon_weighting_rejects_invalid_scale(scale):
    with pytest.raises((ValidationError, TypeError), match="scale|finite"):
        PhotonOptions(
            weighting="exponential",
            scale=scale,
            repetitions=2,
        )


@pytest.mark.parametrize(
    "repetitions",
    [0, -1, 1.5, "2", True],
)
def test_exponential_photon_weighting_requires_strict_positive_repetitions(repetitions):
    with pytest.raises(ValidationError, match="repetitions"):
        PhotonOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=repetitions,
        )


@pytest.mark.parametrize(
    "seed",
    [-1, 1.5, "2", True],
)
def test_exponential_photon_weighting_requires_strict_nonnegative_seed(seed):
    with pytest.raises(ValidationError, match="seed"):
        PhotonOptions(
            weighting="exponential",
            scale=1.0,
            repetitions=2,
            seed=seed,
        )


def test_photon_options_are_frozen():
    options = PhotonOptions(weighting="unit")

    with pytest.raises(ValidationError, match="frozen"):
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


@pytest.mark.parametrize(
    ("batch_size", "expected_error"),
    [
        pytest.param(0, ValidationError, id="zero"),
        pytest.param(-1, ValidationError, id="negative"),
        pytest.param(1.5, ValidationError, id="non-integer"),
        pytest.param(True, TypeError, id="boolean"),
    ],
)
def test_spectrum_config_rejects_invalid_spectral_estimates_per_batch(
    batch_size,
    expected_error,
):
    with pytest.raises(expected_error, match="spectral_estimates_per_batch"):
        SpectrumConfig(spectral_estimates_per_batch=batch_size)


def test_spectrum_config_accepts_short_term_uncertainty_estimation():
    config = SpectrumConfig(uncertainty_estimation="short_term", m_var=4)

    assert config.uncertainty_estimation == "short_term"
    assert config.m_var == 4


def test_spectrum_config_rejects_unknown_uncertainty_estimation():
    with pytest.raises(ValidationError, match="uncertainty_estimation"):
        SpectrumConfig(uncertainty_estimation="local")


@pytest.mark.parametrize("m_var", [1, 0, -1, 1.5, True])
def test_spectrum_config_rejects_invalid_m_var(m_var):
    with pytest.raises(ValidationError, match="m_var"):
        SpectrumConfig(uncertainty_estimation="short_term", m_var=m_var)


def test_spectrum_config_accepts_positive_frequency_spacing():
    assert SpectrumConfig(df=0.125).df == 0.125


@pytest.mark.parametrize("df", [0.0, -0.125, np.inf, np.nan])
def test_spectrum_config_rejects_invalid_frequency_spacing(df):
    with pytest.raises(ValidationError, match="df"):
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


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan, True])
def test_sampled_channel_rejects_invalid_dt(dt):
    with pytest.raises((ValidationError, TypeError), match="dt"):
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
    ],
)
def test_sampled_channel_rejects_invalid_data(data, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        SampledChannel(data=data, dt=1.0)


def test_sampled_channel_rejects_non_cpu_tensor_storage():
    with pytest.raises(ValidationError, match="stored on the CPU"):
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
        pytest.param(torch.empty(2, device="meta"), "stored on the CPU", id="non-cpu"),
    ],
)
def test_timestamped_channel_rejects_invalid_timestamps(timestamps, message):
    with pytest.raises((ValidationError, TypeError), match=message):
        TimestampedChannel(timestamps=timestamps)


def test_explicit_data_models_are_frozen_without_freezing_referenced_arrays():
    data = np.arange(8)
    sampled = SampledChannel(data=data, dt=1.0)
    timestamped = TimestampedChannel(timestamps=np.array([0.0, 1.0]))
    source = HDF5Source(file="data.h5", dataset="/x", selection=(slice(None),))

    with pytest.raises(ValidationError, match="frozen"):
        sampled.dt = 2.0
    with pytest.raises(ValidationError, match="frozen"):
        timestamped.timestamps = np.array([])
    with pytest.raises(ValidationError, match="frozen"):
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
    with pytest.raises(ValidationError, match="at least 1 item"):
        DataConfig(channels=())


def test_data_config_rejects_old_global_dt():
    channel = SampledChannel(data=np.arange(8), dt=1.0)

    with pytest.raises(ValidationError, match="dt"):
        DataConfig(channels=(channel,), dt=1.0)


@pytest.mark.parametrize(
    ("start", "stop"),
    [(0.0, 0.0), (1.0, 0.0)],
)
def test_data_config_rejects_unordered_observation_interval(start, stop):
    channel = SampledChannel(data=np.arange(8), dt=1.0)

    with pytest.raises(ValidationError, match="observation_start"):
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
    with pytest.raises((ValidationError, TypeError), match=message):
        HDF5Source(
            file=Path("data.h5"),
            dataset=dataset,
            selection=selection,
        )


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
    with pytest.raises(ValidationError, match="device|device type|numbered"):
        SpectrumConfig(device=device)
