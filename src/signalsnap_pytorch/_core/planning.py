# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import math
import secrets
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ..configurators import (
    DataConfig,
    PhotonOptions,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
)
from .data_access import RuntimeSource, get_source_length, validate_timestamp_source
from .utils import ChannelIndex, FrequencyUnits, TimeUnits, unit_conversion_time_to_freq

_MAX_AMPLITUDE_REPETITIONS_PER_BATCH = 100


@dataclass(frozen=True, slots=True)
class SampledChannelPlan:
    """Resolved instructions for one active sampled channel."""

    sample_count: int
    dt: float


@dataclass(frozen=True, slots=True)
class TimestampedChannelPlan:
    """Resolved access and amplitude instructions for one timestamped channel."""

    event_count: int
    weighting: Literal["unit", "exponential"]
    scale: float | None


ChannelPlan = SampledChannelPlan | TimestampedChannelPlan


@dataclass(frozen=True, slots=True)
class RepetitionPlan:
    """Resolved calculation-wide amplitude-realization iteration."""

    count: int
    batch_size: int
    resolved_seed: int | None

    def iter_batches(self) -> Iterator[range]:
        """Yield bounded batches of stable realization IDs."""

        for start in range(0, self.count, self.batch_size):
            stop = min(start + self.batch_size, self.count)
            yield range(start, stop)


def _resolve_repetition_plan(photon_options: PhotonOptions | None) -> RepetitionPlan:
    """Resolve shared amplitude-repetition iteration for one calculation."""

    if photon_options is None or photon_options.weighting == "unit":
        return RepetitionPlan(count=1, batch_size=1, resolved_seed=None)

    assert photon_options.repetitions is not None

    resolved_seed = photon_options.seed if photon_options.seed is not None else secrets.randbits(63)

    return RepetitionPlan(
        count=photon_options.repetitions,
        batch_size=min(photon_options.repetitions, _MAX_AMPLITUDE_REPETITIONS_PER_BATCH),
        resolved_seed=resolved_seed,
    )


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """Resolved half-open observation interval, physical-window, and batching settings."""

    observation_start: float
    observation_stop: float
    duration: float
    windows_per_estimate: int
    unshifted_estimate_count: int
    shifted_estimate_count: int
    estimates_per_batch: int
    interlacing_offset: float


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """One batch of shared physical windows."""

    relative_starts: NDArray[np.float64]
    duration: float
    estimate_count: int
    shifted: bool


@dataclass(frozen=True, slots=True)
class SampledFrequencyPlan:
    """Resolved shifted FFT grid and hard-bounded sampled band view."""

    full_fft_frequencies: NDArray[np.floating[Any]]
    band_frequencies: NDArray[np.floating[Any]]
    band_start: int
    band_stop: int

    @property
    def window_points(self) -> int:
        """Return the number of frequencies in the complete shifted FFT grid."""

        return int(self.full_fft_frequencies.size)


@dataclass(frozen=True, slots=True)
class TimestampFrequencyPlan:
    """Resolved hard-bounded direct-transform frequency grid."""

    band_frequencies: NDArray[np.float64]


FrequencyPlan = SampledFrequencyPlan | TimestampFrequencyPlan


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for;
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults,
    data-size constraints, frequency axes, and device details have been resolved.

    Attributes
    ----------
    active_data_channels : tuple[int, ...]
        Data-channel indices used by the calculation.
    spectrum_frequency_plans : dict[tuple[ChannelIndex, ...], FrequencyPlan]
        Selected frequency plan for each requested spectrum tuple, in request order.
    repetition_plan : RepetitionPlan
        Shared amplitude-realization count, bounded batch size, and resolved seed.
    uncertainty_estimation : Literal["global", "short_term"]
        Uncertainty-estimation method.
    m_var : int
        Effective number of consecutive spectral estimates per short-term batch. This may be lower
        than the requested value if insufficient unshifted estimates are available.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
    old_window : bool
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from the
        old API is used as a window function.
    """

    active_data_channels: tuple[int, ...]
    spectrum_frequency_plans: dict[tuple[ChannelIndex, ...], FrequencyPlan]
    channel_plans: dict[int, ChannelPlan]
    window_plan: WindowPlan
    repetition_plan: RepetitionPlan
    uncertainty_estimation: Literal["global", "short_term"]
    m_var: int
    freq_unit: FrequencyUnits
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    old_window: bool


def _resolve_device(device_name: str) -> torch.device:
    """Validate and resolve a requested PyTorch device.

    Supported examples are ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ``"mps"``,
    ``"xpu"``, and ``"xpu:0"``.
    """
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device {device_name!r}.") from exc

    if device.type == "cpu":
        return device

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available. Check that PyTorch was installed "
                "with CUDA support and that an NVIDIA GPU and compatible driver are available."
            )

        index = device.index
        if index is None:
            index = torch.cuda.current_device()

        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f"CUDA device {index} was requested, but only {device_count} CUDA device(s) are "
                "available."
            )

        # Return an explicit device such as cuda:0.
        return torch.device("cuda", index)

    if device.type == "mps":
        if device.index is not None:
            raise ValueError("MPS devices do not support numbered device indices.")

        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS was requested, but this PyTorch installation was not built with MPS support."
            )

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available on this system.")

        return torch.device("mps")

    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError(
                "XPU was requested, but XPU is not available. Check that PyTorch was installed "
                "with Intel GPU support and that a supported Intel GPU and driver are available."
            )

        index = device.index
        if index is None:
            index = xpu.current_device()

        device_count = xpu.device_count()
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f"XPU device {index} was requested, but only {device_count} XPU device(s) are "
                "available."
            )

        return torch.device("xpu", index)

    raise ValueError(
        f"Unsupported device type {device.type!r}. Use 'cpu', 'cuda', 'cuda:N', 'mps', 'xpu', "
        "or 'xpu:N'."
    )


def normalize_channel_index(channel: object, channel_count: int) -> int:
    """Validate and normalize one data-channel index.

    Parameters
    ----------
    channel : object
        Candidate integer index. NumPy integers are accepted; Booleans are rejected.
    channel_count : int
        Number of available channels.

    Returns
    -------
    int
        Normalized built-in integer index.

    Raises
    ------
    TypeError
        If ``channel`` is not an integer or is Boolean.
    ValueError
        If ``channel`` is negative or outside the available channel range.
    """

    if isinstance(channel, (bool, np.bool_)):
        raise TypeError(f"Channel indices must be integers; received {channel!r}.")

    if not isinstance(channel, (int, np.integer)):
        raise TypeError(f"Channel indices must be integers; received {channel!r}.")

    normalized = int(channel)

    if normalized < 0:
        raise ValueError(f"Channel indices must be nonnegative; received {normalized}.")

    if normalized >= channel_count:
        raise ValueError(
            f"Channel {normalized} is out of bounds for {channel_count} available data channels."
        )

    return normalized


def resolve_channels(
    requested_spectra: list[tuple[int, ...]] | None,
    channel_count: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate and normalize the requested spectra.

    If ``requested_spectra`` is ``None``, generate auto-spectra of
    orders one through four for every available data channel.

    Parameters
    ----------
    requested_spectra : list[tuple[int, ...]] | None
        Explicit order-one through order-four channel tuples, or ``None`` for all default
        auto-spectra.
    channel_count : int
        Number of available input channels.

    Returns
    -------
    tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]
        Normalized spectrum requests and unique active data channels in first-use order.

    Raises
    ------
    TypeError
        If a request is not a tuple or contains a non-integer channel index.
    ValueError
        If no channels or requests are supplied, an order or index is invalid, or a request is
        duplicated.
    """
    if channel_count < 1:
        raise ValueError("At least one data channel is required.")

    if requested_spectra is None:
        spectra_channels = tuple(
            (channel,) * order for channel in range(channel_count) for order in range(1, 5)
        )
    else:
        if not requested_spectra:
            raise ValueError("requested_spectra must contain at least one spectrum.")

        resolved: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        for spectrum in requested_spectra:
            if not isinstance(spectrum, tuple):
                raise TypeError("Each spectrum request must be a tuple of channel indices.")

            order = len(spectrum)
            if not 1 <= order <= 4:
                raise ValueError(f"Spectrum order must be between 1 and 4; received order {order}.")

            resolved_channels: list[int] = []

            for channel in spectrum:
                resolved_channels.append(normalize_channel_index(channel, channel_count))

            resolved_spectrum = tuple(resolved_channels)

            if resolved_spectrum in seen:
                raise ValueError(
                    f"requested_spectra cannot contain duplicates; "
                    f"{resolved_spectrum} was requested more than once."
                )

            seen.add(resolved_spectrum)
            resolved.append(resolved_spectrum)

        spectra_channels = tuple(resolved)

    active_data_channels = tuple(
        dict.fromkeys(channel for spectrum in spectra_channels for channel in spectrum)
    )

    return spectra_channels, active_data_channels


def _build_channel_plans(
    data_config: DataConfig,
    opened_channels: Mapping[int, RuntimeSource],
    photon_options: PhotonOptions | None,
) -> tuple[dict[int, ChannelPlan], TimeUnits]:
    """Build plans for active channels and validate shared sampled properties."""

    channel_plans: dict[int, ChannelPlan] = {}
    first_sampled_channel: int | None = None
    first_sampled_plan: SampledChannelPlan | None = None

    for channel, source in opened_channels.items():
        channel_config = data_config.channels[channel]
        source_length = get_source_length(source)

        if isinstance(channel_config, SampledChannel):
            plan = SampledChannelPlan(sample_count=source_length, dt=channel_config.dt)

            if first_sampled_plan is None:
                first_sampled_channel = channel
                first_sampled_plan = plan
            else:
                if plan.dt != first_sampled_plan.dt:
                    raise ValueError(
                        f"Channel {channel} has dt={plan.dt}, but channel "
                        f"{first_sampled_channel} has dt={first_sampled_plan.dt}."
                    )

                if plan.sample_count != first_sampled_plan.sample_count:
                    raise ValueError(
                        f"Channel {channel} contains {plan.sample_count} samples, "
                        f"but channel {first_sampled_channel} contains "
                        f"{first_sampled_plan.sample_count} samples."
                    )

        elif isinstance(channel_config, TimestampedChannel):
            if photon_options is None:
                raise RuntimeError(
                    "Internal error: timestamped channel planning requires resolved PhotonOptions."
                )

            plan = TimestampedChannelPlan(
                event_count=source_length,
                weighting=photon_options.weighting,
                scale=photon_options.scale,
            )

        else:
            raise TypeError(
                f"Channel {channel} has unsupported type {type(channel_config).__name__}."
            )

        channel_plans[channel] = plan

    return channel_plans, data_config.t_unit


def _resolve_observation_interval(
    data_config: DataConfig,
    channel_plans: Mapping[int, ChannelPlan],
) -> tuple[float, float]:
    """Resolve the common half-open observation interval."""

    has_timestamped_channel = any(
        isinstance(plan, TimestampedChannelPlan) for plan in channel_plans.values()
    )
    sampled_plan = next(
        (plan for plan in channel_plans.values() if isinstance(plan, SampledChannelPlan)),
        None,
    )

    if has_timestamped_channel:
        if data_config.observation_start is None or data_config.observation_stop is None:
            raise ValueError(
                "Timestamped calculations require explicit observation_start and observation_stop."
            )

        observation_start = data_config.observation_start
        observation_stop = data_config.observation_stop
    elif sampled_plan is not None:
        observation_start = (
            0.0 if data_config.observation_start is None else data_config.observation_start
        )
        sampled_duration = sampled_plan.sample_count * sampled_plan.dt
        observation_stop = (
            observation_start + sampled_duration
            if data_config.observation_stop is None
            else data_config.observation_stop
        )
    else:
        raise RuntimeError("Internal error: no active channel plan is available.")

    if sampled_plan is not None:
        sampled_duration = sampled_plan.sample_count * sampled_plan.dt
        observation_duration = observation_stop - observation_start

        duration_tolerance = 4.0 * max(
            math.ulp(observation_start),
            math.ulp(observation_stop),
            math.ulp(observation_duration),
            math.ulp(sampled_duration),
        )

        if abs(observation_duration - sampled_duration) > duration_tolerance:
            raise ValueError(
                f"The configured observation interval has duration {observation_duration}, but "
                f"{sampled_plan.sample_count} samples with dt={sampled_plan.dt} span "
                f"{sampled_duration}."
            )

    return observation_start, observation_stop


def resolve_sampled_frequencies(
    spectrum_config: SpectrumConfig,
    dt: float,
) -> tuple[int, SampledFrequencyPlan]:
    """Resolve the sampled FFT grid and hard-bounded output band.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        Spectrum configuration options.
    dt : float
        Time step of the specified data channels.

    Returns
    -------
    tuple[int, SampledFrequencyPlan]
        Resolved sampled window length and frequency plan.

    Raises
    ------
    ValueError
        If the resolved window length is zero or the intersection of the requested
        band and sampled FFT support contains no frequencies.
    """
    # An omitted upper bound retains the sampled-only Nyquist default. Explicit
    # bounds may extend beyond sampled support because mixed calculations can also
    # contain timestamp-only spectra using the wider requested view.
    f_max = 1 / (2 * dt) if spectrum_config.f_max is None else spectrum_config.f_max

    # Compute how many points must be taken into account in one window to achieve the required
    # frequency spacing in the given frequency bounds.
    if spectrum_config.df is None:
        window_points = 1000
    else:
        window_points = int(np.round(1 / (spectrum_config.df * dt)))

    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    # get the frequency axis
    freq_all = np.fft.fftfreq(window_points, dt)
    freq_all = np.fft.fftshift(freq_all)

    band_indices = np.flatnonzero((freq_all >= spectrum_config.f_min) & (freq_all <= f_max))

    if band_indices.size == 0:
        raise ValueError(
            f"The requested frequency band "
            f"[{spectrum_config.f_min}, {f_max}] does not contain "
            "any FFT frequencies at the resolved frequency spacing."
        )

    band_start = int(band_indices[0])
    band_stop = int(band_indices[-1]) + 1
    band_frequencies = freq_all[band_start:band_stop]

    frequency_plan = SampledFrequencyPlan(
        full_fft_frequencies=freq_all,
        band_frequencies=band_frequencies,
        band_start=band_start,
        band_stop=band_stop,
    )

    return window_points, frequency_plan


def resolve_timestamp_frequencies(
    *,
    f_min: float,
    f_max: float,
    window_duration: float,
) -> TimestampFrequencyPlan:
    """Resolve the direct-transform grid inside inclusive hard bounds."""

    actual_df = 1.0 / window_duration
    first_candidate = math.floor(f_min / actual_df) - 1
    last_candidate = math.ceil(f_max / actual_df) + 1

    grid_indices = np.arange(first_candidate, last_candidate + 1, dtype=np.int64)
    candidate_frequencies = grid_indices.astype(np.float64) * actual_df
    within_bounds = (candidate_frequencies >= f_min) & (candidate_frequencies <= f_max)
    band_frequencies = candidate_frequencies[within_bounds]

    if band_frequencies.size == 0:
        raise ValueError(
            f"The requested frequency band [{f_min}, {f_max}] does not contain "
            "any timestamp frequencies at the resolved frequency spacing."
        )

    return TimestampFrequencyPlan(band_frequencies=band_frequencies)


def _count_complete_windows(available_duration: float, window_duration: float) -> int:
    """Count complete windows while tolerating only ULP-scale boundary rounding."""

    if available_duration <= 0.0:
        return 0

    window_ratio = available_duration / window_duration
    if not math.isfinite(window_ratio):
        raise ValueError("The observation interval contains too many physical windows.")

    nearest_integer = round(window_ratio)
    boundary_tolerance = 4.0 * math.ulp(window_ratio)

    if abs(window_ratio - nearest_integer) <= boundary_tolerance:
        return nearest_integer

    return math.floor(window_ratio)


def _resolve_window_plan(
    spectrum_config: SpectrumConfig,
    *,
    observation_start: float,
    observation_stop: float,
    window_duration: float,
    interlacing_offset: float,
    available_unshifted_windows: int,
    available_shifted_windows: int,
    orders: tuple[int, ...],
) -> tuple[WindowPlan, int]:
    """Resolve shared physical estimates, batching, and uncertainty grouping."""

    if available_unshifted_windows < spectrum_config.m:
        windows_per_estimate = available_unshifted_windows
        warnings.warn(
            f"Not enough data points are available for m={spectrum_config.m}; "
            f"using m={windows_per_estimate} instead.",
            UserWarning,
            stacklevel=4,
        )
    else:
        windows_per_estimate = spectrum_config.m

    if windows_per_estimate < max(orders):
        raise ValueError("Not enough data points")

    available_unshifted_estimates = available_unshifted_windows // windows_per_estimate
    if spectrum_config.spectral_estimates_max is None:
        unshifted_estimate_count = available_unshifted_estimates
    else:
        unshifted_estimate_count = min(
            spectrum_config.spectral_estimates_max,
            available_unshifted_estimates,
        )

    shifted_estimate_count = 0
    if spectrum_config.interlacing:
        available_shifted_estimates = available_shifted_windows // windows_per_estimate
        shifted_estimate_count = min(unshifted_estimate_count, available_shifted_estimates)
        if shifted_estimate_count < 1:
            raise ValueError(
                "Interlacing was requested, but the data is too short for a shifted "
                "spectral estimate. Disable interlacing or provide more data."
            )

    m_var = spectrum_config.m_var
    if (
        spectrum_config.uncertainty_estimation == "short_term"
        and 2 <= unshifted_estimate_count < m_var
    ):
        m_var = unshifted_estimate_count
        warnings.warn(
            f"Only {unshifted_estimate_count} unshifted spectral estimates are "
            f"available; using m_var={m_var} instead.",
            UserWarning,
            stacklevel=4,
        )

    estimates_per_batch = spectrum_config.spectral_estimates_per_batch
    if spectrum_config.uncertainty_estimation == "short_term" and estimates_per_batch >= m_var:
        estimates_per_batch -= estimates_per_batch % m_var

    return (
        WindowPlan(
            observation_start=observation_start,
            observation_stop=observation_stop,
            duration=window_duration,
            windows_per_estimate=windows_per_estimate,
            unshifted_estimate_count=unshifted_estimate_count,
            shifted_estimate_count=shifted_estimate_count,
            estimates_per_batch=estimates_per_batch,
            interlacing_offset=interlacing_offset,
        ),
        m_var,
    )


def build_runtime_config(
    data_config: DataConfig,
    opened_channels: Mapping[int, RuntimeSource],
    spectrum_config: SpectrumConfig,
    spectra_channels: tuple[tuple[int, ...], ...],
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates active channels, resolves shared physical windows and per-spectrum frequency views,
    and selects the calculation dtypes and device.

    Parameters
    ----------
    data_config : :class:`DataConfig`
        Data configurations containing the input data and sampling metadata.
    opened_channels : Mapping[int, Any | HDF5SourceState]
        Opened runtime representation of ``data_config.channels``. Array channels are retained
        directly; HDF5 channels are represented by
        :class:`~signalsnap_pytorch._core.data_access.HDF5SourceState` instances.
    spectrum_config : :class:`SpectrumConfig`
        User configuration for frequency bounds, precision, device, windowing, and
        related calculation options.
    spectra_channels : tuple[tuple[int, ...], ...]
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index which matches the index
        in ``data_config.channels``.

    Returns
    -------
    RuntimeConfig
        Resolved runtime configuration.

    Warns
    -----
    UserWarning
        If the configured ``m`` requires more samples than are available and is reduced at runtime.

    Raises
    ------
    ValueError
        If active channels have unequal lengths, frequency resolution fails, the effective ``m`` is
        below the highest requested order, or interlacing cannot produce a shifted estimate.
    RuntimeError
        If the requested accelerator is unavailable or its device index is invalid.
    """

    # Validate and read the channels, number of data points, and the time step from the
    # SpectrumConfig and DataConfigs
    active_data_channels = tuple(opened_channels)
    has_timestamped_channel = any(
        isinstance(data_config.channels[channel], TimestampedChannel)
        for channel in active_data_channels
    )
    photon_options = spectrum_config.photon_options

    if has_timestamped_channel and photon_options is None:
        raise ValueError("PhotonOptions are required when an active channel is timestamped.")

    if not has_timestamped_channel and photon_options is not None:
        raise ValueError("PhotonOptions cannot be used in a sampled-only calculation.")

    channel_plans, t_unit = _build_channel_plans(
        data_config=data_config,
        opened_channels=opened_channels,
        photon_options=photon_options,
    )

    observation_start, observation_stop = _resolve_observation_interval(data_config, channel_plans)
    for channel, channel_plan in channel_plans.items():
        if isinstance(channel_plan, TimestampedChannelPlan):
            validate_timestamp_source(
                opened_channels[channel],
                observation_start,
                observation_stop,
                label=f"Timestamped channel {channel}",
            )
    repetition_plan = _resolve_repetition_plan(photon_options)
    orders = tuple(sorted({len(channels) for channels in spectra_channels}))

    sampled_plan = next(
        (plan for plan in channel_plans.values() if isinstance(plan, SampledChannelPlan)),
        None,
    )

    if sampled_plan is None:
        if spectrum_config.df is None or spectrum_config.f_max is None:
            raise ValueError("Timestamp-only calculations require explicit df and f_max.")

        sampled_frequency_plan = None
        window_duration = 1.0 / spectrum_config.df
        interlacing_offset = window_duration / 2.0
        observation_duration = observation_stop - observation_start

        available_unshifted_windows = _count_complete_windows(observation_duration, window_duration)
        available_shifted_windows = _count_complete_windows(
            observation_duration - interlacing_offset,
            window_duration,
        )
    else:
        window_points, sampled_frequency_plan = resolve_sampled_frequencies(
            spectrum_config,
            sampled_plan.dt,
        )
        window_duration = window_points * sampled_plan.dt
        sampled_interlacing_offset = window_points // 2
        interlacing_offset = sampled_interlacing_offset * sampled_plan.dt

        available_unshifted_windows = sampled_plan.sample_count // window_points
        available_shifted_windows = max(
            0,
            (sampled_plan.sample_count - sampled_interlacing_offset) // window_points,
        )

    window_plan, m_var = _resolve_window_plan(
        spectrum_config,
        observation_start=observation_start,
        observation_stop=observation_stop,
        window_duration=window_duration,
        interlacing_offset=interlacing_offset,
        available_unshifted_windows=available_unshifted_windows,
        available_shifted_windows=available_shifted_windows,
        orders=orders,
    )

    timestamp_frequency_plan = None
    if has_timestamped_channel:
        timestamp_f_max = spectrum_config.f_max
        if timestamp_f_max is None:
            assert sampled_plan is not None
            timestamp_f_max = 1.0 / (2.0 * sampled_plan.dt)

        timestamp_frequency_plan = resolve_timestamp_frequencies(
            f_min=spectrum_config.f_min,
            f_max=timestamp_f_max,
            window_duration=window_duration,
        )

    spectrum_frequency_plans: dict[tuple[ChannelIndex, ...], FrequencyPlan] = {}

    for channels in spectra_channels:
        contains_sampled_channel = any(
            isinstance(channel_plans[channel], SampledChannelPlan) for channel in channels
        )

        if contains_sampled_channel:
            assert sampled_frequency_plan is not None
            frequency_plan: FrequencyPlan = sampled_frequency_plan
        else:
            assert timestamp_frequency_plan is not None
            frequency_plan = timestamp_frequency_plan

        spectrum_frequency_plans[tuple(channels)] = frequency_plan

    device = _resolve_device(spectrum_config.device)

    # determine the data types based on the given precision
    if spectrum_config.precision == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif spectrum_config.precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        if device.type in {"mps", "xpu"}:
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

    return RuntimeConfig(
        active_data_channels=active_data_channels,
        spectrum_frequency_plans=spectrum_frequency_plans,
        channel_plans=channel_plans,
        window_plan=window_plan,
        repetition_plan=repetition_plan,
        uncertainty_estimation=spectrum_config.uncertainty_estimation,
        m_var=m_var,
        freq_unit=unit_conversion_time_to_freq(t_unit),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=device,
        old_window=spectrum_config.old_window,
    )


def physical_estimate_count(plan: WindowPlan) -> int:
    """Return the total number of physical spectral estimates."""

    return plan.unshifted_estimate_count + plan.shifted_estimate_count


def iter_window_batches(plan: WindowPlan) -> Iterator[WindowBatch]:
    """Yield shared physical-window batches in relative observation time."""

    def iter_placement(
        estimate_count: int,
        relative_offset: float,
        shifted: bool,
    ) -> Iterator[WindowBatch]:
        for first_estimate in range(0, estimate_count, plan.estimates_per_batch):
            batch_size = min(plan.estimates_per_batch, estimate_count - first_estimate)
            first_window = (
                relative_offset + first_estimate * plan.windows_per_estimate * plan.duration
            )
            relative_starts = (
                first_window
                + np.arange(batch_size * plan.windows_per_estimate, dtype=np.float64)
                * plan.duration
            ).reshape(batch_size, plan.windows_per_estimate)

            yield WindowBatch(
                relative_starts=relative_starts,
                duration=plan.duration,
                estimate_count=batch_size,
                shifted=shifted,
            )

    yield from iter_placement(plan.unshifted_estimate_count, relative_offset=0, shifted=False)

    yield from iter_placement(
        plan.shifted_estimate_count,
        relative_offset=plan.interlacing_offset,
        shifted=True,
    )
