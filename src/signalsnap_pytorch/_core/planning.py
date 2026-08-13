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
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ..configurators import (
    DataConfig,
    SampledChannel,
    SpectrumConfig,
    TimestampedChannel,
    TimestampOptions,
)
from .data_access import (
    HDF5SourceState,
    RuntimeSource,
    get_source_length,
    validate_sampled_hdf5_source,
    validate_timestamp_source,
)
from .utils import FrequencyUnits, TimeUnits, unit_conversion_time_to_freq

_MAX_AMPLITUDE_REPETITIONS_PER_BATCH = 10


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
    """Resolved calculation-wide amplitude-realization plan.

    Attributes
    ----------
    count : int
        Total number of amplitude realizations. Calculations without exponential timestamp
        weighting use one realization; exponential weighting uses the configured repetition
        count.
    batch_size : int
        Maximum number of realizations processed together.
    resolved_seed : int | None
        Seed used for keyed exponential amplitudes. This is the configured seed or a seed generated
        during planning. It is ``None`` when exponential weighting is not active.
    """

    count: int
    batch_size: int
    resolved_seed: int | None

    def iter_batches(self) -> Iterator[range]:
        """Yield consecutive batches of calculation-wide realization IDs.

        The IDs cover ``range(count)`` exactly once, in ascending order, with no batch larger than
        ``batch_size``. They remain independent of repetition batching so that keyed timestamp
        amplitudes are reproducible when the batch size changes.

        Yields
        ------
        range
            Consecutive realization IDs for one repetition batch.
        """

        for start in range(0, self.count, self.batch_size):
            stop = min(start + self.batch_size, self.count)
            yield range(start, stop)


def _resolve_repetition_plan(timestamp_options: TimestampOptions | None) -> RepetitionPlan:
    """Resolve shared amplitude-repetition iteration for one calculation."""

    if timestamp_options is None or timestamp_options.weighting == "unit":
        return RepetitionPlan(count=1, batch_size=1, resolved_seed=None)

    assert timestamp_options.repetitions is not None

    resolved_seed = (
        timestamp_options.seed if timestamp_options.seed is not None else secrets.randbits(63)
    )

    requested_batch_size = (
        timestamp_options.repetitions_per_batch
        if timestamp_options.repetitions_per_batch is not None
        else _MAX_AMPLITUDE_REPETITIONS_PER_BATCH
    )

    return RepetitionPlan(
        count=timestamp_options.repetitions,
        batch_size=min(timestamp_options.repetitions, requested_batch_size),
        resolved_seed=resolved_seed,
    )


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """Resolved observation interval, physical-window layout, and batching plan.

    Attributes
    ----------
    observation_start, observation_stop : float
        Bounds of the common half-open observation interval.
    duration : float
        Duration of one physical window, in the configured time unit.
    windows_per_estimate : int
        Effective number of physical windows combined into one spectral estimate. This may be lower
        than the requested ``SpectrumConfig.m`` when insufficient data is available.
    unshifted_estimate_count : int
        Number of complete unshifted spectral estimates to calculate after applying
        ``SpectrumConfig.spectral_estimates_max``.
    shifted_estimate_count : int
        Number of complete interlaced estimates to calculate. Zero when interlacing is disabled.
    estimates_per_batch : int
        Effective maximum number of spectral estimates processed together. In short-term
        uncertainty mode, this may be reduced to a multiple of the effective ``m_var``.
    interlacing_offset : float
        Offset of the shifted placement relative to the unshifted placement. For sampled data, this
        is aligned to a sample boundary.
    """

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
    """One batch of physical windows from a single placement group.

    Attributes
    ----------
    relative_starts : NDArray[np.float64]
        Physical-window start times relative to ``WindowPlan.observation_start``, with shape
        ``(B, m)``. Rows identify spectral estimates and columns identify the physical windows
        belonging to each estimate.
    duration : float
        Duration of every physical window in the batch.
    estimate_count : int
        Number of spectral estimates in the batch, equal to ``B``.
    shifted : bool
        Whether the batch belongs to the shifted interlacing placement.
    relative_stop : float or None
        Exclusive stop of the final physical window. Batches produced by
        :func:`iter_window_batches` derive it from the same boundary grid as
        ``relative_starts``. ``None`` supports manually constructed batches.
    """

    relative_starts: NDArray[np.float64]
    duration: float
    estimate_count: int
    shifted: bool
    relative_stop: float | None = None


@dataclass(frozen=True, slots=True)
class FFTFrequencyPlan:
    """Resolved shifted FFT grid and hard-bounded sampled band view.

    Attributes
    ----------
    shifted_full_fft_frequencies : NDArray[np.floating[Any]]
        Full frequency grid of the FFT after ``fftshift``.
    band_start, band_stop : int
        Half-open slice bounds such that
        ``band_frequencies = shifted_full_fft_frequencies[band_start:band_stop]``. The slice
        contains the FFT frequencies within the requested frequency bounds.
    """

    shifted_full_fft_frequencies: NDArray[np.floating[Any]]
    band_start: int
    band_stop: int

    @property
    def band_frequencies(self) -> NDArray[np.floating[Any]]:
        """Return the frequencies inside the requested frequency bounds."""
        return self.shifted_full_fft_frequencies[self.band_start : self.band_stop]

    @property
    def window_points(self) -> int:
        """Return the number of frequencies in the complete shifted FFT grid."""

        return int(self.shifted_full_fft_frequencies.size)


@dataclass(frozen=True, slots=True)
class DirectFrequencyPlan:
    """Resolved hard-bounded direct-transform frequency grid.

    Attributes
    ----------
    actual_df : float
        Resolved grid spacing, equal to ``1 / window_duration``.
    grid_indices : NDArray[np.int64]
        Integer Fourier-grid coordinates specifying the frequencies inside the requested frequency
        bounds: ``band_frequencies = grid_indices * actual_df``.
    """

    actual_df: float
    grid_indices: NDArray[np.int64]

    @property
    def band_frequencies(self) -> NDArray[np.float64]:
        """Return the frequencies inside the requested frequency bounds."""
        return cast(NDArray[np.float64], self.grid_indices * self.actual_df)


FrequencyPlan = FFTFrequencyPlan | DirectFrequencyPlan


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for;
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults, data
    constraints, frequency grids, batching, numerical precision, and device availability have been
    resolved.

    Attributes
    ----------
    active_data_channels : tuple[int, ...]
        Unique data-channel indices referenced by ``requested_spectra``, in first-use order.
    requested_spectra : tuple[tuple[int, ...], ...]
        Validated channel tuples identifying the requested spectra, in request order.
    fft_frequency_plan : FFTFrequencyPlan | None
        Shared shifted-FFT frequency plan used by every spectrum containing at least one sampled
        channel. ``None`` when no sampled channel is active.
    direct_frequency_plan : DirectFrequencyPlan | None
        Shared direct-transform frequency plan used by timestamp-only spectra. ``None`` when no
        timestamped channel is active.
    channel_plans : dict[int, ChannelPlan]
        Resolved source metadata and processing options for each active data channel.
    window_plan : WindowPlan
        Shared observation interval, physical-window layout, interlacing, estimate counts, and
        processing batch size.
    repetition_plan : RepetitionPlan
        Shared amplitude-realization count, repetition batch size, and resolved random seed.
        Sampled data and unit-weighted timestamp data use a single deterministic realization.
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
    requested_spectra: tuple[tuple[int, ...], ...]
    fft_frequency_plan: FFTFrequencyPlan | None
    direct_frequency_plan: DirectFrequencyPlan | None
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

    def frequency_plan_for(self, channels: tuple[int, ...]) -> FrequencyPlan:
        """Return the frequency plan used for a spectrum channel tuple.

        Spectra containing at least one sampled channel use ``fft_frequency_plan``.
        Timestamp-only spectra use ``direct_frequency_plan``.

        Parameters
        ----------
        channels : tuple[int, ...]
            Data-channel indices identifying the spectrum.

        Returns
        -------
        FrequencyPlan
            Shared frequency plan applicable to the spectrum.

        Raises
        ------
        RuntimeError
            If the required frequency plan was not created.
        """
        if any(isinstance(self.channel_plans[channel], SampledChannelPlan) for channel in channels):
            plan = self.fft_frequency_plan
        else:
            plan = self.direct_frequency_plan

        if plan is None:
            raise RuntimeError(f"No frequency plan is available for {channels}.")

        return plan


@dataclass(slots=True)
class CoefficientPreparationPlan:
    """Coefficient requirements for spectra sharing one frequency plan.

    Attributes
    ----------
    frequency_plan : FFTFrequencyPlan | DirectFrequencyPlan
        Frequency grid shared by the associated requested spectra.
    required_channels : set[int]
        All channels referenced by those spectra. Every required channel needs a zero-frequency
        coefficient.
    direct_transform_channels : tuple[int, ...]
        Required timestamped channels whose coefficients must be calculated by direct
        transformation.
    band_coefficient_channels : set[int]
        Channels requiring coefficients on the output-frequency band. This includes every channel
        in second- and fourth-order spectra and the first two channel positions in third-order
        spectra.
    third_order_closing_frequency_channels : set[int]
        Channels occurring in the third position of a third-order spectrum and therefore requiring
        coefficients at ``f3 = -(f1 + f2)``.
    """

    frequency_plan: FrequencyPlan
    required_channels: set[int] = field(default_factory=set)
    direct_transform_channels: tuple[int, ...] = ()
    band_coefficient_channels: set[int] = field(default_factory=set)
    third_order_closing_frequency_channels: set[int] = field(default_factory=set)


def build_coefficient_preparation_plans(
    runtime: RuntimeConfig,
) -> dict[type[FFTFrequencyPlan | DirectFrequencyPlan], CoefficientPreparationPlan]:
    """Plan the Fourier coefficients required for every requested spectrum.

    Requested spectra are grouped by frequency-plan type because mixed calculations can use an FFT
    grid for spectra containing sampled channels and a direct-transform grid for timestamp-only
    spectra.

    For first-order spectra, only the zero-frequency coefficient is required. For second- and
    fourth-order spectra, every referenced channel requires output-band coefficients. For
    third-order spectra, the first two channels require output-band coefficients and the third
    requires coefficients at the closing frequencies ``f3 = -(f1 + f2)``.

    All required timestamped channels are marked for direct transformation. Sampled channels are
    transformed separately by the FFT path.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved channel types, requested spectra, and frequency plans.

    Returns
    -------
    dict[type[FFTFrequencyPlan | DirectFrequencyPlan], CoefficientPreparationPlan]
        One preparation plan for each frequency-plan type used by the requested spectra. Each plan
        records all required channels, timestamped channels that require direct transformation,
        channels needing output-band coefficients, and channels needing third-order
        closing-frequency coefficients.

    Raises
    ------
    RuntimeError
        If different frequency-plan objects of the same type occur in one runtime configuration.
    """

    preparation_plans_by_type: dict[
        type[FFTFrequencyPlan | DirectFrequencyPlan],
        CoefficientPreparationPlan,
    ] = {}

    for spectrum_channels in runtime.requested_spectra:
        frequency_plan = runtime.frequency_plan_for(spectrum_channels)
        plan_type = type(frequency_plan)
        preparation_plan = preparation_plans_by_type.get(plan_type)

        if preparation_plan is None:
            preparation_plan = CoefficientPreparationPlan(frequency_plan=frequency_plan)
            preparation_plans_by_type[plan_type] = preparation_plan
        elif preparation_plan.frequency_plan is not frequency_plan:
            raise RuntimeError(f"Multiple {plan_type.__name__} instances were planned.")

        preparation_plan.required_channels.update(spectrum_channels)
        order = len(spectrum_channels)

        if order == 3:
            preparation_plan.band_coefficient_channels.update(spectrum_channels[:2])
            preparation_plan.third_order_closing_frequency_channels.add(spectrum_channels[2])
        elif order > 1:
            preparation_plan.band_coefficient_channels.update(spectrum_channels)

    for preparation_plan in preparation_plans_by_type.values():
        preparation_plan.direct_transform_channels = tuple(
            channel
            for channel, channel_plan in runtime.channel_plans.items()
            if channel in preparation_plan.required_channels
            and isinstance(channel_plan, TimestampedChannelPlan)
        )

    return preparation_plans_by_type


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


def resolve_requested_spectra(
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
    timestamp_options: TimestampOptions | None,
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
            if timestamp_options is None:
                raise RuntimeError(
                    "Internal error: timestamped channel planning requires resolved "
                    "TimestampOptions."
                )

            plan = TimestampedChannelPlan(
                event_count=source_length,
                weighting=timestamp_options.weighting,
                scale=timestamp_options.scale,
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
    """Resolve and validate the common half-open observation interval.

    Timestamped calculations require both bounds explicitly. For sampled-only calculations,
    ``observation_start`` defaults to zero and ``observation_stop`` defaults to
    ``observation_start + sample_count * dt``.

    Whenever a sampled channel is active, the resolved interval duration must match the
    sampled-data duration. Only differences consistent with ULP-scale floating-point rounding are
    tolerated.

    Parameters
    ----------
    data_config : DataConfig
        Configured observation bounds and channel definitions.
    channel_plans : Mapping[int, ChannelPlan]
        Resolved plans for the active channels.

    Returns
    -------
    tuple[float, float]
        Resolved ``(observation_start, observation_stop)`` bounds.

    Raises
    ------
    ValueError
        If a timestamped calculation lacks an explicit bound, or if the observation duration does
        not match the duration of an active sampled channel.
    RuntimeError
        If no active sampled or timestamped channel plan is available.
    """

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

        if not math.isfinite(observation_duration) or not math.isfinite(sampled_duration):
            raise ValueError("The resolved observation or sampled duration is not finite.")

        duration_tolerance = 4.0 * max(
            math.ulp(float(observation_duration)),
            math.ulp(float(sampled_duration)),
        )

        if not math.isclose(
            observation_duration,
            sampled_duration,
            rel_tol=0.0,
            abs_tol=duration_tolerance,
        ):
            raise ValueError(
                f"The configured observation interval has duration {observation_duration}, but "
                f"{sampled_plan.sample_count} samples with dt={sampled_plan.dt} span "
                f"{sampled_duration}."
            )

    return observation_start, observation_stop


def resolve_sampled_frequencies(
    spectrum_config: SpectrumConfig,
    dt: float,
) -> tuple[int, FFTFrequencyPlan]:
    """Resolve the sampled FFT grid and selected output-frequency band.

    When ``SpectrumConfig.df`` is given, the physical-window length is calculated as
    ``round(1 / (dt * df))`` samples. Because this length must be an integer, the actual frequency
    spacing is ``1 / (window_points * dt)`` and may differ from the requested value. If ``df`` is
    omitted, the window length defaults to 1000 samples.

    The complete FFT grid is zero-anchored and shifted into ascending order. The output band
    contains every available FFT frequency within the inclusive configured bounds. An omitted
    ``f_max`` uses the sampled Nyquist bound.

    Parameters
    ----------
    spectrum_config : SpectrumConfig
        Requested frequency spacing and inclusive frequency bounds.
    dt : float
        Positive sampling interval shared by the active sampled channels.

    Returns
    -------
    tuple[int, FFTFrequencyPlan]
        Resolved number of samples per physical window and the shifted FFT plan containing the
        complete grid and selected output-band slice.

    Raises
    ------
    ValueError
        If the requested spacing resolves to a nonpositive window length, or if the requested
        frequency bounds contain no frequency on the resolved FFT grid.
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

    frequency_plan = FFTFrequencyPlan(
        shifted_full_fft_frequencies=freq_all,
        band_start=band_start,
        band_stop=band_stop,
    )

    return window_points, frequency_plan


def _resolve_timestamp_frequencies(
    *,
    f_min: float,
    f_max: float,
    window_duration: float,
) -> DirectFrequencyPlan:
    """Resolve the direct-transform frequency grid inside inclusive bounds.

    The grid is anchored at zero and consists of integer multiples of
    ``actual_df = 1 / window_duration``. In mixed calculations, this spacing is determined by the
    resolved sampled window and may differ from the originally requested ``SpectrumConfig.df``.

    Parameters
    ----------
    f_min, f_max : float
        Inclusive lower and upper frequency bounds.
    window_duration : float
        Positive duration of one physical window.

    Returns
    -------
    DirectFrequencyPlan
        Resolved frequency spacing and the ascending integer grid coordinates whose frequencies lie
        within ``[f_min, f_max]``.

    Raises
    ------
    ValueError
        If the requested bounds contain no frequency on the resolved grid.
    """

    actual_df = 1.0 / window_duration
    first_candidate = math.floor(f_min / actual_df) - 1
    last_candidate = math.ceil(f_max / actual_df) + 1

    grid_indices = np.arange(first_candidate, last_candidate + 1, dtype=np.int64)
    candidate_frequencies = grid_indices.astype(np.float64) * actual_df
    within_bounds = (candidate_frequencies >= f_min) & (candidate_frequencies <= f_max)
    band_frequencies = candidate_frequencies[within_bounds]
    band_grid_indices = grid_indices[within_bounds]

    if band_frequencies.size == 0:
        raise ValueError(
            f"The requested frequency band [{f_min}, {f_max}] does not contain "
            "any timestamp frequencies at the resolved frequency spacing."
        )

    return DirectFrequencyPlan(
        actual_df=actual_df,
        grid_indices=band_grid_indices,
    )


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
    """Resolve estimate counts, interlacing, batching, and uncertainty grouping.

    The effective number of physical windows per estimate is initially ``SpectrumConfig.m`` and is
    reduced when fewer unshifted windows are available. It must still be at least the highest
    requested spectrum order.

    Complete physical windows are grouped into estimates, and the number of unshifted estimates is
    limited by ``spectral_estimates_max``. When interlacing is enabled, the shifted count is
    limited to the smaller of the available shifted and selected unshifted counts.

    In short-term uncertainty mode, ``m_var`` may be reduced to the available unshifted estimate
    count. A processing batch large enough to contain a complete uncertainty group is rounded down
    to a multiple of the effective ``m_var``.

    Parameters
    ----------
    spectrum_config : SpectrumConfig
        Requested window count, estimate limit, batching, interlacing, and
        uncertainty settings.
    observation_start, observation_stop : float
        Bounds of the common half-open observation interval.
    window_duration : float
        Duration of one physical window.
    interlacing_offset : float
        Offset applied to shifted physical windows.
    available_unshifted_windows : int
        Number of complete physical windows in the unshifted placement.
    available_shifted_windows : int
        Number of complete physical windows in the shifted placement.
    orders : tuple[int, ...]
        Distinct requested spectrum orders.

    Returns
    -------
    tuple[WindowPlan, int]
        Resolved window plan and effective ``m_var``.

    Warns
    -----
    UserWarning
        If ``m`` is reduced because too few physical windows are available, or if
        ``m_var`` is reduced because too few unshifted estimates are available.

    Raises
    ------
    ValueError
        If the effective number of windows per estimate is below the highest
        requested order, or if interlacing is requested but no complete shifted
        estimate fits.
    """

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
    """Resolve user configuration and active sources into runtime calculation settings.

    Only channels present in ``opened_channels`` participate in planning. The function validates
    their shared timing constraints, resolves the observation interval, constructs sampled and
    direct frequency grids, determines the physical window layout and batching, resolves timestamp
    amplitude repetitions, checks the requested device, and selects the calculation dtypes.

    Active sampled channels must have equal lengths and sampling intervals. Their sample count and
    interval must span the configured observation interval. Timestamped calculations require
    explicit observation bounds, and timestamp-only calculations additionally require explicit
    ``df`` and ``f_max``.

    Parameters
    ----------
    data_config : DataConfig
        Channel definitions, observation bounds, and shared time unit.
    opened_channels : Mapping[int, RuntimeSource]
        Opened runtime sources keyed by their indices in ``data_config.channels``. Only these
        channels are considered active.
    spectrum_config : SpectrumConfig
        Requested frequency bounds, windowing, repetition, uncertainty, precision, batching,
        interlacing, and device options.
    spectra_channels : tuple[tuple[int, ...], ...]
        Validated channel tuples identifying the requested spectra.

    Returns
    -------
    RuntimeConfig
        Resolved channel plans, frequency plans, window and repetition plans, uncertainty settings,
        frequency unit, dtypes, and calculation device.

    Warns
    -----
    UserWarning
        If ``m`` is reduced because too few physical windows are available, or if ``m_var`` is
        reduced because too few unshifted spectral estimates are available.

    Raises
    ------
    TypeError
        If an active timestamp source contains Boolean values or an active channel has an
        unsupported type.
    ValueError
        If channel timing is inconsistent; timestamp options or observation bounds are invalid; an
        active timestamp lies outside the observation interval or cannot be represented safely;
        frequency or window resolution fails; the effective ``m`` is below the highest requested
        order; interlacing cannot produce an estimate; or the device specification is invalid.
    RuntimeError
        If the requested accelerator is unavailable, its device index is invalid, or the runtime
        configuration is internally inconsistent.
    """

    # Validate and read the channels, number of data points, and the time step from the
    # SpectrumConfig and DataConfigs
    active_data_channels = tuple(opened_channels)
    has_timestamped_channel = any(
        isinstance(data_config.channels[channel], TimestampedChannel)
        for channel in active_data_channels
    )
    timestamp_options = spectrum_config.timestamp_options

    if has_timestamped_channel and timestamp_options is None:
        raise ValueError("TimestampOptions are required when an active channel is timestamped.")

    if not has_timestamped_channel and timestamp_options is not None:
        raise ValueError("TimestampOptions cannot be used in a sampled-only calculation.")

    channel_plans, t_unit = _build_channel_plans(
        data_config=data_config,
        opened_channels=opened_channels,
        timestamp_options=timestamp_options,
    )

    observation_start, observation_stop = _resolve_observation_interval(data_config, channel_plans)
    for channel, channel_plan in channel_plans.items():
        source = opened_channels[channel]

        # In-memory sampled sources were already validated by SampledChannel.
        # HDF5 sources require a bounded scan after the dataset has been opened.
        if isinstance(channel_plan, SampledChannelPlan) and isinstance(source, HDF5SourceState):
            validate_sampled_hdf5_source(source, label=f"Sampled channel {channel}")
        if isinstance(channel_plan, TimestampedChannelPlan):
            validate_timestamp_source(
                opened_channels[channel],
                observation_start,
                observation_stop,
                label=f"Timestamped channel {channel}",
            )
    repetition_plan = _resolve_repetition_plan(timestamp_options)
    orders = tuple(sorted({len(channels) for channels in spectra_channels}))

    sampled_plan = next(
        (plan for plan in channel_plans.values() if isinstance(plan, SampledChannelPlan)),
        None,
    )

    if sampled_plan is None:
        if spectrum_config.df is None or spectrum_config.f_max is None:
            raise ValueError("Timestamp-only calculations require explicit df and f_max.")

        fft_frequency_plan = None
        window_duration = 1.0 / spectrum_config.df
        interlacing_offset = window_duration / 2.0
        observation_duration = observation_stop - observation_start

        available_unshifted_windows = _count_complete_windows(observation_duration, window_duration)
        available_shifted_windows = _count_complete_windows(
            observation_duration - interlacing_offset,
            window_duration,
        )
    else:
        window_points, fft_frequency_plan = resolve_sampled_frequencies(
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

    direct_frequency_plan = None
    if has_timestamped_channel:
        timestamp_f_max = spectrum_config.f_max
        if timestamp_f_max is None:
            assert sampled_plan is not None
            timestamp_f_max = 1.0 / (2.0 * sampled_plan.dt)

        direct_frequency_plan = _resolve_timestamp_frequencies(
            f_min=spectrum_config.f_min,
            f_max=timestamp_f_max,
            window_duration=window_duration,
        )

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
        requested_spectra=spectra_channels,
        fft_frequency_plan=fft_frequency_plan,
        direct_frequency_plan=direct_frequency_plan,
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
    """Yield physical-window batches in observation-relative time.

    Each row of ``WindowBatch.relative_starts`` describes one spectral estimate and contains the
    starts of its ``m = plan.windows_per_estimate`` consecutive physical windows. Starts are
    measured relative to ``plan.observation_start``.

    Unshifted batches are yielded first in chronological order. If interlacing is enabled, shifted
    batches follow in chronological order with ``plan.interlacing_offset`` applied. Each batch
    contains at most ``plan.estimates_per_batch`` estimates.

    Parameters
    ----------
    plan : WindowPlan
        Resolved window duration, placement counts, interlacing offset, windows per estimate, and
        estimate batch size.

    Yields
    ------
    WindowBatch
        A batch whose ``relative_starts`` has shape ``(B, m)``, where ``B`` equals
        ``estimate_count``. The ``shifted`` flag identifies its placement group.
    """

    def iter_placement(
        estimate_count: int,
        relative_offset: float,
        shifted: bool,
    ) -> Iterator[WindowBatch]:
        windows_per_estimate = plan.windows_per_estimate

        for first_estimate in range(0, estimate_count, plan.estimates_per_batch):
            batch_size = min(plan.estimates_per_batch, estimate_count - first_estimate)
            first_window_index = first_estimate * windows_per_estimate
            stop_window_index = (first_estimate + batch_size) * windows_per_estimate

            boundary_indices = np.arange(first_window_index, stop_window_index + 1, dtype=np.int64)
            boundaries = relative_offset + boundary_indices.astype(np.float64) * plan.duration
            relative_starts = boundaries[:-1].reshape(batch_size, windows_per_estimate)

            yield WindowBatch(
                relative_starts=relative_starts,
                duration=plan.duration,
                estimate_count=batch_size,
                shifted=shifted,
                relative_stop=float(boundaries[-1]),
            )

    yield from iter_placement(plan.unshifted_estimate_count, relative_offset=0.0, shifted=False)

    yield from iter_placement(
        plan.shifted_estimate_count,
        relative_offset=plan.interlacing_offset,
        shifted=True,
    )
