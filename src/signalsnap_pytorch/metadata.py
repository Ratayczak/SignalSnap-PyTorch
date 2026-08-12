# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ._core.planning import (
    FFTFrequencyPlan,
    RuntimeConfig,
    SampledChannelPlan,
    TimestampedChannelPlan,
)
from ._core.utils import FrequencyUnits as _FrequencyUnits
from ._core.utils import TimeUnits as _TimeUnits
from .configurators import DataConfig, PhotonOptions, SampledChannel, SpectrumConfig

__all__ = ["CalculationMetadata", "SpectrumMetadata"]

_ChannelKind = Literal["sampled", "timestamped"]
_FrequencyView = Literal["sampled_fft", "direct_transform"]
_NormalizationConvention = Literal[
    "sampled_discrete_default",
    "sampled_discrete_legacy",
    "timestamp_continuous_default",
    "timestamp_fixed_grid_legacy",
    "mixed_discrete_overlap_default",
    "mixed_discrete_overlap_legacy",
]
_ClosingFrequencySupport = Literal["sampled_fft", "direct_transform", "not_applicable"]
_WindowConvention = Literal["confined_gaussian", "legacy_confined_gaussian"]
_PhotonWeighting = Literal["unit", "exponential"]
_UncertaintyEstimation = Literal["global", "short_term"]


@dataclass(frozen=True, slots=True)
class SpectrumMetadata:
    """Immutable metadata describing one planned spectrum.

    An instance is created for every requested spectrum during calculation planning. Its presence
    does not imply that the corresponding calculation completed successfully.

    Attributes
    ----------
    channels : tuple[int, ...]
        Channel tuple defining the requested spectrum.
    frequency_view : Literal["sampled_fft", "direct_transform"]
        Convention used for the output frequency view. ``"sampled_fft"`` denotes a frequency view
        constrained by a sampled-channel FFT grid. ``"direct_transform"`` denotes direct evaluation
        of timestamped data.
    effective_f_min : float
        Lower bound of the output-frequency view. For a successful result, this equals the first
        value of ``result.freq``. It is zero for a first-order spectrum.
    effective_f_max : float
        Upper bound of the output-frequency view. For a successful result, this equals the last
        value of ``result.freq``. It is zero for a first-order spectrum.
    normalization_convention : Literal[
        "sampled_discrete_default",
        "sampled_discrete_legacy",
        "timestamp_continuous_default",
        "timestamp_fixed_grid_legacy",
        "mixed_discrete_overlap_default",
        "mixed_discrete_overlap_legacy",
    ]
        Window-normalization convention used for this spectrum. The value records both the
        participating channel kinds and whether the default or legacy confined-Gaussian window was
        selected.
    closing_frequency_support : Literal[
        "sampled_fft",
        "direct_transform",
        "not_applicable",
    ]
        Frequency-support convention for the closing frequency of a third-order spectrum.
        ``"sampled_fft"`` means that the closing channel is restricted to sampled FFT support.
        ``"direct_transform"`` means that its timestamped closing channel is evaluated directly.
        ``"not_applicable"`` is used for spectrum orders other than three.
    """

    channels: tuple[int, ...]
    frequency_view: _FrequencyView
    effective_f_min: float
    effective_f_max: float
    normalization_convention: _NormalizationConvention
    closing_frequency_support: _ClosingFrequencySupport

    @property
    def order(self) -> int:
        """Return the requested spectrum order."""
        return len(self.channels)


@dataclass(frozen=True, slots=True)
class CalculationMetadata:
    """Calculation-wide metadata shared by all returned results.

    This object records the resolved calculation plan and the user settings needed to interpret or
    reproduce the calculation. Per-spectrum conventions and frequency bounds are stored separately
    in :class:`SpectrumMetadata`.

    Attributes
    ----------
    channel_kinds : tuple[Literal["sampled", "timestamped"], ...]
        Kind of every configured data channel, in data-configuration order.
    active_channels : tuple[int, ...]
        Indices of the channels used by at least one requested spectrum.
    requested_spectra : tuple[tuple[int, ...], ...]
        Resolved channel tuples for every planned spectrum, in request order. Individual
        calculations may fail, so not every tuple is necessarily present in the final result store's
        ``results`` mapping.
    observation_start : int | float
        Start of the resolved common observation interval, expressed in ``time_unit``.
    observation_stop : int | float
        End of the resolved common observation interval, expressed in ``time_unit``.
    time_unit : Literal["s", "ms", "us", "ns", "ps", "fs"]
        Unit of observation times, window durations, and placement offsets.
    frequency_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of frequency-related values and returned frequency axes.
    requested_df : float | None
        Frequency spacing requested by the user. ``None`` indicates that no explicit spacing was
        requested.
    actual_df : float
        Resolved Fourier frequency spacing, equal to the reciprocal of the physical window duration
        after unit conversion.
    requested_f_min : float
        Lower frequency bound requested by the user.
    requested_f_max : float | None
        Upper frequency bound requested by the user. ``None`` indicates that no explicit upper bound
        was configured.
    window_duration : float
        Duration of one physical coefficient window, expressed in ``time_unit``.
    unshifted_offset : float
        Placement offset of the unshifted window group. This is zero.
    shifted_offset : float | None
        Placement offset of the interlaced window group, expressed in ``time_unit``, or ``None``
        when interlacing is disabled.
    window_convention : Literal[
        "confined_gaussian",
        "legacy_confined_gaussian",
    ]
        Window convention used by the calculation.
    photon_weighting : Literal["unit", "exponential"] | None
        Timestamp amplitude model, or ``None`` for a sampled-only calculation.
    exponential_scale : float | None
        Scale of exponentially distributed timestamp amplitudes, or ``None`` when exponential
        weighting is not used.
    repetition_count : int
        Number of timestamp-amplitude realizations. This is one when exponential weighting is not
        used.
    requested_repetition_batch_size : int | None
        Repetition batch size configured by the user, or ``None`` when exponential weighting is not
        used.
    resolved_repetition_batch_size : int
        Repetition batch size actually used by the calculation.
    user_seed : int | None
        Random seed supplied by the user for exponential timestamp weighting. This is ``None`` when
        no seed was supplied or exponential weighting was not used.
    resolved_seed : int | None
        Random seed actually used for exponential timestamp weighting, including an automatically
        generated seed. This is ``None`` when exponential weighting was not used.
    requested_m : int
        Number of coefficient windows per spectral estimate requested by the user.
    effective_m : int
        Number of coefficient windows per spectral estimate actually used.
    requested_m_var : int
        Short-term uncertainty group size requested by the user. This setting is ignored when
        ``uncertainty_estimation`` is ``"global"``.
    effective_m_var : int
        Short-term uncertainty group size resolved during planning. This may be smaller than
        ``requested_m_var`` when fewer unshifted estimates are available. It is ignored when
        ``uncertainty_estimation`` is ``"global"``.
    uncertainty_estimation : Literal["global", "short_term"]
        Uncertainty-estimation method used by the calculation.
    unshifted_physical_estimate_count : int
        Number of unshifted spectral estimates available to the calculation.
    shifted_physical_estimate_count : int
        Number of interlaced spectral estimates available to the calculation. This is zero when
        interlacing is disabled.
    unshifted_coefficient_window_count : int
        Total number of coefficient windows contributing to unshifted estimates.
    shifted_coefficient_window_count : int
        Total number of coefficient windows contributing to interlaced estimates. This is zero when
        interlacing is disabled.
    real_dtype : str
        Resolved real-valued calculation dtype.
    complex_dtype : str
        Resolved complex-valued calculation dtype.
    requested_device : str
        PyTorch device requested by the user.
    resolved_device : str
        PyTorch device actually used after device resolution.
    """

    channel_kinds: tuple[_ChannelKind, ...]
    active_channels: tuple[int, ...]
    requested_spectra: tuple[tuple[int, ...], ...]

    observation_start: int | float
    observation_stop: int | float
    time_unit: _TimeUnits
    frequency_unit: _FrequencyUnits

    requested_df: float | None
    actual_df: float
    requested_f_min: float
    requested_f_max: float | None

    window_duration: float
    unshifted_offset: float
    shifted_offset: float | None
    window_convention: _WindowConvention

    photon_weighting: _PhotonWeighting | None
    exponential_scale: float | None
    repetition_count: int
    requested_repetition_batch_size: int | None
    resolved_repetition_batch_size: int
    user_seed: int | None
    resolved_seed: int | None

    requested_m: int
    effective_m: int
    requested_m_var: int
    effective_m_var: int
    uncertainty_estimation: _UncertaintyEstimation
    unshifted_physical_estimate_count: int
    shifted_physical_estimate_count: int
    unshifted_coefficient_window_count: int
    shifted_coefficient_window_count: int

    real_dtype: str
    complex_dtype: str
    requested_device: str
    resolved_device: str


def _build_spectrum_metadata(runtime: RuntimeConfig, channels: tuple[int, ...]) -> SpectrumMetadata:
    """Build metadata for one planned spectrum.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved runtime plan for the complete calculation.
    channels : tuple[int, ...]
        Channel tuple identifying the planned spectrum.

    Returns
    -------
    SpectrumMetadata
        Metadata describing the spectrum's frequency view, effective bounds, normalization, and
        closing-frequency support.
    """
    frequency_plan = runtime.frequency_plan_for(channels)
    channel_plans = tuple(runtime.channel_plans[channel] for channel in channels)

    if isinstance(frequency_plan, FFTFrequencyPlan):
        frequency_view = "sampled_fft"
    else:
        frequency_view = "direct_transform"

    if len(channels) == 1:
        effective_f_min = 0.0
        effective_f_max = 0.0
    else:
        frequencies = frequency_plan.band_frequencies
        effective_f_min = float(frequencies[0])
        effective_f_max = float(frequencies[-1])

    all_sampled = all(
        isinstance(channel_plan, SampledChannelPlan) for channel_plan in channel_plans
    )
    all_timestamped = all(
        isinstance(channel_plan, TimestampedChannelPlan) for channel_plan in channel_plans
    )

    if all_sampled:
        normalization_convention = (
            "sampled_discrete_legacy" if runtime.old_window else "sampled_discrete_default"
        )
    elif all_timestamped:
        normalization_convention = (
            "timestamp_fixed_grid_legacy" if runtime.old_window else "timestamp_continuous_default"
        )
    else:
        normalization_convention = (
            "mixed_discrete_overlap_legacy"
            if runtime.old_window
            else "mixed_discrete_overlap_default"
        )

    if len(channels) != 3:
        closing_frequency_support = "not_applicable"
    elif isinstance(runtime.channel_plans[channels[2]], SampledChannelPlan):
        closing_frequency_support = "sampled_fft"
    else:
        closing_frequency_support = "direct_transform"

    return SpectrumMetadata(
        channels=channels,
        frequency_view=frequency_view,
        effective_f_min=effective_f_min,
        effective_f_max=effective_f_max,
        normalization_convention=normalization_convention,
        closing_frequency_support=closing_frequency_support,
    )


def _photon_values(
    photon_options: PhotonOptions | None,
) -> tuple[_PhotonWeighting | None, float | None, int | None, int | None]:
    """Extract durable photon-weighting settings from the configuration.

    Parameters
    ----------
    photon_options : PhotonOptions | None
        Configured timestamp-weighting options, or ``None`` for a calculation without photon
        weighting.

    Returns
    -------
    tuple[
        Literal["unit", "exponential"] | None,
        float | None,
        int | None,
        int | None,
    ]
        Photon-weighting convention, exponential scale, requested repetition batch size, and
        user-supplied seed, respectively. All values are ``None`` when ``photon_options`` is
        ``None``.
    """
    if photon_options is None:
        return None, None, None, None

    return (
        photon_options.weighting,
        photon_options.scale,
        photon_options.repetitions_per_batch,
        photon_options.seed,
    )


def build_result_metadata(
    data_config: DataConfig,
    spectrum_config: SpectrumConfig,
    runtime: RuntimeConfig,
) -> tuple[CalculationMetadata, Mapping[tuple[int, ...], SpectrumMetadata]]:
    """Build the public metadata for a successfully planned calculation.

    Metadata is constructed before individual spectra are evaluated. Consequently, the returned
    spectrum-metadata mapping describes every planned spectrum, including spectra that may later
    fail at an isolated failure boundary.

    Parameters
    ----------
    data_config : DataConfig
        Validated input-channel configuration.
    spectrum_config : SpectrumConfig
        User configuration for the spectral calculation.
    runtime : RuntimeConfig
        Resolved runtime plan derived from the data and spectrum configurations.

    Returns
    -------
    calculation_metadata : CalculationMetadata
        Calculation-wide metadata shared by the result store and all successfully returned results.
    spectra_metadata : Mapping[tuple[int, ...], SpectrumMetadata]
        Mapping from every requested channel tuple to its spectrum-specific metadata. Iteration
        order matches the resolved request order.
    """
    spectra_metadata = MappingProxyType(
        {
            channels: _build_spectrum_metadata(runtime, channels)
            for channels in runtime.requested_spectra
        }
    )
    photon_weighting, exponential_scale, requested_batch_size, user_seed = _photon_values(
        spectrum_config.photon_options
    )
    window_plan = runtime.window_plan
    repetition_plan = runtime.repetition_plan

    channel_kinds: tuple[_ChannelKind, ...] = tuple(
        "sampled" if isinstance(channel, SampledChannel) else "timestamped"
        for channel in data_config.channels
    )

    calculation_metadata = CalculationMetadata(
        channel_kinds=channel_kinds,
        active_channels=runtime.active_data_channels,
        requested_spectra=runtime.requested_spectra,
        observation_start=window_plan.observation_start,
        observation_stop=window_plan.observation_stop,
        time_unit=data_config.t_unit,
        frequency_unit=runtime.freq_unit,
        requested_df=spectrum_config.df,
        actual_df=1.0 / window_plan.duration,
        requested_f_min=spectrum_config.f_min,
        requested_f_max=spectrum_config.f_max,
        window_duration=window_plan.duration,
        unshifted_offset=0.0,
        shifted_offset=(window_plan.interlacing_offset if spectrum_config.interlacing else None),
        window_convention=(
            "legacy_confined_gaussian" if runtime.old_window else "confined_gaussian"
        ),
        photon_weighting=photon_weighting,
        exponential_scale=exponential_scale,
        repetition_count=repetition_plan.count,
        requested_repetition_batch_size=requested_batch_size,
        resolved_repetition_batch_size=repetition_plan.batch_size,
        user_seed=user_seed,
        resolved_seed=repetition_plan.resolved_seed,
        requested_m=spectrum_config.m,
        effective_m=window_plan.windows_per_estimate,
        requested_m_var=spectrum_config.m_var,
        effective_m_var=runtime.m_var,
        uncertainty_estimation=runtime.uncertainty_estimation,
        unshifted_physical_estimate_count=window_plan.unshifted_estimate_count,
        shifted_physical_estimate_count=window_plan.shifted_estimate_count,
        unshifted_coefficient_window_count=(
            window_plan.unshifted_estimate_count * window_plan.windows_per_estimate
        ),
        shifted_coefficient_window_count=(
            window_plan.shifted_estimate_count * window_plan.windows_per_estimate
        ),
        real_dtype=str(runtime.real_dtype),
        complex_dtype=str(runtime.complex_dtype),
        requested_device=spectrum_config.device,
        resolved_device=str(runtime.device),
    )

    return calculation_metadata, spectra_metadata
