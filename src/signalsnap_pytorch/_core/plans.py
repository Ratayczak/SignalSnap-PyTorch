# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .utils import FrequencyUnits


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