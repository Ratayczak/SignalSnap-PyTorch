# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from ..results import SpectrumResult
from .utils import FrequencyUnits

if TYPE_CHECKING:
    from .planning import RuntimeConfig


@dataclass(slots=True)
class ShortTermUncertaintyState:
    """Tracks the calculation of Welford's online variance algorithm.

    Attributes
    ----------
    current_count : int = 0
        Number of estimates incorporated into the current incomplete batch.
        Reset to zero whenever a batch is completed.
    current_mean_re, current_mean_im : Tensor | None = None
        Running component-wise means for the current batch.
    current_m2_re, current_m2_im : Tensor | None = None
        Welford sums of squared deviations for the current batch.
    variance_sum_re, variance_sum_im : Tensor | None = None
        Sums of the real and imaginary variance-of-mean estimates from all completed batches.
    completed_batches : int = 0
        Number of complete short-term batches.
    """

    current_count: int = 0

    current_mean_re: Tensor | None = None
    current_mean_im: Tensor | None = None
    current_m2_re: Tensor | None = None
    current_m2_im: Tensor | None = None

    variance_sum_re: Tensor | None = None
    variance_sum_im: Tensor | None = None

    completed_batches: int = 0


@dataclass(slots=True)
class GroupAccumulator:
    """Accumulator for the group of either unshifted or shifted spectral estimates.

    Attributes
    ----------
    spectrum_sum : Tensor | None = None
        Running total sum of the calculated spectra on the active torch device.
    count : int = 0
        Number of accumulated spectral estimates.
    squared_sum : Tensor | None = None
        Running total squared sum of the real and imaginary parts of the calculated spectra on the
        active torch device. Real and imaginary parts are squared separately.
    short_term : :class:`ShortTermUncertaintyState`
        State used to construct short-term uncertainty estimates. It remains empty in global mode.
    """

    spectrum_sum: Tensor | None = None
    count: int = 0

    squared_sum: Tensor | None = None

    short_term: ShortTermUncertaintyState = field(default_factory=ShortTermUncertaintyState)


@dataclass(slots=True)
class SpectrumAccumulator:
    """Data container for the accumulation of spectral estimates.

    Stores the configuration metadata, accumulated hardware states, and uncertainty buffers for a
    specific higher-order auto- or cross-spectrum calculation.

    Attributes
    ----------
    channels : tuple[int, ...]
        The indices identifying which channels are part of this calculation. For example,
        ``(0, 0, 0)`` indicates a third-order auto-spectrum on channel 0, while ``(0, 1)`` indicates
        a cross-spectrum between channels 0 and 1.
    freq : np.ndarray
        Frequency axis associated with the spectrum.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    uncertainty_estimation : Literal["global", "short_term"]
        "global": estimate the standard error based on all spectral estimates. "short_term":
        estimate a typical local uncertainty for ``m_var`` spectral estimates at a time. At the end
        all variance-of-mean estimates are averaged before taking the square root.
    m_var : int
        Number of spectral estimates per short-term uncertainty batch used at runtime.
    unshifted : :class:`GroupAccumulator`
        Accumulator for the unshifted spectral estimates.
    shifted : :class:`GroupAccumulator`
        Accumulator for the shifted spectral estimates.
    """

    channels: tuple[int, ...]
    freq: np.ndarray
    freq_unit: FrequencyUnits

    uncertainty_estimation: Literal["global", "short_term"]
    m_var: int

    unshifted: GroupAccumulator = field(default_factory=GroupAccumulator)
    shifted: GroupAccumulator = field(default_factory=GroupAccumulator)

    @property
    def order(self) -> int:
        """Order of the accumulated spectrum, derived from its channel tuple."""
        return len(self.channels)


@dataclass(slots=True)
class SpectrumAccumulatorStore:
    """Container for all :class:`SpectrumAccumulator` used by a calculation pipeline.

    Stores one :class:`SpectrumAccumulator` per channel tuple. Accumulators are indexed by
    ``channels``, where ``channels`` is a tuple of data-channel indices.

    This class owns collection-level bookkeeping only. Numerical accumulation, uncertainty
    estimation, and finalization are handled elsewhere.

    Attributes
    ----------
    accumulators : dict[tuple[int, ...], SpectrumAccumulator]
        Mapping from ``channels`` to the corresponding :class:`SpectrumAccumulator`. For example,
        ``(0, 0)`` identifies the second-order auto-spectrum of channel 0, while ``(0, 1)``
        identifies a second-order cross-spectrum between channels 0 and 1.
    """

    accumulators: dict[tuple[int, ...], SpectrumAccumulator] = field(default_factory=dict)

    def __iter__(self) -> Iterator[SpectrumAccumulator]:
        """Iterate over accumulators in insertion order."""
        return iter(self.accumulators.values())

    def get(self, channels: tuple[int, ...]) -> SpectrumAccumulator:
        """Return the accumulator for a channel tuple.

        Raises
        ------
        KeyError
            If ``channels`` is not present.
        """
        return self.accumulators[channels]

    def add(self, accumulator: SpectrumAccumulator) -> None:
        """Add an accumulator, replacing one with the same channel tuple."""
        self.accumulators[accumulator.channels] = accumulator


def initialize_accumulator_store(runtime: RuntimeConfig) -> SpectrumAccumulatorStore:
    """Create an initialized :class:`SpectrumAccumulatorStore` for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumAccumulator` with matching channels.

    Parameters
    ----------
    runtime : :class:`RuntimeConfig`
        :class:`RuntimeConfig` that contains all necessary information to initialize a
        :class:`SpectrumAccumulator`.

    Returns
    -------
    SpectrumAccumulatorStore
        Store containing one initialized :class:`SpectrumAccumulator` per task.
    """

    store = SpectrumAccumulatorStore()
    for channels in runtime.spectra_channels:
        freq = np.asarray([0.0]) if len(channels) == 1 else runtime.freq_band
        store.add(
            SpectrumAccumulator(
                channels,
                freq=freq,
                freq_unit=runtime.freq_unit,
                uncertainty_estimation=runtime.uncertainty_estimation,
                m_var=runtime.m_var,
            )
        )
    return store


def _get_group_accumulator(accumulator: SpectrumAccumulator, shifted: bool) -> GroupAccumulator:
    """Helper to return the specified :class:`GroupAccumulator`. This keeps branching out of the
    code.
    """

    return accumulator.shifted if shifted else accumulator.unshifted


def _accumulate_global_uncertainty(accumulator: GroupAccumulator, spectrum: Tensor) -> None:
    """Accumulate the squared sum of :class:`GroupAccumulator`. Real and imaginary squared sums are
    encoded as one complex number, which should not be interpreted as a complex number.
    """

    squared = torch.complex(torch.square(spectrum.real), torch.square(spectrum.imag))

    if accumulator.squared_sum is None:
        accumulator.squared_sum = squared
    else:
        accumulator.squared_sum += squared


def _complete_short_term_batch(state: ShortTermUncertaintyState, m_var: int) -> None:
    """Calculate the variance-of-mean for one completed uncertainty batch."""

    assert state.current_m2_re is not None
    assert state.current_m2_im is not None

    # divide by (n-1) to obtain variance and divide by n to obtain the variance-of-mean
    denominator = m_var * (m_var - 1)

    variance_re = state.current_m2_re / denominator
    variance_im = state.current_m2_im / denominator

    if state.variance_sum_re is None:
        state.variance_sum_re = variance_re.clone()
        state.variance_sum_im = variance_im.clone()
    else:
        assert state.variance_sum_im is not None
        state.variance_sum_re += variance_re
        state.variance_sum_im += variance_im

    state.completed_batches += 1
    state.current_count = 0


def _accumulate_short_term_uncertainty(
    state: ShortTermUncertaintyState, spectrum: Tensor, m_var: int
) -> None:
    """One step in Welford's online variance algorithm."""

    state.current_count += 1
    count = state.current_count

    if count == 1:
        if state.current_mean_re is None:
            state.current_mean_re = spectrum.real.clone()
            state.current_mean_im = spectrum.imag.clone()
            state.current_m2_re = torch.zeros_like(spectrum.real)
            state.current_m2_im = torch.zeros_like(spectrum.imag)
        else:
            assert state.current_mean_im is not None
            assert state.current_m2_re is not None
            assert state.current_m2_im is not None

            state.current_mean_re.copy_(spectrum.real)
            state.current_mean_im.copy_(spectrum.imag)
            state.current_m2_re.zero_()
            state.current_m2_im.zero_()
    else:
        assert state.current_mean_re is not None
        assert state.current_m2_re is not None
        assert state.current_mean_im is not None
        assert state.current_m2_im is not None

        delta_re = spectrum.real - state.current_mean_re
        state.current_mean_re += delta_re / count
        state.current_m2_re += delta_re * (spectrum.real - state.current_mean_re)

        delta_im = spectrum.imag - state.current_mean_im
        state.current_mean_im += delta_im / count
        state.current_m2_im += delta_im * (spectrum.imag - state.current_mean_im)

    if count == m_var:
        _complete_short_term_batch(state, m_var)


def accumulate_spectrum(
    accumulator: SpectrumAccumulator, single_spectrum: Tensor, shifted: bool = False
) -> None:
    """Accumulate one spectral estimate into the :class:`SpectrumAccumulator`.

    Adds the spectral estimate to the running sum and updates either global squared sums or
    short-term Welford states. Spectral estimates and their squared components are accumulated
    separately for shifted and unshifted data.

    Parameters
    ----------
    accumulator : SpectrumAccumulator
        Mutable accumulation state for one requested spectrum.
    single_spectrum : Tensor
        One spectral estimate. Its shape must match prior estimates accumulated into the same group.
    shifted : bool, default=False
        Store the estimate in the shifted interlacing group instead of the unshifted group.
    """

    group = _get_group_accumulator(accumulator, shifted)

    if group.spectrum_sum is None:
        group.spectrum_sum = single_spectrum.clone()
    else:
        group.spectrum_sum += single_spectrum

    group.count += 1

    if accumulator.uncertainty_estimation == "global":
        _accumulate_global_uncertainty(group, single_spectrum)
    elif accumulator.uncertainty_estimation == "short_term":
        _accumulate_short_term_uncertainty(group.short_term, single_spectrum, accumulator.m_var)
    else:
        raise RuntimeError(
            f"Unknown uncertainty-estimation method {accumulator.uncertainty_estimation!r}."
        )


def _check_accumulator_group(
    group: GroupAccumulator, *, uncertainty_estimation: Literal["global", "short_term"], m_var: int
) -> GroupAccumulator | None:
    """Validate one shifted or unshifted accumulator group.

    Returns ``None`` for a consistently empty group and the validated group otherwise.

    Raises
    ------
    RuntimeError
        If the group state is inconsistent with its configured uncertainty-estimation method.
    """
    if uncertainty_estimation not in {"global", "short_term"}:
        raise RuntimeError(f"Unknown uncertainty-estimation method {uncertainty_estimation!r}.")

    if uncertainty_estimation == "short_term" and m_var < 2:
        raise RuntimeError("Short-term uncertainty estimation requires m_var >= 2.")

    st_state = group.short_term

    welford_tensors = (
        st_state.current_mean_re,
        st_state.current_mean_im,
        st_state.current_m2_re,
        st_state.current_m2_im,
    )
    variance_tensors = (st_state.variance_sum_re, st_state.variance_sum_im)
    short_term_tensors = welford_tensors + variance_tensors

    # Validate an empty group.
    if group.spectrum_sum is None:
        if (
            group.count != 0
            or group.squared_sum is not None
            or st_state.current_count != 0
            or st_state.completed_batches != 0
            or any(tensor is not None for tensor in short_term_tensors)
        ):
            raise RuntimeError("Spectrum accumulator group is inconsistent.")

        return None

    # Validate state shared by both uncertainty-estimation methods.
    if group.count <= 0:
        raise RuntimeError("Spectrum accumulator group is inconsistent.")

    expected_shape = group.spectrum_sum.shape

    if uncertainty_estimation == "global":
        if group.squared_sum is None:
            raise RuntimeError("Global uncertainty accumulator has no squared-sum state.")

        if group.squared_sum.shape != expected_shape:
            raise RuntimeError("Global squared-sum shape does not match the spectrum-sum shape.")

        if (
            st_state.current_count != 0
            or st_state.completed_batches != 0
            or any(tensor is not None for tensor in short_term_tensors)
        ):
            raise RuntimeError("Short-term state must remain empty in global uncertainty mode.")

        return group

    # Short-term mode must not populate the global squared-sum state.
    if group.squared_sum is not None:
        raise RuntimeError("Global squared-sum state must remain empty in short-term mode.")

    expected_batches = group.count // m_var
    expected_remainder = group.count % m_var

    if st_state.completed_batches != expected_batches:
        raise RuntimeError(
            "Completed short-term batch count is inconsistent with the "
            "number of accumulated spectra."
        )

    if st_state.current_count != expected_remainder:
        raise RuntimeError(
            "Current short-term batch count is inconsistent with the number of accumulated spectra."
        )

    # A populated short-term group always retains its Welford buffers,
    # including after a batch has just been completed.
    if any(tensor is None for tensor in welford_tensors):
        raise RuntimeError("Populated short-term state is missing Welford buffers.")

    for tensor in welford_tensors:
        assert tensor is not None
        if tensor.shape != expected_shape:
            raise RuntimeError(
                "Short-term Welford-buffer shape does not match the spectrum-sum shape."
            )

    if st_state.completed_batches == 0:
        if any(tensor is not None for tensor in variance_tensors):
            raise RuntimeError("Short-term variance sums exist without a completed batch.")
    else:
        if any(tensor is None for tensor in variance_tensors):
            raise RuntimeError("Completed short-term batches have no variance-sum state.")

        for tensor in variance_tensors:
            assert tensor is not None
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    "Short-term variance-sum shape does not match the spectrum-sum shape."
                )

    return group


def _finalize_global_uncertainty(
    group: GroupAccumulator,
) -> Tensor | None:
    """Calculate the component-wise global standard error for one group."""
    if group.count < 2:
        return None

    if group.spectrum_sum is None or group.squared_sum is None:
        raise RuntimeError("Global uncertainty accumulator is inconsistent.")

    mean = group.spectrum_sum / group.count
    mean_squared = group.squared_sum / group.count

    variance = (group.count / (group.count - 1)) * (
        mean_squared - torch.complex(torch.square(mean.real), torch.square(mean.imag))
    )

    var_re = torch.clamp_min(variance.real, 0.0)
    var_im = torch.clamp_min(variance.imag, 0.0)

    return torch.complex(torch.sqrt(var_re / group.count), torch.sqrt(var_im / group.count))


def _finalize_short_term_uncertainty(
    state: ShortTermUncertaintyState,
) -> Tensor | None:
    """Calculate the component-wise short-term uncertainty from completed batches."""
    if state.completed_batches == 0:
        return None

    if state.variance_sum_re is None or state.variance_sum_im is None:
        raise RuntimeError("Short-term uncertainty accumulator is inconsistent.")

    mean_variance_re = state.variance_sum_re / state.completed_batches
    mean_variance_im = state.variance_sum_im / state.completed_batches

    return torch.complex(
        torch.sqrt(torch.clamp_min(mean_variance_re, 0.0)),
        torch.sqrt(torch.clamp_min(mean_variance_im, 0.0)),
    )


def _finalize_group_uncertainty(
    group: GroupAccumulator,
    uncertainty_estimation: Literal["global", "short_term"],
) -> Tensor | None:
    """Finalize one group's configured uncertainty estimate."""
    if uncertainty_estimation == "global":
        return _finalize_global_uncertainty(group)

    if uncertainty_estimation == "short_term":
        return _finalize_short_term_uncertainty(group.short_term)

    raise RuntimeError(f"Unknown uncertainty-estimation method {uncertainty_estimation!r}.")


def _combine_group_uncertainties(uncertainties: list[Tensor]) -> Tensor | None:
    """Combine placement uncertainties using their component-wise maximum."""
    if not uncertainties:
        return None

    if len(uncertainties) == 1:
        return uncertainties[0]

    uncertainty_re = uncertainties[0].real
    uncertainty_im = uncertainties[0].imag

    for uncertainty in uncertainties[1:]:
        uncertainty_re = torch.maximum(uncertainty_re, uncertainty.real)
        uncertainty_im = torch.maximum(uncertainty_im, uncertainty.imag)

    return torch.complex(uncertainty_re, uncertainty_im)


def finalize_result(accumulator: SpectrumAccumulator) -> SpectrumResult:
    """Create a CPU-backed result from accumulated spectral estimates.

    Shifted and unshifted estimates are combined into one count-weighted
    spectrum. Uncertainties are calculated separately for each placement
    group using the configured method. If both groups provide an uncertainty,
    their component-wise maximum is reported.

    Parameters
    ----------
    accumulator : SpectrumAccumulator
        Completed accumulation state for one requested spectrum.

    Returns
    -------
    SpectrumResult
        Final spectrum, frequency metadata, and optional component-wise
        uncertainty estimate.

    Warns
    -----
    RuntimeWarning
        If neither placement group contains enough estimates for the
        configured uncertainty-estimation method.

    Raises
    ------
    RuntimeError
        If no unshifted spectra were accumulated or an accumulator group
        is inconsistent.
    """
    unshifted_group = _check_accumulator_group(
        accumulator.unshifted,
        uncertainty_estimation=accumulator.uncertainty_estimation,
        m_var=accumulator.m_var,
    )

    if unshifted_group is None:
        raise RuntimeError(
            f"Cannot finalize channels {accumulator.channels}: no spectra were accumulated."
        )

    groups: list[GroupAccumulator] = [unshifted_group]

    shifted_group = _check_accumulator_group(
        accumulator.shifted,
        uncertainty_estimation=accumulator.uncertainty_estimation,
        m_var=accumulator.m_var,
    )

    if shifted_group is not None:
        groups.append(shifted_group)

    if unshifted_group.spectrum_sum is None:
        raise RuntimeError("Unshifted spectrum accumulator is inconsistent.")

    total_spectrum = unshifted_group.spectrum_sum.clone()
    total_count = unshifted_group.count

    for group in groups[1:]:
        if group.spectrum_sum is None:
            raise RuntimeError("Spectrum accumulator group is inconsistent.")

        total_spectrum += group.spectrum_sum
        total_count += group.count

    spectrum = (total_spectrum / total_count).cpu().resolve_conj().numpy()

    uncertainties: list[Tensor] = []

    for group in groups:
        uncertainty = _finalize_group_uncertainty(
            group, accumulator.uncertainty_estimation
        )

        if uncertainty is not None:
            uncertainties.append(uncertainty)

    combined_uncertainty = _combine_group_uncertainties(uncertainties)

    if combined_uncertainty is None:
        spectrum_uncertainty = None

        if accumulator.uncertainty_estimation == "global":
            message = (
                "Need at least two spectral estimates in one placement "
                "group for global uncertainty estimation."
            )
        else:
            message = (
                f"Need at least one complete batch of m_var="
                f"{accumulator.m_var} spectral estimates in one placement "
                "group for short-term uncertainty estimation."
            )

        warnings.warn(message, RuntimeWarning, stacklevel=3)
    else:
        spectrum_uncertainty = combined_uncertainty.cpu().resolve_conj().numpy()

    return SpectrumResult(
        channels=accumulator.channels,
        freq=accumulator.freq,
        freq_unit=accumulator.freq_unit,
        spectrum=spectrum,
        spectrum_uncertainty=spectrum_uncertainty,
    )
