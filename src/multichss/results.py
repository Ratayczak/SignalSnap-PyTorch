# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .planning import RuntimeConfig
    from torch import Tensor


@dataclass(slots=True)
class SpectrumResult:
    """Data container for the state and results of a single spectral
    calculation.

    Stores the configuration metadata, accumulated hardware states, error
    buffers, and final computed results for a specific higher-order
    auto- or cross-spectrum calculation.

    Parameters
    ----------
    order : int
        The order of the polyspectrum (e.g., 2 for power
        spectrum, 3 for bispectrum, 4 for trispectrum).
    channels : tuple[int, ...]
        The indices identifying which channels are part of this
        calculation. For example, `(0,)` indicates an auto-spectrum on
        channel 0, while `(0, 1)` indicates a cross-spectrum between
        channels 0 and 1.

    Attributes
    ----------
    freq : np.ndarray | tuple[np.ndarray, np.ndarray] | None
        The frequency axes associated with the spectrum. Single 1D array
        for a power spectrum, or a tuple of two 1D arrays for higher-order
        polyspectra.
    spectrum : np.ndarray | None
        The final normalized spectral values transferred back to the CPU.
    spectrum_error : np.ndarray | None
        The final calculated standard error of the mean (SEM) or variance
        values transferred back to the CPU. # see TODO
    spectrum_accumulator : torch.Tensor | None
        Running total accumulation buffer of the calculated spectra on the
        active torch device
    error_accumulator_x_squared : torch.Tensor | None
        Running total accumulation buffer of the real and imaginary parts
        of the spectra squared on the active torch device.
    chunks_processed : int
        The total number of individual signal windows integrated into
        `spectrum_accumulator`.
    """

    order: int
    channels: tuple[int, ...]

    freq: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
    spectrum: np.ndarray | None = None  # s
    spectrum_error: np.ndarray | None = None  # s_err

    spectrum_accumulator: Tensor | None = None  # s_gpu
    error_accumulator_x_squared: Tensor | None = None

    chunks_processed: int = 0  # n_chunks_processed

    def reset_state(self):
        """Clears accumulators to prepare for a fresh calculation."""
        self.freq = None
        self.spectrum = None
        self.spectrum_error = None
        self.spectrum_accumulator = None
        self.error_accumulator_x_squared = None
        self.chunks_processed = 0

    def initialize_arrays(self, freq_band: np.ndarray, runtime: RuntimeConfig) -> None:
        order = self.order
        f_size = freq_band.shape[0]

        if order == 3:
            half_size = f_size // 2
            self.freq = freq_band[:half_size]

            if runtime.s3_calc == "1/2":
                self.freq = np.concatenate((-self.freq[:0:-1], self.freq))
        else:
            self.freq = freq_band


@dataclass(slots=True)
class SpectrumResultStore:
    """Container for all spectrum results produced by a calculation
    pipeline.

    Stores one :class:`SpectrumResult` per channel tuple and spectrum
    order.
    Results are indexed by ``(channels, order)``, where ``channels`` is a
    tuple of data-channel indices and ``order`` is the polyspectrum order.

    This class owns collection-level bookkeeping only. Numerical
    accumulation, error estimation, and finalization are handled elsewhere.

    Attributes
    ----------
    results : dict[tuple[tuple[int, ...], int], SpectrumResult]
        Mapping from ``(channels, order)`` to the corresponding spectrum
        result. For example, ``((0,), 2)`` identifies the second-order
        auto-spectrum of channel 0, while ``((0, 1), 2)`` identifies a
        second-order cross-spectrum between channels 0 and 1.
    """

    results: dict[tuple[tuple[int, ...], int], SpectrumResult] = field(
        default_factory=dict
    )

    def get(self, channels: tuple[int, ...], order: int) -> SpectrumResult:
        """Return the result for a channel tuple and spectrum order."""
        return self.results[(channels, order)]

    def add(self, result: SpectrumResult) -> None:
        """Add or replace a spectrum result using its channels and order."""
        self.results[(result.channels, result.order)] = result

    def reset_all_states(self) -> None:
        """Reset the mutable calculation state of all stored results."""
        for result in self.results.values():
            result.reset_state()

    def initialize_arrays(self, runtime: RuntimeConfig) -> None:
        freq_band = runtime.freq_band

        for result in self.results.values():
            result.initialize_arrays(freq_band, runtime)
