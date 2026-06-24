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
import torch

if TYPE_CHECKING:
    from .planning import RuntimeConfig
    from .utils import S3Calcs


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
    error_buffer : torch.Tensor | None
        Temporary device buffer holding a subset batch of `m` window
        chunks to compute intermediate variance profiles.
    error_accumulator : np.ndarray | None
        Running summation of variance metrics (split across real and
        imaginary parts) accumulated on the host CPU.
    error_buffer_index : int
        The current write position pointer inside `error_buffer`.
    chunks_processed : int
        The total number of individual signal windows integrated into
        `spectrum_accumulator`.
    error_batches_processed : int
        The total number of variance/error batches completed and merged.
    """

    order: int
    channels: tuple[int, ...]

    freq: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
    spectrum: np.ndarray | None = None  # s
    spectrum_error: np.ndarray | None = None  # s_err

    spectrum_accumulator: torch.Tensor | None = None  # s_gpu
    error_buffer: torch.Tensor | None = None  # s_errs_buffer
    error_accumulator: np.ndarray | None = None  # s_err_accumulated

    error_buffer_index: int = 0  # err_counter
    chunks_processed: int = 0  # n_chunks_processed
    error_batches_processed: int = 0  # n_error_estimates

    @property
    def is_initialized(self) -> bool:
        """Check if the device accumulators have been allocated."""
        return self.error_buffer is not None

    def reset_state(self):
        """Clears accumulators to prepare for a fresh calculation."""
        self.freq = None
        self.spectrum = None
        self.spectrum_error = None
        self.spectrum_accumulator = None
        self.error_buffer = None
        self.error_accumulator = None
        self.error_buffer_index = 0
        self.chunks_processed = 0
        self.error_batches_processed = 0

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

        self.error_buffer = allocate_error_buffer(
            order=order,
            f_size=f_size,
            m=runtime.m,
            device=runtime.device,
            s3_calc=runtime.s3_calc,
        )


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
        freq_band = runtime.freq_all[runtime.f_min_idx : runtime.f_max_idx]

        for result in self.results.values():
            result.initialize_arrays(freq_band, runtime)


def allocate_error_buffer(
    order: int,
    f_size: int,
    m: int,
    device: torch.device,
    s3_calc: S3Calcs,
):
    if order == 1:
        shape = (1, m)
    elif order == 2:
        shape = (f_size, m)
    elif order == 3:
        if s3_calc == "1/2":
            shape = (f_size // 2, 2 * (f_size // 2) - 1, m)
        else:
            shape = (f_size // 2, f_size // 2, m)
    elif order == 4:
        shape = (f_size, f_size, m)
    else:
        raise ValueError(f"{order} not a valid order.")

    return torch.ones(shape, device=device, dtype=torch.complex64)
