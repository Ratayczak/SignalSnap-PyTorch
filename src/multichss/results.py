# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
import numpy as np
import torch


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
    s : np.ndarray | None
        The final normalized spectral values transferred back to the CPU.
    s_err : np.ndarray | None
        The final calculated standard error of the mean (SEM) or variance 
        values transferred back to the CPU. # TODO
    s_gpu : torch.Tensor | None
        Running total accumulation buffer of the calculated spectra on the
        gpu
    s_errs_buffer : torch.Tensor | None
        Temporary device buffer holding a subset batch of `m_var` window chunks 
        to compute intermediate variance profiles.
    err_counter : int
        The current write position pointer inside `s_errs_buffer`.
    n_chunks_processed : int
        The total number of individual signal windows integrated into `s_gpu`.
    n_error_estimates : int
        The total number of variance batches completed and merged.
    s_err_accumulated : np.ndarray | None
        Running summation of variance metrics (split across real and imaginary parts) 
        accumulated on the host CPU.
    """

    order: int
    keys: tuple[int, ...]
    freq: np.ndarray | tuple[np.ndarray, np.ndarray]| None = None
    s: np.ndarray | None = None
    s_err: np.ndarray | None = None
    s_gpu: torch.Tensor | None = None
    s_errs_buffer: torch.Tensor | None = None
    err_counter: int = 0
    n_chunks_processed: int = 0
    n_error_estimates: int = 0
    s_err_accumulated: np.ndarray | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if the GPU accumulators have been allocated."""
        return self.s_gpu is not None

    def reset_state(self):
        """Clears accumulators to prepare for a fresh calculation."""
        self.freq = None
        self.s = None
        self.s_err = None
        self.s_gpu = None
        self.s_errs_buffer = None
        self.err_counter = 0
        self.n_chunks_processed = 0
        self.n_error_estimates = 0
        self.s_err_accumulated = None
