# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .planning import RuntimeConfig
    from .results import SpectrumResult


def accumulate_spectrum(
    result: SpectrumResult,
    single_spectrum: torch.Tensor,
    runtime: RuntimeConfig,
) -> None:
    """Accumulate one single-window spectrum into a result object.

    Updates the running spectrum sum and, every ``runtime.m`` spectra,
    reduces the error buffer into an accumulated variance-of-mean estimate.
    """
    if result.error_buffer is None:
        raise ValueError("Result must be initialized before accumulation.")

    if result.spectrum_accumulator is None:
        result.spectrum_accumulator = single_spectrum.clone()
    else:
        result.spectrum_accumulator += single_spectrum

    if result.order == 1:
        result.error_buffer[0, result.error_buffer_index] = single_spectrum
    elif result.order == 2:
        result.error_buffer[:, result.error_buffer_index] = single_spectrum
    else:
        result.error_buffer[:, :, result.error_buffer_index] = single_spectrum

    result.error_buffer_index += 1
    result.chunks_processed += 1

    if result.error_buffer_index != runtime.m:
        return

    dim = 1 if result.order in (1, 2) else 2
    factor = runtime.m / (runtime.m - 1)

    x = result.error_buffer
    xr = x.real
    xi = x.imag

    mean_xr2 = torch.mean(xr**2, dim=dim)
    mean_xr = torch.mean(xr, dim=dim)
    var_mean_re = factor * (mean_xr2 - mean_xr**2) / runtime.m

    mean_xi2 = torch.mean(xi**2, dim=dim)
    mean_xi = torch.mean(xi, dim=dim)
    var_mean_im = factor * (mean_xi2 - mean_xi**2) / runtime.m

    batch_error = var_mean_re.cpu().numpy() + 1j * var_mean_im.cpu().numpy()

    if result.error_accumulator is None:
        result.error_accumulator = batch_error
    else:
        result.error_accumulator += batch_error

    result.error_batches_processed += 1
    result.error_buffer_index = 0


def finalize_result(result: SpectrumResult) -> None:
    """Finalize accumulated spectra and error estimates on a result object."""
    if result.spectrum_accumulator is None:
        result.spectrum = None
        result.spectrum_error = None
        return

    if result.chunks_processed == 0:
        raise ValueError("Cannot finalize result without processed chunks.")

    result.spectrum_accumulator /= result.chunks_processed
    result.spectrum = result.spectrum_accumulator.cpu().resolve_conj().numpy()

    n_estimates = result.error_batches_processed
    if n_estimates and result.error_accumulator is not None:
        var_mean = result.error_accumulator / n_estimates
        var_re = np.maximum(np.real(var_mean), 0.0)
        var_im = np.maximum(np.imag(var_mean), 0.0)

        sem_re = np.sqrt(var_re) / 2
        sem_im = np.sqrt(var_im) / 2
        result.spectrum_error = sem_re + 1j * sem_im
    else:
        result.spectrum_error = None
