# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor

from .results import SpectrumResult


def accumulate_spectrum(result: SpectrumResult, single_spectrum: Tensor) -> None:
    """Accumulate one single-window spectrum into a result object.

    Updates the running spectrum sum and running real and imaginary squared sums
    """
    
    if result.freq is None:
        raise ValueError("SpectrumResult must be initialized before accumulation.")

    if result.spectrum_accumulator is None:
        result.spectrum_accumulator = single_spectrum.clone()
    else:
        result.spectrum_accumulator += single_spectrum

    if result.error_accumulator_x_squared is None:
        result.error_accumulator_x_squared = torch.complex(
            torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
        )
    else:
        result.error_accumulator_x_squared += torch.complex(
            torch.square(single_spectrum.real), torch.square(single_spectrum.imag)
        )

    result.chunks_processed += 1


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

    assert result.error_accumulator_x_squared is not None

    if result.chunks_processed == 1:
        result.spectrum_error = None
        print("Need at least two spectra estimates for an error estimation.")
    else:
        var_factor = result.chunks_processed / (result.chunks_processed - 1)
        result.error_accumulator_x_squared /= result.chunks_processed
        spectrum_variance = var_factor * (
            result.error_accumulator_x_squared
            - torch.complex(
                torch.square(result.spectrum_accumulator.real),
                torch.square(result.spectrum_accumulator.imag),
            )
        )
        var_re = torch.clamp_min(spectrum_variance.real, 0.0)
        var_im = torch.clamp_min(spectrum_variance.imag, 0.0)
        result.spectrum_error = (
            torch.complex(
                torch.sqrt(var_re / result.chunks_processed),
                torch.sqrt(var_im / result.chunks_processed),
            )
            .cpu()
            .resolve_conj()
            .numpy()
        )
