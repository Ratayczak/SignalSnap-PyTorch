# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .plans import RuntimeConfig


def reshape_window_chunk(
    chunk: np.ndarray,
    estimate_count: int,
    windows_per_estimate: int,
    window_points: int,
) -> np.ndarray:
    """Reshape one flat signal slice into the window batch used by the FFT.

    Parameters
    ----------
    chunk : np.ndarray
        One-dimensional signal slice with shape ``(B * m * N,)``.
    estimate_count : int
        Number of spectral estimates represented by ``chunk``.
    windows_per_estimate : int
        Number of coefficient windows in each spectral estimate.
    window_points : int
        Number of sampled points in each coefficient window.

    Returns
    -------
    np.ndarray
        Reshaped chunk with shape ``(B, m, N)``.

    Raises
    ------
    ValueError
        If ``chunk`` does not contain exactly ``B * m * N`` samples.
    """
    expected_size = estimate_count * window_points * windows_per_estimate

    if chunk.shape[0] != expected_size:
        raise ValueError(f"Expected chunk with {expected_size} samples, got {chunk.shape[0]}.")

    return chunk.reshape(estimate_count, windows_per_estimate, window_points)


def to_device(array: np.ndarray, runtime: RuntimeConfig) -> Tensor:
    """Convert a NumPy array to a torch tensor using the runtime dtype and device.

    The input shape is preserved. In the main calculation pipeline this is typically called with a
    reshaped signal chunk of shape ``(B, m, N)``.

    Parameters
    ----------
    array : np.ndarray
        NumPy array to transfer.
    runtime : RuntimeConfig
        Resolved real dtype and torch device.

    Returns
    -------
    Tensor
        Tensor on ``runtime.device`` with dtype ``runtime.real_dtype``.
    """

    return torch.as_tensor(array, dtype=runtime.real_dtype, device=runtime.device)


def compute_fft(chunk: Tensor, window: Tensor, dt: float) -> Tensor:
    """Window a signal chunk and compute its Fourier coefficients.

    ``coeffs`` is computed via the inverse FFT with forward normalization because the SignalSnap
    convention uses the opposite Fourier-transform sign from PyTorch.

    Parameters
    ----------
    chunk : Tensor
        Real-valued signal chunk with shape ``(B, m, N)``.
    window : Tensor
        Window tensor with shape ``(N,)``.
    dt : float
        Sampling interval of the channel.

    Returns
    -------
    Tensor
        Shifted complex Fourier coefficients scaled by ``dt``, with shape ``(1, B, m, N)``. The
        leading dimension is the number of different realizations.
    """

    coeffs = torch.fft.ifft(window * chunk, dim=-1, norm="forward")
    coeffs = torch.fft.fftshift(coeffs, dim=-1)

    return (coeffs * dt).unsqueeze(0)
