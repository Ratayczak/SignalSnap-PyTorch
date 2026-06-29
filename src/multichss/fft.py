# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .planning import RuntimeConfig


### old code from previous api
def _old_gaussian_window(x: Tensor, n_windows: int, l: int, sigma_t: float) -> Tensor:
    """
    Approx. confined Gaussian window (see DOI:10.1016/j.sigpro.2014.03.033)
    """

    center = n_windows * 0.5
    denom = 2.0 * l * sigma_t

    t = (x - center) / denom
    return torch.exp(-t * t)


def _old_calc_window(x: Tensor, n_windows: int, l: int, sigma_t: float) -> Tensor:
    """
    Helper function to calculate the approx. confined gaussian window
    as defined in https://doi.org/10.1016/j.sigpro.2014.03.033
    """

    h: Tensor = x.new_tensor(-0.5)

    term_x = _old_gaussian_window(x, n_windows, l, sigma_t)
    term_h = _old_gaussian_window(h, n_windows, l, sigma_t)
    term_x_p_l = _old_gaussian_window(x + l, n_windows, l, sigma_t)
    term_x_m_l = _old_gaussian_window(x - l, n_windows, l, sigma_t)
    term_h_p_l = _old_gaussian_window(h + l, n_windows, l, sigma_t)
    term_h_m_l = _old_gaussian_window(h - l, n_windows, l, sigma_t)

    denom = term_h_p_l + term_h_m_l
    win = term_x - (term_h * (term_x_p_l + term_x_m_l)) / denom

    return win


def _old_cg_window(
    n_windows: int,
    fs: float,
    torch_device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Helper function to calculate the approx. confined gaussian window
    as defined in https://doi.org/10.1016/j.sigpro.2014.03.033
    """

    x = torch.linspace(0, n_windows, n_windows, device=torch_device, dtype=dtype)
    l = n_windows + 1
    sigma_t = 0.14

    window = _old_calc_window(x, n_windows, l, sigma_t)
    norm_t = (window * window).sum() / fs

    window_full = window / torch.sqrt(norm_t)

    return window_full


### end of old code


def _gaussian(x: Tensor, N: int, sigma_t_prefactor: float) -> Tensor:
    """
    Helper function to calculate the Gaussian
        G(x) = exp{- dt^2 [x - (N-1) / 2]^2 / [2 * sigma_t]^2}.

    sigma_t is the temporal width of the Gaussian. Here, it is given in terms
    of the window duration T
        sigma_t = sigma_t_prefactor * T = sigma_t_prefactor * N * dt
    so that we effectively calculate
        G(x) = exp{- [x - (N-1) / 2]^2 / [2 * N * sigma_t_prefactor]^2}

    This Gaussian is used to construct the discrete approximate confined
    Gaussian window function for N-point Fourier transforms.
    (reference: DOI:10.1016/j.sigpro.2014.03.033)
    """
    center = (N - 1) * 0.5
    denom = 2.0 * N * sigma_t_prefactor

    t = (x - center) / denom
    return torch.exp(-t * t)


def acG_window_func(
    N: int,
    sigma_t: float = 0.14,
    torch_device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Helper function to calculate the approximate confined Gaussian window
    function
        g_k^(acG) \\propto G(k) - G(-1/2) * [G(k + N) + G(k - N)]/
                                            [G(-1/2 + N) + G(-1/2 - N)]
    for N-point Fourier transforms.
    (reference: DOI:10.1016/j.sigpro.2014.03.033)

    sigma_t is given in terms of the time duration T = N * dt per window.

    To minimize floating-point precision errors, the window is
    normalized such that the maximum is equal to 1.
    """
    h = torch.tensor(-0.5, device=torch_device, dtype=dtype)
    k = torch.arange(N, device=torch_device, dtype=dtype)

    term_k = _gaussian(k, N, sigma_t)
    term_h = _gaussian(h, N, sigma_t)
    term_k_p_N = _gaussian(k + N, N, sigma_t)
    term_k_m_N = _gaussian(k - N, N, sigma_t)
    term_h_p_N = term_h
    term_h_m_N = _gaussian(h - N, N, sigma_t)

    acG_k = term_k - (term_h * (term_k_p_N + term_k_m_N)) / (term_h_p_N + term_h_m_N)

    return acG_k / torch.max(acG_k)


def to_device(array: np.ndarray, runtime: RuntimeConfig) -> Tensor:
    """Copy np.array to torch.device using the correct data type"""
    if np.iscomplexobj(array):
        raise TypeError("Input data cannot be complex.")

    return torch.as_tensor(array, dtype=runtime.real_dtype, device=runtime.device)


def compute_fft(chunk: Tensor, window: Tensor, runtime: RuntimeConfig) -> Tensor:
    """Compute the FFT as specified in DOI:10.1016/j.dsp.2026.105893"""
    weighted_chunk = window * chunk

    if runtime.use_full_fft:
        coeffs = torch.fft.fft(weighted_chunk, dim=1)
        coeffs = torch.fft.fftshift(coeffs, dim=1)
    else:
        coeffs = torch.fft.rfft(weighted_chunk, dim=1)

    return coeffs * runtime.dt


def prepare_windows(runtime: RuntimeConfig) -> tuple[Tensor, Tensor]:
    """Return window for m chunks in the correct shape"""
    if runtime.old_window:
        single_window = _old_cg_window(
            runtime.window_points,
            fs=1,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )
    else:
        single_window = acG_window_func(
            runtime.window_points,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )

    repeated_window = single_window.reshape(1, runtime.window_points, 1).repeat(
        runtime.m, 1, 1
    )
    return single_window, repeated_window


def iter_window_slices(runtime: RuntimeConfig) -> Iterator[tuple[int, int]]:
    """Return the window slice indices"""
    chunk_size = runtime.window_points * runtime.m

    for window_index in range(runtime.n_windows):
        base = window_index * chunk_size
        for shift in (0, runtime.window_points // 2):
            start = base + shift
            end = start + chunk_size
            if end <= runtime.n_data_points:
                yield start, end


def reshape_window_chunk(
    chunk: np.ndarray,
    runtime: RuntimeConfig,
) -> np.ndarray:
    """Reshape each chunk to m windows"""
    expected_size = runtime.window_points * runtime.m

    if chunk.shape[0] != expected_size:
        raise ValueError(
            f"Expected chunk with {expected_size} samples, got {chunk.shape[0]}."
        )

    return chunk.reshape(runtime.m, runtime.window_points, 1)
