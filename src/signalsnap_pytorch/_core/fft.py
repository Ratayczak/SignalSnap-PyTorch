# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .planning import RuntimeConfig

_CPU_DEVICE = torch.device("cpu")


### ------------------------
# old code from previous api
### ------------------------
def _old_gaussian_window(x: Tensor, n_windows: int, l: int, sigma_t: float) -> Tensor:
    """Approx. confined Gaussian window (see DOI:10.1016/j.sigpro.2014.03.033)."""

    center = n_windows * 0.5
    denom = 2.0 * l * sigma_t

    t = (x - center) / denom
    return torch.exp(-t * t)


def _old_calc_window(x: Tensor, n_windows: int, l: int, sigma_t: float) -> Tensor:
    """
    Helper function to calculate the approx. confined gaussian window as defined in
    https://doi.org/10.1016/j.sigpro.2014.03.033
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
    torch_device: torch.device = _CPU_DEVICE,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Helper function to calculate the approx. confined gaussian window as defined in
    https://doi.org/10.1016/j.sigpro.2014.03.033
    """

    x = torch.linspace(0, n_windows, n_windows, device=torch_device, dtype=dtype)
    l = n_windows + 1
    sigma_t = 0.14

    window = _old_calc_window(x, n_windows, l, sigma_t)
    norm_t = (window * window).sum() / fs

    window_full = window / torch.sqrt(norm_t)

    return window_full


### -------------
# end of old code
### -------------


@dataclass(frozen=True, slots=True)
class WindowBuffer:
    """Store the window function and its reusable normalization factors.

    Attributes
    ----------
    window : Tensor
        Real window function with shape ``(N,)``, where ``N = runtime.window_points``.
    norm_all_orders : tuple[Tensor, Tensor, Tensor, Tensor]
        Scalar normalization for orders one through four. Entry ``n - 1`` is
        ``runtime.dt * sum(window**n)``.
    """

    window: Tensor
    norm_all_orders: tuple[Tensor, Tensor, Tensor, Tensor]

    def norm(self, order: int) -> Tensor:
        """Return the scalar normalization for an order from one through four.

        Callers must supply a supported positive order; this internal accessor does not perform
        explicit range validation.
        """
        return self.norm_all_orders[order - 1]


def _gaussian(x: Tensor, N: int, sigma_t_prefactor: float) -> Tensor:
    """
    Helper function to calculate the Gaussian
        G(x) = exp{- dt^2 [x - (N-1) / 2]^2 / [2 * sigma_t]^2}.

    sigma_t is the temporal width of the Gaussian. Here, it is given in terms of the window
    duration T
        sigma_t = sigma_t_prefactor * T = sigma_t_prefactor * N * dt
    so that we effectively calculate
        G(x) = exp{- [x - (N-1) / 2]^2 / [2 * N * sigma_t_prefactor]^2}

    This Gaussian is used to construct the discrete approximate confined Gaussian window function
    for N-point Fourier transforms.
    (reference: DOI:10.1016/j.sigpro.2014.03.033)
    """

    center = (N - 1) * 0.5
    denom = 2.0 * N * sigma_t_prefactor

    t = (x - center) / denom
    return torch.exp(-t * t)


def _acg_window(
    N: int,
    sigma_t: float = 0.14,
    torch_device: torch.device = _CPU_DEVICE,
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

    To minimize floating-point precision errors, the window is normalized such that the maximum is
    equal to 1.
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


def compute_fft(chunk: Tensor, window: Tensor, runtime: RuntimeConfig) -> Tensor:
    """Window a signal chunk and compute its Fourier coefficients.

    ``coeffs`` is computed via the inverse FFT with forward normalization because the SignalSnap
    convention uses the opposite Fourier-transform sign from PyTorch.

    Parameters
    ----------
    chunk : Tensor
        Real-valued signal chunk with shape ``(B, m, N)``.
    window : Tensor
        Window tensor with shape ``(N,)``.
    runtime : RuntimeConfig
        Runtime settings defining FFT mode, sample spacing, and dtypes.

    Returns
    -------
    Tensor
        Shifted complex Fourier coefficients scaled by ``runtime.dt``, with shape ``(B, m, N)``.
    """

    coeffs = torch.fft.ifft(window * chunk, dim=-1, norm="forward")
    coeffs = torch.fft.fftshift(coeffs, dim=-1)

    return coeffs * runtime.dt


def prepare_window(runtime: RuntimeConfig) -> WindowBuffer:
    """Build the window tensors used for each spectral estimate.

    Parameters
    ----------
    runtime : :class:`RuntimeConfig`
        Runtime parameters resolved from the user configs.

    Returns
    -------
    WindowBuffer
        ``window`` has shape ``(N,)``, where ``N = runtime.window_points``. ``norm_all_orders``
        contains the scalar normalization for orders one through four.
    """

    if runtime.old_window:
        window = _old_cg_window(
            runtime.window_points,
            fs=1,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )
    else:
        window = _acg_window(
            runtime.window_points,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )

    return WindowBuffer(
        window=window,
        norm_all_orders=(
            runtime.dt * window.sum(),
            runtime.dt * (window**2).sum(),
            runtime.dt * (window**3).sum(),
            runtime.dt * (window**4).sum(),
        ),
    )


def reshape_window_chunk(
    chunk: np.ndarray,
    runtime: RuntimeConfig,
    estimate_count: int,
) -> np.ndarray:
    """Reshape one flat signal slice into the window batch used by the FFT.

    Parameters
    ----------
    chunk : np.ndarray
        One-dimensional signal slice with shape ``(B * m * N,)``, where ``B=estimate_count``,
        ``m = runtime.m`` and ``N = runtime.window_points``.
    runtime : RuntimeConfig
        Resolved window count and window length.
    estimate_count : int
        Number of spectral estimates represented by ``chunk``.

    Returns
    -------
    np.ndarray
        Reshaped chunk with shape ``(B, m, N)``.

    Raises
    ------
    ValueError
        If ``chunk`` does not contain exactly ``B * m * N`` samples.
    """

    expected_size = estimate_count * runtime.window_points * runtime.m

    if chunk.shape[0] != expected_size:
        raise ValueError(f"Expected chunk with {expected_size} samples, got {chunk.shape[0]}.")

    return chunk.reshape(estimate_count, runtime.m, runtime.window_points)


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
