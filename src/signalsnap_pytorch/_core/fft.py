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
_TIMESTAMP_WINDOW_SIGMA = 0.14
_TIMESTAMP_QUADRATURE_POINTS = 128
_LEGACY_TIMESTAMP_REFERENCE_POINTS = 70


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
        Real window function with shape ``(N,)``.
    norm_all_orders : tuple[Tensor, Tensor, Tensor, Tensor]
        Scalar normalization for orders one through four. Entry ``n - 1`` is
        ``dt * sum(window**n)``.
    """

    window: Tensor
    norm_all_orders: tuple[Tensor, Tensor, Tensor, Tensor]

    def norm(self, order: int) -> Tensor:
        """Return the scalar normalization for an order from one through four.

        Callers must supply a supported positive order; this internal accessor does not perform
        explicit range validation.
        """
        return self.norm_all_orders[order - 1]


@dataclass(frozen=True, slots=True)
class DefaultTimestampWindow:
    """Continuous timestamp window and order-one through order-four normalizers."""

    duration: float
    norm_all_orders: tuple[Tensor, Tensor, Tensor, Tensor]

    def evaluate(self, relative_times: Tensor) -> Tensor:
        """Evaluate the window at times relative to the current window start."""

        normalized_times = relative_times / self.duration
        return _default_timestamp_window(normalized_times)

    def norm(self, order: int) -> Tensor:
        """Return the continuous normalization for an order from one through four."""

        return self.norm_all_orders[order - 1]


@dataclass(frozen=True, slots=True)
class LegacyTimestampWindow:
    """Fixed 70-point v1 timestamp window and normalization convention."""

    duration: float
    reference_dt: float
    scale: Tensor
    norm_all_orders: tuple[Tensor, Tensor, Tensor, Tensor]

    def evaluate(self, relative_times: Tensor) -> Tensor:
        """Evaluate the normalized v1 window at relative event times."""

        reference_positions = relative_times / self.reference_dt
        raw = _old_calc_window(
            reference_positions,
            _LEGACY_TIMESTAMP_REFERENCE_POINTS,
            _LEGACY_TIMESTAMP_REFERENCE_POINTS + 1,
            _TIMESTAMP_WINDOW_SIGMA,
        )
        return raw * self.scale

    def norm(self, order: int) -> Tensor:
        """Return the v1 discrete reference normalization for an order."""

        return self.norm_all_orders[order - 1]


def _default_timestamp_window(normalized_times: Tensor) -> Tensor:
    """Evaluate the continuous confined-Gaussian family using a Torch tensor."""

    def gaussian(values: Tensor) -> Tensor:
        scaled = (values - 0.5) / (2.0 * _TIMESTAMP_WINDOW_SIGMA)
        return torch.exp(-(scaled**2))

    zero = normalized_times.new_tensor(0.0)
    one = normalized_times.new_tensor(1.0)
    edge = gaussian(zero)
    denominator = gaussian(one) + gaussian(-one)
    raw = (
        gaussian(normalized_times)
        - edge * (gaussian(normalized_times + one) + gaussian(normalized_times - one)) / denominator
    )

    midpoint = normalized_times.new_tensor(0.5)
    midpoint_raw = (
        gaussian(midpoint)
        - edge * (gaussian(midpoint + one) + gaussian(midpoint - one)) / denominator
    )

    return raw / midpoint_raw


def _default_timestamp_normalizations(duration: float) -> tuple[float, ...]:
    """Calculate continuous normalizers using CPU float64 quadrature."""

    nodes_array, weights_array = np.polynomial.legendre.leggauss(_TIMESTAMP_QUADRATURE_POINTS)
    normalized_times = torch.from_numpy((nodes_array + 1.0) / 2.0)
    weights = torch.from_numpy(weights_array)
    window = _default_timestamp_window(normalized_times)

    return tuple(duration * 0.5 * torch.dot(weights, window**order).item() for order in range(1, 5))


def _prepare_default_timestamp_window(runtime: RuntimeConfig) -> DefaultTimestampWindow:
    """Prepare the default continuous window for a timestamp calculation."""

    duration = runtime.window_plan.duration
    normalizations = _default_timestamp_normalizations(duration)
    norms = torch.as_tensor(normalizations, dtype=runtime.real_dtype, device=runtime.device)

    return DefaultTimestampWindow(
        duration=duration,
        norm_all_orders=(norms[0], norms[1], norms[2], norms[3]),
    )


def _legacy_timestamp_raw_numpy(reference_positions: np.ndarray) -> np.ndarray:
    """Evaluate the unnormalized v1 timestamp window."""

    reference_points = _LEGACY_TIMESTAMP_REFERENCE_POINTS
    length = reference_points + 1

    def gaussian(values: np.ndarray | float):
        scaled = (values - reference_points / 2.0) / (2.0 * length * _TIMESTAMP_WINDOW_SIGMA)
        return np.exp(-(scaled**2))

    edge = gaussian(-0.5)
    denominator = gaussian(-0.5 + length) + gaussian(-0.5 - length)

    return (
        gaussian(reference_positions)
        - edge
        * (gaussian(reference_positions + length) + gaussian(reference_positions - length))
        / denominator
    )


def _prepare_legacy_timestamp_window(runtime: RuntimeConfig) -> LegacyTimestampWindow:
    """Prepare the exact fixed-grid v1 timestamp window convention."""

    duration = runtime.window_plan.duration
    reference_points = _LEGACY_TIMESTAMP_REFERENCE_POINTS
    reference_dt = duration / reference_points
    reference_grid = np.linspace(0.0, float(reference_points), reference_points, dtype=np.float64)

    raw = _legacy_timestamp_raw_numpy(reference_grid)
    norm2 = reference_dt * float(np.sum(raw**2))
    scale = 1.0 / np.sqrt(norm2)
    normalized_window = raw * scale

    normalizations = tuple(
        reference_dt * float(np.sum(normalized_window**order)) for order in range(1, 5)
    )
    values = torch.as_tensor(
        (scale, *normalizations),
        dtype=runtime.real_dtype,
        device=runtime.device,
    )

    return LegacyTimestampWindow(
        duration=duration,
        reference_dt=reference_dt,
        scale=values[0],
        norm_all_orders=(values[1], values[2], values[3], values[4]),
    )


TimestampWindow = DefaultTimestampWindow | LegacyTimestampWindow


def prepare_timestamp_window(runtime: RuntimeConfig) -> TimestampWindow:
    """Prepare timestamp event weights and order-dependent normalizations.

    The returned object evaluates the selected window at event times relative to a physical-window
    start and supplies normalization factors for spectrum orders one through four.
    ``runtime.old_window`` selects the fixed-grid legacy convention; otherwise, the default
    continuous timestamp-window convention is used.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved window duration, numeric dtype, calculation device, and compatibility setting.

    Returns
    -------
    DefaultTimestampWindow | LegacyTimestampWindow
        Prepared timestamp window whose normalization tensors use ``runtime.real_dtype`` on
        ``runtime.device``.
    """

    if runtime.old_window:
        return _prepare_legacy_timestamp_window(runtime)

    return _prepare_default_timestamp_window(runtime)


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


def prepare_window(runtime: RuntimeConfig, dt: float, window_points: int) -> WindowBuffer:
    """Build the window tensors used for each spectral estimate.

    Parameters
    ----------
    runtime : :class:`RuntimeConfig`
        Runtime parameters resolved from the user configs.
    dt : float
        Sampling interval used for discrete window normalization.
    window_points : int
        Number of sampled points in one coefficient window.

    Returns
    -------
    WindowBuffer
        ``window`` has shape ``(N,)``, where ``N=window_points``. ``norm_all_orders`` contains the
        scalar normalization for orders one through four.
    """

    if runtime.old_window:
        window = _old_cg_window(
            window_points,
            fs=1,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )
    else:
        window = _acg_window(
            window_points,
            torch_device=runtime.device,
            dtype=runtime.real_dtype,
        )

    return WindowBuffer(
        window=window,
        norm_all_orders=(
            dt * window.sum(),
            dt * (window**2).sum(),
            dt * (window**3).sum(),
            dt * (window**4).sum(),
        ),
    )


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
