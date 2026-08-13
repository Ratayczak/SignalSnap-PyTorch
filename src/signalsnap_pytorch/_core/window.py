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

from .legacy_window import LegacyTimestampWindow, old_cg_window, prepare_legacy_timestamp_window
from .plans import RuntimeConfig

_CPU_DEVICE = torch.device("cpu")
_TIMESTAMP_WINDOW_SIGMA = 0.14
_TIMESTAMP_QUADRATURE_POINTS = 128



@dataclass(frozen=True, slots=True)
class SampledWindow:
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
        return prepare_legacy_timestamp_window(runtime)

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


def prepare_window(runtime: RuntimeConfig, dt: float, window_points: int) -> SampledWindow:
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
    SampledWindow
        ``window`` has shape ``(N,)``, where ``N=window_points``. ``norm_all_orders`` contains the
        scalar normalization for orders one through four.
    """

    if runtime.old_window:
        window = old_cg_window(
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

    return SampledWindow(
        window=window,
        norm_all_orders=(
            dt * window.sum(),
            dt * (window**2).sum(),
            dt * (window**3).sum(),
            dt * (window**4).sum(),
        ),
    )

