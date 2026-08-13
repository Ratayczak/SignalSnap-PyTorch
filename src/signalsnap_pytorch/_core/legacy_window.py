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

from .plans import RuntimeConfig

_CPU_DEVICE = torch.device("cpu")
_TIMESTAMP_WINDOW_SIGMA = 0.14
_LEGACY_TIMESTAMP_REFERENCE_POINTS = 70

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


def old_cg_window(
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


def prepare_legacy_timestamp_window(runtime: RuntimeConfig) -> LegacyTimestampWindow:
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
