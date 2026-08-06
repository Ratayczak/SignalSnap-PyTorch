# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor


def build_s3_target_indices(axis_indices: Tensor, fft_freq_count: int) -> tuple[Tensor, Tensor]:
    """Map output bins ``(w1, w2)`` to the shifted-FFT bin at ``w3 = -(w1 + w2)``.

    Parameters
    ----------
    axis_indices : Tensor
        One-dimensional integer indices for the selected frequency band.
    fft_freq_count : int
        Number of bins in the full shifted FFT.

    Returns
    -------
    tuple[Tensor, Tensor]
        Safe target indices and a Boolean validity mask, both with shape ``(F, F)``. Invalid target
        indices are replaced by zero so they can be gathered before the result is masked.
    """

    zero_idx = fft_freq_count // 2
    axis_offsets = axis_indices - zero_idx

    target_offsets = -(axis_offsets[:, None] + axis_offsets[None, :])
    target_indices = target_offsets + zero_idx

    valid_mask = (target_indices >= 0) & (target_indices < fft_freq_count)
    safe_indices = torch.where(valid_mask, target_indices, torch.zeros_like(target_indices))
    return safe_indices, valid_mask


def gather_s3_third_factor(coeffs: Tensor, target_indices: Tensor) -> Tensor:
    """Gather coefficients at ``w3 = -(w1 + w2)`` for a third-order cumulant.

    Parameters
    ----------
    coeffs : Tensor
        Full shifted Fourier coefficients with shape ``(..., m, N)``.
    target_indices : Tensor
        Target-bin grid with shape ``(F, F)``.

    Returns
    -------
    Tensor
        Gathered coefficients with shape ``(..., m, F, F)``.
    """
    return coeffs[..., target_indices]


def _mean_outer(m: int, a: Tensor, b: Tensor) -> Tensor:
    """Compute an average outer product over the window axis.

    Parameters
    ----------
    m : int
        Number of windows represented by the penultimate tensor dimension.
    a, b : Tensor
        Input tensors with shape ``(..., m, F)``.

    Returns
    -------
    Tensor
        Tensor with shape ``(..., F, F)`` whose entry ``[..., f, g]`` is
        ``sum_i(a[..., i, f] * b[..., i, g]) / m``.
    """
    return torch.einsum("...mf,...mg->...fg", a, b) / m


def c2_factorized(m: int, centered_x: Tensor, centered_y: Tensor) -> Tensor:
    """Compute the unbiased second-order multivariate cumulant estimate.

        C2(x, y) = m/(m-1) * ((x-x.mean)*(y-y.mean)).mean

    Parameters
    ----------
    m : int
        Number of windows in the estimate. Must exceed one.
    centered_x, centered_y : Tensor
        Centered Fourier coefficients with shape ``(..., m, F)``.

    Returns
    -------
    Tensor
        Cumulant estimate with shape ``(..., F)``.
    """

    s2 = m / (m - 1) * torch.mean(centered_x * centered_y, dim=-2)
    return s2


def c3_factorized(m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor) -> Tensor:
    """Compute the unbiased third-order multivariate cumulant estimate.

        C3(x, y, z) = (m^2)/((m-1)(m-2)) * ((x-x.mean)*(y-y.mean)*(z-z.mean)).mean

    Parameters
    ----------
    m : int
        Number of windows in the estimate. Must exceed two.
    centered_x, centered_y : Tensor
        Centered Fourier coefficients with shape ``(..., m, F)``.
    centered_z : Tensor
        Centered third-frequency coefficients with shape ``(..., m, F, F)``.

    Returns
    -------
    Tensor
        Cumulant estimate with shape ``(..., F, F)``.
    """

    s3 = (
        m**2
        / ((m - 1) * (m - 2))
        * torch.mean(centered_x[..., :, :, None] * centered_y[..., :, None, :] * centered_z, dim=-3)
    )
    return s3


def c4_factorized(
    m: int, centered_x: Tensor, centered_y: Tensor, centered_z: Tensor, centered_w: Tensor
) -> Tensor:
    """Compute the unbiased fourth-order multivariate cumulant estimate.

        C4(x, y, z, w) = (m^2)/((m-1)(m-2)(m-3))
                        * ((m+1) * ((x-x.mean)*(y-y.mean)*(z-z.mean)*(w-w.mean)).mean
                            -(m-1) *(
                                    (((x-x.mean)*(y-y.mean)).mean * ((z-z.mean)*(w-w.mean)).mean)
                                    +(((x-x.mean)*(z-z.mean)).mean * ((y-y.mean)*(w-w.mean)).mean)
                                    +(((x-x.mean)*(w-w.mean)).mean * ((y-y.mean)*(z-z.mean)).mean)
                            )
                        )

    Parameters
    ----------
    m : int
        Number of windows in the estimate. Must exceed three.
    centered_x, centered_y, centered_z, centered_w : Tensor
        Centered Fourier coefficients with shape ``(..., m, F)``. The ``x`` and ``y`` factors vary
        along the first frequency axis; ``z`` and ``w`` vary along the second frequency axis.

    Returns
    -------
    Tensor
        Cumulant estimate with shape ``(..., F, F)``.
    """

    centered_xy = centered_x * centered_y
    centered_zw = centered_z * centered_w

    s4 = (
        m**2
        / ((m - 1) * (m - 2) * (m - 3))
        * (
            (m + 1) * _mean_outer(m, centered_xy, centered_zw)
            - (m - 1)
            * (
                torch.einsum("...f,...g->...fg", centered_xy.mean(dim=-2), centered_zw.mean(dim=-2))
                + _mean_outer(m, centered_x, centered_z) * _mean_outer(m, centered_y, centered_w)
                + _mean_outer(m, centered_x, centered_w) * _mean_outer(m, centered_y, centered_z)
            )
        )
    )
    return s4
