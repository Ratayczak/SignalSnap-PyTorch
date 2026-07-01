# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor

from .utils import S3Calcs


def c1(use_full_fft: bool, a_w: Tensor) -> Tensor:
    """First-order cumulant.
    
    
    ``a_w`` has shape ``(m, F, 1)`` with ``F = runtime.f_max_idx - runtime.f_min_idx``. The returned
    tensor has shape ``(1,)`` and contains the DC component: index 0 for real FFT input, or the
    center frequency for full FFT input.
    """

    s1 = torch.mean(a_w, dim=0)
    if use_full_fft:
        dc_index = s1.shape[0] // 2
        result = s1[dc_index]
    else:
        result = s1[0]

    return result


def c2(m: int, a_w1: Tensor, a_w2: Tensor) -> Tensor:
    """Second-order cumulant is the covariance.
    
    ``a_w1`` and ``a_w2`` must both have the same shape ``(m, X, 1)``. The returned tensor has shape
    ``(X,)``.
    """

    a_w2_star = torch.conj(a_w2)

    factor = m / (m - 1)
    term_1 = torch.mean(a_w1 * a_w2_star, dim=0)
    term_2 = torch.mean(a_w1, dim=0) * torch.mean(a_w2_star, dim=0)
    s2 = factor * (term_1 - term_2)
    return s2.squeeze(-1)


def a_w3_gen(
    s3_calc: S3Calcs,
    f_max_idx: int,
    m: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """
    Generates an initialization tensor which will be used to calculate c3. Here sconfig.s3_calc can
    either be 1/2 or 1/4, which is chosen by the user. 1/2 to calculate positive and negative
    frequencies for x axis. 1/4 to calculate positive frequencies only.

    For ``s3_calc="1/4"``, returns a tensor with shape ``(H, H, m)`` for 
    ``H = runtime.f_max_idx // 2``. For ``s3_calc="1/2"``, returns shape ``(H, 2 * H - 1, m)``.
    """

    if s3_calc == "1/2":
        n = 2 * (f_max_idx // 2) - 1
        a_w3 = torch.ones((f_max_idx // 2, n, m), device=device, dtype=dtype) * 1j
    elif s3_calc == "1/4":
        a_w3 = torch.ones((f_max_idx // 2, f_max_idx // 2, m), device=device, dtype=dtype) * 1j
    else:
        raise ValueError(f"Unknown s3_calc: {s3_calc}")

    return a_w3


def index_generation_to_aw_3(
    s3_calc: S3Calcs, f_max_idx: int, device: torch.device | None = None
) -> Tensor:
    """
    Constructs an index matrix to correctly place elements of the Fourier coefficients of the signal
    in the desired order, accounting for potential spectrum symmetry. Here sconfig.s3_calc can
    either be 1/2 or 1/4, which is chosen by the user. 1/2 to calculate positive and negative
    frequencies for x axis. 1/4 to calculate positive frequencies only.


    The returned index tensor matches the first two dimensions of the third-order cache: ``(H, H)``
    for ``s3_calc="1/4"`` and ``(H, 2 * H - 1)`` for ``s3_calc="1/2"``, where
    ``H = f_max_idx // 2``.
    """

    if s3_calc == "1/2":
        n = f_max_idx // 2
        indices = torch.arange(n, device=device).unsqueeze(1) + torch.arange(
            -(n - 1), n, device=device
        )
    elif s3_calc == "1/4":
        indices = torch.arange(f_max_idx // 2, device=device).unsqueeze(1) + torch.arange(
            f_max_idx // 2, device=device
        )
    else:
        raise ValueError(f"Unknown s3_calc: {s3_calc}")

    return indices


def calc_a_w3(a_w_all: Tensor, f_max_idx: int, m: int, a_w3: Tensor, indices: Tensor) -> Tensor:
    """Fill the cached third-order coefficient tensor using precomputed frequency indices.

    The returned tensor contains the conjugated coefficients for ``w3 = -w1 - w2`` and is moved to
    the same dtype/device as ``a_w_all`` when needed.


    Parameters
    ----------
    a_w_all : Tensor
        Fourier coefficients for the third channel, rearranged to shape ``(K, 1, m)`` for 
        ``K = N if runtime.use_full_fft else N // 2 + 1``. For ``s3_calc="1/2"``, this tensor has to
        be already been extended with conjugated negative-frequency coefficients along the first
        axis to shape ``(2 * K - 1, 1, m)``.
    f_max_idx : int
        Upper frequency-band index from the runtime configuration. ``H = f_max_idx // 2`` determines
        the first dimension of the third-order output.
    m : int
        Number of windows in one spectral estimate.
    a_w3 : Tensor
        Cached work tensor with shape ``(H, H, m)`` for ``s3_calc="1/4"`` or
        ``(H, 2 * H - 1, m)`` for ``s3_calc="1/2"``.
    indices : Tensor
        Index tensor matching the first two dimensions of ``a_w3``.

    Returns
    -------
    Tensor
        Conjugated third-order coefficient tensor with the same shape as ``a_w3``.
    """

    if a_w3.dtype != a_w_all.dtype:
        a_w3 = a_w3.to(a_w_all.dtype)

    if a_w3.device != a_w_all.device:
        a_w3 = a_w3.to(a_w_all.device)

    row_ids = torch.arange(f_max_idx // 2, device=a_w_all.device)
    a_w3[row_ids, :, :] = a_w_all[indices, 0, :m]
    return a_w3.conj()


def c3(m: int, a_w1: Tensor, a_w2: Tensor, a_w3: Tensor) -> Tensor:
    """
    Third-order cumulant::

        C_3 = m^2 / [(m - 1) * (m - 2)] * {
                < a_w1 * a_w2 * a_w3 >
                - < a_w1 >< a_w2 * a_w3 > - < a_w1 * a_w2 >< a_w3 > - < a_w1 * a_w3 >< a_w2 >
                + 2 < a_w1 >< a_w2 >< a_w3 >
            }
    
    with w3 = - w1 - w2 and as before <...> denotes the mean and the factor m^2 / (m - 1)(m - 2) is
    the unbiased estimator for the third order cumulant. ``m`` is the number of windows per spectral 
    estimate. The estimator requires ``m > 2``.
    (see arXiv:1904.12154)

    For ``s3_calc="1/4"``, ``a_w1`` and ``a_w2`` have shape ``(m, H, 1)``, ``a_w3`` has shape
    ``(H, H, m)``, and the returned tensor has shape ``(H, H)``.  Here ``H = f_max_idx // 2``.

    For ``s3_calc="1/2"``, ``a_w1`` has shape ``(m, 2 * H - 1, 1)``, ``a_w2`` has shape
    ``(m, H, 1)``, ``a_w3`` has shape ``(H, 2 * H - 1, m)``, and the returned tensor has shape
    ``(H, 2 * H - 1)``.

    These shapes assume the third-order calculation starts at ``f_min_idx == 0``.
    """

    a_w1_modified = a_w1.transpose(-1, -2)
    a_w1_modified_stacked = a_w1_modified.expand(
        a_w1_modified.size(0), a_w2.size(1), a_w1_modified.size(2)
    )

    a_w2_modified_stacked = a_w2.expand((a_w2.size(0), a_w2.size(1), a_w1.size(1)))

    a_w3_modified = a_w3.permute(2, 0, 1)

    d_12 = a_w1_modified_stacked * a_w2_modified_stacked
    d_13 = a_w1_modified_stacked * a_w3_modified
    d_23 = a_w2_modified_stacked * a_w3_modified
    d_123 = d_12 * a_w3_modified

    d_means = [
        torch.mean(d, dim=0)
        for d in [
            a_w1_modified_stacked,
            a_w2_modified_stacked,
            a_w3_modified,
            d_12,
            d_13,
            d_23,
            d_123,
        ]
    ]

    d_1_mean, d_2_mean, d_3_mean, d_12_mean, d_13_mean, d_23_mean, d_123_mean = d_means
    s3 = (
        m**2
        / ((m - 1) * (m - 2))
        * (
            d_123_mean
            - d_12_mean * d_3_mean
            - d_13_mean * d_2_mean
            - d_23_mean * d_1_mean
            + 2 * d_1_mean * d_2_mean * d_3_mean
        )
    )

    return s3


def c4(m: int, a_w1: Tensor, a_w2: Tensor, a_w3: Tensor, a_w4: Tensor) -> Tensor:
    """
    Fourth-order cumulant::

        C_4 = m^2 / [(m - 1) * (m - 2) * (m - 3)] * {
                (m + 1) * <(a_w1 - <a_w1>) * (a_w2 - <a_w2>) * (a_w3 - <a_w3>) * (a_w4 - <a_w4>)>
                -(m - 1) * [
                           <(a_w1 - <a_w1>) * (a_w2 - <a_w2>)> * <(a_w3 - <a_w3>) * (a_w4 - <a_w4>)>
                           + 2 o.p.
                ]
            }
    
    <...> denotes the mean. ``m`` is the number of windows per spectral 
    estimate. The estimator requires ``m > 3``.
    (see arXiv:1904.12154)

    All input tensors must have shape ``(m, F, 1)``, where 
    ``F = runtime.f_max_idx - runtime.f_min_idx`` is the selected frequency-band length.
    The returned tensor has shape ``(F, F)``.

    The second and fourth inputs are conjugated internally, matching the convention used for 
    fourth-order auto- and cross-spectra.
    """

    # --- for a better readability ---
    x = a_w1
    y = torch.conj(a_w2)
    z = a_w3
    w = torch.conj(a_w4)
    # --------------------------------

    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    z_mean = z - z.mean(dim=0, keepdim=True)
    w_mean = w - w.mean(dim=0, keepdim=True)

    # Compute product and various partial means
    xyzw = torch.matmul((x_mean * y_mean), (z_mean * w_mean).transpose(-1, -2))
    xyzw_mean = xyzw.mean(dim=0)

    xy_mean = (x_mean * y_mean).mean(dim=0)
    zw_mean = (z_mean * w_mean).mean(dim=0)
    xy_zw_mean = torch.matmul(xy_mean, zw_mean.transpose(-1, -2))

    xz_mean = torch.matmul(x_mean, z_mean.transpose(-1, -2)).mean(dim=0)
    yw_mean = torch.matmul(y_mean, w_mean.transpose(-1, -2)).mean(dim=0)
    xz_yw_mean = xz_mean * yw_mean

    xw_mean = torch.matmul(x_mean, w_mean.transpose(-1, -2)).mean(dim=0)
    yz_mean = torch.matmul(y_mean, z_mean.transpose(-1, -2)).mean(dim=0)
    xw_yz_mean = xw_mean * yz_mean

    # Final combination
    s4 = (m**2 / ((m - 1) * (m - 2) * (m - 3))) * (
        (m + 1) * xyzw_mean - (m - 1) * (xy_zw_mean + xz_yw_mean + xw_yz_mean)
    )

    return s4
