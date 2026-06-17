# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import Tensor


def _gaussian(x: Tensor, N: int, sigma_t: float) -> Tensor:
    """
    Helper function to calculate the Gaussian G(x).

    This is used to construct the discrete approximate confined Gaussian
    window function for N-point Fourier transforms.
    (see DOI:10.1016/j.sigpro.2014.03.033)
    """
    center = (N - 1) * 0.5
    denom = 2.0 * N * sigma_t

    t = (x - center) / denom
    return torch.exp(-t * t)


def _prop_acG_window_func(k: Tensor, N: int, sigma_t: float) -> Tensor:
    """
    Helper function to calculate the *proportionallity* of the discrete
    approximate confined Gaussian window function g_k^(acG) for N-point Fourier
    transforms.
    (see DOI:10.1016/j.sigpro.2014.03.033)
    """
    h = k.new_tensor(-0.5)

    term_k = _gaussian(k, N, sigma_t)
    term_h = _gaussian(h, N, sigma_t)
    term_k_p_N = _gaussian(k + N, N, sigma_t)
    term_k_m_N = _gaussian(k - N, N, sigma_t)
    term_h_p_N = _gaussian(h + N, N, sigma_t)
    term_h_m_N = _gaussian(h - N, N, sigma_t)

    denom = term_h_p_N + term_h_m_N
    acG_k = term_k - (term_h * (term_k_p_N + term_k_m_N)) / denom

    return acG_k


def _acG_window_func(N: int, frequency_step: float) -> tuple[Tensor, float]:
    """
    Helper function to calculate the discrete approximate confined Gaussian
    window function g_k^(acG) for N-point Fourier transforms.
    (see DOI:10.1016/j.sigpro.2014.03.033)
    """

    k = torch.linspace(0, N - 1, N)
    sigma_t = 0.14

    prop_acG_window_func = _prop_acG_window_func(k, N, sigma_t)
    norm_t = (
        prop_acG_window_func * prop_acG_window_func
    ).sum() / frequency_step

    acG_window_func = prop_acG_window_func / torch.sqrt(norm_t)
    norm = float(norm_t.item())

    return acG_window_func, norm
