# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import torch
from torch import Tensor


def _gaussian(x: Tensor, N: int, sigma_t_prefactor: float) -> Tensor:
    """
    Helper function to calculate the Gaussian 
        G(x) = exp{- dt^2 [x - (N-1) / 2]^2 / [2 * sigma_t]^2}.

    sigma_t is the temporal width of the Gaussian. Here, it is given in terms 
    of the window duration T
        sigma_t = sigma_t_prefactor * T = sigma_t_prefactor * N * dt
    so that we effectively calculate
        G(x) = exp{- [x - (N-1) / 2]^2 / [2 * N * sigma_t_prefactor]^2 }

    This is used to construct the discrete approximate confined Gaussian
    window function for N-point Fourier transforms.
    (reference: DOI:10.1016/j.sigpro.2014.03.033)
    """
    center = (N - 1) * 0.5
    denom = 2.0 * N * sigma_t_prefactor

    t = (x - center) / denom
    return torch.exp(-t * t)


def _acG_window_func(N: int, sigma_t: float = 0.14) -> Tensor:
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
    k = torch.linspace(0, N-1, N)
    print(k)
    h = torch.tensor(-0.5)

    term_k = _gaussian(k, N, sigma_t)
    term_h = _gaussian(h, N, sigma_t)
    term_k_p_N = _gaussian(k + N, N, sigma_t)
    term_k_m_N = _gaussian(k - N, N, sigma_t)
    term_h_p_N = _gaussian(h + N, N, sigma_t)
    term_h_m_N = _gaussian(h - N, N, sigma_t)

    denom = term_h_p_N + term_h_m_N
    acG_k = term_k - (term_h * (term_k_p_N + term_k_m_N)) / denom

    acG_k_norm = acG_k / torch.max(acG_k)

    return acG_k_norm