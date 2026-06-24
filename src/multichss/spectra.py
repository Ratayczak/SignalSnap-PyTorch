# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .cumulants import (
    a_w3_gen,
    c1,
    c2,
    c3,
    c4,
    calc_a_w3,
    index_generation_to_aw_3,
)

if TYPE_CHECKING:
    from .planning import RuntimeConfig, SpectrumTask


@dataclass(slots=True)
class ThirdOrderCache:
    a_w3_init: Tensor
    indices: Tensor


def build_third_order_cache(runtime: RuntimeConfig) -> ThirdOrderCache:
    return ThirdOrderCache(
        a_w3_init=a_w3_gen(
            runtime.s3_calc,
            runtime.f_max_idx,
            runtime.m,
            device=runtime.device,
        ),
        indices=index_generation_to_aw_3(
            runtime.s3_calc,
            runtime.f_max_idx,
            device=runtime.device,
        ),
    )


def compute_single_spectrum(
    task: SpectrumTask,
    coeffs_by_channel: dict[int, Tensor],
    single_window: Tensor,
    runtime: RuntimeConfig,
    third_order_cache: ThirdOrderCache | None = None,
) -> Tensor:
    order = task.order
    channels = task.channels
    f_min_idx = runtime.f_min_idx
    f_max_idx = runtime.f_max_idx

    if len(channels) != order:
        raise ValueError(
            f"Number of selected channels ({len(channels)}) does not match order ({order})"
        )

    if order == 1:
        coeffs = coeffs_by_channel[channels[0]]
        a_w = coeffs[:, f_min_idx:f_max_idx, :]
        single_spectrum = c1(runtime.use_full_fft, a_w)
        norm = runtime.dt * single_window.mean() * single_window.shape[0]

    elif order == 2:
        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx:f_max_idx, :]
        a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx:f_max_idx, :]
        single_spectrum = c2(runtime.m, a_w1, a_w2)
        norm = runtime.dt * (single_window ** 2).sum()

    elif order == 3:
        if third_order_cache is None:
            raise ValueError("Third-order spectra require third_order_cache.")

        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx:f_max_idx // 2, :]
        if channels[0] == channels[1]:
            a_w2 = a_w1
        else:
            a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx:f_max_idx // 2, :]

        coeffs_gpu_p = coeffs_by_channel[channels[2]].permute((1, 2, 0))

        if runtime.s3_calc == "1/2":
            a_w1 = torch.cat((a_w1[:, 1:, :].flip([1]).conj(), a_w1), dim=1)
            coeffs_gpu_p = torch.cat(
                (coeffs_gpu_p, torch.conj(coeffs_gpu_p[1:, :, :].flip([0]))),
                dim=0,
            )

        a_w3 = calc_a_w3(
            coeffs_gpu_p,
            f_max_idx,
            runtime.m,
            third_order_cache.a_w3_init,
            third_order_cache.indices,
        )

        single_spectrum = c3(runtime.m, a_w1, a_w2, a_w3)
        norm = runtime.dt * (single_window**3).sum()

    elif order == 4:
        a_w1 = coeffs_by_channel[channels[0]][:, f_min_idx:f_max_idx, :]
        a_w2 = coeffs_by_channel[channels[1]][:, f_min_idx:f_max_idx, :]
        a_w3 = coeffs_by_channel[channels[2]][:, f_min_idx:f_max_idx, :]
        a_w4 = coeffs_by_channel[channels[3]][:, f_min_idx:f_max_idx, :]

        single_spectrum = c4(runtime.m, a_w1, a_w2, a_w3, a_w4)
        norm = runtime.dt * (single_window**4).sum()

    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return torch.conj(single_spectrum / norm)