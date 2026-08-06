# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .cumulants import build_s3_target_indices, c2_factorized, c3_factorized, c4_factorized
from .fft import TimestampWindow, WindowBuffer
from .planning import RuntimeConfig, SampledFrequencyPlan, TimestampFrequencyPlan


@dataclass(slots=True)
class ThirdOrderIndexCache:
    """Compact mapping from output-frequency pairs to closing FFT coefficients.

    Attributes
    ----------
    closing_fft_indices : Tensor
        Sorted unique indices into the shifted FFT, with shape ``(K,)``. These contain every valid
        closing frequency required by the output grid.
    gather_indices : Tensor
        Integer tensor with shape ``(F, F)`` mapping each output-frequency pair to an entry in the
        compact ``K`` axis. Invalid pairs contain placeholder index zero and must be interpreted
        using ``valid_mask``.
    valid_mask : Tensor
        Boolean tensor with shape ``(F, F)`` identifying output-frequency pairs whose closing
        frequency lies within the sampled FFT support.
    """

    closing_fft_indices: Tensor
    gather_indices: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class TimestampThirdOrderFrequencyCache:
    """Compact closing-frequency mapping for a timestamp frequency grid."""

    closing_frequencies: NDArray[np.float64]
    gather_indices: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class ThirdOrderCoefficients:
    """Compact closing-frequency coefficients for one channel.

    Attributes
    ----------
    values : Tensor
        Closing-frequency coefficients with shape ``(R, B, m, K)``.
    gather_indices : Tensor
        Integer tensor with shape ``(F, F)`` mapping output-frequency pairs onto the compact ``K``
        axis.
    valid_mask : Tensor
        Boolean tensor with shape ``(F, F)`` identifying valid sampled closing frequencies.
    """

    values: Tensor
    gather_indices: Tensor
    valid_mask: Tensor

    def gathered_centered_values(self) -> Tensor:
        """Center over ``m`` and gather onto the closing-frequency grid.

        Returns
        -------
        Tensor
            Temporary tensor with shape ``(R, B, m, F, F)``. Entries outside
            ``valid_mask`` are placeholders and must be masked by the caller.
        """
        centered = self.values - self.values.mean(dim=-2, keepdim=True)

        if centered.shape[-1] == 0:
            output_shape = centered.shape[:-1] + self.gather_indices.shape
            return centered.new_zeros(output_shape)

        return centered[..., self.gather_indices]


@dataclass(slots=True)
class ChannelCoefficients:
    """Prepared source-independent coefficients for one channel."""

    dc: Tensor
    output: Tensor
    third_order: ThirdOrderCoefficients | None = None
    _centered_output: Tensor | None = field(default=None, init=False, repr=False)

    def centered_output(self, conjugated: bool = False) -> Tensor:
        """Return output coefficients centered only over ``m``."""

        if self._centered_output is None:
            self._centered_output = self.output - self.output.mean(dim=-2, keepdim=True)

        if conjugated:
            return torch.conj(self._centered_output)

        return self._centered_output


@dataclass(slots=True)
class CoefficientBatch:
    """Compact coefficients for every active channel in one batch."""

    by_channel: dict[int, ChannelCoefficients]


def build_third_order_cache(
    runtime: RuntimeConfig,
    frequency_plan: SampledFrequencyPlan,
) -> ThirdOrderIndexCache:
    """Build the frequency-index mapping reused by all third-order spectra.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved frequency band, FFT length, and device.
    frequency_plan : SampledFrequencyPlan
        Common sampled frequency plan used by the current coefficient preparation path.

    Returns
    -------
    ThirdOrderIndexCache
        Compact shifted-FFT indices with shape ``(K,)`` and output-grid mapping tensors with shape
        ``(F, F)`` on the runtime device.
    """
    axis_indices = torch.arange(
        frequency_plan.band_start,
        frequency_plan.band_stop,
        device=runtime.device,
    )
    target_indices, valid_mask = build_s3_target_indices(axis_indices, frequency_plan.window_points)
    closing_fft_indices, inverse_indices = torch.unique(
        target_indices[valid_mask],
        sorted=True,
        return_inverse=True,
    )
    gather_indices = torch.zeros_like(target_indices)
    gather_indices[valid_mask] = inverse_indices

    return ThirdOrderIndexCache(
        closing_fft_indices=closing_fft_indices,
        gather_indices=gather_indices,
        valid_mask=valid_mask,
    )


def build_timestamp_third_order_cache(
    runtime: RuntimeConfig,
    frequency_plan: TimestampFrequencyPlan,
) -> TimestampThirdOrderFrequencyCache:
    """Build compact closing frequencies for a direct timestamp transform."""

    grid_indices = frequency_plan.grid_indices
    target_grid_indices = -(grid_indices[:, None] + grid_indices[None, :])
    closing_grid_indices, inverse_indices = np.unique(target_grid_indices, return_inverse=True)
    gather_indices = torch.as_tensor(
        inverse_indices.reshape(target_grid_indices.shape),
        dtype=torch.long,
        device=runtime.device,
    )

    return TimestampThirdOrderFrequencyCache(
        closing_frequencies=(closing_grid_indices.astype(np.float64) * frequency_plan.actual_df),
        gather_indices=gather_indices,
        valid_mask=torch.ones(target_grid_indices.shape, dtype=torch.bool, device=runtime.device),
    )


def build_coefficient_batch(
    frequency_plan: SampledFrequencyPlan,
    coeffs_by_channel: dict[int, Tensor],
    third_order_cache: ThirdOrderIndexCache | None,
) -> CoefficientBatch:
    """Select compact source-independent coefficients from sampled FFTs."""

    by_channel: dict[int, ChannelCoefficients] = {}

    for channel, coeffs in coeffs_by_channel.items():
        dc_index = coeffs.shape[-1] // 2
        third_order = None

        if third_order_cache is not None:
            third_order = ThirdOrderCoefficients(
                values=coeffs[..., third_order_cache.closing_fft_indices],
                gather_indices=third_order_cache.gather_indices,
                valid_mask=third_order_cache.valid_mask,
            )

        by_channel[channel] = ChannelCoefficients(
            dc=coeffs[..., dc_index].clone(),
            output=coeffs[..., frequency_plan.band_start : frequency_plan.band_stop].clone(),
            third_order=third_order,
        )

    return CoefficientBatch(by_channel=by_channel)


def compute_spectral_estimates(
    channels: tuple[int, ...],
    coefficient_batch: CoefficientBatch,
    window_buffer: WindowBuffer | TimestampWindow,
    runtime: RuntimeConfig,
) -> Tensor:
    """Compute normalized spectral estimates from channel Fourier coefficients.

    Dispatches to the cumulant implementation for orders 1 through 4 and applies the matching window
    normalization.

    Parameters
    ----------
    channels : tuple[int, ...]
        Specifies the corresponding channels of the spectrum, e.g. ``(0, 0, 0)`` for a
        third-order auto-spectrum.
    coefficient_batch : :class:`CoefficientBatch`
        Stores the precomputed Fourier coefficients and bands for the current slice.
    window_buffer : :class:`WindowBuffer`
        Stores all information related to the window function.
    runtime : :class:`RuntimeConfig`
        Resolved calculation settings derived from user configuration.

    Returns
    -------
    Tensor
        Spectral estimates for the specified spectrum. Output shape depends on order: order 1
        returns ``(R, B, 1)``, order 2 returns ``(R, B, F)``, and orders 3 and 4 return
        ``(R, B, F, F)``, where ``F`` is the selected-band length. Invalid third-order points, where
        ``w3 = -(w1 + w2)`` lies outside the shifted FFT support, are filled with ``NaN``.

    Raises
    ------
    ValueError
        If the channel tuple does not describe an order-one through order-four spectrum.
    """
    order = len(channels)
    windows_per_estimate = runtime.window_plan.windows_per_estimate

    if order == 1:
        coefficients = coefficient_batch.by_channel[channels[0]]
        cumulants = coefficients.dc.mean(dim=-1, keepdim=True)

    elif order == 2:
        cumulants = c2_factorized(
            windows_per_estimate,
            coefficient_batch.by_channel[channels[0]].centered_output(),
            coefficient_batch.by_channel[channels[1]].centered_output(conjugated=True),
        )

    elif order == 3:
        third_order = coefficient_batch.by_channel[channels[2]].third_order
        if third_order is None:
            raise ValueError("Third-order coefficients were not prepared.")

        cumulants = c3_factorized(
            windows_per_estimate,
            coefficient_batch.by_channel[channels[0]].centered_output(),
            coefficient_batch.by_channel[channels[1]].centered_output(),
            third_order.gathered_centered_values(),
        )

        nan_value = torch.full_like(cumulants, complex(float("nan"), 0.0))
        cumulants = torch.where(third_order.valid_mask, cumulants, nan_value)

    elif order == 4:
        cumulants = c4_factorized(
            windows_per_estimate,
            coefficient_batch.by_channel[channels[0]].centered_output(),
            coefficient_batch.by_channel[channels[1]].centered_output(conjugated=True),
            coefficient_batch.by_channel[channels[2]].centered_output(),
            coefficient_batch.by_channel[channels[3]].centered_output(conjugated=True),
        )
    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return cumulants / window_buffer.norm(order)
