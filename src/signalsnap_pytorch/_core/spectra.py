# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .cumulants import (
    build_s3_target_indices,
    c2_factorized,
    c3_factorized,
    c4_factorized,
    gather_s3_third_factor,
)
from .fft import WindowBuffer
from .planning import RuntimeConfig, SampledFrequencyPlan


@dataclass(slots=True)
class ThirdOrderIndexCache:
    """Cached mapping from output frequency pairs to the implied third frequency.

    Attributes
    ----------
    target_indices : Tensor
        Integer tensor with shape ``(F, F)``. Each entry indexes the shifted full FFT at
        ``w3 = -(w1 + w2)``. Invalid entries contain the safe placeholder index zero.
    valid_mask : Tensor
        Boolean tensor with shape ``(F, F)`` identifying entries whose implied third frequency lies
        within the full FFT support.
    """

    target_indices: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class ThirdOrderFactor:
    """Prepared third factor for a third-order cumulant.

    Attributes
    ----------
    centered_a_w3 : Tensor
        Centered Fourier coefficients with shape ``(B, m, F, F)`` gathered at the implied third
        frequencies.
    valid_mask : Tensor
        Boolean validity mask with shape ``(F, F)``.
    """

    centered_a_w3: Tensor
    valid_mask: Tensor


@dataclass(slots=True)
class IntermediateSliceBuffer:
    """Stores precomputed intermediate results used in :func:`compute_spectral_estimates`.

    Attributes
    ----------
    band_start_idx, band_end_idx : int
        Start-inclusive and end-exclusive indices selecting the requested band from the shifted
        full Fourier coefficients.
    m : int
        Number of windows per spectral estimate.
    fft_freq_count : int
        Length of the full Fourier coefficients.
    coeffs_by_channel : dict[int, Tensor]
        Shifted full Fourier coefficients by channel, each with shape ``(B, m, N)``.
    third_order_cache : :class:`ThirdOrderIndexCache` | None
        Indices of the third frequency for the corresponding frequency axis.
    """

    band_start_idx: int
    band_end_idx: int
    m: int
    fft_freq_count: int
    coeffs_by_channel: dict[int, Tensor] = field(default_factory=dict)
    third_order_cache: ThirdOrderIndexCache | None = None

    _centered_coeffs_by_channel_band: dict[int, Tensor] = field(default_factory=dict)
    _centered_c3_third_factor_by_channel: dict[int, ThirdOrderFactor] = field(default_factory=dict)

    def centered_coeffs_by_channel_band(self, channel: int, conjugated: bool = False) -> Tensor:
        """Return cached centered coefficients for one channel in the selected band.

        Parameters
        ----------
        channel : int
            Data-channel index present in ``coeffs_by_channel``.
        conjugated : bool, default=False
            Return the complex conjugate of the cached coefficients.

        Returns
        -------
        Tensor
            Tensor with shape ``(B, m, F)`` centered over the window axis.
        """
        if channel not in self._centered_coeffs_by_channel_band:
            coeffs = self.coeffs_by_channel[channel][..., self.band_start_idx : self.band_end_idx]
            self._centered_coeffs_by_channel_band[channel] = coeffs - coeffs.mean(
                dim=1, keepdim=True
            )

        if conjugated:
            return torch.conj(self._centered_coeffs_by_channel_band[channel])
        else:
            return self._centered_coeffs_by_channel_band[channel]

    def centered_c3_third_factor_by_channel(self, channel: int) -> ThirdOrderFactor:
        """Return the cached third-order factor for one channel.

        Parameters
        ----------
        channel : int
            Data-channel index present in ``coeffs_by_channel``.

        Returns
        -------
        ThirdOrderFactor
            Centered coefficients gathered on the ``(w1, w2)`` output grid.

        Raises
        ------
        ValueError
            If no :class:`ThirdOrderIndexCache` was supplied.
        """
        if channel not in self._centered_c3_third_factor_by_channel:
            if self.third_order_cache is None:
                raise ValueError("Third-order spectra require third_order_cache.")

            coeffs = self.coeffs_by_channel[channel]
            centered_a_w3 = gather_s3_third_factor(
                coeffs - coeffs.mean(dim=1, keepdim=True),
                self.third_order_cache.target_indices,
            )
            self._centered_c3_third_factor_by_channel[channel] = ThirdOrderFactor(
                centered_a_w3=centered_a_w3,
                valid_mask=self.third_order_cache.valid_mask,
            )

        return self._centered_c3_third_factor_by_channel[channel]


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
        Target indices and validity mask, each with shape ``(F, F)`` on the runtime device.
    """
    axis_indices = torch.arange(
        frequency_plan.band_start,
        frequency_plan.band_stop,
        device=runtime.device,
    )
    target_indices, valid_mask = build_s3_target_indices(axis_indices, frequency_plan.window_points)
    return ThirdOrderIndexCache(target_indices=target_indices, valid_mask=valid_mask)


def build_intermediate_slice_buffer(
    runtime: RuntimeConfig,
    frequency_plan: SampledFrequencyPlan,
    coeffs_by_channel: dict[int, Tensor],
    third_order_cache: ThirdOrderIndexCache | None,
) -> IntermediateSliceBuffer:
    """Create the reusable intermediate buffer for one calculation batch.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved band indices, FFT length, and window count.
    frequency_plan : SampledFrequencyPlan
        Common sampled frequency plan used by the current coefficient preparation path.
    coeffs_by_channel : dict[int, Tensor]
        Shifted full Fourier coefficients with shape ``(B, m, N)`` for each active channel.
    third_order_cache : ThirdOrderIndexCache | None
        Shared third-order index mapping, required when an order-three spectrum is requested.

    Returns
    -------
    IntermediateSliceBuffer
        Buffer that lazily caches centered and gathered coefficients.
    """
    return IntermediateSliceBuffer(
        band_start_idx=frequency_plan.band_start,
        band_end_idx=frequency_plan.band_stop,
        m=runtime.window_plan.windows_per_estimate,
        fft_freq_count=frequency_plan.window_points,
        coeffs_by_channel=coeffs_by_channel,
        third_order_cache=third_order_cache,
    )


def compute_spectral_estimates(
    channels: tuple[int, ...],
    intermediate_buffer: IntermediateSliceBuffer,
    window_buffer: WindowBuffer,
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
    intermediate_buffer : :class:`IntermediateSliceBuffer`
        Stores the precomputed Fourier coefficients and bands for the current slice.
    window_buffer : :class:`WindowBuffer`
        Stores all information related to the window function.
    runtime : :class:`RuntimeConfig`
        Resolved calculation settings derived from user configuration.

    Returns
    -------
    Tensor
        Spectral estimates for the specified spectrum. Output shape depends on order: order 1
        returns ``(B, 1)``, order 2 returns ``(B, F)``, and orders 3 and 4 return ``(B, F, F)``,
        where ``F`` is the selected-band length. Invalid third-order points, where
        ``w3 = -(w1 + w2)`` lies outside the shifted FFT support, are filled with ``NaN``.

    Raises
    ------
    ValueError
        If the channel tuple does not describe an order-one through order-four spectrum.
    """
    order = len(channels)
    windows_per_estimate = runtime.window_plan.windows_per_estimate

    if order == 1:
        a_w = intermediate_buffer.coeffs_by_channel[channels[0]]
        dc_index = a_w.shape[-1] // 2
        cumulants = a_w[:, :, dc_index].mean(dim=1, keepdim=True)

    elif order == 2:
        cumulants = c2_factorized(
            windows_per_estimate,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1], conjugated=True),
        )

    elif order == 3:
        prepared = intermediate_buffer.centered_c3_third_factor_by_channel(channels[2])

        cumulants = c3_factorized(
            windows_per_estimate,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1]),
            prepared.centered_a_w3,
        )

        nan_value = torch.full_like(cumulants, complex(float("nan"), 0.0))
        cumulants = torch.where(prepared.valid_mask, cumulants, nan_value)

    elif order == 4:
        cumulants = c4_factorized(
            windows_per_estimate,
            intermediate_buffer.centered_coeffs_by_channel_band(channels[0]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[1], conjugated=True),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[2]),
            intermediate_buffer.centered_coeffs_by_channel_band(channels[3], conjugated=True),
        )
    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return cumulants / window_buffer.norm(order)
