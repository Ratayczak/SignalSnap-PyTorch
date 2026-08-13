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
from .data_access import RuntimeSource, read_source
from .fft import compute_fft, reshape_window_chunk, to_device
from .plans import (
    DirectFrequencyPlan,
    FFTFrequencyPlan,
    FrequencyPlan,
    RuntimeConfig,
    SampledChannelPlan,
    TimestampedChannelPlan,
    WindowBatch,
)
from .window import SampledWindow, TimestampWindow

_COEFFICIENT_ROLE_CONJUGATIONS = {
    1: (False,),
    2: (False, True),
    3: (False, False, False),
    4: (False, True, False, True),
}


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
    """Compact closing-frequency mapping for timestamp coefficients.

    Attributes
    ----------
    closing_frequencies : NDArray[np.float64]
        Sorted unique frequencies ``f3 = -(f1 + f2)`` required by the output grid, with shape
        ``(K,)``.
    gather_indices : Tensor
        Integer tensor with shape ``(F, F)`` mapping every output-frequency pair to the
        corresponding entry on the compact ``K`` axis.
    valid_mask : Tensor
        Boolean tensor with shape ``(F, F)``. It is entirely true because timestamp coefficients can
        be directly evaluated outside sampled FFT support.
    """

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
    """Prepared source-independent Fourier coefficients for one channel.

    The same representation is used for sampled FFT coefficients and directly transformed timestamp
    coefficients.

    Attributes
    ----------
    dc : Tensor
        Zero-frequency coefficients with shape ``(R, B, m)``.
    output : Tensor | None
        Coefficients on the selected output-frequency band, with shape ``(R, B, m, F)``, or ``None``
        when the output band was not required.
    third_order : ThirdOrderCoefficients | None
        Compact coefficients at third-order closing frequencies, or ``None`` when the channel is not
        used as the closing-frequency factor.
    """

    dc: Tensor
    output: Tensor | None
    third_order: ThirdOrderCoefficients | None = None
    _centered_output: Tensor | None = field(default=None, init=False, repr=False)

    @property
    def realization_count(self) -> int:
        """Return the size of the shared realization axis."""

        return int(self.dc.shape[0])

    def centered_output(self, conjugated: bool = False) -> Tensor:
        """Return output-band coefficients centered over physical windows.

        The mean is removed along the ``m`` axis while preserving the realization, estimate, and
        frequency axes. The centered tensor is calculated lazily and cached for reuse by multiple
        requested spectra.

        Parameters
        ----------
        conjugated : bool, default=False
            Return the complex conjugate of the centered coefficients.

        Returns
        -------
        Tensor
            Centered coefficients with the same ``(R, B, m, F)`` shape as ``output``.

        Raises
        ------
        RuntimeError
            If output-band coefficients were not prepared.
        """

        if self.output is None:
            raise RuntimeError("Output-band coefficients were not prepared.")

        if self._centered_output is None:
            self._centered_output = self.output - self.output.mean(dim=-2, keepdim=True)

        if conjugated:
            return torch.conj(self._centered_output)

        return self._centered_output


def prepare_spectrum_normalizations(
    runtime: RuntimeConfig,
    sampled_window: SampledWindow | None,
    timestamp_window: TimestampWindow | None,
) -> dict[tuple[int, ...], Tensor]:
    """Prepare the scalar normalization for every requested spectrum.

    Homogeneous sampled and timestamp tuples use their order-specific window norms. Mixed
    polyspectra use the discrete overlap of the sampled and timestamped windows evaluated on the
    sampled-data time grid.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved requested spectra and active channel types.
    sampled_window : SampledWindow | None
        Prepared sampled-data window, or ``None`` if no sampled channel is active.
    timestamp_window : TimestampWindow | None
        Prepared timestamp window, or ``None`` if no timestamped channel is active.

    Returns
    -------
    dict[tuple[int, ...], Tensor]
        Scalar normalization keyed by requested spectrum tuple.

    Raises
    ------
    RuntimeError
        If a requested spectrum requires a window that was not prepared.
    TypeError
        If a requested spectrum references an unsupported channel plan.
    ValueError
        If the spectrum order is unsupported or a mixed overlap is exactly zero or numerically
        negligible.
    """

    sampled_plans = (
        plan for plan in runtime.channel_plans.values() if isinstance(plan, SampledChannelPlan)
    )
    first_sampled_plan = next(sampled_plans, None)
    sampled_dt = first_sampled_plan.dt if first_sampled_plan is not None else None

    sample_times = None
    timestamp_window_on_sample_grid = None
    if sampled_dt is not None:
        if sampled_window is None:
            raise RuntimeError("Sampled window was not prepared.")

        sample_times = (
            torch.arange(
                sampled_window.window.numel(),
                dtype=runtime.real_dtype,
                device=runtime.device,
            )
            * sampled_dt
        )
        if timestamp_window is not None:
            timestamp_window_on_sample_grid = timestamp_window.evaluate(sample_times)

    normalizations: dict[tuple[int, ...], Tensor] = {}

    for spectrum_channels in runtime.requested_spectra:
        order = len(spectrum_channels)
        try:
            conjugated_roles = _COEFFICIENT_ROLE_CONJUGATIONS[order]
        except KeyError as exc:
            raise ValueError(f"Unsupported spectrum order: {order}.") from exc

        channel_plans = tuple(runtime.channel_plans[channel] for channel in spectrum_channels)
        all_sampled = all(isinstance(plan, SampledChannelPlan) for plan in channel_plans)
        all_timestamped = all(isinstance(plan, TimestampedChannelPlan) for plan in channel_plans)

        if all_sampled:
            if sampled_window is None:
                raise RuntimeError("Sampled window was not prepared.")
            normalization_value = sampled_window.norm(order)
        elif all_timestamped:
            if timestamp_window is None:
                raise RuntimeError("Timestamp normalization was not prepared.")
            normalization_value = timestamp_window.norm(order)
        else:
            if sampled_window is None or sampled_dt is None or sample_times is None:
                raise RuntimeError("Sampled window was not prepared.")
            if timestamp_window is None or timestamp_window_on_sample_grid is None:
                raise RuntimeError("Timestamp window was not prepared.")

            factors: list[Tensor] = []
            for channel, channel_plan, conjugated in zip(
                spectrum_channels, channel_plans, conjugated_roles
            ):
                if isinstance(channel_plan, SampledChannelPlan):
                    factor = sampled_window.window
                elif isinstance(channel_plan, TimestampedChannelPlan):
                    factor = timestamp_window_on_sample_grid
                else:
                    raise TypeError(
                        f"Channel {channel} has unsupported plan {type(channel_plan).__name__}."
                    )

                if conjugated:
                    factor = torch.conj(factor)

                factors.append(factor)

            factor_product = torch.prod(torch.stack(factors), dim=0)
            factor_sum = factor_product.sum()
            magnitude_scale = torch.abs(factor_product).sum()
            threshold = torch.finfo(runtime.real_dtype).eps * magnitude_scale
            if bool(torch.abs(factor_sum) <= threshold):
                raise ValueError(
                    f"Mixed window overlap for spectrum {spectrum_channels} is zero or "
                    "numerically negligible."
                )

            normalization_value = sampled_dt * factor_sum

        normalizations[spectrum_channels] = normalization_value

    return normalizations


def build_third_order_cache(
    runtime: RuntimeConfig,
    frequency_plan: FFTFrequencyPlan,
) -> ThirdOrderIndexCache:
    """Build the frequency-index mapping reused by all third-order spectra.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved frequency band, FFT length, and device.
    frequency_plan : FFTFrequencyPlan
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
    frequency_plan: FrequencyPlan,
) -> TimestampThirdOrderFrequencyCache:
    """Build compact timestamp closing frequencies for a third-order output grid.

    For every output-frequency pair ``(f1, f2)``, the third factor is evaluated at
    ``f3 = -(f1 + f2)``. Unique closing frequencies are stored once and a gather map reconstructs
    the full ``(F, F)`` grid.

    Timestamp coefficients are evaluated by direct transformation, even when the output grid comes
    from an FFT plan. Consequently, closing frequencies outside the sampled FFT support remain
    valid.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved calculation device.
    frequency_plan : FFTFrequencyPlan | DirectFrequencyPlan
        Output-frequency grid whose closing frequencies are required.

    Returns
    -------
    TimestampThirdOrderFrequencyCache
        ``closing_frequencies`` contains ``K`` unique frequencies. ``gather_indices`` has shape
        ``(F, F)`` on ``runtime.device`` and maps each output pair to the compact ``K`` axis.
        ``valid_mask`` has shape ``(F, F)`` and is entirely true.

    Raises
    ------
    TypeError
        If ``frequency_plan`` has an unsupported type.
    """

    if isinstance(frequency_plan, DirectFrequencyPlan):
        grid_indices = frequency_plan.grid_indices
        actual_df = frequency_plan.actual_df
    elif isinstance(frequency_plan, FFTFrequencyPlan):
        grid_indices = (
            np.arange(frequency_plan.band_start, frequency_plan.band_stop, dtype=np.int64)
            - frequency_plan.window_points // 2
        )
        actual_df = 1.0 / runtime.window_plan.duration
    else:
        raise TypeError(f"Unsupported frequency plan {type(frequency_plan).__name__}.")

    target_grid_indices = -(grid_indices[:, None] + grid_indices[None, :])
    closing_grid_indices, inverse_indices = np.unique(target_grid_indices, return_inverse=True)
    gather_indices = torch.as_tensor(
        inverse_indices.reshape(target_grid_indices.shape),
        dtype=torch.long,
        device=runtime.device,
    )

    return TimestampThirdOrderFrequencyCache(
        closing_frequencies=(closing_grid_indices.astype(np.float64) * actual_df),
        gather_indices=gather_indices,
        valid_mask=torch.ones(target_grid_indices.shape, dtype=torch.bool, device=runtime.device),
    )


def _build_coefficient_batch(
    frequency_plan: FFTFrequencyPlan,
    coeffs_by_channel: dict[int, Tensor],
    third_order_cache: ThirdOrderIndexCache | None,
) -> dict[int, ChannelCoefficients]:
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

    return by_channel


def prepare_sampled_channel_coefficients(
    channel_index: int,
    source: RuntimeSource,
    channel_plan: SampledChannelPlan,
    batch: WindowBatch,
    frequency_plan: FFTFrequencyPlan,
    sampled_window: SampledWindow,
    runtime: RuntimeConfig,
    third_order_cache: ThirdOrderIndexCache | None,
) -> ChannelCoefficients:
    """Read and transform one sampled channel for a physical-window batch.

    The function reads the contiguous source range covered by ``batch``, reshapes it to
    ``(B, m, N)``, transfers it to the calculation device, applies the prepared window, and computes
    shifted FFT coefficients. Here ``B`` is the number of estimates, ``m`` is the number of physical
    windows per estimate, and ``N`` is the number of samples per physical window.

    Parameters
    ----------
    channel_index : int
        Index identifying the source channel.
    source : RuntimeSource
        Opened one-dimensional source for the channel.
    channel_plan : SampledChannelPlan
        Resolved sample count and sampling interval.
    batch : WindowBatch
        Physical windows to read and transform.
    frequency_plan : FFTFrequencyPlan
        Shifted FFT grid and selected output band.
    sampled_window : SampledWindow
        Sampled window tensor and order-normalization factors.
    runtime : RuntimeConfig
        Resolved window layout, device, and numeric dtypes.
    third_order_cache : ThirdOrderIndexCache | None
        Closing-frequency selection for third-order spectra, if required.

    Returns
    -------
    ChannelCoefficients
        Coefficients with one deterministic realization. ``dc`` has shape ``(1, B, m)``, ``output``
        has shape ``(1, B, m, F)``, and, when requested, ``third_order.values`` has shape
        ``(1, B, m, K)``.
    """

    window_points = round(batch.duration / channel_plan.dt)
    start = round(float(batch.relative_starts[0, 0]) / channel_plan.dt)
    stop = start + batch.estimate_count * runtime.window_plan.windows_per_estimate * window_points
    data = read_source(source, start, stop)
    chunk = reshape_window_chunk(
        chunk=data,
        estimate_count=batch.estimate_count,
        windows_per_estimate=runtime.window_plan.windows_per_estimate,
        window_points=window_points,
    )
    chunk = to_device(chunk, runtime)
    coefficients = compute_fft(chunk=chunk, window=sampled_window.window, dt=channel_plan.dt)
    coefficients_by_channel = _build_coefficient_batch(
        frequency_plan=frequency_plan,
        coeffs_by_channel={channel_index: coefficients},
        third_order_cache=third_order_cache,
    )
    return coefficients_by_channel[channel_index]


def expand_deterministic_coefficients(
    coefficients_by_channel: dict[int, ChannelCoefficients],
    realization_count: int,
) -> dict[int, ChannelCoefficients]:
    """Expand deterministic sampled coefficients across a realization batch.

    Only the leading realization axis is expanded. For more than one realization, PyTorch expansion
    views are used, so the coefficient data is not copied. Cache mapping tensors are shared
    unchanged.

    Parameters
    ----------
    coefficients_by_channel : dict[int, ChannelCoefficients]
        Sampled coefficients whose leading realization axis has length one.
    realization_count : int
        Required length of the expanded realization axis.

    Returns
    -------
    dict[int, ChannelCoefficients]
        The original mapping when ``realization_count`` is one; otherwise, a new mapping containing
        coefficient objects with expanded tensor views.

    Raises
    ------
    ValueError
        If ``realization_count`` is less than one.
    RuntimeError
        If a coefficient tensor does not have exactly one deterministic realization.
    """

    if realization_count < 1:
        raise ValueError("At least one realization is required.")

    if realization_count == 1:
        return coefficients_by_channel

    def expand(values: Tensor) -> Tensor:
        if values.shape[0] != 1:
            raise RuntimeError("Deterministic coefficients must contain exactly one realization.")

        return values.expand(realization_count, *values.shape[1:])

    expanded_channels = {}

    for channel, coefficients in coefficients_by_channel.items():
        third_order = coefficients.third_order

        if third_order is not None:
            third_order = ThirdOrderCoefficients(
                values=expand(third_order.values),
                gather_indices=third_order.gather_indices,
                valid_mask=third_order.valid_mask,
            )

        expanded_channels[channel] = ChannelCoefficients(
            dc=expand(coefficients.dc),
            output=expand(coefficients.output) if coefficients.output is not None else None,
            third_order=third_order,
        )

    return expanded_channels


def compute_spectral_estimates(
    channels: tuple[int, ...],
    coefficients_by_channel: dict[int, ChannelCoefficients],
    normalization: Tensor,
    runtime: RuntimeConfig,
) -> Tensor:
    """Compute normalized spectral estimates from prepared Fourier coefficients.

    The channel tuple determines both the spectrum order and the role of each coefficient:

    - ``(a,)`` averages the zero-frequency coefficient of channel ``a``.
    - ``(a, b)`` combines ``X_a(f)`` with ``conj(X_b(f))``.
    - ``(a, b, c)`` combines ``X_a(f1)``, ``X_b(f2)``, and ``X_c(-(f1 + f2))``.
    - ``(a, b, c, d)`` combines ``X_a(f1)``, ``conj(X_b(f1))``, ``X_c(f2)``, and ``conj(X_d(f2))``.

    The appropriate unbiased multivariate cumulant estimator is evaluated over the ``m`` physical
    windows in each spectral estimate and divided by the prepared normalization.

    Parameters
    ----------
    channels : tuple[int, ...]
        One through four channel indices defining the spectrum and coefficient roles.
    coefficients_by_channel : dict[int, ChannelCoefficients]
        Prepared coefficients for every channel referenced by ``channels``.
    normalization : Tensor
        Scalar window normalization.
    runtime : RuntimeConfig
        Resolved number of physical windows per estimate.

    Returns
    -------
    Tensor
        Normalized estimates. Order one has shape ``(R, B, 1)``, order two has shape ``(R, B, F)``,
        and orders three and four have shape ``(R, B, F, F)``. ``R`` is the realization count, ``B``
        is the estimate batch size, and ``F`` is the output-band length. Third-order points whose
        closing frequency lies outside sampled FFT support are filled with ``NaN``; timestamp
        closing frequencies remain valid because they are transformed directly.

    Raises
    ------
    KeyError
        If coefficients for a referenced channel are missing.
    ValueError
        If the requested order is unsupported or required third-order coefficients were not
        prepared.
    RuntimeError
        If required output-band coefficients were not prepared.
    """

    order = len(channels)
    windows_per_estimate = runtime.window_plan.windows_per_estimate

    if order == 1:
        coefficients = coefficients_by_channel[channels[0]]
        cumulants = coefficients.dc.mean(dim=-1, keepdim=True)

    elif order == 2:
        cumulants = c2_factorized(
            windows_per_estimate,
            coefficients_by_channel[channels[0]].centered_output(),
            coefficients_by_channel[channels[1]].centered_output(conjugated=True),
        )

    elif order == 3:
        third_order = coefficients_by_channel[channels[2]].third_order
        if third_order is None:
            raise ValueError("Third-order coefficients were not prepared.")

        cumulants = c3_factorized(
            windows_per_estimate,
            coefficients_by_channel[channels[0]].centered_output(),
            coefficients_by_channel[channels[1]].centered_output(),
            third_order.gathered_centered_values(),
        )

        nan_value = torch.full_like(cumulants, complex(float("nan"), 0.0))
        cumulants = torch.where(third_order.valid_mask, cumulants, nan_value)

    elif order == 4:
        cumulants = c4_factorized(
            windows_per_estimate,
            coefficients_by_channel[channels[0]].centered_output(),
            coefficients_by_channel[channels[1]].centered_output(conjugated=True),
            coefficients_by_channel[channels[2]].centered_output(),
            coefficients_by_channel[channels[3]].centered_output(conjugated=True),
        )
    else:
        raise ValueError(f"Unsupported spectrum order: {order}.")

    return cumulants / normalization
