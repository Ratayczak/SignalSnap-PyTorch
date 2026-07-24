# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemad, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..configurators import DataConfig, SpectrumConfig
from .data_access import RuntimeChannel, get_sample_count
from .utils import ChannelIndex, FrequencyUnits, TimeUnits, unit_conversion_time_to_freq


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for;
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults,
    data-size constraints, frequency axes, and device details have been resolved.

    Attributes
    ----------
    active_data_channels : tuple[int, ...]
        Data-channel indices used by the calculation.
    spectra_channels : tuple[tuple[int, ...], ...]
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index.
    orders : tuple[int, ...]
        Orders at which spectra are computed.
    dt : float
        Sampling interval shared by all selected data channels.
    window_points : int
        Number of samples per window.
    m : int
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be positive.
    n_data_points : int
        Number of samples in each selected data channel.
    freq_all : np.ndarray
        Full frequency axis.
    freq_band : np.ndarray
        Selected frequency axis.
    band_start_idx, band_end_idx : int
        Slice indices selecting the configured frequency band.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
    spectral_estimates: int
        Number of unshifted spectral estimates processed by the base calculation. If
        ``interlacing=True``, up to the same number of additional shifted estimates are calculated
        when enough data is available.
    interlacing : bool
        Compute additional spectral estimates for windows shifted by half a window size, to
        compensate the low weight of data points produced by the window function near the original
        window edges.
    old_window : bool
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from the
        old API is used as a window function.
    """

    active_data_channels: tuple[int, ...]
    spectra_channels: tuple[tuple[ChannelIndex, ...], ...]
    orders: tuple[int, ...]
    dt: float
    window_points: int
    m: int
    n_data_points: int
    freq_all: np.ndarray
    freq_band: np.ndarray
    band_start_idx: int
    band_end_idx: int
    freq_unit: FrequencyUnits
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    spectral_estimates: int
    interlacing: bool
    old_window: bool


def _resolve_device(device_name: str) -> torch.device:
    """Validate and resolve a requested PyTorch device.

    Supported examples are ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ``"mps"``,
    ``"xpu"``, and ``"xpu:0"``.
    """
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device {device_name!r}.") from exc

    if device.type == "cpu":
        return device

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available. Check that PyTorch was installed "
                "with CUDA support and that an NVIDIA GPU and compatible driver are available."
            )

        index = device.index
        if index is None:
            index = torch.cuda.current_device()

        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f"CUDA device {index} was requested, but only {device_count} CUDA device(s) are "
                "available."
            )

        # Return an explicit device such as cuda:0.
        return torch.device("cuda", index)

    if device.type == "mps":
        if device.index is not None:
            raise ValueError("MPS devices do not support numbered device indices.")

        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS was requested, but this PyTorch installation was not built with MPS support."
            )

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available on this system.")

        return torch.device("mps")

    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError(
                "XPU was requested, but XPU is not available. Check that PyTorch was installed "
                "with Intel GPU support and that a supported Intel GPU and driver are available."
            )

        index = device.index
        if index is None:
            index = xpu.current_device()

        device_count = xpu.device_count()
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f"XPU device {index} was requested, but only {device_count} XPU device(s) are "
                "available."
            )

        return torch.device("xpu", index)

    raise ValueError(
        f"Unsupported device type {device.type!r}. Use 'cpu', 'cuda', 'cuda:N', 'mps', 'xpu', "
        "or 'xpu:N'."
    )


def normalize_channel_index(channel: object, channel_count: int) -> int:
    """Validate and normalize one data-channel index.

    Parameters
    ----------
    channel : object
        Candidate integer index. NumPy integers are accepted; Booleans are rejected.
    channel_count : int
        Number of available channels.

    Returns
    -------
    int
        Normalized built-in integer index.

    Raises
    ------
    TypeError
        If ``channel`` is not an integer or is Boolean.
    ValueError
        If ``channel`` is negative or outside the available channel range.
    """

    if isinstance(channel, (bool, np.bool_)):
        raise TypeError(f"Channel indices must be integers; received {channel!r}.")

    if not isinstance(channel, (int, np.integer)):
        raise TypeError(f"Channel indices must be integers; received {channel!r}.")

    normalized = int(channel)

    if normalized < 0:
        raise ValueError(f"Channel indices must be nonnegative; received {normalized}.")

    if normalized >= channel_count:
        raise ValueError(
            f"Channel {normalized} is out of bounds for {channel_count} available data channels."
        )

    return normalized


def resolve_channels(
    requested_spectra: list[tuple[int, ...]] | None,
    channel_count: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate and normalize the requested spectra.

    If ``requested_spectra`` is ``None``, generate auto-spectra of
    orders one through four for every available data channel.

    Parameters
    ----------
    requested_spectra : list[tuple[int, ...]] | None
        Explicit order-one through order-four channel tuples, or ``None`` for all default
        auto-spectra.
    channel_count : int
        Number of available input channels.

    Returns
    -------
    tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]
        Normalized spectrum requests and unique active data channels in first-use order.

    Raises
    ------
    TypeError
        If a request is not a tuple or contains a non-integer channel index.
    ValueError
        If no channels or requests are supplied, an order or index is invalid, or a request is
        duplicated.
    """
    if channel_count < 1:
        raise ValueError("At least one data channel is required.")

    if requested_spectra is None:
        spectra_channels = tuple(
            (channel,) * order for channel in range(channel_count) for order in range(1, 5)
        )
    else:
        if not requested_spectra:
            raise ValueError("requested_spectra must contain at least one spectrum.")

        resolved: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        for spectrum in requested_spectra:
            if not isinstance(spectrum, tuple):
                raise TypeError("Each spectrum request must be a tuple of channel indices.")

            order = len(spectrum)
            if not 1 <= order <= 4:
                raise ValueError(f"Spectrum order must be between 1 and 4; received order {order}.")

            resolved_channels: list[int] = []

            for channel in spectrum:
                resolved_channels.append(normalize_channel_index(channel, channel_count))

            resolved_spectrum = tuple(resolved_channels)

            if resolved_spectrum in seen:
                raise ValueError(
                    f"requested_spectra cannot contain duplicates; "
                    f"{resolved_spectrum} was requested more than once."
                )

            seen.add(resolved_spectrum)
            resolved.append(resolved_spectrum)

        spectra_channels = tuple(resolved)

    active_data_channels = tuple(
        dict.fromkeys(channel for spectrum in spectra_channels for channel in spectrum)
    )

    return spectra_channels, active_data_channels


def _get_and_validate_selected_channels(
    data_config: DataConfig,
    opened_channels: Mapping[int, RuntimeChannel],
) -> tuple[tuple[int, ...], int, float, TimeUnits]:
    """Resolve active channels and require them to have equal sample counts.

    Returns the active indices, common sample count, sampling interval, and time unit.
    """

    active_data_channels = tuple(opened_channels)
    first_channel = active_data_channels[0]
    first_sample_count = get_sample_count(opened_channels[first_channel])

    for channel in active_data_channels[1:]:
        sample_count = get_sample_count(opened_channels[channel])

        if sample_count != first_sample_count:
            raise ValueError(
                f"Channel {channel} contains {sample_count} samples, but channel {first_channel} "
                f"contains {first_sample_count} samples."
            )

    return active_data_channels, first_sample_count, data_config.dt, data_config.t_unit


def resolve_frequencies(
    spectrum_config: SpectrumConfig, dt: float
) -> tuple[int, NDArray[np.floating[Any]], int, int]:
    """Resolve frequencies based on the user's :class:`SpectrumConfig`.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        Spectrum configuration options.
    dt : float
        Time step of the specified data channels.

    Returns
    -------
    tuple[int, NDArray[np.floating[Any]], int, int]
        Resolved window length, full shifted FFT frequency grid, and start-inclusive/end-exclusive
        indices selecting the requested frequency band.

    Raises
    ------
    ValueError
        If the requested bounds exceed the Nyquist interval, the resolved window length is zero, or
        the requested band contains no FFT bins.
    """
    # Validate and resolve the frequency bounds
    f_max_allowed = 1 / (2 * dt)
    f_max = spectrum_config.f_max

    if f_max is None:
        f_max = f_max_allowed

        if f_max <= spectrum_config.f_min:
            raise ValueError("f_min is larger than the Nyquist frequency.")

    if f_max > f_max_allowed:
        raise ValueError("f_max is larger than the Nyquist frequency.")

    if spectrum_config.f_min < -f_max_allowed:
        raise ValueError("f_min outside of Nyquist frequency bounds.")

    # Compute how many points must be taken into account in one window to achieve the required
    # frequency spacing in the given frequency bounds.
    if spectrum_config.df is None:
        window_points = 1000
    else:
        window_points = int(np.round(1 / (spectrum_config.df * dt)))

    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    # get the frequency axis
    freq_all = np.fft.fftfreq(window_points, dt)
    freq_all = np.fft.fftshift(freq_all)

    band_start_idx = int(np.sum(freq_all < spectrum_config.f_min))
    band_end_idx = int(np.sum(freq_all <= f_max))

    if band_start_idx >= band_end_idx:
        raise ValueError(
            f"The requested frequency band [{spectrum_config.f_min}, {f_max}] does not contain "
            "any FFT frequencies at the resolved frequency spacing."
        )

    return window_points, freq_all, band_start_idx, band_end_idx


def build_runtime_config(
    data_config: DataConfig,
    opened_channels: Mapping[int, RuntimeChannel],
    spectrum_config: SpectrumConfig,
    spectra_channels: tuple[tuple[int, ...], ...],
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates the selected data channels, derives the frequency axis and frequency-band indices,
    checks Nyquist-frequency bounds, resolves the effective window size, and
    selects torch dtypes and device settings used by the spectrum calculation.

    Parameters
    ----------
    data_config : :class:`DataConfig`
        Data configurations containing the input data and sampling metadata.
    opened_channels : Mapping[int, Any | :class:`~signalsnap_pytorch._core.data_access.HDF5ChannelState`]
        Opened runtime representation of ``data_config.channels``. Array channels are retained
        directly; HDF5 channels are represented by :class:`~signalsnap_pytorch._core.data_access.HDF5ChannelState` instances.
    spectrum_config : :class:`SpectrumConfig`
        User configuration for frequency bounds, precision, device, windowing, and
        related calculation options.
    spectra_channels : tuple[tuple[int, ...], ...]
        Specifies which (multi-channel) spectra will be calculated. Each tuple represents one auto-
        or cross-correlation spectrum. Each tuple entry is a channel index which matches the index
        in ``data_config.channels``.

    Returns
    -------
    RuntimeConfig
        Resolved runtime configuration.

    Warns
    -----
    UserWarning
        If the configured ``m`` requires more samples than are available and is reduced at runtime.

    Raises
    ------
    ValueError
        If active channels have unequal lengths, frequency resolution fails, the effective ``m`` is
        below the highest requested order, or interlacing cannot produce a shifted estimate.
    RuntimeError
        If the requested accelerator is unavailable or its device index is invalid.
    """

    # Validate and read the channels, number of data points, and the time step from the
    # SpectrumConfig and DataConfigs
    active_data_channels, n_data_points, dt, t_unit = _get_and_validate_selected_channels(
        data_config=data_config,
        opened_channels=opened_channels,
    )

    window_points, freq_all, band_start_idx, band_end_idx = resolve_frequencies(spectrum_config, dt)

    # Check if enough data is available and try to lower the window count per cumulant/spectrum
    # estimate if needed
    required_points = window_points * spectrum_config.m
    if required_points > n_data_points:
        m = n_data_points // window_points
        warnings.warn(
            f"Not enough data points are available for m={spectrum_config.m}; using m={m} instead.",
            UserWarning,
            stacklevel=3,
        )
    else:
        m = spectrum_config.m

    orders = tuple(sorted({len(channels) for channels in spectra_channels}))
    if m < max(orders):
        raise ValueError("Not enough data points")

    device = _resolve_device(spectrum_config.device)

    # determine the data types based on the given precision
    if spectrum_config.precision == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif spectrum_config.precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        if device.type in {"mps", "xpu"}:
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

    # Determine the number of spectral estimates
    chunk_size = m * window_points
    unshifted_estimates = n_data_points // chunk_size

    if spectrum_config.spectral_estimates_max is None:
        spectral_estimates = unshifted_estimates
    else:
        spectral_estimates = min(spectrum_config.spectral_estimates_max, unshifted_estimates)

    # raise ValueError, if not a single shifted spectral estimate can be calculated when interlacing
    # is enabled.
    if spectrum_config.interlacing:
        shifted_estimates = (n_data_points - window_points // 2) // chunk_size
        if shifted_estimates < 1:
            raise ValueError(
                "Interlacing was requested, but the data is too short for a shifted spectral "
                "estimate. Disable interlacing or provide more data."
            )

    return RuntimeConfig(
        active_data_channels=active_data_channels,
        spectra_channels=tuple(spectra_channels),
        orders=orders,
        dt=dt,
        window_points=window_points,
        m=m,
        n_data_points=n_data_points,
        freq_all=freq_all,
        freq_band=freq_all[band_start_idx:band_end_idx],
        band_start_idx=band_start_idx,
        band_end_idx=band_end_idx,
        freq_unit=unit_conversion_time_to_freq(t_unit),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=device,
        spectral_estimates=spectral_estimates,
        interlacing=spectrum_config.interlacing,
        old_window=spectrum_config.old_window,
    )


def iter_window_slices(runtime: RuntimeConfig) -> Iterator[tuple[int, int, bool]]:
    """Return the window slice indices.

    Each yielded ``(start, end, shifted)`` selects ``m * N`` samples from a one-dimensional data
    channel, where ``m = runtime.m`` and ``N = runtime.window_points``. With interlacing enabled,
    additional slices shifted by ``N // 2`` are yielded when they still fit inside the signal.

    Parameters
    ----------
    runtime : RuntimeConfig
        Resolved window size, estimate limit, data length, and interlacing setting.

    Yields
    ------
    tuple[int, int, bool]
        Half-open sample bounds and whether the slice belongs to the shifted placement group.
    """

    chunk_size = runtime.window_points * runtime.m

    for chunk_index in range(runtime.spectral_estimates):
        start = chunk_index * chunk_size
        end = start + chunk_size
        yield start, end, False

    if runtime.interlacing:
        shift = runtime.window_points // 2
        shifted_estimates = window_slice_count(runtime) - runtime.spectral_estimates
        for chunk_index in range(shifted_estimates):
            start = chunk_index * chunk_size + shift
            end = start + chunk_size
            yield start, end, True


def window_slice_count(runtime: RuntimeConfig) -> int:
    """Return the total number of unshifted and shifted spectral-estimate slices.

    The shifted count is limited both by available data and by the resolved unshifted estimate
    count, which already incorporates ``spectral_estimates_max``.
    """

    total = runtime.spectral_estimates
    if not runtime.interlacing:
        return total

    chunk_size = runtime.window_points * runtime.m
    available_shifted = max(0, (runtime.n_data_points - runtime.window_points // 2) // chunk_size)
    return total + min(runtime.spectral_estimates, available_shifted)
