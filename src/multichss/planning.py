# This file is part of SignalSnap (PyTorch): Signal Analysis In Python Made Easy
# Copyright (c) 2024 and later, Armin Ghorbanietemed, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .configurators import CrossConfig, DataConfig, SpectrumConfig
from .results import SpectrumResult, SpectrumResultStore
from .utils import FrequencyUnits, S3Calcs, TimeUnits, unit_conversion_time_to_freq


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved calculation settings derived from user configuration.

    :class:`SpectrumConfig` and :class:`DataConfig` describe what the user asked for
    :class:`RuntimeConfig` describes what the calculation will actually use after defaults,
    data-size constraints, frequency axes, and device details have been resolved.

    Attributes
    ----------
    selected_channels : tuple[int, ...]
        Data-channel indices used by the calculation.
    orders : tuple[int, ...]
        Spectrum orders to calculate.
    dt : float
        Sampling interval shared by all selected data channels.
    window_points : int
        Number of samples per window.
    m : int
        Number of windows used per spectral estimate. This may be reduced at runtime if the signal
        is too short. Must be positive.
    n_data_points : int
        Number of samples in each selected data channel.
    n_windows : int
        Number of window groups processed by the calculation.
    freq_band : np.ndarray
        Selected frequency axis.
    freq_unit : Literal["Hz", "kHz", "MHz", "GHz", "THz"]
        Unit of the frequency axis.
    f_min_idx, f_max_idx : int
        Slice indices selecting the configured frequency band.
    use_full_fft : bool
        Whether negative frequencies require full FFT handling.
    real_dtype : torch.dtype
        Sets the dtype of floats.
    complex_dtype : torch.dtype
        Sets the dtype of complex numbers.
    device : torch.device
        Torch device used for calculation.
    s3_calc : Literal["1/4", "1/2"]
        Method used for third-order spectrum calculation.
    spectral_estimates: int
        Number of spectral estimates.
    old_window : bool
        Compatibility option. If set to ``True``, the approximated confined Gaussian window from the
        old API is used as a window function.
    """

    selected_channels: tuple[int, ...]
    orders: tuple[int, ...]
    dt: float
    window_points: int
    m: int
    n_data_points: int
    n_windows: int
    freq_band: np.ndarray
    freq_unit: FrequencyUnits
    f_min_idx: int
    f_max_idx: int
    use_full_fft: bool
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    device: torch.device
    s3_calc: S3Calcs
    spectral_estimates: int
    old_window: bool


@dataclass(frozen=True, slots=True)
class SpectrumTask:
    """Description of one spectrum that should be calculated.

    A task is the normalized representation of a user request after :class:`SpectrumConfig` and
    :class:`CrossConfig` have been expanded. It identifies one polyspectrum order and the channel
    tuple that should be used for that calculation.

    Parameters
    ----------
    order : int
        The polyspectrum order to calculate.
    channels : tuple[int, ...]
        The channel indices used by this calculation. Auto-spectra repeat the same channel once per
        order, e.g. ``(0, 0)`` for the second-order spectrum of channel 0 and ``(0, 0, 0)`` for the
        third-order spectrum. Cross-spectra are represented by the configured channel tuple, e.g.
        ``(0, 1)`` for a second-order cross-spectrum.
    """

    order: int
    channels: tuple[int, ...]


def _normalize_selected(
    data_config_list: list[DataConfig], selected: list[int] | None = None
) -> tuple[int, ...]:
    """Resolve selected data-channel indices."""

    if selected is None:
        return tuple(range(len(data_config_list)))

    if not selected:
        raise ValueError("At least one data channel must be selected.")

    if len(selected) != len(set(selected)):
        raise ValueError("Selected data channels cannot contain duplicates.")

    n_data_configs = len(data_config_list)
    for channel in selected:
        if channel < 0 or channel >= n_data_configs:
            raise IndexError(
                f"Selected channel {channel} is outside available data "
                f"channels 0..{n_data_configs - 1}."
            )

    return tuple(selected)


def _validate_data_configs(
    data_config_list: list[DataConfig], selected: tuple[int, ...]
) -> tuple[int, float, TimeUnits]:
    """Validate selected data and return ``(n_data_points, dt)``."""

    if not data_config_list:
        raise ValueError("At least one DataConfig is required.")

    if not selected:
        raise ValueError("At least one data channel must be selected.")

    first_config = data_config_list[selected[0]]

    for channel in selected:
        data_config = data_config_list[channel]
        if data_config.data.shape[0] != first_config.data.shape[0]:
            raise ValueError("Imported data must have same length!")
        if data_config.dt != first_config.dt or data_config.t_unit != first_config.t_unit:
            raise ValueError("Selected data channels must use the same dt and t_unit.")

    return first_config.data.shape[0], first_config.dt, first_config.t_unit


def build_runtime_config(
    spectrum_config: SpectrumConfig,
    data_config_list: list[DataConfig],
    selected: list[int] | None = None,
) -> RuntimeConfig:
    """Resolve user configuration into immutable runtime calculation settings.

    Validates the selected data channels, derives the frequency axis and frequency-band indices,
    checks Nyquist-frequency bounds, resolves the effective window size and window count, and
    selects torch dtypes and device settings used by the spectrum calculation.

    Parameters
    ----------
    spectrum_config : :class:`SpectrumConfig`
        User configuration for spectrum orders, frequency bounds, precision, device, windowing, and
        related calculation options.
    data_config_list : list[:class:`DataConfig`]
        Data configurations containing the input data and sampling metadata.
    selected : list[int] | None, optional
        Data-channel indices to use. If ``None``, all data configurations are selected.
    """

    # Validate and read the channels, number of data points, and the time step from the DataConfigs
    selected_channels = _normalize_selected(data_config_list, selected)
    n_data_points, dt, t_unit = _validate_data_configs(data_config_list, selected_channels)

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
    # frequency spacing in the given frequency bounds
    window_T = (spectrum_config.frequency_points - 1) / (f_max - spectrum_config.f_min)
    window_points = int(np.round(window_T / dt))
    if window_points <= 0:
        raise ValueError("Calculated window_points must be greater than zero.")

    # Resolve orders='all' to [1, 2, 3, 4]
    orders = [1, 2, 3, 4] if spectrum_config.orders == "all" else list(spectrum_config.orders)

    # Check if enough data is available and try to lower the window count per cumulant/spectrum
    # estimate if needed
    required_points = window_points * spectrum_config.m + window_points // 2
    if not required_points < n_data_points:
        m = (n_data_points - window_points // 2) // window_points
        if m < max(orders):
            raise ValueError("Not enough data points")
        print(
            "Values have been changed, because not enough data points were available."
            f"Old m: {spectrum_config.m}, new m: {m}"
        )
    else:
        m = spectrum_config.m

    # get the frequency axis
    n_windows = int(np.floor(n_data_points / (m * window_points)))
    use_full_fft = spectrum_config.f_min < 0
    if use_full_fft:
        freq_all = np.fft.fftfreq(window_points, dt)
        freq_all = np.fft.fftshift(freq_all)
    else:
        freq_all = np.fft.rfftfreq(window_points, dt)

    f_max_idx = int(np.sum(freq_all <= f_max))
    f_min_idx = int(np.sum(freq_all < spectrum_config.f_min))

    # determine the data types based on the given precision
    if spectrum_config.precision == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif spectrum_config.precision == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        if spectrum_config.device == "mps":
            real_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            real_dtype = torch.float64
            complex_dtype = torch.complex128

    # Determine the number of spectral estimates
    chunk_size = m * window_points
    half_shift = window_points // 2

    unshifted_estimates = n_data_points // chunk_size
    shifted_estimates = max(0, (n_data_points - half_shift) // chunk_size)

    available_estimates = unshifted_estimates + shifted_estimates

    if spectrum_config.spectral_estimates_max is None:
        spectral_estimates = available_estimates
    else:
        spectral_estimates = min(spectrum_config.spectral_estimates_max, available_estimates)

    return RuntimeConfig(
        selected_channels=selected_channels,
        orders=tuple(orders),
        dt=dt,
        window_points=window_points,
        m=m,
        n_data_points=n_data_points,
        n_windows=n_windows,
        freq_band=freq_all[f_min_idx:f_max_idx],
        freq_unit=unit_conversion_time_to_freq(t_unit),
        f_min_idx=f_min_idx,
        f_max_idx=f_max_idx,
        use_full_fft=use_full_fft,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        device=torch.device(spectrum_config.device),
        s3_calc=spectrum_config.s3_calc,
        spectral_estimates=spectral_estimates,
        old_window=spectrum_config.old_window,
    )


def build_spectrum_tasks(
    runtime_config: RuntimeConfig, cross_config: CrossConfig
) -> list[SpectrumTask]:
    """Build the concrete spectrum tasks requested by the configuration.

    Expands the high-level configuration into one :class:`SpectrumTask` per spectrum that should be
    calculated. Auto-correlation tasks are generated for each selected channel when
    ``cross_config.auto_corr`` is enabled. Cross tasks are generated from ``cross_corr_2``,
    ``cross_corr_3``, and ``cross_corr_4`` when their corresponding orders are requested.

    Parameters
    ----------
    runtime_config : :class:`RuntimeConfig`
        Configuration for spectrum order, frequency bounds, and numerical calculation settings.
    cross_config : :class:`CrossConfig`
        Configuration describing whether auto-spectra and which cross spectra should be calculated.

    Returns
    -------
    list[:class:`SpectrumTask`]
        Ordered list of concrete spectrum calculations to perform.
    """

    tasks: list[SpectrumTask] = []

    if cross_config.auto_corr:
        for channel in runtime_config.selected_channels:
            for order in runtime_config.orders:
                channels = (channel,) * order
                tasks.append(SpectrumTask(order=order, channels=channels))

    cross_specs = {
        2: cross_config.cross_corr_2 or [],
        3: cross_config.cross_corr_3 or [],
        4: cross_config.cross_corr_4 or [],
    }

    for order, channel_groups in cross_specs.items():
        if order not in runtime_config.orders:
            continue
        for channels in channel_groups:
            channels = tuple(channels)
            for channel in channels:
                if channel not in runtime_config.selected_channels:
                    raise ValueError(
                        f"Cross spectrum {channels} references channel {channel}, "
                        f"which is not in selected channels {runtime_config.selected_channels}."
                    )
            tasks.append(SpectrumTask(channels=channels, order=order))

    if not tasks:
        raise ValueError(
            "No spectrum tasks were requested. This may be because the requested cross-correlation "
            "spectra are not matching the specified orders in SpectrumConfig."
        )

    task_keys = [(task.channels, task.order) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("Duplicate spectrum tasks were requested.")

    return tasks


def initialize_result_store(
    tasks: list[SpectrumTask], runtime: RuntimeConfig
) -> SpectrumResultStore:
    """Create an initialized result store for a list of spectrum tasks.

    Each task is converted into a :class:`SpectrumResult` with matching order and channels.

    Parameters
    ----------
    tasks : list[:class:`SpectrumTask`]
        Spectrum tasks that should receive corresponding result containers.
    runtime : :class:`RuntimeConfig`
        :class:`RuntimeConfig` that contains all necessary information to initialize result arrays.

    Returns
    -------
    SpectrumResultStore
        Store containing one initialized :class:`SpectrumResult` per task.
    """

    store = SpectrumResultStore()
    for task in tasks:
        store.add(SpectrumResult(order=task.order, channels=task.channels))
    store.initialize_arrays(runtime)
    return store
